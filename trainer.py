import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import time
from tqdm import tqdm
import os

# Import our modules
from dataset_simple import SimpleDataset
from models import Generator, Discriminator, weights_init


class Pix2PixTrainer:
    """
    Pix2Pix GAN Trainer for Sketch-to-Face translation
    """
    def __init__(self, config):
        self.config = config
        
        # Auto-detect best device
        self.device = self.get_best_device()
        print(f"Using device: {self.device}")
        
        # Create output directories
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)
        (self.output_dir / "samples").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        
        # Initialize models
        self.generator = Generator(
            input_channels=3,
            output_channels=3,
            features=config['gen_features']
        ).to(self.device)
        
        self.discriminator = Discriminator(
            input_channels=6,
            features=config['disc_features']
        ).to(self.device)
        
        # Apply weight initialization
        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)
        
        # Loss functions
        self.adversarial_loss = nn.BCELoss()
        self.l1_loss = nn.L1Loss()
        
        # Optimizers
        self.optimizer_G = optim.Adam(
            self.generator.parameters(),
            lr=config['lr'],
            betas=(config['beta1'], 0.999)
        )
        
        self.optimizer_D = optim.Adam(
            self.discriminator.parameters(),
            lr=config['lr'],
            betas=(config['beta1'], 0.999)
        )
        
        # Training metrics
        self.g_losses = []
        self.d_losses = []
        self.l1_losses = []
        
        print("✅ Pix2Pix trainer initialized")
    
    def get_best_device(self):
        """Auto-detect the best available device"""
        # Check for CUDA (NVIDIA)
        if torch.cuda.is_available():
            return torch.device('cuda')
        
        # Check for ROCm (AMD)
        try:
            if hasattr(torch.version, 'hip') and torch.version.hip is not None:
                return torch.device('cuda')  # ROCm uses cuda syntax
        except:
            pass
        
        # Check for DirectML (AMD/Intel)
        try:
            import torch_directml
            if torch_directml.is_available():
                return torch_directml.device()
        except ImportError:
            pass
        
        # Fallback to CPU
        return torch.device('cpu')
    
    def create_data_loaders(self):
        """Create training and validation data loaders"""
        
        # Training dataset
        train_dataset = SimpleDataset(
            images_dir=self.config['data_dir'] + "/images",
            sketches_dir=self.config['data_dir'] + "/sketches",
            max_items=self.config['max_train_samples']
        )
        
        # Validation dataset (smaller subset)
        val_dataset = SimpleDataset(
            images_dir=self.config['data_dir'] + "/images", 
            sketches_dir=self.config['data_dir'] + "/sketches",
            max_items=self.config['max_val_samples']
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        print(f"Training batches: {len(self.train_loader)}")
        print(f"Validation batches: {len(self.val_loader)}")
    
    def train_discriminator(self, real_sketches, real_images, fake_images):
        """Train discriminator"""
        self.optimizer_D.zero_grad()
        
        batch_size = real_sketches.size(0)
        
        # Real labels (1) and fake labels (0)
        real_labels = torch.ones(batch_size, 1, 30, 30).to(self.device)  # PatchGAN output size
        fake_labels = torch.zeros(batch_size, 1, 30, 30).to(self.device)
        
        # Train with real images
        real_output = self.discriminator(real_sketches, real_images)
        real_loss = self.adversarial_loss(real_output, real_labels)
        
        # Train with fake images
        fake_output = self.discriminator(real_sketches, fake_images.detach())
        fake_loss = self.adversarial_loss(fake_output, fake_labels)
        
        # Total discriminator loss
        d_loss = (real_loss + fake_loss) * 0.5
        d_loss.backward()
        self.optimizer_D.step()
        
        return d_loss.item()
    
    def train_generator(self, real_sketches, real_images, fake_images):
        """Train generator"""
        self.optimizer_G.zero_grad()
        
        batch_size = real_sketches.size(0)
        
        # Generator wants discriminator to classify fake images as real
        real_labels = torch.ones(batch_size, 1, 30, 30).to(self.device)
        
        # Adversarial loss
        fake_output = self.discriminator(real_sketches, fake_images)
        adv_loss = self.adversarial_loss(fake_output, real_labels)
        
        # L1 loss for pixel-wise accuracy
        l1_loss = self.l1_loss(fake_images, real_images)
        
        # Total generator loss
        g_loss = adv_loss + self.config['lambda_l1'] * l1_loss
        g_loss.backward()
        self.optimizer_G.step()
        
        return g_loss.item(), l1_loss.item()
    
    def save_sample_images(self, epoch, batch_idx):
        """Save sample generated images"""
        self.generator.eval()
        
        with torch.no_grad():
            # Get a batch from validation set
            val_batch = next(iter(self.val_loader))
            real_sketches = val_batch['sketch'][:4].to(self.device)
            real_images = val_batch['real_image'][:4].to(self.device)
            
            # Generate fake images
            fake_images = self.generator(real_sketches)
            
            # Denormalize images from [-1, 1] to [0, 1]
            real_sketches = (real_sketches + 1) / 2
            real_images = (real_images + 1) / 2
            fake_images = (fake_images + 1) / 2
            
            # Create comparison grid
            comparison = torch.cat([
                real_sketches,
                fake_images,
                real_images
            ], dim=0)
            
            # Save image
            save_path = self.output_dir / "samples" / f"epoch_{epoch:03d}_batch_{batch_idx:04d}.png"
            vutils.save_image(comparison, save_path, nrow=4, normalize=False)
        
        self.generator.train()
    
    def save_checkpoint(self, epoch):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'g_losses': self.g_losses,
            'd_losses': self.d_losses,
            'l1_losses': self.l1_losses,
            'config': self.config
        }
        
        checkpoint_path = self.output_dir / "checkpoints" / f"checkpoint_epoch_{epoch:03d}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save latest checkpoint
        latest_path = self.output_dir / "checkpoints" / "latest.pth"
        torch.save(checkpoint, latest_path)
        
        print(f"✅ Checkpoint saved: {checkpoint_path}")
    
    def plot_losses(self):
        """Plot training losses"""
        if len(self.g_losses) == 0:
            return
        
        plt.figure(figsize=(15, 5))
        
        # Generator and Discriminator losses
        plt.subplot(1, 3, 1)
        plt.plot(self.g_losses, label='Generator Loss')
        plt.plot(self.d_losses, label='Discriminator Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Adversarial Losses')
        plt.legend()
        plt.grid(True)
        
        # L1 Loss
        plt.subplot(1, 3, 2)
        plt.plot(self.l1_losses, label='L1 Loss')
        plt.xlabel('Iteration')
        plt.ylabel('L1 Loss')
        plt.title('L1 Reconstruction Loss')
        plt.legend()
        plt.grid(True)
        
        # Combined view
        plt.subplot(1, 3, 3)
        plt.plot(self.g_losses, label='Generator', alpha=0.7)
        plt.plot(self.d_losses, label='Discriminator', alpha=0.7)
        plt.plot(np.array(self.l1_losses) * 10, label='L1 x10', alpha=0.7)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('All Losses')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "logs" / "training_losses.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def train(self):
        """Main training loop"""
        print("🚀 Starting Pix2Pix training...")
        
        # Create data loaders
        self.create_data_loaders()
        
        # Training loop
        start_time = time.time()
        iteration = 0
        
        for epoch in range(self.config['num_epochs']):
            epoch_start_time = time.time()
            
            # Training
            self.generator.train()
            self.discriminator.train()
            
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            epoch_l1_loss = 0.0
            
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['num_epochs']}")
            
            for batch_idx, batch in enumerate(pbar):
                real_sketches = batch['sketch'].to(self.device)
                real_images = batch['real_image'].to(self.device)
                
                # Generate fake images
                fake_images = self.generator(real_sketches)
                
                # Train Discriminator
                d_loss = self.train_discriminator(real_sketches, real_images, fake_images)
                
                # Train Generator
                g_loss, l1_loss = self.train_generator(real_sketches, real_images, fake_images)
                
                # Record losses
                self.g_losses.append(g_loss)
                self.d_losses.append(d_loss)
                self.l1_losses.append(l1_loss)
                
                epoch_g_loss += g_loss
                epoch_d_loss += d_loss
                epoch_l1_loss += l1_loss
                
                # Update progress bar
                pbar.set_postfix({
                    'G_Loss': f'{g_loss:.4f}',
                    'D_Loss': f'{d_loss:.4f}',
                    'L1_Loss': f'{l1_loss:.4f}'
                })
                
                # Save sample images
                if iteration % self.config['sample_interval'] == 0:
                    self.save_sample_images(epoch, batch_idx)
                
                iteration += 1
            
            # Epoch summary
            epoch_time = time.time() - epoch_start_time
            avg_g_loss = epoch_g_loss / len(self.train_loader)
            avg_d_loss = epoch_d_loss / len(self.train_loader)
            avg_l1_loss = epoch_l1_loss / len(self.train_loader)
            
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Time: {epoch_time:.2f}s")
            print(f"  Avg G Loss: {avg_g_loss:.4f}")
            print(f"  Avg D Loss: {avg_d_loss:.4f}")
            print(f"  Avg L1 Loss: {avg_l1_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % self.config['checkpoint_interval'] == 0:
                self.save_checkpoint(epoch + 1)
            
            # Plot losses
            if (epoch + 1) % self.config['plot_interval'] == 0:
                self.plot_losses()
        
        # Final save
        self.save_checkpoint(self.config['num_epochs'])
        self.plot_losses()
        
        total_time = time.time() - start_time
        print(f"\n🎉 Training completed in {total_time/60:.2f} minutes!")


def get_training_config():
    """Get training configuration"""
    return {
        # Data
        'data_dir': 'd:/AI2/processed_data',
        'max_train_samples': 5000,  # Start with smaller dataset for testing
        'max_val_samples': 100,
        
        # Model
        'gen_features': 64,
        'disc_features': 64,
        
        # Training
        'num_epochs': 50,
        'batch_size': 8,  # Smaller batch size for safety
        'lr': 0.0002,
        'beta1': 0.5,
        'lambda_l1': 100,  # L1 loss weight
        
        # Output
        'output_dir': 'd:/AI2/training_output',
        'sample_interval': 100,  # Save samples every N iterations
        'checkpoint_interval': 5,  # Save checkpoint every N epochs
        'plot_interval': 5,  # Plot losses every N epochs
    }


if __name__ == "__main__":
    # Get configuration
    config = get_training_config()
    
    # Create trainer
    trainer = Pix2PixTrainer(config)
    
    # Start training
    trainer.train()
