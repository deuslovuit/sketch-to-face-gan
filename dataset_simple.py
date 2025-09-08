import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

class SketchToFaceDataset(Dataset):
    """
    Dataset class for Sketch to Face translation
    """
    def __init__(self, images_dir, sketches_dir, transform=None, mode='train'):
        """
        Initialize dataset
        
        Args:
            images_dir (str): Directory containing real face images
            sketches_dir (str): Directory containing sketch images
            transform: Data augmentation transforms
            mode (str): 'train', 'val', or 'test'
        """
        self.images_dir = Path(images_dir)
        self.sketches_dir = Path(sketches_dir)
        self.transform = transform
        self.mode = mode
        
        # Get all image files
        self.image_files = sorted([f for f in os.listdir(images_dir) 
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        print(f"Found {len(self.image_files)} image pairs for {mode} mode")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Get file name
        img_name = self.image_files[idx]
        
        # Load sketch and real image
        sketch_path = self.sketches_dir / img_name
        image_path = self.images_dir / img_name
        
        try:
            # Load images as RGB
            sketch = Image.open(sketch_path).convert('RGB')
            real_image = Image.open(image_path).convert('RGB')
            
            # Convert to numpy arrays
            sketch = np.array(sketch, dtype=np.float32) / 255.0
            real_image = np.array(real_image, dtype=np.float32) / 255.0
            
            # Apply transforms if provided
            if self.transform:
                sketch = self.transform(sketch)
                real_image = self.transform(real_image)
            else:
                # Convert to tensors and normalize to [-1, 1]
                sketch = torch.FloatTensor(sketch).permute(2, 0, 1) * 2.0 - 1.0
                real_image = torch.FloatTensor(real_image).permute(2, 0, 1) * 2.0 - 1.0
            
            return {
                'sketch': sketch,
                'real_image': real_image,
                'filename': img_name
            }
            
        except Exception as e:
            print(f"Error loading {img_name}: {e}")
            # Return a dummy sample in case of error
            return {
                'sketch': torch.zeros(3, 256, 256),
                'real_image': torch.zeros(3, 256, 256),
                'filename': img_name
            }


class SimpleDataset(Dataset):
    """
    Simple dataset with limited items for testing
    """
    def __init__(self, images_dir, sketches_dir, max_items=None):
        self.images_dir = Path(images_dir)
        self.sketches_dir = Path(sketches_dir)
        
        # Get all image files
        all_files = sorted([f for f in os.listdir(images_dir) 
                           if f.lower().endswith('.png')])
        
        # Limit number of items if specified
        if max_items:
            self.image_files = all_files[:max_items]
        else:
            self.image_files = all_files
            
        print(f"Using {len(self.image_files)} files")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        
        # Load images
        sketch_path = self.sketches_dir / img_name
        image_path = self.images_dir / img_name
        
        # Load as PIL images and convert to numpy
        sketch = np.array(Image.open(sketch_path).convert('RGB'))
        real_image = np.array(Image.open(image_path).convert('RGB'))
        
        # Normalize to [-1, 1] and convert to tensors
        sketch = torch.FloatTensor(sketch / 127.5 - 1.0).permute(2, 0, 1)
        real_image = torch.FloatTensor(real_image / 127.5 - 1.0).permute(2, 0, 1)
        
        return {
            'sketch': sketch,
            'real_image': real_image,
            'filename': img_name
        }


def get_basic_transforms(image_size=256):
    """
    Get basic transforms for preprocessing
    
    Args:
        image_size (int): Target image size
        
    Returns:
        torchvision.transforms.Compose: Transform pipeline
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])


def create_simple_data_loaders(data_dir, batch_size=16, num_workers=0, train_split=0.8, val_split=0.1):
    """
    Create simple data loaders without complex splitting
    
    Args:
        data_dir (str): Directory containing processed data
        batch_size (int): Batch size for data loaders
        num_workers (int): Number of worker processes
        train_split (float): Fraction of data for training
        val_split (float): Fraction of data for validation
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    data_path = Path(data_dir)
    images_dir = data_path / "images"
    sketches_dir = data_path / "sketches"
    
    # Get all image files
    all_files = sorted([f for f in os.listdir(images_dir) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    # Split indices
    n_total = len(all_files)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)
    
    train_files = all_files[:n_train]
    val_files = all_files[n_train:n_train + n_val]
    test_files = all_files[n_train + n_val:]
    
    print(f"Data split: Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
    
    # Create custom datasets for each split
    class SplitDataset(SketchToFaceDataset):
        def __init__(self, images_dir, sketches_dir, file_list, transform=None, mode='train'):
            self.images_dir = Path(images_dir)
            self.sketches_dir = Path(sketches_dir)
            self.transform = transform
            self.mode = mode
            self.image_files = file_list
            print(f"Split dataset initialized with {len(self.image_files)} files for {mode} mode")
    
    # Create datasets
    transform = get_basic_transforms()
    
    train_dataset = SplitDataset(
        images_dir=images_dir,
        sketches_dir=sketches_dir,
        file_list=train_files,
        transform=transform,
        mode='train'
    )
    
    val_dataset = SplitDataset(
        images_dir=images_dir,
        sketches_dir=sketches_dir,
        file_list=val_files,
        transform=transform,
        mode='val'
    )
    
    test_dataset = SplitDataset(
        images_dir=images_dir,
        sketches_dir=sketches_dir,
        file_list=test_files,
        transform=transform,
        mode='test'
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader, test_loader


def visualize_batch(data_loader, num_samples=4, save_path='data_visualization.png'):
    """
    Visualize a batch of data
    
    Args:
        data_loader: DataLoader to visualize
        num_samples (int): Number of samples to show
        save_path (str): Path to save visualization
    """
    # Get a batch
    batch = next(iter(data_loader))
    sketches = batch['sketch']
    real_images = batch['real_image']
    
    # Denormalize images from [-1, 1] to [0, 1]
    def denormalize(tensor):
        return (tensor * 0.5 + 0.5).clamp(0, 1)
    
    sketches = denormalize(sketches)
    real_images = denormalize(real_images)
    
    # Plot
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples*3, 6))
    
    for i in range(min(num_samples, sketches.size(0))):
        # Sketch
        sketch_np = sketches[i].permute(1, 2, 0).numpy()
        axes[0, i].imshow(sketch_np)
        axes[0, i].set_title('Sketch')
        axes[0, i].axis('off')
        
        # Real image
        real_np = real_images[i].permute(1, 2, 0).numpy()
        axes[1, i].imshow(real_np)
        axes[1, i].set_title('Real Image')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {save_path}")


if __name__ == "__main__":
    # Test data loading
    data_dir = "d:/AI2/processed_data"
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_simple_data_loaders(
        data_dir=data_dir,
        batch_size=8,
        num_workers=0
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Test loading a batch
    print("\nTesting batch loading...")
    batch = next(iter(train_loader))
    
    print(f"Sketch shape: {batch['sketch'].shape}")
    print(f"Real image shape: {batch['real_image'].shape}")
    print(f"Sketch value range: {batch['sketch'].min():.3f} to {batch['sketch'].max():.3f}")
    print(f"Real image value range: {batch['real_image'].min():.3f} to {batch['real_image'].max():.3f}")
    
    # Visualize some samples
    print("\nCreating data visualization...")
    visualize_batch(train_loader)
