"""
🚀 Pix2Pix GAN Training Script
สำหรับการฝึกสอนโมเดลแปลงภาพสเก็ตเป็นหน้าคนจริง
"""

import torch
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from trainer import Pix2PixTrainer, get_training_config

def main():
    """Main training function"""
    print("="*60)
    print("🎨 Pix2Pix GAN: Sketch to Face Translation")
    print("="*60)
    
    # Check system requirements
    print("\n📋 System Check:")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Get training configuration
    config = get_training_config()
    
    print(f"\n⚙️ Training Configuration:")
    print(f"Dataset: {config['data_dir']}")
    print(f"Training samples: {config['max_train_samples']:,}")
    print(f"Validation samples: {config['max_val_samples']:,}")
    print(f"Epochs: {config['num_epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['lr']}")
    print(f"Output directory: {config['output_dir']}")
    
    # Verify data exists
    data_path = Path(config['data_dir'])
    if not data_path.exists():
        print(f"❌ Error: Data directory not found: {data_path}")
        return
    
    images_dir = data_path / "images"
    sketches_dir = data_path / "sketches"
    
    if not images_dir.exists() or not sketches_dir.exists():
        print(f"❌ Error: Images or sketches directory not found")
        return
    
    image_count = len(list(images_dir.glob("*.png")))
    sketch_count = len(list(sketches_dir.glob("*.png")))
    
    print(f"\n📊 Dataset Information:")
    print(f"Real images: {image_count:,}")
    print(f"Sketch images: {sketch_count:,}")
    
    if image_count == 0 or sketch_count == 0:
        print("❌ Error: No images found in dataset")
        return
    
    # Ask for confirmation
    print(f"\n🤔 Ready to start training?")
    print(f"This will train for {config['num_epochs']} epochs with {config['max_train_samples']} samples")
    print(f"Estimated time: ~{config['num_epochs'] * config['max_train_samples'] / config['batch_size'] / 60:.0f} minutes")
    
    response = input("Continue? (y/n): ").lower().strip()
    if response != 'y':
        print("Training cancelled.")
        return
    
    try:
        # Create trainer
        print(f"\n🔧 Initializing trainer...")
        trainer = Pix2PixTrainer(config)
        
        # Start training
        print(f"\n🚀 Starting training...")
        trainer.train()
        
        print(f"\n🎉 Training completed successfully!")
        print(f"Check results in: {config['output_dir']}")
        
    except KeyboardInterrupt:
        print(f"\n⏹️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
