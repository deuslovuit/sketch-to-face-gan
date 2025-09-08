"""
🎨 วิธีใช้โมเดล Sketch-to-Face ที่เทรนเสร็จแล้ว
How to use your trained Sketch-to-Face model
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
import os

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from models import Generator

class SketchToFaceInference:
    """Class สำหรับใช้โมเดลที่เทรนแล้ว"""
    
    def __init__(self, checkpoint_path):
        """
        Load โมเดลจาก checkpoint
        
        Args:
            checkpoint_path (str): Path ไปยังไฟล์ checkpoint (.pth)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load checkpoint
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        print(f"Loading checkpoint from epoch {checkpoint['epoch']}")
        
        # Create and load generator
        self.generator = Generator(input_channels=3, output_channels=3, features=64)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.generator.to(self.device)
        self.generator.eval()
        
        # Transform สำหรับ preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [-1, 1]
        ])
        
        print("✅ Model loaded successfully!")
    
    def generate_face(self, sketch_path, output_path=None):
        """
        สร้างหน้าคนจากภาพสเก็ต
        
        Args:
            sketch_path (str): Path ไปยังภาพสเก็ต
            output_path (str): Path สำหรับบันทึกผลลัพธ์
            
        Returns:
            PIL.Image: ภาพหน้าคนที่สร้างขึ้น
        """
        # Load และ preprocess ภาพสเก็ต
        sketch_image = Image.open(sketch_path).convert('RGB')
        sketch_tensor = self.transform(sketch_image).unsqueeze(0).to(self.device)
        
        # Generate ใบหน้า
        with torch.no_grad():
            fake_face = self.generator(sketch_tensor)
        
        # Convert กลับเป็น PIL Image
        fake_face = (fake_face.squeeze().cpu() + 1) / 2  # [-1,1] -> [0,1]
        fake_face = transforms.ToPILImage()(fake_face)
        
        # บันทึกถ้าระบุ path
        if output_path:
            fake_face.save(output_path)
            print(f"✅ Generated face saved to: {output_path}")
        
        return fake_face
    
    def generate_batch(self, sketch_folder, output_folder):
        """
        สร้างหน้าคนจากสเก็ตหลายๆ ภาพ
        
        Args:
            sketch_folder (str): โฟลเดอร์ที่มีภาพสเก็ต
            output_folder (str): โฟลเดอร์สำหรับบันทึกผลลัพธ์
        """
        sketch_folder = Path(sketch_folder)
        output_folder = Path(output_folder)
        output_folder.mkdir(exist_ok=True)
        
        sketch_files = list(sketch_folder.glob("*.png")) + list(sketch_folder.glob("*.jpg"))
        
        print(f"Processing {len(sketch_files)} sketches...")
        
        for sketch_file in sketch_files:
            output_file = output_folder / f"generated_{sketch_file.name}"
            try:
                self.generate_face(str(sketch_file), str(output_file))
            except Exception as e:
                print(f"❌ Error processing {sketch_file.name}: {e}")
        
        print(f"✅ Batch processing completed! Check {output_folder}")
    
    def create_comparison(self, sketch_path, output_path):
        """
        สร้างภาพเปรียบเทียบระหว่างสเก็ตกับหน้าคนที่สร้าง
        
        Args:
            sketch_path (str): Path ไปยังภาพสเก็ต
            output_path (str): Path สำหรับบันทึกภาพเปรียบเทียบ
        """
        # Load ภาพสเก็ต
        sketch_image = Image.open(sketch_path).convert('RGB')
        
        # Generate หน้าคน
        generated_face = self.generate_face(sketch_path)
        
        # สร้างภาพเปรียบเทียบ
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        axes[0].imshow(sketch_image)
        axes[0].set_title('Input Sketch')
        axes[0].axis('off')
        
        axes[1].imshow(generated_face)
        axes[1].set_title('Generated Face')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Comparison saved to: {output_path}")


def find_best_checkpoint():
    """หา checkpoint ที่ดีที่สุด"""
    checkpoint_dir = Path("d:/AI2/training_output/checkpoints")
    
    if not checkpoint_dir.exists():
        print("❌ No checkpoints found!")
        return None
    
    checkpoints = list(checkpoint_dir.glob("checkpoint_epoch_*.pth"))
    
    if not checkpoints:
        # ลอง latest checkpoint
        latest_path = checkpoint_dir / "latest.pth"
        if latest_path.exists():
            return str(latest_path)
        print("❌ No checkpoints found!")
        return None
    
    # เลือก checkpoint ล่าสุด
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.stem.split('_')[-1]))
    print(f"🎯 Found checkpoint: {latest_checkpoint}")
    return str(latest_checkpoint)


def demo_usage():
    """ตัวอย่างการใช้งาน"""
    print("🎨 Demo: Using Trained Sketch-to-Face Model")
    print("="*50)
    
    # หา checkpoint ที่ดีที่สุด
    checkpoint_path = find_best_checkpoint()
    if not checkpoint_path:
        print("❌ No trained model found!")
        print("💡 Make sure training has created checkpoints in:")
        print("   d:/AI2/training_output/checkpoints/")
        return
    
    try:
        # Load โมเดล
        model = SketchToFaceInference(checkpoint_path)
        
        # ทดสอบกับภาพสเก็ตจาก validation set
        test_sketch_dir = Path("d:/AI2/processed_data/sketches")
        test_sketches = list(test_sketch_dir.glob("*.png"))[:5]  # เอา 5 ภาพแรก
        
        if test_sketches:
            print(f"\n🧪 Testing with {len(test_sketches)} sample sketches...")
            
            output_dir = Path("d:/AI2/generated_faces")
            output_dir.mkdir(exist_ok=True)
            
            for i, sketch_path in enumerate(test_sketches):
                print(f"\nProcessing sketch {i+1}: {sketch_path.name}")
                
                # Generate face
                output_path = output_dir / f"generated_face_{i+1}.png"
                model.generate_face(str(sketch_path), str(output_path))
                
                # Create comparison
                comparison_path = output_dir / f"comparison_{i+1}.png"
                model.create_comparison(str(sketch_path), str(comparison_path))
            
            print(f"\n✅ Demo completed!")
            print(f"📁 Check results in: {output_dir}")
            print(f"🖼️ Generated faces: generated_face_*.png")
            print(f"📊 Comparisons: comparison_*.png")
        
        else:
            print("❌ No test sketches found!")
    
    except Exception as e:
        print(f"❌ Error: {e}")


def show_model_locations():
    """แสดงตำแหน่งไฟล์โมเดลและผลลัพธ์"""
    print("📁 ตำแหน่งไฟล์โมเดลและผลลัพธ์:")
    print("="*45)
    
    locations = {
        "🎯 Trained Models": "d:/AI2/training_output/checkpoints/",
        "📊 Training Logs": "d:/AI2/training_output/logs/", 
        "🖼️ Sample Images": "d:/AI2/training_output/samples/",
        "🎨 Generated Faces": "d:/AI2/generated_faces/",
        "📝 Original Data": "d:/AI2/processed_data/"
    }
    
    for desc, path in locations.items():
        path_obj = Path(path)
        status = "✅ EXISTS" if path_obj.exists() else "❌ NOT FOUND"
        print(f"{desc}: {path} [{status}]")
        
        if path_obj.exists() and "checkpoints" in path:
            checkpoints = list(path_obj.glob("*.pth"))
            print(f"   📦 Found {len(checkpoints)} checkpoint files")


def main():
    """Main function"""
    print("🎨 Sketch-to-Face Model Usage Guide")
    print("="*60)
    
    show_model_locations()
    
    print("\n" + "="*60)
    print("🚀 Usage Options:")
    print("1. 🧪 Run demo with sample sketches")
    print("2. 🎨 Use your own sketch images")
    print("3. 📚 Read the code examples")
    print("="*60)
    
    # รัน demo
    demo_usage()


if __name__ == "__main__":
    main()
