import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil
from pathlib import Path

class CelebADataProcessor:
    def __init__(self, data_dir, output_dir, target_size=(256, 256)):
        """
        Initialize CelebA data processor
        
        Args:
            data_dir (str): Path to CelebA images directory
            output_dir (str): Path to output processed data
            target_size (tuple): Target image size (height, width)
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.target_size = target_size
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "sketches").mkdir(exist_ok=True)
        (self.output_dir / "processed").mkdir(exist_ok=True)
        
    def check_image_quality(self, image_path):
        """
        Check if image meets quality standards
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            bool: True if image passes quality check
        """
        try:
            # Load image
            img = cv2.imread(str(image_path))
            if img is None:
                return False
            
            # Check minimum size
            height, width = img.shape[:2]
            if height < 64 or width < 64:
                return False
            
            # Check if image is corrupted (all pixels same color)
            if np.std(img) < 10:
                return False
                
            # Check if image is too dark or too bright
            mean_brightness = np.mean(img)
            if mean_brightness < 20 or mean_brightness > 235:
                return False
                
            return True
            
        except Exception as e:
            print(f"Error checking image {image_path}: {e}")
            return False
    
    def preprocess_image(self, image_path):
        """
        Preprocess individual image
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            numpy.ndarray: Processed image or None if failed
        """
        try:
            # Load image
            img = cv2.imread(str(image_path))
            if img is None:
                return None
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize image
            img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LANCZOS4)
            
            # Normalize to [0, 1]
            img = img.astype(np.float32) / 255.0
            
            # Apply histogram equalization to improve contrast
            img_yuv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2YUV)
            img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
            img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB).astype(np.float32) / 255.0
            
            return img
            
        except Exception as e:
            print(f"Error preprocessing image {image_path}: {e}")
            return None
    
    def create_sketch_from_image(self, image):
        """
        Create sketch from real image using edge detection
        
        Args:
            image (numpy.ndarray): Input image [0, 1]
            
        Returns:
            numpy.ndarray: Sketch image
        """
        # Convert to grayscale
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection using Canny
        edges = cv2.Canny(blurred, 50, 150)
        
        # Dilate edges to make them thicker
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Invert edges (white background, black lines)
        sketch = 255 - edges
        
        # Convert to 3 channel
        sketch = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)
        
        # Normalize to [0, 1]
        sketch = sketch.astype(np.float32) / 255.0
        
        return sketch
    
    def get_image_files(self):
        """
        Get list of valid image files
        
        Returns:
            list: List of image file paths
        """
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        image_files = []
        
        for file_path in self.data_dir.iterdir():
            if file_path.suffix.lower() in image_extensions:
                image_files.append(file_path)
        
        return sorted(image_files)
    
    def process_dataset(self, max_images=None):
        """
        Process entire dataset
        
        Args:
            max_images (int): Maximum number of images to process (None for all)
        """
        print("Starting CelebA dataset processing...")
        
        # Get all image files
        image_files = self.get_image_files()
        print(f"Found {len(image_files)} image files")
        
        if max_images:
            image_files = image_files[:max_images]
            print(f"Processing first {len(image_files)} images")
        
        # Process images
        processed_count = 0
        failed_count = 0
        
        for i, image_path in enumerate(tqdm(image_files, desc="Processing images")):
            try:
                # Check image quality
                if not self.check_image_quality(image_path):
                    failed_count += 1
                    continue
                
                # Preprocess image
                processed_img = self.preprocess_image(image_path)
                if processed_img is None:
                    failed_count += 1
                    continue
                
                # Create sketch
                sketch = self.create_sketch_from_image(processed_img)
                
                # Save processed image and sketch
                output_name = f"{processed_count:06d}.png"
                
                # Save real image
                real_img_path = self.output_dir / "images" / output_name
                Image.fromarray((processed_img * 255).astype(np.uint8)).save(real_img_path)
                
                # Save sketch
                sketch_path = self.output_dir / "sketches" / output_name
                Image.fromarray((sketch * 255).astype(np.uint8)).save(sketch_path)
                
                processed_count += 1
                
                # Save sample every 1000 images
                if processed_count % 1000 == 0:
                    self.save_sample_visualization(processed_img, sketch, processed_count)
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                failed_count += 1
                continue
        
        print(f"\nProcessing completed!")
        print(f"Successfully processed: {processed_count} images")
        print(f"Failed: {failed_count} images")
        print(f"Success rate: {processed_count/(processed_count+failed_count)*100:.2f}%")
        
        # Save dataset statistics
        self.save_dataset_stats(processed_count, failed_count)
    
    def save_sample_visualization(self, real_img, sketch, count):
        """
        Save sample visualization for monitoring
        """
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        axes[0].imshow(sketch)
        axes[0].set_title('Generated Sketch')
        axes[0].axis('off')
        
        axes[1].imshow(real_img)
        axes[1].set_title('Real Image')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"sample_{count}.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_dataset_stats(self, processed_count, failed_count):
        """
        Save dataset processing statistics
        """
        stats = {
            'total_processed': processed_count,
            'total_failed': failed_count,
            'success_rate': processed_count/(processed_count+failed_count)*100,
            'target_size': self.target_size,
            'output_directory': str(self.output_dir)
        }
        
        stats_text = f"""
CelebA Dataset Processing Statistics
====================================
Total processed: {stats['total_processed']}
Total failed: {stats['total_failed']}
Success rate: {stats['success_rate']:.2f}%
Target size: {stats['target_size']}
Output directory: {stats['output_directory']}
        """
        
        with open(self.output_dir / "processing_stats.txt", "w") as f:
            f.write(stats_text)
        
        print(stats_text)


def main():
    """
    Main function to run data processing
    """
    # Configuration
    data_dir = "d:/AI2/celebA"
    output_dir = "d:/AI2/processed_data"
    target_size = (256, 256)
    max_images = None  # Process all images, set to number if you want to limit
    
    # Initialize processor
    processor = CelebADataProcessor(data_dir, output_dir, target_size)
    
    # Process dataset
    processor.process_dataset(max_images=max_images)


if __name__ == "__main__":
    main()
