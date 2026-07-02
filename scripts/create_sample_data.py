"""
Generate sample plant images for testing.

Run this script to create dummy images for training and testing.
This is useful for getting the system running quickly without real data.

Usage:
    python scripts/create_sample_data.py

Creates:
    data/raw/healthy/
    data/raw/powdery_mildew/
    data/raw/rust/
    data/raw/leaf_spot/
    data/raw/blight/
"""

import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
import argparse


def create_sample_images(num_samples_per_class: int = 20):
    """
    Create synthetic plant leaf images for testing.
    
    Args:
        num_samples_per_class: Number of images to create per disease class
    """
    classes = {
        "healthy": {"color": (34, 139, 34), "pattern": "solid"},
        "powdery_mildew": {"color": (192, 192, 192), "pattern": "spots"},
        "rust": {"color": (184, 92, 0), "pattern": "dots"},
        "leaf_spot": {"color": (85, 85, 85), "pattern": "patches"},
        "blight": {"color": (139, 69, 19), "pattern": "stripes"},
    }
    
    base_dir = Path("data/raw")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating sample plant leaf images...")
    print(f"Generating {num_samples_per_class} images per class\n")
    
    for class_name, properties in classes.items():
        class_dir = base_dir / class_name
        class_dir.mkdir(exist_ok=True)
        
        print(f"Creating images for class: {class_name}")
        
        for idx in range(num_samples_per_class):
            # Create image (224x224 like ImageNet)
            img = Image.new('RGB', (224, 224), color=(144, 238, 144))  # Light green background
            draw = ImageDraw.Draw(img, 'RGBA')
            
            # Draw leaf shape (ellipse)
            leaf_color = properties["color"]
            draw.ellipse([50, 50, 174, 174], fill=leaf_color, outline="black", width=2)
            
            # Add disease pattern based on class
            pattern = properties["pattern"]
            
            if pattern == "spots":
                # Powdery mildew: white spots
                for _ in range(np.random.randint(5, 15)):
                    x = np.random.randint(60, 164)
                    y = np.random.randint(60, 164)
                    r = np.random.randint(5, 15)
                    draw.ellipse([x-r, y-r, x+r, y+r], fill=(255, 255, 255, 200))
            
            elif pattern == "dots":
                # Rust: orange/brown dots
                for _ in range(np.random.randint(10, 25)):
                    x = np.random.randint(60, 164)
                    y = np.random.randint(60, 164)
                    r = np.random.randint(2, 8)
                    draw.ellipse([x-r, y-r, x+r, y+r], fill=(255, 165, 0, 150))
            
            elif pattern == "patches":
                # Leaf spot: dark patches
                for _ in range(np.random.randint(3, 8)):
                    x = np.random.randint(60, 164)
                    y = np.random.randint(60, 164)
                    w = np.random.randint(15, 40)
                    h = np.random.randint(15, 40)
                    draw.rectangle([x-w//2, y-h//2, x+w//2, y+h//2], 
                                 fill=(0, 0, 0, 100))
            
            elif pattern == "stripes":
                # Blight: streaky dark areas
                for _ in range(np.random.randint(2, 5)):
                    x1 = np.random.randint(60, 164)
                    y1 = np.random.randint(60, 164)
                    x2 = x1 + np.random.randint(-50, 50)
                    y2 = y1 + np.random.randint(-50, 50)
                    draw.line([(x1, y1), (x2, y2)], fill=(0, 0, 0, 150), width=3)
            
            # Add some noise
            img_array = np.array(img)
            noise = np.random.randint(-10, 10, img_array.shape)
            img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_array)
            
            # Save image
            img_path = class_dir / f"{class_name}_{idx:03d}.jpg"
            img.save(img_path, quality=95)
        
        print(f"  ✓ Created {num_samples_per_class} images in {class_dir}")
    
    print("\n" + "="*60)
    print("Sample data created successfully!")
    print(f"Location: {base_dir.absolute()}")
    print("\nDirectory structure:")
    for class_name in classes.keys():
        count = len(list((base_dir / class_name).glob("*.jpg")))
        print(f"  {class_name}/: {count} images")
    print("="*60)
    print("\nNext steps:")
    print("1. Train model: python src/train.py")
    print("2. Start API: python src/api.py")
    print("3. Open browser: http://localhost:8000/docs")


def main():
    parser = argparse.ArgumentParser(
        description="Create sample plant leaf images for testing"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="Number of images to create per class (default: 20)"
    )
    
    args = parser.parse_args()
    create_sample_images(num_samples_per_class=args.num_samples)


if __name__ == "__main__":
    main()
