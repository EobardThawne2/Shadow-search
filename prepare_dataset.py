"""
Data Preparation Utilities for Bharatanatyam Mudra Classification

This script provides utilities for organizing and preparing the dataset.
"""

import os
import shutil
import random
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

class DatasetPreparer:
    def __init__(self, base_path='data/bharatanatyam_mudras'):
        self.base_path = Path(base_path)
        self.mudra_classes = [
            'pataka',      # Open palm gesture
            'tripataka',   # Three-flag gesture  
            'ardhapataka', # Half-flag gesture
            'kartarimukha',# Scissors gesture
            'mayura',      # Peacock gesture
            'ardhachandra',# Half-moon gesture
            'arala',       # Bent gesture
            'shukatunda',  # Parrot's beak gesture
            'mushtika',    # Fist gesture
            'sikhara'      # Peak gesture
        ]
        
    def create_directory_structure(self):
        """Create the required directory structure"""
        splits = ['train', 'validation', 'test']
        
        for split in splits:
            for mudra_class in self.mudra_classes:
                dir_path = self.base_path / split / mudra_class
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"Created directory: {dir_path}")
                
        print(f"\nDirectory structure created at: {self.base_path}")
        
    def split_dataset(self, source_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        """Split dataset into train, validation, and test sets"""
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
            
        source_path = Path(source_dir)
        if not source_path.exists():
            raise FileNotFoundError(f"Source directory not found: {source_dir}")
            
        print(f"Splitting dataset from: {source_path}")
        
        for mudra_class in self.mudra_classes:
            class_source = source_path / mudra_class
            if not class_source.exists():
                print(f"Warning: Class directory not found: {class_source}")
                continue
                
            # Get all image files
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                image_files.extend(list(class_source.glob(ext)))
                
            if not image_files:
                print(f"Warning: No images found in {class_source}")
                continue
                
            # Shuffle files
            random.shuffle(image_files)
            
            # Calculate split indices
            n_total = len(image_files)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)
            
            # Split files
            train_files = image_files[:n_train]
            val_files = image_files[n_train:n_train + n_val]
            test_files = image_files[n_train + n_val:]
            
            # Copy files to respective directories
            splits = {
                'train': train_files,
                'validation': val_files,
                'test': test_files
            }
            
            for split_name, files in splits.items():
                dest_dir = self.base_path / split_name / mudra_class
                dest_dir.mkdir(parents=True, exist_ok=True)
                
                for file in tqdm(files, desc=f"Copying {mudra_class} to {split_name}"):
                    dest_file = dest_dir / file.name
                    shutil.copy2(file, dest_file)
                    
            print(f"{mudra_class}: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
            
    def augment_dataset(self, target_samples_per_class=1000):
        """Augment dataset to have target number of samples per class"""
        train_dir = self.base_path / 'train'
        
        for mudra_class in self.mudra_classes:
            class_dir = train_dir / mudra_class
            if not class_dir.exists():
                continue
                
            # Get existing images
            existing_images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
            current_count = len(existing_images)
            
            if current_count >= target_samples_per_class:
                print(f"{mudra_class}: Already has {current_count} samples (target: {target_samples_per_class})")
                continue
                
            needed_samples = target_samples_per_class - current_count
            print(f"{mudra_class}: Generating {needed_samples} augmented samples...")
            
            # Generate augmented images
            for i in tqdm(range(needed_samples), desc=f"Augmenting {mudra_class}"):
                # Select random source image
                source_img_path = random.choice(existing_images)
                
                # Load image
                img = cv2.imread(str(source_img_path))
                if img is None:
                    continue
                    
                # Apply random augmentations
                augmented_img = self.apply_augmentations(img)
                
                # Save augmented image
                output_path = class_dir / f"aug_{i:04d}_{source_img_path.stem}.jpg"
                cv2.imwrite(str(output_path), augmented_img)
                
    def apply_augmentations(self, img):
        """Apply random augmentations to an image"""
        # Random rotation
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            center = (img.shape[1] // 2, img.shape[0] // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]))
            
        # Random brightness adjustment
        if random.random() > 0.5:
            brightness = random.uniform(0.8, 1.2)
            img = cv2.convertScaleAbs(img, alpha=brightness, beta=0)
            
        # Random horizontal flip
        if random.random() > 0.5:
            img = cv2.flip(img, 1)
            
        # Random blur
        if random.random() > 0.7:
            kernel_size = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
            
        # Random noise
        if random.random() > 0.8:
            noise = np.random.randint(0, 25, img.shape, dtype=np.uint8)
            img = cv2.add(img, noise)
            
        return img
        
    def validate_dataset(self):
        """Validate the dataset structure and count samples"""
        print("Dataset Validation Report")
        print("=" * 50)
        
        total_samples = 0
        splits = ['train', 'validation', 'test']
        
        for split in splits:
            print(f"\n{split.upper()} SET:")
            split_total = 0
            
            for mudra_class in self.mudra_classes:
                class_dir = self.base_path / split / mudra_class
                
                if class_dir.exists():
                    # Count image files
                    image_count = len(list(class_dir.glob('*.jpg'))) + \
                                len(list(class_dir.glob('*.png'))) + \
                                len(list(class_dir.glob('*.jpeg'))) + \
                                len(list(class_dir.glob('*.bmp')))
                    
                    print(f"  {mudra_class}: {image_count} images")
                    split_total += image_count
                else:
                    print(f"  {mudra_class}: Directory not found")
                    
            print(f"  Total: {split_total} images")
            total_samples += split_total
            
        print(f"\nGRAND TOTAL: {total_samples} images")
        
        # Check for empty directories
        print(f"\nEmpty Directories Check:")
        empty_dirs = []
        for split in splits:
            for mudra_class in self.mudra_classes:
                class_dir = self.base_path / split / mudra_class
                if class_dir.exists():
                    image_count = len(list(class_dir.glob('*')))
                    if image_count == 0:
                        empty_dirs.append(str(class_dir))
                        
        if empty_dirs:
            print("  Found empty directories:")
            for empty_dir in empty_dirs:
                print(f"    {empty_dir}")
        else:
            print("  No empty directories found ✓")


def create_sample_data():
    """Create sample data structure for demonstration"""
    print("Creating sample data structure...")
    
    preparer = DatasetPreparer()
    preparer.create_directory_structure()
    
    # Create some placeholder files for demonstration
    for split in ['train', 'validation', 'test']:
        for mudra_class in preparer.mudra_classes:
            class_dir = preparer.base_path / split / mudra_class
            
            # Create a placeholder text file
            placeholder_file = class_dir / 'README.txt'
            with open(placeholder_file, 'w') as f:
                f.write(f"Place {mudra_class} images here for {split} set.\n")
                f.write(f"Supported formats: .jpg, .jpeg, .png, .bmp\n")
                f.write(f"Recommended image size: 224x224 or larger\n")
                
    print("Sample data structure created!")
    print("Please add your mudra images to the respective directories.")


def main():
    """Main function with example usage"""
    preparer = DatasetPreparer()
    
    # Create directory structure
    preparer.create_directory_structure()
    
    # If you have a source directory with all images, you can split it
    # preparer.split_dataset('path/to/your/source/images')
    
    # Augment dataset if needed
    # preparer.augment_dataset(target_samples_per_class=1000)
    
    # Validate dataset
    preparer.validate_dataset()
    
    print("\nDataset preparation completed!")
    print("Next steps:")
    print("1. Add your mudra images to the respective directories")
    print("2. Run train_model.py to train the classification model")


if __name__ == "__main__":
    main()