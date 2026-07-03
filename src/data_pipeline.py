"""
Data Pipeline for Image Classification.

Production thinking:
- Data is the foundation of ML. Spend 80% of time here, 20% on models.
- Pipeline must be reproducible (same input → same output every time)
- Pipeline must be efficient (handle large datasets)
- Pipeline must be debuggable (what went wrong? why?)
"""

import os
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
import logging
import pickle

from sklearn.model_selection import train_test_split
from PIL import Image
import cv2
from tensorflow import keras

# ============================================================================
# DATA LOADER - Load images from disk efficiently
# ============================================================================

class ImageDataLoader:
    """
    Loads images from directory structure.
    
    Assumes directory structure:
    data/
    ├── disease1/
    │   ├── image1.jpg
    │   ├── image2.jpg
    ├── disease2/
    │   ├── image1.jpg
    """
    
    def __init__(self, config: Dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.image_size = config["data"]["image_size"]
        self.classes = config["classes"]
    
    def load_images(self, data_dir: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load all images from directory structure.
        
        Returns:
            images: array of shape (N, height, width, 3)
            labels: array of shape (N,) with class indices
            file_paths: list of original file paths (for debugging)
        
        Production insight: Keep file paths for debugging. If a prediction is wrong,
        you need to know which image it was!
        """
        self.logger.info(f"Loading images from {data_dir}")
        
        images = []
        labels = []
        file_paths = []
        
        # Iterate through class directories
        for class_idx, class_name in enumerate(self.classes):
            class_dir = Path(data_dir) / class_name
            
            if not class_dir.exists():
                self.logger.warning(f"Class directory not found: {class_dir}")
                continue
            
            # Load all images in this class
            image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.png"))
            for image_file in image_files:
                try:
                    # Load and preprocess image
                    image = self._load_and_preprocess_image(str(image_file))
                    
                    if image is not None:
                        images.append(image)
                        labels.append(class_idx)
                        file_paths.append(str(image_file))
                
                except Exception as e:
                    self.logger.warning(f"Failed to load {image_file}: {str(e)}")
                    continue
            
            self.logger.info(f"Loaded {len([l for l in labels if l == class_idx])} images for class '{class_name}'")
        
        # Convert to numpy arrays
        images = np.array(images, dtype=np.float32)
        labels = np.array(labels)
        
        self.logger.info(f"Total loaded: {len(images)} images")
        self.logger.info(f"Data shape: {images.shape}, Label shape: {labels.shape}")
        
        return images, labels, file_paths
    
    def _load_and_preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Load image and preprocess it.
        
        Steps:
        1. Load from disk
        2. Resize to standard size
        3. Normalize pixel values
        
        Production insight: Consistent preprocessing is CRITICAL.
        If you preprocess differently during training vs inference, 
        your model will perform poorly in production.
        """
        # Load image
        image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Convert BGR to RGB (OpenCV loads as BGR by default)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to standard size
        image = cv2.resize(image, (self.image_size, self.image_size))
        
        # Normalize pixel values to [0, 1]
        # This is CRITICAL - ResNet expects normalized inputs
        image = image.astype(np.float32) / 255.0
        
        return image


# ============================================================================
# DATA AUGMENTATION - Prevent overfitting
# ============================================================================

class DataAugmentor:
    """
    Apply data augmentation during training.
    
    Production insight: Augmentation prevents overfitting and makes model
    more robust to variations in real-world images.
    
    Why augment:
    - More training data without collecting more images
    - Makes model invariant to rotations, brightness changes, etc.
    - Models trained with augmentation generalize better
    """
    
    def __init__(self, config: Dict, logger: logging.Logger):
        self.config = config["data"]["augmentation"]
        self.logger = logger
    
    def augment_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply random augmentations to image.
        
        Only use during TRAINING, not during validation/testing.
        """
        # Random rotation
        angle = np.random.uniform(-self.config["rotation_range"], 
                                  self.config["rotation_range"])
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, rotation_matrix, (width, height))
        
        # Random zoom
        if self.config["zoom_range"] > 0:
            zoom_factor = np.random.uniform(1 - self.config["zoom_range"], 
                                           1 + self.config["zoom_range"])
            new_size = int(height * zoom_factor)
            if new_size > 0:
                resized = cv2.resize(image, (new_size, new_size))
                # Crop or pad to original size
                if new_size > height:
                    start = (new_size - height) // 2
                    image = resized[start:start+height, start:start+width]
                else:
                    pad_top = (height - new_size) // 2
                    pad_bottom = height - new_size - pad_top
                    pad_left = (width - new_size) // 2
                    pad_right = width - new_size - pad_left
                    image = cv2.copyMakeBorder(resized, pad_top, pad_bottom, 
                                              pad_left, pad_right, cv2.BORDER_REFLECT)
        
        # Random horizontal flip
        if self.config["horizontal_flip"] and np.random.random() > 0.5:
            image = cv2.flip(image, 1)
        
        # Random brightness
        if self.config["brightness_range"]:
            brightness_factor = np.random.uniform(self.config["brightness_range"][0],
                                                 self.config["brightness_range"][1])
            image = np.clip(image * brightness_factor, 0, 1)
        
        return image


# ============================================================================
# DATA SPLITTER - Create train/val/test sets
# ============================================================================

class DataSplitter:
    """
    Split data into train, validation, and test sets.
    
    Production insight:
    - NEVER tune hyperparameters on test set
    - Never look at test set during training
    - Use validation set to pick best model
    - Use test set only for final evaluation
    
    Standard splits: 70% train, 15% val, 15% test
    """
    
    def __init__(self, config: Dict, logger: logging.Logger, random_state: int = 42):
        self.config = config
        self.logger = logger
        self.random_state = random_state  # For reproducibility
    
    def split_data(self, images: np.ndarray, labels: np.ndarray, 
                   file_paths: List[str]) -> Dict:
        """
        Split into train, validation, and test sets.
        
        Returns:
            Dict with 'train', 'val', 'test' keys, each containing:
            - images, labels, file_paths
        """
        self.logger.info("Splitting data into train/val/test")
        
        # First split: train+val vs test (80% vs 20%)
        train_val_idx, test_idx = train_test_split(
            range(len(images)),
            test_size=1 - self.config["data"]["train_test_split"],
            random_state=self.random_state,
            stratify=labels  # Keep class distribution same in splits
        )
        
        # Second split: train vs val (80% of train_val becomes train, 20% becomes val)
        val_fraction = self.config["data"]["val_split"] / self.config["data"]["train_test_split"]
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_fraction,
            random_state=self.random_state,
            stratify=labels[train_val_idx]
        )
        
        # Create split dictionaries
        splits = {
            "train": {
                "images": images[train_idx],
                "labels": labels[train_idx],
                "file_paths": [file_paths[i] for i in train_idx]
            },
            "val": {
                "images": images[val_idx],
                "labels": labels[val_idx],
                "file_paths": [file_paths[i] for i in val_idx]
            },
            "test": {
                "images": images[test_idx],
                "labels": labels[test_idx],
                "file_paths": [file_paths[i] for i in test_idx]
            }
        }
        
        # Log split statistics
        for split_name, split_data in splits.items():
            self.logger.info(f"{split_name.upper()}: {len(split_data['images'])} images")
            # Show class distribution
            unique, counts = np.unique(split_data['labels'], return_counts=True)
            for class_idx, count in zip(unique, counts):
                self.logger.info(f"  Class {self.config['classes'][class_idx]}: {count}")
        
        return splits


# ============================================================================
# DATA GENERATOR - Efficient batching for training
# ============================================================================

class DataGenerator(keras.utils.Sequence):
    """
    Generate batches of data for training.
    
    Production insight: Don't load all data into memory. Generate batches
    on-the-fly. This allows training on datasets larger than RAM.
    """
    
    def __init__(self, images: np.ndarray, labels: np.ndarray, 
                 batch_size: int, augment: bool = False, 
                 augmentor: DataAugmentor = None):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.augment = augment
        self.augmentor = augmentor
        self.num_samples = len(images)
        self._num_batches = int(np.ceil(self.num_samples / batch_size))
        
        # Shuffle indices for randomness
        self.indices = np.arange(self.num_samples)
        self.on_epoch_end()
    
    def __len__(self) -> int:
        """Number of batches."""
        return self._num_batches
        
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get batch at index idx."""
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.num_samples)
        batch_indices = self.indices[start_idx:end_idx]
        
        # Get batch data
        batch_images = self.images[batch_indices].copy()
        batch_labels = self.labels[batch_indices]
        
        # Apply augmentation if training
        if self.augment and self.augmentor:
            batch_images = np.array([
                self.augmentor.augment_image(img) for img in batch_images
            ])
        
        # Convert labels to one-hot encoding
        num_classes = len(np.unique(self.labels))
        batch_labels_onehot = np.eye(num_classes)[batch_labels]
        
        return batch_images, batch_labels_onehot
        
    def on_epoch_end(self):
        """Shuffle indices at the end of each epoch."""
        np.random.shuffle(self.indices)
