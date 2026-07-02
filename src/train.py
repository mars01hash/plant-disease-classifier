"""
Main Training Script - Orchestrates the full ML pipeline.

This is the entry point for training. It:
1. Loads configuration
2. Loads data
3. Trains model
4. Evaluates model
5. Saves everything

Run with: python train.py
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from utils import load_config, setup_logging, save_model_metadata
from data_pipeline import ImageDataLoader, DataSplitter
from model_training import train_full_pipeline


def main():
    """
    Main training pipeline.
    
    My approach to building production systems:
    1. Load config (single source of truth)
    2. Setup logging (track what happens)
    3. Load and validate data
    4. Split into train/val/test
    5. Train model
    6. Evaluate on test set
    7. Save model + metadata
    """
    
    print("="*70)
    print("PLANT DISEASE CLASSIFIER - TRAINING PIPELINE")
    print("="*70)
    
    # ==================== CONFIGURATION ====================
    print("\n[1/6] Loading configuration...")
    config = load_config("./config.yaml")
    logger = setup_logging(config)
    
    logger.info("Configuration loaded successfully")
    logger.info(f"Classes: {config['classes']}")
    logger.info(f"Image size: {config['data']['image_size']}x{config['data']['image_size']}")
    
    # ==================== DATA LOADING ====================
    print("\n[2/6] Loading images from disk...")
    
    loader = ImageDataLoader(config, logger)
    images, labels, file_paths = loader.load_images(config["data"]["raw_dir"])
    
    logger.info(f"Loaded {len(images)} images with shape {images.shape}")
    
    # ==================== DATA SPLITTING ====================
    print("\n[3/6] Splitting data into train/val/test...")
    
    splitter = DataSplitter(config, logger)
    splits = splitter.split_data(images, labels, file_paths)
    
    # Extract split data
    train_images = splits["train"]["images"]
    train_labels = splits["train"]["labels"]
    val_images = splits["val"]["images"]
    val_labels = splits["val"]["labels"]
    test_images = splits["test"]["images"]
    test_labels = splits["test"]["labels"]
    
    logger.info(f"Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")
    
    # ==================== MODEL TRAINING ====================
    print("\n[4/6] Building and training model...")
    
    results = train_full_pipeline(
        config, logger,
        train_images, train_labels,
        val_images, val_labels,
        test_images, test_labels
    )
    
    model = results['model']
    metrics = results['metrics']
    
    # ==================== SAVE MODEL ====================
    print("\n[5/6] Saving model and metadata...")
    
    model_path = config["model"]["save_path"]
    model.save(model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Save metadata
    metadata_path = save_model_metadata(model_path, config, metrics, logger)
    
    # ==================== SUMMARY ====================
    print("\n[6/6] Training complete!")
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    
    print(f"\nModel saved to: {model_path}")
    print(f"Metadata saved to: {metadata_path}")
    
    print("\nKey Metrics:")
    print(f"  - Training Accuracy: {metrics['train_accuracy']:.4f}")
    print(f"  - Validation Accuracy: {metrics['val_accuracy']:.4f}")
    print(f"  - Test Accuracy: {metrics['test']['overall']['accuracy']:.4f}")
    print(f"  - Test Precision: {metrics['test']['overall']['precision']:.4f}")
    print(f"  - Test Recall: {metrics['test']['overall']['recall']:.4f}")
    print(f"  - Test F1 Score: {metrics['test']['overall']['f1_score']:.4f}")
    
    print("\n" + "="*70)
    print("Next steps:")
    print("1. Review metrics above")
    print("2. Start API server: python src/api.py")
    print("3. Test predictions at http://localhost:8000/docs")
    print("="*70)


if __name__ == "__main__":
    main()
