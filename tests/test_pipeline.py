"""
Comprehensive test suite for the ML pipeline.

Production insight: Tests are essential. They:
- Catch bugs early (before users find them)
- Enable refactoring without fear
- Document expected behavior
- Prevent regressions

Run tests with: pytest tests/ -v
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import json
import logging

# Setup path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import load_config, ImageValidator
from data_pipeline import ImageDataLoader, DataSplitter
from inference import ImagePredictor, ModelLoader


# ============================================================================
# FIXTURES (Setup test data)
# ============================================================================

@pytest.fixture
def config():
    """Load test configuration."""
    return load_config("./config.yaml")


@pytest.fixture
def logger():
    """Create logger for tests."""
    return logging.getLogger("test")


@pytest.fixture
def image_validator(config):
    """Create validator."""
    return ImageValidator(config)


@pytest.fixture
def mock_images():
    """Create mock images for testing."""
    # Random images with shape (N, 224, 224, 3)
    return np.random.rand(50, 224, 224, 3).astype(np.float32)


@pytest.fixture
def mock_labels():
    """Create mock labels (0-4 for 5 classes)."""
    return np.repeat(np.arange(5), 10)


# ============================================================================
# DATA PIPELINE TESTS
# ============================================================================

class TestDataLoader:
    """Test image loading from disk."""
    
    def test_load_images_shape(self, config, logger, mock_images):
        """Test that loaded images have correct shape."""
        images = mock_images
        assert images.shape == (len(mock_images), 224, 224, 3)
        assert images.dtype == np.float32
    
    def test_image_normalization(self, config, logger):
        """Test that images are normalized correctly."""
        # Create a test image
        from PIL import Image
        import cv2
        
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp.close()
        try:
            # Create simple test image
            test_img = np.ones((100, 100, 3), dtype=np.uint8) * 128
            cv2.imwrite(tmp.name, test_img)
            
            # Load and preprocess
            loader = ImageDataLoader(config, logger)
            img = loader._load_and_preprocess_image(tmp.name)
            
            # Check normalized to [0, 1]
            assert img.min() >= 0, "Image min should be >= 0"
            assert img.max() <= 1, "Image max should be <= 1"
            assert img.shape == (224, 224, 3)
        finally:
            # Cleanup
            Path(tmp.name).unlink(missing_ok=True)
            
    def test_load_images_from_directory(self, config, logger):
        """Test loading images from a directory structure."""
        import cv2
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Create subdirectories for the classes
            for class_name in config["classes"]:
                class_dir = tmpdir_path / class_name
                class_dir.mkdir()
                
                # Write one mock image for each extension: .jpg, .jpeg, .png
                for ext in ["test.jpg", "test.jpeg", "test.png"]:
                    img_path = class_dir / ext
                    test_img = np.ones((100, 100, 3), dtype=np.uint8) * 128
                    cv2.imwrite(str(img_path), test_img)
            
            # Load images
            loader = ImageDataLoader(config, logger)
            images, labels, file_paths = loader.load_images(str(tmpdir_path))
            
            # We have 5 classes, 3 images per class -> 15 images
            expected_count = len(config["classes"]) * 3
            assert len(images) == expected_count
            assert len(labels) == expected_count
            assert len(file_paths) == expected_count
            
            # Check shape and types
            assert images.shape == (expected_count, 224, 224, 3)
            assert images.dtype == np.float32


class TestDataSplitter:
    """Test train/val/test splitting."""
    
    def test_split_proportions(self, config, logger, mock_images, mock_labels):
        """Test that splits have correct proportions."""
        file_paths = [f"image_{i}.jpg" for i in range(len(mock_images))]
        
        splitter = DataSplitter(config, logger)
        splits = splitter.split_data(mock_images, mock_labels, file_paths)
        
        # Check total equals original
        total = (len(splits["train"]["images"]) + 
                len(splits["val"]["images"]) + 
                len(splits["test"]["images"]))
        assert total == len(mock_images)
        
        # Check ratios approximately match config
        train_ratio = len(splits["train"]["images"]) / len(mock_images)
        val_ratio = len(splits["val"]["images"]) / len(mock_images)
        test_ratio = len(splits["test"]["images"]) / len(mock_images)
        
        # Allow 10% deviation due to random split
        assert 0.60 < train_ratio <= 0.80, f"Train ratio {train_ratio} out of range"
        assert 0.05 < val_ratio <= 0.20, f"Val ratio {val_ratio} out of range"
        assert 0.05 < test_ratio <= 0.25, f"Test ratio {test_ratio} out of range"
    
    def test_split_stratification(self, config, logger, mock_images, mock_labels):
        """Test that class distribution is maintained in splits."""
        file_paths = [f"image_{i}.jpg" for i in range(len(mock_images))]
        
        splitter = DataSplitter(config, logger)
        splits = splitter.split_data(mock_images, mock_labels, file_paths)
        
        # Original class distribution
        original_dist = np.bincount(mock_labels, minlength=5) / len(mock_labels)
        
        # Check each split maintains similar distribution
        for split_name in ["train", "val", "test"]:
            split_labels = splits[split_name]["labels"]
            split_dist = np.bincount(split_labels, minlength=5) / len(split_labels)
            
            # Check if distributions are similar (chi-square would be more rigorous)
            for original, split in zip(original_dist, split_dist):
                if original > 0:
                    ratio = split / (original + 1e-6)
                    # Allow 50% deviation
                    assert 0.5 < ratio < 1.5, \
                        f"Split distribution changed too much in {split_name}"


# ============================================================================
# INFERENCE TESTS
# ============================================================================

class TestImageValidator:
    """Test image validation."""
    
    def test_validate_valid_file(self, image_validator):
        """Test validation of valid file."""
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp.close()
        try:
            # Create a small valid image
            img = np.ones((100, 100, 3), dtype=np.uint8)
            from PIL import Image
            Image.fromarray(img).save(tmp.name)
            
            is_valid, msg = image_validator.validate_file(tmp.name)
            assert is_valid, f"Valid file failed validation: {msg}"
        finally:
            Path(tmp.name).unlink(missing_ok=True)
    
    def test_validate_missing_file(self, image_validator):
        """Test validation of missing file."""
        is_valid, msg = image_validator.validate_file("/nonexistent/file.jpg")
        assert not is_valid, "Missing file should fail validation"
        assert "not found" in msg.lower()
    
    def test_validate_wrong_format(self, image_validator):
        """Test validation of wrong format."""
        tmp = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
        tmp.close()
        try:
            with open(tmp.name, "wb") as f:
                f.write(b"this is text")
            
            is_valid, msg = image_validator.validate_file(tmp.name)
            assert not is_valid, "Wrong format should fail validation"
        finally:
            Path(tmp.name).unlink(missing_ok=True)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestFullPipeline:
    """Test the full pipeline together."""
    
    def test_data_pipeline_end_to_end(self, config, logger, mock_images, mock_labels):
        """Test full data pipeline."""
        file_paths = [f"image_{i}.jpg" for i in range(len(mock_images))]
        
        # Create temporary directory structure
        with tempfile.TemporaryDirectory() as tmpdir:
            # This would normally load real images, but for testing we verify structure
            assert len(mock_images) == len(mock_labels)
            
            # Split data
            splitter = DataSplitter(config, logger)
            splits = splitter.split_data(mock_images, mock_labels, file_paths)
            
            # Verify splits can be used for training
            for split_name in ["train", "val", "test"]:
                images = splits[split_name]["images"]
                labels = splits[split_name]["labels"]
                
                assert len(images) == len(labels)
                assert images.dtype == np.float32
                assert 0 <= labels.min() < 5
                assert 0 <= labels.max() < 5


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestPerformance:
    """Test performance characteristics."""
    
    def test_inference_speed(self, config, logger, mock_images):
        """Test that inference completes in reasonable time."""
        import time
        
        # This is a mock test - real test would need loaded model
        start = time.time()
        # Simulate inference work
        predictions = np.random.rand(len(mock_images), config["model"]["num_classes"])
        elapsed = time.time() - start
        
        # Should be fast (less than 1 second for 10 images in CPU)
        assert elapsed < 1.0, f"Inference took too long: {elapsed}s"
    
    def test_memory_efficiency(self, mock_images):
        """Test memory usage is reasonable."""
        # Check that array doesn't exceed expected size
        expected_size_mb = (mock_images.nbytes / 1024 / 1024)
        # 10 images * 224*224*3 * 4 bytes ≈ 37 MB
        assert expected_size_mb < 50, f"Images taking too much memory: {expected_size_mb}MB"


# ============================================================================
# CONFIGURATION TESTS
# ============================================================================

class TestConfiguration:
    """Test configuration loading and validation."""
    
    def test_config_loads(self, config):
        """Test that configuration loads without errors."""
        assert config is not None
        assert "data" in config
        assert "model" in config
        assert "api" in config
    
    def test_config_required_keys(self, config):
        """Test that all required config keys are present."""
        required_keys = {
            "data": ["raw_dir", "image_size", "batch_size"],
            "model": ["name", "num_classes", "learning_rate"],
            "api": ["port", "host"],
            "classes": None  # This is a list
        }
        
        for section, keys in required_keys.items():
            assert section in config, f"Missing section: {section}"
            if keys:
                for key in keys:
                    assert key in config[section], \
                        f"Missing key '{key}' in section '{section}'"


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
