"""
Inference Module for Making Predictions.

This is what gets called when the API receives a request.
Key considerations:
- Must be fast (users wait for response)
- Must be thread-safe (multiple requests at same time)
- Must handle errors gracefully
- Must be memory efficient
"""

import numpy as np
from typing import Dict, List, Tuple
import logging
from pathlib import Path
import cv2

import tensorflow as tf

# ============================================================================
# MODEL LOADER
# ============================================================================

class ModelLoader:
    """
    Load and cache model.
    
    Production insight: Loading a model is expensive (~1 second).
    Load once at startup, reuse for all predictions.
    This is why you don't load model in every prediction function!
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._model = None  # Cache the model
        self._loaded_path = None
    
    def load_model(self, model_path: str) -> tf.keras.Model:
        """
        Load model from disk (or return cached version if already loaded).
        """
        # Return cached model if already loaded
        if self._model is not None and self._loaded_path == model_path:
            self.logger.debug(f"Using cached model from {model_path}")
            return self._model
        
        self.logger.info(f"Loading model from {model_path}")
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load model
        model = tf.keras.models.load_model(model_path)
        
        # Cache it
        self._model = model
        self._loaded_path = model_path
        
        self.logger.info(f"Model loaded successfully. Parameters: {model.count_params():,}")
        
        return model


# ============================================================================
# PREDICTOR
# ============================================================================

class ImagePredictor:
    """
    Make predictions on images.
    
    Core logic:
    1. Load image from disk or accept as array
    2. Preprocess (resize, normalize)
    3. Run inference
    4. Convert to human-readable format
    """
    
    def __init__(self, model: tf.keras.Model, config: Dict, 
                 logger: logging.Logger, classes: List[str]):
        self.model = model
        self.config = config
        self.logger = logger
        self.classes = classes
        self.image_size = config["data"]["image_size"]
        self.confidence_threshold = config["api"]["confidence_threshold"]
    
    def predict_from_file(self, image_path: str) -> Dict:
        """
        Make prediction from image file path.
        
        Returns:
            {
                'class': 'powdery_mildew',
                'confidence': 0.92,
                'all_predictions': {'healthy': 0.02, 'powdery_mildew': 0.92, ...},
                'is_confident': True,
                'model_version': '1.0'
            }
        """
        # Load image
        image = self._load_image(image_path)
        
        # Make prediction
        return self.predict_from_array(image, source=image_path)
    
    def predict_from_array(self, image: np.ndarray, source: str = "array") -> Dict:
        """
        Make prediction from image array.
        
        Production insight: Accept both file paths and arrays for flexibility.
        """
        # Preprocess image
        processed_image = self._preprocess_image(image)
        
        # Add batch dimension (model expects batch)
        batch = np.expand_dims(processed_image, axis=0)
        
        # Run inference
        predictions = self.model.predict(batch, verbose=0)[0]
        
        # Get top prediction
        predicted_class_idx = np.argmax(predictions)
        confidence = float(predictions[predicted_class_idx])
        predicted_class = self.classes[predicted_class_idx]
        
        # Check if confident enough
        is_confident = confidence >= self.confidence_threshold
        
        # All predictions (for debugging/understanding)
        all_predictions = {
            class_name: float(pred)
            for class_name, pred in zip(self.classes, predictions)
        }
        
        # Format response
        result = {
            'prediction': {
                'class': predicted_class,
                'confidence': confidence,
                'is_confident': is_confident,
            },
            'all_predictions': all_predictions,
            'metadata': {
                'image_source': source,
                'model_version': 'v1.0',
                'confidence_threshold': self.confidence_threshold,
            }
        }
        
        # Log prediction
        self.logger.info(f"Prediction: {predicted_class} ({confidence:.2%}) from {source}")
        
        return result
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """
        Load image from disk.
        
        Production insight: Handle different formats and validate.
        """
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load with OpenCV
        image = cv2.imread(str(image_path))
        
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}. Check format.")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for model.
        
        CRITICAL: This must match preprocessing used during training!
        If different:
        - During training: image resized to 224x224, normalized to [0,1]
        - During inference: image resized differently = wrong predictions
        
        This is one of the most common bugs in production ML systems.
        """
        # Resize to model input size
        image = cv2.resize(image, (self.image_size, self.image_size))
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        return image


# ============================================================================
# BATCH PREDICTOR
# ============================================================================

class BatchPredictor:
    """
    Make predictions on multiple images efficiently.
    
    Production insight: Batch processing is much faster than individual predictions.
    Benefits:
    - Better GPU utilization
    - Amortizes model loading cost
    - Makes API responses faster
    """
    
    def __init__(self, model: tf.keras.Model, config: Dict,
                 logger: logging.Logger, classes: List[str]):
        self.predictor = ImagePredictor(model, config, logger, classes)
        self.config = config
        self.logger = logger
    
    def predict_batch(self, image_paths: List[str]) -> List[Dict]:
        """
        Make predictions on multiple images.
        
        Returns:
            List of prediction dictionaries
        """
        max_batch = self.config["api"]["max_batch_size"]
        
        if len(image_paths) > max_batch:
            self.logger.warning(
                f"Batch size {len(image_paths)} exceeds max {max_batch}. "
                f"Processing only first {max_batch}"
            )
            image_paths = image_paths[:max_batch]
        
        results = []
        
        # Load all images
        images = []
        for image_path in image_paths:
            try:
                image = self.predictor._load_image(image_path)
                images.append(image)
            except Exception as e:
                self.logger.error(f"Failed to load {image_path}: {e}")
                results.append({
                    'image_path': image_path,
                    'error': str(e),
                    'success': False
                })
                continue
        
        # Preprocess all images
        processed_images = np.array([
            self.predictor._preprocess_image(img) for img in images
        ])
        
        # Batch inference (much faster than individual predictions)
        batch_predictions = self.predictor.model.predict(processed_images, verbose=0)
        
        # Format results
        for image_path, predictions in zip(
            [ip for ip in image_paths if Path(ip).exists()],
            batch_predictions
        ):
            predicted_class_idx = np.argmax(predictions)
            confidence = float(predictions[predicted_class_idx])
            
            results.append({
                'image_path': image_path,
                'class': self.predictor.classes[predicted_class_idx],
                'confidence': confidence,
                'all_predictions': {
                    class_name: float(pred)
                    for class_name, pred in zip(self.predictor.classes, predictions)
                },
                'success': True
            })
        
        self.logger.info(f"Batch prediction completed for {len(results)} images")
        
        return results


# ============================================================================
# INFERENCE CACHE (Optional but recommended for production)
# ============================================================================

class PredictionCache:
    """
    Cache predictions to avoid redundant computation.
    
    Production insight: Many users might send the same image.
    Caching saves computation and makes API faster.
    """
    
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
    
    def get(self, image_hash: str) -> Dict = None:
        """Get cached prediction if exists."""
        return self.cache.get(image_hash)
    
    def set(self, image_hash: str, prediction: Dict):
        """Cache prediction."""
        if len(self.cache) >= self.max_size:
            # Remove oldest entry (simple FIFO)
            self.cache.pop(next(iter(self.cache)))
        
        self.cache[image_hash] = prediction
    
    def hash_image(self, image_path: str) -> str:
        """Generate hash of image for caching."""
        import hashlib
        
        with open(image_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
