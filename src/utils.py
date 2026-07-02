"""
Utility functions and configuration management.
This is production-level utilities for logging, config, and helpers.
"""

import yaml
import logging
import os
from pathlib import Path
from typing import Dict, Any
import json

# ============================================================================
# LOGGING SETUP
# ============================================================================
# Production systems MUST have structured logging.
# Why? So you can track what happened when something breaks in production.

def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """
    Setup logging with both file and console handlers.
    
    Production insight:
    - File logging: persists events for debugging production issues
    - Console logging: allows developers to see real-time events
    - Structured format: makes logs machine-readable for monitoring systems
    """
    logger = logging.getLogger("plant_classifier")
    logger.setLevel(getattr(logging, config["logging"]["level"]))
    
    # Create logs directory if it doesn't exist
    log_dir = Path(config["logging"]["file"]).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Format: timestamp - logger_name - level - message
    formatter = logging.Formatter(config["logging"]["format"])
    
    # File handler (persists logs)
    fh = logging.FileHandler(config["logging"]["file"])
    fh.setLevel(getattr(logging, config["logging"]["level"]))
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # Console handler (real-time visibility)
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, config["logging"]["level"]))
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger


# ============================================================================
# CONFIG LOADER
# ============================================================================

def load_config(config_path: str = "./config.yaml") -> Dict[str, Any]:
    """
    Load and validate configuration from YAML file.
    
    Production insight:
    - Single source of truth for all settings
    - Easier to test different configs (change file, not code)
    - Can be overridden by environment variables in production
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create necessary directories
    for directory in [config["data"]["raw_dir"], 
                      config["data"]["processed_dir"],
                      Path(config["model"]["save_path"]).parent,
                      Path(config["logging"]["file"]).parent]:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    return config


# ============================================================================
# DATA VALIDATION HELPERS
# ============================================================================

class ImageValidator:
    """
    Validates incoming image data before processing.
    
    Production insight: NEVER trust user input. Always validate.
    This prevents:
    - Malformed images crashing your service
    - Memory exhaustion from huge images
    - Invalid formats wasting processing
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.max_size_mb = config["api"]["max_image_size_mb"]
        self.allowed_formats = config["api"]["allowed_formats"]
        self.max_size_bytes = self.max_size_mb * 1024 * 1024
    
    def validate_file(self, file_path: str) -> tuple[bool, str]:
        """
        Validate image file.
        
        Returns:
            (is_valid, error_message)
        """
        # Check file exists
        if not Path(file_path).exists():
            return False, f"File not found: {file_path}"
        
        # Check file size
        file_size = Path(file_path).stat().st_size
        if file_size > self.max_size_bytes:
            return False, f"File too large: {file_size/1024/1024:.2f}MB (max: {self.max_size_mb}MB)"
        
        # Check file format
        extension = Path(file_path).suffix.lower().lstrip('.')
        if extension not in self.allowed_formats:
            return False, f"Invalid format: {extension}. Allowed: {self.allowed_formats}"
        
        return True, "Valid"
    
    def validate_image_content(self, image) -> tuple[bool, str]:
        """
        Validate image content (shape, values, etc).
        """
        import numpy as np
        
        if image is None:
            return False, "Image is None"
        
        if len(image.shape) != 3 or image.shape[2] not in [3, 4]:
            return False, f"Invalid image shape: {image.shape}. Expected (H, W, 3) or (H, W, 4)"
        
        return True, "Valid"


# ============================================================================
# ERROR TRACKING & MONITORING
# ============================================================================

class ErrorTracker:
    """
    Track errors for monitoring and alerting.
    
    Production insight: Know when your system is having issues BEFORE users report it.
    """
    
    def __init__(self, log_file: str = "./logs/errors.jsonl"):
        self.log_file = log_file
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    def log_error(self, error_type: str, message: str, context: Dict = None):
        """Log error in JSON format for structured monitoring."""
        import datetime
        
        error_record = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "error_type": error_type,
            "message": message,
            "context": context or {}
        }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(error_record) + "\n")


# ============================================================================
# MODEL UTILITIES
# ============================================================================

def save_model_metadata(model_path: str, config: Dict[str, Any], 
                        metrics: Dict[str, float], logger: logging.Logger):
    """
    Save model metadata (config, performance metrics) alongside the model.
    
    Production insight: When you load a model 6 months later, you need to know:
    - What config was it trained with?
    - What was its performance?
    - When was it trained?
    """
    import datetime
    
    metadata = {
        "model_path": model_path,
        "training_config": {
            "learning_rate": config["model"]["learning_rate"],
            "epochs": config["model"]["epochs"],
            "batch_size": config["data"]["batch_size"],
        },
        "performance_metrics": metrics,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "classes": config["classes"]
    }
    
    metadata_path = model_path.replace(".h5", "_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved model metadata to {metadata_path}")
    return metadata_path


def load_model_metadata(model_path: str) -> Dict[str, Any]:
    """Load metadata saved with the model."""
    metadata_path = model_path.replace(".h5", "_metadata.json")
    
    if not Path(metadata_path).exists():
        return {}
    
    with open(metadata_path, 'r') as f:
        return json.load(f)
