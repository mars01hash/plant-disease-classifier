"""
FastAPI Application for Serving ML Model Predictions.

Production approach:
- Pydantic models for request/response validation
- Proper error handling with HTTP status codes
- Logging for all requests
- Health checks for monitoring
- Documentation (Swagger UI)
- File upload handling
"""

import logging
from typing import List, Optional
from pathlib import Path
import json
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Import our modules
from utils import load_config, setup_logging, ImageValidator
from inference import ModelLoader, ImagePredictor, BatchPredictor


# ============================================================================
# PYDANTIC MODELS (Request/Response validation)
# ============================================================================
# Pydantic automatically validates incoming data and documents API

class PredictionResponse(BaseModel):
    """Response from prediction."""
    success: bool
    class_name: str
    confidence: float
    is_confident: bool
    all_predictions: dict = Field(description="Confidence for all classes")
    metadata: dict = Field(description="Prediction metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "class_name": "powdery_mildew",
                "confidence": 0.92,
                "is_confident": True,
                "all_predictions": {
                    "healthy": 0.02,
                    "powdery_mildew": 0.92,
                    "rust": 0.03,
                    "leaf_spot": 0.02,
                    "blight": 0.01
                },
                "metadata": {
                    "model_version": "v1.0",
                    "processing_time_ms": 125
                }
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    model_loaded: bool
    version: str = "1.0"
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-15T10:30:00Z",
                "model_loaded": True,
                "version": "1.0"
            }
        }


class BatchPredictionResponse(BaseModel):
    """Response from batch prediction."""
    success: bool
    predictions: List[dict]
    summary: dict = Field(description="Summary statistics for batch")


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

class MLApp:
    """
    Production ML API wrapper.
    
    This pattern is useful because:
    - Can manage app lifecycle (startup, shutdown)
    - Can hold state (loaded model, config)
    - Easier to test
    """
    
    def __init__(self, config_path: str = "./config.yaml"):
        # Load configuration
        self.config = load_config(config_path)
        
        # Setup logging
        self.logger = setup_logging(self.config)
        
        # Create FastAPI app
        self.app = FastAPI(
            title="Plant Disease Classifier",
            description="Classify plant diseases from images using deep learning",
            version="1.0.0"
        )
        
        # Initialize components (will be loaded at startup)
        self.model = None
        self.predictor = None
        self.batch_predictor = None
        self.validator = ImageValidator(self.config)
        
        # Setup routes
        self._setup_routes()
        
        # Setup lifecycle events
        self._setup_lifecycle_events()
        
        self.logger.info("MLApp initialized")
    
    def _setup_lifecycle_events(self):
        """Setup startup and shutdown events."""
        
        @self.app.on_event("startup")
        async def startup():
            """Load model and prepare for requests."""
            self.logger.info("=" * 60)
            self.logger.info("STARTING UP APPLICATION")
            self.logger.info("=" * 60)
            
            try:
                # Load model
                loader = ModelLoader(self.logger)
                model_path = self.config["model"]["save_path"]
                self.model = loader.load_model(model_path)
                
                # Create predictors
                self.predictor = ImagePredictor(
                    self.model, self.config, self.logger, self.config["classes"]
                )
                
                self.batch_predictor = BatchPredictor(
                    self.model, self.config, self.logger, self.config["classes"]
                )
                
                self.logger.info("Model loaded successfully at startup")
                self.logger.info("API is ready for requests!")
                
            except Exception as e:
                self.logger.error(f"Failed to load model: {e}")
                raise
        
        @self.app.on_event("shutdown")
        async def shutdown():
            """Cleanup on shutdown."""
            self.logger.info("Shutting down application")
            self.logger.info("=" * 60)
    
    def _setup_routes(self):
        """Setup API routes."""
        
        # ====== HEALTH CHECK ======
        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Check if API is running and model is loaded."""
            return {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "model_loaded": self.model is not None,
                "version": "1.0"
            }
        
        # ====== SINGLE IMAGE PREDICTION ======
        @self.app.post("/predict", response_model=PredictionResponse)
        async def predict(file: UploadFile = File(...)):
            """
            Predict disease class for a single image.
            
            - **file**: Image file (JPG, PNG)
            - **returns**: Prediction with confidence scores
            
            Production insight: 
            - Validate file first
            - Handle errors with proper HTTP status codes
            - Return detailed information for debugging
            """
            import time
            
            start_time = time.time()
            
            try:
                # Validate uploaded file
                if file.filename is None:
                    raise HTTPException(
                        status_code=400,
                        detail="No filename provided"
                    )
                
                # Check file size before saving
                contents = await file.read()
                if len(contents) > self.config["api"]["max_image_size_mb"] * 1024 * 1024:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Max: {self.config['api']['max_image_size_mb']}MB"
                    )
                
                # Check file format
                extension = Path(file.filename).suffix.lower().lstrip('.')
                if extension not in self.config["api"]["allowed_formats"]:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid format: {extension}. Allowed: {self.config['api']['allowed_formats']}"
                    )
                
                # Save temporarily
                temp_path = f"/tmp/{file.filename}"
                with open(temp_path, 'wb') as f:
                    f.write(contents)
                
                # Make prediction
                result = self.predictor.predict_from_file(temp_path)
                
                # Cleanup
                Path(temp_path).unlink()
                
                # Calculate processing time
                processing_time_ms = (time.time() - start_time) * 1000
                
                # Format response
                return {
                    "success": True,
                    "class_name": result['prediction']['class'],
                    "confidence": result['prediction']['confidence'],
                    "is_confident": result['prediction']['is_confident'],
                    "all_predictions": result['all_predictions'],
                    "metadata": {
                        "model_version": result['metadata']['model_version'],
                        "processing_time_ms": processing_time_ms
                    }
                }
            
            except HTTPException as e:
                self.logger.error(f"HTTP error: {e.detail}")
                raise e
            
            except Exception as e:
                self.logger.error(f"Unexpected error during prediction: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Prediction failed: {str(e)}"
                )
        
        # ====== BATCH PREDICTION ======
        @self.app.post("/predict-batch", response_model=BatchPredictionResponse)
        async def predict_batch(files: List[UploadFile] = File(...)):
            """
            Make predictions for multiple images at once.
            
            Production insight: Batch processing is much faster than individual requests.
            """
            import time
            
            start_time = time.time()
            
            try:
                if not files:
                    raise HTTPException(
                        status_code=400,
                        detail="No files provided"
                    )
                
                if len(files) > self.config["api"]["max_batch_size"]:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Too many files. Max: {self.config['api']['max_batch_size']}"
                    )
                
                # Save all files temporarily
                temp_paths = []
                for file in files:
                    contents = await file.read()
                    temp_path = f"/tmp/{file.filename}"
                    with open(temp_path, 'wb') as f:
                        f.write(contents)
                    temp_paths.append(temp_path)
                
                # Batch prediction
                results = self.batch_predictor.predict_batch(temp_paths)
                
                # Cleanup
                for temp_path in temp_paths:
                    Path(temp_path).unlink()
                
                # Summary statistics
                successful = [r for r in results if r.get('success', False)]
                summary = {
                    "total": len(results),
                    "successful": len(successful),
                    "failed": len(results) - len(successful),
                    "processing_time_ms": (time.time() - start_time) * 1000
                }
                
                return {
                    "success": len(successful) == len(results),
                    "predictions": results,
                    "summary": summary
                }
            
            except HTTPException as e:
                raise e
            except Exception as e:
                self.logger.error(f"Batch prediction error: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Batch prediction failed: {str(e)}"
                )
        
        # ====== MODEL INFO ======
        @self.app.get("/model-info")
        async def model_info():
            """Get information about loaded model."""
            if self.model is None:
                raise HTTPException(
                    status_code=503,
                    detail="Model not loaded"
                )
            
            return {
                "model_type": "ResNet50 (Transfer Learning)",
                "num_classes": self.config["model"]["num_classes"],
                "classes": self.config["classes"],
                "input_size": self.config["data"]["image_size"],
                "version": "1.0",
                "description": "Plant disease classification model trained on agricultural dataset"
            }


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Start the API server."""
    
    # Create app
    ml_app = MLApp("./config.yaml")
    app = ml_app.app
    
    # Run with Uvicorn
    # Production deployment would use gunicorn + uvicorn workers
    uvicorn.run(
        app,
        host=ml_app.config["api"]["host"],
        port=ml_app.config["api"]["port"],
        workers=ml_app.config["api"]["workers"],
        log_level="info"
    )


if __name__ == "__main__":
    main()
