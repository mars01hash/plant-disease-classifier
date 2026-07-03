"""
ENHANCED FastAPI Application - With Production Features

This is how to integrate the 4 new features into your existing api.py:
1. Prediction Database
2. Advanced Logging
3. Prediction Caching
4. API Authentication

Just copy the highlighted sections into your api.py
"""

import logging
import time
import traceback
import tempfile
from typing import List
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Header, Request, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Import our new modules (add these!)
from utils import load_config, setup_logging, ImageValidator
from inference import ModelLoader, ImagePredictor, BatchPredictor
from prediction_database import PredictionDatabase  # ✨ NEW
from advanced_logging import StructuredLogger, AnalyticsAggregator  # ✨ NEW
from prediction_caching import PredictionCache, ResponseTimeOptimizer  # ✨ NEW
from authentication import APIKeyManager, RateLimiter, setup_auth  # ✨ NEW


# ============================================================================
# RESPONSE MODELS (unchanged)
# ============================================================================

class PredictionResponse(BaseModel):
    """Response from prediction."""
    success: bool
    class_name: str
    confidence: float
    is_confident: bool
    all_predictions: dict
    prediction_id: int = Field(description="Use this for feedback")  # ✨ NEW
    rate_limit: dict = Field(description="Rate limit info")  # ✨ NEW
    metadata: dict


# ============================================================================
# ENHANCED ML APPLICATION
# ============================================================================

class EnhancedMLApp:
    """
    Production ML API with:
    - Database (store predictions)
    - Logging (track events)
    - Caching (speed up)
    - Authentication (security)
    """
    
    def __init__(self, config_path: str = "./config.yaml"):
        # Load configuration
        self.config = load_config(config_path)
        
        # Setup logging
        self.logger = setup_logging(self.config)
        
        # Create FastAPI app
        self.app = FastAPI(
            title="Plant Disease Classifier - Production Edition",
            description="ML API with database, logging, caching, and auth",
            version="2.0.0"
        )
        
        # ✨ NEW: Initialize production components
        self.db = PredictionDatabase(logger=self.logger)
        self.analytics = StructuredLogger("./logs/events.jsonl")
        self.cache = PredictionCache(max_size=1000, ttl_hours=24)
        self.optimizer = ResponseTimeOptimizer()
        self.key_manager = APIKeyManager()
        self.limiter = RateLimiter(
            requests_per_hour=self.config.get("api", {}).get("rate_limit", 1000)
        )
        
        # Initialize model components
        self.model = None
        self.predictor = None
        self.batch_predictor = None
        self.validator = ImageValidator(self.config)
        
        # Setup auth
        self.verify_api_key, self.verify_rate_limit = setup_auth(
            self.app, self.key_manager, self.limiter
        )
        
        # Setup routes
        self._setup_routes()
        
        # Setup lifecycle events
        self._setup_lifecycle_events()
        
        self.logger.info("EnhancedMLApp initialized with production features")
    
    def _setup_lifecycle_events(self):
        """Setup startup and shutdown events."""
        
        @self.app.on_event("startup")
        async def startup():
            """Load model and prepare for requests."""
            self.logger.info("="*60)
            self.logger.info("STARTING UP - PRODUCTION MODE")
            self.logger.info("="*60)
            
            try:
                loader = ModelLoader(self.logger)
                model_path = self.config["model"]["save_path"]
                self.model = loader.load_model(model_path)
                
                self.predictor = ImagePredictor(
                    self.model, self.config, self.logger, self.config["classes"]
                )
                
                self.batch_predictor = BatchPredictor(
                    self.model, self.config, self.logger, self.config["classes"]
                )
                
                self.logger.info("✓ Model loaded")
                self.logger.info("✓ Database initialized")
                self.logger.info("✓ Logging enabled")
                self.logger.info("✓ Caching enabled")
                self.logger.info("✓ Authentication enabled")
                self.logger.info("API is ready for requests!")
            
            except Exception as e:
                self.logger.error(f"Failed to startup: {e}")
                self.analytics.log_error(f"Startup failed: {e}", traceback.format_exc())
                raise
        
        @self.app.on_event("shutdown")
        async def shutdown():
            """Cleanup on shutdown."""
            self.logger.info("Shutting down application")
            self.analytics.log_system_health({
                'total_predictions': self.db.get_statistics()['total_predictions'],
                'cache_stats': self.cache.get_stats()
            })
    
    def _setup_routes(self):
        """Setup API routes."""
        
        # ====== HEALTH CHECK ======
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            stats = self.cache.get_stats()
            return {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "model_loaded": self.model is not None,
                "cache_hit_rate": stats.get('hit_rate', 0),
                "version": "2.0.0"
            }
        
        # ====== ENHANCED /predict ======
        @self.app.post("/predict", response_model=PredictionResponse)
        async def predict(
            file: UploadFile = File(...),
            auth: dict = Depends(self.verify_api_key),
            request: Request = None,  # ✨ NEW: For rate limiting
            background_tasks: BackgroundTasks = None  # ✨ NEW: For async logging
        ):
            """
            Predict disease class for a single image.
            
            ✨ NEW FEATURES:
            - Authentication (X-API-Key header required)
            - Rate limiting (requests per hour)
            - Database storage (predictions saved)
            - Caching (repeated images are instant)
            - Logging (all events tracked)
            """
            
            start_time = time.time()
            
            try:
                user_id = auth['user_id']
                x_api_key = auth['key_info']['key']
                
                # ✨ NEW: Check rate limit
                allowed, rate_stats = self.limiter.is_allowed(user_id)
                if not allowed:
                    self.analytics.log_error(
                        "Rate limit exceeded",
                        context={'user_id': user_id, 'stats': rate_stats}
                    )
                    raise HTTPException(
                        status_code=429,
                        detail=rate_stats['message'],
                        headers={
                            'X-RateLimit-Limit': str(rate_stats['requests_limit']),
                            'X-RateLimit-Used': str(rate_stats['requests_used'])
                        }
                    )
                
                # Validate file
                if file.filename is None:
                    raise HTTPException(status_code=400, detail="No filename")
                
                contents = await file.read()
                if len(contents) > self.config["api"]["max_image_size_mb"] * 1024 * 1024:
                    raise HTTPException(status_code=413, detail="File too large")
                
                extension = Path(file.filename).suffix.lower().lstrip('.')
                if extension not in self.config["api"]["allowed_formats"]:
                    raise HTTPException(status_code=400, detail="Invalid format")
                
                # Save temporarily
                temp_path = str(Path(tempfile.gettempdir()) / file.filename)
                with open(temp_path, 'wb') as f:
                    f.write(contents)
                
                # ✨ NEW: Try cache first
                cached_result = self.cache.get(temp_path)
                if cached_result:
                    self.logger.info(f"Cache hit for {file.filename}")
                    
                    # Record API usage
                    self.key_manager.record_usage(x_api_key)
                    
                    return {
                        "success": True,
                        "class_name": cached_result['class'],
                        "confidence": cached_result['confidence'],
                        "is_confident": cached_result['is_confident'],
                        "all_predictions": cached_result['all_predictions'],
                        "prediction_id": cached_result.get('prediction_id', -1),  # -1 = cached
                        "rate_limit": rate_stats,
                        "metadata": {"from_cache": True}
                    }
                
                # Make prediction
                result = self.predictor.predict_from_file(temp_path)
                
                # ✨ NEW: Cache the result
                self.cache.set(temp_path, result)
                
                processing_time_ms = (time.time() - start_time) * 1000
                self.optimizer.record('prediction', processing_time_ms)
                
                # ✨ NEW: Store in database
                prediction_id = self.db.save_prediction({
                    'predicted_class': result['prediction']['class'],
                    'confidence': result['prediction']['confidence'],
                    'all_predictions': result['all_predictions'],
                    'image_file': file.filename,
                    'user_id': user_id,
                    'model_version': result['metadata']['model_version'],
                    'processing_time_ms': processing_time_ms,
                    'ip_address': request.client.host if request.client else None
                })
                
                # ✨ NEW: Log the prediction
                self.analytics.log_prediction({
                    'class': result['prediction']['class'],
                    'confidence': result['prediction']['confidence'],
                    'processing_time_ms': processing_time_ms,
                    'model_version': result['metadata']['model_version'],
                    'user_id': user_id,
                    'image_file': file.filename,
                    'from_cache': False
                })
                
                # Record API usage
                self.key_manager.record_usage(x_api_key)
                
                # Cleanup
                Path(temp_path).unlink()
                
                return {
                    "success": True,
                    "class_name": result['prediction']['class'],
                    "confidence": result['prediction']['confidence'],
                    "is_confident": result['prediction']['is_confident'],
                    "all_predictions": result['all_predictions'],
                    "prediction_id": prediction_id,
                    "rate_limit": rate_stats,
                    "metadata": {
                        "processing_time_ms": processing_time_ms,
                        "from_cache": False
                    }
                }
            
            except HTTPException as e:
                raise e
            
            except Exception as e:
                # ✨ NEW: Log error with full context
                self.analytics.log_error(
                    f"Prediction failed: {str(e)}",
                    traceback.format_exc(),
                    context={'file': file.filename if file else None}
                )
                self.logger.error(f"Prediction error: {e}")
                
                raise HTTPException(
                    status_code=500,
                    detail=f"Prediction failed: {str(e)}"
                )
        
        # ====== ✨ NEW ENDPOINTS ======
        
        @self.app.get("/stats")
        async def get_stats():
            """Get prediction statistics (database)."""
            return self.db.get_statistics(hours=24)
        
        @self.app.post("/feedback/{prediction_id}")
        async def submit_feedback(
            prediction_id: int,
            feedback: str,
            correct_class: str = None,
            auth: dict = Depends(self.verify_api_key)
        ):
            """User reports prediction was wrong."""
            
            self.db.save_feedback(prediction_id, feedback, correct_class)
            self.analytics.log_event("feedback", {
                "prediction_id": prediction_id,
                "feedback": feedback,
                "correct_class": correct_class
            })
            
            return {
                "success": True,
                "message": "Thank you for feedback! We'll improve."
            }
        
        @self.app.get("/cache-stats")
        async def cache_stats():
            """Get cache performance."""
            return self.cache.get_stats()
        
        @self.app.get("/analytics")
        async def get_analytics():
            """Get system analytics."""
            aggregator = AnalyticsAggregator()
            return {
                "statistics": aggregator.get_statistics(),
                "class_distribution": aggregator.get_class_distribution(),
                "top_errors": aggregator.get_error_distribution(),
                "slowest_predictions": aggregator.get_slowest_predictions(limit=5)
            }
        
        @self.app.post("/admin/generate-key")
        async def generate_api_key(user_id: str, name: str = None):
            """⚠️  This should require admin auth in production!"""
            api_key = self.key_manager.generate_key(user_id, name)
            return {
                'success': True,
                'api_key': api_key,
                'user_id': user_id,
                'message': 'Save this key securely!'
            }


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Start the enhanced API server."""
    ml_app = EnhancedMLApp("./config.yaml")
    app = ml_app.app
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=1,
        log_level="info"
    )


if __name__ == "__main__":
    main()


# ============================================================================
# TESTING THE ENHANCED API
# ============================================================================

"""
How to use the enhanced API:

1. Generate API key:
   curl -X POST "http://localhost:8000/admin/generate-key?user_id=test_user"
   # Response: {"api_key": "sk-abc123..."}

2. Make prediction:
   curl -X POST "http://localhost:8000/predict" \
     -H "X-API-Key: sk-abc123..." \
     -F "file=@leaf.jpg"
   
   # Response includes prediction_id for feedback

3. Submit feedback:
   curl -X POST "http://localhost:8000/feedback/123" \
     -H "X-API-Key: sk-abc123..." \
     -d "feedback=wrong&correct_class=powdery_mildew"

4. Get statistics:
   curl http://localhost:8000/stats

5. Get cache performance:
   curl http://localhost:8000/cache-stats

6. Get analytics:
   curl http://localhost:8000/analytics

7. Health check:
   curl http://localhost:8000/health
"""