"""
Advanced Logging & Analytics

Provides structured logging for:
1. Tracking all predictions
2. Detecting anomalies
3. Performance monitoring
4. Debugging issues
"""

import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
from enum import Enum

class EventType(Enum):
    """Different types of events to track."""
    PREDICTION = "prediction"
    ERROR = "error"
    WARNING = "warning"
    API_REQUEST = "api_request"
    MODEL_TRAINING = "model_training"
    SYSTEM_HEALTH = "system_health"
    FEEDBACK = "feedback"

class StructuredLogger:
    """
    Logs events in JSON format for easy analysis.
    
    Example JSON output:
    {
        "timestamp": "2024-01-15T10:30:45.123Z",
        "event_type": "prediction",
        "class": "powdery_mildew",
        "confidence": 0.92,
        "processing_time_ms": 125,
        "user_id": "user_123",
        "ip_address": "192.168.1.1"
    }
    
    Advantages:
    - Can parse with tools (ELK, Datadog, etc.)
    - Easy to query and analyze
    - Machine-readable
    - Works with monitoring systems
    """
    
    def __init__(self, log_file: str = "./logs/events.jsonl"):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Python logger for console output
        self.logger = logging.getLogger("analytics")
        self.logger.setLevel(logging.INFO)
    
    def log_event(self, event_type: EventType, data: Dict[str, Any]) -> str:
        """
        Log a structured event.
        
        Args:
            event_type: Type of event
            data: Event details as dict
        
        Returns:
            Event ID for tracking
        """
        event_type_str = event_type.value if isinstance(event_type, EventType) else str(event_type)
        event_id = f"{int(time.time() * 1000)}_{event_type_str}"
        
        event = {
            "event_id": event_id,
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type_str,
            **data
        }
        
        # Write to file (JSON Lines format)
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(event) + "\n")
        
        return event_id
    
    def log_prediction(self, prediction_data: Dict):
        """Log a prediction."""
        self.log_event(EventType.PREDICTION, {
            "class": prediction_data['class'],
            "confidence": prediction_data['confidence'],
            "processing_time_ms": prediction_data.get('processing_time_ms'),
            "model_version": prediction_data.get('model_version'),
            "user_id": prediction_data.get('user_id'),
            "image_file": prediction_data.get('image_file')
        })
    
    def log_error(self, error_message: str, traceback: str = None, context: Dict = None):
        """Log an error."""
        self.log_event(EventType.ERROR, {
            "message": error_message,
            "traceback": traceback,
            "context": context or {}
        })
    
    def log_api_request(self, method: str, endpoint: str, 
                        status_code: int, response_time_ms: float, 
                        user_id: str = None, ip_address: str = None):
        """Log API request."""
        self.log_event(EventType.API_REQUEST, {
            "method": method,
            "endpoint": endpoint,
            "status_code": status_code,
            "response_time_ms": response_time_ms,
            "user_id": user_id,
            "ip_address": ip_address
        })
    
    def log_model_training(self, metrics: Dict):
        """Log model training completion."""
        self.log_event(EventType.MODEL_TRAINING, {
            "accuracy": metrics.get('accuracy'),
            "loss": metrics.get('loss'),
            "epochs": metrics.get('epochs'),
            "training_time_hours": metrics.get('training_time_hours'),
            "model_version": metrics.get('model_version')
        })
    
    def log_system_health(self, health_data: Dict):
        """Log system health metrics."""
        self.log_event(EventType.SYSTEM_HEALTH, {
            "memory_usage_mb": health_data.get('memory_usage_mb'),
            "gpu_usage_percent": health_data.get('gpu_usage_percent'),
            "avg_prediction_time_ms": health_data.get('avg_prediction_time_ms'),
            "total_predictions": health_data.get('total_predictions'),
            "error_count": health_data.get('error_count')
        })


class AnalyticsAggregator:
    """
    Analyze logs to extract insights.
    
    Example queries:
    - Average response time per hour
    - Top 10 slowest predictions
    - Error rate over time
    - User with most predictions
    """
    
    def __init__(self, log_file: str = "./logs/events.jsonl"):
        self.log_file = log_file
    
    def load_events(self, event_type: str = None) -> list:
        """Load and parse log file."""
        events = []
        
        if not Path(self.log_file).exists():
            return events
        
        with open(self.log_file, 'r') as f:
            for line in f:
                try:
                    event = json.loads(line)
                    if event_type is None or event.get('event_type') == event_type:
                        events.append(event)
                except json.JSONDecodeError:
                    continue
        
        return events
    
    def get_statistics(self) -> Dict:
        """Get overall statistics."""
        predictions = self.load_events('prediction')
        errors = self.load_events('error')
        
        if not predictions:
            return {}
        
        processing_times = [p.get('processing_time_ms', 0) for p in predictions]
        
        return {
            'total_predictions': len(predictions),
            'total_errors': len(errors),
            'error_rate': len(errors) / len(predictions) if predictions else 0,
            'avg_processing_time_ms': sum(processing_times) / len(processing_times),
            'min_processing_time_ms': min(processing_times),
            'max_processing_time_ms': max(processing_times),
            'p95_processing_time_ms': sorted(processing_times)[int(0.95 * len(processing_times))]
        }
    
    def get_class_distribution(self) -> Dict:
        """What classes are being predicted most?"""
        predictions = self.load_events('prediction')
        
        distribution = {}
        for pred in predictions:
            class_name = pred.get('class')
            distribution[class_name] = distribution.get(class_name, 0) + 1
        
        return distribution
    
    def get_slowest_predictions(self, limit: int = 10) -> list:
        """Find predictions that took longest."""
        predictions = self.load_events('prediction')
        
        sorted_preds = sorted(
            predictions,
            key=lambda x: x.get('processing_time_ms', 0),
            reverse=True
        )
        
        return sorted_preds[:limit]
    
    def get_error_distribution(self) -> Dict:
        """What types of errors are happening?"""
        errors = self.load_events('error')
        
        distribution = {}
        for error in errors:
            message = error.get('message', 'Unknown')
            distribution[message] = distribution.get(message, 0) + 1
        
        return distribution
    
    def get_predictions_by_user(self) -> Dict:
        """Which users are making most predictions?"""
        predictions = self.load_events('prediction')
        
        by_user = {}
        for pred in predictions:
            user_id = pred.get('user_id', 'anonymous')
            by_user[user_id] = by_user.get(user_id, 0) + 1
        
        # Sort by count
        return dict(sorted(by_user.items(), key=lambda x: x[1], reverse=True))


# ============================================================================
# INTEGRATION WITH FastAPI
# ============================================================================

def setup_analytics_middleware(app, analytics_logger: StructuredLogger):
    """
    Add middleware to automatically log all requests/responses.
    
    Usage in api.py:
    
    from this_module import StructuredLogger, setup_analytics_middleware
    
    analytics = StructuredLogger("./logs/events.jsonl")
    setup_analytics_middleware(app, analytics)
    """
    
    @app.middleware("http")
    async def log_requests(request, call_next):
        start_time = time.time()
        
        response = await call_next(request)
        
        process_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Log the request
        analytics_logger.log_api_request(
            method=request.method,
            endpoint=request.url.path,
            status_code=response.status_code,
            response_time_ms=process_time,
            ip_address=request.client.host if request.client else None
        )
        
        return response


# ============================================================================
# EXAMPLE: MONITORING DASHBOARD (can be built later)
# ============================================================================

"""
Example queries for monitoring dashboard:

1. Predictions per hour (trend)
2. Average response time (trending up = problem!)
3. Error rate (should be <1%)
4. Class distribution (useful for capacity planning)
5. Slowest endpoints (where to optimize)
6. Top errors (what to fix first)

Can integrate with:
- Grafana (visualization)
- ELK Stack (log analysis)
- Datadog (monitoring)
- Prometheus (metrics)
"""


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Create logger
    logger = StructuredLogger("./logs/events_demo.jsonl")
    
    # Log some events
    print("Logging events...")
    
    for i in range(5):
        logger.log_prediction({
            'class': 'healthy' if i % 2 == 0 else 'disease',
            'confidence': 0.85 + (i * 0.01),
            'processing_time_ms': 100 + i * 10,
            'model_version': 'v1.0',
            'user_id': f'user_{i}'
        })
    
    logger.log_error("Connection timeout", "socket.timeout: timeout()")
    
    # Analyze
    print("\n=== Analytics ===")
    aggregator = AnalyticsAggregator("./logs/events_demo.jsonl")
    
    stats = aggregator.get_statistics()
    print(f"Total predictions: {stats.get('total_predictions')}")
    print(f"Error rate: {stats.get('error_rate'):.2%}")
    print(f"Avg time: {stats.get('avg_processing_time_ms'):.1f}ms")
    
    print("\nClass distribution:", aggregator.get_class_distribution())
    print("Predictions by user:", aggregator.get_predictions_by_user())
    print("Top errors:", aggregator.get_error_distribution())