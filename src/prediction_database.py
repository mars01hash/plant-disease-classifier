"""
Database Integration for Storing Predictions

This module stores all predictions for:
1. Audit trail (compliance)
2. Performance monitoring (detect model drift)
3. Feedback collection (users can say "this is wrong!")
4. Retraining data (improve model with real predictions)
"""

import sqlite3
import json
from datetime import datetime
from typing import Dict, Optional
import logging
from pathlib import Path

class PredictionDatabase:
    """
    Store predictions in SQLite database.
    
    Production insight:
    - SQLite: Simple, file-based, no server needed (for MVP)
    - PostgreSQL: Better for scaling to 1000s of requests/day
    - MongoDB: Good for flexible schema
    
    Start with SQLite, migrate to PostgreSQL when needed.
    """
    
    def __init__(self, db_path: str = "./logs/predictions.db", logger: logging.Logger = None):
        self.db_path = db_path
        self.logger = logger or logging.getLogger(__name__)
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Create database tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Predictions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    image_file TEXT,
                    image_hash TEXT,
                    predicted_class TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    all_predictions TEXT,
                    user_id TEXT,
                    model_version TEXT,
                    processing_time_ms REAL,
                    ip_address TEXT,
                    user_feedback TEXT DEFAULT NULL,
                    feedback_timestamp TEXT DEFAULT NULL
                )
            """)
            
            # Create index on timestamp for fast queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON predictions(timestamp)
            """)
            
            # Create index on user for analytics
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_id 
                ON predictions(user_id)
            """)
            
            conn.commit()
        
        self.logger.info(f"Database initialized: {self.db_path}")
    
    def save_prediction(self, prediction_data: Dict) -> int:
        """
        Save a prediction to database.
        
        Args:
            prediction_data: Dict with keys:
                - predicted_class: str
                - confidence: float
                - all_predictions: dict
                - image_file: str (optional)
                - user_id: str (optional)
                - model_version: str
                - processing_time_ms: float
                - ip_address: str (optional)
        
        Returns:
            Prediction ID (for feedback later)
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO predictions (
                    timestamp,
                    image_file,
                    predicted_class,
                    confidence,
                    all_predictions,
                    user_id,
                    model_version,
                    processing_time_ms,
                    ip_address
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.utcnow().isoformat(),
                prediction_data.get('image_file'),
                prediction_data['predicted_class'],
                prediction_data['confidence'],
                json.dumps(prediction_data.get('all_predictions', {})),
                prediction_data.get('user_id'),
                prediction_data.get('model_version', 'unknown'),
                prediction_data.get('processing_time_ms', 0),
                prediction_data.get('ip_address')
            ))
            
            conn.commit()
            prediction_id = cursor.lastrowid
        
        return prediction_id
    
    def save_feedback(self, prediction_id: int, feedback: str, correct_class: Optional[str] = None):
        """
        User says "this prediction was wrong"
        
        Args:
            prediction_id: ID from save_prediction()
            feedback: "This is wrong"
            correct_class: "powdery_mildew" (if user knows correct answer)
        
        Production insight:
        - This creates a feedback loop
        - After collecting ~100 corrections, retrain model
        - Model improves with real-world data
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE predictions
                SET user_feedback = ?, feedback_timestamp = ?
                WHERE id = ?
            """, (
                feedback,
                datetime.utcnow().isoformat(),
                prediction_id
            ))
            
            conn.commit()
        
        self.logger.warning(f"Feedback recorded for prediction {prediction_id}: {feedback}")
    
    def get_statistics(self, hours: int = 24) -> Dict:
        """
        Get model performance statistics.
        
        Returns:
            {
                'total_predictions': 1234,
                'average_confidence': 0.92,
                'class_distribution': {...},
                'error_count': 45,
                'avg_processing_time_ms': 125
            }
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get time window
            since = datetime.fromtimestamp(
                (datetime.utcnow().timestamp() - hours * 3600)
            ).isoformat()
            
            # Total predictions
            cursor.execute("""
                SELECT COUNT(*) as total FROM predictions
                WHERE timestamp > ?
            """, (since,))
            total = cursor.fetchone()['total']
            
            # Average confidence
            cursor.execute("""
                SELECT AVG(confidence) as avg_confidence FROM predictions
                WHERE timestamp > ?
            """, (since,))
            avg_confidence = cursor.fetchone()['avg_confidence'] or 0
            
            # Class distribution
            cursor.execute("""
                SELECT predicted_class, COUNT(*) as count
                FROM predictions
                WHERE timestamp > ?
                GROUP BY predicted_class
            """, (since,))
            class_dist = {row['predicted_class']: row['count'] 
                         for row in cursor.fetchall()}
            
            # Feedback count (errors)
            cursor.execute("""
                SELECT COUNT(*) as errors FROM predictions
                WHERE timestamp > ? AND user_feedback IS NOT NULL
            """, (since,))
            errors = cursor.fetchone()['errors']
            
            # Average processing time
            cursor.execute("""
                SELECT AVG(processing_time_ms) as avg_time FROM predictions
                WHERE timestamp > ?
            """, (since,))
            avg_time = cursor.fetchone()['avg_time'] or 0
            
            return {
                'total_predictions': total,
                'average_confidence': float(avg_confidence),
                'class_distribution': class_dist,
                'feedback_count': errors,
                'avg_processing_time_ms': float(avg_time),
                'time_window_hours': hours,
                'accuracy_estimate': 1 - (errors / max(total, 1))  # Rough estimate
            }
    
    def get_predictions_needing_feedback(self, limit: int = 10) -> list:
        """
        Get predictions with low confidence (likely to be wrong).
        
        Use for:
        - Manual review
        - Prioritize which predictions to ask users about
        - Find edge cases for retraining
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM predictions
                WHERE confidence < 0.7
                ORDER BY confidence ASC
                LIMIT ?
            """, (limit,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def export_for_retraining(self, feedback_only: bool = False) -> list:
        """
        Export predictions for model retraining.
        
        Use when:
        - Collected 100+ predictions with feedback
        - Ready to retrain model with real-world data
        
        Args:
            feedback_only: Only export predictions users marked as wrong
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if feedback_only:
                cursor.execute("""
                    SELECT * FROM predictions
                    WHERE user_feedback IS NOT NULL
                    ORDER BY feedback_timestamp DESC
                """)
            else:
                cursor.execute("SELECT * FROM predictions")
            
            return [dict(row) for row in cursor.fetchall()]


# ============================================================================
# INTEGRATION WITH FastAPI (add to api.py)
# ============================================================================

def integrate_database_with_api(app, db: PredictionDatabase, logger):
    """
    Add database integration to FastAPI app.
    
    Usage in api.py:
    
    from this_module import PredictionDatabase, integrate_database_with_api
    
    db = PredictionDatabase("./logs/predictions.db", logger)
    integrate_database_with_api(app, db, logger)
    """
    
    # Store db in app state
    app.db = db
    
    # Enhanced /predict endpoint (store in database)
    original_predict = None
    for route in app.routes:
        if hasattr(route, 'path') and route.path == '/predict':
            original_predict = route
    
    # Add database stats endpoint
    @app.get("/stats")
    async def get_stats():
        """Get model performance statistics."""
        stats = db.get_statistics(hours=24)
        return stats
    
    # Add feedback endpoint
    @app.post("/feedback/{prediction_id}")
    async def submit_feedback(prediction_id: int, feedback: str, correct_class: str = None):
        """
        User reports prediction was wrong.
        
        Example:
            POST /feedback/123?feedback=wrong&correct_class=powdery_mildew
        """
        db.save_feedback(prediction_id, feedback, correct_class)
        return {
            "success": True,
            "message": "Feedback recorded. Thank you!",
            "prediction_id": prediction_id
        }
    
    logger.info("Database integration added to API")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Create database
    db = PredictionDatabase(logger=logger)
    
    # Simulate some predictions
    for i in range(10):
        db.save_prediction({
            'predicted_class': 'healthy' if i % 2 == 0 else 'powdery_mildew',
            'confidence': 0.85 + (i * 0.01),
            'all_predictions': {
                'healthy': 0.85,
                'powdery_mildew': 0.15
            },
            'user_id': f'user_{i}',
            'model_version': 'v1.0',
            'processing_time_ms': 125 + i,
            'ip_address': '127.0.0.1'
        })
    
    # Get statistics
    stats = db.get_statistics(hours=24)
    print("\n=== Model Statistics (Last 24 hours) ===")
    print(f"Total predictions: {stats['total_predictions']}")
    print(f"Average confidence: {stats['average_confidence']:.2%}")
    print(f"Feedback count: {stats['feedback_count']}")
    print(f"Class distribution: {stats['class_distribution']}")
    print(f"Avg processing time: {stats['avg_processing_time_ms']:.1f}ms")
    print(f"Estimated accuracy: {stats['accuracy_estimate']:.2%}")