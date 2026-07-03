"""
Prediction Caching System

Cache predictions to avoid redundant computation.

Example:
- User uploads leaf.jpg → 100ms prediction
- Same user uploads leaf.jpg again → <1ms (from cache!)
- Different user uploads same leaf.jpg → <1ms (from cache!)

Benefits:
1. Response time: 100x faster for cached images
2. GPU usage: Save compute for new predictions
3. Cost: Cheaper (less GPU hours)
4. Scalability: Handle more users with same hardware
"""

import hashlib
import time
from typing import Dict, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime, timedelta


class PredictionCache:
    """
    In-memory cache with optional persistence.
    
    Strategies:
    1. LRU (Least Recently Used) - Remove oldest when full
    2. LFU (Least Frequently Used) - Remove least-used when full
    3. TTL (Time To Live) - Expire after N hours
    
    This implementation uses LRU + TTL.
    """
    
    def __init__(self, max_size: int = 1000, ttl_hours: int = 24, persist_path: Optional[str] = None):
        """
        Args:
            max_size: Max predictions to cache (memory limit)
            ttl_hours: How long to keep cached predictions
            persist_path: Optional file to persist cache to disk
        """
        self.cache = {}  # {image_hash: prediction}
        self.access_times = {}  # {image_hash: last_access_time}
        self.create_times = {}  # {image_hash: creation_time}
        self.hit_count = 0
        self.miss_count = 0
        self.max_size = max_size
        self.ttl = timedelta(hours=ttl_hours)
        self.persist_path = persist_path
        
        # Load from disk if available
        if persist_path and Path(persist_path).exists():
            self._load_from_disk()
    
    def _hash_image(self, image_path: str) -> str:
        """Generate hash of image file."""
        with open(image_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    
    def get(self, image_path: str) -> Optional[Dict]:
        """
        Get cached prediction.
        
        Returns:
            Prediction dict if cached and valid, None otherwise
        """
        image_hash = self._hash_image(image_path)
        
        # Check if cached
        if image_hash not in self.cache:
            self.miss_count += 1
            return None
        
        # Check if expired
        created = self.create_times[image_hash]
        if datetime.utcnow() - created > self.ttl:
            # Expired, remove
            del self.cache[image_hash]
            del self.access_times[image_hash]
            del self.create_times[image_hash]
            self.miss_count += 1
            return None
        
        # Update access time (for LRU eviction)
        self.access_times[image_hash] = datetime.utcnow()
        self.hit_count += 1
        
        return self.cache[image_hash]
    
    def set(self, image_path: str, prediction: Dict) -> None:
        """
        Cache a prediction.
        
        Args:
            image_path: Path to image file
            prediction: Prediction dict to cache
        """
        image_hash = self._hash_image(image_path)
        
        # Check if need to evict
        if len(self.cache) >= self.max_size:
            self._evict_lru()
        
        # Store
        self.cache[image_hash] = prediction
        self.access_times[image_hash] = datetime.utcnow()
        self.create_times[image_hash] = datetime.utcnow()
        
        # Persist if enabled
        if self.persist_path:
            self._save_to_disk()
    
    def _evict_lru(self) -> None:
        """Remove least recently used item."""
        if not self.cache:
            return
        
        # Find least recently accessed
        lru_hash = min(
            self.access_times.keys(),
            key=lambda h: self.access_times[h]
        )
        
        # Remove it
        del self.cache[lru_hash]
        del self.access_times[lru_hash]
        del self.create_times[lru_hash]
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0
        
        return {
            'cache_size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hit_count,
            'misses': self.miss_count,
            'hit_rate': hit_rate,
            'total_requests': total_requests,
            'memory_usage_percent': (len(self.cache) / self.max_size) * 100
        }
    
    def clear(self) -> None:
        """Clear all cached predictions."""
        self.cache.clear()
        self.access_times.clear()
        self.create_times.clear()
        self.hit_count = 0
        self.miss_count = 0
        
        if self.persist_path and Path(self.persist_path).exists():
            Path(self.persist_path).unlink()
    
    def _save_to_disk(self) -> None:
        """Persist cache to disk for recovery."""
        cache_data = {
            'cache': self.cache,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        with open(self.persist_path, 'w') as f:
            json.dump(cache_data, f, default=str)
    
    def _load_from_disk(self) -> None:
        """Restore cache from disk."""
        try:
            with open(self.persist_path, 'r') as f:
                cache_data = json.load(f)
            
            self.cache = cache_data.get('cache', {})
            self.hit_count = cache_data.get('hit_count', 0)
            self.miss_count = cache_data.get('miss_count', 0)
            
            # Reinitialize timestamps (since we lost them)
            now = datetime.utcnow()
            for image_hash in self.cache.keys():
                self.access_times[image_hash] = now
                self.create_times[image_hash] = now
        
        except Exception as e:
            print(f"Failed to load cache from disk: {e}")


class ResponseTimeOptimizer:
    """
    Track response times to find bottlenecks.
    
    Example output:
    - Avg prediction: 125ms
    - Avg cache hit: 2ms
    - Avg preprocessing: 25ms
    - Bottleneck: GPU inference (98ms)
    """
    
    def __init__(self):
        self.timings = []  # List of timing records
    
    def record(self, stage: str, duration_ms: float) -> None:
        """Record timing for a stage."""
        self.timings.append({
            'stage': stage,
            'duration_ms': duration_ms,
            'timestamp': datetime.utcnow()
        })
    
    def get_average_by_stage(self, minutes: int = 60) -> Dict:
        """Get average duration per stage in last N minutes."""
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
        
        recent = [t for t in self.timings if t['timestamp'] > cutoff_time]
        
        stages = {}
        for timing in recent:
            stage = timing['stage']
            if stage not in stages:
                stages[stage] = []
            stages[stage].append(timing['duration_ms'])
        
        return {
            stage: {
                'avg_ms': sum(times) / len(times),
                'min_ms': min(times),
                'max_ms': max(times),
                'count': len(times)
            }
            for stage, times in stages.items()
        }
    
    def get_bottleneck(self) -> Tuple[str, float]:
        """Find slowest stage."""
        if not self.timings:
            return None, 0
        
        stats = self.get_average_by_stage()
        
        slowest = max(
            stats.items(),
            key=lambda x: x[1]['avg_ms']
        )
        
        return slowest[0], slowest[1]['avg_ms']


# ============================================================================
# INTEGRATION WITH FastAPI (add to api.py)
# ============================================================================

def add_caching_to_api(app, cache: PredictionCache, optimizer: ResponseTimeOptimizer):
    """
    Add caching to prediction endpoint.
    
    Usage in api.py:
    
    from this_module import PredictionCache, ResponseTimeOptimizer, add_caching_to_api
    
    cache = PredictionCache(max_size=1000, ttl_hours=24)
    optimizer = ResponseTimeOptimizer()
    add_caching_to_api(app, cache, optimizer)
    
    
    # Then in /predict endpoint:
    
    @app.post("/predict")
    async def predict(file: UploadFile):
        start = time.time()
        
        # Try cache first
        cached = cache.get(temp_path)
        if cached:
            return cached  # Instant response!
        
        # If not cached, compute
        result = predictor.predict_from_file(temp_path)
        
        # Store in cache
        cache.set(temp_path, result)
        
        return result
    """
    
    # Add cache stats endpoint
    @app.get("/cache-stats")
    async def cache_stats():
        """Get cache performance statistics."""
        return cache.get_stats()
    
    # Add performance stats endpoint
    @app.get("/performance-stats")
    async def performance_stats():
        """Get response time breakdown."""
        bottleneck_stage, bottleneck_time = optimizer.get_bottleneck()
        
        return {
            'timings_by_stage': optimizer.get_average_by_stage(),
            'bottleneck': {
                'stage': bottleneck_stage,
                'avg_ms': bottleneck_time
            }
        }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    from pathlib import Path
    import tempfile
    import numpy as np
    from PIL import Image
    
    # Create temp image
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        img = np.ones((224, 224, 3), dtype=np.uint8) * 128
        Image.fromarray(img).save(tmp.name)
        temp_image = tmp.name
    
    # Create cache
    cache = PredictionCache(max_size=100, ttl_hours=24)
    
    print("=== Caching Demo ===\n")
    
    # First request (cache miss)
    start = time.time()
    prediction = cache.get(temp_image)
    print(f"First request (cache miss): {prediction}")
    
    # Simulate prediction
    test_prediction = {
        'class': 'healthy',
        'confidence': 0.92,
        'all_predictions': {'healthy': 0.92, 'disease': 0.08}
    }
    cache.set(temp_image, test_prediction)
    print(f"Stored prediction in cache")
    
    # Second request (cache hit)
    start = time.time()
    cached = cache.get(temp_image)
    elapsed = (time.time() - start) * 1000
    print(f"\nSecond request (cache hit): {elapsed:.2f}ms")
    print(f"Result: {cached}\n")
    
    # Stats
    stats = cache.get_stats()
    print(f"Cache Statistics:")
    print(f"  Hit rate: {stats['hit_rate']:.1%}")
    print(f"  Cache size: {stats['cache_size']}/{stats['max_size']}")
    print(f"  Total hits: {stats['hits']}")
    print(f"  Total misses: {stats['misses']}")
    
    # Cleanup
    Path(temp_image).unlink()