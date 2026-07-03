"""
API Authentication & Rate Limiting

Protect API with API keys and rate limiting.

Features:
1. API key authentication
2. Rate limiting (requests per hour per user)
3. Usage tracking
4. Revocable keys
"""

import secrets
import json
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from pathlib import Path
from enum import Enum
from functools import wraps

from fastapi import HTTPException, Header, Request, Depends
from fastapi.security import APIKeyHeader
import logging


class APIKeyManager:
    """
    Manage API keys for authentication.
    
    Usage:
    1. Admin generates API key for user
    2. User includes key in requests
    3. System validates key
    4. Request is allowed/denied based on key
    """
    
    def __init__(self, keys_file: str = "./logs/api_keys.json"):
        self.keys_file = keys_file
        Path(keys_file).parent.mkdir(parents=True, exist_ok=True)
        self.keys = self._load_keys()
        self.logger = logging.getLogger(__name__)
    
    def generate_key(self, user_id: str, name: str = None) -> str:
        """
        Generate new API key for user.
        
        Args:
            user_id: User identifier
            name: Friendly name (e.g., "Production API")
        
        Returns:
            API key (e.g., "sk-abc123def456...")
        """
        # Generate random token
        token = secrets.token_urlsafe(32)
        api_key = f"sk-{token}"
        
        # Store with metadata
        if user_id not in self.keys:
            self.keys[user_id] = []
        
        self.keys[user_id].append({
            'key': api_key,
            'name': name or f"Key created {datetime.utcnow().isoformat()}",
            'created': datetime.utcnow().isoformat(),
            'last_used': None,
            'active': True,
            'usage_count': 0
        })
        
        self._save_keys()
        self.logger.info(f"Generated API key for user {user_id}")
        
        return api_key
    
    def validate_key(self, api_key: str) -> Optional[Dict]:
        """
        Validate API key and get user info.
        
        Returns:
            Dict with user_id and key info if valid, None if invalid
        """
        for user_id, keys in self.keys.items():
            for key_info in keys:
                if key_info['key'] == api_key and key_info['active']:
                    return {
                        'user_id': user_id,
                        'key_info': key_info
                    }
        
        return None
    
    def revoke_key(self, api_key: str) -> bool:
        """Disable an API key."""
        for user_id, keys in self.keys.items():
            for key_info in keys:
                if key_info['key'] == api_key:
                    key_info['active'] = False
                    self._save_keys()
                    self.logger.warning(f"Revoked API key {api_key}")
                    return True
        
        return False
    
    def record_usage(self, api_key: str) -> bool:
        """Record API key usage."""
        for user_id, keys in self.keys.items():
            for key_info in keys:
                if key_info['key'] == api_key and key_info['active']:
                    key_info['usage_count'] += 1
                    key_info['last_used'] = datetime.utcnow().isoformat()
                    self._save_keys()
                    return True
        
        return False
    
    def _load_keys(self) -> Dict:
        """Load keys from file."""
        if Path(self.keys_file).exists():
            with open(self.keys_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_keys(self) -> None:
        """Save keys to file."""
        with open(self.keys_file, 'w') as f:
            json.dump(self.keys, f, indent=2)


class RateLimiter:
    """
    Implement rate limiting (requests per hour per user).
    
    Example:
    - User can make 1000 requests per hour
    - After 1000, requests are rejected
    - Limit resets every hour
    """
    
    def __init__(self, requests_per_hour: int = 1000, tracking_file: str = "./logs/rate_limits.json"):
        self.requests_per_hour = requests_per_hour
        self.tracking_file = tracking_file
        self.limits = self._load_limits()
        self.logger = logging.getLogger(__name__)
    
    def is_allowed(self, user_id: str) -> Tuple[bool, Dict]:
        """
        Check if user can make request.
        
        Returns:
            (is_allowed, stats)
        """
        now = datetime.utcnow()
        
        if user_id not in self.limits:
            # New user
            self.limits[user_id] = {
                'requests': [now.isoformat()],
                'hour_start': now.isoformat()
            }
            self._save_limits()
            
            return True, {
                'requests_used': 1,
                'requests_limit': self.requests_per_hour,
                'reset_time': (now + timedelta(hours=1)).isoformat()
            }
        
        user_limit = self.limits[user_id]
        hour_start = datetime.fromisoformat(user_limit['hour_start'])
        
        # Check if hour expired
        if now - hour_start > timedelta(hours=1):
            # Reset
            user_limit['requests'] = [now.isoformat()]
            user_limit['hour_start'] = now.isoformat()
            self._save_limits()
            
            return True, {
                'requests_used': 1,
                'requests_limit': self.requests_per_hour,
                'reset_time': (now + timedelta(hours=1)).isoformat()
            }
        
        # Check limit
        request_count = len(user_limit['requests'])
        
        if request_count >= self.requests_per_hour:
            # Over limit!
            return False, {
                'requests_used': request_count,
                'requests_limit': self.requests_per_hour,
                'reset_time': (hour_start + timedelta(hours=1)).isoformat(),
                'message': f'Rate limit exceeded. Reset in {int((hour_start + timedelta(hours=1) - now).total_seconds())}s'
            }
        
        # Record this request
        user_limit['requests'].append(now.isoformat())
        self._save_limits()
        
        return True, {
            'requests_used': request_count + 1,
            'requests_limit': self.requests_per_hour,
            'reset_time': (hour_start + timedelta(hours=1)).isoformat()
        }
    
    def _load_limits(self) -> Dict:
        """Load rate limit tracking from file."""
        if Path(self.tracking_file).exists():
            with open(self.tracking_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_limits(self) -> None:
        """Save rate limit tracking."""
        with open(self.tracking_file, 'w') as f:
            json.dump(self.limits, f, indent=2)


# ============================================================================
# FASTAPI INTEGRATION
# ============================================================================

def setup_auth(app, api_key_manager: APIKeyManager, rate_limiter: RateLimiter):
    """
    Add authentication to FastAPI app.
    
    Usage:
    
    from this_module import APIKeyManager, RateLimiter, setup_auth
    
    key_manager = APIKeyManager()
    limiter = RateLimiter(requests_per_hour=1000)
    setup_auth(app, key_manager, limiter)
    
    # Now all endpoints require API key:
    # curl -H "X-API-Key: sk-abc123..." http://localhost:8000/predict
    """
    
    logger = logging.getLogger(__name__)
    
    # Add admin endpoints to generate/manage keys
    @app.post("/admin/generate-key")
    async def generate_api_key(user_id: str, name: str = None):
        """
        Generate new API key.
        
        ** WARNING: In production, this should require admin authentication! **
        """
        api_key = api_key_manager.generate_key(user_id, name)
        
        return {
            'success': True,
            'api_key': api_key,
            'user_id': user_id,
            'message': 'Save this key securely. You won\'t be able to see it again!'
        }
    
    @app.post("/admin/revoke-key")
    async def revoke_api_key(api_key: str):
        """Disable an API key."""
        success = api_key_manager.revoke_key(api_key)
        
        return {
            'success': success,
            'message': 'Key revoked successfully' if success else 'Key not found'
        }
    
    api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
    
    # Add authenticated endpoint decorator
    async def verify_api_key(x_api_key: str = Depends(api_key_header)):
        """Dependency for API key verification."""
        if not x_api_key:
            raise HTTPException(
                status_code=401,
                detail="API key required. Include 'X-API-Key' header."
            )
        
        key_info = api_key_manager.validate_key(x_api_key)
        if not key_info:
            raise HTTPException(
                status_code=403,
                detail="Invalid or inactive API key"
            )
        
        return key_info
    
    # Wrap protected endpoints
    async def verify_rate_limit(user_id: str, request: Request):
        """Check rate limit."""
        allowed, stats = rate_limiter.is_allowed(user_id)
        
        # Add stats to response header
        request.state.rate_limit_stats = stats
        
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail=stats['message'],
                headers={
                    'X-RateLimit-Limit': str(stats['requests_limit']),
                    'X-RateLimit-Used': str(stats['requests_used']),
                    'X-RateLimit-Reset': stats['reset_time']
                }
            )
        
        return stats
    
    # Return dependencies for use in endpoints
    return verify_api_key, verify_rate_limit


# ============================================================================
# USAGE IN api.py
# ============================================================================

"""
Example of how to use in api.py:

from this_module import APIKeyManager, RateLimiter, setup_auth
from fastapi import Depends

key_manager = APIKeyManager()
limiter = RateLimiter(requests_per_hour=1000)
verify_api_key, verify_rate_limit = setup_auth(app, key_manager, limiter)


@app.post("/predict")
async def predict(
    file: UploadFile,
    auth: dict = Depends(verify_api_key),
    request: Request = None
):
    user_id = auth['user_id']
    
    # Check rate limit
    stats = await verify_rate_limit(user_id, request)
    
    # Add user_id to tracking
    prediction_data = {
        'user_id': user_id,
        'class': result['class'],
        'confidence': result['confidence'],
        ...
    }
    
    # Record API key usage
    key_manager.record_usage(auth['key_info']['key'])
    
    return {
        'prediction': result,
        'rate_limit': stats
    }
"""

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=== API Authentication Demo ===\n")
    
    # Setup
    key_manager = APIKeyManager()
    limiter = RateLimiter(requests_per_hour=5)  # Low limit for demo
    
    # Generate keys for users
    key1 = key_manager.generate_key("user_alice", "Alice's Production Key")
    key2 = key_manager.generate_key("user_bob", "Bob's Test Key")
    
    print(f"Generated keys:")
    print(f"  Alice: {key1}")
    print(f"  Bob: {key2}\n")
    
    # Validate keys
    print("Validating keys:")
    info1 = key_manager.validate_key(key1)
    print(f"  Key1 valid: {info1 is not None} (user: {info1['user_id'] if info1 else 'N/A'})")
    
    info_invalid = key_manager.validate_key("sk-invalid")
    print(f"  Invalid key valid: {info_invalid is not None}\n")
    
    # Test rate limiting
    print("Testing rate limit (5 requests per hour):")
    for i in range(7):
        allowed, stats = limiter.is_allowed("user_alice")
        status = "✓ Allowed" if allowed else "✗ Blocked"
        print(f"  Request {i+1}: {status} ({stats['requests_used']}/{stats['requests_limit']})")
    
    print("\nDone!")