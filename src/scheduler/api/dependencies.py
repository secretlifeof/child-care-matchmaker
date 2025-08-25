"""
FastAPI dependency injection for the scheduler API
"""
import time
import logging
from typing import Optional, Dict, Any
from functools import lru_cache
from datetime import datetime, timedelta

from fastapi import HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from ..core.optimizer import ScheduleOptimizer
from ..core.cache import CacheManager
from ..utils.profiler import PerformanceProfiler
from ..utils.exceptions import ConfigurationError
from ..config import settings

logger = logging.getLogger(__name__)

# Global instances - initialized once
_optimizer_instance: Optional[ScheduleOptimizer] = None
_cache_manager_instance: Optional[CacheManager] = None
_profiler_instance: Optional[PerformanceProfiler] = None

# Rate limiting storage (in production, use Redis)
_rate_limit_storage: Dict[str, Dict[str, Any]] = {}

# Security
security = HTTPBearer(auto_error=False)


@lru_cache()
def get_settings():
    """Get application settings (cached)"""
    return settings


def get_optimizer() -> ScheduleOptimizer:
    """Get the schedule optimizer instance (singleton)"""
    global _optimizer_instance
    
    if _optimizer_instance is None:
        try:
            _optimizer_instance = ScheduleOptimizer()
            logger.info("Schedule optimizer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize optimizer: {e}")
            raise ConfigurationError(f"Optimizer initialization failed: {e}")
    
    return _optimizer_instance


def get_cache_manager() -> CacheManager:
    """Get the cache manager instance (singleton)"""
    global _cache_manager_instance
    
    if _cache_manager_instance is None:
        try:
            _cache_manager_instance = CacheManager()
            logger.info("Cache manager initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize cache manager: {e}")
            # Return a basic in-memory cache as fallback
            _cache_manager_instance = CacheManager()
    
    return _cache_manager_instance


def get_profiler() -> PerformanceProfiler:
    """Get the performance profiler instance (singleton)"""
    global _profiler_instance
    
    if _profiler_instance is None:
        _profiler_instance = PerformanceProfiler()
        logger.info("Performance profiler initialized")
    
    return _profiler_instance


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[Dict[str, Any]]:
    """
    Get current user from API key (optional authentication)
    
    In production, this would validate the API key against a database
    For now, it's a simple check against environment variable
    """
    if not settings.ENABLE_API_KEY_AUTH:
        return None
    
    if not credentials:
        if settings.REQUIRE_AUTH:
            raise HTTPException(
                status_code=401,
                detail="API key required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return None
    
    # Simple API key validation (in production, use proper auth service)
    expected_key = settings.API_KEY
    if expected_key and credentials.credentials == expected_key:
        return {
            "user_id": "api_user",
            "permissions": ["schedule:read", "schedule:write"],
            "authenticated": True
        }
    
    raise HTTPException(
        status_code=401,
        detail="Invalid API key",
        headers={"WWW-Authenticate": "Bearer"},
    )


def require_auth(user: Optional[Dict[str, Any]] = Depends(get_current_user)) -> Dict[str, Any]:
    """Require authentication (force authentication check)"""
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Authentication required"
        )
    return user


def require_permission(permission: str):
    """Require specific permission"""
    def permission_check(user: Dict[str, Any] = Depends(require_auth)) -> Dict[str, Any]:
        if permission not in user.get("permissions", []):
            raise HTTPException(
                status_code=403,
                detail=f"Permission '{permission}' required"
            )
        return user
    return permission_check


async def rate_limit(
    request: Request,
    max_requests: int = 60,  # per minute
    window_seconds: int = 60
) -> bool:
    """
    Simple rate limiting based on client IP
    
    In production, use Redis with sliding window or token bucket algorithm
    """
    # Disable rate limiting in debug mode
    if settings.DEBUG:
        return True
    
    client_ip = request.client.host
    current_time = time.time()
    
    # Clean old entries
    cutoff_time = current_time - window_seconds
    for ip in list(_rate_limit_storage.keys()):
        _rate_limit_storage[ip]["requests"] = [
            req_time for req_time in _rate_limit_storage[ip]["requests"]
            if req_time > cutoff_time
        ]
        if not _rate_limit_storage[ip]["requests"]:
            del _rate_limit_storage[ip]
    
    # Check rate limit for this IP
    if client_ip not in _rate_limit_storage:
        _rate_limit_storage[client_ip] = {"requests": []}
    
    client_requests = _rate_limit_storage[client_ip]["requests"]
    
    if len(client_requests) >= max_requests:
        # Rate limit exceeded
        retry_after = int(min(client_requests) + window_seconds - current_time) + 1
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
            headers={"Retry-After": str(retry_after)}
        )
    
    # Add current request
    client_requests.append(current_time)
    
    return True


def get_request_context(request: Request) -> Dict[str, Any]:
    """Extract request context for logging and analytics"""
    return {
        "method": request.method,
        "url": str(request.url),
        "client_ip": request.client.host,
        "user_agent": request.headers.get("user-agent"),
        "request_id": request.headers.get("x-request-id"),
        "timestamp": datetime.now().isoformat()
    }


async def validate_content_type(request: Request) -> bool:
    """Validate content type for POST/PUT requests"""
    if request.method in ["POST", "PUT", "PATCH"]:
        content_type = request.headers.get("content-type", "").lower()
        if not content_type.startswith("application/json"):
            raise HTTPException(
                status_code=415,
                detail="Content-Type must be application/json"
            )
    return True


async def check_maintenance_mode() -> bool:
    """Check if the system is in maintenance mode"""
    maintenance_mode = settings.MAINTENANCE_MODE
    
    if maintenance_mode:
        maintenance_message = settings.MAINTENANCE_MESSAGE
        raise HTTPException(
            status_code=503,
            detail=maintenance_message,
            headers={"Retry-After": "3600"}  # 1 hour
        )
    
    return True


class DatabaseSession:
    """Database session dependency (for future use)"""
    
    def __init__(self):
        # In future, this would create a database session
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Close database session
        pass


async def get_db_session() -> DatabaseSession:
    """Get database session (placeholder for future database integration)"""
    return DatabaseSession()


def log_request_start(request: Request, profiler: PerformanceProfiler = Depends(get_profiler)):
    """Log request start and begin performance tracking"""
    request_context = get_request_context(request)
    request_id = request_context["request_id"] or f"req_{int(time.time() * 1000)}"
    
    logger.info(f"[{request_id}] {request.method} {request.url.path} - Request started")
    profiler.start_timer(f"request_{request_id}")
    
    # Store request context for later use
    request.state.request_id = request_id
    request.state.start_time = time.time()


def get_client_info(request: Request) -> Dict[str, str]:
    """Extract client information from request headers"""
    return {
        "ip": request.client.host,
        "user_agent": request.headers.get("user-agent", "unknown"),
        "origin": request.headers.get("origin", "unknown"),
        "referer": request.headers.get("referer", "unknown"),
        "forwarded_for": request.headers.get("x-forwarded-for", ""),
        "real_ip": request.headers.get("x-real-ip", "")
    }


async def validate_request_size(
    request: Request,
    max_size_mb: int = 10
) -> bool:
    """Validate request payload size"""
    if request.method in ["POST", "PUT", "PATCH"]:
        content_length = request.headers.get("content-length")
        if content_length:
            size_mb = int(content_length) / (1024 * 1024)
            if size_mb > max_size_mb:
                raise HTTPException(
                    status_code=413,
                    detail=f"Request payload too large. Maximum {max_size_mb}MB allowed."
                )
    return True


def get_timezone_from_request(request: Request) -> str:
    """Get timezone from request headers or default"""
    timezone = request.headers.get("x-timezone", "UTC")
    
    # Validate timezone (basic check)
    try:
        from datetime import timezone as dt_timezone
        # This is a simple validation - in production, use pytz
        if timezone not in ["UTC", "EST", "PST", "CST", "MST"]:
            timezone = "UTC"
    except Exception:
        timezone = "UTC"
    
    return timezone


# Health check dependencies
async def check_optimizer_health(optimizer: ScheduleOptimizer = Depends(get_optimizer)) -> bool:
    """Check if optimizer is healthy"""
    try:
        # Basic health check - could be expanded
        return hasattr(optimizer, 'generate_schedule')
    except Exception:
        return False


async def check_cache_health(cache: CacheManager = Depends(get_cache_manager)) -> bool:
    """Check if cache is healthy"""
    try:
        # Test cache connectivity
        await cache.set("health_check", "ok", ttl=10)
        result = await cache.get("health_check")
        await cache.delete("health_check")
        return result == "ok"
    except Exception:
        return False


# Startup/shutdown event handlers
async def startup_event():
    """Initialize dependencies on startup"""
    logger.info("Initializing application dependencies...")
    
    try:
        # Initialize optimizer
        get_optimizer()
        
        # Initialize cache
        cache = get_cache_manager()
        await cache.set("startup_time", datetime.now().isoformat(), ttl=86400)
        
        # Initialize profiler
        get_profiler()
        
        logger.info("All dependencies initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize dependencies: {e}")
        raise


async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down application dependencies...")
    
    try:
        # Cleanup cache
        if _cache_manager_instance:
            await _cache_manager_instance.clear()
        
        # Reset profiler
        if _profiler_instance:
            _profiler_instance.reset()
        
        logger.info("Dependencies cleaned up successfully")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# Custom dependency for request tracing
class RequestTracer:
    """Request tracing dependency for distributed tracing"""
    
    def __init__(self, request: Request):
        self.request = request
        self.trace_id = request.headers.get("x-trace-id") or f"trace_{int(time.time() * 1000)}"
        self.span_id = f"span_{int(time.time() * 1000000)}"
    
    def get_trace_context(self) -> Dict[str, str]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.request.headers.get("x-parent-span-id"),
        }


def get_request_tracer(request: Request) -> RequestTracer:
    """Get request tracer for distributed tracing"""
    return RequestTracer(request)