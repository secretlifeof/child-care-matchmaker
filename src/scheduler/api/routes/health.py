"""
Health check and system status routes
"""

from datetime import datetime

from fastapi import APIRouter, Depends

from ...config import settings
from ...utils import PerformanceProfiler
from ..dependencies import get_cache_manager, get_profiler

router = APIRouter(tags=["System"])


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": settings.app_name,
        "version": settings.app_version,
        "timestamp": datetime.now().isoformat(),
    }


@router.get("/stats")
async def get_performance_stats(
    profiler: PerformanceProfiler = Depends(get_profiler),
    cache_manager = Depends(get_cache_manager)
):
    """Get performance statistics"""
    return {
        "performance": profiler.get_stats(),
        "cache_info": {
            "redis_available": cache_manager.redis_client is not None,
            "memory_cache_size": len(cache_manager.memory_cache),
        },
    }
