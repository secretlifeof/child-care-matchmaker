"""
Administrative routes for configuration and cache management
"""

import logging

from fastapi import APIRouter, Depends, HTTPException

from ...config import settings

try:
    from ...config import CONSTRAINT_TYPES, OPTIMIZATION_GOALS
except ImportError:
    CONSTRAINT_TYPES = {}
    OPTIMIZATION_GOALS = {}
from ..dependencies import get_cache_manager

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Administration"])


@router.get("/config/constraints")
async def get_constraint_types():
    """Get available constraint types and their configurations"""
    return {
        "constraint_types": CONSTRAINT_TYPES,
        "optimization_goals": list(OPTIMIZATION_GOALS.keys()),
        "default_settings": {
            "max_solver_time": settings.max_solver_time_seconds,
            "max_consecutive_hours": settings.max_consecutive_hours,
            "min_break_between_shifts": settings.min_break_between_shifts,
            "max_weekly_hours": settings.max_weekly_hours,
        },
    }


@router.get("/config/ratios")
async def get_staffing_ratios():
    """Get default staff-to-child ratios by age group"""
    return {
        "ratios": {
            "infant": settings.infant_ratio,
            "toddler": settings.toddler_ratio,
            "preschool": settings.preschool_ratio,
        },
        "description": "Staff-to-child ratios (1 staff per X children)",
    }


@router.delete("/cache/clear", tags=["Cache Management"])
async def clear_cache(cache_manager = Depends(get_cache_manager)):
    """Clear all cached optimization results (Admin only)"""
    if not settings.DEBUG:
        raise HTTPException(status_code=403, detail="Not available in production")

    success = await cache_manager.clear()
    if success:
        logger.info("Cache cleared by admin request")
        return {"message": "Cache cleared successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to clear cache")


@router.get("/cache/stats", tags=["Cache Management"])
async def get_cache_stats(cache_manager = Depends(get_cache_manager)):
    """Get cache statistics (Admin only)"""
    if not settings.DEBUG:
        raise HTTPException(status_code=403, detail="Not available in production")

    return {
        "redis_available": cache_manager.redis_client is not None,
        "memory_cache_size": len(cache_manager.memory_cache),
        "cache_timestamps": len(cache_manager.cache_timestamps),
    }
