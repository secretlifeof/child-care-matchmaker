"""
API module for the Daycare Schedule Optimizer

This module contains all HTTP API related components including:
- Route handlers for different endpoints
- Middleware for request/response processing
- Dependency injection for API components
- Authentication and authorization logic
"""

from .dependencies import (
    get_optimizer,
    get_cache_manager,
    get_profiler,
    get_current_user,
    require_auth,
    rate_limit
)

__all__ = [
    "get_optimizer",
    "get_cache_manager", 
    "get_profiler",
    "get_current_user",
    "require_auth",
    "rate_limit"
]