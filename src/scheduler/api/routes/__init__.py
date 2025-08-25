"""
API routes for the Daycare Schedule Optimizer

This module contains all route handlers for different API endpoints:
- Schedule generation and optimization
- Schedule validation
- Health checks and system status
- Administrative functions
"""

# Import route modules when they're available
try:
    from .schedule import router as schedule_router
except ImportError:
    schedule_router = None

try:
    from .validation import router as validation_router
except ImportError:
    validation_router = None

try:
    from .health import router as health_router
except ImportError:
    health_router = None

try:
    from .admin import router as admin_router
except ImportError:
    admin_router = None

__all__ = [
    "schedule_router",
    "validation_router", 
    "health_router",
    "admin_router"
]

# List of all available routers
AVAILABLE_ROUTERS = [
    schedule_router,
    validation_router,
    health_router, 
    admin_router
]

# Filter out None routers (not yet implemented)
ROUTERS = [router for router in AVAILABLE_ROUTERS if router is not None]