"""
FastAPI main application for the Daycare Schedule Optimization Service
"""

import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.openapi.utils import get_openapi
import uvicorn

from .config import settings
from .models import *
from .core.optimizer import ScheduleOptimizer
from .core.cache import CacheManager
from .utils.profiler import PerformanceProfiler
from .logging_config import setup_logging
from .api.routes import schedule_router
from .api.routes.health import router as health_router
from .api.routes.admin import router as admin_router
from .api.routes.validation import router as validation_router




# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Global instances
optimizer = ScheduleOptimizer()
cache_manager = CacheManager()
profiler = PerformanceProfiler()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events"""
    # Startup
    logger.info("Starting Daycare Schedule Optimization Service")
    logger.info(f"Version: {settings.app_version}")
    logger.info(f"Environment: {'Development' if settings.DEBUG else 'Production'}")

    # Test cache connection
    await cache_manager.set("startup_test", "success", 60)
    test_result = await cache_manager.get("startup_test")
    if test_result == "success":
        logger.info("Cache system operational")
        await cache_manager.delete("startup_test")
    else:
        logger.warning("Cache system not operational - using fallback")

    yield

    # Shutdown
    logger.info("Shutting down Daycare Schedule Optimization Service")
    await cache_manager.clear()


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="""
    An advanced constraint programming service for optimizing work schedules in daycare centers.
    
    This service uses OR-Tools CP-SAT solver to generate optimal staff schedules while considering:
    - Staff availability and preferences
    - Qualification requirements
    - Labor law constraints
    - Business rules and policies
    - Cost optimization
    - Fairness and satisfaction
    
    Features:
    - Real-time schedule generation
    - Conflict detection and resolution
    - Schedule validation
    - Performance analytics
    - Caching for improved performance
    """,
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.DEBUG else ["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include routers
try:
    app.include_router(schedule_router, prefix="/api")
    logger.info("Schedule router included")
except Exception as e:
    logger.warning(f"Failed to include schedule router: {e}")

try:
    app.include_router(health_router)
    logger.info("Health router included")
except Exception as e:
    logger.warning(f"Failed to include health router: {e}")

try:
    app.include_router(validation_router, prefix="/api")
    logger.info("Validation router included")
except Exception as e:
    logger.warning(f"Failed to include validation router: {e}")

try:
    app.include_router(admin_router, prefix="/api")
    logger.info("Admin router included")
except Exception as e:
    logger.warning(f"Failed to include admin router: {e}")


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    logger.warning(f"HTTP {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500, content={"error": "Internal server error", "status_code": 500}
    )


# Custom OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=settings.app_name,
        version=settings.app_version,
        description=app.description,
        routes=app.routes,
    )

    # Add custom schema elements
    openapi_schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
    }

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


# Run the application
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.DEBUG,
        log_level="debug" if settings.DEBUG else "info",
        workers=1 if settings.DEBUG else 4,
    )
