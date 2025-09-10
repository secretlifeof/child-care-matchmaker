"""
Daycare Schedule Optimizer

An advanced constraint programming API service for automatically generating
optimal work schedules for daycare centers.

Features:
- Constraint programming optimization using OR-Tools
- Multi-objective optimization (cost, satisfaction, fairness)
- Staff priority weights and preferences
- Age-specific ratios and qualification requirements
- Real-time schedule generation and validation
- FastAPI REST API with comprehensive documentation
- Redis caching for performance
- Docker deployment ready

Example:
    Basic usage with the API:

    ```python
    import httpx

    request_data = {
        "center_id": "uuid",
        "week_start_date": "2024-01-15",
        "staff": [...],
        "groups": [...],
        "staffing_requirements": [...]
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/schedule/generate",
            json=request_data
        )
        schedule = response.json()
    ```

    Direct usage with the optimizer:

    ```python
    from scheduler import ScheduleOptimizer, ScheduleGenerationRequest

    optimizer = ScheduleOptimizer()
    request = ScheduleGenerationRequest(...)
    response = await optimizer.generate_schedule(request)
    ```
"""

__version__ = "1.0.0"
__author__ = "Your Organization"
__email__ = "contact@yourorg.com"
__license__ = "MIT"
__description__ = (
    "Advanced constraint programming API service for daycare staff scheduling"
)
__url__ = "https://github.com/yourorg/daycare-schedule-optimizer"

# Package metadata
__all__ = [
    "ScheduleOptimizer",
    "ScheduleSolver",
    "ScheduleSolver",
    "ScheduleGenerationRequest",
    "ScheduleGenerationResponse",
    "ScheduleValidationRequest",
    "ScheduleValidationResponse",
    "Staff",
    "Group",
    "StaffingRequirement",
    "ScheduledShift",
    "OptimizationConfig",
    "OptimizationGoal",
    "StaffRole",
    "AgeGroup",
    "PreferenceType",
    "PriorityLevel",
]

# Import main classes for convenience
try:
    from .config import settings
    from .enchanced_solver import ScheduleSolver
    from .models import (
        AgeGroup,
        Group,
        OptimizationConfig,
        OptimizationGoal,
        PreferenceType,
        PriorityLevel,
        ScheduledShift,
        ScheduleGenerationRequest,
        ScheduleGenerationResponse,
        ScheduleValidationRequest,
        ScheduleValidationResponse,
        Staff,
        StaffingRequirement,
        StaffRole,
    )
    from .scheduler import ScheduleOptimizer
    from .solver import ScheduleSolver
except ImportError:
    # Handle import errors gracefully during package building
    pass

# Version info tuple
VERSION_INFO = tuple(int(x) for x in __version__.split("."))

# Check Python version compatibility
import sys

if sys.version_info < (3, 11):
    raise RuntimeError(
        f"Daycare Schedule Optimizer requires Python 3.11 or higher. "
        f"You are using Python {sys.version_info.major}.{sys.version_info.minor}."
    )


# Optional dependency checks with helpful error messages
def _check_optional_dependencies():
    """Check for optional dependencies and provide helpful messages."""
    missing_deps = []

    try:
        import redis
    except ImportError:
        missing_deps.append("redis")

    try:
        import prometheus_client
    except ImportError:
        # This is truly optional for monitoring
        pass

    if missing_deps:
        import warnings

        warnings.warn(
            f"Optional dependencies not found: {', '.join(missing_deps)}. "
            f"Some features may not be available. "
            f"Install with: pip install daycare-schedule-optimizer[all]",
            ImportWarning,
            stacklevel=2,
        )


# Check optional dependencies on import
_check_optional_dependencies()


# Package-level configuration
def get_version():
    """Get the package version."""
    return __version__


def get_version_info():
    """Get the package version as a tuple."""
    return VERSION_INFO


# Health check function for the package
def health_check():
    """
    Perform a basic health check of the package and its dependencies.

    Returns:
        dict: Health check results
    """
    health = {
        "package": "scheduler",
        "version": __version__,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "dependencies": {},
        "status": "healthy",
    }

    # Check core dependencies
    try:
        import fastapi

        health["dependencies"]["fastapi"] = fastapi.__version__
    except ImportError:
        health["dependencies"]["fastapi"] = "missing"
        health["status"] = "unhealthy"

    try:
        import ortools

        health["dependencies"]["ortools"] = ortools.__version__
    except ImportError:
        health["dependencies"]["ortools"] = "missing"
        health["status"] = "unhealthy"

    try:
        import pydantic

        health["dependencies"]["pydantic"] = pydantic.__version__
    except ImportError:
        health["dependencies"]["pydantic"] = "missing"
        health["status"] = "unhealthy"

    # Check optional dependencies
    try:
        import redis

        health["dependencies"]["redis"] = redis.__version__
    except ImportError:
        health["dependencies"]["redis"] = "optional"

    return health


# Logging configuration for the package
def setup_package_logging():
    """Set up basic logging for the package."""
    import logging

    # Create logger for the package
    logger = logging.getLogger(__name__)

    # Only add handler if none exists
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger


# Initialize package logging
_logger = setup_package_logging()
_logger.info(f"Daycare Schedule Optimizer v{__version__} loaded")


# Expose commonly used exceptions
class SchedulerError(Exception):
    """Base exception for scheduler-related errors."""

    pass


class OptimizationError(SchedulerError):
    """Exception raised when optimization fails."""

    pass


class ValidationError(SchedulerError):
    """Exception raised when validation fails."""

    pass


class ConfigurationError(SchedulerError):
    """Exception raised when configuration is invalid."""

    pass


# Add to __all__
__all__.extend(
    [
        "SchedulerError",
        "OptimizationError",
        "ValidationError",
        "ConfigurationError",
        "health_check",
        "get_version",
        "get_version_info",
        "setup_package_logging",
    ]
)

# Package metadata for introspection
__package_info__ = {
    "name": "daycare-schedule-optimizer",
    "version": __version__,
    "author": __author__,
    "email": __email__,
    "description": __description__,
    "url": __url__,
    "license": __license__,
    "python_requires": ">=3.11",
    "dependencies": [
        "fastapi>=0.104.1",
        "uvicorn>=0.24.0",
        "pydantic>=2.5.0",
        "ortools>=9.8.3296",
        "numpy>=1.25.2",
        "structlog>=23.2.0",
    ],
    "optional_dependencies": {
        "cache": ["redis>=5.0.1"],
        "monitoring": ["prometheus-client>=0.19.0"],
        "docs": ["mkdocs>=1.5.3", "mkdocs-material>=9.4.8"],
        "dev": ["pytest>=7.4.3", "black>=23.11.0", "mypy>=1.7.1"],
    },
}
