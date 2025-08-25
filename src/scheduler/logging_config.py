"""
Logging configuration for the Schedule Optimization Service
"""
import logging
import logging.config
import sys
from datetime import datetime
from pathlib import Path

import structlog
from pythonjsonlogger import jsonlogger

from .config import settings


def setup_logging():
    """Setup structured logging configuration"""
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure standard logging
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "detailed": {
                "format": "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(funcName)s(): %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "json": {
                "()": jsonlogger.JsonFormatter,
                "format": "%(asctime)s %(name)s %(levelname)s %(message)s %(pathname)s %(lineno)d %(funcName)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "DEBUG" if settings.DEBUG else "INFO",
                "formatter": "standard",
                "stream": sys.stdout
            },
            "file_info": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "INFO",
                "formatter": "detailed",
                "filename": log_dir / "scheduler.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
                "encoding": "utf8"
            },
            "file_error": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "ERROR",
                "formatter": "detailed",
                "filename": log_dir / "scheduler_errors.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
                "encoding": "utf8"
            },
            "json_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "INFO",
                "formatter": "json",
                "filename": log_dir / "scheduler_structured.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
                "encoding": "utf8"
            }
        },
        "loggers": {
            # Root logger
            "": {
                "level": "INFO",
                "handlers": ["console", "file_info", "file_error"],
                "propagate": False
            },
            # Application loggers
            "main": {
                "level": "DEBUG" if settings.DEBUG else "INFO",
                "handlers": ["console", "file_info", "json_file"],
                "propagate": False
            },
            "scheduler": {
                "level": "DEBUG" if settings.DEBUG else "INFO",
                "handlers": ["console", "file_info", "json_file"],
                "propagate": False
            },
            "solver": {
                "level": "DEBUG" if settings.DEBUG else "INFO",
                "handlers": ["console", "file_info", "json_file"],
                "propagate": False
            },
            "constraints": {
                "level": "DEBUG" if settings.DEBUG else "INFO",
                "handlers": ["console", "file_info", "json_file"],
                "propagate": False
            },
            "utils": {
                "level": "INFO",
                "handlers": ["console", "file_info"],
                "propagate": False
            },
            # Third-party loggers
            "uvicorn": {
                "level": "INFO",
                "handlers": ["console", "file_info"],
                "propagate": False
            },
            "uvicorn.error": {
                "level": "INFO",
                "handlers": ["console", "file_error"],
                "propagate": False
            },
            "uvicorn.access": {
                "level": "INFO",
                "handlers": ["file_info"],
                "propagate": False
            },
            "fastapi": {
                "level": "INFO",
                "handlers": ["console", "file_info"],
                "propagate": False
            },
            # Suppress noisy loggers
            "ortools": {
                "level": "WARNING",
                "handlers": ["file_info"],
                "propagate": False
            },
            "redis": {
                "level": "WARNING",
                "handlers": ["file_info"],
                "propagate": False
            }
        }
    }
    
    # Apply logging configuration
    logging.config.dictConfig(logging_config)
    
    # Configure structlog for structured logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Log startup message
    logger = logging.getLogger("main")
    logger.info(f"Logging configured - Debug: {settings.DEBUG}")


class PerformanceLogger:
    """Performance logging utility for tracking optimization metrics"""
    
    def __init__(self, logger_name: str = "performance"):
        self.logger = logging.getLogger(logger_name)
        self.structured_logger = structlog.get_logger(logger_name)
    
    def log_optimization_start(self, center_id: str, staff_count: int, group_count: int, week_date: str):
        """Log optimization start"""
        self.structured_logger.info(
            "optimization_started",
            center_id=center_id,
            staff_count=staff_count,
            group_count=group_count,
            week_date=week_date,
            timestamp=datetime.now().isoformat()
        )
    
    def log_optimization_result(
        self,
        center_id: str,
        status: str,
        solve_time: float,
        objective_value: float,
        shift_count: int,
        conflict_count: int
    ):
        """Log optimization result"""
        self.structured_logger.info(
            "optimization_completed",
            center_id=center_id,
            status=status,
            solve_time_seconds=solve_time,
            objective_value=objective_value,
            shift_count=shift_count,
            conflict_count=conflict_count,
            timestamp=datetime.now().isoformat()
        )
    
    def log_constraint_violation(
        self,
        constraint_type: str,
        severity: str,
        description: str,
        staff_id: str = None,
        group_id: str = None
    ):
        """Log constraint violation"""
        self.structured_logger.warning(
            "constraint_violation",
            constraint_type=constraint_type,
            severity=severity,
            description=description,
            staff_id=staff_id,
            group_id=group_id,
            timestamp=datetime.now().isoformat()
        )
    
    def log_cache_operation(self, operation: str, key: str, success: bool, duration: float = None):
        """Log cache operation"""
        self.structured_logger.debug(
            "cache_operation",
            operation=operation,
            key=key,
            success=success,
            duration_ms=duration * 1000 if duration else None,
            timestamp=datetime.now().isoformat()
        )
    
    def log_api_request(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        duration: float,
        request_size: int = None,
        response_size: int = None
    ):
        """Log API request"""
        self.structured_logger.info(
            "api_request",
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            duration_ms=duration * 1000,
            request_size_bytes=request_size,
            response_size_bytes=response_size,
            timestamp=datetime.now().isoformat()
        )


class SolverLogger:
    """Specialized logger for OR-Tools solver operations"""
    
    def __init__(self):
        self.logger = logging.getLogger("solver")
        self.structured_logger = structlog.get_logger("solver")
    
    def log_model_creation(self, variables_count: int, constraints_count: int):
        """Log model creation statistics"""
        self.structured_logger.info(
            "model_created",
            variables_count=variables_count,
            constraints_count=constraints_count,
            timestamp=datetime.now().isoformat()
        )
    
    def log_solver_progress(self, iteration: int, objective_value: float, time_elapsed: float):
        """Log solver progress"""
        self.structured_logger.debug(
            "solver_progress",
            iteration=iteration,
            objective_value=objective_value,
            time_elapsed_seconds=time_elapsed,
            timestamp=datetime.now().isoformat()
        )
    
    def log_constraint_added(self, constraint_type: str, count: int):
        """Log constraint addition"""
        self.structured_logger.debug(
            "constraints_added",
            constraint_type=constraint_type,
            count=count,
            timestamp=datetime.now().isoformat()
        )
    
    def log_infeasibility_analysis(self, conflicting_constraints: list):
        """Log infeasibility analysis results"""
        self.structured_logger.warning(
            "infeasibility_detected",
            conflicting_constraints=conflicting_constraints,
            timestamp=datetime.now().isoformat()
        )


class RequestLogger:
    """Logger for API request/response tracking"""
    
    def __init__(self):
        self.logger = logging.getLogger("requests")
        self.performance_logger = PerformanceLogger("requests")
    
    async def log_request(self, request, call_next):
        """Middleware for logging requests"""
        start_time = datetime.now()
        
        # Log request start
        self.logger.info(f"Request started: {request.method} {request.url.path}")
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration = (datetime.now() - start_time).total_seconds()
        
        # Log request completion
        self.performance_logger.log_api_request(
            endpoint=request.url.path,
            method=request.method,
            status_code=response.status_code,
            duration=duration
        )
        
        # Log response
        if response.status_code >= 400:
            self.logger.warning(
                f"Request completed with error: {request.method} {request.url.path} "
                f"-> {response.status_code} in {duration:.3f}s"
            )
        else:
            self.logger.info(
                f"Request completed: {request.method} {request.url.path} "
                f"-> {response.status_code} in {duration:.3f}s"
            )
        
        return response


# Export logger instances
performance_logger = PerformanceLogger()
solver_logger = SolverLogger()
request_logger = RequestLogger()