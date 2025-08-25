"""
Configuration settings for the Schedule Optimization Service
"""

import os
from typing import List, Dict, Any, Optional
from pydantic import Field, validator
from pydantic_settings import BaseSettings

from datetime import time
from enum import Enum


class Settings(BaseSettings):
    # API Settings
    app_name: str = "Daycare Schedule Optimizer"
    app_version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    # Redis Settings (for caching optimization results)
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = ""
    cache_ttl: int = 3600  # 1 hour

    # Optimization Settings
    max_solver_time_seconds: int = 60
    max_staff_count: int = 100
    max_time_slots_per_week: int = 168  # 24 hours * 7 days

    # Constraint Weights (0.0 to 1.0)
    preference_weight: float = 0.3
    overtime_penalty_weight: float = 0.4
    fairness_weight: float = 0.2
    continuity_weight: float = 0.1

    # Business Rules
    max_consecutive_hours: int = 8
    min_break_between_shifts: int = 10  # hours
    max_weekly_hours: int = 40
    min_shift_duration: float = 2.0  # hours
    max_shift_duration: float = 10.0  # hours

    # Staff-to-Child Ratios by Age Group
    infant_ratio: int = 4  # 1 staff per 4 infants
    toddler_ratio: int = 6  # 1 staff per 6 toddlers
    preschool_ratio: int = 10  # 1 staff per 10 preschoolers

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = 'ignore'


# Global settings instance
settings = Settings()

# Optimization Goals Priority Order
OPTIMIZATION_GOALS = {
    "minimize_cost": 1,
    "maximize_satisfaction": 2,
    "minimize_overtime": 3,
    "maximize_fairness": 4,
    "maximize_continuity": 5,
}

# Constraint Types
CONSTRAINT_TYPES = {
    "HARD": {
        "staffing_ratio": "Minimum staff per group per time slot",
        "qualifications": "Required certifications for specific roles",
        "availability": "Staff cannot be scheduled when unavailable",
        "labor_laws": "Maximum hours, required breaks",
        "shift_continuity": "Minimum shift durations",
    },
    "SOFT": {
        "preferences": "Staff preferred days/times",
        "overtime": "Minimize excessive hours",
        "fairness": "Fair distribution across staff",
        "travel_time": "Minimize quick shift changes",
        "seniority": "Preference by experience level",
    },
}

# Default conflict resolution strategies
CONFLICT_RESOLUTION = {
    "understaffed": ["hire_substitute", "extend_shift", "call_backup"],
    "overstaffed": ["reduce_hours", "reassign_group", "schedule_break"],
    "qualification_missing": ["find_qualified", "provide_supervision", "reschedule"],
    "overtime_violation": ["split_shift", "reassign_staff", "adjust_hours"],
}


class OptimizationStrategy(str, Enum):
    """Available optimization strategies"""

    STANDARD = "standard"
    CHUNKED = "chunked"
    PARALLEL = "parallel"
    HIERARCHICAL = "hierarchical"
    INCREMENTAL = "incremental"
    AUTO = "auto"


class CacheBackend(str, Enum):
    """Cache backend options"""

    MEMORY = "memory"
    REDIS = "redis"
    MEMCACHED = "memcached"
    DATABASE = "database"


class EnhancedSchedulerSettings(BaseSettings):
    """Enhanced settings for the schedule optimization system"""
    app_name: str = "Daycare Schedule Optimizer"
    app_version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    # Redis Settings (for caching optimization results)
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = ""
    cache_ttl: int = 3600  # 1 hour

    # Optimization Settings
    max_solver_time_seconds: int = 60
    max_staff_count: int = 100
    max_time_slots_per_week: int = 168  # 24 hours * 7 days

    # Constraint Weights (0.0 to 1.0)
    preference_weight: float = 0.3
    overtime_penalty_weight: float = 0.4
    fairness_weight: float = 0.2
    continuity_weight: float = 0.1

    # Business Rules
    max_consecutive_hours: int = 8
    min_break_between_shifts: int = 10  # hours
    max_weekly_hours: int = 40
    min_shift_duration: float = 2.0  # hours
    max_shift_duration: float = 10.0  # hours

    # Staff-to-Child Ratios by Age Group
    infant_ratio: int = 4  # 1 staff per 4 infants
    toddler_ratio: int = 6  # 1 staff per 6 toddlers
    preschool_ratio: int = 10  # 1 staff per 10 preschoolers

    # Basic settings (extend your existing ones)
    DEBUG: bool = Field(default=False, env="DEBUG")
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    # DATABASE_URL: str = Field(env="DATABASE_URL")
    
    # Authentication Settings
    ENABLE_API_KEY_AUTH: bool = Field(default=False, env="ENABLE_API_KEY_AUTH")
    REQUIRE_AUTH: bool = Field(default=False, env="REQUIRE_AUTH")
    API_KEY: Optional[str] = Field(default=None, env="API_KEY")
    
    # Maintenance Settings
    MAINTENANCE_MODE: bool = Field(default=False, env="MAINTENANCE_MODE")
    MAINTENANCE_MESSAGE: str = Field(default="System maintenance in progress", env="MAINTENANCE_MESSAGE")

    # Solver Configuration
    DEFAULT_SOLVER_TIME: int = Field(
        default=300,
        env="DEFAULT_SOLVER_TIME",
        description="Default solver time in seconds",
    )
    MAX_SOLVER_TIME: int = Field(
        default=1800, env="MAX_SOLVER_TIME", description="Maximum allowed solver time"
    )
    SOLVER_WORKERS: int = Field(
        default=4, env="SOLVER_WORKERS", description="Number of solver worker threads"
    )
    LOG_SOLVER_PROGRESS: bool = Field(default=True, env="LOG_SOLVER_PROGRESS")

    # Operating Hours
    OPERATING_START_HOUR: int = Field(
        default=6,
        env="OPERATING_START_HOUR",
        description="Daily operating start hour (24h format)",
    )
    OPERATING_END_HOUR: int = Field(
        default=20,
        env="OPERATING_END_HOUR",
        description="Daily operating end hour (24h format)",
    )

    # Schedule Constraints
    MAX_DAILY_HOURS: int = Field(
        default=12, env="MAX_DAILY_HOURS", description="Maximum hours per day per staff"
    )
    MAX_CONSECUTIVE_DAYS: int = Field(
        default=6,
        env="MAX_CONSECUTIVE_DAYS",
        description="Maximum consecutive working days",
    )
    MIN_SHIFT_DURATION: float = Field(
        default=2.0,
        env="MIN_SHIFT_DURATION",
        description="Minimum shift duration in hours",
    )
    MAX_SHIFT_DURATION: float = Field(
        default=12.0,
        env="MAX_SHIFT_DURATION",
        description="Maximum shift duration in hours",
    )
    MIN_REST_HOURS: int = Field(
        default=8, env="MIN_REST_HOURS", description="Minimum hours between shifts"
    )

    # Date Range Limits
    MAX_DATE_RANGE_DAYS: int = Field(
        default=365,
        env="MAX_DATE_RANGE_DAYS",
        description="Maximum scheduling period in days",
    )
    DEFAULT_WEEK_DAYS: int = Field(default=7, env="DEFAULT_WEEK_DAYS")
    CHUNK_SIZE_THRESHOLD: int = Field(
        default=14,
        env="CHUNK_SIZE_THRESHOLD",
        description="Days threshold for chunking strategy",
    )

    # Performance Settings
    OPTIMIZATION_STRATEGY: OptimizationStrategy = Field(
        default=OptimizationStrategy.AUTO, env="OPTIMIZATION_STRATEGY"
    )
    AUTO_STRATEGY_SELECTION: bool = Field(
        default=True,
        env="AUTO_STRATEGY_SELECTION",
        description="Automatically select optimization strategy",
    )
    PARALLEL_THRESHOLD_GROUPS: int = Field(
        default=3,
        env="PARALLEL_THRESHOLD_GROUPS",
        description="Minimum groups for parallel optimization",
    )
    PARALLEL_THRESHOLD_STAFF: int = Field(
        default=10,
        env="PARALLEL_THRESHOLD_STAFF",
        description="Minimum staff for parallel optimization",
    )

    # Memory Management
    MAX_MEMORY_MB: int = Field(
        default=8192, env="MAX_MEMORY_MB", description="Maximum memory usage in MB"
    )
    MEMORY_WARNING_THRESHOLD: float = Field(
        default=0.8,
        env="MEMORY_WARNING_THRESHOLD",
        description="Memory usage warning threshold (0-1)",
    )
    ENABLE_MEMORY_MONITORING: bool = Field(default=True, env="ENABLE_MEMORY_MONITORING")

    # Cache Configuration
    CACHE_BACKEND: CacheBackend = Field(
        default=CacheBackend.MEMORY, env="CACHE_BACKEND"
    )
    CACHE_TTL_SECONDS: int = Field(
        default=3600,
        env="CACHE_TTL_SECONDS",
        description="Cache time-to-live in seconds",
    )
    CACHE_MAX_SIZE: int = Field(
        default=1000, env="CACHE_MAX_SIZE", description="Maximum cache entries"
    )
    REDIS_URL: Optional[str] = Field(default=None, env="REDIS_URL")
    MEMCACHED_SERVERS: List[str] = Field(
        default=["localhost:11211"], env="MEMCACHED_SERVERS"
    )

    # Optimization Weights
    PREFERENCE_WEIGHT: float = Field(
        default=1.0, env="PREFERENCE_WEIGHT", description="Weight for staff preferences"
    )
    FAIRNESS_WEIGHT: float = Field(
        default=1.0, env="FAIRNESS_WEIGHT", description="Weight for workload fairness"
    )
    COST_WEIGHT: float = Field(
        default=1.0, env="COST_WEIGHT", description="Weight for cost optimization"
    )
    CONTINUITY_WEIGHT: float = Field(
        default=0.5, env="CONTINUITY_WEIGHT", description="Weight for shift continuity"
    )

    # Business Rules
    ALLOW_SPLIT_SHIFTS: bool = Field(
        default=True,
        env="ALLOW_SPLIT_SHIFTS",
        description="Allow split shifts for same staff/day",
    )
    PREFER_FULL_DAYS: bool = Field(
        default=False, env="PREFER_FULL_DAYS", description="Prefer full day assignments"
    )
    ENABLE_OVERTIME: bool = Field(
        default=True, env="ENABLE_OVERTIME", description="Allow overtime assignments"
    )
    OVERTIME_MULTIPLIER: float = Field(
        default=1.5, env="OVERTIME_MULTIPLIER", description="Overtime pay multiplier"
    )

    # Validation Settings
    STRICT_VALIDATION: bool = Field(
        default=True,
        env="STRICT_VALIDATION",
        description="Enable strict input validation",
    )
    ALLOW_UNDERSTAFFING: bool = Field(
        default=False,
        env="ALLOW_UNDERSTAFFING",
        description="Allow understaffing if no solution",
    )
    MAX_CONFLICTS_ALLOWED: int = Field(
        default=10,
        env="MAX_CONFLICTS_ALLOWED",
        description="Maximum conflicts before failing",
    )

    # Export/Import Settings
    DEFAULT_EXPORT_FORMAT: str = Field(default="json", env="DEFAULT_EXPORT_FORMAT")
    EXPORT_INCLUDE_ANALYTICS: bool = Field(default=True, env="EXPORT_INCLUDE_ANALYTICS")
    IMPORT_VALIDATION_ENABLED: bool = Field(
        default=True, env="IMPORT_VALIDATION_ENABLED"
    )
    BACKUP_GENERATED_SCHEDULES: bool = Field(
        default=True, env="BACKUP_GENERATED_SCHEDULES"
    )

    # Performance Monitoring
    ENABLE_PERFORMANCE_MONITORING: bool = Field(
        default=True, env="ENABLE_PERFORMANCE_MONITORING"
    )
    PERFORMANCE_SNAPSHOT_RETENTION_HOURS: int = Field(
        default=168,
        env="PERFORMANCE_SNAPSHOT_RETENTION_HOURS",
        description="Performance data retention (168h = 1 week)",
    )
    PERFORMANCE_ALERTS_ENABLED: bool = Field(
        default=True, env="PERFORMANCE_ALERTS_ENABLED"
    )
    SLOW_OPERATION_THRESHOLD_SECONDS: int = Field(
        default=300, env="SLOW_OPERATION_THRESHOLD_SECONDS"
    )

    # API Settings
    API_RATE_LIMIT_PER_MINUTE: int = Field(default=60, env="API_RATE_LIMIT_PER_MINUTE")
    API_MAX_REQUEST_SIZE_MB: int = Field(default=10, env="API_MAX_REQUEST_SIZE_MB")
    ASYNC_GENERATION_ENABLED: bool = Field(default=True, env="ASYNC_GENERATION_ENABLED")
    BACKGROUND_TASK_TIMEOUT: int = Field(default=3600, env="BACKGROUND_TASK_TIMEOUT")

    # Integration Settings
    WEBHOOK_ENABLED: bool = Field(default=False, env="WEBHOOK_ENABLED")
    WEBHOOK_URL: Optional[str] = Field(default=None, env="WEBHOOK_URL")
    EMAIL_NOTIFICATIONS_ENABLED: bool = Field(
        default=False, env="EMAIL_NOTIFICATIONS_ENABLED"
    )
    SLACK_NOTIFICATIONS_ENABLED: bool = Field(
        default=False, env="SLACK_NOTIFICATIONS_ENABLED"
    )
    SLACK_WEBHOOK_URL: Optional[str] = Field(default=None, env="SLACK_WEBHOOK_URL")

    # Development/Testing
    ENABLE_TEST_MODE: bool = Field(default=False, env="ENABLE_TEST_MODE")
    MOCK_SOLVER_RESPONSES: bool = Field(default=False, env="MOCK_SOLVER_RESPONSES")
    DEVELOPMENT_SEED: Optional[int] = Field(default=None, env="DEVELOPMENT_SEED")

    @validator("OPERATING_START_HOUR", "OPERATING_END_HOUR")
    def validate_operating_hours(cls, v):
        if not 0 <= v <= 23:
            raise ValueError("Operating hours must be between 0 and 23")
        return v

    @validator("OPERATING_END_HOUR")
    def validate_operating_end_after_start(cls, v, values):
        if "OPERATING_START_HOUR" in values and v <= values["OPERATING_START_HOUR"]:
            raise ValueError("Operating end hour must be after start hour")
        return v

    @validator("MAX_DAILY_HOURS")
    def validate_max_daily_hours(cls, v):
        if not 1 <= v <= 24:
            raise ValueError("Max daily hours must be between 1 and 24")
        return v

    @validator(
        "PREFERENCE_WEIGHT", "FAIRNESS_WEIGHT", "COST_WEIGHT", "CONTINUITY_WEIGHT"
    )
    def validate_weights(cls, v):
        if v < 0:
            raise ValueError("Optimization weights must be non-negative")
        return v

    @validator("MEMORY_WARNING_THRESHOLD")
    def validate_memory_threshold(cls, v):
        if not 0 < v <= 1:
            raise ValueError("Memory warning threshold must be between 0 and 1")
        return v

    @property
    def operating_hours_range(self) -> tuple[time, time]:
        """Get operating hours as time objects"""
        return (time(self.OPERATING_START_HOUR), time(self.OPERATING_END_HOUR))

    @property
    def total_operating_hours(self) -> int:
        """Get total operating hours per day"""
        return self.OPERATING_END_HOUR - self.OPERATING_START_HOUR

    @property
    def optimization_weights(self) -> Dict[str, float]:
        """Get optimization weights as dictionary"""
        return {
            "preference": self.PREFERENCE_WEIGHT,
            "fairness": self.FAIRNESS_WEIGHT,
            "cost": self.COST_WEIGHT,
            "continuity": self.CONTINUITY_WEIGHT,
        }

    @property
    def should_use_chunking(self) -> bool:
        """Determine if chunking should be used based on settings"""
        return self.OPTIMIZATION_STRATEGY == OptimizationStrategy.CHUNKED or (
            self.OPTIMIZATION_STRATEGY == OptimizationStrategy.AUTO
            and self.AUTO_STRATEGY_SELECTION
        )

    @property
    def solver_config(self) -> Dict[str, Any]:
        """Get solver configuration dictionary"""
        return {
            "max_time": self.DEFAULT_SOLVER_TIME,
            "workers": self.SOLVER_WORKERS,
            "log_progress": self.LOG_SOLVER_PROGRESS,
            "memory_limit_mb": self.MAX_MEMORY_MB,
        }

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = 'ignore'


class EnvironmentConfig:
    """Environment-specific configuration management"""

    @staticmethod
    def get_development_config() -> EnhancedSchedulerSettings:
        """Get development environment configuration"""
        return EnhancedSchedulerSettings(
            DEBUG=True,
            LOG_LEVEL="DEBUG",
            DEFAULT_SOLVER_TIME=60,  # Shorter for development
            ENABLE_TEST_MODE=True,
            STRICT_VALIDATION=False,
            CACHE_TTL_SECONDS=300,  # Shorter cache for development
            PERFORMANCE_ALERTS_ENABLED=False,
        )

    @staticmethod
    def get_testing_config() -> EnhancedSchedulerSettings:
        """Get testing environment configuration"""
        return EnhancedSchedulerSettings(
            DEBUG=True,
            LOG_LEVEL="WARNING",
            DEFAULT_SOLVER_TIME=30,  # Very short for tests
            ENABLE_TEST_MODE=True,
            MOCK_SOLVER_RESPONSES=True,
            CACHE_BACKEND=CacheBackend.MEMORY,
            ENABLE_PERFORMANCE_MONITORING=False,
            DEVELOPMENT_SEED=12345,  # Consistent seed for tests
        )

    @staticmethod
    def get_production_config() -> EnhancedSchedulerSettings:
        """Get production environment configuration"""
        return EnhancedSchedulerSettings(
            DEBUG=False,
            LOG_LEVEL="INFO",
            DEFAULT_SOLVER_TIME=300,
            MAX_SOLVER_TIME=1800,
            STRICT_VALIDATION=True,
            ENABLE_PERFORMANCE_MONITORING=True,
            CACHE_BACKEND=CacheBackend.REDIS,
            PERFORMANCE_ALERTS_ENABLED=True,
            BACKUP_GENERATED_SCHEDULES=True,
        )


class ConfigValidator:
    """Validate configuration settings"""

    @staticmethod
    def validate_solver_config(settings: EnhancedSchedulerSettings) -> List[str]:
        """Validate solver-related configuration"""
        issues = []

        if settings.DEFAULT_SOLVER_TIME > settings.MAX_SOLVER_TIME:
            issues.append("DEFAULT_SOLVER_TIME cannot exceed MAX_SOLVER_TIME")

        if settings.SOLVER_WORKERS < 1:
            issues.append("SOLVER_WORKERS must be at least 1")

        if settings.SOLVER_WORKERS > 16:
            issues.append("SOLVER_WORKERS should not exceed 16 for optimal performance")

        return issues

    @staticmethod
    def validate_business_rules(settings: EnhancedSchedulerSettings) -> List[str]:
        """Validate business rule configuration"""
        issues = []

        if settings.MIN_SHIFT_DURATION > settings.MAX_SHIFT_DURATION:
            issues.append("MIN_SHIFT_DURATION cannot exceed MAX_SHIFT_DURATION")

        if settings.MAX_SHIFT_DURATION > settings.MAX_DAILY_HOURS:
            issues.append("MAX_SHIFT_DURATION cannot exceed MAX_DAILY_HOURS")

        if settings.MIN_REST_HOURS < 4:
            issues.append("MIN_REST_HOURS should be at least 4 for staff wellbeing")

        return issues

    @staticmethod
    def validate_performance_config(settings: EnhancedSchedulerSettings) -> List[str]:
        """Validate performance-related configuration"""
        issues = []

        if settings.MAX_MEMORY_MB < 1024:
            issues.append("MAX_MEMORY_MB should be at least 1024 MB")

        if settings.CACHE_TTL_SECONDS < 60:
            issues.append("CACHE_TTL_SECONDS should be at least 60 seconds")

        if settings.MAX_DATE_RANGE_DAYS > 730:  # 2 years
            issues.append("MAX_DATE_RANGE_DAYS should not exceed 730 days")

        return issues

    @staticmethod
    def validate_all(settings: EnhancedSchedulerSettings) -> Dict[str, List[str]]:
        """Validate all configuration settings"""
        return {
            "solver": ConfigValidator.validate_solver_config(settings),
            "business_rules": ConfigValidator.validate_business_rules(settings),
            "performance": ConfigValidator.validate_performance_config(settings),
        }


# Global settings instance
def get_settings() -> EnhancedSchedulerSettings:
    """Get application settings based on environment"""

    environment = os.getenv("ENVIRONMENT", "development").lower()

    if environment == "production":
        return EnvironmentConfig.get_production_config()
    elif environment == "testing":
        return EnvironmentConfig.get_testing_config()
    else:
        return EnvironmentConfig.get_development_config()


# Configuration factory
class ConfigFactory:
    """Factory for creating configuration objects"""

    @staticmethod
    def create_optimization_config(
        goals: List[str] = None,
        max_solver_time: int = None,
        strategy: OptimizationStrategy = None,
    ) -> Dict[str, Any]:
        """Create optimization configuration dictionary"""

        settings = get_settings()

        config = {
            "goals": goals or ["MAXIMIZE_SATISFACTION", "MINIMIZE_COST"],
            "max_solver_time": max_solver_time or settings.DEFAULT_SOLVER_TIME,
            "consider_preferences": True,
            "weights": settings.optimization_weights,
        }

        if strategy:
            config["strategy"] = strategy.value

        return config

    @staticmethod
    def create_chunking_config(total_days: int) -> Dict[str, Any]:
        """Create chunking configuration based on total days"""

        settings = get_settings()

        if total_days <= 7:
            chunk_size = total_days
        elif total_days <= 14:
            chunk_size = 7
        elif total_days <= 30:
            chunk_size = 10
        else:
            chunk_size = 14

        return {
            "enabled": total_days > settings.CHUNK_SIZE_THRESHOLD,
            "chunk_size": chunk_size,
            "overlap_days": 1,
            "merge_strategy": "continuous",
        }

    @staticmethod
    def create_validation_config(strict: bool = None) -> Dict[str, Any]:
        """Create validation configuration"""

        settings = get_settings()

        return {
            "strict_validation": strict
            if strict is not None
            else settings.STRICT_VALIDATION,
            "allow_understaffing": settings.ALLOW_UNDERSTAFFING,
            "max_conflicts": settings.MAX_CONFLICTS_ALLOWED,
            "validate_imports": settings.IMPORT_VALIDATION_ENABLED,
        }


# Usage examples and helper functions
def load_config_from_file(config_path: str) -> EnhancedSchedulerSettings:
    """Load configuration from a specific file"""

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    return EnhancedSchedulerSettings(_env_file=config_path)


def validate_current_config() -> Dict[str, Any]:
    """Validate the current configuration and return issues"""

    settings = get_settings()
    validation_results = ConfigValidator.validate_all(settings)

    # Count total issues
    total_issues = sum(len(issues) for issues in validation_results.values())

    return {
        "valid": total_issues == 0,
        "total_issues": total_issues,
        "issues_by_category": validation_results,
        "settings_summary": {
            "environment": os.getenv("ENVIRONMENT", "development"),
            "solver_time": settings.DEFAULT_SOLVER_TIME,
            "max_memory": settings.MAX_MEMORY_MB,
            "cache_backend": settings.CACHE_BACKEND.value,
            "optimization_strategy": settings.OPTIMIZATION_STRATEGY.value,
        },
    }


def get_runtime_config() -> Dict[str, Any]:
    """Get runtime configuration information"""

    settings = get_settings()

    return {
        "scheduler": {
            "version": "2.0.0",  # Your version
            "operating_hours": f"{settings.OPERATING_START_HOUR:02d}:00 - {settings.OPERATING_END_HOUR:02d}:00",
            "max_date_range": f"{settings.MAX_DATE_RANGE_DAYS} days",
            "default_solver_time": f"{settings.DEFAULT_SOLVER_TIME} seconds",
        },
        "performance": {
            "max_memory": f"{settings.MAX_MEMORY_MB} MB",
            "solver_workers": settings.SOLVER_WORKERS,
            "cache_backend": settings.CACHE_BACKEND.value,
            "monitoring_enabled": settings.ENABLE_PERFORMANCE_MONITORING,
        },
        "features": {
            "date_range_scheduling": True,
            "multi_week_generation": True,
            "parallel_optimization": True,
            "chunked_optimization": True,
            "performance_monitoring": settings.ENABLE_PERFORMANCE_MONITORING,
            "async_generation": settings.ASYNC_GENERATION_ENABLED,
        },
    }


# Export the main settings instance
settings = get_settings()

# Validate configuration on import
config_validation = validate_current_config()
if not config_validation["valid"]:
    import warnings

    warnings.warn(
        f"Configuration validation failed with {config_validation['total_issues']} issues"
    )
