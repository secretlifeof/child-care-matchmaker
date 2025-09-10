"""
Utility functions and classes for the Daycare Schedule Optimizer

This module contains utility functions, helper classes, and common tools:
- Performance profiling and monitoring
- Custom exceptions for error handling
- Helper functions for data processing
- Common constants and enumerations
- Schedule validation and analysis
- Cache management
"""

from ..core.analyzer import ScheduleAnalyzer
from ..core.cache import CacheManager

# Import from core module for proper organization
from ..core.validator import ScheduleValidator
from .exceptions import (
    APIError,
    AuthenticationError,
    AuthorizationError,
    AvailabilityError,
    CacheError,
    ConfigurationError,
    DataInconsistencyError,
    InfeasibleScheduleError,
    OptimizationError,
    PriorityWeightError,
    QualificationError,
    RateLimitError,
    ResourceLimitError,
    SchedulerError,
    ServiceUnavailableError,
    SolverTimeoutError,
    StaffingRatioError,
    ValidationError,
)
from .helpers import (
    calculate_priority_score,
    calculate_schedule_density,
    calculate_staff_child_ratio,
    calculate_time_overlap,
    calculate_workload_balance,
    chunk_list,
    clamp,
    convert_to_serializable,
    deep_merge_dicts,
    estimate_optimization_complexity,
    find_schedule_gaps,
    flatten_list,
    format_duration,
    format_money,
    format_percentage,
    format_time_range,
    generate_color_for_staff,
    generate_hash,
    generate_request_id,
    get_age_group_from_age,
    get_business_days_in_month,
    get_monday_of_week,
    get_week_dates,
    is_weekend,
    log_performance_warning,
    minutes_to_time,
    normalize_name,
    parse_csv_line,
    parse_time_range,
    safe_division,
    sanitize_filename,
    suggest_shift_consolidation,
    time_to_minutes,
    validate_email,
    validate_phone,
    validate_priority_weights,
)
from .profiler import OptimizationProfiler, PerformanceProfiler, RequestProfiler

__all__ = [
    # Profiling classes
    "PerformanceProfiler",
    "OptimizationProfiler",
    "RequestProfiler",

    # Validator classes
    "ScheduleValidator",
    "ScheduleAnalyzer",
    "CacheManager",

    # Exception classes
    "SchedulerError",
    "OptimizationError",
    "InfeasibleScheduleError",
    "ValidationError",
    "ConfigurationError",
    "DataInconsistencyError",
    "ResourceLimitError",
    "SolverTimeoutError",
    "CacheError",
    "QualificationError",
    "AvailabilityError",
    "StaffingRatioError",
    "PriorityWeightError",
    "APIError",
    "RateLimitError",
    "AuthenticationError",
    "AuthorizationError",
    "ServiceUnavailableError",

    # Helper functions
    "generate_request_id",
    "generate_hash",
    "safe_division",
    "clamp",
    "normalize_name",
    "validate_email",
    "validate_phone",
    "format_duration",
    "format_percentage",
    "get_week_dates",
    "get_monday_of_week",
    "time_to_minutes",
    "minutes_to_time",
    "calculate_time_overlap",
    "is_weekend",
    "get_age_group_from_age",
    "calculate_staff_child_ratio",
    "parse_time_range",
    "format_time_range",
    "get_business_days_in_month",
    "chunk_list",
    "flatten_list",
    "deep_merge_dicts",
    "sanitize_filename",
    "convert_to_serializable",
    "calculate_priority_score",
    "validate_priority_weights",
    "format_money",
    "parse_csv_line",
    "generate_color_for_staff",
    "calculate_schedule_density",
    "find_schedule_gaps",
    "suggest_shift_consolidation",
    "calculate_workload_balance",
    "estimate_optimization_complexity",
    "log_performance_warning"
]

# Utility version
UTILS_VERSION = "1.0.0"

# Exception hierarchies for easy importing
OPTIMIZATION_EXCEPTIONS = (
    OptimizationError,
    InfeasibleScheduleError,
    SolverTimeoutError
)

VALIDATION_EXCEPTIONS = (
    ValidationError,
    DataInconsistencyError,
    QualificationError,
    AvailabilityError,
    StaffingRatioError,
    PriorityWeightError
)

API_EXCEPTIONS = (
    APIError,
    RateLimitError,
    AuthenticationError,
    AuthorizationError,
    ServiceUnavailableError
)

SYSTEM_EXCEPTIONS = (
    ConfigurationError,
    ResourceLimitError,
    CacheError
)

ALL_EXCEPTIONS = (
    OPTIMIZATION_EXCEPTIONS +
    VALIDATION_EXCEPTIONS +
    API_EXCEPTIONS +
    SYSTEM_EXCEPTIONS
)
