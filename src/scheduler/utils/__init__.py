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

from .profiler import PerformanceProfiler, OptimizationProfiler, RequestProfiler
from .exceptions import (
    SchedulerError,
    OptimizationError,
    InfeasibleScheduleError,
    ValidationError,
    ConfigurationError,
    DataInconsistencyError,
    ResourceLimitError,
    SolverTimeoutError,
    CacheError,
    QualificationError,
    AvailabilityError,
    StaffingRatioError,
    PriorityWeightError,
    APIError,
    RateLimitError,
    AuthenticationError,
    AuthorizationError,
    ServiceUnavailableError
)
from .helpers import (
    generate_request_id,
    generate_hash,
    safe_division,
    clamp,
    normalize_name,
    validate_email,
    validate_phone,
    format_duration,
    format_percentage,
    get_week_dates,
    get_monday_of_week,
    time_to_minutes,
    minutes_to_time,
    calculate_time_overlap,
    is_weekend,
    get_age_group_from_age,
    calculate_staff_child_ratio,
    parse_time_range,
    format_time_range,
    get_business_days_in_month,
    chunk_list,
    flatten_list,
    deep_merge_dicts,
    sanitize_filename,
    convert_to_serializable,
    calculate_priority_score,
    validate_priority_weights,
    format_money,
    parse_csv_line,
    generate_color_for_staff,
    calculate_schedule_density,
    find_schedule_gaps,
    suggest_shift_consolidation,
    calculate_workload_balance,
    estimate_optimization_complexity,
    log_performance_warning
)
# Import from core module for proper organization
from ..core.validator import ScheduleValidator
from ..core.analyzer import ScheduleAnalyzer
from ..core.cache import CacheManager

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