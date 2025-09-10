"""
Custom exceptions for the scheduler service
"""
from typing import Any


class SchedulerError(Exception):
    """Base exception for scheduler-related errors"""

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        details: dict[str, Any] | None = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__.upper()
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for API responses"""
        return {
            "error": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details
        }


class OptimizationError(SchedulerError):
    """Exception raised when optimization fails"""

    def __init__(
        self,
        message: str,
        solver_status: str | None = None,
        solve_time: float | None = None,
        iterations: int | None = None,
        details: dict[str, Any] | None = None
    ):
        super().__init__(message, "OPTIMIZATION_FAILED", details)
        self.solver_status = solver_status
        self.solve_time = solve_time
        self.iterations = iterations

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result["solver_info"] = {
            "status": self.solver_status,
            "solve_time": self.solve_time,
            "iterations": self.iterations
        }
        return result


class InfeasibleScheduleError(OptimizationError):
    """Exception raised when no feasible schedule can be found"""

    def __init__(
        self,
        message: str,
        conflicting_constraints: list[str] | None = None,
        suggestions: list[str] | None = None,
        details: dict[str, Any] | None = None
    ):
        super().__init__(message, "INFEASIBLE", details=details)
        self.conflicting_constraints = conflicting_constraints or []
        self.suggestions = suggestions or []

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result["conflicting_constraints"] = self.conflicting_constraints
        result["suggestions"] = self.suggestions
        return result


class ValidationError(SchedulerError):
    """Exception raised when validation fails"""

    def __init__(
        self,
        message: str,
        field: str | None = None,
        value: Any | None = None,
        validation_rules: list[str] | None = None,
        details: dict[str, Any] | None = None
    ):
        super().__init__(message, "VALIDATION_FAILED", details)
        self.field = field
        self.value = value
        self.validation_rules = validation_rules or []

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result["validation_info"] = {
            "field": self.field,
            "value": self.value,
            "validation_rules": self.validation_rules
        }
        return result


class ConfigurationError(SchedulerError):
    """Exception raised when configuration is invalid"""

    def __init__(
        self,
        message: str,
        config_key: str | None = None,
        expected_type: str | None = None,
        actual_value: Any | None = None,
        details: dict[str, Any] | None = None
    ):
        super().__init__(message, "CONFIGURATION_ERROR", details)
        self.config_key = config_key
        self.expected_type = expected_type
        self.actual_value = actual_value

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result["config_info"] = {
            "key": self.config_key,
            "expected_type": self.expected_type,
            "actual_value": self.actual_value
        }
        return result


class DataInconsistencyError(SchedulerError):
    """Exception raised when data is inconsistent"""

    def __init__(
        self,
        message: str,
        entity_type: str | None = None,
        entity_id: str | None = None,
        inconsistency_type: str | None = None,
        details: dict[str, Any] | None = None
    ):
        super().__init__(message, "DATA_INCONSISTENCY", details)
        self.entity_type = entity_type
        self.entity_id = entity_id
        self.inconsistency_type = inconsistency_type

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result["data_info"] = {
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "inconsistency_type": self.inconsistency_type
        }
        return result


class ResourceLimitError(SchedulerError):
    """Exception raised when resource limits are exceeded"""

    def __init__(
        self,
        message: str,
        resource_type: str | None = None,
        limit: Any | None = None,
        current_usage: Any | None = None,
        details: dict[str, Any] | None = None
    ):
        super().__init__(message, "RESOURCE_LIMIT_EXCEEDED", details)
        self.resource_type = resource_type
        self.limit = limit
        self.current_usage = current_usage

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result["resource_info"] = {
            "type": self.resource_type,
            "limit": self.limit,
            "current_usage": self.current_usage
        }
        return result


class SolverTimeoutError(OptimizationError):
    """Exception raised when solver times out"""

    def __init__(
        self,
        message: str,
        timeout_seconds: float | None = None,
        partial_solution: bool | None = None,
        details: dict[str, Any] | None = None
    ):
        super().__init__(message, "TIMEOUT", details=details)
        self.timeout_seconds = timeout_seconds
        self.partial_solution = partial_solution

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result["timeout_info"] = {
            "timeout_seconds": self.timeout_seconds,
            "partial_solution_available": self.partial_solution
        }
        return result


class CacheError(SchedulerError):
    """Exception raised when cache operations fail"""

    def __init__(
        self,
        message: str,
        operation: str | None = None,
        cache_key: str | None = None,
        backend: str | None = None,
        details: dict[str, Any] | None = None
    ):
        super().__init__(message, "CACHE_ERROR", details)
        self.operation = operation
        self.cache_key = cache_key
        self.backend = backend

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result["cache_info"] = {
            "operation": self.operation,
            "cache_key": self.cache_key,
            "backend": self.backend
        }
        return result


class QualificationError(SchedulerError):
    """Exception raised when qualification requirements are not met"""

    def __init__(
        self,
        message: str,
        staff_id: str | None = None,
        required_qualifications: list[str] | None = None,
        missing_qualifications: list[str] | None = None,
        details: dict[str, Any] | None = None
    ):
        super().__init__(message, "QUALIFICATION_ERROR", details)
        self.staff_id = staff_id
        self.required_qualifications = required_qualifications or []
        self.missing_qualifications = missing_qualifications or []

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result["qualification_info"] = {
            "staff_id": self.staff_id,
            "required": self.required_qualifications,
            "missing": self.missing_qualifications
        }
        return result


class AvailabilityError(SchedulerError):
    """Exception raised when staff availability conflicts occur"""

    def __init__(
        self,
        message: str,
        staff_id: str | None = None,
        requested_time: str | None = None,
        available_times: list[str] | None = None,
        details: dict[str, Any] | None = None
    ):
        super().__init__(message, "AVAILABILITY_ERROR", details)
        self.staff_id = staff_id
        self.requested_time = requested_time
        self.available_times = available_times or []

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result["availability_info"] = {
            "staff_id": self.staff_id,
            "requested_time": self.requested_time,
            "available_times": self.available_times
        }
        return result


class StaffingRatioError(SchedulerError):
    """Exception raised when staffing ratios are violated"""

    def __init__(
        self,
        message: str,
        group_id: str | None = None,
        required_ratio: str | None = None,
        current_ratio: str | None = None,
        child_count: int | None = None,
        staff_count: int | None = None,
        details: dict[str, Any] | None = None
    ):
        super().__init__(message, "STAFFING_RATIO_ERROR", details)
        self.group_id = group_id
        self.required_ratio = required_ratio
        self.current_ratio = current_ratio
        self.child_count = child_count
        self.staff_count = staff_count

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result["ratio_info"] = {
            "group_id": self.group_id,
            "required_ratio": self.required_ratio,
            "current_ratio": self.current_ratio,
            "child_count": self.child_count,
            "staff_count": self.staff_count
        }
        return result


class PriorityWeightError(SchedulerError):
    """Exception raised when priority weight configuration is invalid"""

    def __init__(
        self,
        message: str,
        staff_id: str | None = None,
        weight_type: str | None = None,
        weight_value: float | None = None,
        valid_range: tuple | None = None,
        details: dict[str, Any] | None = None
    ):
        super().__init__(message, "PRIORITY_WEIGHT_ERROR", details)
        self.staff_id = staff_id
        self.weight_type = weight_type
        self.weight_value = weight_value
        self.valid_range = valid_range

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result["weight_info"] = {
            "staff_id": self.staff_id,
            "weight_type": self.weight_type,
            "weight_value": self.weight_value,
            "valid_range": self.valid_range
        }
        return result


class APIError(SchedulerError):
    """Base exception for API-related errors"""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_code: str | None = None,
        details: dict[str, Any] | None = None
    ):
        super().__init__(message, error_code, details)
        self.status_code = status_code


class RateLimitError(APIError):
    """Exception raised when rate limits are exceeded"""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: int | None = None,
        details: dict[str, Any] | None = None
    ):
        super().__init__(message, 429, "RATE_LIMIT_EXCEEDED", details)
        self.retry_after = retry_after

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result["retry_after"] = self.retry_after
        return result


class AuthenticationError(APIError):
    """Exception raised when authentication fails"""

    def __init__(
        self,
        message: str = "Authentication failed",
        details: dict[str, Any] | None = None
    ):
        super().__init__(message, 401, "AUTHENTICATION_FAILED", details)


class AuthorizationError(APIError):
    """Exception raised when authorization fails"""

    def __init__(
        self,
        message: str = "Access denied",
        required_permission: str | None = None,
        details: dict[str, Any] | None = None
    ):
        super().__init__(message, 403, "ACCESS_DENIED", details)
        self.required_permission = required_permission

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result["required_permission"] = self.required_permission
        return result


class ServiceUnavailableError(APIError):
    """Exception raised when service is temporarily unavailable"""

    def __init__(
        self,
        message: str = "Service temporarily unavailable",
        maintenance_mode: bool = False,
        retry_after: int | None = None,
        details: dict[str, Any] | None = None
    ):
        super().__init__(message, 503, "SERVICE_UNAVAILABLE", details)
        self.maintenance_mode = maintenance_mode
        self.retry_after = retry_after

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result["maintenance_mode"] = self.maintenance_mode
        result["retry_after"] = self.retry_after
        return result


# Exception hierarchy for easy catching
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

ALL_SCHEDULER_EXCEPTIONS = (
    OPTIMIZATION_EXCEPTIONS +
    VALIDATION_EXCEPTIONS +
    API_EXCEPTIONS +
    SYSTEM_EXCEPTIONS
)


def create_error_response(exception: SchedulerError) -> dict[str, Any]:
    """Create standardized error response from exception"""
    response = exception.to_dict()
    response["timestamp"] = str(Exception.__new__(Exception).__init__)
    return response


def handle_optimization_exception(e: Exception) -> OptimizationError:
    """Convert generic optimization exceptions to OptimizationError"""
    if isinstance(e, OPTIMIZATION_EXCEPTIONS):
        return e

    # Map common OR-Tools exceptions
    error_message = str(e).lower()

    if "infeasible" in error_message:
        return InfeasibleScheduleError(
            "No feasible solution found",
            suggestions=[
                "Reduce staffing requirements",
                "Increase staff availability",
                "Adjust qualification requirements",
                "Review constraint configuration"
            ]
        )

    elif "timeout" in error_message or "time limit" in error_message:
        return SolverTimeoutError(
            "Optimization timed out",
            partial_solution=True
        )

    else:
        return OptimizationError(
            f"Optimization failed: {str(e)}",
            details={"original_exception": type(e).__name__}
        )
