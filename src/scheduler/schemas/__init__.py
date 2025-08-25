"""
API schemas for the Daycare Schedule Optimizer

This module contains Pydantic schemas for API requests and responses:
- Request schemas for validation
- Response schemas for consistent output
- Base schemas for common patterns
- Error schemas for exception handling
"""

# Import from main models for now - can be split later if needed
from ..models import (
    # Core data models
    Staff,
    Group,
    StaffingRequirement,
    ScheduledShift,
    TimeSlot,
    Qualification,
    StaffAvailability,
    StaffPreference,
    ScheduleConstraint,
    OptimizationConfig,
    
    # Request models
    ScheduleGenerationRequest,
    ScheduleValidationRequest,
    
    # Response models  
    ScheduleGenerationResponse,
    ScheduleValidationResponse,
    ScheduleConflict,
    OptimizationResult,
    
    # Enums
    StaffRole,
    AgeGroup,
    ShiftStatus,
    PreferenceType,
    PriorityLevel,
    OptimizationGoal
)

__all__ = [
    # Core data models
    "Staff",
    "Group", 
    "StaffingRequirement",
    "ScheduledShift",
    "TimeSlot",
    "Qualification",
    "StaffAvailability",
    "StaffPreference", 
    "ScheduleConstraint",
    "OptimizationConfig",
    
    # Request models
    "ScheduleGenerationRequest",
    "ScheduleValidationRequest",
    
    # Response models
    "ScheduleGenerationResponse", 
    "ScheduleValidationResponse",
    "ScheduleConflict",
    "OptimizationResult",
    
    # Enums
    "StaffRole",
    "AgeGroup",
    "ShiftStatus", 
    "PreferenceType",
    "PriorityLevel",
    "OptimizationGoal"
]

# Schema version for API compatibility
SCHEMA_VERSION = "1.0.0"

# Schema categories for organization
REQUEST_SCHEMAS = [
    ScheduleGenerationRequest,
    ScheduleValidationRequest
]

RESPONSE_SCHEMAS = [
    ScheduleGenerationResponse,
    ScheduleValidationResponse,
    ScheduleConflict,
    OptimizationResult  
]

DATA_SCHEMAS = [
    Staff,
    Group,
    StaffingRequirement,
    ScheduledShift,
    TimeSlot,
    Qualification,
    StaffAvailability,
    StaffPreference,
    ScheduleConstraint,
    OptimizationConfig
]

ENUM_SCHEMAS = [
    StaffRole,
    AgeGroup,
    ShiftStatus,
    PreferenceType,
    PriorityLevel,
    OptimizationGoal
]

def get_schema_by_name(schema_name: str):
    """Get a schema class by name"""
    schema_map = {cls.__name__: cls for cls in globals().values() if hasattr(cls, '__name__')}
    return schema_map.get(schema_name)

def list_schemas_by_category():
    """List all schemas organized by category"""
    return {
        "request": [cls.__name__ for cls in REQUEST_SCHEMAS],
        "response": [cls.__name__ for cls in RESPONSE_SCHEMAS], 
        "data": [cls.__name__ for cls in DATA_SCHEMAS],
        "enums": [cls.__name__ for cls in ENUM_SCHEMAS]
    }