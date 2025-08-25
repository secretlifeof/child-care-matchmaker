"""
Pydantic models for the Schedule Optimization Service
"""

from __future__ import annotations
from datetime import datetime, date, time, timedelta
from typing import List, Optional, Dict, Any, Union
from enum import Enum
from pydantic import BaseModel, Field, validator
from uuid import UUID


class StaffRole(str, Enum):
    TEACHER = "teacher"
    ASSISTANT = "assistant"
    SUPERVISOR = "supervisor"
    SUBSTITUTE = "substitute"
    ADMIN = "admin"
    STAFF = 'staff'


class AgeGroup(str, Enum):
    INFANT = "infant"  # 6 weeks - 18 months
    TODDLER = "toddler"  # 18 months - 3 years
    PRESCHOOL = "preschool"  # 3-5 years
    MIXED = "mixed"


class ShiftStatus(str, Enum):
    SCHEDULED = "scheduled"
    CONFIRMED = "confirmed"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ABSENT = "absent"
    CANCELLED = "cancelled"


class PreferenceType(str, Enum):
    PREFERRED_TIME = "preferred_time"
    UNAVAILABLE = "unavailable"
    MAX_HOURS = "max_hours"
    MIN_HOURS = "min_hours"
    PREFERRED_DAYS = "preferred_days"
    AVOID_DAYS = "avoid_days"


class PriorityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class OptimizationGoal(str, Enum):
    MINIMIZE_COST = "minimize_cost"
    MAXIMIZE_SATISFACTION = "maximize_satisfaction"
    MINIMIZE_OVERTIME = "minimize_overtime"
    MAXIMIZE_FAIRNESS = "maximize_fairness"
    MAXIMIZE_CONTINUITY = "maximize_continuity"
    MAXIMIZE_GROUP_CONTINUITY = "maximize_group_continuity"
    RESPECT_GROUP_ASSIGNMENTS = "respect_group_assignments"


class GroupAssignmentType(str, Enum):
    PRIMARY = "primary"
    SECONDARY = "secondary"
    SUBSTITUTE = "substitute"


class AbsenceType(str, Enum):
    SICK_LEAVE = "sick_leave"
    VACATION = "vacation"
    PERSONAL = "personal"
    TRAINING = "training"
    UNAVAILABLE = "unavailable"


class ShiftType(str, Enum):
    MORNING = "morning"
    AFTERNOON = "afternoon"
    EVENING = "evening"
    FULL_DAY = "full_day"
    SPLIT = "split"
    CUSTOM = "custom"


# Base Models
class TimeSlot(BaseModel):
    start_time: time
    end_time: time
    day_of_week: Optional[int] = Field(None, ge=0, le=6)  # 0=Monday, 6=Sunday, None=all days

    @validator("end_time")
    def end_after_start(cls, v, values):
        if "start_time" in values and v <= values["start_time"]:
            raise ValueError("end_time must be after start_time")
        return v
    
    @property
    def applies_to_all_days(self) -> bool:
        """Check if this time slot applies to all days"""
        return self.day_of_week is None
    
    @classmethod
    def daily_slot(cls, start_time: time, end_time: time) -> "TimeSlot":
        """Create a time slot that applies to all days"""
        return cls(start_time=start_time, end_time=end_time, day_of_week=None)
    
    @classmethod
    def weekly_slot(cls, day_of_week: int, start_time: time, end_time: time) -> "TimeSlot":
        """Create a time slot for a specific day of the week"""
        return cls(start_time=start_time, end_time=end_time, day_of_week=day_of_week)


# Enhanced Models for New Requirements
class CenterConfiguration(BaseModel):
    """Center-wide configuration including opening hours and ratios"""
    center_id: UUID
    name: str
    
    # Opening hours
    opening_hours: List[TimeSlot] = Field(
        description="Center opening hours for each day of the week"
    )
    
    # Staff-to-child ratios by age group (1 staff per X children)
    staff_child_ratios: Dict[AgeGroup, float] = Field(
        default={
            AgeGroup.INFANT: 4.0,
            AgeGroup.TODDLER: 6.0, 
            AgeGroup.PRESCHOOL: 10.0,
            AgeGroup.MIXED: 8.0
        },
        description="Staff-to-child ratios (1 staff per X children)"
    )
    
    # Overtime limits (center-wide)
    max_daily_overtime_hours: float = Field(2.0, ge=0, le=8)
    max_weekly_overtime_hours: float = Field(10.0, ge=0, le=20)
    overtime_threshold_daily: float = Field(8.0, gt=0, le=12)
    overtime_threshold_weekly: float = Field(40.0, gt=0, le=60)
    
    # Break requirements
    min_break_between_shifts_hours: float = Field(10.0, ge=0, le=24)
    
    @validator("staff_child_ratios")
    def validate_ratios(cls, v):
        for age_group, ratio in v.items():
            if ratio <= 0:
                raise ValueError(f"Ratio for {age_group} must be positive")
        return v
    
    @validator("max_daily_overtime_hours")
    def validate_daily_overtime(cls, v, values):
        if v < 0:
            raise ValueError("Daily overtime hours cannot be negative")
        if v > 8:
            raise ValueError("Daily overtime hours cannot exceed 8 hours")
        return v
    
    @validator("max_weekly_overtime_hours")
    def validate_weekly_overtime(cls, v, values):
        if v < 0:
            raise ValueError("Weekly overtime hours cannot be negative")
        if v > 20:
            raise ValueError("Weekly overtime hours cannot exceed 20 hours")
        return v


class Qualification(BaseModel):
    qualification_type: Optional[str] = None
    qualification_name: str
    issuing_organization: Optional[str] = None
    issue_date: Optional[date] = None
    expiry_date: Optional[date] = None
    is_verified: bool = False


class StaffAvailability(BaseModel):
    day_of_week: int = Field(..., ge=0, le=6)
    start_time: time
    end_time: time
    is_available: bool = True


class StaffPreference(BaseModel):
    preference_type: PreferenceType
    day_of_week: Optional[int] = Field(None, ge=0, le=6)
    time_range_start: Optional[time] = None
    time_range_end: Optional[time] = None
    max_hours_per_day: Optional[float] = None
    min_hours_per_day: Optional[float] = None
    weight: float = Field(1.0, ge=0.0, le=1.0)
    reason: Optional[str] = None


class StaffAbsence(BaseModel):
    """Staff absence model with date range and type"""
    absence_type: AbsenceType
    start_date: date
    end_date: date
    start_time: Optional[time] = None  # If None, full day absence
    end_time: Optional[time] = None    # If None, full day absence
    reason: Optional[str] = None
    is_approved: bool = True
    
    @validator("end_date")
    def end_after_start(cls, v, values):
        if "start_date" in values and v < values["start_date"]:
            raise ValueError("end_date must be after or equal to start_date")
        return v
    
    @validator("end_time")
    def validate_time_range(cls, v, values):
        if v is not None and "start_time" in values and values["start_time"] is not None:
            if v <= values["start_time"]:
                raise ValueError("end_time must be after start_time")
        return v
    
    @property
    def is_full_day(self) -> bool:
        """Check if this is a full day absence"""
        return self.start_time is None or self.end_time is None


class GroupAssignment(BaseModel):
    """Staff assignment to a group with primary/secondary designation"""
    group_id: UUID
    assignment_type: GroupAssignmentType
    priority_weight: float = Field(1.0, ge=0.0, le=2.0)
    notes: Optional[str] = None


class ShiftTemplate(BaseModel):
    """Work shift template that can be assigned to groups"""
    shift_template_id: UUID
    name: str
    shift_type: ShiftType
    start_time: time
    end_time: time
    break_duration_minutes: Optional[int] = Field(None, ge=0, le=120)
    required_qualifications: List[str] = Field(default_factory=list)
    is_active: bool = True
    
    @validator("end_time")
    def validate_time_range(cls, v, values):
        if "start_time" in values:
            start_dt = datetime.combine(date.today(), values["start_time"])
            end_dt = datetime.combine(date.today(), v)
            if end_dt <= start_dt:
                # Allow overnight shifts
                end_dt += timedelta(days=1)
            duration_hours = (end_dt - start_dt).total_seconds() / 3600
            if duration_hours > 16:
                raise ValueError("Shift duration cannot exceed 16 hours")
        return v
    
    @property
    def duration_hours(self) -> float:
        """Calculate shift duration in hours"""
        start_dt = datetime.combine(date.today(), self.start_time)
        end_dt = datetime.combine(date.today(), self.end_time)
        if end_dt <= start_dt:
            end_dt += timedelta(days=1)  # Handle overnight shifts
        duration = (end_dt - start_dt).total_seconds() / 3600
        if self.break_duration_minutes:
            duration -= self.break_duration_minutes / 60
        return duration


class ShiftTemplateRequirement(BaseModel):
    """Defines how many shifts of a specific template a group needs per day"""
    shift_template_id: UUID
    group_id: UUID
    required_count: int = Field(1, ge=0)
    preferred_count: Optional[int] = Field(None, ge=0)
    day_of_week: Optional[int] = Field(None, ge=0, le=6)  # None means all days
    
    @validator("preferred_count")
    def validate_preferred_count(cls, v, values):
        if v is not None and "required_count" in values and v < values["required_count"]:
            raise ValueError("preferred_count cannot be less than required_count")
        return v


class Staff(BaseModel):
    staff_id: UUID
    name: str
    role: StaffRole
    qualifications: List[Qualification] = []
    availability: List[StaffAvailability] = []
    preferences: List[StaffPreference] = []
    seniority_weight: float = Field(1.0, ge=0.0, le=2.0)
    performance_weight: float = Field(1.0, ge=0.0, le=2.0)
    flexibility_score: float = Field(1.0, ge=0.0, le=2.0)
    overall_priority: PriorityLevel = PriorityLevel.MEDIUM
    max_weekly_hours: float = Field(40.0, gt=0, le=60)
    hourly_rate: Optional[float] = Field(None, gt=0)
    overtime_rate: Optional[float] = Field(None, gt=0)

    # Shift length constraints (optional, overrides center/group defaults)
    min_shift_hours: Optional[float] = Field(None, gt=0, le=12)
    max_shift_hours: Optional[float] = Field(None, gt=0, le=16)
    
    # Enhanced fields for new requirements
    absences: List[StaffAbsence] = Field(default_factory=list)
    group_assignments: List[GroupAssignment] = Field(default_factory=list)
    max_daily_hours: Optional[float] = Field(None, gt=0, le=16)
    
    @validator("overtime_rate")
    def overtime_rate_validation(cls, v, values):
        if v is not None and "hourly_rate" in values and values["hourly_rate"] is not None:
            if v < values["hourly_rate"]:
                raise ValueError("overtime_rate should be greater than or equal to hourly_rate")
        return v
    
    @validator("group_assignments")
    def validate_primary_assignments(cls, v):
        """Ensure staff can only be primary to one group"""
        primary_assignments = [a for a in v if a.assignment_type == GroupAssignmentType.PRIMARY]
        if len(primary_assignments) > 1:
            raise ValueError("Staff can only be assigned as primary to one group")
        return v

    @property
    def priority_score(self) -> float:
        """Calculate overall priority score for preferences and satisfaction"""
        base_scores = {
            PriorityLevel.LOW: 0.5,
            PriorityLevel.MEDIUM: 1.0,
            PriorityLevel.HIGH: 1.5,
            PriorityLevel.CRITICAL: 2.0,
        }
        return (
            base_scores[self.overall_priority]
            * self.seniority_weight
            * self.performance_weight
        )


class Group(BaseModel):
    group_id: UUID
    name: str
    age_group: AgeGroup
    capacity: int = Field(..., gt=0)
    current_enrollment: int = Field(0, ge=0)
    required_qualifications: List[str] = []
    preferred_qualifications: List[str] = []

    # Group-specific shift length constraints (optional, overrides center defaults)
    min_shift_hours: Optional[float] = Field(None, gt=0, le=12)
    max_shift_hours: Optional[float] = Field(None, gt=0, le=16)
    
    # Shift templates assigned to this group
    shift_template_ids: List[UUID] = Field(default_factory=list)

    @validator("current_enrollment")
    def enrollment_not_exceed_capacity(cls, v, values):
        if "capacity" in values and v > values["capacity"]:
            raise ValueError("current_enrollment cannot exceed capacity")
        return v


class StaffingRequirement(BaseModel):
    group_id: UUID
    time_slot: TimeSlot
    min_staff_count: int = Field(1, ge=1)
    max_staff_count: Optional[int] = None
    required_qualifications: List[str] = []
    preferred_qualifications: List[str] = []

    @validator("max_staff_count")
    def max_not_less_than_min(cls, v, values):
        if (
            v is not None
            and "min_staff_count" in values
            and v < values["min_staff_count"]
        ):
            raise ValueError("max_staff_count cannot be less than min_staff_count")
        return v


class ScheduleConstraint(BaseModel):
    constraint_type: str
    is_mandatory: bool = True
    weight: float = Field(1.0, ge=0.0, le=1.0)
    config: Dict[str, Any] = {}


class OptimizationConfig(BaseModel):
    goals: List[OptimizationGoal] = [OptimizationGoal.MINIMIZE_COST]
    max_solver_time: int = Field(60, gt=0, le=300)  # seconds
    consider_preferences: bool = True
    minimize_overtime: bool = True
    ensure_fairness: bool = True
    max_consecutive_hours: int = Field(8, gt=0, le=12)
    min_break_between_shifts: int = Field(10, ge=0, le=24)  # hours

    # Priority-based hour allocation (default: equal distribution)
    enable_priority_based_hours: bool = Field(
        False,
        description="If True, higher priority staff get more hours. If False, hours are distributed equally.",
    )

    # Fairness strictness (0.0 = no fairness enforcement, 1.0 = strict equal distribution)
    fairness_strictness: float = Field(
        0.8,
        ge=0.0,
        le=1.0,
        description="How strictly to enforce equal hour distribution",
    )


# Request Models
class ScheduleGenerationRequest(BaseModel):
    """Legacy weekly schedule request - kept for backward compatibility"""
    center_id: UUID
    week_start_date: date
    staff: List[Staff]
    groups: List[Group]
    staffing_requirements: List[StaffingRequirement]
    constraints: List[ScheduleConstraint] = []
    optimization_config: OptimizationConfig = OptimizationConfig()
    existing_schedules: Optional[List["ScheduledShift"]] = None

    # Extra shift constraints (HARD CONSTRAINT)
    extra_shift_eligible_staff_ids: List[UUID] = Field(
        default=[],
        description="List of staff IDs who can be assigned extra shifts beyond normal allocation. If empty, no extra shifts are allowed.",
    )

    # Center-level shift length constraints (can be overridden by group or staff)
    default_min_shift_hours: float = Field(
        2.0, gt=0, le=12, description="Default minimum shift length in hours"
    )
    default_max_shift_hours: float = Field(
        10.0, gt=0, le=16, description="Default maximum shift length in hours"
    )

    @validator("week_start_date")
    def week_start_is_monday(cls, v):
        if v.weekday() != 0:  # 0 = Monday
            raise ValueError("week_start_date must be a Monday")
        return v

    @validator("default_max_shift_hours")
    def max_shift_greater_than_min(cls, v, values):
        if (
            "default_min_shift_hours" in values
            and v <= values["default_min_shift_hours"]
        ):
            raise ValueError(
                "default_max_shift_hours must be greater than default_min_shift_hours"
            )
        return v


class EnhancedScheduleGenerationRequest(BaseModel):
    """Enhanced request model with flexible date range and new features"""
    center_id: UUID
    schedule_start_date: Union[date, str] = Field(description="Start date of the scheduling period (date or ISO 8601 datetime string)")
    schedule_end_date: Optional[Union[date, str]] = Field(
        default=None,
        description="End date of the scheduling period (inclusive). If not provided, defaults to 7 days from start. (date or ISO 8601 datetime string)",
    )
    
    # Core scheduling data
    staff: List[Staff]
    groups: List[Group]
    staffing_requirements: List[StaffingRequirement]
    center_config: CenterConfiguration
    
    # Shift templates
    shift_templates: List[ShiftTemplate] = Field(default_factory=list)
    shift_template_requirements: List[ShiftTemplateRequirement] = Field(default_factory=list)
    
    # Optional configurations
    constraints: List[ScheduleConstraint] = Field(default_factory=list)
    optimization_config: OptimizationConfig = Field(default_factory=OptimizationConfig)
    existing_schedules: Optional[List["ScheduledShift"]] = None
    
    # Extra shift constraints (HARD CONSTRAINT)
    extra_shift_eligible_staff_ids: List[UUID] = Field(
        default_factory=list,
        description="List of staff IDs who can be assigned extra shifts beyond normal allocation. If empty, no extra shifts are allowed.",
    )
    
    @validator("schedule_start_date", pre=True)
    def parse_start_date(cls, v):
        """Parse start date from string (ISO 8601 datetime) or return date as-is"""
        if isinstance(v, str):
            try:
                # Parse ISO 8601 datetime string and extract date
                parsed_dt = datetime.fromisoformat(v.replace('Z', '+00:00'))
                return parsed_dt.date()
            except ValueError:
                try:
                    # Try parsing as date only
                    return datetime.strptime(v, '%Y-%m-%d').date()
                except ValueError:
                    raise ValueError(f"Invalid date format: {v}. Expected date or ISO 8601 datetime string.")
        elif isinstance(v, datetime):
            return v.date()
        elif isinstance(v, date):
            return v
        else:
            raise ValueError(f"Invalid date type: {type(v)}. Expected date, datetime, or string.")
    
    @validator("schedule_end_date", pre=True)
    def parse_end_date(cls, v):
        """Parse end date from string (ISO 8601 datetime) or return date as-is"""
        if v is None:
            return v
        if isinstance(v, str):
            try:
                # Parse ISO 8601 datetime string and extract date
                parsed_dt = datetime.fromisoformat(v.replace('Z', '+00:00'))
                return parsed_dt.date()
            except ValueError:
                try:
                    # Try parsing as date only
                    return datetime.strptime(v, '%Y-%m-%d').date()
                except ValueError:
                    raise ValueError(f"Invalid date format: {v}. Expected date or ISO 8601 datetime string.")
        elif isinstance(v, datetime):
            return v.date()
        elif isinstance(v, date):
            return v
        else:
            raise ValueError(f"Invalid date type: {type(v)}. Expected date, datetime, or string.")
    
    @validator("schedule_end_date")
    def validate_end_date(cls, v, values):
        if v is not None and "schedule_start_date" in values:
            start_date = values["schedule_start_date"]
            if v < start_date:
                raise ValueError("End date must be after start date")

            total_days = (v - start_date).days + 1
            if total_days > 365:
                raise ValueError(
                    f"Date range too large: {total_days} days (maximum 365)"
                )
            if total_days == 0:
                raise ValueError("Date range must be at least 1 day")
        return v
    
    @validator("staff")
    def validate_staff_list(cls, v):
        if not v:
            raise ValueError("At least one staff member is required")
        if len(v) > 100:
            raise ValueError("Too many staff members (maximum 100)")
        return v
    
    @validator("groups")
    def validate_groups_list(cls, v):
        if not v:
            raise ValueError("At least one group is required")
        if len(v) > 20:
            raise ValueError("Too many groups (maximum 20)")
        return v
    
    @property
    def total_days(self) -> int:
        """Calculate total days in the scheduling period"""
        end_date = self.schedule_end_date or (
            self.schedule_start_date + timedelta(days=6)
        )
        return (end_date - self.schedule_start_date).days + 1

    @property
    def effective_end_date(self) -> date:
        """Get the effective end date (computed if not provided)"""
        return self.schedule_end_date or (self.schedule_start_date + timedelta(days=6))
    
    def get_staff_availability_for_date(self, staff_id: UUID, target_date: date) -> List[StaffAvailability]:
        """Get staff availability for a specific date, considering absences and preferences"""
        staff_member = next((s for s in self.staff if s.staff_id == staff_id), None)
        if not staff_member:
            return []
        
        day_of_week = target_date.weekday()
        
        # Check for absences (highest priority - hard constraint)
        for absence in staff_member.absences:
            if absence.start_date <= target_date <= absence.end_date:
                if absence.is_full_day:
                    return []  # No availability on this day
                # TODO: Handle partial day absences
        
        # Get preferences or fallback to center opening hours
        available_times = []
        
        # Check if staff has any availability preferences for this day
        staff_availability = [av for av in staff_member.availability if av.day_of_week == day_of_week and av.is_available]
        
        if staff_availability:
            # Use staff preferences
            available_times = staff_availability
        else:
            # Use center opening hours as fallback
            center_opening = [oh for oh in self.center_config.opening_hours if oh.day_of_week == day_of_week]
            if center_opening:
                # Convert TimeSlot to StaffAvailability
                for opening in center_opening:
                    available_times.append(StaffAvailability(
                        day_of_week=day_of_week,
                        start_time=opening.start_time,
                        end_time=opening.end_time,
                        is_available=True
                    ))
        
        return available_times


# Response Models
class ScheduledShift(BaseModel):
    schedule_id: Optional[UUID] = None
    staff_id: UUID
    group_id: UUID
    shift_template_id: Optional[UUID] = None
    date: Union[date, str]
    start_time: Union[time, str]
    end_time: Union[time, str]
    scheduled_hours: float
    status: ShiftStatus = ShiftStatus.SCHEDULED
    notes: Optional[str] = None
    is_overtime: bool = False
    assignment_type: Optional[GroupAssignmentType] = None
    timestamptz: Optional[str] = Field(None, description="ISO 8601 datetime string combining date and start_time")

    @validator("date", pre=True)
    def parse_shift_date(cls, v):
        """Parse shift date from string (ISO 8601 datetime) or return date as-is"""
        if isinstance(v, str):
            try:
                # Parse ISO 8601 datetime string and extract date
                parsed_dt = datetime.fromisoformat(v.replace('Z', '+00:00'))
                return parsed_dt.date()
            except ValueError:
                try:
                    # Try parsing as date only
                    return datetime.strptime(v, '%Y-%m-%d').date()
                except ValueError:
                    raise ValueError(f"Invalid date format: {v}. Expected date or ISO 8601 datetime string.") from None
        elif isinstance(v, datetime):
            return v.date()
        elif isinstance(v, date):
            return v
        else:
            raise ValueError(f"Invalid date type: {type(v)}. Expected date, datetime, or string.")

    @validator("start_time", pre=True)
    def parse_start_time(cls, v):
        """Parse start time from string or return time as-is"""
        if isinstance(v, str):
            try:
                # Try parsing as time only first
                return datetime.strptime(v, '%H:%M:%S').time()
            except ValueError:
                try:
                    # Try parsing as ISO 8601 datetime string and extract time
                    parsed_dt = datetime.fromisoformat(v.replace('Z', '+00:00'))
                    return parsed_dt.time()
                except ValueError:
                    raise ValueError(f"Invalid time format: {v}. Expected time or ISO 8601 datetime string.") from None
        elif isinstance(v, datetime):
            return v.time()
        elif isinstance(v, time):
            return v
        else:
            raise ValueError(f"Invalid time type: {type(v)}. Expected time, datetime, or string.")

    @validator("end_time", pre=True)
    def parse_end_time(cls, v):
        """Parse end time from string or return time as-is"""
        if isinstance(v, str):
            try:
                # Try parsing as time only first
                return datetime.strptime(v, '%H:%M:%S').time()
            except ValueError:
                try:
                    # Try parsing as ISO 8601 datetime string and extract time
                    parsed_dt = datetime.fromisoformat(v.replace('Z', '+00:00'))
                    return parsed_dt.time()
                except ValueError:
                    raise ValueError(f"Invalid time format: {v}. Expected time or ISO 8601 datetime string.") from None
        elif isinstance(v, datetime):
            return v.time()
        elif isinstance(v, time):
            return v
        else:
            raise ValueError(f"Invalid time type: {type(v)}. Expected time, datetime, or string.")

    @validator("timestamptz", always=True)
    def generate_timestamptz(cls, v, values):
        """Generate ISO 8601 datetime string from date and start_time"""
        if "date" in values and "start_time" in values:
            date_val = values["date"]
            time_val = values["start_time"]
            if isinstance(date_val, date) and isinstance(time_val, time):
                combined_dt = datetime.combine(date_val, time_val)
                return combined_dt.isoformat() + "Z"
        return v

    @property
    def duration_hours(self) -> float:
        """Calculate shift duration in hours"""
        start_dt = datetime.combine(self.date, self.start_time)
        end_dt = datetime.combine(self.date, self.end_time)
        if end_dt <= start_dt:
            end_dt += timedelta(days=1)  # Handle overnight shifts
        return (end_dt - start_dt).total_seconds() / 3600


class ScheduleConflict(BaseModel):
    conflict_type: str
    severity: str  # 'error', 'warning', 'info'
    group_id: Optional[UUID] = None
    staff_id: Optional[UUID] = None
    time_slot: Optional[TimeSlot] = None
    description: str
    suggested_solutions: List[str] = []


class OptimizationResult(BaseModel):
    objective_value: float
    solve_time_seconds: float
    status: str  # 'OPTIMAL', 'FEASIBLE', 'INFEASIBLE', 'TIMEOUT'
    iterations: int = 0
    conflicts_resolved: int = 0


class ScheduleGenerationResponse(BaseModel):
    success: bool
    schedules: List[ScheduledShift] = []  # All schedules (existing + new)
    new_schedules: List[ScheduledShift] = []  # Only newly generated schedules
    conflicts: List[ScheduleConflict] = []
    optimization_result: OptimizationResult
    total_cost: Optional[float] = None
    total_hours: float = 0
    staff_utilization: Dict[UUID, float] = {}
    satisfaction_score: float = 0.0
    message: str = ""


class ScheduleValidationRequest(BaseModel):
    schedule: List[ScheduledShift]
    staff: List[Staff]
    groups: List[Group]
    staffing_requirements: List[StaffingRequirement]
    constraints: List[ScheduleConstraint] = []


class ScheduleValidationResponse(BaseModel):
    is_valid: bool
    conflicts: List[ScheduleConflict] = []
    warnings: List[str] = []
    suggestions: List[str] = []




class EnhancedScheduleResponse(BaseModel):
    """Enhanced response with date range information"""

    # Response fields
    success: bool
    schedules: List[ScheduledShift] = []  # All schedules (existing + new) 
    new_schedules: List[ScheduledShift] = []  # Only newly generated schedules
    conflicts: List[ScheduleConflict] = []
    optimization_result: OptimizationResult
    total_cost: float = 0.0
    total_hours: float = 0.0
    staff_utilization: Dict[UUID, float] = {}
    satisfaction_score: float = 0.0
    message: str = ""

    # Enhanced fields for date range support
    schedule_start_date: Optional[date] = None
    schedule_end_date: Optional[date] = None
    total_days: Optional[int] = None

    # Additional metrics for longer periods
    daily_averages: Dict[str, float] = {}
    period_coverage: float = 0.0
    
    # Schedule counts
    total_schedules_count: int = 0
    new_schedules_count: int = 0
    existing_schedules_count: int = 0

    @property
    def period_description(self) -> str:
        """Get a human-readable description of the scheduling period"""
        if self.schedule_start_date and self.schedule_end_date:
            if self.total_days == 1:
                return f"Single day: {self.schedule_start_date}"
            elif self.total_days == 7:
                return f"Week: {self.schedule_start_date} to {self.schedule_end_date}"
            else:
                return f"{self.total_days} days: {self.schedule_start_date} to {self.schedule_end_date}"
        return "Unknown period"


class PerformanceMetrics(BaseModel):
    """Performance metrics for optimization tracking"""

    total_time_seconds: float
    solver_time_seconds: float
    preprocessing_time_seconds: float
    postprocessing_time_seconds: float

    total_variables: int
    total_constraints: int
    memory_usage_mb: float

    problem_size_score: float
    optimization_efficiency: float

    chunk_count: int = 1
    parallel_chunks: int = 1
    cache_hits: int = 0


# Enhanced optimization configuration
class EnhancedOptimizationConfig(OptimizationConfig):
    """Extended optimization config with period-aware settings"""

    # Period-specific settings
    max_consecutive_days: Optional[int] = Field(
        default=None, description="Maximum consecutive working days for staff"
    )
    min_days_off_per_week: Optional[int] = Field(
        default=None, description="Minimum days off per week for staff"
    )
    balance_workload_over_period: bool = Field(
        default=True,
        description="Whether to balance workload evenly across the entire period",
    )

    # Overtime settings for longer periods
    overtime_threshold_per_week: float = Field(
        default=40.0, description="Hours per week before overtime applies"
    )
    max_overtime_per_week: float = Field(
        default=10.0, description="Maximum overtime hours per week"
    )


# Helper functions for creating common request types


def create_weekly_request(
    center_id: UUID,
    week_start: date,
    staff: List[Staff],
    groups: List[Group],
    requirements: List[StaffingRequirement],
    center_config: CenterConfiguration,
    existing_config: Optional[OptimizationConfig] = None,
) -> EnhancedScheduleGenerationRequest:
    """Helper function to create a standard weekly schedule request"""

    config = existing_config or EnhancedOptimizationConfig(
        goals=[
            OptimizationGoal.MAXIMIZE_SATISFACTION,
            OptimizationGoal.MINIMIZE_COST,
            OptimizationGoal.MAXIMIZE_FAIRNESS,
        ]
    )

    return EnhancedScheduleGenerationRequest(
        center_id=center_id,
        schedule_start_date=week_start,
        schedule_end_date=week_start + timedelta(days=6),
        staff=staff,
        groups=groups,
        staffing_requirements=requirements,
        center_config=center_config,
        optimization_config=config,
    )


def create_monthly_request(
    center_id: UUID,
    month_start: date,
    staff: List[Staff],
    groups: List[Group],
    requirements: List[StaffingRequirement],
    center_config: CenterConfiguration,
) -> EnhancedScheduleGenerationRequest:
    """Helper function to create a monthly schedule request"""

    # Calculate month end (approximately)
    if month_start.month == 12:
        next_month = month_start.replace(year=month_start.year + 1, month=1)
    else:
        next_month = month_start.replace(month=month_start.month + 1)

    month_end = next_month - timedelta(days=1)

    return EnhancedScheduleGenerationRequest(
        center_id=center_id,
        schedule_start_date=month_start,
        schedule_end_date=month_end,
        staff=staff,
        groups=groups,
        staffing_requirements=requirements,
        center_config=center_config,
        optimization_config=EnhancedOptimizationConfig(
            goals=[
                OptimizationGoal.MAXIMIZE_FAIRNESS,
                OptimizationGoal.MAXIMIZE_SATISFACTION,
                OptimizationGoal.MINIMIZE_OVERTIME,
            ],
            max_solver_time=600,  # Longer time for monthly optimization
            balance_workload_over_period=True,
            max_consecutive_days=5,
            min_days_off_per_week=2,
        ),
    )


# Update forward references
ScheduleGenerationRequest.model_rebuild()
