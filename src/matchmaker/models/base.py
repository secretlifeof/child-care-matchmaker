"""Base models for the matchmaker service."""

from datetime import date, datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from uuid import UUID
from pydantic import BaseModel, Field, ConfigDict


class MatchMode(str, Enum):
    """Matching operation modes."""
    RECOMMEND = "recommend"  # Parent-centric recommendations
    ALLOCATE = "allocate"    # Global optimal allocation
    WAITLIST = "waitlist"    # Center-centric waitlist ranking


class ConstraintType(str, Enum):
    """Types of matching constraints."""
    MUST_HAVE = "must_have"
    NICE_TO_HAVE = "nice_to_have"
    EXCLUDE = "exclude"


class PreferenceStrength(str, Enum):
    """Preference strength levels for better LLM extraction."""
    REQUIRED = "required"        # Must have (dealbreaker if missing)
    PREFERRED = "preferred"      # Strong preference but flexible  
    NICE_TO_HAVE = "nice_to_have" # Would be nice but not important
    AVOID = "avoid"             # Must NOT have (dealbreaker if present)


class PropertyCategory(str, Enum):
    """Categories of center properties."""
    FACILITY = "facility"
    CERTIFICATE = "certificate"
    APPROACH = "approach"
    AWARD = "award"
    SERVICE = "service"
    DIETARY = "dietary"
    PROGRAM = "program"


class PropertyType(str, Enum):
    """Types of property values - renamed for clarity."""
    TEXT = "text"
    BOOLEAN = "boolean"
    NUMBER = "number"
    DATE = "date"
    LIST = "list"


# Backward compatibility alias
ValueType = PropertyType


class SourceType(str, Enum):
    """Source of property data."""
    MANUAL = "manual"
    AI_EXTRACTED = "ai_extracted"
    VERIFIED = "verified"


class ComparisonOperator(str, Enum):
    """Operators for preference matching."""
    EQUALS = "equals"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    BETWEEN = "between"
    WITHIN_DISTANCE = "within_distance"


class Location(BaseModel):
    """Geographic location."""
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    address: Optional[str] = None
    postal_code: Optional[str] = None
    city: Optional[str] = None
    country_code: str = Field(default="DE", pattern="^[A-Z]{2}$")


class AgeGroup(BaseModel):
    """Age group definition."""
    min_age_months: int = Field(..., ge=0)
    max_age_months: int = Field(..., ge=0)
    name: str
    
    def contains_age(self, age_months: int) -> bool:
        """Check if age falls within this group."""
        return self.min_age_months <= age_months <= self.max_age_months


class TimeSlot(BaseModel):
    """Time slot for availability using minutes from start of day or time strings."""
    day_of_week: int = Field(..., ge=0, le=6)  # 0=Monday, 6=Sunday
    start_minute: Optional[int] = Field(None, ge=0, le=1439)  # 0-1439 (24*60-1)
    end_minute: Optional[int] = Field(None, ge=0, le=1440)    # 0-1440 (24*60)
    start: Optional[str] = Field(None, pattern=r"^([01]?[0-9]|2[0-3]):[0-5][0-9]$")  # HH:MM format
    end: Optional[str] = Field(None, pattern=r"^([01]?[0-9]|2[0-3]):[0-5][0-9]$")    # HH:MM format
    
    def __init__(self, **data):
        # Convert time strings to minutes if provided
        if 'start' in data and data['start'] is not None:
            if 'start_minute' not in data:
                data['start_minute'] = self._time_string_to_minutes(data['start'])
        if 'end' in data and data['end'] is not None:
            if 'end_minute' not in data:
                data['end_minute'] = self._time_string_to_minutes(data['end'])
            
        # Ensure we have the required minute fields after conversion
        if data.get('start_minute') is None:
            raise ValueError("TimeSlot requires start_minute or start field")
        if data.get('end_minute') is None:
            raise ValueError("TimeSlot requires end_minute or end field")
            
        super().__init__(**data)
    
    @staticmethod
    def _time_string_to_minutes(time_str: str) -> int:
        """Convert HH:MM string to minutes from start of day."""
        hours, minutes = map(int, time_str.split(':'))
        return hours * 60 + minutes
    
    @property
    def start_hour(self) -> int:
        """Get start hour for backward compatibility."""
        return self.start_minute // 60
    
    @property 
    def start_minute_in_hour(self) -> int:
        """Get minute within the hour."""
        return self.start_minute % 60
        
    @property
    def end_hour(self) -> int:
        """Get end hour for backward compatibility."""
        return self.end_minute // 60
    
    @property
    def end_minute_in_hour(self) -> int:
        """Get minute within the hour."""
        return self.end_minute % 60
    
    @property
    def start_time_string(self) -> str:
        """Get start time as HH:MM string."""
        hours, minutes = divmod(self.start_minute, 60)
        return f"{hours:02d}:{minutes:02d}"
    
    @property
    def end_time_string(self) -> str:
        """Get end time as HH:MM string."""
        hours, minutes = divmod(self.end_minute, 60)
        return f"{hours:02d}:{minutes:02d}"
    
    @classmethod
    def from_time_strings(cls, day_of_week: int, start: str, end: str):
        """Create TimeSlot from HH:MM time strings."""
        return cls(
            day_of_week=day_of_week,
            start=start,
            end=end
        )
    
    @classmethod
    def from_hours(cls, day_of_week: int, start_hour: int, end_hour: int, 
                   start_min: int = 0, end_min: int = 0):
        """Create TimeSlot from hour format for backward compatibility."""
        return cls(
            day_of_week=day_of_week,
            start_minute=start_hour * 60 + start_min,
            end_minute=end_hour * 60 + end_min
        )
    
    def overlaps_with(self, other: "TimeSlot") -> bool:
        """Check if time slots overlap."""
        if self.day_of_week != other.day_of_week:
            return False
        return not (self.end_minute <= other.start_minute or other.end_minute <= self.start_minute)
    
    def to_time_string(self) -> str:
        """Convert to readable time string."""
        start_h, start_m = divmod(self.start_minute, 60)
        end_h, end_m = divmod(self.end_minute, 60)
        return f"{start_h:02d}:{start_m:02d}-{end_h:02d}:{end_m:02d}"


class CapacityBucket(BaseModel):
    """Capacity bucket for a specific center group."""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    center_id: UUID
    group_id: Optional[UUID] = None
    age_band: AgeGroup
    start_month: date
    total_capacity: int = Field(..., ge=0)
    available_capacity: int = Field(..., ge=0)
    reserved_label: Optional[str] = None  # e.g., "sibling", "municipality"
    
    @property
    def is_available(self) -> bool:
        """Check if bucket has available capacity."""
        return self.available_capacity > 0


class CenterProperty(BaseModel):
    """Property of a daycare center."""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    center_id: UUID
    property_type_id: UUID
    property_key: str  # e.g., 'cert_ofsted', 'facility_gym'
    category: PropertyCategory
    value_text: Optional[str] = None
    value_numeric: Optional[float] = None
    value_boolean: Optional[bool] = None
    value_date: Optional[date] = None
    value_list: Optional[List[str]] = None
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0)
    source: SourceType = SourceType.MANUAL
    
    def get_value(self) -> Any:
        """Get the actual value regardless of type."""
        if self.value_boolean is not None:
            return self.value_boolean
        if self.value_numeric is not None:
            return self.value_numeric
        if self.value_date is not None:
            return self.value_date
        if self.value_list is not None:
            return self.value_list
        return self.value_text


class Center(BaseModel):
    """Daycare center."""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    name: str
    location: Location
    short_description: Optional[str] = None
    long_description: Optional[str] = None
    country_code: str = Field(default="DE")
    opening_hours: List[TimeSlot] = Field(default_factory=list)
    properties: List[CenterProperty] = Field(default_factory=list)
    capacity_buckets: List[CapacityBucket] = Field(default_factory=list)
    
    def has_property(self, property_key: str) -> bool:
        """Check if center has a specific property."""
        return any(p.property_key == property_key for p in self.properties)
    
    def get_property(self, property_key: str) -> Optional[CenterProperty]:
        """Get a specific property by key."""
        return next((p for p in self.properties if p.property_key == property_key), None)


class ParentPreference(BaseModel):
    """Parent's preference for matching with categorical strength system."""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    profile_id: UUID
    property_type_id: Optional[UUID] = None
    property_key: str
    property_type: PropertyType
    operator: ComparisonOperator
    
    # Preference strength (replaces weight/threshold)
    strength: PreferenceStrength = PreferenceStrength.PREFERRED
    
    # Values to match against
    value_text: Optional[str] = None
    value_numeric: Optional[float] = None
    value_boolean: Optional[bool] = None
    value_date: Optional[date] = None
    value_list: Optional[List[str]] = None
    
    # For numeric ranges
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    
    # Backward compatibility (deprecated)
    weight: float = Field(default=0.5, ge=0.0, le=1.0)
    threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    
    @property
    def is_absolute(self) -> bool:
        """Check if this is an absolute requirement."""
        return self.strength in [PreferenceStrength.REQUIRED, PreferenceStrength.AVOID]
    
    @property
    def computed_weight(self) -> float:
        """Convert strength to numeric weight."""
        strength_weights = {
            PreferenceStrength.REQUIRED: 1.0,
            PreferenceStrength.PREFERRED: 0.8,
            PreferenceStrength.NICE_TO_HAVE: 0.5,
            PreferenceStrength.AVOID: -1.0
        }
        return strength_weights[self.strength]
    
    @property
    def constraint_type(self) -> ConstraintType:
        """Derive constraint type from strength."""
        if self.strength == PreferenceStrength.REQUIRED:
            return ConstraintType.MUST_HAVE
        elif self.strength == PreferenceStrength.AVOID:
            return ConstraintType.EXCLUDE
        return ConstraintType.NICE_TO_HAVE
    
    def get_value(self) -> Any:
        """Get the actual value regardless of type."""
        if self.value_boolean is not None:
            return self.value_boolean
        if self.value_numeric is not None:
            return self.value_numeric
        if self.value_date is not None:
            return self.value_date
        if self.value_list is not None:
            return self.value_list
        return self.value_text


class Child(BaseModel):
    """Child information."""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    family_id: UUID
    name: str
    birth_date: date
    special_needs: Optional[Dict[str, Any]] = None
    
    @property
    def age_months(self) -> int:
        """Calculate age in months."""
        today = date.today()
        months = (today.year - self.birth_date.year) * 12
        months += today.month - self.birth_date.month
        if today.day < self.birth_date.day:
            months -= 1
        return max(0, months)


class Application(BaseModel):
    """Application for daycare placement."""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    family_id: UUID
    children: List[Child]
    home_location: Location
    preferences: List[ParentPreference]
    desired_start_date: date
    desired_hours: List[TimeSlot]
    priority_flags: List[str] = Field(default_factory=list)  # e.g., ["sibling", "low_income"]
    max_distance_km: Optional[float] = None
    
    def has_siblings(self) -> bool:
        """Check if application involves multiple children."""
        return len(self.children) > 1