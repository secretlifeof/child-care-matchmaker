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


class PropertyCategory(str, Enum):
    """Categories of center properties."""
    FACILITY = "facility"
    CERTIFICATE = "certificate"
    APPROACH = "approach"
    AWARD = "award"
    SERVICE = "service"
    DIETARY = "dietary"
    PROGRAM = "program"


class ValueType(str, Enum):
    """Types of property values."""
    TEXT = "text"
    BOOLEAN = "boolean"
    NUMBER = "number"
    DATE = "date"
    LIST = "list"


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
    """Time slot for availability."""
    day_of_week: int = Field(..., ge=0, le=6)  # 0=Monday, 6=Sunday
    start_hour: int = Field(..., ge=0, le=23)
    start_minute: int = Field(default=0, ge=0, le=59)
    end_hour: int = Field(..., ge=0, le=23)
    end_minute: int = Field(default=0, ge=0, le=59)
    
    def overlaps_with(self, other: "TimeSlot") -> bool:
        """Check if time slots overlap."""
        if self.day_of_week != other.day_of_week:
            return False
        self_start = self.start_hour * 60 + self.start_minute
        self_end = self.end_hour * 60 + self.end_minute
        other_start = other.start_hour * 60 + other.start_minute
        other_end = other.end_hour * 60 + other.end_minute
        return not (self_end <= other_start or other_end <= self_start)


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
    """Parent's preference for matching."""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    profile_id: UUID
    property_type_id: Optional[UUID] = None
    property_key: str
    operator: ComparisonOperator
    value_text: Optional[str] = None
    value_numeric: Optional[float] = None
    value_boolean: Optional[bool] = None
    value_date: Optional[date] = None
    value_list: Optional[List[str]] = None
    weight: float = Field(default=0.5, ge=0.0, le=1.0)
    threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    
    @property
    def constraint_type(self) -> ConstraintType:
        """Derive constraint type from threshold and weight."""
        if self.threshold >= 0.9:
            return ConstraintType.MUST_HAVE
        elif self.threshold <= 0.1:
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