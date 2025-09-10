"""Enhanced models matching the database schema."""

from datetime import date, datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from .base import AgeGroup, Location, TimeSlot


class ApplicationStatus(BaseModel):
    """Application status definition."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    description: str | None = None
    is_terminal: bool | None = None
    is_default: bool = True
    is_enabled: bool = True
    created_by: UUID | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class EnhancedApplication(BaseModel):
    """Enhanced application matching database schema."""
    model_config = ConfigDict(from_attributes=True)

    # Core fields from Applications table
    id: UUID
    profile_id: UUID | None = None
    child_id: UUID | None = None
    center_id: UUID | None = None
    status_id: UUID | None = None

    # Dates
    application_date: datetime = Field(default_factory=datetime.utcnow)
    desired_start_date: date
    earliest_start_date: date
    latest_start_date: date

    # Additional fields
    notes: str | None = None
    match_score: float | None = None

    # Audit fields
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Derived/computed fields for matching
    children: list["Child"] = Field(default_factory=list)
    home_location: Location | None = None
    preferences: list["ApplicationPreference"] = Field(default_factory=list)
    property_values: list["ApplicationPropertyValue"] = Field(default_factory=list)
    desired_hours: list[TimeSlot] = Field(default_factory=list)
    priority_flags: list[str] = Field(default_factory=list)
    max_distance_km: float | None = None

    def has_siblings(self) -> bool:
        """Check if application involves multiple children."""
        return len(self.children) > 1


class ApplicationStatusHistory(BaseModel):
    """Application status change history."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    application_id: UUID
    status_id: UUID
    changed_by_profile_id: UUID | None = None
    status_date: datetime
    notes: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class SpotOfferStatus(str, Enum):
    """Status of spot offers."""
    PENDING = "pending"
    ACCEPTED = "accepted"
    DECLINED = "declined"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class ApplicationSpotOffer(BaseModel):
    """Spot offer for an application."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    application_id: UUID
    center_id: UUID | None = None
    group_id: UUID | None = None

    # Offer details
    offer_date: datetime = Field(default_factory=datetime.utcnow)
    expiry_date: datetime | None = None
    proposed_start: datetime | None = None
    seat_label: str | None = None
    notes: str | None = None

    # Workflow
    status: SpotOfferStatus = SpotOfferStatus.PENDING
    offered_by: UUID | None = None
    responded_by: UUID | None = None
    responded_at: datetime | None = None
    cancelled_by: UUID | None = None
    cancelled_at: datetime | None = None

    # Audit
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class QuestionType(str, Enum):
    """Types of application property questions."""
    TEXT = "text"
    NUMBER = "number"
    BOOLEAN = "boolean"
    SELECT = "select"
    MULTI_SELECT = "multi_select"
    DATE = "date"
    TIME = "time"
    EMAIL = "email"
    PHONE = "phone"
    URL = "url"
    TEXTAREA = "textarea"


class PlatformCategoryScope(str, Enum):
    """Scope for property definitions."""
    PLATFORM = "platform"
    ORGANIZATION = "organization"
    CENTER = "center"


class ApplicationPropertyDefinition(BaseModel):
    """Definition of application properties."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    description: str | None = None
    question_type: QuestionType
    is_required: bool = False
    default_value: str | None = None
    display_order: int = 0

    # Validation and UI
    validations: dict[str, Any] = Field(default_factory=dict)
    properties: dict[str, Any] = Field(default_factory=dict)

    # Scope
    scope: PlatformCategoryScope = PlatformCategoryScope.PLATFORM
    scope_id: UUID | None = None
    country_code: str | None = None
    region_code: str | None = None

    # Status
    is_active: bool = True
    created_by: UUID | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ApplicationPropertyOption(BaseModel):
    """Options for select-type properties."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    property_definition_id: UUID
    option_text: str
    option_value: str | None = None
    display_order: int = 0

    # Workflow control
    blocks_submission: bool = False
    block_message: str | None = None

    # Status
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ApplicationPropertyValue(BaseModel):
    """Value of an application property."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    application_id: UUID
    property_definition_id: UUID
    answer_value: Any  # JSON field - can be any type

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class PropertyScope(str, Enum):
    """Scope for property types."""
    PLATFORM = "platform"
    ORGANIZATION = "organization"
    CENTER = "center"


class PropertyEntityType(str, Enum):
    """Types of entities that can have properties."""
    CENTER = "center"
    ORGANIZATION = "organization"
    PROFILE = "profile"
    APPLICATION = "application"


class EnhancedPropertyType(BaseModel):
    """Enhanced property type matching database schema."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    description: str | None = None
    property_key: str | None = None
    category: str | None = None
    value_type: str = "text"
    is_searchable: bool = True
    is_filterable: bool = True
    display_order: int = 0

    # Scope
    scope: PropertyScope = PropertyScope.PLATFORM
    country_code: str | None = None
    region_code: str | None = None
    owner_profile_id: UUID | None = None

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class EnhancedEntityProperty(BaseModel):
    """Enhanced entity property matching database schema."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    property_type_id: UUID
    property_value: str | None = None
    value_numeric: float | None = None
    value_boolean: bool | None = None
    value_date: date | None = None
    confidence_score: float | None = None
    source: str | None = None

    # Entity reference
    entity_type: PropertyEntityType
    entity_id: UUID

    description: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def get_value(self) -> Any:
        """Get the actual value regardless of type."""
        if self.value_boolean is not None:
            return self.value_boolean
        if self.value_numeric is not None:
            return self.value_numeric
        if self.value_date is not None:
            return self.value_date
        return self.property_value


class WaitingListContainer(BaseModel):
    """Waiting list container for center groups."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    center_id: UUID | None = None
    group_id: UUID | None = None
    name: str | None = None
    description: str | None = None
    min_child_age_months: int
    max_child_age_months: int

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def age_band(self) -> AgeGroup:
        """Get age band as AgeGroup object."""
        return AgeGroup(
            min_age_months=self.min_child_age_months,
            max_age_months=self.max_child_age_months,
            name=self.name or f"{self.min_child_age_months}-{self.max_child_age_months} months"
        )


class WaitingListEntry(BaseModel):
    """Entry in a waiting list."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    waiting_list_container_id: UUID | None = None
    application_id: UUID | None = None
    position: int
    earliest_start_date: datetime
    latest_start_date: datetime
    special_needs_accommodated: bool | None = None
    priority_factors: dict[str, Any] | None = None
    notes: str | None = None
    estimated_availability_date: datetime | None = None

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class EnhancedCenter(BaseModel):
    """Enhanced center model matching database schema."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    license_number: str | None = None
    description: str | None = None
    short_description: str | None = None
    long_description: str | None = None
    raw_text_content: str | None = None
    last_ai_processing: datetime | None = None
    country_code: str = "DE"
    total_capacity: int

    # Contact info
    address_id: UUID | None = None
    contact_person_id: UUID | None = None
    phone_number: str | None = None
    website: str | None = None
    email: str | None = None

    # Business info
    currency_code: str = "EUR"
    tax_id: str | None = None
    payment_terms_days: int = 30

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Derived fields for matching
    location: Location | None = None
    opening_hours: list[TimeSlot] = Field(default_factory=list)
    properties: list[EnhancedEntityProperty] = Field(default_factory=list)
    capacity_buckets: list["CapacityBucket"] = Field(default_factory=list)
    waiting_list_containers: list[WaitingListContainer] = Field(default_factory=list)

    def has_property(self, property_key: str) -> bool:
        """Check if center has a specific property."""
        return any(
            p for p in self.properties
            if hasattr(p, 'property_key') and p.property_key == property_key
        )

    def get_property(self, property_key: str) -> EnhancedEntityProperty | None:
        """Get a specific property by key."""
        return next(
            (p for p in self.properties
             if hasattr(p, 'property_key') and p.property_key == property_key),
            None
        )


class Child(BaseModel):
    """Child information."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    family_id: UUID
    name: str
    birth_date: date
    special_needs: dict[str, Any] | None = None

    @property
    def age_months(self) -> int:
        """Calculate age in months."""
        today = date.today()
        months = (today.year - self.birth_date.year) * 12
        months += today.month - self.birth_date.month
        if today.day < self.birth_date.day:
            months -= 1
        return max(0, months)


class ApplicationPreference(BaseModel):
    """Application preference derived from property values."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    profile_id: UUID
    property_type_id: UUID | None = None
    property_key: str
    operator: str = "equals"
    value_text: str | None = None
    value_numeric: float | None = None
    value_boolean: bool | None = None
    value_date: date | None = None
    value_list: list[str] | None = None
    weight: float = Field(default=0.5, ge=0.0, le=1.0)
    threshold: float = Field(default=0.0, ge=0.0, le=1.0)

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


# Forward references
EnhancedApplication.model_rebuild()
EnhancedCenter.model_rebuild()
