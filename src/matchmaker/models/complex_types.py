"""Complex value type models for the schema registry system."""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, validator


class ComplexValueType(BaseModel):
    """Schema registry entry for complex preference structures."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    type_name: str
    description: str | None = None
    schema_definition: dict[str, Any]  # JSON Schema for validation
    examples: list[dict[str, Any]] | None = None
    version: int = 1
    is_active: bool = True
    created_at: datetime
    updated_at: datetime

    @validator('schema_definition')
    def validate_json_schema(cls, v):
        """Ensure schema is a valid JSON Schema."""
        required_keys = {'type', 'properties'}
        if not all(k in v for k in required_keys):
            raise ValueError("Schema must have 'type' and 'properties' keys")
        return v


class LocationDistance(BaseModel):
    """Complex type for multi-location distance constraints."""
    locations: list[dict[str, Any]]

    @validator('locations')
    def validate_locations(cls, v):
        """Validate each location has required fields."""
        for loc in v:
            if 'name' not in loc:
                raise ValueError("Each location must have a 'name'")
            if 'max_distance_km' not in loc:
                raise ValueError("Each location must have 'max_distance_km'")
            # Must have either lat/long or address
            if not (('latitude' in loc and 'longitude' in loc) or 'address' in loc):
                raise ValueError("Location must have either lat/long or address")
        return v


class ScheduleRange(BaseModel):
    """Complex type for flexible time schedules."""
    start_time: str = Field(..., pattern=r'^[0-9]{2}:[0-9]{2}$')
    end_time: str = Field(..., pattern=r'^[0-9]{2}:[0-9]{2}$')
    flexibility_minutes: int | None = Field(default=0, ge=0)
    days_of_week: list[str] | None = None
    frequency: str | None = Field(default="daily")

    @validator('days_of_week')
    def validate_days(cls, v):
        """Validate days of week."""
        valid_days = {'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'}
        if v:
            for day in v:
                if day.lower() not in valid_days:
                    raise ValueError(f"Invalid day: {day}")
        return v


class SocialConnection(BaseModel):
    """Complex type for family/social relationships."""
    connection_type: str  # sibling, friend, family
    entity_ids: list[UUID]
    priority_level: str | None = "preferred"  # required, preferred, nice_to_have
    same_center_required: bool = False
    same_group_required: bool = False


class EducationalApproach(BaseModel):
    """Complex type for pedagogical preferences."""
    approaches: list[str]  # montessori, waldorf, reggio_emilia, etc.
    importance_weights: dict[str, float] | None = None
    must_have_all: bool = False
    certification_required: bool = False


class RoutePreference(BaseModel):
    """Complex type for commute route integration."""
    start_location: dict[str, Any]  # lat/long or address
    end_location: dict[str, Any]
    max_detour_minutes: int
    transport_mode: str = "car"  # car, public_transport, bike, walk
    avoid_highways: bool = False
    prefer_main_roads: bool = True


class ComplexPreferenceValue(BaseModel):
    """Wrapper for complex preference values with type information."""
    complex_type_id: UUID
    type_name: str
    value_data: dict[str, Any]

    def validate_against_schema(self, schema_definition: dict[str, Any]) -> bool:
        """Validate value_data against the provided JSON schema."""
        # This would use jsonschema library in production
        # For now, basic validation
        if schema_definition.get('required'):
            for req_field in schema_definition['required']:
                if req_field not in self.value_data:
                    return False
        return True


# Predefined complex type schemas
COMPLEX_TYPE_SCHEMAS = {
    "location_distance": {
        "type": "object",
        "properties": {
            "locations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "address": {"type": "string"},
                        "latitude": {"type": "number"},
                        "longitude": {"type": "number"},
                        "max_distance_km": {"type": "number"},
                        "preferred_distance_km": {"type": "number"}
                    },
                    "required": ["name", "max_distance_km"]
                }
            }
        },
        "required": ["locations"]
    },
    "schedule_range": {
        "type": "object",
        "properties": {
            "start_time": {"type": "string", "pattern": "^[0-9]{2}:[0-9]{2}$"},
            "end_time": {"type": "string", "pattern": "^[0-9]{2}:[0-9]{2}$"},
            "flexibility_minutes": {"type": "number", "minimum": 0},
            "days_of_week": {"type": "array", "items": {"type": "string"}},
            "frequency": {"type": "string", "enum": ["daily", "weekly", "occasionally"]}
        },
        "required": ["start_time", "end_time"]
    },
    "social_connection": {
        "type": "object",
        "properties": {
            "connection_type": {"type": "string"},
            "entity_ids": {"type": "array", "items": {"type": "string"}},
            "priority_level": {"type": "string"},
            "same_center_required": {"type": "boolean"},
            "same_group_required": {"type": "boolean"}
        },
        "required": ["connection_type", "entity_ids"]
    },
    "educational_approach": {
        "type": "object",
        "properties": {
            "approaches": {"type": "array", "items": {"type": "string"}},
            "importance_weights": {"type": "object"},
            "must_have_all": {"type": "boolean"},
            "certification_required": {"type": "boolean"}
        },
        "required": ["approaches"]
    },
    "route_preference": {
        "type": "object",
        "properties": {
            "start_location": {"type": "object"},
            "end_location": {"type": "object"},
            "max_detour_minutes": {"type": "number"},
            "transport_mode": {"type": "string"},
            "avoid_highways": {"type": "boolean"},
            "prefer_main_roads": {"type": "boolean"}
        },
        "required": ["start_location", "end_location", "max_detour_minutes"]
    }
}


# Example data for LLM training
COMPLEX_TYPE_EXAMPLES = {
    "location_distance": [
        {
            "locations": [
                {"name": "home", "latitude": 52.52, "longitude": 13.405, "max_distance_km": 3},
                {"name": "work", "address": "Berliner Str. 15, 10713 Berlin", "max_distance_km": 5, "preferred_distance_km": 3}
            ]
        }
    ],
    "schedule_range": [
        {
            "start_time": "16:00",
            "end_time": "18:00",
            "flexibility_minutes": 30,
            "days_of_week": ["monday", "tuesday", "wednesday", "thursday", "friday"]
        }
    ],
    "social_connection": [
        {
            "connection_type": "sibling",
            "entity_ids": ["550e8400-e29b-41d4-a716-446655440001"],
            "priority_level": "required",
            "same_center_required": True,
            "same_group_required": False
        }
    ],
    "educational_approach": [
        {
            "approaches": ["montessori", "waldorf"],
            "importance_weights": {"montessori": 0.8, "waldorf": 0.6},
            "must_have_all": False,
            "certification_required": True
        }
    ],
    "route_preference": [
        {
            "start_location": {"latitude": 52.52, "longitude": 13.405},
            "end_location": {"address": "Hauptstraße 42, 10827 Berlin"},
            "max_detour_minutes": 10,
            "transport_mode": "car"
        }
    ]
}
