"""Database queries for complex preference system."""

import logging
from typing import Any
from uuid import UUID

import asyncpg

from ..models.base import Center, Location, ParentPreference
from ..models.complex_types import ComplexValueType

logger = logging.getLogger(__name__)


class ComplexPreferenceRepository:
    """Repository for complex preference and schema registry operations."""

    def __init__(self, connection_pool: asyncpg.Pool):
        self.pool = connection_pool

    async def get_complex_value_types(self) -> list[ComplexValueType]:
        """Get all active complex value types from schema registry."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT id, type_name, description, schema as schema_definition, examples,
                       version, is_active, agent_class, plugin_pipeline, 
                       supports_user_interaction, default_config, created_at, updated_at
                FROM matching.Complex_Value_Types
                WHERE is_active = true
                ORDER BY type_name
            """)

            return [ComplexValueType(**dict(row)) for row in rows]

    async def get_complex_value_type(self, type_name: str) -> ComplexValueType | None:
        """Get a specific complex value type by name."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT id, type_name, description, schema as schema_definition, examples,
                       version, is_active, agent_class, plugin_pipeline, 
                       supports_user_interaction, default_config, created_at, updated_at
                FROM matching.Complex_Value_Types
                WHERE type_name = $1 AND is_active = true
            """, type_name)

            if row:
                return ComplexValueType(**dict(row))
            return None
    
    async def get_complex_value_type_by_id(self, type_id: UUID) -> ComplexValueType | None:
        """Get a specific complex value type by ID."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT id, type_name, description, schema as schema_definition, examples,
                       version, is_active, agent_class, plugin_pipeline, 
                       supports_user_interaction, default_config, created_at, updated_at
                FROM matching.Complex_Value_Types
                WHERE id = $1 AND is_active = true
            """, type_id)

            if row:
                return ComplexValueType(**dict(row))
            return None

    async def get_parent_preferences(self, parent_id: UUID) -> list[ParentPreference]:
        """Get all preferences for a parent, including complex ones."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT pp.parent_id, pp.feature_key, pp.value_type, pp.value_bool,
                       pp.value_num, pp.value_text, pp.value_list, pp.unit,
                       pp.preference, pp.confidence, pp.source_text,
                       pp.complex_value_type_id, pp.value_data, pp.updated_at,
                       cvt.type_name as complex_type_name
                FROM matching.Parent_Preferences pp
                LEFT JOIN matching.Complex_Value_Types cvt ON pp.complex_value_type_id = cvt.id
                WHERE pp.parent_id = $1
                ORDER BY pp.feature_key
            """, parent_id)

            preferences = []
            for row in rows:
                # Convert to ParentPreference model
                pref_data = {
                    'id': UUID(str(parent_id)),  # Temporary ID
                    'profile_id': row['parent_id'],
                    'property_key': row['feature_key'],
                    'property_type': 'text',  # Default, would map from value_type enum
                    'operator': 'equals',  # Default, would be derived
                    'complex_value_type_id': row['complex_value_type_id'],
                    'value_data': row['value_data'],
                    'value_text': row['value_text'],
                    'value_numeric': row['value_num'],
                    'value_boolean': row['value_bool'],
                    'value_list': row['value_list'],
                    'confidence': row['confidence'] or 1.0,
                    'source_text': row['source_text']
                }

                # Map preference enum to strength
                pref_mapping = {
                    'required': 'REQUIRED',
                    'preferred': 'PREFERRED',
                    'nice_to_have': 'NICE_TO_HAVE',
                    'exclude': 'EXCLUDE'
                }
                pref_data['strength'] = pref_mapping.get(row['preference'], 'PREFERRED')

                preferences.append(ParentPreference(**pref_data))

            return preferences

    async def get_center_features(self, center_id: UUID) -> list[dict[str, Any]]:
        """Get all features for a center, including complex ones."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT cf.center_id, cf.feature_key, cf.value_bool, cf.value_num,
                       cf.unit, cf.source, cf.confidence, cf.raw_phrase,
                       cf.complex_value_type_id, cf.value_data, cf.created_at, cf.updated_at,
                       cvt.type_name as complex_type_name
                FROM matching.Center_Features cf
                LEFT JOIN matching.Complex_Value_Types cvt ON cf.complex_value_type_id = cvt.id
                WHERE cf.center_id = $1
                ORDER BY cf.feature_key
            """, center_id)

            return [dict(row) for row in rows]

    async def discover_centers_with_complex_constraints(
        self,
        parent_id: UUID,
        max_distance_km: float = 10.0,
        location_constraints: list[dict[str, Any]] | None = None,
        limit: int = 50
    ) -> list[UUID]:
        """Discover centers using complex spatial and other constraints."""

        # Get parent location first
        parent_location = await self._get_parent_location(parent_id)
        if not parent_location:
            logger.warning(f"No location found for parent {parent_id}")
            return []

        async with self.pool.acquire() as conn:
            # Base spatial query using PostGIS with join to Addresses table
            base_query = """
                SELECT c.id, c.name,
                       ST_Distance(
                           ST_MakePoint($2, $1)::geography,
                           ST_MakePoint(a.longitude, a.latitude)::geography
                       ) / 1000 as distance_km
                FROM Centers c
                JOIN Addresses a ON c.address_id = a.id
                WHERE ST_DWithin(
                    ST_MakePoint($2, $1)::geography,
                    ST_MakePoint(a.longitude, a.latitude)::geography,
                    $3 * 1000
                )
            """

            # Initialize params for the base query
            params = [parent_location.latitude, parent_location.longitude, max_distance_km]
            
            # Process location constraints if provided
            if location_constraints:
                additional_constraints = []
                param_count = 4

                for constraint in location_constraints:
                    if constraint.get('type') == 'location_distance':
                        locations = constraint.get('locations', [])
                        for loc in locations:
                            if 'latitude' in loc and 'longitude' in loc:
                                max_dist = loc.get('max_distance_km', 10.0)
                                additional_constraints.append(f"""
                                    ST_DWithin(
                                        ST_MakePoint(${param_count+1}, ${param_count})::geography,
                                        ST_MakePoint(a.longitude, a.latitude)::geography,
                                        ${param_count+2} * 1000
                                    )
                                """)
                                params.extend([loc['latitude'], loc['longitude'], max_dist])
                                param_count += 3

                if additional_constraints:
                    base_query += " AND (" + " OR ".join(additional_constraints) + ")"

            base_query += f" ORDER BY distance_km LIMIT {limit}"

            rows = await conn.fetch(base_query, *params)
            return [row['id'] for row in rows]

    async def _get_parent_location(self, parent_id: UUID) -> Location | None:
        """Get parent's home location."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT a.latitude, a.longitude, a.street_address, a.postal_code, a.city
                FROM Profiles p
                JOIN Addresses a ON p.address_id = a.id
                WHERE p.id = $1
            """, parent_id)

            if row and row['latitude'] and row['longitude']:
                return Location(
                    latitude=row['latitude'],
                    longitude=row['longitude'],
                    address=row['street_address'],
                    postal_code=row['postal_code'],
                    city=row['city']
                )
            return None

    async def get_centers_with_features(self, center_ids: list[UUID]) -> list[Center]:
        """Get full center details with all features."""
        if not center_ids:
            return []

        async with self.pool.acquire() as conn:
            # Get basic center info
            center_rows = await conn.fetch("""
                SELECT c.id, c.name, a.latitude, a.longitude, a.street_address, a.postal_code, a.city,
                       c.short_description, c.long_description, c.country_code
                FROM Centers c
                JOIN Addresses a ON c.address_id = a.id
                WHERE c.id = ANY($1)
            """, center_ids)

            centers = {}
            for row in center_rows:
                center_id = row['id']
                centers[center_id] = Center(
                    id=center_id,
                    name=row['name'],
                    location=Location(
                        latitude=row['latitude'],
                        longitude=row['longitude'],
                        address=row['street_address'],
                        postal_code=row['postal_code'],
                        city=row['city']
                    ),
                    short_description=row['short_description'],
                    long_description=row['long_description'],
                    country_code=row['country_code'] or 'DE',
                    opening_hours=[],  # Would load separately
                    properties=[],     # Loaded below
                    capacity_buckets=[] # Would load separately
                )

            # Get features for all centers
            if centers:
                feature_rows = await conn.fetch("""
                    SELECT cf.center_id, cf.feature_key, cf.value_bool, cf.value_num,
                           cf.unit, cf.source, cf.confidence, cf.raw_phrase,
                           cf.complex_value_type_id, cf.value_data
                    FROM matching.Center_Features cf
                    WHERE cf.center_id = ANY($1)
                    ORDER BY cf.center_id, cf.feature_key
                """, center_ids)

                for row in feature_rows:
                    center_id = row['center_id']
                    if center_id in centers:
                        # Convert to CenterProperty (simplified)
                        property_data = {
                            'id': UUID(str(center_id)),  # Temp ID
                            'center_id': center_id,
                            'property_type_id': UUID(str(center_id)),  # Temp
                            'property_key': row['feature_key'],
                            'category': 'facility',  # Default
                            'value_boolean': row['value_bool'],
                            'value_numeric': row['value_num'],
                            'complex_value_type_id': row['complex_value_type_id'],
                            'value_data': row['value_data'],
                            'confidence_score': row['confidence'] or 1.0
                        }

                        # Add to center properties (simplified conversion)
                        centers[center_id].properties.append(property_data)

            return list(centers.values())

    async def validate_complex_preference(
        self,
        type_name: str,
        value_data: dict[str, Any]
    ) -> tuple[bool, list[str]]:
        """Validate complex preference data against its schema."""
        complex_type = await self.get_complex_value_type(type_name)
        if not complex_type:
            return False, [f"Unknown complex type: {type_name}"]

        # Basic validation - in production would use jsonschema library
        schema = complex_type.schema
        errors = []

        if schema.get('required'):
            for req_field in schema['required']:
                if req_field not in value_data:
                    errors.append(f"Missing required field: {req_field}")

        # Type validation for properties
        properties = schema.get('properties', {})
        for field, field_schema in properties.items():
            if field in value_data:
                expected_type = field_schema.get('type')
                actual_value = value_data[field]

                if expected_type == 'string' and not isinstance(actual_value, str):
                    errors.append(f"Field {field} must be string")
                elif expected_type == 'number' and not isinstance(actual_value, int | float):
                    errors.append(f"Field {field} must be number")
                elif expected_type == 'boolean' and not isinstance(actual_value, bool):
                    errors.append(f"Field {field} must be boolean")
                elif expected_type == 'array' and not isinstance(actual_value, list):
                    errors.append(f"Field {field} must be array")

        return len(errors) == 0, errors
