-- Example configurations for complex types with hybrid agent-plugin architecture
-- Run this after the schema has been updated with new columns

-- First, add the new columns to the existing table
ALTER TABLE matching.Complex_Value_Types
  ADD COLUMN IF NOT EXISTS agent_class VARCHAR(100),
  ADD COLUMN IF NOT EXISTS plugin_pipeline JSONB,
  ADD COLUMN IF NOT EXISTS supports_user_interaction BOOLEAN DEFAULT false,
  ADD COLUMN IF NOT EXISTS default_config JSONB;

-- Insert/Update location_distance complex type with plugin pipeline
INSERT INTO matching.Complex_Value_Types (
    id, type_name, description, schema, agent_class, plugin_pipeline, 
    supports_user_interaction, default_config, examples, version, is_active
) VALUES (
    gen_random_uuid(),
    'location_distance',
    'Distance constraints from multiple locations with travel time support',
    '{
        "type": "object",
        "properties": {
            "locations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Location name (home, work, etc.)"},
                        "address": {"type": "string", "description": "Street address"},
                        "latitude": {"type": "number"},
                        "longitude": {"type": "number"},
                        "max_distance_km": {"type": "number", "description": "Maximum distance in km"},
                        "preferred_distance_km": {"type": "number", "description": "Preferred distance in km"},
                        "max_travel_minutes": {"type": "number", "description": "Maximum travel time in minutes"},
                        "preferred_travel_minutes": {"type": "number", "description": "Preferred travel time in minutes"},
                        "transport_mode": {
                            "type": "string", 
                            "enum": ["walking", "bike", "cycling", "car", "driving", "public_transport", "transit"],
                            "default": "walking"
                        }
                    },
                    "required": ["name"],
                    "anyOf": [
                        {"required": ["max_distance_km"]},
                        {"required": ["max_travel_minutes"]}
                    ]
                }
            }
        },
        "required": ["locations"]
    }',
    'LocationDistanceAgent',
    '[
        {"name": "location_resolver", "config": {"geocoding_timeout": 5000}},
        {"name": "distance_calculator", "config": {"precision": "high"}},
        {"name": "location_scorer", "config": {"scoring_method": "minimum_constraint"}}
    ]',
    true,
    '{"geocoding_service": "google_maps", "cache_duration": 3600}',
    '[
        {
            "description": "Distance constraints from home and work",
            "data": {
                "locations": [
                    {
                        "name": "home",
                        "max_distance_km": 3,
                        "preferred_distance_km": 2
                    },
                    {
                        "name": "work",
                        "address": "Berliner Str. 15, 10713 Berlin",
                        "max_distance_km": 5,
                        "preferred_distance_km": 3
                    }
                ]
            }
        },
        {
            "description": "Travel time constraint with specific location",
            "data": {
                "locations": [
                    {
                        "name": "home",
                        "max_travel_minutes": 15,
                        "preferred_travel_minutes": 10,
                        "transport_mode": "walking"
                    },
                    {
                        "name": "work",
                        "latitude": 52.5200,
                        "longitude": 13.4050,
                        "max_travel_minutes": 20,
                        "transport_mode": "bike"
                    }
                ]
            }
        }
    ]',
    1,
    true
) ON CONFLICT (type_name) DO UPDATE SET
    description = EXCLUDED.description,
    schema = EXCLUDED.schema,
    agent_class = EXCLUDED.agent_class,
    plugin_pipeline = EXCLUDED.plugin_pipeline,
    supports_user_interaction = EXCLUDED.supports_user_interaction,
    default_config = EXCLUDED.default_config,
    examples = EXCLUDED.examples,
    updated_at = NOW();

-- Insert/Update schedule_range complex type (no plugins yet, uses legacy processing)
INSERT INTO matching.Complex_Value_Types (
    id, type_name, description, schema, agent_class, plugin_pipeline,
    supports_user_interaction, default_config, examples, version, is_active
) VALUES (
    gen_random_uuid(),
    'schedule_range',
    'Flexible time schedule constraints with day-of-week support',
    '{
        "type": "object",
        "properties": {
            "start_time": {"type": "string", "pattern": "^[0-9]{2}:[0-9]{2}$"},
            "end_time": {"type": "string", "pattern": "^[0-9]{2}:[0-9]{2}$"},
            "flexibility_minutes": {"type": "number", "minimum": 0, "default": 0},
            "days_of_week": {
                "type": "array", 
                "items": {"type": "string", "enum": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]},
                "default": ["monday", "tuesday", "wednesday", "thursday", "friday"]
            },
            "frequency": {"type": "string", "enum": ["daily", "weekly", "occasionally"], "default": "daily"}
        },
        "required": ["start_time", "end_time"]
    }',
    'ScheduleRangeAgent',
    null,
    false,
    '{"default_flexibility": 30}',
    '[
        {
            "description": "Weekday afternoon pickup",
            "data": {
                "start_time": "16:00",
                "end_time": "18:00",
                "flexibility_minutes": 30,
                "days_of_week": ["monday", "tuesday", "wednesday", "thursday", "friday"]
            }
        }
    ]',
    1,
    true
) ON CONFLICT (type_name) DO UPDATE SET
    description = EXCLUDED.description,
    schema = EXCLUDED.schema,
    agent_class = EXCLUDED.agent_class,
    plugin_pipeline = EXCLUDED.plugin_pipeline,
    supports_user_interaction = EXCLUDED.supports_user_interaction,
    default_config = EXCLUDED.default_config,
    examples = EXCLUDED.examples,
    updated_at = NOW();

-- Insert/Update educational_approach complex type (no plugins, uses semantic matching)
INSERT INTO matching.Complex_Value_Types (
    id, type_name, description, schema, agent_class, plugin_pipeline,
    supports_user_interaction, default_config, examples, version, is_active
) VALUES (
    gen_random_uuid(),
    'educational_approach',
    'Educational philosophy and pedagogical approach preferences',
    '{
        "type": "object",
        "properties": {
            "approaches": {
                "type": "array", 
                "items": {"type": "string"},
                "description": "List of preferred educational approaches"
            },
            "importance_weights": {
                "type": "object",
                "description": "Weight for each approach (0.0-1.0)"
            },
            "must_have_all": {"type": "boolean", "default": false},
            "certification_required": {"type": "boolean", "default": false}
        },
        "required": ["approaches"]
    }',
    'SemanticEnhancedAgent',
    null,
    false,
    '{"semantic_threshold": 0.3, "max_matches": 10}',
    '[
        {
            "description": "Montessori and Waldorf preferences",
            "data": {
                "approaches": ["montessori", "waldorf"],
                "importance_weights": {"montessori": 0.8, "waldorf": 0.6},
                "must_have_all": false,
                "certification_required": true
            }
        }
    ]',
    1,
    true
) ON CONFLICT (type_name) DO UPDATE SET
    description = EXCLUDED.description,
    schema = EXCLUDED.schema,
    agent_class = EXCLUDED.agent_class,
    plugin_pipeline = EXCLUDED.plugin_pipeline,
    supports_user_interaction = EXCLUDED.supports_user_interaction,
    default_config = EXCLUDED.default_config,
    examples = EXCLUDED.examples,
    updated_at = NOW();

-- Insert social_connection complex type with user interaction support
INSERT INTO matching.Complex_Value_Types (
    id, type_name, description, schema, agent_class, plugin_pipeline,
    supports_user_interaction, default_config, examples, version, is_active
) VALUES (
    gen_random_uuid(),
    'social_connection',
    'Family and social relationship constraints (siblings, friends)',
    '{
        "type": "object",
        "properties": {
            "connection_type": {"type": "string", "enum": ["sibling", "friend", "family", "neighbor"]},
            "entity_ids": {"type": "array", "items": {"type": "string", "format": "uuid"}},
            "priority_level": {"type": "string", "enum": ["required", "preferred", "nice_to_have"], "default": "preferred"},
            "same_center_required": {"type": "boolean", "default": false},
            "same_group_required": {"type": "boolean", "default": false}
        },
        "required": ["connection_type", "entity_ids"]
    }',
    null,
    null,
    true,
    '{"lookup_timeout": 5000}',
    '[
        {
            "description": "Sibling must be at same center",
            "data": {
                "connection_type": "sibling",
                "entity_ids": ["550e8400-e29b-41d4-a716-446655440201"],
                "priority_level": "required",
                "same_center_required": true,
                "same_group_required": false
            }
        }
    ]',
    1,
    true
) ON CONFLICT (type_name) DO UPDATE SET
    description = EXCLUDED.description,
    schema = EXCLUDED.schema,
    agent_class = EXCLUDED.agent_class,
    plugin_pipeline = EXCLUDED.plugin_pipeline,
    supports_user_interaction = EXCLUDED.supports_user_interaction,
    default_config = EXCLUDED.default_config,
    examples = EXCLUDED.examples,
    updated_at = NOW();

-- Insert route_preference complex type (for future implementation)
INSERT INTO matching.Complex_Value_Types (
    id, type_name, description, schema, agent_class, plugin_pipeline,
    supports_user_interaction, default_config, examples, version, is_active
) VALUES (
    gen_random_uuid(),
    'route_preference',
    'Commute route integration with detour time constraints',
    '{
        "type": "object",
        "properties": {
            "start_location": {"type": "object", "description": "Starting point with lat/lng or address"},
            "end_location": {"type": "object", "description": "Destination with lat/lng or address"},
            "max_detour_minutes": {"type": "number", "description": "Maximum acceptable detour time"},
            "transport_mode": {"type": "string", "enum": ["car", "public_transport", "bike", "walk"], "default": "car"},
            "avoid_highways": {"type": "boolean", "default": false},
            "prefer_main_roads": {"type": "boolean", "default": true}
        },
        "required": ["start_location", "end_location", "max_detour_minutes"]
    }',
    null,
    null,
    true,
    '{"routing_service": "google_maps"}',
    '[
        {
            "description": "Home to work commute with max 10min detour",
            "data": {
                "start_location": {"latitude": 52.52, "longitude": 13.405},
                "end_location": {"address": "Hauptstraße 42, 10827 Berlin"},
                "max_detour_minutes": 10,
                "transport_mode": "car"
            }
        }
    ]',
    1,
    true
) ON CONFLICT (type_name) DO UPDATE SET
    description = EXCLUDED.description,
    schema = EXCLUDED.schema,
    agent_class = EXCLUDED.agent_class,
    plugin_pipeline = EXCLUDED.plugin_pipeline,
    supports_user_interaction = EXCLUDED.supports_user_interaction,
    default_config = EXCLUDED.default_config,
    examples = EXCLUDED.examples,
    updated_at = NOW();

-- Sample parent preferences with complex types
INSERT INTO matching.Parent_Preferences (
    parent_id, feature_key, value_type, complex_value_type_id, value_data, 
    preference, confidence, source_text
) VALUES 
(
    '550e8400-e29b-41d4-a716-446655440001'::uuid,
    'location_constraint',
    'complex',
    (SELECT id FROM matching.Complex_Value_Types WHERE type_name = 'location_distance'),
    '{
        "locations": [
            {
                "name": "home",
                "max_distance_km": 5,
                "preferred_distance_km": 3
            },
            {
                "name": "work",
                "address": "Potsdamer Platz 1, Berlin",
                "max_travel_minutes": 15,
                "transport_mode": "walking"
            }
        ]
    }',
    'required',
    0.95,
    'Must be within 5km of home and 15 minutes walking from work'
),
(
    '550e8400-e29b-41d4-a716-446655440001'::uuid,
    'pickup_schedule',
    'complex', 
    (SELECT id FROM matching.Complex_Value_Types WHERE type_name = 'schedule_range'),
    '{
        "start_time": "16:00",
        "end_time": "18:30",
        "flexibility_minutes": 30,
        "days_of_week": ["monday", "tuesday", "wednesday", "thursday", "friday"]
    }',
    'preferred',
    0.8,
    'Prefer pickup between 4-6:30 PM on weekdays with 30min flexibility'
)
ON CONFLICT (parent_id, feature_key) DO UPDATE SET
    value_data = EXCLUDED.value_data,
    preference = EXCLUDED.preference,
    confidence = EXCLUDED.confidence,
    source_text = EXCLUDED.source_text,
    complex_value_type_id = EXCLUDED.complex_value_type_id;