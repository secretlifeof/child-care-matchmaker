# Request Examples

This document provides complete examples of API requests with the new complex preference system.

## Basic Recommendation Request

Get personalized recommendations using parent ID for database lookup:

```json
{
  "parent_id": "550e8400-e29b-41d4-a716-446655440001",
  "limit": 10,
  "max_distance_km": 5.0,
  "include_full_centers": false,
  "include_explanations": true,
  "matching_config": {
    "scoring_weights": {
      "spatial_weight": 0.3,
      "preference_weight": 0.4,
      "quality_weight": 0.3
    }
  }
}
```

## Complex Preference Examples

The system automatically loads preferences from the database. Here are examples of the complex preference data structures:

### Location Distance Preference (Enhanced with Travel Time Support)
```json
{
  "parent_id": "uuid",
  "feature_key": "location_constraint",
  "complex_value_type_id": "location_distance_type_id",
  "value_data": {
    "locations": [
      {
        "name": "home",
        "latitude": 52.52,
        "longitude": 13.405,
        "max_distance_km": 3
      },
      {
        "name": "work", 
        "address": "Berliner Str. 15, Berlin",
        "max_distance_km": 5,
        "preferred_distance_km": 3
      }
    ]
  },
  "preference": "required",
  "confidence": 0.95
}
```

### Location Distance with Travel Time Constraints
```json
{
  "parent_id": "uuid",
  "feature_key": "commute_constraint",
  "complex_value_type_id": "location_distance_type_id",
  "value_data": {
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
      },
      {
        "name": "specific_address",
        "address": "Reuterstraße 33, Berlin",
        "max_travel_minutes": 25,
        "transport_mode": "public_transport"
      }
    ]
  },
  "preference": "required",
  "confidence": 0.9
}
```

### Schedule Range Preference
```json
{
  "parent_id": "uuid", 
  "feature_key": "schedule_constraint",
  "complex_value_type_id": "schedule_range_type_id",
  "value_data": {
    "start_time": "16:00",
    "end_time": "18:00",
    "flexibility_minutes": 30,
    "days_of_week": ["monday", "tuesday", "wednesday", "thursday", "friday"]
  },
  "preference": "preferred",
  "confidence": 0.85
}
```

### Social Connection Preference  
```json
{
  "parent_id": "uuid",
  "feature_key": "sibling_constraint", 
  "complex_value_type_id": "social_connection_type_id",
  "value_data": {
    "connection_type": "sibling",
    "entity_ids": ["550e8400-e29b-41d4-a716-446655440201"],
    "priority_level": "required",
    "same_center_required": true,
    "same_group_required": false
  },
  "preference": "required",
  "confidence": 1.0
}
```

### Educational Approach Preference
```json
{
  "parent_id": "uuid",
  "feature_key": "educational_approach",
  "complex_value_type_id": "educational_approach_type_id", 
  "value_data": {
    "approaches": ["montessori", "waldorf"],
    "importance_weights": {"montessori": 0.8, "waldorf": 0.6},
    "must_have_all": false,
    "certification_required": true
  },
  "preference": "preferred",
  "confidence": 0.9
}
```

### Route Preference
```json
{
  "parent_id": "uuid",
  "feature_key": "commute_route",
  "complex_value_type_id": "route_preference_type_id",
  "value_data": {
    "start_location": {"latitude": 52.52, "longitude": 13.405},
    "end_location": {"address": "Hauptstraße 42, 10827 Berlin"},
    "max_detour_minutes": 10,
    "transport_mode": "car"
  },
  "preference": "preferred",
  "confidence": 0.8
}
```

## API Response Examples

### Enhanced Recommendation Response

```json
{
  "offers": [
    {
      "center_id": "550e8400-e29b-41d4-a716-446655440401",
      "center_name": "Green Garden Daycare",
      "score": 0.87,
      "preference_matches": {
        "simple_preferences": {
          "satisfied": 8,
          "total": 10
        },
        "complex_preferences": {
          "location_distance": {
            "satisfied": true,
            "details": "home: ideal walking time (8min); work: acceptable bike time (18min)",
            "score": 0.95,
            "processing_metadata": {
              "pipeline_used": true,
              "pipeline_name": "location_resolver->distance_calculator->location_scorer",
              "execution_times": [120, 45, 30]
            }
          },
          "schedule_range": {
            "satisfied": true,
            "details": "Fully supports schedule 16:00-18:00 (with 30min flexibility)",
            "score": 1.0,
            "processing_metadata": {
              "pipeline_used": false,
              "agent_type": "ScheduleRangeAgent"
            }
          },
          "social_connection": {
            "satisfied": true,
            "details": "Siblings already enrolled at this center",
            "score": 1.0
          },
          "educational_approach": {
            "satisfied": true,
            "details": "Offers: montessori, waldorf (no certification)",
            "score": 0.7,
            "semantic_enhancements": [
              {"approach": "montessori", "matched_feature": "montessori_materials", "similarity": 0.92},
              {"approach": "waldorf", "matched_feature": "nature_play", "similarity": 0.78}
            ]
          }
        }
      },
      "center_details": {
        "location": {
          "latitude": 52.5150,
          "longitude": 13.4100,
          "address": "Gartenstraße 45"
        },
        "description": "Modern daycare with outdoor space and Montessori approach"
      }
    }
  ],
  "processing_details": {
    "centers_evaluated": 45,
    "complex_types_processed": ["location_distance", "schedule_range", "social_connection", "educational_approach"],
    "processing_time_ms": 234
  },
  "success": true
}
```

### Response with User Interaction Required

```json
{
  "offers": [],
  "success": false,
  "message": "User interaction required for complex preference processing",
  "user_interactions": [
    {
      "preference_id": "550e8400-e29b-41d4-a716-446655440001",
      "complex_type": "location_distance", 
      "interaction": {
        "required": true,
        "type": "location_input",
        "question": "Where is your work located?",
        "field_name": "work_location",
        "context": {
          "preference_id": "550e8400-e29b-41d4-a716-446655440001",
          "location_reference": {
            "name": "work",
            "max_travel_minutes": 15,
            "transport_mode": "walking"
          }
        }
      }
    }
  ],
  "processing_details": {
    "centers_evaluated": 0,
    "complex_types_processed": ["location_distance"],
    "processing_time_ms": 156,
    "interaction_required": true
  }
}
```

### Complex Types Discovery Response

```json
{
  "complex_types": [
    {
      "type_name": "location_distance",
      "description": "Distance constraints from multiple locations",
      "schema": {
        "type": "object",
        "properties": {
          "locations": {
            "type": "array",
            "items": {
              "type": "object", 
              "properties": {
                "name": {"type": "string"},
                "latitude": {"type": "number"},
                "longitude": {"type": "number"},
                "address": {"type": "string"},
                "max_distance_km": {"type": "number"},
                "preferred_distance_km": {"type": "number"}
              },
              "required": ["name", "max_distance_km"]
            }
          }
        },
        "required": ["locations"]
      },
      "examples": [
        {
          "locations": [
            {
              "name": "home",
              "latitude": 52.52,
              "longitude": 13.405,
              "max_distance_km": 3
            },
            {
              "name": "work",
              "address": "Berliner Str. 15",
              "max_distance_km": 5,
              "preferred_distance_km": 3
            }
          ]
        }
      ],
      "version": 1
    }
  ],
  "count": 5
}
```

### Service Statistics Response

```json
{
  "status": "healthy", 
  "version": "2.1.0",
  "features": {
    "complex_preferences": true,
    "schema_registry": true,
    "spatial_queries": true,
    "detailed_explanations": true,
    "database_integration": true
  },
  "complex_types_supported": [
    "location_distance",
    "schedule_range",
    "social_connection",
    "educational_approach", 
    "route_preference"
  ],
  "endpoints": {
    "recommend": "Generate personalized recommendations with complex preferences",
    "allocate": "Global optimal allocation",
    "waitlist": "Center-specific waitlist",
    "batch": "Batch processing",
    "complex-types": "Get available complex preference types"
  },
  "api_changes": {
    "parameter_changes": {
      "top_k": "replaced with 'limit'",
      "parent_id": "now primary identifier for database lookup"
    }
  }
}
```

## Error Response Examples

### Parent Not Found
```json
{
  "offers": [],
  "success": false,
  "message": "No preferences found for parent",
  "processing_details": {
    "centers_evaluated": 0,
    "complex_types_processed": [],
    "processing_time_ms": 15
  }
}
```

### No Centers Found
```json
{
  "offers": [],
  "success": true,
  "message": "No centers found within constraints", 
  "processing_details": {
    "centers_evaluated": 0,
    "complex_types_processed": ["location_distance"],
    "processing_time_ms": 45
  }
}
```

### Processing Error
```json
{
  "offers": [],
  "success": false,
  "message": "Processing error: Database connection failed",
  "processing_details": {
    "centers_evaluated": 0,
    "complex_types_processed": [],
    "processing_time_ms": 125
  }
}
```

## Integration Notes

### Database-Driven Approach
- **Preference Loading**: All preferences (simple and complex) loaded from database using `parent_id`
- **Schema Registry**: Complex types defined in `matching.Complex_Value_Types` table
- **Spatial Queries**: Uses PostGIS for efficient geographic distance calculations
- **No Inline Data**: Application/center data no longer provided in requests

### Key Changes from Legacy API
- **Parameter Change**: `top_k` → `limit`
- **Database Lookup**: Use `parent_id` instead of inline application data  
- **Complex Support**: Automatic processing of complex preference types
- **Enhanced Response**: Detailed explanations for all preference matches
- **Processing Metadata**: Information about complex types used and performance

### Best Practices
- **Parent ID Required**: Always provide valid `parent_id` for database lookup
- **Distance Limits**: Set reasonable `max_distance_km` to limit search scope
- **Explanation Details**: Enable `include_explanations` for debugging and user transparency
- **Error Handling**: Check `success` field and `message` for error details