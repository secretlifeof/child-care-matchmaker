# API Parameters Reference

This document describes all available parameters for the Parent-Daycare Matchmaker service API.

## Common Request Parameters

### Core Parameters

```json
{
  "parent_id": "uuid",                   // Required: Database ID for parent lookup
  "limit": 10,                           // Number of results (1-100, default: 10)  
  "max_distance_km": 10.0,               // Distance limit in kilometers
  "include_full_centers": false,         // Include detailed center info
  "include_explanations": true           // Include match explanations
}
```

### Scoring Configuration

```json
{
  "matching_config": {
    "scoring_weights": {
      "spatial_weight": 0.3,             // Distance & route preferences
      "preference_weight": 0.4,          // User preferences  
      "quality_weight": 0.3              // Center quality
    }
  }
}
```

## Complex Preference Support

The API now supports complex preferences stored in the database:

### Location Distance
Multi-location constraints with home, work, or other important locations:
```json
{
  "complex_value_type_id": "uuid",
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
        "address": "Berliner Str. 15",
        "max_distance_km": 5,
        "preferred_distance_km": 3
      }
    ]
  }
}
```

### Schedule Range  
Flexible time schedules with day-of-week support:
```json
{
  "complex_value_type_id": "uuid",
  "value_data": {
    "start_time": "16:00",
    "end_time": "18:00", 
    "flexibility_minutes": 30,
    "days_of_week": ["monday", "tuesday", "wednesday", "thursday", "friday"]
  }
}
```

### Social Connection
Sibling and family relationship constraints:
```json
{
  "complex_value_type_id": "uuid",
  "value_data": {
    "connection_type": "sibling",
    "entity_ids": ["550e8400-e29b-41d4-a716-446655440001"],
    "same_center_required": true
  }
}
```

### Educational Approach
Pedagogical method preferences:
```json
{
  "complex_value_type_id": "uuid", 
  "value_data": {
    "approaches": ["montessori", "waldorf"],
    "importance_weights": {"montessori": 0.8, "waldorf": 0.6},
    "certification_required": true
  }
}
```

### Route Preference
Commute route integration:
```json
{
  "complex_value_type_id": "uuid",
  "value_data": {
    "start_location": {"latitude": 52.52, "longitude": 13.405},
    "end_location": {"address": "Hauptstraße 42, 10827 Berlin"},  
    "max_detour_minutes": 10,
    "transport_mode": "car"
  }
}
```

## API Endpoints

### POST /api/matches/recommend

Get personalized recommendations with complex preference support.

**Request:**
```json
{
  "parent_id": "uuid",
  "limit": 10,
  "max_distance_km": 10.0,
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

**Response:**
```json
{
  "offers": [
    {
      "center_id": "uuid",
      "center_name": "Happy Kids Daycare", 
      "score": 0.87,
      "preference_matches": {
        "simple_preferences": {
          "satisfied": 8,
          "total": 10
        },
        "complex_preferences": {
          "location_distance": {
            "satisfied": true,
            "details": "Within range of home and work",
            "score": 0.95
          },
          "schedule_range": {
            "satisfied": true,
            "details": "Supports flexible pickup 15:30-18:30", 
            "score": 0.85
          }
        }
      },
      "center_details": {
        "location": {
          "latitude": 52.5150,
          "longitude": 13.4100,
          "address": "Kindergarten Str. 123"
        },
        "description": "Modern daycare with outdoor space"
      }
    }
  ],
  "processing_details": {
    "centers_evaluated": 45,
    "complex_types_processed": ["location_distance", "schedule_range"],
    "processing_time_ms": 234
  },
  "success": true
}
```

### GET /api/matches/complex-types

Get available complex preference types from schema registry.

**Response:**
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
                "max_distance_km": {"type": "number"}
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

### GET /api/matches/stats

Get service statistics and capabilities.

**Response:**
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

## Legacy Parameters (Deprecated)

- `top_k` → Use `limit` instead
- `application_id` → Use `parent_id` for database lookup
- Inline application data → Preferences now loaded from database

## Error Responses

All endpoints return consistent error responses:

```json
{
  "offers": [],
  "success": false,
  "message": "Processing error: Parent not found",
  "processing_details": {
    "centers_evaluated": 0,
    "complex_types_processed": [],
    "processing_time_ms": 15
  }
}
```

Common error scenarios:
- **Parent not found**: Invalid `parent_id` 
- **No preferences**: Parent has no stored preferences
- **No centers found**: No centers match spatial constraints
- **Database error**: Connection or query issues
- **Processing timeout**: Complex preference processing took too long