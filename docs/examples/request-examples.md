# Request Examples

This document provides complete examples of API requests for different scenarios.

## Basic Recommendation Request

Get personalized recommendations for a single parent:

```json
{
  "application": {
    "id": "550e8400-e29b-41d4-a716-446655440001",
    "family_id": "550e8400-e29b-41d4-a716-446655440101",
    "children": [
      {
        "id": "550e8400-e29b-41d4-a716-446655440201",
        "family_id": "550e8400-e29b-41d4-a716-446655440101",
        "name": "Emma Johnson",
        "birth_date": "2022-03-15"
      }
    ],
    "home_location": {
      "latitude": 52.5200,
      "longitude": 13.4050,
      "address": "Hauptstraße 123",
      "postal_code": "10117",
      "city": "Berlin",
      "country_code": "DE"
    },
    "preferences": [
      {
        "id": "550e8400-e29b-41d4-a716-446655440301",
        "profile_id": "550e8400-e29b-41d4-a716-446655440001",
        "property_key": "organic_food",
        "operator": "equals",
        "value_boolean": true,
        "weight": 0.8,
        "threshold": 0.7
      }
    ],
    "desired_start_date": "2024-09-01",
    "desired_hours": [
      {"day_of_week": 0, "start_hour": 8, "end_hour": 16},
      {"day_of_week": 1, "start_hour": 8, "end_hour": 16},
      {"day_of_week": 2, "start_hour": 8, "end_hour": 16},
      {"day_of_week": 3, "start_hour": 8, "end_hour": 16},
      {"day_of_week": 4, "start_hour": 8, "end_hour": 16}
    ],
    "max_distance_km": 5.0,
    "priority_flags": []
  },
  "centers": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440401",
      "name": "Green Garden Daycare",
      "location": {
        "latitude": 52.5150,
        "longitude": 13.4100,
        "address": "Gartenstraße 45",
        "postal_code": "10115",
        "city": "Berlin",
        "country_code": "DE"
      },
      "properties": [
        {
          "id": "550e8400-e29b-41d4-a716-446655440501",
          "center_id": "550e8400-e29b-41d4-a716-446655440401",
          "property_key": "organic_food",
          "category": "service",
          "value_boolean": true,
          "source": "verified"
        }
      ],
      "capacity_buckets": [
        {
          "id": "550e8400-e29b-41d4-a716-446655440701",
          "center_id": "550e8400-e29b-41d4-a716-446655440401",
          "age_band": {
            "min_age_months": 12,
            "max_age_months": 36,
            "name": "Toddlers"
          },
          "start_month": "2024-09-01",
          "total_capacity": 20,
          "available_capacity": 5
        }
      ]
    }
  ],
  "top_k": 10,
  "include_full_centers": false,
  "include_explanations": true
}
```

## Global Allocation Request

Optimal allocation across multiple applications:

```json
{
  "applications": [
    {
      "id": "app1",
      "children": [...],
      "preferences": [...],
      "priority_flags": ["sibling"]
    },
    {
      "id": "app2", 
      "children": [...],
      "preferences": [...],
      "priority_flags": ["low_income"]
    }
  ],
  "centers": [...],
  "respect_capacity": true,
  "prioritize_siblings": true,
  "optimization_objective": "fairness",
  "matching_config": {
    "policy_settings": {
      "priority_tiers": {
        "sibling": 1000,
        "low_income": 500
      }
    }
  }
}
```

## Waitlist Generation Request

Generate priority-ordered waitlist for a center:

```json
{
  "center_id": "550e8400-e29b-41d4-a716-446655440405",
  "applications": [
    {
      "id": "app1",
      "priority_flags": ["sibling"]
    },
    {
      "id": "app2", 
      "priority_flags": ["municipality"]
    },
    {
      "id": "app3",
      "priority_flags": []
    }
  ],
  "center": {
    "id": "550e8400-e29b-41d4-a716-446655440405",
    "name": "Popular Daycare",
    "capacity_buckets": [
      {
        "id": "bucket1",
        "total_capacity": 20,
        "available_capacity": 0
      }
    ]
  },
  "policy_overrides": {
    "priority_tiers": {
      "sibling": 1000,
      "municipality": 800,
      "low_income": 500
    }
  },
  "include_estimated_dates": true
}
```

## Advanced Configuration Request

Request with custom scoring and performance settings:

```json
{
  "application": {...},
  "centers": [...],
  "top_k": 5,
  "min_score_threshold": 0.6,
  "matching_config": {
    "scoring_weights": {
      "preference_weight": 0.5,
      "property_weight": 0.2,
      "availability_weight": 0.2,
      "quality_weight": 0.1,
      "sibling_bonus": 0.3
    },
    "distance_weights": {
      "decay_factor": 0.08,
      "preferred_radius_km": 2.0
    },
    "performance": {
      "max_edges_per_application": 30,
      "enable_edge_pruning": true,
      "edge_pruning_threshold": 0.2,
      "parallel_scoring": true
    },
    "debug": {
      "include_timing_breakdown": true,
      "include_graph_stats": true,
      "log_level": "DEBUG"
    }
  }
}
```

## Batch Processing Request

Process multiple operations in one call:

```json
{
  "requests": [
    {
      "type": "recommend",
      "application_id": "app1",
      "parameters": {
        "top_k": 5,
        "include_explanations": true
      }
    },
    {
      "type": "waitlist",
      "center_id": "center1", 
      "parameters": {
        "group_by_age": true
      }
    },
    {
      "type": "allocate",
      "application_ids": ["app2", "app3", "app4"],
      "parameters": {
        "optimization_objective": "efficiency"
      }
    }
  ],
  "shared_centers": [...],
  "matching_config": {
    "scoring_weights": {...}
  },
  "parallel_processing": true
}
```

## Response Examples

### Recommendation Response

```json
{
  "mode": "recommend",
  "offers": [
    {
      "application_id": "550e8400-e29b-41d4-a716-446655440001",
      "center_id": "550e8400-e29b-41d4-a716-446655440401",
      "bucket_id": "550e8400-e29b-41d4-a716-446655440701",
      "rank": 1,
      "score": 0.87,
      "is_available": true,
      "explanation": {
        "distance_km": 2.3,
        "preference_score": 0.9,
        "components": {
          "organic_food": 1.0,
          "outdoor_space": 0.8,
          "distance_penalty": 0.23
        },
        "constraints_satisfied": ["age_match", "hours_overlap"],
        "reason_codes": ["preference_match", "close_distance"]
      }
    }
  ],
  "success": true,
  "processing_time_ms": 234
}
```

### Allocation Response

```json
{
  "mode": "allocate",
  "offers": [
    {
      "application_id": "app1",
      "center_id": "center1",
      "bucket_id": "bucket1",
      "score": 0.89,
      "is_allocated": true
    }
  ],
  "statistics": {
    "total_applications": 150,
    "matched_applications": 142,
    "coverage_rate": 0.947,
    "average_score": 0.73,
    "solver_status": "OPTIMAL"
  },
  "unmatched_applications": [
    {
      "application_id": "app99",
      "reasons": ["no_capacity", "distance_exceeded"]
    }
  ],
  "success": true,
  "processing_time_ms": 1847
}
```

### Waitlist Response

```json
{
  "mode": "waitlist",
  "center_id": "550e8400-e29b-41d4-a716-446655440405",
  "waitlist_entries": [
    {
      "application_id": "app1",
      "rank": 1,
      "score": 0.94,
      "tier": "sibling",
      "estimated_wait_days": 45,
      "explanation": {
        "tier_bonus": 1000,
        "base_score": 0.75
      }
    },
    {
      "application_id": "app2", 
      "rank": 2,
      "score": 0.83,
      "tier": "municipality",
      "estimated_wait_days": 67
    }
  ],
  "statistics": {
    "total_interested": 87,
    "by_tier": {
      "sibling": 12,
      "municipality": 34,
      "regular": 41
    }
  },
  "success": true,
  "processing_time_ms": 156
}
```

## Error Response Example

```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid application data",
    "details": {
      "field": "children",
      "issue": "At least one child is required",
      "provided_value": []
    }
  },
  "timestamp": "2024-01-15T10:30:00Z",
  "request_id": "550e8400-e29b-41d4-a716-446655440999"
}
```

## Integration Notes

- **Center Data**: Always provide centers in the request to limit matching scope and avoid additional API calls
- **Progressive Loading**: If centers not provided, service will progressively load from main API starting with nearest 100
- **Capacity Respect**: Set `respect_capacity: false` to include full centers for waitlist generation
- **Custom Scoring**: Use `matching_config` to adjust scoring weights for specific requirements
- **Batch Efficiency**: Use batch requests for multiple operations to reduce overhead