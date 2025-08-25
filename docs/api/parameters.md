# API Parameters Reference

This document describes all available parameters for the Parent-Daycare Matchmaker service API.

## Common Request Parameters

### Application Data
Applications can be provided in two ways:
1. **By ID**: Reference existing applications in the main database
2. **Inline**: Full application data embedded in the request

```json
{
  "applications": [
    {
      "id": "uuid",
      "profile_id": "uuid", 
      "child_id": "uuid",
      "children": [...],
      "preferences": [...],
      "home_location": {...},
      "status_id": "uuid",
      "desired_start_date": "2024-09-01",
      "earliest_start_date": "2024-08-15", 
      "latest_start_date": "2024-10-01",
      "priority_flags": ["sibling", "low_income"],
      "max_distance_km": 5.0,
      "notes": "Special requirements text"
    }
  ]
}
```

### Center Data
Centers should be provided inline with the request to avoid additional API calls:

```json
{
  "centers": [
    {
      "id": "uuid",
      "name": "Happy Kids Daycare",
      "license_number": "DAY-2024-001",
      "description": "Modern daycare with outdoor space",
      "location": {
        "latitude": 52.5150,
        "longitude": 13.4100,
        "address": "Kindergarten Str. 123",
        "postal_code": "10115",
        "city": "Berlin",
        "country_code": "DE"
      },
      "total_capacity": 60,
      "properties": [...],
      "capacity_buckets": [...],
      "opening_hours": [...],
      "waiting_list_containers": [...]
    }
  ]
}
```

### Matching Configuration

#### Distance Parameters
```json
{
  "max_distance_km": 10.0,           // Overall distance limit
  "distance_weights": {
    "decay_factor": 0.1,             // How much distance penalizes score
    "preferred_radius_km": 2.0       // Distance within which no penalty
  }
}
```

#### Progressive Loading (when centers not provided)
```json
{
  "progressive_loading": {
    "initial_batch_size": 100,       // Start with N nearest centers
    "expansion_factor": 2.0,         // Multiply batch size each round
    "max_total_centers": 1000,       // Hard limit on centers to consider
    "min_quality_threshold": 0.7,    // Stop if enough good matches found
    "target_good_matches": 10        // How many quality matches we want
  }
}
```

#### Scoring Configuration
```json
{
  "scoring_weights": {
    "preference_weight": 0.4,        // Parent preferences importance
    "property_weight": 0.3,          // Center properties match
    "availability_weight": 0.2,      // Capacity availability
    "quality_weight": 0.1,           // Quality indicators
    "distance_weight": 0.15,         // Distance penalty
    "sibling_bonus": 0.2             // Bonus for sibling co-assignment
  }
}
```

#### Policy Configuration
```json
{
  "policy_settings": {
    "priority_tiers": {
      "sibling": 1000,               // Highest priority
      "municipality": 800,           
      "low_income": 500,
      "special_needs": 300,
      "staff_children": 200
    },
    "reserved_capacity": {
      "sibling_reserved_pct": 0.1,   // 10% reserved for siblings
      "municipality_reserved_pct": 0.3
    }
  }
}
```

## Endpoint-Specific Parameters

### POST /api/matches/recommend

Get personalized recommendations for a parent application.

```json
{
  "application": {...},              // Single application (required)
  "centers": [...],                  // List of centers to consider (optional)
  "top_k": 10,                       // Number of recommendations (1-100)
  "include_full_centers": false,     // Include centers at capacity
  "include_explanations": true,      // Include scoring explanations
  "force_center_ids": [],            // Always include these centers
  "exclude_center_ids": [],          // Never include these centers
  "min_score_threshold": 0.1,        // Minimum score to include
  "matching_config": {...}           // Configuration overrides
}
```

**Response Structure:**
```json
{
  "mode": "recommend",
  "offers": [
    {
      "application_id": "uuid",
      "center_id": "uuid", 
      "bucket_id": "uuid",
      "rank": 1,
      "score": 0.87,
      "is_available": true,
      "waiting_list_position": null,
      "explanation": {
        "distance_km": 2.3,
        "preference_score": 0.9,
        "components": {
          "organic_food": 1.0,
          "outdoor_space": 0.8
        }
      }
    }
  ]
}
```

### POST /api/matches/allocate

Perform global optimal allocation across multiple applications.

```json
{
  "applications": [...],             // Multiple applications (required)
  "centers": [...],                  // Available centers (optional)
  "respect_capacity": true,          // Honor capacity constraints
  "prioritize_siblings": true,       // Keep families together
  "allow_partial_families": false,   // Split siblings if necessary
  "optimization_objective": "fairness", // "efficiency" | "fairness" | "utilization"
  "max_iterations": 1000,            // Solver iterations limit
  "time_limit_seconds": 30,          // Solver time limit
  "seed": 42,                        // For reproducible results
  "matching_config": {...}
}
```

**Response Structure:**
```json
{
  "mode": "allocate",
  "offers": [...],                   // Allocated matches
  "statistics": {
    "total_applications": 150,
    "matched_applications": 142,
    "coverage_rate": 0.947,
    "average_score": 0.73,
    "solver_status": "OPTIMAL",
    "processing_time_ms": 2340
  },
  "unmatched_applications": [        // Applications without matches
    {
      "application_id": "uuid",
      "reasons": ["no_capacity", "distance_exceeded"]
    }
  ]
}
```

### POST /api/matches/waitlist

Generate ranked waitlist for a specific center.

```json
{
  "center_id": "uuid",               // Target center (required)
  "waiting_list_container_id": "uuid", // Specific age group (optional)
  "applications": [...],             // Interested applications (optional)
  "include_current_students": false, // Include already enrolled
  "policy_overrides": {              // Override default policies
    "priority_tiers": {...}
  },
  "group_by_age": true,              // Separate waitlists by age group
  "include_estimated_dates": true,   // Estimate availability dates
  "matching_config": {...}
}
```

**Response Structure:**
```json
{
  "mode": "waitlist",
  "center_id": "uuid",
  "waitlist_entries": [
    {
      "application_id": "uuid",
      "rank": 1,
      "score": 0.94,
      "tier": "sibling",
      "estimated_wait_days": 45,
      "waiting_list_container": {
        "id": "uuid",
        "name": "Toddler Program",
        "age_range": "18-36 months"
      }
    }
  ],
  "statistics": {
    "total_interested": 87,
    "by_tier": {
      "sibling": 12,
      "municipality": 34,
      "regular": 41
    }
  }
}
```

### POST /api/matches/batch

Process multiple matching requests in a single call.

```json
{
  "requests": [
    {
      "type": "recommend",
      "application_id": "uuid",
      "parameters": {...}
    },
    {
      "type": "waitlist", 
      "center_id": "uuid",
      "parameters": {...}
    }
  ],
  "centers": [...],                  // Shared center data
  "matching_config": {...},          // Global configuration
  "parallel_processing": true        // Process requests in parallel
}
```

## Data Validation Parameters

### Application Validation
```json
{
  "validation_rules": {
    "require_preferences": false,     // Must have at least one preference
    "require_children": true,         // Must have at least one child
    "max_children": 5,                // Limit children per application
    "validate_dates": true,           // Check date consistency
    "require_location": true          // Must have home location
  }
}
```

### Center Validation
```json
{
  "center_validation": {
    "require_capacity_buckets": true, // Must have capacity defined
    "require_opening_hours": true,    // Must have hours defined
    "validate_location": true,        // Must have valid coordinates
    "require_properties": false       // Must have properties defined
  }
}
```

## Advanced Parameters

### Caching Configuration
```json
{
  "caching": {
    "enable_result_caching": true,    // Cache computed matches
    "cache_ttl_seconds": 3600,        // Cache lifetime
    "cache_key_prefix": "match_",     // Redis key prefix
    "invalidate_on_capacity_change": true
  }
}
```

### Performance Tuning
```json
{
  "performance": {
    "max_edges_per_application": 50,  // Limit graph edges
    "enable_edge_pruning": true,      // Remove low-score edges
    "edge_pruning_threshold": 0.1,    // Minimum edge weight
    "parallel_scoring": true,         // Use multiple cores
    "batch_size_scoring": 100         // Score in batches
  }
}
```

### Debugging and Monitoring
```json
{
  "debug": {
    "include_graph_stats": false,     // Return graph statistics
    "include_timing_breakdown": true, // Detailed performance metrics
    "log_level": "INFO",              // DEBUG, INFO, WARN, ERROR
    "trace_requests": false           // Full request tracing
  }
}
```

## Default Values

The service uses these defaults when parameters are not specified:

```json
{
  "top_k": 10,
  "max_distance_km": 10.0,
  "respect_capacity": true,
  "prioritize_siblings": true,
  "include_explanations": true,
  "scoring_weights": {
    "preference_weight": 0.4,
    "property_weight": 0.3,
    "availability_weight": 0.2,
    "quality_weight": 0.1
  },
  "progressive_loading": {
    "initial_batch_size": 100,
    "expansion_factor": 2.0,
    "max_total_centers": 1000,
    "target_good_matches": 10
  },
  "policy_settings": {
    "priority_tiers": {
      "sibling": 1000,
      "municipality": 500,
      "low_income": 300
    }
  }
}
```

## Error Responses

All endpoints return consistent error responses:

```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid application data",
    "details": {
      "field": "children",
      "issue": "At least one child is required"
    }
  },
  "timestamp": "2024-01-15T10:30:00Z",
  "request_id": "uuid"
}
```

Common error codes:
- `VALIDATION_ERROR`: Invalid input data
- `NO_MATCHES_FOUND`: No valid matches available
- `CAPACITY_EXCEEDED`: Too many requests
- `TIMEOUT`: Processing took too long
- `INTERNAL_ERROR`: Unexpected server error