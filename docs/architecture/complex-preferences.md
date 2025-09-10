# Complex Preference System

This document describes the complex preference system implementation using a schema registry approach.

## Overview

The complex preference system allows parents to specify sophisticated constraints beyond simple property matching:

- **Location Distance**: Multi-location constraints (home, work, etc.)
- **Schedule Range**: Flexible time schedules with flexibility windows
- **Social Connection**: Sibling and family relationships
- **Educational Approach**: Pedagogical method preferences
- **Route Preference**: Commute route integration

## Architecture

### Schema Registry

Complex preferences are defined using JSON Schema in the database:

```sql
CREATE TABLE matching.Complex_Value_Types (
  id UUID PRIMARY KEY,
  type_name TEXT UNIQUE NOT NULL,
  description TEXT,
  schema JSONB NOT NULL,
  examples JSONB,
  version INTEGER DEFAULT 1,
  is_active BOOLEAN DEFAULT true
);
```

### Database Integration

Parent preferences support both simple and complex types:

```sql
ALTER TABLE matching.Parent_Preferences 
ADD COLUMN complex_value_type_id UUID REFERENCES matching.Complex_Value_Types(id),
ADD COLUMN value_data JSONB;
```

## Complex Preference Types

### 1. Location Distance

Multiple location constraints with preferred and maximum distances:

```json
{
  "type_name": "location_distance",
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

**Processing Logic**:
- Calculate distance to each location using Haversine formula
- Score based on preferred vs maximum distance
- Overall score is minimum across all locations (all must be satisfied)

### 2. Schedule Range

Flexible time schedules with day-of-week support:

```json
{
  "type_name": "schedule_range",
  "value_data": {
    "start_time": "16:00",
    "end_time": "18:00",
    "flexibility_minutes": 30,
    "days_of_week": ["monday", "tuesday", "wednesday", "thursday", "friday"]
  }
}
```

**Processing Logic**:
- Convert time strings to minutes from start of day
- Apply flexibility buffer to start/end times
- Check center operating hours for each required day
- Score is percentage of days that can be accommodated

### 3. Social Connection

Sibling and family relationship constraints:

```json
{
  "type_name": "social_connection", 
  "value_data": {
    "connection_type": "sibling",
    "entity_ids": ["550e8400-e29b-41d4-a716-446655440001"],
    "priority_level": "required",
    "same_center_required": true,
    "same_group_required": false
  }
}
```

**Processing Logic**:
- Query database for entity placements
- Check if connected entities are at the center
- Apply priority bonuses for sibling relationships
- Handle same-center vs same-group requirements

### 4. Educational Approach

Pedagogical method preferences with importance weighting:

```json
{
  "type_name": "educational_approach",
  "value_data": {
    "approaches": ["montessori", "waldorf"],
    "importance_weights": {"montessori": 0.8, "waldorf": 0.6},
    "must_have_all": false,
    "certification_required": true
  }
}
```

**Processing Logic**:
- Extract educational approaches from center properties
- Calculate weighted average of matched approaches
- Check for required certifications
- Handle must_have_all vs preferred logic

### 5. Route Preference

Commute route integration with transport modes:

```json
{
  "type_name": "route_preference",
  "value_data": {
    "start_location": {"latitude": 52.52, "longitude": 13.405},
    "end_location": {"address": "Hauptstraße 42, 10827 Berlin"},
    "max_detour_minutes": 10,
    "transport_mode": "car"
  }
}
```

**Processing Logic**:
- Calculate direct travel time between start and end
- Calculate travel time with center detour
- Compare detour time against maximum acceptable
- Score based on detour efficiency

## Processing Architecture

### Complex Preference Processors

Each complex type has a dedicated processor:

```python
class ComplexPreferenceProcessor(ABC):
    @abstractmethod
    def process(self, preference: ParentPreference, center: Center) -> Tuple[float, str]:
        """Process preference and return (score, explanation)"""
        pass
```

### Processor Factory

The factory manages processor instances and routing:

```python
class ComplexPreferenceProcessorFactory:
    def process_preference(self, preference: ParentPreference, center: Center) -> Tuple[float, str]:
        # Determine processor type from value_data structure
        # Route to appropriate processor
        # Return score and explanation
```

## Scoring Integration

Complex preferences integrate with the existing scoring system:

```python
def _score_preferences(self, application: Application, center: Center) -> float:
    total_score = 0
    total_weight = 0
    
    for pref in application.preferences:
        if pref.complex_value_type_id and pref.value_data:
            # Process complex preference
            satisfaction, _ = self.complex_processor_factory.process_preference(pref, center)
        else:
            # Process simple preference
            satisfaction = self._evaluate_preference(pref, center)
        
        weight = abs(pref.computed_weight)
        total_score += satisfaction * weight
        total_weight += weight
    
    return total_score / total_weight if total_weight > 0 else 0.5
```

## API Integration

### Request Format

Requests now use `limit` instead of `top_k` and support `parent_id` for database lookup:

```json
{
  "parent_id": "uuid",
  "max_distance_km": 10.0,
  "limit": 10,
  "include_full_centers": false,
  "include_explanations": true
}
```

### Response Format

Enhanced responses include complex preference details:

```json
{
  "offers": [
    {
      "center_id": "uuid",
      "center_name": "Happy Kids Daycare",
      "score": 0.87,
      "preference_matches": {
        "simple_preferences": {"satisfied": 8, "total": 10},
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
      }
    }
  ],
  "processing_details": {
    "centers_evaluated": 45,
    "complex_types_processed": ["location_distance", "schedule_range"],
    "processing_time_ms": 234
  }
}
```

## Database Queries

### Spatial Queries

Location-based complex preferences use PostGIS for efficient spatial queries:

```sql
SELECT c.id, c.name,
       ST_Distance(
           ST_GeogFromText('POINT(' || $2 || ' ' || $1 || ')'),
           ST_GeogFromText('POINT(' || c.longitude || ' ' || c.latitude || ')')
       ) / 1000 as distance_km
FROM Centers c
WHERE ST_DWithin(
    ST_GeogFromText('POINT(' || $2 || ' ' || $1 || ')'),
    ST_GeogFromText('POINT(' || c.longitude || ' ' || c.latitude || ')'),$3 * 1000
)
ORDER BY distance_km LIMIT $4
```

### Complex Preference Loading

Parent preferences are loaded with complex type information:

```sql
SELECT pp.parent_id, pp.feature_key, pp.value_type, pp.value_bool,
       pp.value_num, pp.value_text, pp.value_list, pp.unit,
       pp.preference, pp.confidence, pp.source_text,
       pp.complex_value_type_id, pp.value_data, pp.updated_at,
       cvt.type_name as complex_type_name
FROM matching.Parent_Preferences pp
LEFT JOIN matching.Complex_Value_Types cvt ON pp.complex_value_type_id = cvt.id
WHERE pp.parent_id = $1
```

## Benefits

### LLM Integration
- **Structured extraction**: JSON Schema provides clear format
- **Example-based learning**: Examples included for each complex type
- **Validation**: Invalid structures are automatically rejected
- **Graceful fallback**: Failed complex extraction falls back to simple preferences

### Maintainability
- **Centralized definitions**: All complex types in schema registry
- **Versioned evolution**: Schema versions support backwards compatibility
- **Easy extension**: New complex types added without code changes
- **Clear documentation**: Schema includes descriptions and examples

### Performance
- **Spatial indexing**: PostGIS provides efficient geographic queries  
- **Parallel processing**: Complex preferences processed independently
- **Selective loading**: Only relevant centers loaded based on constraints
- **Caching**: Database connection pooling and query result caching