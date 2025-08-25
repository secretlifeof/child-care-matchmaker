# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Parent-Daycare Matchmaker Service - A graph-based matching service using NetworkX and OR-Tools for optimal parent-daycare center matching with complex constraints and preferences.

## Development Commands

### Docker
```bash
# Build and run service
docker build -t matchmaker-service .
docker run -p 8001:8001 matchmaker-service

# Development with auto-reload
docker build --target development -t matchmaker-dev .
docker run -p 8001:8001 -v $(pwd)/src:/app/src matchmaker-dev
```

### Current endpoints (called from where you are)

Main service endpoint: http://localhost:8001

API endpoints:
- `POST /api/matches/recommend` - Get personalized recommendations for a parent
- `POST /api/matches/allocate` - Perform global optimal allocation
- `POST /api/matches/waitlist` - Generate waitlist ranking for a center  
- `POST /api/matches/batch` - Batch matching operations

### Running the Service Locally
```bash
# Install dependencies
pip install -r requirements.txt

# Run development server
python -m src.matchmaker.main

# Or using uvicorn directly
uvicorn src.matchmaker.main:app --reload --host 0.0.0.0 --port 8001
```

### Testing
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_edge_cases.py -v

# Run with coverage
pytest --cov=src/matchmaker --cov-report=html

# Run integration tests
pytest tests/test_integration.py -v
```

## Architecture Overview

The service uses a **graph-based matching architecture**:

1. **Graph Layer** (`src/matchmaker/graph/`)
   - `builder.py`: Constructs bipartite matching graph (applications ↔ capacity buckets)
   - `matcher.py`: Implements three matching modes (recommend, allocate, waitlist)

2. **Scoring System** (`src/matchmaker/scoring/`)
   - `composite_scorer.py`: Multi-factor scoring combining preferences, properties, availability, quality
   - Future: ML-based scoring for acceptance prediction

3. **Data Layer** (`src/matchmaker/data/`)
   - `loaders/progressive.py`: Progressive loading strategy (starts with nearest 100 centers)
   - `repositories/`: Data access patterns (future implementation)

4. **Models** (`src/matchmaker/models/`)
   - `base.py`: Core domain models (Application, Center, Preferences, etc.)
   - `results.py`: Result models (MatchOffer, WaitlistEntry, etc.)

5. **API Layer** (`src/matchmaker/api/`)
   - `routes/matches.py`: RESTful endpoints for all matching operations
   - Async FastAPI with dependency injection

6. **Utilities** (`src/matchmaker/utils/`)
   - `filters.py`: Hard constraint filtering
   - Future: caching, monitoring, etc.

## Key Technical Decisions

1. **Graph-Based Matching**: Uses NetworkX for graph representation and algorithms (Hungarian, min-cost flow via OR-Tools)

2. **Three Matching Modes**:
   - **Recommend**: Parent-centric top-K recommendations
   - **Allocate**: Global optimal allocation using min-cost flow
   - **Waitlist**: Center-centric ranking with policy tiers

3. **Progressive Loading**: Starts with 100 nearest centers, expands until sufficient quality matches found

4. **Continuous Scoring**: Uses 0-1 scoring instead of hard categories (must_have/exclude derived from thresholds)

5. **Policy Compliance**: Reserved capacity buckets and tiered priorities for fair allocation

## Data Models

### Key Models

**Application**: Parent's application with children and preferences
```python
{
  "children": [Child],
  "preferences": [ParentPreference],
  "home_location": Location,
  "priority_flags": ["sibling", "low_income"],
  "max_distance_km": 5.0
}
```

**ParentPreference**: Continuous scoring approach
```python
{
  "property_key": "organic_food",
  "operator": "equals", 
  "value_boolean": true,
  "weight": 0.8,        # Importance (0-1)
  "threshold": 0.9      # Min acceptable (>0.9 = must_have, <0.1 = exclude)
}
```

**Center**: Daycare with properties and capacity buckets
```python
{
  "properties": [CenterProperty],
  "capacity_buckets": [CapacityBucket],  # (center × group × age_band × start_month)
  "opening_hours": [TimeSlot]
}
```

## Algorithm Details

### Recommend Mode (Top-K)
1. Build edges from application to all valid buckets
2. Score each edge using CompositeScorer
3. Sort by weight descending, return top K

### Allocate Mode (Min-Cost Flow) 
1. Source → Applications (capacity=1) → Buckets (capacity=N) → Sink
2. Edge costs = -weight (maximize total weight)
3. OR-Tools SimpleMinCostFlow solver
4. Handle sibling co-assignment with post-processing

### Waitlist Mode (Policy Tiers + Scoring)
1. Find all applications interested in center
2. Apply policy tier bonuses (sibling=1000, low_income=500, etc.)
3. Sort by adjusted weight
4. Return ranked list with explanations

## Common Development Tasks

### Adding New Matching Constraints
1. Update hard filters in `src/matchmaker/utils/filters.py`
2. Or add to scoring in `src/matchmaker/scoring/composite_scorer.py`
3. Add property types to `src/matchmaker/models/base.py`
4. Test with edge cases

### Adding New API Endpoints
1. Add route to `src/matchmaker/api/routes/matches.py`
2. Update main app in `src/matchmaker/main.py`
3. Add request/response models
4. Write integration tests

### Modifying Scoring Weights
1. Update `CompositeScorer` weights in constructor
2. Or make configurable via environment variables
3. Test impact on match quality

### Performance Optimization
1. Adjust progressive loading parameters
2. Prune graph edges more aggressively
3. Add Redis caching layer
4. Monitor via `/stats` endpoint

## Testing Strategy

- **Edge Cases** (`tests/test_edge_cases.py`): No matches, capacity constraints, conflicting preferences, etc.
- **Integration** (`tests/test_integration.py`): End-to-end scenarios with realistic data
- **Unit Tests**: Individual component testing
- All async tests with proper fixtures

## Configuration

Environment variables:
```env
API_BASE_URL=http://localhost:8000  # Main application API
INITIAL_BATCH_SIZE=100
MAX_CENTERS_PER_REQUEST=1000
MIN_QUALITY_THRESHOLD=0.7
TARGET_MATCHES=10

# Scoring weights
PREFERENCE_WEIGHT=0.4
PROPERTY_WEIGHT=0.3
AVAILABILITY_WEIGHT=0.2
QUALITY_WEIGHT=0.1
```

## Future Enhancements

1. **Machine Learning**: Train acceptance prediction models, integrate as edge weights
2. **Real-time Updates**: WebSocket notifications for waitlist changes
3. **Multi-objective**: Lexicographic optimization for complex policy requirements
4. **Caching**: Redis layer for computed matches and explanations
5. **Analytics**: Match success tracking, recommendation quality metrics

## Integration with Main Application

This service expects to integrate with a main application API that provides:
- `GET /api/applications/{id}` - Application details
- `GET /api/centers/nearby` - Centers by location  
- `GET /api/centers/{id}` - Center details
- `GET /api/applications/interested` - Applications for a center

The service operates as a recommendation and allocation engine, not the primary data store.