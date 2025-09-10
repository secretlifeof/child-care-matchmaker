# Parent-Daycare Matchmaker Service

A graph-based matching service using NetworkX and OR-Tools for optimal parent-daycare center matching with complex constraints and preferences.

## Overview

This service implements a sophisticated matching system that operates in three modes:
- **Recommend Mode**: Parent-centric recommendations based on preferences
- **Allocate Mode**: Global optimal allocation respecting capacity constraints  
- **Waitlist Mode**: Center-centric waitlist ranking with policy tiers

## Architecture

The service uses a **graph-based matching architecture** where:
- **Nodes**: Applications (parents) and capacity buckets (center slots)
- **Edges**: Weighted connections representing match quality
- **Algorithms**: Min-cost flow, Hungarian algorithm, and custom scoring

### Key Components

```
src/matchmaker/
├── graph/           # Graph construction and matching algorithms
│   ├── builder.py   # Builds bipartite matching graph
│   └── matcher.py   # Implements matching modes (recommend/allocate/waitlist)
├── scoring/         # Scoring and weight calculation
│   └── composite_scorer.py  # Multi-factor scoring system
├── data/           # Data access and loading
│   └── loaders/    
│       └── progressive.py  # Progressive loading strategy
├── models/         # Data models
│   ├── base.py     # Core domain models
│   └── results.py  # Result and response models
└── api/            # REST API endpoints
    └── routes/
        └── matches.py  # Matching endpoints
```

## Features

### Semantic Matching with AI Embeddings
- **OpenAI Integration**: Uses text-embedding-3-small for feature similarity
- **Automatic Relationship Discovery**: No manual setup of feature relationships 
- **Smart Categorization**: Separates semantic matching (descriptions) from hard matching (location/ratings)
- **Multi-language Support**: Works with German, English, and other languages
- **Cost Optimized**: ~$1-5/month for typical usage with cached embeddings

### Progressive Loading
- Starts with nearest 100 centers
- Expands search radius until sufficient quality matches found
- Reduces API calls and memory usage

### Constraint Handling
- **Hard Constraints** (filters):
  - Age compatibility
  - Distance limits (PostGIS spatial queries)
  - Opening hours overlap
  - Must-have requirements
  - Exclusions
- **Soft Constraints** (scoring):
  - Preference satisfaction (0-1 continuous)
  - Property matching with semantic enhancement
  - Quality indicators
  - Distance penalties

### Intelligent Feature Extraction
- **Categorical Preferences**: Uses "required", "preferred", "nice_to_have", "exclude" 
- **LLM-Friendly**: Better than numeric thresholds for AI extraction
- **Automatic Weights**: Converts categories to scoring weights internally
- **Confidence Scoring**: All extractions include confidence levels

### Sibling Co-Assignment
- Bonus scoring for keeping siblings together
- Post-allocation adjustment in global matching
- Family-aware waitlist ranking

### Policy Compliance
- Reserved capacity buckets (municipality, low-income, siblings)
- Tiered waitlist priorities
- Deterministic tie-breaking

## API Endpoints

### Core Matching Endpoints

#### Get Recommendations
```http
POST /api/matches/recommend
{
  "application_id": "uuid",
  "top_k": 10,
  "include_full": false,
  "max_distance_km": 5.0
}
```

#### Global Allocation
```http
POST /api/matches/allocate
{
  "application_ids": ["uuid1", "uuid2"],
  "respect_capacity": true,
  "prioritize_siblings": true,
  "seed": 42
}
```

#### Generate Waitlist
```http
POST /api/matches/waitlist
{
  "center_id": "uuid",
  "bucket_id": "uuid",  // optional
  "include_all": false
}
```

### Semantic Matching Endpoints

#### Process Center Features
```http
POST /api/semantic/centers/{center_id}/features
{
  "features": [
    {
      "feature_key": "space.garden",
      "value_bool": true,
      "confidence": 0.92,
      "raw_phrase": "Unser Garten mit altem Baumbestand"
    }
  ]
}
```

#### Process Parent Preferences
```http
POST /api/semantic/parents/{parent_id}/preferences
{
  "original_text": "near forest with outdoor activities",
  "extracted_features": [
    {
      "feature_key": "location.near_forest",
      "confidence": 0.8,
      "preference": "nice_to_have",
      "raw_phrase": "near forest"
    }
  ]
}
```

#### Get Match Score
```http
GET /api/semantic/match-score/{parent_id}/{center_id}
```

#### Search Features
```http
GET /api/semantic/search-features?query=garden&limit=5
```

#### Initialize Embeddings (One-time Setup)
```http
POST /api/semantic/initialize-embeddings
```

## Quick Start

### 1. Using Docker (Recommended)
```bash
# Clone and start the service
git clone <repository>
cd child-care-matchmaker
docker-compose --profile dev up --build
```

### 2. Set up PostgreSQL with pgvector
```sql
-- In your PostgreSQL database
CREATE EXTENSION IF NOT EXISTS vector;

-- Add embedding support to ontology features
ALTER TABLE ontology.Ontology_Features 
ADD COLUMN IF NOT EXISTS embedding vector(1536),
ADD COLUMN IF NOT EXISTS embedding_model TEXT DEFAULT 'text-embedding-3-small',
ADD COLUMN IF NOT EXISTS embedding_updated_at TIMESTAMPTZ;

-- Create vector index for fast similarity search
CREATE INDEX idx_ontology_features_embedding 
ON ontology.Ontology_Features 
USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 100);
```

### 3. Initialize Semantic Matching
```bash
# One-time setup: Create embeddings for your ontology features
curl -X POST "http://localhost:8002/api/semantic/initialize-embeddings"
```

### 4. Test the Integration
```bash
# Search for similar features
curl "http://localhost:8002/api/semantic/search-features?query=garden&limit=5"

# Check service health
curl "http://localhost:8002/health"
```

## Installation

### Using Docker
```bash
docker build -t matchmaker-service .
docker run -p 8001:8001 matchmaker-service
```

### Local Development
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run service
python -m src.matchmaker.main
```

## Configuration

### Environment Variables
```env
# API Configuration
API_BASE_URL=http://localhost:8000
MAX_CENTERS_PER_REQUEST=1000

# Matching Parameters
INITIAL_BATCH_SIZE=100
EXPANSION_FACTOR=2.0
MIN_QUALITY_THRESHOLD=0.7
TARGET_MATCHES=10

# Scoring Weights
PREFERENCE_WEIGHT=0.4
PROPERTY_WEIGHT=0.3
AVAILABILITY_WEIGHT=0.2
QUALITY_WEIGHT=0.1

# Distance
DISTANCE_DECAY_FACTOR=0.1

# OpenAI Integration (for semantic matching)
OPENAI_API_KEY=your_openai_api_key_here
EMBEDDING_MODEL_ID=text-embedding-3-small

# Database (PostgreSQL with pgvector extension)
DATABASE_URL=postgresql://user:pass@localhost:5432/database

# Graph Database (optional for advanced relationships)
GRAPH_DB_TYPE=neo4j  # Options: "tigergraph" or "neo4j"
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

## Data Models

### Key Models

**Application**: Parent application with children, preferences, and constraints
```python
{
  "id": "uuid",
  "family_id": "uuid",
  "children": [...],
  "home_location": {...},
  "preferences": [...],
  "desired_start_date": "2024-09-01",
  "max_distance_km": 5.0,
  "priority_flags": ["sibling", "low_income"]
}
```

**Center**: Daycare center with properties and capacity
```python
{
  "id": "uuid",
  "name": "Happy Kids Daycare",
  "location": {...},
  "properties": [...],
  "capacity_buckets": [...],
  "opening_hours": [...]
}
```

**ParentPreference**: Continuous scoring (0-1) instead of hard categories
```python
{
  "property_key": "organic_food",
  "operator": "equals",
  "value_boolean": true,
  "weight": 0.8,      # Importance
  "threshold": 0.9    # Min acceptable score (>0.9 = must-have)
}
```

## Algorithms

### Min-Cost Flow (Global Allocation)
- Source → Applications (capacity 1) → Buckets (capacity N) → Sink
- Edge costs = negative weights (maximize weight = minimize cost)
- OR-Tools SimpleMinCostFlow solver

### Top-K Ranking (Recommendations)
- Build edges for single application
- Sort by composite weight
- Return top K matches

### Policy-Tiered Sorting (Waitlist)
- Apply policy tier bonuses
- Sort by adjusted weight
- Maintain stable ordering

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_edge_cases.py -v

# Run with coverage
pytest --cov=src/matchmaker --cov-report=html
```

## Performance Optimization

1. **Pre-pruning**: Filter edges before graph construction
2. **Progressive Loading**: Load centers incrementally
3. **Edge Capping**: Limit edges per application (default: 50)
4. **Caching**: Redis-based result caching (optional)
5. **Parallel Processing**: Async operations for data loading

## Future Enhancements

### Machine Learning Integration
- Train acceptance prediction model
- Personalized scoring based on historical data
- Retention prediction

### Advanced Features
- Real-time updates via WebSocket
- Multi-objective optimization
- Stable marriage algorithm option
- Explanation generation for decisions

## License

MIT License - See LICENSE file for details

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Support

For issues and questions, please use the GitHub issue tracker.