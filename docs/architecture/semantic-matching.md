# Semantic Matching Architecture

This document describes the semantic matching system that enables intelligent relationship discovery between parent preferences and daycare center features using OpenAI embeddings.

## Overview

The semantic matching system automatically finds relationships between features without manual configuration. When a parent mentions "garden", the system can automatically match centers with "outdoor activities", "nature access", and other related features.

## Architecture Components

```
┌─────────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│   Main Application │    │   Matchmaker API     │    │     PostgreSQL      │
│                     │    │                      │    │                     │
│ - Center Profiles   │───▶│ - Feature Extraction │───▶│ - pgvector Storage  │
│ - Parent Preferences│    │ - Semantic Enhancement│    │ - Vector Similarity │
│ - LLM Extraction    │    │ - Match Scoring      │    │ - Cached Embeddings │
└─────────────────────┘    └──────────────────────┘    └─────────────────────┘
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │     OpenAI API       │
                           │                      │
                           │ - text-embedding-3   │
                           │ - ~1536 dimensions   │
                           │ - Multilingual       │
                           └──────────────────────┘
```

## Core Components

### 1. SemanticMatchingService

**Location**: `src/matchmaker/services/semantic_matching.py`

**Key Methods**:
- `initialize_ontology_embeddings()`: One-time setup of feature embeddings
- `find_semantic_matches()`: Vector similarity search
- `process_parent_preference()`: Enhance parent preferences with semantic matches
- `enhance_center_with_semantic_features()`: Add semantic features to centers
- `calculate_match_score()`: Score parent-center compatibility

### 2. Database Schema

#### Ontology Features (Enhanced)
```sql
-- Existing ontology table with semantic enhancements
ALTER TABLE ontology.Ontology_Features 
ADD COLUMN embedding vector(1536),                    -- OpenAI embedding
ADD COLUMN embedding_model TEXT DEFAULT 'text-embedding-3-small',
ADD COLUMN embedding_updated_at TIMESTAMPTZ;

-- Vector similarity index
CREATE INDEX idx_ontology_features_embedding 
ON ontology.Ontology_Features 
USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 100);
```

#### Center Features (Enhanced)
```sql
-- Track semantic vs explicit matches
ALTER TABLE matching.Center_Features 
ADD COLUMN is_semantic_match BOOLEAN DEFAULT FALSE,
ADD COLUMN semantic_source_feature TEXT,             -- Which feature generated this
ADD COLUMN semantic_similarity_score FLOAT;         -- Cosine similarity score
```

#### Parent Preferences (Enhanced)
```sql
-- Store original text and phrase embeddings
ALTER TABLE matching.Parent_Preferences
ADD COLUMN original_text TEXT,                      -- Full parent input
ADD COLUMN embedding vector(1536),                  -- Phrase embedding
ADD COLUMN embedding_updated_at TIMESTAMPTZ;
```

### 3. API Routes

**Location**: `src/matchmaker/api/routes/semantic.py`

**Endpoints**:
- `POST /api/semantic/centers/{center_id}/features`: Process center features
- `POST /api/semantic/parents/{parent_id}/preferences`: Process parent preferences  
- `GET /api/semantic/match-score/{parent_id}/{center_id}`: Calculate match score
- `GET /api/semantic/search-features`: Search similar features
- `POST /api/semantic/initialize-embeddings`: One-time setup

## Data Flow

### Center Processing Flow

1. **LLM Extraction** (Main App)
   ```typescript
   const features = await extractFeatures(centerDescription, ontology);
   // Result: [{ feature_key: 'space.garden', raw_phrase: 'Garten', confidence: 0.9 }]
   ```

2. **Semantic Enhancement** (Matchmaker API)
   ```python
   # Store explicit feature
   await store_center_feature(center_id, 'space.garden', confidence=0.9)
   
   # Find semantic matches
   matches = await find_semantic_matches('Garten', threshold=0.65)
   # Results: ['outdoor_activities': 0.78, 'nature_access': 0.72]
   
   # Store semantic features
   for match in matches:
       await store_semantic_feature(center_id, match.feature_key, 
                                   source='space.garden', similarity=match.score)
   ```

3. **Database Storage**
   ```sql
   -- Explicit feature
   INSERT INTO matching.Center_Features (center_id, feature_key, confidence, is_semantic_match)
   VALUES ('center-1', 'space.garden', 0.9, FALSE);
   
   -- Semantic features
   INSERT INTO matching.Center_Features (center_id, feature_key, confidence, 
                                        is_semantic_match, semantic_source_feature, semantic_similarity_score)
   VALUES ('center-1', 'outdoor_activities', 0.67, TRUE, 'space.garden', 0.78);
   ```

### Parent Preference Flow

1. **LLM Extraction** (Main App)
   ```typescript
   const preferences = await extractPreferences(parentInput, ontology);
   // Result: [{ feature_key: 'location.near_forest', preference: 'nice_to_have', raw_phrase: 'near forest' }]
   ```

2. **Semantic Enhancement** (Matchmaker API)
   ```python
   # Store explicit preference
   await store_parent_preference(parent_id, 'location.near_forest', 'nice_to_have')
   
   # Find semantic matches (excluding location features)
   if not feature_key.startswith('location.'):
       matches = await find_semantic_matches('near forest', threshold=0.6)
       # Results: ['forest_kindergarten': 0.85, 'nature_activities': 0.72]
   ```

### Match Scoring Flow

1. **Retrieve Data**
   ```sql
   -- Get parent preferences (explicit + semantic)
   SELECT feature_key, preference, confidence FROM matching.Parent_Preferences 
   WHERE parent_id = $1;
   
   -- Get center features (explicit + semantic)
   SELECT feature_key, confidence, is_semantic_match FROM matching.Center_Features 
   WHERE center_id = $1;
   ```

2. **Score Calculation**
   ```python
   for preference in parent_preferences:
       if preference.feature_key in center_features:
           base_score = min(preference.confidence, center_feature.confidence)
           
           # Reduce score for semantic matches
           if center_feature.is_semantic_match:
               base_score *= center_feature.similarity_score
           
           # Apply preference weight
           weighted_score = base_score * preference_weights[preference.preference]
           total_score += weighted_score
   
   final_score = total_score / total_weight  # Normalize to [0,1]
   ```

## Semantic vs Hard Matching Categories

### ✅ Semantic Matching (Uses Embeddings)
Features that benefit from conceptual similarity:

- **Space Features**: `space.garden`, `space.playground`, `space.library`
- **Food Features**: `food.organic`, `food.vegetarian_options`, `food.home_cooked`
- **Education Features**: `education_approach.*` (montessori, waldorf, etc.)
- **Cultural Features**: `inclusion_culture.*` (diverse_families, multilingual)
- **Activity Features**: `outdoor_activities`, `arts_integration`, `music_program`

**Why**: These features have semantic relationships (garden ↔ outdoor activities)

### 🚫 Hard Matching (Direct Queries)
Features that require precise matching:

- **Location Features**: `location.distance_from_home_km`, `location.near_forest`
- **Rating Features**: `rating.overall`, `rating.food_quality` 
- **Ratio Features**: `ratio.staff_to_children`, `ratio.children_per_room`
- **Capacity Features**: Age groups, availability, pricing

**Why**: These are hard data that need exact geographic/numeric comparison

## Performance Optimizations

### 1. Embedding Caching
- **Storage**: Embeddings cached in PostgreSQL `vector(1536)` columns
- **Updates**: Only recalculate when feature text changes
- **Cost**: One-time OpenAI API cost per feature (~$0.0001 each)

### 2. Vector Indexing
```sql
-- IVFFlat index for fast similarity search
CREATE INDEX idx_ontology_features_embedding 
ON ontology.Ontology_Features 
USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 100);
```

### 3. Query Optimization
- **Batch Processing**: Group multiple features in single API calls
- **Threshold Filtering**: Only store semantic matches above similarity threshold
- **Feature Categorization**: Skip inappropriate features for semantic matching

### 4. Similarity Thresholds
- **Ontology Initialization**: 0.6 (moderate similarity required)
- **Center Enhancement**: 0.65 (higher quality for center features)
- **Parent Preferences**: 0.6 (more permissive for preferences)

## Cost Analysis

### OpenAI API Costs
- **Model**: text-embedding-3-small
- **Cost**: $0.02 per 1M tokens
- **Feature Size**: ~10 tokens average
- **1000 Features**: ~$0.0002 (one-time)
- **Monthly Usage**: ~$1-5 for typical deployment

### Performance Metrics
- **Embedding Generation**: ~50-100ms per feature
- **Vector Search**: ~1-5ms per query (with index)
- **Match Scoring**: ~10-20ms per parent-center pair

## Configuration

### Environment Variables
```env
OPENAI_API_KEY=your_key_here
EMBEDDING_MODEL_ID=text-embedding-3-small
```

### Tunable Parameters
```python
# In SemanticMatchingService
SIMILARITY_THRESHOLD = 0.6        # Minimum similarity for matches
SEMANTIC_CONFIDENCE_FACTOR = 0.7  # Reduce confidence for semantic matches
MAX_SEMANTIC_MATCHES = 10         # Limit semantic matches per feature
```

## Testing and Monitoring

### Health Checks
```bash
# Test embedding generation
curl "http://localhost:8002/api/semantic/search-features?query=test&limit=1"

# Check initialization status
curl -X POST "http://localhost:8002/api/semantic/initialize-embeddings"
```

### Database Queries
```sql
-- Check embedding coverage
SELECT 
    COUNT(*) as total_features,
    COUNT(embedding) as with_embeddings,
    ROUND(COUNT(embedding) * 100.0 / COUNT(*), 1) as coverage_percent
FROM ontology.Ontology_Features 
WHERE is_active = true;

-- View semantic matches for a center
SELECT cf.feature_key, cf.confidence, cf.is_semantic_match, 
       cf.semantic_source_feature, cf.semantic_similarity_score
FROM matching.Center_Features cf
WHERE center_id = 'your-center-id'
ORDER BY cf.confidence DESC;
```

## Future Enhancements

### 1. Advanced Embeddings
- **Fine-tuned Models**: Train on daycare-specific text
- **Multilingual Models**: Better German/English handling
- **Domain Adaptation**: Custom embeddings for childcare domain

### 2. Machine Learning Integration
- **Success Prediction**: Train on successful matches
- **Preference Learning**: Adapt weights based on parent behavior
- **Quality Scoring**: ML-based quality assessment

### 3. Real-time Updates
- **Incremental Updates**: Update embeddings as features change
- **Cache Warming**: Pre-calculate common semantic matches
- **Stream Processing**: Real-time preference processing

## Security Considerations

### Data Privacy
- **No PII in Embeddings**: Only feature descriptions, not personal data
- **Secure Storage**: Embeddings stored in encrypted database
- **API Key Security**: OpenAI keys in secure environment variables

### Rate Limiting
- **OpenAI Limits**: Respect API rate limits during initialization
- **Graceful Degradation**: Fall back to non-semantic matching if API unavailable
- **Error Handling**: Robust error handling for API failures

## Troubleshooting

### Common Issues

1. **Empty Search Results**
   - Check if embeddings are initialized
   - Verify ontology features exist
   - Lower similarity threshold

2. **High OpenAI Costs**
   - Ensure embeddings are cached properly
   - Check for duplicate API calls
   - Monitor token usage

3. **Slow Performance**
   - Verify vector indexes are created
   - Check PostgreSQL performance
   - Monitor embedding cache hit rate

4. **Poor Match Quality**
   - Adjust similarity thresholds
   - Review feature extraction quality
   - Check semantic vs hard matching categorization

This architecture enables intelligent, automatic relationship discovery while maintaining performance and cost efficiency.