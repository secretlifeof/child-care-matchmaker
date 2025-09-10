# Main App Integration Guide for Semantic Matching

This guide shows how to integrate your main application with the matchmaker service's semantic matching capabilities.

## Overview

The semantic matching system enhances your existing LLM-based feature extraction with intelligent relationship discovery. When a parent mentions "garden", the system automatically finds centers with "outdoor activities", "nature access", and other related features.

## Integration Architecture

```
┌─────────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│   Main Application │───▶│   Matchmaker API     │───▶│     PostgreSQL      │
│                     │    │                      │    │                     │
│ - LLM Extraction    │    │ - Semantic Enhancement│    │ - Cached Embeddings │
│ - Center Processing │    │ - Match Scoring      │    │ - Vector Similarity │
│ - Parent Preferences│    │ - Auto Relationships │    │ - Feature Storage   │
└─────────────────────┘    └──────────────────────┘    └─────────────────────┘
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │     OpenAI API       │
                           │   (text-embedding)   │
                           └──────────────────────┘
```

## Prerequisites

### 1. Database Setup

Add these columns to your PostgreSQL database:

```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Add embedding support to ontology features
ALTER TABLE ontology.Ontology_Features 
ADD COLUMN IF NOT EXISTS embedding vector(1536),
ADD COLUMN IF NOT EXISTS embedding_model TEXT DEFAULT 'text-embedding-3-small',
ADD COLUMN IF NOT EXISTS embedding_updated_at TIMESTAMPTZ;

-- Create vector index for fast similarity search
CREATE INDEX IF NOT EXISTS idx_ontology_features_embedding 
ON ontology.Ontology_Features 
USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 100);

-- Add semantic metadata to center features
ALTER TABLE matching.Center_Features 
ADD COLUMN IF NOT EXISTS is_semantic_match BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS semantic_source_feature TEXT,
ADD COLUMN IF NOT EXISTS semantic_similarity_score FLOAT;

-- Add semantic support to parent preferences  
ALTER TABLE matching.Parent_Preferences
ADD COLUMN IF NOT EXISTS original_text TEXT,
ADD COLUMN IF NOT EXISTS embedding vector(1536),
ADD COLUMN IF NOT EXISTS embedding_updated_at TIMESTAMPTZ;
```

### 2. Environment Variables

Add to your main app's environment:

```env
# Matchmaker service URL
MATCHING_SERVICE_URL=http://localhost:8002

# OpenAI API key (same for both apps)
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. One-Time Setup

Initialize embeddings for your ontology features:

```bash
curl -X POST "${MATCHING_SERVICE_URL}/api/semantic/initialize-embeddings"
```

## Integration Steps

### Step 1: Update Center Processing

Enhance your existing center processing flow:

```typescript
// Your existing LLM extraction (keep unchanged)
async function extractCenterFeatures(centerDescription: string, ontology: any) {
  // Your current LLM prompt and extraction logic
  const prompt = `Extract features from this daycare description...`;
  const response = await openai.chat.completions.create({
    model: "gpt-4",
    messages: [{ role: "user", content: prompt }],
  });
  
  return JSON.parse(response.choices[0].message.content);
}

// Enhanced processing with semantic matching
async function processCenterWithSemantics(centerId: string, centerDescription: string) {
  // 1. Your existing LLM extraction
  const extractedFeatures = await extractCenterFeatures(centerDescription, ontology);
  
  // 2. Store in your database (existing flow)
  await storeCenterFeatures(centerId, extractedFeatures);
  
  // 3. NEW: Enhance with semantic matching
  const semanticResponse = await fetch(`${process.env.MATCHING_SERVICE_URL}/api/semantic/centers/${centerId}/features`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      features: extractedFeatures.map(f => ({
        feature_key: f.feature_key,
        value_bool: f.type === 'boolean' ? f.value : null,
        value_num: f.type === 'number' ? f.value : null,
        value_text: f.type === 'text' ? f.value : null,
        confidence: f.confidence,
        raw_phrase: f.raw_phrase
      }))
    })
  });
  
  const result = await semanticResponse.json();
  
  console.log(`✅ Processed ${result.processing_summary.explicit_features} features`);
  console.log(`🔗 Found ${result.processing_summary.semantic_enhancements} semantic matches`);
  
  return result;
}
```

### Step 2: Update Parent Preference Processing

Enhance parent preference collection:

```typescript
// Your existing LLM extraction (keep unchanged)
async function extractParentPreferences(parentInput: string, ontology: any) {
  // Your current preference extraction logic
  const prompt = `Extract preferences from parent input using categorical levels...`;
  const response = await openai.chat.completions.create({
    model: "gpt-4",
    messages: [{ role: "user", content: prompt }],
  });
  
  return JSON.parse(response.choices[0].message.content);
}

// Enhanced processing with semantic matching
async function processParentPreferencesWithSemantics(parentId: string, parentInput: string) {
  // 1. Your existing LLM extraction
  const extractedPreferences = await extractParentPreferences(parentInput, ontology);
  
  // 2. Store in your database (existing flow)
  await storeParentPreferences(parentId, extractedPreferences);
  
  // 3. NEW: Enhance with semantic matching
  const semanticResponse = await fetch(`${process.env.MATCHING_SERVICE_URL}/api/semantic/parents/${parentId}/preferences`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      original_text: parentInput,
      extracted_features: extractedPreferences.map(p => ({
        feature_key: p.feature_key,
        confidence: p.confidence,
        preference: p.preference, // "required", "preferred", "nice_to_have", "exclude"
        raw_phrase: p.raw_phrase
      }))
    })
  });
  
  const result = await semanticResponse.json();
  
  console.log(`✅ Stored ${result.matches.length} explicit preferences`);
  console.log(`🔗 Generated ${result.semantic_enhancements.length} semantic matches`);
  
  // Log discoveries for debugging
  result.semantic_enhancements.forEach(match => {
    console.log(`  - "${match.derived_from}" → "${match.feature_key}" (${match.similarity_score.toFixed(2)})`);
  });
  
  return result;
}
```

### Step 3: Update Matching Logic

Replace your current matching with semantic scoring:

```typescript
// NEW: Use semantic match scores
async function getSemanticMatchScore(parentId: string, centerId: string) {
  const response = await fetch(`${process.env.MATCHING_SERVICE_URL}/api/semantic/match-score/${parentId}/${centerId}`);
  const result = await response.json();
  
  return {
    score: result.score,  // 0.0 to 1.0
    matches: result.matches,
    totalPreferences: result.total_preferences,
    matchedPreferences: result.matched_preferences,
    explanation: {
      explicitMatches: result.matches.filter(m => !m.is_semantic).length,
      semanticMatches: result.matches.filter(m => m.is_semantic).length
    }
  };
}

// Enhanced recommendation endpoint
app.post('/api/recommendations', async (req, res) => {
  const { parentId, maxDistance = 10, limit = 20 } = req.body;
  
  try {
    // 1. Find nearby centers (your existing geo logic)
    const nearbyCenters = await findNearbyCenters(parentId, maxDistance);
    
    // 2. Get semantic match scores for all centers
    const matches = await Promise.all(
      nearbyCenters.map(async center => ({
        center,
        ...(await getSemanticMatchScore(parentId, center.id))
      }))
    );
    
    // 3. Sort by score and return top matches
    const topMatches = matches
      .sort((a, b) => b.score - a.score)
      .slice(0, limit);
    
    const recommendations = topMatches.map(match => ({
      center: match.center,
      score: match.score,
      explanation: {
        matchedPreferences: `${match.matchedPreferences}/${match.totalPreferences}`,
        semanticDiscoveries: match.matches
          .filter(m => m.is_semantic)
          .map(m => ({
            feature: m.feature_key,
            confidence: m.match_score,
            source: 'AI discovered relationship'
          }))
      }
    }));
    
    res.json({ recommendations });
    
  } catch (error) {
    console.error('Matching error:', error);
    res.status(500).json({ error: 'Matching service error' });
  }
});
```

## Feature Categorization

The system automatically handles different types of features:

### ✅ Semantic Matching (Uses AI Embeddings)
- **Space Features**: garden, playground, library, quiet_area
- **Food Features**: organic, vegetarian_options, home_cooked
- **Education Features**: montessori, waldorf, arts_integration
- **Activities**: outdoor_activities, music_program, sports

### 🚫 Hard Matching (Direct Database Queries)
- **Location Features**: distance_from_home_km, near_forest
- **Rating Features**: overall_rating, food_quality_rating
- **Capacity Features**: available_spots, age_groups

## Expected Results

### Example: Parent Input "near forest with outdoor activities"

**Your LLM extracts:**
```json
[
  {
    "feature_key": "location.near_forest",
    "preference": "nice_to_have",
    "confidence": 0.8,
    "raw_phrase": "near forest"
  },
  {
    "feature_key": "outdoor_activities", 
    "preference": "preferred",
    "confidence": 0.9,
    "raw_phrase": "outdoor activities"
  }
]
```

**Semantic service adds:**
```json
[
  {
    "feature_key": "forest_kindergarten",
    "preference": "preferred",
    "confidence": 0.68,
    "similarity_score": 0.85,
    "derived_from": "outdoor_activities"
  },
  {
    "feature_key": "space.garden",
    "preference": "nice_to_have", 
    "confidence": 0.56,
    "similarity_score": 0.70,
    "derived_from": "outdoor_activities"
  }
]
```

### Example: Center with "Waldkindergarten mit großem Garten"

**Your LLM extracts:**
```json
[
  {
    "feature_key": "forest_kindergarten",
    "confidence": 0.95,
    "raw_phrase": "Waldkindergarten"
  },
  {
    "feature_key": "space.garden",
    "confidence": 0.88,
    "raw_phrase": "großem Garten"
  }
]
```

**Semantic service adds:**
```json
[
  {
    "feature_key": "outdoor_activities",
    "confidence": 0.72,
    "similarity_score": 0.72,
    "source_feature": "forest_kindergarten"
  },
  {
    "feature_key": "nature_access",
    "confidence": 0.68,
    "similarity_score": 0.68, 
    "source_feature": "space.garden"
  }
]
```

**Final Match Score: ~0.85** 🎯

## Testing Your Integration

### 1. Test Center Processing
```bash
curl -X POST "${MATCHING_SERVICE_URL}/api/semantic/centers/test-center/features" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [
      {
        "feature_key": "space.garden",
        "value_bool": true,
        "confidence": 0.92,
        "raw_phrase": "schöner Garten"
      }
    ]
  }'
```

### 2. Test Parent Preferences
```bash
curl -X POST "${MATCHING_SERVICE_URL}/api/semantic/parents/test-parent/preferences" \
  -H "Content-Type: application/json" \
  -d '{
    "original_text": "we want outdoor activities",
    "extracted_features": [
      {
        "feature_key": "outdoor_activities",
        "confidence": 0.9,
        "preference": "preferred",
        "raw_phrase": "outdoor activities"
      }
    ]
  }'
```

### 3. Test Match Scoring
```bash
curl "${MATCHING_SERVICE_URL}/api/semantic/match-score/test-parent/test-center"
```

## Error Handling

Add proper error handling for semantic service calls:

```typescript
async function callSemanticService(url: string, data: any) {
  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
      timeout: 10000 // 10 second timeout
    });
    
    if (!response.ok) {
      throw new Error(`Semantic service error: ${response.status}`);
    }
    
    return await response.json();
    
  } catch (error) {
    console.error('Semantic service unavailable:', error);
    
    // Graceful degradation - continue without semantic enhancement
    return {
      processing_summary: { explicit_features: data.features?.length || 0, semantic_enhancements: 0 },
      semantic_matches: []
    };
  }
}
```

## Performance Considerations

### 1. Async Processing
Process semantic enhancement asynchronously for better user experience:

```typescript
// Process immediately for user feedback
const extractedFeatures = await extractCenterFeatures(centerDescription);
await storeCenterFeatures(centerId, extractedFeatures);

// Enhance semantically in background
setTimeout(async () => {
  try {
    await processCenterWithSemantics(centerId, extractedFeatures);
  } catch (error) {
    console.error('Background semantic processing failed:', error);
  }
}, 100);
```

### 2. Caching
Cache match scores for frequently requested parent-center pairs:

```typescript
const matchScoreCache = new Map();

async function getCachedMatchScore(parentId: string, centerId: string) {
  const cacheKey = `${parentId}:${centerId}`;
  
  if (matchScoreCache.has(cacheKey)) {
    return matchScoreCache.get(cacheKey);
  }
  
  const score = await getSemanticMatchScore(parentId, centerId);
  matchScoreCache.set(cacheKey, score);
  
  // Cache for 1 hour
  setTimeout(() => matchScoreCache.delete(cacheKey), 3600000);
  
  return score;
}
```

## Monitoring and Debugging

### 1. Log Semantic Discoveries
```typescript
function logSemanticMatches(result: any, type: 'center' | 'parent') {
  const enhancements = type === 'center' ? 
    result.semantic_matches : 
    result.semantic_enhancements;
    
  enhancements.forEach(match => {
    console.log(`🔗 Semantic: "${match.source_feature || match.derived_from}" → "${match.feature_key}" (${match.similarity_score?.toFixed(2)})`);
  });
}
```

### 2. Track Service Health
```typescript
async function checkSemanticServiceHealth() {
  try {
    const response = await fetch(`${process.env.MATCHING_SERVICE_URL}/health`);
    const health = await response.json();
    
    if (health.features?.semantic_matching) {
      console.log('✅ Semantic matching service healthy');
    } else {
      console.warn('⚠️ Semantic matching not available');
    }
    
    return health;
  } catch (error) {
    console.error('❌ Semantic service unreachable:', error);
    return { healthy: false };
  }
}
```

## Migration Strategy

### Phase 1: Parallel Operation
- Keep your existing matching logic
- Add semantic enhancement calls
- Compare results and tune thresholds

### Phase 2: Gradual Adoption  
- Start using semantic scores for new centers/parents
- Backfill existing data with semantic enhancements
- Monitor match quality improvements

### Phase 3: Full Integration
- Replace old matching with semantic scoring
- Remove redundant feature relationship logic
- Optimize performance with caching

## Cost Optimization

### OpenAI Usage
- **Embeddings cached** in database (one-time cost)
- **No runtime API calls** for matching (uses cached vectors)
- **Typical cost**: $1-5/month for 1000 features + 10k preferences

### Database Optimization
```sql
-- Monitor embedding storage
SELECT 
    pg_size_pretty(pg_total_relation_size('ontology.ontology_features')) as ontology_size,
    pg_size_pretty(pg_total_relation_size('matching.center_features')) as center_features_size;

-- Check vector index usage
EXPLAIN (ANALYZE, BUFFERS) 
SELECT feature_key FROM ontology.Ontology_Features 
WHERE embedding <=> '[0.1,0.2,...]'::vector < 0.4;
```

This integration enables intelligent matching without manual feature relationship setup, while maintaining backward compatibility with your existing LLM extraction pipeline.