"""API routes for semantic matching functionality."""

import logging
import os
from typing import List, Dict, Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from ...database import get_database_manager
from ...services.semantic_matching import SemanticMatchingService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/semantic", tags=["semantic"])


class FeatureExtraction(BaseModel):
    """Single feature extraction from center/parent text."""
    feature_key: str
    value_bool: bool | None = None
    value_num: float | None = None
    value_text: str | None = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    raw_phrase: str | None = None


class CenterFeaturesRequest(BaseModel):
    """Request to store center features with semantic enhancement."""
    features: List[FeatureExtraction]


class ParentPreferenceFeature(BaseModel):
    """Extracted feature from parent preference."""
    feature_key: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    preference: str = Field(..., pattern="^(required|preferred|nice_to_have|exclude)$")
    raw_phrase: str | None = None  # The specific phrase that led to this extraction


class ParentPreferenceRequest(BaseModel):
    """Request to store parent preferences with semantic enhancement."""
    original_text: str
    extracted_features: List[ParentPreferenceFeature]


class SemanticMatchResponse(BaseModel):
    """Response with semantic matching results."""
    matches: List[Dict[str, Any]]
    semantic_enhancements: List[Dict[str, Any]]
    processing_summary: Dict[str, Any]


class MatchScoreResponse(BaseModel):
    """Response with calculated match score."""
    score: float
    matches: List[Dict[str, Any]]
    total_preferences: int
    matched_preferences: int
    explanation: Dict[str, Any]


async def get_semantic_service() -> SemanticMatchingService:
    """Get semantic matching service instance."""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    
    db_manager = await get_database_manager()
    return SemanticMatchingService(db_manager.pool, openai_api_key)


@router.post("/centers/{center_id}/features", response_model=SemanticMatchResponse)
async def store_center_features_with_semantics(
    center_id: UUID,
    request: CenterFeaturesRequest,
    semantic_service: SemanticMatchingService = Depends(get_semantic_service)
):
    """
    Store center features and enhance with semantic matches.
    
    The main app should call this after extracting features from center descriptions.
    This will:
    1. Store the explicitly extracted features
    2. Find semantically related features
    3. Store semantic matches with reduced confidence
    
    Example:
    - Main app extracts: "space.garden" from "Unser schöner Garten..."  
    - Service finds semantic matches: "outdoor_activities", "nature_access"
    - Parent searching "near forest" will match both explicit and semantic features
    """
    try:
        # Store explicit features first
        async with semantic_service.pool.acquire() as conn:
            for feature in request.features:
                # Check if feature already exists
                existing = await conn.fetchrow("""
                    SELECT id, confidence FROM matching.Center_Features 
                    WHERE center_id = $1 AND feature_key = $2
                """, center_id, feature.feature_key)
                
                if existing:
                    # Update existing feature if new confidence is higher
                    if feature.confidence > existing['confidence']:
                        await conn.execute("""
                            UPDATE matching.Center_Features 
                            SET confidence = $1,
                                value_bool = $2,
                                value_num = $3,
                                raw_phrase = $4,
                                updated_at = NOW()
                            WHERE id = $5
                        """, feature.confidence, feature.value_bool, 
                        feature.value_num, feature.raw_phrase, existing['id'])
                else:
                    # Insert new feature
                    await conn.execute("""
                        INSERT INTO matching.Center_Features (
                            center_id, feature_key, value_bool, value_num, 
                            confidence, source, raw_phrase, created_at, updated_at
                        ) VALUES ($1, $2, $3, $4, $5, 'extraction', $6, NOW(), NOW())
                    """, 
                    center_id, feature.feature_key, feature.value_bool, 
                    feature.value_num, feature.confidence, feature.raw_phrase)
        
        # Enhance with semantic matches
        semantic_enhancements = await semantic_service.enhance_center_with_semantic_features(center_id)
        
        return SemanticMatchResponse(
            matches=[f.dict() for f in request.features],
            semantic_enhancements=semantic_enhancements,
            processing_summary={
                "explicit_features": len(request.features),
                "semantic_enhancements": len(semantic_enhancements),
                "center_id": str(center_id)
            }
        )
        
    except Exception as e:
        logger.error(f"Error processing center features: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/parents/{parent_id}/preferences")
async def store_parent_preferences_with_semantics(
    parent_id: UUID,
    request: ParentPreferenceRequest,
    semantic_service: SemanticMatchingService = Depends(get_semantic_service)
):
    """
    Store parent preferences and enhance with semantic matches.
    
    IMPORTANT: This endpoint now detects complex preferences and recommends
    using the interactive processing endpoint when user interaction may be needed.
    
    For Node.js integration, your processParentPreferences() method should:
    1. Try this endpoint first for simple preferences
    2. If complex preferences detected, use /api/preferences/process/interactive
    3. Handle the streaming responses for user interactions
    """
    try:
        # Check if any preferences might be complex types that need user interaction
        complex_features_detected = []
        for feature in request.extracted_features:
            complex_type = await detect_potential_complex_type(feature)
            if complex_type:
                complex_features_detected.append({
                    "feature_key": feature.feature_key,
                    "complex_type": complex_type,
                    "raw_phrase": feature.raw_phrase,
                    "requires_interaction": await might_require_user_input(feature, complex_type)
                })
        
        # If complex features detected that might need interaction, recommend interactive processing
        interactive_needed = any(cf["requires_interaction"] for cf in complex_features_detected)
        
        if interactive_needed:
            return {
                "success": False,
                "message": "Complex preferences detected that may require user interaction",
                "complex_features_detected": complex_features_detected,
                "recommendation": {
                    "use_endpoint": "/api/preferences/process/interactive",
                    "method": "POST", 
                    "reason": "These preferences may require additional user input (e.g., work location, schedule clarification)",
                    "example_usage": {
                        "request_body": {
                            "parent_id": str(parent_id),
                            "original_text": request.original_text,
                            "extracted_features": [f.dict() for f in request.extracted_features]
                        },
                        "response_type": "streaming (Server-Sent Events)"
                    }
                },
                "fallback_processed": False
            }
        
        # Process simple preferences with semantic matching
        results = await semantic_service.process_parent_preference(
            parent_id=parent_id,
            original_text=request.original_text,
            extracted_features=[f.dict() for f in request.extracted_features]
        )
        
        # Separate explicit and semantic results
        explicit_matches = [r for r in results if r['match_type'] == 'explicit']
        semantic_matches = [r for r in results if r['match_type'] == 'semantic']
        
        return {
            "success": True,
            "matches": explicit_matches,
            "semantic_enhancements": semantic_matches,
            "complex_features_detected": complex_features_detected,
            "processing_summary": {
                "original_text": request.original_text,
                "explicit_features": len(explicit_matches),
                "semantic_enhancements": len(semantic_matches),
                "parent_id": str(parent_id),
                "processing_type": "simple_preferences_only"
            }
        }
        
    except Exception as e:
        logger.error(f"Error processing parent preferences: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def detect_potential_complex_type(feature) -> str | None:
    """Detect if this feature might be a complex preference type."""
    phrase = feature.raw_phrase.lower() if feature.raw_phrase else ""
    
    # Location-related keywords (distance, travel time, specific locations)
    if any(word in phrase for word in ["km", "distance", "minutes", "walking", "driving", "biking", "cycling", "home", "work", "office"]):
        return "location_distance"
    
    # Schedule-related keywords  
    if any(word in phrase for word in ["time", "schedule", "pickup", "drop", "hours", "morning", "afternoon", "evening", "flexible"]):
        return "schedule_range"
    
    # Educational approach keywords
    if any(word in phrase for word in ["montessori", "waldorf", "reggio", "approach", "pedagogy", "philosophy", "method", "curriculum"]):
        return "educational_approach"
    
    # Social connection keywords
    if any(word in phrase for word in ["sibling", "brother", "sister", "friend", "family", "same center", "together"]):
        return "social_connection"
    
    return None


async def might_require_user_input(feature, complex_type: str) -> bool:
    """Check if this complex feature might require user input."""
    phrase = feature.raw_phrase.lower() if feature.raw_phrase else ""
    
    if complex_type == "location_distance":
        # Work location often needs clarification
        if any(word in phrase for word in ["work", "office", "job"]):
            return True
        # Vague references might need clarification
        if any(word in phrase for word in ["there", "that place", "where"]):
            return True
    
    elif complex_type == "schedule_range":
        # Ambiguous time references
        if any(word in phrase for word in ["flexible", "varies", "depends", "sometimes"]):
            return True
    
    elif complex_type == "social_connection":
        # References to other children without specific IDs
        if any(word in phrase for word in ["sibling", "brother", "sister", "friend"]):
            return True
    
    return False


@router.get("/match-score/{parent_id}/{center_id}", response_model=MatchScoreResponse)
async def calculate_semantic_match_score(
    parent_id: UUID,
    center_id: UUID,
    semantic_service: SemanticMatchingService = Depends(get_semantic_service)
):
    """
    Calculate semantic match score between a parent and center.
    
    This is the main matching endpoint that combines:
    1. Explicit preference matches
    2. Semantic preference matches  
    3. Explicit center features
    4. Semantic center features
    
    Returns a score from 0.0 to 1.0 with detailed explanation.
    """
    try:
        result = await semantic_service.calculate_match_score(parent_id, center_id)
        
        return MatchScoreResponse(
            score=result['score'],
            matches=result['matches'],
            total_preferences=result['total_preferences'],
            matched_preferences=result['matched_preferences'],
            explanation={
                "raw_score": result['raw_score'],
                "total_weight": result['total_weight'],
                "calculation": "score = sum(preference_weight * feature_confidence * semantic_factor) / total_weight"
            }
        )
        
    except Exception as e:
        logger.error(f"Error calculating match score: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/initialize-embeddings")
async def initialize_embeddings_cache(
    semantic_service: SemanticMatchingService = Depends(get_semantic_service)
):
    """
    Initialize embeddings cache for all ontology features.
    
    Run this once after setting up the database to create embeddings
    for all features in your ontology. This is required for semantic matching.
    
    This is a one-time operation that can take several minutes depending
    on the number of features.
    """
    try:
        await semantic_service.initialize_ontology_embeddings()
        return {"message": "Embeddings cache initialized successfully"}
    
    except Exception as e:
        logger.error(f"Error initializing embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/get-embedding")
async def get_embedding_for_text(
    text: str,
    semantic_service: SemanticMatchingService = Depends(get_semantic_service)
):
    """Get embedding vector for given text - for testing purposes."""
    try:
        embedding = await semantic_service.get_embedding(text)
        return {"text": text, "embedding": embedding, "dimensions": len(embedding)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search-features")
async def search_semantic_features(
    query: str,
    limit: int = 10,
    threshold: float = 0.3,
    semantic_service: SemanticMatchingService = Depends(get_semantic_service)
):
    """
    Search for semantically similar features in the ontology.
    
    Useful for:
    1. Testing semantic matching
    2. Exploring feature relationships  
    3. Debugging matching issues
    4. Building UIs for feature selection
    
    Example: query="garden" might return "space.garden", "outdoor_activities", "nature_access"
    """
    try:
        matches = await semantic_service.find_semantic_matches(
            query_text=query,
            similarity_threshold=threshold,
            limit=limit
        )
        
        return {
            "query": query,
            "matches": matches,
            "count": len(matches)
        }
        
    except Exception as e:
        logger.error(f"Error searching features: {e}")
        raise HTTPException(status_code=500, detail=str(e))