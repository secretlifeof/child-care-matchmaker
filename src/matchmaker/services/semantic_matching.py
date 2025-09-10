"""Semantic matching using OpenAI embeddings."""

import logging
import openai
import asyncio
from typing import List, Dict, Optional, Any, Tuple
from uuid import UUID
import asyncpg
import numpy as np

logger = logging.getLogger(__name__)

class SemanticMatchingService:
    """Handles embedding-based semantic matching between preferences and features."""
    
    def __init__(self, connection_pool: asyncpg.Pool, openai_api_key: str):
        self.pool = connection_pool
        self.openai_client = openai.AsyncOpenAI(api_key=openai_api_key)
        
    async def get_embedding(self, text: str) -> List[float]:
        """Get OpenAI embedding for text."""
        try:
            response = await self.openai_client.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting embedding for '{text[:50]}...': {e}")
            # Return zero vector as fallback
            return [0.0] * 1536
    
    async def initialize_ontology_embeddings(self) -> None:
        """One-time setup: Create embeddings for all ontology features."""
        logger.info("🔧 Building embeddings cache for ontology features...")
        
        async with self.pool.acquire() as conn:
            # Check if embedding column exists
            embedding_column_exists = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.columns 
                    WHERE table_name = 'ontology_features' 
                    AND column_name = 'embedding'
                    AND table_schema = 'ontology'
                )
            """)
            
            if not embedding_column_exists:
                logger.warning("Embedding column not found in ontology.Ontology_Features, skipping semantic initialization")
                return
            
            # Get all features without embeddings
            features = await conn.fetch("""
                SELECT feature_key, title, description 
                FROM ontology.Ontology_Features 
                WHERE is_active = true 
                AND (embedding IS NULL OR embedding_updated_at < updated_at)
            """)
            
            # Debug logging
            total_features = await conn.fetchval("SELECT COUNT(*) FROM ontology.Ontology_Features WHERE is_active = true")
            features_with_embeddings = await conn.fetchval("SELECT COUNT(*) FROM ontology.Ontology_Features WHERE is_active = true AND embedding IS NOT NULL")
            logger.info(f"Found {total_features} total active features, {features_with_embeddings} with embeddings, {len(features)} need processing")
            
            if not features:
                logger.info("All embeddings are up to date")
                return
            
            for i, feature in enumerate(features):
                try:
                    # Create natural language text for embedding (title + description works better than technical keys)
                    embedding_text = feature['title']
                    if feature['description']:
                        embedding_text += f" - {feature['description']}"
                    
                    # Get embedding
                    embedding = await self.get_embedding(embedding_text)
                    
                    # Try to store embedding, fall back if pgvector not available
                    try:
                        # Convert list to PostgreSQL vector format
                        vector_str = str(embedding)
                        await conn.execute("""
                            UPDATE ontology.Ontology_Features 
                            SET embedding = $1::vector, 
                                embedding_model = 'text-embedding-3-small',
                                embedding_updated_at = NOW()
                            WHERE feature_key = $2
                        """, vector_str, feature['feature_key'])
                    except Exception as db_error:
                        if "vector" in str(db_error).lower():
                            logger.warning(f"pgvector not available, skipping embedding storage. Install pgvector extension to enable semantic matching.")
                            return  # Skip remaining features
                        else:
                            raise  # Re-raise if it's a different error
                    
                    if i % 10 == 0:
                        logger.info(f"Processed {i}/{len(features)} features")
                        
                except Exception as e:
                    logger.error(f"Failed to process embedding for {feature['feature_key']}: {e}")
                    # Continue with other features
                    continue
        
        logger.info("Ontology embeddings cache complete!")
    
    async def find_semantic_matches(
        self, 
        query_text: str, 
        similarity_threshold: float = 0.3,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Find semantically similar features using vector search."""
        
        # Auto-initialize embeddings if missing
        await self._ensure_embeddings_exist()
        
        # Get query embedding
        query_embedding = await self.get_embedding(query_text)
        query_embedding_str = str(query_embedding)
        
        async with self.pool.acquire() as conn:
            matches = await conn.fetch("""
                SELECT 
                    feature_key,
                    title,
                    category_key,
                    value_type,
                    description,
                    1 - (embedding <=> $1::vector) as similarity_score
                FROM ontology.Ontology_Features 
                WHERE embedding IS NOT NULL
                  AND 1 - (embedding <=> $1::vector) > $2
                ORDER BY embedding <=> $1::vector
                LIMIT $3
            """, query_embedding_str, similarity_threshold, limit)
            
            return [dict(match) for match in matches]
    
    async def _ensure_embeddings_exist(self):
        """Ensure embeddings exist for ontology features, create if missing."""
        async with self.pool.acquire() as conn:
            # Check if we have any embeddings
            count = await conn.fetchval("""
                SELECT COUNT(*) FROM ontology.Ontology_Features 
                WHERE embedding IS NOT NULL AND is_active = true
            """)
            
            if count == 0:
                logger.info("No embeddings found, auto-initializing...")
                await self.initialize_ontology_embeddings()
            else:
                # Check for any missing embeddings and create them
                missing = await conn.fetch("""
                    SELECT feature_key, title, description 
                    FROM ontology.Ontology_Features 
                    WHERE is_active = true 
                    AND (embedding IS NULL OR embedding_updated_at < updated_at)
                    LIMIT 50
                """)
                
                if missing:
                    logger.info(f"Updating {len(missing)} missing/stale embeddings...")
                    for feature in missing:
                        embedding_text = feature['title']
                        if feature['description']:
                            embedding_text += f" - {feature['description']}"
                        
                        embedding = await self.get_embedding(embedding_text)
                        
                        # Convert list to PostgreSQL vector format
                        vector_str = str(embedding)
                        await conn.execute("""
                            UPDATE ontology.Ontology_Features 
                            SET embedding = $1::vector, 
                                embedding_model = 'text-embedding-3-small',
                                embedding_updated_at = NOW()
                            WHERE feature_key = $2
                        """, vector_str, feature['feature_key'])
    
    async def process_parent_preference(
        self,
        parent_id: UUID,
        original_text: str,
        extracted_features: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Process parent preference and find semantic matches."""
        
        results = []
        
        # For each extracted feature, find semantic matches
        for extracted in extracted_features:
            feature_key = extracted['feature_key']
            confidence = extracted['confidence']
            preference_level = extracted['preference']
            raw_phrase = extracted.get('raw_phrase', '')
            
            # Store the explicitly extracted preference
            await self._store_parent_preference(
                parent_id=parent_id,
                feature_key=feature_key,
                preference=preference_level,
                confidence=confidence,
                original_text=original_text,
                raw_phrase=raw_phrase,
                is_semantic_match=False
            )
            
            results.append({
                'feature_key': feature_key,
                'match_type': 'explicit',
                'confidence': confidence,
                'similarity_score': 1.0,
                'raw_phrase': raw_phrase
            })
            
            # Only do semantic matching for semantic-appropriate features
            # Skip hard data like locations, ratings, ratios
            if feature_key.startswith(('location.', 'rating.', 'ratio.')):
                continue
            
            # Use the raw phrase for semantic matching if available
            search_text = raw_phrase if raw_phrase else f"{feature_key} {original_text}"
            
            # Find semantic matches
            semantic_matches = await self.find_semantic_matches(
                query_text=search_text,
                similarity_threshold=0.4
            )
            
            for match in semantic_matches:
                # Skip if it's the same as explicitly extracted
                if match['feature_key'] == feature_key:
                    continue
                
                # Skip location/rating features for semantic matching
                if match['feature_key'].startswith(('location.', 'rating.', 'ratio.')):
                    continue
                
                # Calculate combined confidence
                semantic_confidence = confidence * match['similarity_score'] * 0.7
                
                # Only store high-confidence semantic matches
                if semantic_confidence > 0.2:
                    await self._store_parent_preference(
                        parent_id=parent_id,
                        feature_key=match['feature_key'],
                        preference=preference_level,
                        confidence=semantic_confidence,
                        original_text=original_text,
                        raw_phrase=f"semantic from: {raw_phrase}",
                        is_semantic_match=True,
                        semantic_source=feature_key,
                        similarity_score=match['similarity_score']
                    )
                    
                    results.append({
                        'feature_key': match['feature_key'],
                        'match_type': 'semantic',
                        'confidence': semantic_confidence,
                        'similarity_score': match['similarity_score'],
                        'derived_from': feature_key,
                        'source_phrase': raw_phrase
                    })
        
        return results
    
    async def enhance_center_with_semantic_features(
        self, 
        center_id: UUID
    ) -> List[Dict[str, Any]]:
        """Enhance center features with semantic matches."""
        
        results = []
        
        async with self.pool.acquire() as conn:
            # Get existing center features that are semantic-matchable
            # Skip location, rating, ratio features that should be hard-coded checks
            existing_features = await conn.fetch("""
                SELECT feature_key, value_bool, value_num, raw_phrase, confidence
                FROM matching.Center_Features
                WHERE center_id = $1 
                AND is_semantic_match = FALSE
                AND NOT feature_key LIKE 'location.%'
                AND NOT feature_key LIKE 'rating.%'
                AND NOT feature_key LIKE 'ratio.%'
                AND raw_phrase IS NOT NULL
            """, center_id)
            
            
            for feature in existing_features:
                # Use just the raw phrase for semantic matching
                search_text = feature['raw_phrase']
                
                # Find semantic matches
                semantic_matches = await self.find_semantic_matches(
                    query_text=search_text,
                    similarity_threshold=0.3
                )
                
                for match in semantic_matches:
                    # Skip self-matches
                    if match['feature_key'] == feature['feature_key']:
                        continue
                    
                    # Skip location/rating features for semantic matching
                    if (match['feature_key'].startswith(('location.', 'rating.', 'ratio.'))):
                        continue
                    
                    # Calculate confidence for semantic feature
                    semantic_confidence = feature['confidence'] * match['similarity_score'] * 0.75
                    
                    if semantic_confidence > 0.3:
                        # Check if semantic feature already exists
                        existing = await conn.fetchrow("""
                            SELECT id, confidence, semantic_similarity_score 
                            FROM matching.Center_Features 
                            WHERE center_id = $1 AND feature_key = $2
                        """, center_id, match['feature_key'])
                        
                        if existing:
                            # Update if new confidence or similarity is higher
                            if semantic_confidence > existing['confidence'] or match['similarity_score'] > (existing['semantic_similarity_score'] or 0):
                                await conn.execute("""
                                    UPDATE matching.Center_Features 
                                    SET confidence = GREATEST(confidence, $1),
                                        semantic_similarity_score = GREATEST(
                                            COALESCE(semantic_similarity_score, 0), $2
                                        ),
                                        is_semantic_match = true,
                                        semantic_source_feature = $3,
                                        updated_at = NOW()
                                    WHERE id = $4
                                """, semantic_confidence, match['similarity_score'], 
                                feature['feature_key'], existing['id'])
                        else:
                            # Insert new semantic feature
                            await conn.execute("""
                                INSERT INTO matching.Center_Features (
                                    center_id, feature_key, value_bool, confidence,
                                    source, is_semantic_match, semantic_source_feature,
                                    semantic_similarity_score, created_at, updated_at
                                ) VALUES ($1, $2, $3, $4, 'semantic', true, $5, $6, NOW(), NOW())
                            """, 
                            center_id, match['feature_key'], feature['value_bool'],
                            semantic_confidence, feature['feature_key'], match['similarity_score'])
                        
                        results.append({
                            'feature_key': match['feature_key'],
                            'source_feature': feature['feature_key'],
                            'source_phrase': feature['raw_phrase'],
                            'similarity_score': match['similarity_score'],
                            'confidence': semantic_confidence
                        })
        
        return results
    
    async def _store_parent_preference(
        self,
        parent_id: UUID,
        feature_key: str,
        preference: str,
        confidence: float,
        original_text: str,
        raw_phrase: str,
        is_semantic_match: bool = False,
        semantic_source: Optional[str] = None,
        similarity_score: Optional[float] = None
    ):
        """Store parent preference with semantic metadata."""
        
        async with self.pool.acquire() as conn:
            # Get feature info
            feature_info = await conn.fetchrow("""
                SELECT value_type FROM ontology.Ontology_Features 
                WHERE feature_key = $1
            """, feature_key)
            
            if not feature_info:
                logger.warning(f"Feature {feature_key} not found in ontology")
                return
            
            # Determine value based on type
            value_bool = None
            value_num = None
            value_text = None
            
            if feature_info['value_type'] == 'boolean':
                value_bool = True  # Assume positive preference
            elif feature_info['value_type'] == 'text':
                value_text = original_text
            
            # Create embedding only for the raw phrase (more efficient)
            phrase_embedding = await self.get_embedding(raw_phrase) if raw_phrase else None
            phrase_embedding_str = str(phrase_embedding) if phrase_embedding else None
            
            # Check if preference already exists
            existing = await conn.fetchrow("""
                SELECT id, confidence FROM matching.Parent_Preferences
                WHERE parent_id = $1 AND feature_key = $2
            """, parent_id, feature_key)
            
            if existing:
                # Update if new confidence is higher
                if confidence > existing['confidence']:
                    await conn.execute("""
                        UPDATE matching.Parent_Preferences
                        SET confidence = $1,
                            preference = $2,
                            value_bool = $3,
                            value_num = $4,
                            value_text = $5,
                            source_text = $6,
                            original_text = COALESCE(original_text, $7),
                            embedding = COALESCE(embedding, $8::vector),
                            embedding_updated_at = CASE 
                                WHEN embedding IS NULL AND $8 IS NOT NULL THEN NOW()
                                ELSE embedding_updated_at
                            END,
                            updated_at = NOW()
                        WHERE id = $9
                    """, confidence, preference, value_bool, value_num, value_text,
                    f"semantic from {semantic_source}: {raw_phrase}" if is_semantic_match else raw_phrase,
                    original_text, phrase_embedding_str, existing['id'])
            else:
                # Insert new preference
                await conn.execute("""
                    INSERT INTO matching.Parent_Preferences (
                        parent_id, feature_key, value_type, value_bool, value_num, 
                        value_text, preference, confidence, source_text,
                        original_text, embedding, embedding_updated_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11::vector, NOW())
                """, 
                parent_id, feature_key, feature_info['value_type'], 
                value_bool, value_num, value_text, preference, confidence,
                f"semantic from {semantic_source}: {raw_phrase}" if is_semantic_match else raw_phrase,
                original_text, phrase_embedding_str)
    
    async def calculate_match_score(
        self,
        parent_id: UUID,
        center_id: UUID
    ) -> Dict[str, Any]:
        """Calculate semantic match score between parent and center."""
        
        async with self.pool.acquire() as conn:
            # Get parent preferences
            preferences = await conn.fetch("""
                SELECT feature_key, preference, confidence, original_text
                FROM matching.Parent_Preferences 
                WHERE parent_id = $1
            """, parent_id)
            
            # Get center features
            features = await conn.fetch("""
                SELECT feature_key, confidence, is_semantic_match, 
                       semantic_similarity_score
                FROM matching.Center_Features 
                WHERE center_id = $1
            """, center_id)
            
            if not preferences or not features:
                return {'score': 0.0, 'matches': [], 'total_preferences': len(preferences)}
            
            # Create lookup for center features
            center_feature_map = {f['feature_key']: f for f in features}
            
            matches = []
            total_score = 0.0
            total_weight = 0.0
            
            # Weight mapping for preference levels
            preference_weights = {
                'required': 1.0,
                'preferred': 0.8, 
                'nice_to_have': 0.5,
                'exclude': -1.0
            }
            
            for pref in preferences:
                pref_weight = preference_weights.get(pref['preference'], 0.5)
                
                if pref['feature_key'] in center_feature_map:
                    center_feature = center_feature_map[pref['feature_key']]
                    
                    # Calculate match score
                    base_score = min(pref['confidence'], center_feature['confidence'])
                    
                    # Reduce score for semantic matches
                    if center_feature['is_semantic_match']:
                        semantic_factor = center_feature['semantic_similarity_score'] or 0.7
                        base_score *= semantic_factor
                    
                    # Apply preference weight
                    weighted_score = base_score * abs(pref_weight)
                    if pref['preference'] == 'exclude':
                        weighted_score *= -1
                    
                    total_score += weighted_score
                    total_weight += abs(pref_weight)
                    
                    matches.append({
                        'feature_key': pref['feature_key'],
                        'preference_level': pref['preference'],
                        'match_score': base_score,
                        'weighted_score': weighted_score,
                        'is_semantic': center_feature['is_semantic_match']
                    })
                else:
                    # No match found, apply penalty for required features
                    if pref['preference'] == 'required':
                        total_score -= pref['confidence']
                        total_weight += 1.0
            
            # Calculate final normalized score
            final_score = total_score / total_weight if total_weight > 0 else 0.0
            final_score = max(0.0, min(1.0, final_score))  # Clamp to [0,1]
            
            return {
                'score': final_score,
                'matches': matches,
                'total_preferences': len(preferences),
                'matched_preferences': len(matches),
                'raw_score': total_score,
                'total_weight': total_weight
            }