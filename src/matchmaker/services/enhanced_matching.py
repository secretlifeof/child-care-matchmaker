"""Enhanced matching service using existing database schema and generic graph interface."""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import date, datetime
from pydantic import BaseModel
from geopy.distance import geodesic

from ..models.base import Application, Center
from ..models.results import MatchResult, MatchOffer, MatchExplanation, MatchReason
from .graph import GraphClient, FeatureMatchResult, get_graph_client
from ..database import DatabaseManager

logger = logging.getLogger(__name__)


class ChildInfo(BaseModel):
    """Child information for matching."""
    age_months: int
    special_needs: Optional[List[str]] = None
    current_center_id: Optional[str] = None


class EnhancedMatchRequest(BaseModel):
    """Enhanced match request with capacity and temporal considerations."""
    parent_id: Optional[str] = None
    preferences: Optional[List[Dict[str, Any]]] = None
    children: List[ChildInfo] = []
    desired_start_date: Optional[date] = None
    required_hours: Optional[Dict[str, Any]] = None  # Time slot requirements
    max_price_per_month: Optional[float] = None
    home_lat: Optional[float] = None
    home_lon: Optional[float] = None
    radius_km: Optional[float] = None
    top_k: int = 50


class EnhancedMatcher:
    """Enhanced matching service with capacity checking and graph-based feature matching."""
    
    def __init__(self, db_manager: DatabaseManager, graph_client: Optional[GraphClient] = None):
        """
        Initialize enhanced matcher.
        
        Args:
            db_manager: Database manager instance
            graph_client: Graph database client (optional, will use global if None)
        """
        self.db_manager = db_manager
        self.graph_client = graph_client
    
    async def match_with_full_context(self, request: EnhancedMatchRequest) -> MatchResult:
        """
        Perform comprehensive matching with capacity, temporal, and feature considerations.
        
        Args:
            request: Enhanced match request
            
        Returns:
            Match result with explanations
        """
        start_time = datetime.utcnow()
        
        try:
            # 1. Geo prefilter using existing spatial functions
            candidates = await self._geo_prefilter(request)
            logger.info(f"Geo prefilter returned {len(candidates)} candidates")
            
            if not candidates:
                return MatchResult(
                    mode="recommend",
                    success=True,
                    message="No centers found within specified radius",
                    offers=[],
                    total_applications=1,
                    matched_applications=0,
                    processing_time_ms=int((datetime.utcnow() - start_time).total_seconds() * 1000)
                )
            
            # 2. Capacity availability filter using existing tables
            if request.children and request.desired_start_date:
                available_centers = await self._check_capacity_availability(
                    candidates, request.children, request.desired_start_date
                )
                logger.info(f"Capacity filter returned {len(available_centers)} available centers")
            else:
                available_centers = candidates
            
            if not available_centers:
                return MatchResult(
                    mode="recommend", 
                    success=True,
                    message="No centers have available capacity for requested dates",
                    offers=[],
                    total_applications=1,
                    matched_applications=0,
                    processing_time_ms=int((datetime.utcnow() - start_time).total_seconds() * 1000)
                )
            
            # 3. Graph-based feature matching
            if request.parent_id:
                graph_client = self.graph_client or await get_graph_client()
                feature_matches = await graph_client.query_feature_matches(
                    request.parent_id, available_centers
                )
                logger.info(f"Graph query returned {len(feature_matches)} feature matches")
            else:
                feature_matches = []
            
            # 4. Score and rank centers
            scored_offers = await self._score_and_rank_centers(
                request, available_centers, feature_matches
            )
            
            # 5. Limit to top K
            final_offers = scored_offers[:request.top_k]
            
            # 6. Log match run
            await self._log_match_run(request, final_offers)
            
            processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            return MatchResult(
                mode="recommend",
                success=True,
                offers=final_offers,
                total_applications=1,
                matched_applications=1 if final_offers else 0,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error in enhanced matching: {e}")
            return MatchResult(
                mode="recommend",
                success=False,
                message=f"Matching failed: {str(e)}",
                offers=[],
                total_applications=1,
                matched_applications=0,
                processing_time_ms=int((datetime.utcnow() - start_time).total_seconds() * 1000)
            )
    
    async def _geo_prefilter(self, request: EnhancedMatchRequest) -> List[str]:
        """Filter centers by geographic proximity using existing PostGIS functions."""
        if not request.home_lat or not request.home_lon:
            # If no location provided, return all centers (or use parent's stored location)
            async with self.db_manager.get_connection() as conn:
                if request.parent_id:
                    # Try to get parent's home location from profiles
                    parent_result = await conn.fetchrow("""
                        SELECT home_latitude, home_longitude 
                        FROM Profiles 
                        WHERE id = $1
                    """, request.parent_id)
                    
                    if parent_result and parent_result['home_latitude'] and parent_result['home_longitude']:
                        request.home_lat = parent_result['home_latitude']
                        request.home_lon = parent_result['home_longitude']
                    else:
                        # Return all centers if no location available
                        all_centers = await conn.fetch("SELECT id FROM Centers LIMIT 1000")
                        return [str(row['id']) for row in all_centers]
                else:
                    # No parent ID and no coordinates
                    all_centers = await conn.fetch("SELECT id FROM Centers LIMIT 1000")
                    return [str(row['id']) for row in all_centers]
        
        radius_km = request.radius_km or 10.0  # Default 10km radius
        
        async with self.db_manager.get_connection() as conn:
            # Use PostGIS to find nearby centers
            query = """
            SELECT c.id, a.latitude, a.longitude,
                   ST_Distance(
                       ST_SetSRID(ST_MakePoint(a.longitude, a.latitude), 4326),
                       ST_SetSRID(ST_MakePoint($2, $1), 4326)
                   ) * 111320 as distance_meters
            FROM Centers c
            JOIN Addresses a ON c.address_id = a.id
            WHERE ST_DWithin(
                ST_SetSRID(ST_MakePoint(a.longitude, a.latitude), 4326),
                ST_SetSRID(ST_MakePoint($2, $1), 4326),
                $3 / 111320.0  -- Convert km to degrees (approximate)
            )
            ORDER BY distance_meters
            LIMIT 500
            """
            
            results = await conn.fetch(query, request.home_lat, request.home_lon, radius_km)
            return [str(row['id']) for row in results]
    
    async def _check_capacity_availability(
        self, 
        center_ids: List[str], 
        children: List[ChildInfo],
        start_date: date
    ) -> List[str]:
        """Check capacity availability using existing waitlist and application tables."""
        if not children:
            return center_ids
        
        async with self.db_manager.get_connection() as conn:
            available_centers = []
            
            for center_id in center_ids:
                has_availability = True
                
                for child in children:
                    # Find suitable waitlist containers (age groups) for this child
                    containers = await conn.fetch("""
                        SELECT id, min_child_age_months, max_child_age_months
                        FROM Waiting_List_Containers 
                        WHERE center_id = $1 
                        AND min_child_age_months <= $2 
                        AND max_child_age_months >= $2
                    """, center_id, child.age_months)
                    
                    if not containers:
                        # No suitable age group for this child
                        has_availability = False
                        break
                    
                    # Check if any suitable container has availability
                    child_has_spot = False
                    
                    for container in containers:
                        # Count current applications/enrollments in this container
                        current_count = await conn.fetchval("""
                            SELECT COUNT(*)
                            FROM Waiting_List_Entries wle
                            JOIN Applications a ON wle.application_id = a.id
                            JOIN Application_Statuses ast ON a.status_id = ast.id
                            WHERE wle.waiting_list_container_id = $1
                            AND ast.name IN ('enrolled', 'accepted', 'waitlisted')
                        """, container['id'])
                        
                        # Get center's total capacity (simplified - using total capacity)
                        center_capacity = await conn.fetchval("""
                            SELECT total_capacity FROM Centers WHERE id = $1
                        """, center_id)
                        
                        # Very simplified capacity check 
                        # In reality, you'd want more sophisticated capacity management per age group
                        estimated_capacity_per_age_group = max(1, center_capacity // 3)  # Rough estimate
                        
                        if current_count < estimated_capacity_per_age_group:
                            child_has_spot = True
                            break
                    
                    if not child_has_spot:
                        has_availability = False
                        break
                
                if has_availability:
                    available_centers.append(center_id)
            
            return available_centers
    
    async def _score_and_rank_centers(
        self,
        request: EnhancedMatchRequest,
        center_ids: List[str],
        feature_matches: List[FeatureMatchResult]
    ) -> List[MatchOffer]:
        """Score and rank centers based on features, distance, and other factors."""
        offers = []
        
        # Group feature matches by center
        center_features = {}
        for match in feature_matches:
            center_id = match.center_id
            if center_id not in center_features:
                center_features[center_id] = []
            center_features[center_id].append(match)
        
        async with self.db_manager.get_connection() as conn:
            for center_id in center_ids:
                try:
                    # Get center basic info
                    center_info = await conn.fetchrow("""
                        SELECT c.id, c.name, a.latitude, a.longitude
                        FROM Centers c
                        JOIN Addresses a ON c.address_id = a.id  
                        WHERE c.id = $1
                    """, center_id)
                    
                    if not center_info:
                        continue
                    
                    # Calculate base score and explanations
                    score, explanation = await self._calculate_center_score(
                        request, center_info, center_features.get(center_id, [])
                    )
                    
                    # Create match offer
                    offer = MatchOffer(
                        application_id=request.parent_id or "unknown",
                        center_id=center_id,
                        bucket_id="", # Would need to determine appropriate bucket
                        score=score,
                        normalized_score=min(1.0, score),  # Normalize for display
                        explanation=explanation,
                        match_quality=self._determine_match_quality(score)
                    )
                    
                    offers.append(offer)
                    
                except Exception as e:
                    logger.error(f"Error scoring center {center_id}: {e}")
                    continue
        
        # Sort by score descending
        offers.sort(key=lambda x: x.score, reverse=True)
        return offers
    
    async def _calculate_center_score(
        self,
        request: EnhancedMatchRequest,
        center_info: Any,
        feature_matches: List[FeatureMatchResult]
    ) -> Tuple[float, MatchExplanation]:
        """Calculate comprehensive score for a center."""
        reasons = []
        total_score = 0.0
        met_absolutes = []
        unmet_preferences = []
        
        # 1. Feature matching score
        feature_score = 0.0
        feature_weight = 0.6  # 60% weight for features
        
        must_haves_satisfied = 0
        must_haves_total = 0
        nice_to_haves_satisfied = 0
        nice_to_haves_total = 0
        
        for match in feature_matches:
            if match.preference == "must":
                must_haves_total += 1
                if match.feature_satisfied:
                    must_haves_satisfied += 1
                    feature_score += 1.0 * match.center_confidence
                    met_absolutes.append(match.feature_key)
                    
                    reasons.append(MatchReason(
                        category="absolute_match",
                        property_key=match.feature_key,
                        explanation=f"Required feature satisfied: {match.feature_key}",
                        score_contribution=1.0,
                        confidence=match.center_confidence
                    ))
                else:
                    # Must-have not satisfied - major penalty
                    feature_score -= 2.0
                    reasons.append(MatchReason(
                        category="absolute_match",
                        property_key=match.feature_key,
                        explanation=f"Required feature NOT satisfied: {match.feature_key}",
                        score_contribution=-2.0,
                        confidence=1.0
                    ))
                        
            elif match.preference in ["nice_to_have", "preferred"]:
                nice_to_haves_total += 1
                if match.feature_satisfied:
                    nice_to_haves_satisfied += 1
                    weight = 0.8 if match.preference == "preferred" else 0.5
                    contribution = weight * match.center_confidence * match.similarity_score
                    feature_score += contribution
                    
                    reasons.append(MatchReason(
                        category="preference_match", 
                        property_key=match.feature_key,
                        explanation=f"Preferred feature satisfied: {match.feature_key}",
                        score_contribution=contribution,
                        confidence=match.center_confidence
                    ))
                else:
                    unmet_preferences.append(match.feature_key)
            
            elif match.preference == "exclude":
                if match.feature_satisfied:
                    # Feature we want to avoid is present - penalty
                    feature_score -= 1.5
                    reasons.append(MatchReason(
                        category="exclusion_violated",
                        property_key=match.feature_key,
                        explanation=f"Unwanted feature present: {match.feature_key}",
                        score_contribution=-1.5,
                        confidence=match.center_confidence
                    ))
        
        # If any must-haves are not satisfied, severely penalize
        if must_haves_total > 0 and must_haves_satisfied < must_haves_total:
            feature_score *= 0.1  # 90% penalty
        
        total_score += feature_score * feature_weight
        
        # 2. Distance score
        distance_score = 0.0
        distance_km = 0.0
        distance_weight = 0.2  # 20% weight for distance
        
        if request.home_lat and request.home_lon and center_info['latitude'] and center_info['longitude']:
            distance_km = geodesic(
                (request.home_lat, request.home_lon),
                (center_info['latitude'], center_info['longitude'])
            ).kilometers
            
            # Distance scoring: closer is better, exponential decay
            max_distance = request.radius_km or 10.0
            distance_score = max(0, 1.0 - (distance_km / max_distance) ** 2)
            
            total_score += distance_score * distance_weight
            
            reasons.append(MatchReason(
                category="distance",
                property_key="distance",
                explanation=f"Distance: {distance_km:.1f}km",
                score_contribution=distance_score * distance_weight,
                confidence=1.0
            ))
        
        # 3. Availability bonus (if we got this far, there's availability)
        availability_score = 1.0
        availability_weight = 0.1
        total_score += availability_score * availability_weight
        
        reasons.append(MatchReason(
            category="availability",
            property_key="capacity",
            explanation="Has available capacity",
            score_contribution=availability_score * availability_weight,
            confidence=1.0
        ))
        
        # 4. Quality baseline (placeholder - could incorporate ratings, certifications)
        quality_score = 0.5  # Neutral baseline
        quality_weight = 0.1
        total_score += quality_score * quality_weight
        
        explanation = MatchExplanation(
            total_score=total_score,
            distance_km=distance_km,
            reasons=reasons,
            met_absolutes=met_absolutes,
            unmet_preferences=unmet_preferences,
            policy_bonuses={},  # Could add sibling bonuses, etc.
            semantic_matches=[]  # Could identify semantic feature matches
        )
        
        return total_score, explanation
    
    def _determine_match_quality(self, score: float) -> str:
        """Determine match quality category from score."""
        if score >= 2.0:
            return "excellent"
        elif score >= 1.0:
            return "good"
        elif score >= 0.3:
            return "fair"
        else:
            return "poor"
    
    async def _log_match_run(self, request: EnhancedMatchRequest, offers: List[MatchOffer]):
        """Log match run to database using existing tables."""
        try:
            async with self.db_manager.get_connection() as conn:
                # Insert into match_runs
                run_id = await conn.fetchval("""
                    INSERT INTO matching.Match_Runs (parent_id, request_json, created_at)
                    VALUES ($1, $2, NOW())
                    RETURNING id
                """, request.parent_id, request.dict())
                
                # Insert results
                for rank, offer in enumerate(offers, 1):
                    await conn.execute("""
                        INSERT INTO matching.Match_Run_Results 
                        (run_id, center_id, rank, score, distance_m, reasons)
                        VALUES ($1, $2, $3, $4, $5, $6)
                    """, 
                    run_id, 
                    offer.center_id,
                    rank,
                    offer.score,
                    offer.explanation.distance_km * 1000 if offer.explanation else None,
                    offer.explanation.dict() if offer.explanation else {}
                    )
                
                logger.info(f"Logged match run {run_id} with {len(offers)} results")
                
        except Exception as e:
            logger.error(f"Error logging match run: {e}")
            # Don't fail the match if logging fails
            pass