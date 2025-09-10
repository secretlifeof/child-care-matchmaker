"""API routes for complex preference matching."""

import logging
import os
import time
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ...data.loaders.progressive import ProgressiveLoader
from ...database import get_database_manager
from ...database.complex_queries import ComplexPreferenceRepository
from ...graph.builder import MatchingGraphBuilder
from ...graph.matcher import GraphMatcher
from ...graph.tigergraph_client import TigerGraphClient
from ...models.base import Application, Center, MatchMode
from ...models.requests import RecommendationRequest
from ...models.results import MatchResult
from ...processing.complex_processors import ComplexPreferenceProcessorFactory
from ...scoring.composite_scorer import CompositeScorer
from ...scoring.explainable_scorer import ExplainableCompositeScorer
from ...services.matching import Matcher
from ...utils.filters import HardConstraintFilter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/matches", tags=["matching"])


class MatchOffer(BaseModel):
    """Match offer with complex preference details."""
    center_id: UUID
    center_name: str
    score: float = Field(ge=0.0, le=1.0)
    preference_matches: dict[str, Any] = Field(default_factory=dict)
    center_details: dict[str, Any] | None = None


class MatchResponse(BaseModel):
    """Response with complex preference processing details."""
    offers: list[MatchOffer]
    processing_details: dict[str, Any] = Field(default_factory=dict)
    success: bool = True
    message: str | None = None


class AllocationRequest(BaseModel):
    """Request for global allocation."""
    application_ids: list[UUID]
    center_ids: list[UUID] | None = None
    respect_capacity: bool = Field(default=True)
    prioritize_siblings: bool = Field(default=True)
    max_centers_per_app: int = Field(default=50)
    seed: int | None = None


class WaitlistRequest(BaseModel):
    """Request for waitlist generation."""
    center_id: UUID
    bucket_id: UUID | None = None
    include_all: bool = Field(default=False)


class BatchMatchRequest(BaseModel):
    """Request for batch matching."""
    applications: list[Application]
    centers: list[Center]
    mode: MatchMode
    params: dict = Field(default_factory=dict)


# Dependency injection for services
def get_scorer() -> CompositeScorer:
    """Get configured scorer instance."""
    return CompositeScorer(
        preference_weight=0.4,
        property_weight=0.3,
        availability_weight=0.2,
        quality_weight=0.1
    )


def get_tigergraph_client() -> TigerGraphClient | None:
    """Get TigerGraph client if configured."""
    host = os.getenv('TIGERGRAPH_HOST')
    username = os.getenv('TIGERGRAPH_USERNAME')
    password = os.getenv('TIGERGRAPH_PASSWORD')

    if host and username and password:
        return TigerGraphClient(
            host=host,
            username=username,
            password=password,
            graph_name=os.getenv('TIGERGRAPH_GRAPH_NAME', 'childcare')
        )
    return None


def get_explainable_scorer(
    tigergraph: TigerGraphClient | None = Depends(get_tigergraph_client)
) -> ExplainableCompositeScorer:
    """Get explainable scorer with optional TigerGraph integration."""
    return ExplainableCompositeScorer(
        preference_weight=0.4,
        property_weight=0.3,
        availability_weight=0.2,
        quality_weight=0.1,
        tigergraph_client=tigergraph
    )


async def get_matcher_service() -> Matcher:
    """Get matcher with database integration."""
    db_manager = await get_database_manager()
    return Matcher(db_manager=db_manager)


def get_filter() -> HardConstraintFilter:
    """Get configured filter instance."""
    return HardConstraintFilter()


def get_graph_builder(
    scorer: CompositeScorer = Depends(get_scorer),
    filter: HardConstraintFilter = Depends(get_filter)
) -> MatchingGraphBuilder:
    """Get configured graph builder."""
    return MatchingGraphBuilder(
        scorer=scorer,
        filter=filter,
        max_candidates_per_application=50,
        distance_decay_factor=0.1
    )


def get_matcher(seed: int | None = None) -> GraphMatcher:
    """Get configured matcher instance."""
    return GraphMatcher(seed=seed)


def get_loader() -> ProgressiveLoader:
    """Get configured progressive loader."""
    return ProgressiveLoader(
        initial_batch=100,
        expansion_factor=2.0,
        max_centers=1000,
        min_quality_threshold=0.7,
        target_matches=10
    )


async def get_complex_repository() -> ComplexPreferenceRepository:
    """Get complex preference repository dependency."""
    db_manager = await get_database_manager()
    return ComplexPreferenceRepository(db_manager.pool)


@router.post("/recommend", response_model=MatchResponse)
async def get_recommendations(
    request: RecommendationRequest,
    repository: ComplexPreferenceRepository = Depends(get_complex_repository)
) -> MatchResponse:
    """
    Get recommendations with complex preference support.
    
    This endpoint:
    - Loads parent preferences including complex types from database
    - Uses schema registry for complex preference validation
    - Performs spatial queries for location-based constraints
    - Returns detailed explanations for all matches
    """
    start_time = time.time()

    try:
        # Get parent_id from request
        parent_id = request.parent_id or request.get_application_id()

        # Load parent preferences from database
        preferences = await repository.get_parent_preferences(parent_id)
        if not preferences:
            return MatchResponse(
                offers=[],
                success=True,
                message="No preferences found for parent",
                processing_details={
                    "centers_evaluated": 0,
                    "complex_types_processed": [],
                    "processing_time_ms": int((time.time() - start_time) * 1000)
                }
            )

        # Extract complex preferences and their types
        complex_types_used = set()
        location_constraints = []

        for pref in preferences:
            if pref.complex_value_type_id and pref.value_data:
                if 'locations' in pref.value_data:
                    complex_types_used.add('location_distance')
                    location_constraints.append({
                        'type': 'location_distance',
                        'locations': pref.value_data['locations']
                    })
                elif 'start_time' in pref.value_data:
                    complex_types_used.add('schedule_range')
                elif 'connection_type' in pref.value_data:
                    complex_types_used.add('social_connection')
                elif 'approaches' in pref.value_data:
                    complex_types_used.add('educational_approach')
                elif 'start_location' in pref.value_data:
                    complex_types_used.add('route_preference')

        # Discover centers using complex spatial constraints
        center_ids = await repository.discover_centers_with_complex_constraints(
            parent_id=parent_id,
            max_distance_km=request.max_distance_km or 10.0,
            location_constraints=location_constraints if location_constraints else None,
            limit=request.limit * 3  # Get more candidates for better selection
        )

        if not center_ids:
            return MatchResponse(
                offers=[],
                success=True,
                message="No centers found within constraints",
                processing_details={
                    "centers_evaluated": 0,
                    "complex_types_processed": list(complex_types_used),
                    "processing_time_ms": int((time.time() - start_time) * 1000)
                }
            )

        # Load full center details
        centers = await repository.get_centers_with_features(center_ids)

        # Initialize complex preference processor and scorer
        db_manager = await get_database_manager()
        processor_factory = ComplexPreferenceProcessorFactory(db_manager)
        scorer = CompositeScorer(
            spatial_weight=0.3,
            preference_weight=0.4,
            quality_weight=0.3,
            db_manager=db_manager
        )

        # Score each center
        offers = []
        for center in centers:
            simple_satisfied = 0
            simple_total = 0
            complex_matches = {}
            overall_score = 0.0
            total_weight = 0.0

            for pref in preferences:
                if pref.complex_value_type_id and pref.value_data:
                    # Process complex preference
                    score, explanation = processor_factory.process_preference(pref, center)

                    # Determine complex type name
                    type_name = 'unknown'
                    if 'locations' in pref.value_data:
                        type_name = 'location_distance'
                    elif 'start_time' in pref.value_data:
                        type_name = 'schedule_range'
                    elif 'connection_type' in pref.value_data:
                        type_name = 'social_connection'
                    elif 'approaches' in pref.value_data:
                        type_name = 'educational_approach'
                    elif 'start_location' in pref.value_data:
                        type_name = 'route_preference'

                    complex_matches[type_name] = {
                        "satisfied": score > 0.5,
                        "details": explanation,
                        "score": score
                    }

                    weight = abs(pref.computed_weight) if hasattr(pref, 'computed_weight') else 0.5
                    overall_score += score * weight
                    total_weight += weight
                else:
                    # Simple preference scoring
                    if hasattr(pref, 'strength'):
                        if pref.strength in ['REQUIRED', 'PREFERRED']:
                            simple_total += 1
                            if center.has_property(pref.property_key):
                                simple_satisfied += 1
                                score = 1.0
                            else:
                                score = 0.3 if pref.strength == 'PREFERRED' else 0.0

                            weight = abs(pref.computed_weight) if hasattr(pref, 'computed_weight') else 0.5
                            overall_score += score * weight
                            total_weight += weight

            # Normalize overall score
            if total_weight > 0:
                final_score = overall_score / total_weight
            else:
                final_score = 0.5

            # Create offer
            offer = MatchOffer(
                center_id=center.id,
                center_name=center.name,
                score=min(1.0, max(0.0, final_score)),
                preference_matches={
                    "simple_preferences": {
                        "satisfied": simple_satisfied,
                        "total": simple_total
                    },
                    "complex_preferences": complex_matches
                },
                center_details={
                    "location": {
                        "latitude": center.location.latitude,
                        "longitude": center.location.longitude,
                        "address": center.location.address
                    },
                    "description": center.short_description
                } if request.include_full_centers else None
            )

            offers.append(offer)

        # Sort by score and limit results
        offers.sort(key=lambda x: x.score, reverse=True)
        offers = offers[:request.limit]

        # Build processing details
        processing_details = {
            "centers_evaluated": len(centers),
            "complex_types_processed": list(complex_types_used),
            "processing_time_ms": int((time.time() - start_time) * 1000)
        }

        logger.info(
            f"Recommendations: {len(offers)} matches for parent {parent_id} "
            f"with {len(complex_types_used)} complex types in {processing_details['processing_time_ms']}ms"
        )

        return MatchResponse(
            offers=offers,
            success=True,
            processing_details=processing_details
        )

    except Exception as e:
        logger.error(f"Error in recommendations: {str(e)}")
        return MatchResponse(
            offers=[],
            success=False,
            message=f"Processing error: {str(e)}",
            processing_details={
                "centers_evaluated": 0,
                "complex_types_processed": [],
                "processing_time_ms": int((time.time() - start_time) * 1000)
            }
        )


@router.post("/allocate", response_model=MatchResult)
async def allocate_globally(
    request: AllocationRequest,
    loader: ProgressiveLoader = Depends(get_loader),
    builder: MatchingGraphBuilder = Depends(get_graph_builder)
) -> MatchResult:
    """
    Perform global optimal allocation of applications to centers.
    
    This uses min-cost flow to find the best overall assignment
    respecting capacity constraints.
    """
    start_time = time.time()

    try:
        # Create matcher with seed for reproducibility
        matcher = GraphMatcher(seed=request.seed)

        # Load applications
        applications = await loader.load_applications(request.application_ids)
        if not applications:
            raise HTTPException(status_code=404, detail="No applications found")

        # Load centers
        if request.center_ids:
            centers = await loader.load_centers(request.center_ids)
        else:
            # Load centers near applications
            centers = await loader.load_centers_for_applications(
                applications,
                max_per_app=request.max_centers_per_app
            )

        if not centers:
            raise HTTPException(status_code=404, detail="No centers found")

        # Build graph
        graph = builder.build_graph(
            applications=applications,
            centers=centers,
            respect_capacity=request.respect_capacity
        )

        # Perform allocation
        result = matcher.match(
            graph,
            mode=MatchMode.ALLOCATE,
            respect_capacity=request.respect_capacity,
            prioritize_siblings=request.prioritize_siblings
        )

        # Add processing time
        result.processing_time_ms = int((time.time() - start_time) * 1000)

        logger.info(
            f"Allocated {result.matched_applications}/{result.total_applications} "
            f"applications in {result.processing_time_ms}ms"
        )

        return result

    except Exception as e:
        logger.error(f"Error in global allocation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/waitlist", response_model=MatchResult)
async def generate_waitlist(
    request: WaitlistRequest,
    loader: ProgressiveLoader = Depends(get_loader),
    builder: MatchingGraphBuilder = Depends(get_graph_builder),
    matcher: GraphMatcher = Depends(get_matcher)
) -> MatchResult:
    """
    Generate waitlist ranking for a specific center.
    
    Orders all interested applications by match quality and policy tiers.
    """
    start_time = time.time()

    try:
        # Load center
        center = await loader.load_center(request.center_id)
        if not center:
            raise HTTPException(status_code=404, detail="Center not found")

        # Load applications interested in this center
        applications = await loader.load_applications_for_center(
            center_id=request.center_id,
            include_all=request.include_all
        )

        if not applications:
            return MatchResult(
                mode=MatchMode.WAITLIST,
                success=True,
                message="No applications found for this center",
                waitlist_entries=[]
            )

        # Build graph with just this center
        graph = builder.build_graph(
            applications=applications,
            centers=[center],
            respect_capacity=False  # Include all for waitlist
        )

        # Define policy tiers (would come from configuration)
        policy_tiers = {
            "sibling": 1000,  # Highest priority
            "local_district": 500,
            "low_income": 300,
            "special_needs": 200
        }

        # Generate waitlist
        result = matcher.match(
            graph,
            mode=MatchMode.WAITLIST,
            center_id=request.center_id,
            bucket_id=request.bucket_id,
            policy_tiers=policy_tiers
        )

        # Add processing time
        result.processing_time_ms = int((time.time() - start_time) * 1000)

        logger.info(
            f"Generated waitlist with {len(result.waitlist_entries)} entries "
            f"for center {request.center_id} in {result.processing_time_ms}ms"
        )

        return result

    except Exception as e:
        logger.error(f"Error generating waitlist: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch", response_model=MatchResult)
async def batch_match(
    request: BatchMatchRequest,
    builder: MatchingGraphBuilder = Depends(get_graph_builder)
) -> MatchResult:
    """
    Perform batch matching with provided data.
    
    Useful for testing and bulk operations.
    """
    start_time = time.time()

    try:
        # Create matcher
        matcher = GraphMatcher(seed=request.params.get("seed"))

        # Build graph
        graph = builder.build_graph(
            applications=request.applications,
            centers=request.centers,
            respect_capacity=request.params.get("respect_capacity", True)
        )

        # Perform matching based on mode
        result = matcher.match(
            graph,
            mode=request.mode,
            **request.params
        )

        # Add processing time
        result.processing_time_ms = int((time.time() - start_time) * 1000)

        return result

    except Exception as e:
        logger.error(f"Error in batch matching: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))



@router.get("/complex-types")
async def get_complex_value_types(
    repository: ComplexPreferenceRepository = Depends(get_complex_repository)
):
    """Get all available complex value types from the schema registry."""
    try:
        complex_types = await repository.get_complex_value_types()

        return {
            "complex_types": [
                {
                    "type_name": ct.type_name,
                    "description": ct.description,
                    "schema": ct.schema,
                    "examples": ct.examples,
                    "version": ct.version
                }
                for ct in complex_types
            ],
            "count": len(complex_types)
        }

    except Exception as e:
        logger.error(f"Error loading complex types: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_statistics():
    """Get matching service statistics."""
    return {
        "status": "healthy",
        "version": "2.1.0",  # Updated version with complex preferences
        "features": {
            "complex_preferences": True,
            "schema_registry": True,
            "spatial_queries": True,
            "detailed_explanations": True,
            "database_integration": True
        },
        "complex_types_supported": [
            "location_distance",
            "schedule_range",
            "social_connection",
            "educational_approach",
            "route_preference"
        ],
        "endpoints": {
            "recommend": "Generate personalized recommendations with complex preferences",
            "allocate": "Global optimal allocation",
            "waitlist": "Center-specific waitlist",
            "batch": "Batch processing",
            "complex-types": "Get available complex preference types"
        },
        "api_changes": {
            "parameter_changes": {
                "top_k": "replaced with 'limit'",
                "parent_id": "now primary identifier for database lookup"
            }
        }
    }
