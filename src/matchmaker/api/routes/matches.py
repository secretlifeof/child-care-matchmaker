"""API routes for matching operations."""

from typing import List, Optional
from uuid import UUID
import time
import logging

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field

from ...models.base import MatchMode, Application, Center
from ...models.results import MatchResult
from ...graph.builder import MatchingGraphBuilder
from ...graph.matcher import GraphMatcher
from ...scoring.composite_scorer import CompositeScorer
from ...utils.filters import HardConstraintFilter
from ...data.loaders.progressive import ProgressiveLoader

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/matches", tags=["matching"])


class RecommendationRequest(BaseModel):
    """Request for recommendations."""
    application_id: UUID
    top_k: int = Field(default=10, ge=1, le=100)
    include_full: bool = Field(default=False)
    max_distance_km: Optional[float] = Field(default=None, ge=0)


class AllocationRequest(BaseModel):
    """Request for global allocation."""
    application_ids: List[UUID]
    center_ids: Optional[List[UUID]] = None
    respect_capacity: bool = Field(default=True)
    prioritize_siblings: bool = Field(default=True)
    max_centers_per_app: int = Field(default=50)
    seed: Optional[int] = None


class WaitlistRequest(BaseModel):
    """Request for waitlist generation."""
    center_id: UUID
    bucket_id: Optional[UUID] = None
    include_all: bool = Field(default=False)


class BatchMatchRequest(BaseModel):
    """Request for batch matching."""
    applications: List[Application]
    centers: List[Center]
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


def get_matcher(seed: Optional[int] = None) -> GraphMatcher:
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


@router.post("/recommend", response_model=MatchResult)
async def get_recommendations(
    request: RecommendationRequest,
    loader: ProgressiveLoader = Depends(get_loader),
    builder: MatchingGraphBuilder = Depends(get_graph_builder),
    matcher: GraphMatcher = Depends(get_matcher)
) -> MatchResult:
    """
    Get top-K center recommendations for an application.
    
    This endpoint uses progressive loading to efficiently find the best matches
    without loading all centers at once.
    """
    start_time = time.time()
    
    try:
        # Load application (would come from database)
        application = await loader.load_application(request.application_id)
        if not application:
            raise HTTPException(status_code=404, detail="Application not found")
        
        # Override max distance if provided
        if request.max_distance_km:
            application.max_distance_km = request.max_distance_km
        
        # Progressive loading of centers
        centers = await loader.load_centers_progressive(
            application=application,
            target_matches=request.top_k
        )
        
        # Build graph
        graph = builder.build_graph(
            applications=[application],
            centers=centers,
            respect_capacity=not request.include_full
        )
        
        # Prune to top candidates
        graph = builder.prune_edges(
            graph,
            keep_top_k_per_application=request.top_k * 2,  # Keep extra for filtering
            min_weight_threshold=0.1
        )
        
        # Generate recommendations
        result = matcher.match(
            graph,
            mode=MatchMode.RECOMMEND,
            application_id=request.application_id,
            top_k=request.top_k,
            include_full=request.include_full
        )
        
        # Add processing time
        result.processing_time_ms = int((time.time() - start_time) * 1000)
        
        logger.info(
            f"Generated {len(result.offers)} recommendations for "
            f"application {request.application_id} in {result.processing_time_ms}ms"
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


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


@router.get("/stats")
async def get_statistics():
    """Get matching service statistics."""
    # Would connect to monitoring/metrics service
    return {
        "status": "healthy",
        "version": "1.0.0",
        "endpoints": {
            "recommend": "Generate personalized recommendations",
            "allocate": "Global optimal allocation",
            "waitlist": "Center-specific waitlist",
            "batch": "Batch processing"
        }
    }