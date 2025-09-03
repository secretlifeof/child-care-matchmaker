"""API routes for matching operations."""

from typing import List, Optional
from uuid import UUID
import time
import logging

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field

from ...models.base import MatchMode, Application, Center
from ...models.results import MatchResult
from ...models.requests import (
    RecommendationRequest, AllocationRequest,
    WaitlistRequest, BatchMatchRequest, MatchingConfig,
    MatchRequest
)
from ...graph.builder import MatchingGraphBuilder
from ...graph.matcher import GraphMatcher
from ...scoring.composite_scorer import CompositeScorer
from ...scoring.explainable_scorer import ExplainableCompositeScorer
from ...graph.tigergraph_client import TigerGraphClient
from ...utils.filters import HardConstraintFilter
from ...data.loaders.progressive import ProgressiveLoader
from ...services.matching import Matcher, MatchRequest
from ...services.graph import get_graph_client
from ...database import get_database_manager, DatabaseManager
import os

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


def get_tigergraph_client() -> Optional[TigerGraphClient]:
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
    tigergraph: Optional[TigerGraphClient] = Depends(get_tigergraph_client)
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
    request: MatchRequest,
    matcher_service: Matcher = Depends(get_matcher_service)
) -> MatchResult:
    """
    Get recommendations using capacity checking and graph-based feature matching.
    
    This endpoint provides:
    - Capacity availability checking using existing waitlist tables
    - Graph database feature matching (TigerGraph or Neo4j)
    - Comprehensive scoring with detailed explanations
    - Distance-based filtering using PostGIS
    """
    start_time = time.time()
    
    try:
        # Validate database connection
        if not matcher_service.db_manager.is_initialized:
            raise HTTPException(
                status_code=503, 
                detail="Database connection not available."
            )
        
        result = await matcher_service.match_with_full_context(request)
        
        # Add processing time
        result.processing_time_ms = int((time.time() - start_time) * 1000)
        
        logger.info(
            f"Recommendations generated {len(result.offers)} matches "
            f"for parent {request.parent_id} in {result.processing_time_ms}ms"
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
    from ...services.graph.factory import GraphClientFactory, GraphDBType
    
    # Get graph client info
    graph_type = os.getenv('GRAPH_DB_TYPE', 'tigergraph')
    supported_graphs = GraphClientFactory.get_supported_types()
    is_valid, missing_vars = GraphClientFactory.validate_environment()
    
    return {
        "status": "healthy",
        "version": "2.0.0",  # Updated version
        "graph_database": {
            "type": graph_type,
            "supported_types": supported_graphs,
            "environment_valid": is_valid,
            "missing_variables": missing_vars if not is_valid else []
        },
        "endpoints": {
            "recommend": "Generate personalized recommendations with capacity + graph matching", 
            "allocate": "Global optimal allocation",
            "waitlist": "Center-specific waitlist",
            "batch": "Batch processing"
        },
        "features": {
            "graph_databases": ["TigerGraph", "Neo4j"],
            "capacity_checking": True,
            "semantic_matching": True,
            "explainable_results": True,
            "distance_filtering": True
        }
    }