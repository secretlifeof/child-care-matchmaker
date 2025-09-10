"""Request models with flexible parameters."""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from .base import Application, Center, MatchMode


class ScoringWeights(BaseModel):
    """Scoring weight configuration."""
    spatial_weight: float = Field(default=0.3, ge=0.0, le=1.0)  # Distance & route preferences
    preference_weight: float = Field(default=0.4, ge=0.0, le=1.0)  # User preferences
    quality_weight: float = Field(default=0.3, ge=0.0, le=1.0)  # Center quality
    availability_weight: float = Field(default=0.2, ge=0.0, le=1.0)
    property_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    distance_weight: float = Field(default=0.15, ge=0.0, le=1.0)  # Deprecated, use spatial_weight
    sibling_bonus: float = Field(default=0.2, ge=0.0, le=1.0)


class DistanceWeights(BaseModel):
    """Distance-based scoring configuration."""
    decay_factor: float = Field(default=0.1, ge=0.0, le=1.0)
    preferred_radius_km: float = Field(default=2.0, ge=0.0)


class ProgressiveLoadingConfig(BaseModel):
    """Progressive loading configuration."""
    initial_batch_size: int = Field(default=100, ge=1, le=1000)
    expansion_factor: float = Field(default=2.0, ge=1.1, le=5.0)
    max_total_centers: int = Field(default=1000, ge=1, le=10000)
    min_quality_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    target_good_matches: int = Field(default=10, ge=1, le=100)


class PolicySettings(BaseModel):
    """Policy and priority configuration."""
    priority_tiers: dict[str, int] = Field(default_factory=lambda: {
        "sibling": 1000,
        "municipality": 800,
        "staff_children": 600,
        "low_income": 500,
        "special_needs": 300
    })
    reserved_capacity: dict[str, float] = Field(default_factory=lambda: {
        "sibling_reserved_pct": 0.1,
        "municipality_reserved_pct": 0.3
    })


class PerformanceConfig(BaseModel):
    """Performance tuning configuration."""
    max_edges_per_application: int = Field(default=50, ge=1, le=200)
    enable_edge_pruning: bool = Field(default=True)
    edge_pruning_threshold: float = Field(default=0.1, ge=0.0, le=1.0)
    parallel_scoring: bool = Field(default=True)
    batch_size_scoring: int = Field(default=100, ge=1, le=1000)


class CachingConfig(BaseModel):
    """Caching configuration."""
    enable_result_caching: bool = Field(default=True)
    cache_ttl_seconds: int = Field(default=3600, ge=60, le=86400)
    cache_key_prefix: str = Field(default="match_")
    invalidate_on_capacity_change: bool = Field(default=True)


class DebugConfig(BaseModel):
    """Debug and monitoring configuration."""
    include_graph_stats: bool = Field(default=False)
    include_timing_breakdown: bool = Field(default=True)
    log_level: str = Field(default="INFO")
    trace_requests: bool = Field(default=False)


class MatchingConfig(BaseModel):
    """Complete matching configuration."""
    scoring_weights: ScoringWeights = Field(default_factory=ScoringWeights)
    distance_weights: DistanceWeights = Field(default_factory=DistanceWeights)
    progressive_loading: ProgressiveLoadingConfig = Field(default_factory=ProgressiveLoadingConfig)
    policy_settings: PolicySettings = Field(default_factory=PolicySettings)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    caching: CachingConfig = Field(default_factory=CachingConfig)
    debug: DebugConfig = Field(default_factory=DebugConfig)


class ValidationRules(BaseModel):
    """Data validation rules."""
    require_preferences: bool = Field(default=False)
    require_children: bool = Field(default=True)
    max_children: int = Field(default=5, ge=1, le=10)
    validate_dates: bool = Field(default=True)
    require_location: bool = Field(default=True)


class CenterValidation(BaseModel):
    """Center data validation."""
    require_capacity_buckets: bool = Field(default=True)
    require_opening_hours: bool = Field(default=True)
    validate_location: bool = Field(default=True)
    require_properties: bool = Field(default=False)


# Updated request models

class RecommendationRequest(BaseModel):
    """Request for recommendations."""
    # Core data - either inline or by reference
    parent_id: UUID | None = None  # Primary identifier for database lookup
    application: Application | None = None
    application_id: UUID | None = None
    centers: list[Center] | None = None

    # Recommendation parameters
    limit: int = Field(default=10, ge=1, le=100)  # Changed from top_k
    top_k: int | None = None  # Deprecated, kept for backward compatibility
    include_full_centers: bool = Field(default=False)
    include_explanations: bool = Field(default=True)
    min_score_threshold: float = Field(default=0.1, ge=0.0, le=1.0)

    # Override filters
    force_center_ids: list[UUID] = Field(default_factory=list)
    exclude_center_ids: list[UUID] = Field(default_factory=list)
    max_distance_km: float | None = Field(default=10.0, ge=0)

    # Configuration
    matching_config: MatchingConfig | None = None
    validation_rules: ValidationRules | None = None
    center_validation: CenterValidation | None = None

    def get_application_id(self) -> UUID:
        """Get the application ID from either source."""
        if self.application:
            return self.application.id
        elif self.application_id:
            return self.application_id
        else:
            raise ValueError("Either application or application_id must be provided")


class AllocationRequest(BaseModel):
    """Request for global allocation."""
    # Core data
    applications: list[Application] | None = None
    application_ids: list[UUID] | None = None
    centers: list[Center] | None = None
    center_ids: list[UUID] | None = None

    # Allocation parameters
    respect_capacity: bool = Field(default=True)
    prioritize_siblings: bool = Field(default=True)
    allow_partial_families: bool = Field(default=False)
    optimization_objective: str = Field(default="fairness")  # "efficiency" | "fairness" | "utilization"

    # Solver parameters
    max_iterations: int = Field(default=1000, ge=100, le=10000)
    time_limit_seconds: int = Field(default=30, ge=1, le=300)
    seed: int | None = None

    # Configuration
    matching_config: MatchingConfig | None = None
    validation_rules: ValidationRules | None = None


class WaitlistRequest(BaseModel):
    """Request for waitlist generation."""
    # Core data
    center_id: UUID
    center: Center | None = None
    applications: list[Application] | None = None
    waiting_list_container_id: UUID | None = None

    # Waitlist parameters
    include_current_students: bool = Field(default=False)
    group_by_age: bool = Field(default=True)
    include_estimated_dates: bool = Field(default=True)

    # Policy overrides
    policy_overrides: dict[str, Any] | None = None

    # Configuration
    matching_config: MatchingConfig | None = None


class BatchMatchRequest(BaseModel):
    """Request for batch matching operations."""
    requests: list[dict[str, Any]]
    shared_centers: list[Center] | None = None
    shared_applications: list[Application] | None = None
    matching_config: MatchingConfig | None = None
    parallel_processing: bool = Field(default=True)


class MatchRequest(BaseModel):
    """Generic match request supporting all modes."""
    mode: MatchMode

    # Universal data
    applications: list[Application] | None = None
    application_ids: list[UUID] | None = None
    centers: list[Center] | None = None
    center_ids: list[UUID] | None = None

    # Mode-specific parameters
    parameters: dict[str, Any] = Field(default_factory=dict)

    # Configuration
    matching_config: MatchingConfig | None = None
    validation_rules: ValidationRules | None = None

    # Request metadata
    request_id: UUID | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    client_info: dict[str, str] | None = None
