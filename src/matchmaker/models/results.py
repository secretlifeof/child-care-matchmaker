"""Result models for matching operations."""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from .base import MatchMode


class MatchReason(BaseModel):
    """Individual reason for a match score."""
    category: str  # "absolute_match", "preference_match", "semantic_match", "policy_bonus"
    property_key: str
    explanation: str
    score_contribution: float
    confidence: float = Field(default=1.0)
    source: str = Field(default="direct")  # "direct", "semantic", "conceptnet", "embedding"


class MatchExplanation(BaseModel):
    """Comprehensive explanation of match scoring."""
    total_score: float
    distance_km: float = Field(default=0)
    reasons: list[MatchReason] = Field(default_factory=list)
    met_absolutes: list[str] = Field(default_factory=list)  # Required preferences satisfied
    unmet_preferences: list[str] = Field(default_factory=list)  # Preferred but not available
    policy_bonuses: dict[str, float] = Field(default_factory=dict)  # Sibling, income, etc.
    semantic_matches: list[str] = Field(default_factory=list)  # Properties found via semantic search

    # Legacy fields for backward compatibility
    preference_score: float = Field(default=0)
    components: dict[str, float] = Field(default_factory=dict)
    constraints_satisfied: list[str] = Field(default_factory=list)
    constraints_violated: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)


class MatchOffer(BaseModel):
    """An enhanced offer for a match between application and center."""
    application_id: UUID
    center_id: UUID
    bucket_id: UUID
    rank: int | None = None
    score: float = Field(..., ge=0.0)
    normalized_score: float = Field(default=0.0, ge=0.0, le=1.0)  # 0-1 normalized
    is_available: bool = True
    is_allocated: bool = False
    is_tentative: bool = False
    explanation: MatchExplanation | None = None
    match_quality: str = Field(default="good")  # "excellent", "good", "fair", "poor"
    expires_at: datetime | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class WaitlistEntry(BaseModel):
    """Enhanced entry in a center's waitlist."""
    center_id: UUID
    bucket_id: UUID
    application_id: UUID
    rank: int
    score: float = Field(..., ge=0.0)
    adjusted_score: float = Field(default=0.0)  # Score after policy adjustments
    tier: str = Field(default="standard")
    policy_tier: str = Field(default="general")  # "sibling", "low_income", "municipality", "general"
    explanation: MatchExplanation | None = None
    estimated_wait_days: int | None = None
    estimated_position_change: int | None = None  # Expected position change per month
    created_at: datetime = Field(default_factory=datetime.utcnow)


class MatchRun(BaseModel):
    """Record of a matching run."""
    id: UUID
    mode: MatchMode
    seed: int | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
    status: str = Field(default="pending")
    statistics: dict[str, Any] = Field(default_factory=dict)


class MatchResult(BaseModel):
    """Result of a matching operation."""
    mode: MatchMode
    run_id: UUID | None = None
    offers: list[MatchOffer] = Field(default_factory=list)
    waitlist_entries: list[WaitlistEntry] = Field(default_factory=list)
    success: bool = True
    message: str | None = None
    total_applications: int | None = None
    matched_applications: int | None = None
    coverage_rate: float | None = None
    statistics: dict[str, Any] = Field(default_factory=dict)
    processing_time_ms: int | None = None
