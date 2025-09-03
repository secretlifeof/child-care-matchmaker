"""Result models for matching operations."""

from datetime import datetime
from typing import Dict, List, Optional, Any
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
    reasons: List[MatchReason] = Field(default_factory=list)
    met_absolutes: List[str] = Field(default_factory=list)  # Required preferences satisfied
    unmet_preferences: List[str] = Field(default_factory=list)  # Preferred but not available
    policy_bonuses: Dict[str, float] = Field(default_factory=dict)  # Sibling, income, etc.
    semantic_matches: List[str] = Field(default_factory=list)  # Properties found via semantic search
    
    # Legacy fields for backward compatibility
    preference_score: float = Field(default=0)
    components: Dict[str, float] = Field(default_factory=dict)
    constraints_satisfied: List[str] = Field(default_factory=list)
    constraints_violated: List[str] = Field(default_factory=list)
    reason_codes: List[str] = Field(default_factory=list)


class MatchOffer(BaseModel):
    """An enhanced offer for a match between application and center."""
    application_id: UUID
    center_id: UUID
    bucket_id: UUID
    rank: Optional[int] = None
    score: float = Field(..., ge=0.0)
    normalized_score: float = Field(default=0.0, ge=0.0, le=1.0)  # 0-1 normalized
    is_available: bool = True
    is_allocated: bool = False
    is_tentative: bool = False
    explanation: Optional[MatchExplanation] = None
    match_quality: str = Field(default="good")  # "excellent", "good", "fair", "poor"
    expires_at: Optional[datetime] = None
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
    explanation: Optional[MatchExplanation] = None
    estimated_wait_days: Optional[int] = None
    estimated_position_change: Optional[int] = None  # Expected position change per month
    created_at: datetime = Field(default_factory=datetime.utcnow)


class MatchRun(BaseModel):
    """Record of a matching run."""
    id: UUID
    mode: MatchMode
    seed: Optional[int] = None
    params: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    status: str = Field(default="pending")
    statistics: Dict[str, Any] = Field(default_factory=dict)


class MatchResult(BaseModel):
    """Result of a matching operation."""
    mode: MatchMode
    run_id: Optional[UUID] = None
    offers: List[MatchOffer] = Field(default_factory=list)
    waitlist_entries: List[WaitlistEntry] = Field(default_factory=list)
    success: bool = True
    message: Optional[str] = None
    total_applications: Optional[int] = None
    matched_applications: Optional[int] = None
    coverage_rate: Optional[float] = None
    statistics: Dict[str, Any] = Field(default_factory=dict)
    processing_time_ms: Optional[int] = None