"""Explainable composite scorer with semantic property expansion."""

import logging
from typing import Dict, List, Optional, Tuple
from geopy.distance import geodesic

from ..models.base import (
    Application, Center, CapacityBucket,
    ParentPreference, ComparisonOperator, PreferenceStrength, PropertyType
)
from ..models.results import MatchReason, MatchExplanation
from ..graph.tigergraph_client import TigerGraphClient, SemanticPropertyExpander

logger = logging.getLogger(__name__)


class ExplainableCompositeScorer:
    """Enhanced scorer that provides detailed explanations for match scores."""
    
    def __init__(
        self,
        preference_weight: float = 0.4,
        property_weight: float = 0.3,
        availability_weight: float = 0.2,
        quality_weight: float = 0.1,
        tigergraph_client: Optional[TigerGraphClient] = None
    ):
        self.weights = {
            "preference": preference_weight,
            "property": property_weight,
            "availability": availability_weight,
            "quality": quality_weight
        }
        
        # Normalize weights
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v/total for k, v in self.weights.items()}
        
        self.tg_client = tigergraph_client
        self.semantic_expander = (
            SemanticPropertyExpander(tigergraph_client) if tigergraph_client else None
        )
    
    async def score_with_explanation(
        self,
        application: Application,
        center: Center,
        bucket: CapacityBucket
    ) -> Tuple[float, MatchExplanation]:
        """
        Calculate score with detailed explanation.
        
        Returns:
            Tuple of (score, explanation)
        """
        reasons = []
        met_absolutes = []
        unmet_preferences = []
        semantic_matches = []
        policy_bonuses = {}
        
        # Expand preferences with semantic relationships if available
        expanded_preferences = application.preferences
        if self.semantic_expander:
            expanded_preferences = await self.semantic_expander.expand_preferences(
                application.preferences
            )
        
        # Score preferences with explanations
        pref_score, pref_reasons = await self._score_preferences_with_explanation(
            expanded_preferences, center, met_absolutes, unmet_preferences, semantic_matches
        )
        reasons.extend(pref_reasons)
        
        # Score other components
        prop_score, prop_reasons = self._score_properties_with_explanation(center)
        reasons.extend(prop_reasons)
        
        avail_score, avail_reasons = self._score_availability_with_explanation(
            application, center, bucket
        )
        reasons.extend(avail_reasons)
        
        quality_score, quality_reasons = self._score_quality_with_explanation(center)
        reasons.extend(quality_reasons)
        
        # Calculate policy bonuses
        policy_bonuses = self._calculate_policy_bonuses(application)
        for policy, bonus in policy_bonuses.items():
            if bonus > 0:
                reasons.append(MatchReason(
                    category="policy_bonus",
                    property_key=policy,
                    explanation=f"Policy bonus for {policy.replace('_', ' ')}",
                    score_contribution=bonus,
                    confidence=1.0,
                    source="policy"
                ))
        
        # Calculate final score
        component_scores = {
            "preference": pref_score,
            "property": prop_score,
            "availability": avail_score,
            "quality": quality_score
        }
        
        base_score = sum(
            component_scores[component] * weight
            for component, weight in self.weights.items()
        )
        
        # Apply policy bonuses
        total_bonus = sum(policy_bonuses.values())
        final_score = min(2.0, base_score + total_bonus)  # Allow scores > 1 for ranking
        
        # Calculate distance for context
        distance_km = geodesic(
            (application.home_location.latitude, application.home_location.longitude),
            (center.location.latitude, center.location.longitude)
        ).kilometers
        
        explanation = MatchExplanation(
            total_score=final_score,
            distance_km=distance_km,
            reasons=reasons,
            met_absolutes=met_absolutes,
            unmet_preferences=unmet_preferences,
            policy_bonuses=policy_bonuses,
            semantic_matches=semantic_matches,
            # Legacy fields
            preference_score=pref_score,
            components=component_scores
        )
        
        return final_score, explanation
    
    async def _score_preferences_with_explanation(
        self,
        preferences: List[ParentPreference],
        center: Center,
        met_absolutes: List[str],
        unmet_preferences: List[str],
        semantic_matches: List[str]
    ) -> Tuple[float, List[MatchReason]]:
        """Score preferences with detailed explanations."""
        if not preferences:
            return 0.5, []
        
        reasons = []
        total_score = 0
        total_weight = 0
        
        # Group preferences by original vs semantic
        original_prefs = [p for p in preferences if not hasattr(p, '_semantic_source')]
        semantic_prefs = [p for p in preferences if hasattr(p, '_semantic_source')]
        
        for pref in preferences:
            # Skip avoid preferences (handled in filters)
            if pref.strength == PreferenceStrength.AVOID:
                continue
            
            prop = center.get_property(pref.property_key)
            satisfaction, reason = self._evaluate_preference_with_explanation(pref, prop)
            
            if reason:
                # Mark as semantic if it came from expansion
                if hasattr(pref, '_semantic_source'):
                    reason.source = "semantic"
                    if satisfaction > 0:
                        semantic_matches.append(pref.property_key)
                
                reasons.append(reason)
            
            # Track absolute requirements
            if pref.is_absolute and satisfaction > 0.8:
                met_absolutes.append(pref.property_key)
            elif pref.strength == PreferenceStrength.PREFERRED and satisfaction < 0.3:
                unmet_preferences.append(pref.property_key)
            
            # Weight by preference strength
            weight = pref.computed_weight
            total_score += satisfaction * abs(weight)  # Use abs for avoid preferences
            total_weight += abs(weight)
        
        if total_weight == 0:
            return 0.5, reasons
        
        return total_score / total_weight, reasons
    
    def _evaluate_preference_with_explanation(
        self,
        preference: ParentPreference,
        center_property: Optional['CenterProperty']
    ) -> Tuple[float, Optional[MatchReason]]:
        """Evaluate preference with explanation."""
        if not center_property:
            explanation = f"Center doesn't have {preference.property_key.replace('_', ' ')}"
            score = 0.0 if preference.is_absolute else 0.3
            
            return score, MatchReason(
                category="absolute_match" if preference.is_absolute else "preference_match",
                property_key=preference.property_key,
                explanation=explanation,
                score_contribution=score * preference.computed_weight,
                confidence=1.0
            )
        
        score = self._calculate_property_match_score(preference, center_property)
        explanation = self._generate_match_explanation(preference, center_property, score)
        
        if score > 0:
            return score, MatchReason(
                category="absolute_match" if preference.is_absolute else "preference_match",
                property_key=preference.property_key,
                explanation=explanation,
                score_contribution=score * preference.computed_weight,
                confidence=center_property.confidence_score
            )
        
        return score, None
    
    def _calculate_property_match_score(
        self,
        preference: ParentPreference,
        center_property: 'CenterProperty'
    ) -> float:
        """Calculate numeric match score for property."""
        if preference.operator == ComparisonOperator.EQUALS:
            return 1.0 if center_property.get_value() == preference.get_value() else 0.0
        
        elif preference.operator == ComparisonOperator.GREATER_THAN:
            if (center_property.value_numeric is not None and 
                preference.value_numeric is not None):
                if center_property.value_numeric > preference.value_numeric:
                    # Graduated score based on how much greater
                    excess = center_property.value_numeric - preference.value_numeric
                    return min(1.0, 1.0 + (excess / preference.value_numeric) * 0.2)
                return 0.0
        
        elif preference.operator == ComparisonOperator.LESS_THAN:
            if (center_property.value_numeric is not None and 
                preference.value_numeric is not None):
                if center_property.value_numeric < preference.value_numeric:
                    # Graduated score - better if much less
                    ratio = center_property.value_numeric / preference.value_numeric
                    return 1.0 - ratio * 0.5  # Scale to prefer smaller values
                return 0.0
        
        elif preference.operator == ComparisonOperator.BETWEEN:
            if (center_property.value_numeric is not None and
                preference.min_value is not None and 
                preference.max_value is not None):
                value = center_property.value_numeric
                if preference.min_value <= value <= preference.max_value:
                    # Perfect score if in range
                    return 1.0
                elif value < preference.min_value:
                    # Partial score if below range
                    return max(0, 0.5 - (preference.min_value - value) * 0.1)
                else:  # value > preference.max_value
                    # Partial score if above range
                    return max(0, 0.5 - (value - preference.max_value) * 0.1)
        
        elif preference.operator == ComparisonOperator.CONTAINS:
            if preference.property_type == PropertyType.LIST:
                if center_property.value_list and preference.value_list:
                    matches = set(preference.value_list) & set(center_property.value_list)
                    if matches:
                        return len(matches) / len(preference.value_list)
            elif preference.property_type == PropertyType.TEXT:
                if center_property.value_text and preference.value_text:
                    if preference.value_text.lower() in center_property.value_text.lower():
                        return 1.0
        
        elif preference.operator == ComparisonOperator.NOT_CONTAINS:
            if preference.property_type == PropertyType.LIST:
                if center_property.value_list and preference.value_list:
                    matches = set(preference.value_list) & set(center_property.value_list)
                    return 1.0 if not matches else 0.0
            elif preference.property_type == PropertyType.TEXT:
                if center_property.value_text and preference.value_text:
                    return 0.0 if preference.value_text.lower() in center_property.value_text.lower() else 1.0
        
        return 0.0
    
    def _generate_match_explanation(
        self,
        preference: ParentPreference,
        center_property: 'CenterProperty',
        score: float
    ) -> str:
        """Generate human-readable explanation for match."""
        prop_name = preference.property_key.replace('_', ' ')
        
        if preference.property_type == PropertyType.BOOLEAN:
            has_feature = "has" if center_property.value_boolean else "doesn't have"
            return f"Center {has_feature} {prop_name}"
        
        elif preference.property_type == PropertyType.NUMBER:
            value = center_property.value_numeric
            if preference.operator == ComparisonOperator.LESS_THAN:
                return f"Has {value} {prop_name} (wanted <{preference.value_numeric})"
            elif preference.operator == ComparisonOperator.GREATER_THAN:
                return f"Has {value} {prop_name} (wanted >{preference.value_numeric})"
            elif preference.operator == ComparisonOperator.BETWEEN:
                return f"Has {value} {prop_name} (wanted {preference.min_value}-{preference.max_value})"
        
        elif preference.property_type == PropertyType.LIST:
            if preference.operator == ComparisonOperator.CONTAINS:
                matches = set(preference.value_list or []) & set(center_property.value_list or [])
                if matches:
                    return f"Offers {', '.join(matches)} (from your {prop_name} preferences)"
        
        elif preference.property_type == PropertyType.TEXT:
            return f"Matches your {prop_name} preference: {center_property.value_text}"
        
        return f"Matches {prop_name} requirement"
    
    def _score_properties_with_explanation(
        self,
        center: Center
    ) -> Tuple[float, List[MatchReason]]:
        """Score general center properties."""
        reasons = []
        
        # Quality properties that add value
        beneficial_properties = {
            "organic_food": "organic meal program",
            "outdoor_space": "outdoor play areas", 
            "music_program": "music and arts program",
            "language_immersion": "language learning program",
            "stem_activities": "STEM education activities",
            "small_groups": "small group sizes",
            "qualified_teachers": "qualified teaching staff"
        }
        
        score = 0.5  # Base score
        bonus_per_property = 0.1
        
        for prop_key, description in beneficial_properties.items():
            if center.has_property(prop_key):
                prop = center.get_property(prop_key)
                if prop.value_boolean:
                    score += bonus_per_property
                    reasons.append(MatchReason(
                        category="property_bonus",
                        property_key=prop_key,
                        explanation=f"Center has {description}",
                        score_contribution=bonus_per_property,
                        confidence=prop.confidence_score
                    ))
        
        return min(1.0, score), reasons
    
    def _score_availability_with_explanation(
        self,
        application: Application,
        center: Center,
        bucket: CapacityBucket
    ) -> Tuple[float, List[MatchReason]]:
        """Score availability match with explanations."""
        reasons = []
        
        if not bucket.is_available:
            reasons.append(MatchReason(
                category="availability",
                property_key="capacity",
                explanation="No available capacity",
                score_contribution=0.0,
                confidence=1.0
            ))
            return 0.0, reasons
        
        # Capacity score
        capacity_ratio = bucket.available_capacity / bucket.total_capacity
        capacity_score = min(1.0, capacity_ratio * 2)  # Boost high availability
        
        if capacity_ratio > 0.5:
            reasons.append(MatchReason(
                category="availability",
                property_key="capacity",
                explanation=f"Good availability ({bucket.available_capacity}/{bucket.total_capacity} spots)",
                score_contribution=capacity_score * 0.6,
                confidence=1.0
            ))
        
        # Timing score
        months_diff = (
            (bucket.start_month.year - application.desired_start_date.year) * 12 +
            bucket.start_month.month - application.desired_start_date.month
        )
        
        timing_score = 1.0
        timing_explanation = "Perfect timing match"
        
        if months_diff > 0:
            timing_score = max(0.2, 1.0 - months_diff * 0.15)
            timing_explanation = f"Starts {months_diff} month(s) after desired date"
        elif months_diff < 0:
            timing_score = 0.6
            timing_explanation = "Already started (immediate availability)"
        
        reasons.append(MatchReason(
            category="availability",
            property_key="timing",
            explanation=timing_explanation,
            score_contribution=timing_score * 0.4,
            confidence=1.0
        ))
        
        final_score = capacity_score * 0.6 + timing_score * 0.4
        return final_score, reasons
    
    def _score_quality_with_explanation(
        self,
        center: Center
    ) -> Tuple[float, List[MatchReason]]:
        """Score center quality indicators."""
        reasons = []
        base_score = 0.5
        
        # Quality certifications
        quality_indicators = {
            "cert_ofsted_outstanding": ("Outstanding Ofsted rating", 0.2),
            "cert_montessori": ("Montessori certified program", 0.15),
            "cert_waldorf": ("Waldorf/Steiner education approach", 0.15),
            "cert_reggio_emilia": ("Reggio Emilia approach", 0.15),
            "awards_received": ("Recognition and awards", 0.1)
        }
        
        score = base_score
        for cert_key, (description, value) in quality_indicators.items():
            if center.has_property(cert_key):
                prop = center.get_property(cert_key)
                if prop.value_boolean or (prop.value_list and len(prop.value_list) > 0):
                    score += value
                    reasons.append(MatchReason(
                        category="quality_indicator",
                        property_key=cert_key,
                        explanation=description,
                        score_contribution=value,
                        confidence=prop.confidence_score
                    ))
        
        return min(1.0, score), reasons
    
    def _calculate_policy_bonuses(self, application: Application) -> Dict[str, float]:
        """Calculate policy-based score bonuses."""
        bonuses = {}
        
        # Priority flags translate to bonuses (these help with ranking)
        priority_bonuses = {
            "sibling": 0.5,      # Sibling priority
            "low_income": 0.3,   # Low income support
            "municipality": 0.2, # Municipality priority
            "special_needs": 0.4 # Special needs support
        }
        
        for flag in application.priority_flags:
            if flag in priority_bonuses:
                bonuses[flag] = priority_bonuses[flag]
        
        return bonuses
    
    def determine_match_quality(self, score: float) -> str:
        """Determine match quality category from score."""
        if score >= 1.5:
            return "excellent"
        elif score >= 1.0:
            return "good" 
        elif score >= 0.7:
            return "fair"
        else:
            return "poor"