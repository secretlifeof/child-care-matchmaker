"""Composite scoring system for matching."""

from typing import Dict, List, Optional
import logging

from ..models.base import (
    Application, Center, CapacityBucket,
    ParentPreference, ComparisonOperator
)

logger = logging.getLogger(__name__)


class CompositeScorer:
    """Combines multiple scoring components into a single score."""
    
    def __init__(
        self,
        preference_weight: float = 0.4,
        property_weight: float = 0.3,
        availability_weight: float = 0.2,
        quality_weight: float = 0.1
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
    
    def score(
        self,
        application: Application,
        center: Center,
        bucket: CapacityBucket
    ) -> float:
        """
        Calculate composite score for application-center-bucket match.
        
        Returns:
            Score between 0 and 1
        """
        components = self.get_score_breakdown(application, center, bucket)
        
        # Weighted sum
        total_score = sum(
            components.get(component, 0) * weight
            for component, weight in self.weights.items()
        )
        
        return min(1.0, max(0.0, total_score))
    
    def get_score_breakdown(
        self,
        application: Application,
        center: Center,
        bucket: CapacityBucket
    ) -> Dict[str, float]:
        """Get detailed breakdown of score components."""
        return {
            "preference": self._score_preferences(application, center),
            "property": self._score_properties(application, center),
            "availability": self._score_availability(application, center, bucket),
            "quality": self._score_quality(center)
        }
    
    def _score_preferences(
        self,
        application: Application,
        center: Center
    ) -> float:
        """Score based on parent preferences."""
        if not application.preferences:
            return 0.5  # Neutral if no preferences
        
        total_score = 0
        total_weight = 0
        
        for pref in application.preferences:
            # Skip exclusions (handled in hard filters)
            if pref.threshold <= 0.1:
                continue
            
            # Check if center satisfies preference
            satisfaction = self._evaluate_preference(pref, center)
            
            # Weight by preference importance
            total_score += satisfaction * pref.weight
            total_weight += pref.weight
        
        if total_weight == 0:
            return 0.5
        
        return total_score / total_weight
    
    def _evaluate_preference(
        self,
        preference: ParentPreference,
        center: Center
    ) -> float:
        """
        Evaluate how well center satisfies a preference.
        
        Returns:
            Satisfaction score between 0 and 1
        """
        prop = center.get_property(preference.property_key)
        
        if not prop:
            # Property not found
            if preference.threshold >= 0.9:  # Must have
                return 0
            return 0.3  # Partial penalty for nice-to-have
        
        # Evaluate based on operator
        if preference.operator == ComparisonOperator.EQUALS:
            if prop.get_value() == preference.get_value():
                return 1.0
            return 0.0
        
        elif preference.operator == ComparisonOperator.BOOLEAN:
            if preference.value_boolean is not None:
                if prop.value_boolean == preference.value_boolean:
                    return 1.0
                return 0.0
        
        elif preference.operator == ComparisonOperator.GREATER_THAN:
            if prop.value_numeric and preference.value_numeric:
                if prop.value_numeric > preference.value_numeric:
                    # Calculate degree of satisfaction
                    diff = prop.value_numeric - preference.value_numeric
                    return min(1.0, diff / preference.value_numeric)
                return 0.0
        
        elif preference.operator == ComparisonOperator.LESS_THAN:
            if prop.value_numeric and preference.value_numeric:
                if prop.value_numeric < preference.value_numeric:
                    # Calculate degree of satisfaction
                    ratio = prop.value_numeric / preference.value_numeric
                    return 1.0 - ratio
                return 0.0
        
        elif preference.operator == ComparisonOperator.CONTAINS:
            if preference.value_text and prop.value_text:
                if preference.value_text.lower() in prop.value_text.lower():
                    return 1.0
            elif preference.value_list and prop.value_list:
                # Check list overlap
                overlap = set(preference.value_list) & set(prop.value_list)
                if overlap:
                    return len(overlap) / len(preference.value_list)
            return 0.0
        
        elif preference.operator == ComparisonOperator.BETWEEN:
            if isinstance(preference.value_list, list) and len(preference.value_list) == 2:
                if prop.value_numeric:
                    min_val, max_val = preference.value_list
                    if min_val <= prop.value_numeric <= max_val:
                        # Score based on position in range
                        range_size = max_val - min_val
                        if range_size > 0:
                            position = (prop.value_numeric - min_val) / range_size
                            # Prefer middle of range
                            return 1.0 - abs(position - 0.5) * 2
                        return 1.0
            return 0.0
        
        # Default
        return 0.5
    
    def _score_properties(
        self,
        application: Application,
        center: Center
    ) -> float:
        """Score based on center properties matching preferences."""
        # Count beneficial properties
        beneficial_properties = [
            "organic_food",
            "outdoor_space",
            "music_program",
            "language_immersion",
            "stem_activities"
        ]
        
        score = 0
        for prop_key in beneficial_properties:
            if center.has_property(prop_key):
                prop = center.get_property(prop_key)
                if prop.value_boolean:
                    score += 0.2
        
        return min(1.0, score)
    
    def _score_availability(
        self,
        application: Application,
        center: Center,
        bucket: CapacityBucket
    ) -> float:
        """Score based on availability match."""
        # Check capacity availability
        if not bucket.is_available:
            return 0.0
        
        # Score based on capacity ratio
        capacity_ratio = bucket.available_capacity / bucket.total_capacity
        
        # Check timing match
        months_until_start = (
            (bucket.start_month.year - application.desired_start_date.year) * 12 +
            bucket.start_month.month - application.desired_start_date.month
        )
        
        timing_score = 1.0
        if months_until_start > 0:
            # Penalize delayed start
            timing_score = max(0, 1.0 - months_until_start * 0.1)
        elif months_until_start < 0:
            # Already started
            timing_score = 0.5
        
        return capacity_ratio * 0.5 + timing_score * 0.5
    
    def _score_quality(self, center: Center) -> float:
        """Score based on center quality indicators."""
        quality_score = 0.5  # Base score
        
        # Check for quality certifications
        quality_certs = [
            "cert_ofsted_outstanding",
            "cert_montessori",
            "cert_waldorf",
            "cert_reggio_emilia"
        ]
        
        for cert in quality_certs:
            if center.has_property(cert):
                quality_score += 0.125
        
        # Check for awards
        if center.has_property("awards"):
            prop = center.get_property("awards")
            if prop.value_list:
                quality_score += min(0.2, len(prop.value_list) * 0.05)
        
        return min(1.0, quality_score)


class MLScorer:
    """Machine learning based scorer (placeholder for future implementation)."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        if model_path:
            self._load_model(model_path)
    
    def _load_model(self, path: str):
        """Load pre-trained model."""
        # Placeholder for ML model loading
        logger.info(f"Would load ML model from {path}")
    
    def score(
        self,
        application: Application,
        center: Center,
        bucket: CapacityBucket
    ) -> float:
        """
        Predict match quality using ML model.
        
        Returns:
            Predicted acceptance probability or quality score
        """
        if not self.model:
            return 0.5  # Default score if no model
        
        # Extract features
        features = self._extract_features(application, center, bucket)
        
        # Predict (placeholder)
        # prediction = self.model.predict(features)
        
        return 0.5  # Placeholder
    
    def _extract_features(
        self,
        application: Application,
        center: Center,
        bucket: CapacityBucket
    ) -> Dict:
        """Extract features for ML model."""
        # Placeholder for feature extraction
        return {}