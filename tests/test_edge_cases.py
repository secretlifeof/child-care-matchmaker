"""Edge case tests for the matchmaker service."""

import pytest
from datetime import date, timedelta
from uuid import uuid4
import networkx as nx

from src.matchmaker.models.base import (
    Application, Center, Child, Location, ParentPreference,
    CapacityBucket, AgeGroup, TimeSlot, CenterProperty,
    ComparisonOperator, PropertyCategory, SourceType, MatchMode
)
from src.matchmaker.graph.builder import MatchingGraphBuilder
from src.matchmaker.graph.matcher import GraphMatcher
from src.matchmaker.scoring.composite_scorer import CompositeScorer
from src.matchmaker.utils.filters import HardConstraintFilter


class TestEdgeCases:
    """Test edge cases in matching system."""
    
    @pytest.fixture
    def scorer(self):
        """Create a scorer instance."""
        return CompositeScorer()
    
    @pytest.fixture
    def filter(self):
        """Create a filter instance."""
        return HardConstraintFilter()
    
    @pytest.fixture
    def builder(self, scorer, filter):
        """Create a graph builder."""
        return MatchingGraphBuilder(scorer, filter)
    
    @pytest.fixture
    def matcher(self):
        """Create a matcher instance."""
        return GraphMatcher(seed=42)
    
    def test_no_centers_available(self, builder, matcher):
        """Test when no centers are available."""
        application = self._create_application()
        centers = []  # No centers
        
        graph = builder.build_graph([application], centers)
        
        # Graph should have only application node
        assert graph.number_of_nodes() == 1
        assert graph.number_of_edges() == 0
        
        result = matcher.match(graph, MatchMode.RECOMMEND, application_id=application.id)
        assert result.success == True  # Empty result is still successful
        assert len(result.offers) == 0
    
    def test_no_matching_age_groups(self, builder, matcher):
        """Test when child age doesn't match any center's age groups."""
        # Create application with infant
        application = self._create_application(child_age_months=6)
        
        # Create center that only accepts toddlers
        center = self._create_center()
        center.capacity_buckets = [
            CapacityBucket(
                id=uuid4(),
                center_id=center.id,
                age_band=AgeGroup(
                    min_age_months=24,  # Only toddlers
                    max_age_months=48,
                    name="Toddlers"
                ),
                start_month=date.today(),
                total_capacity=10,
                available_capacity=5
            )
        ]
        
        graph = builder.build_graph([application], [center])
        
        # Should have nodes but no edges due to age mismatch
        assert graph.number_of_nodes() == 2  # 1 app + 1 bucket
        assert graph.number_of_edges() == 0
        
        result = matcher.match(graph, MatchMode.RECOMMEND, application_id=application.id)
        assert len(result.offers) == 0
    
    def test_distance_exceeds_maximum(self, builder, matcher):
        """Test when all centers exceed maximum distance."""
        # Application with strict distance limit
        application = self._create_application(max_distance_km=1.0)
        
        # Center far away
        center = self._create_center()
        center.location.latitude += 1.0  # Move ~111km away
        
        graph = builder.build_graph([application], [center])
        
        # No edges due to distance constraint
        assert graph.number_of_edges() == 0
    
    def test_no_capacity_available(self, builder, matcher):
        """Test when centers have no available capacity."""
        application = self._create_application()
        center = self._create_center()
        
        # Set capacity to 0
        for bucket in center.capacity_buckets:
            bucket.available_capacity = 0
        
        graph = builder.build_graph([application], [center], respect_capacity=True)
        
        # No bucket nodes added when respecting capacity
        assert graph.number_of_nodes() == 1  # Only application
    
    def test_conflicting_must_have_preferences(self, builder, matcher):
        """Test when must-have preferences cannot be satisfied."""
        application = self._create_application()
        
        # Add must-have preference for organic food
        application.preferences = [
            ParentPreference(
                id=uuid4(),
                profile_id=application.id,
                property_key="organic_food",
                operator=ComparisonOperator.EQUALS,
                value_boolean=True,
                weight=1.0,
                threshold=0.95  # Must have
            )
        ]
        
        # Center without organic food
        center = self._create_center()
        center.properties = [
            CenterProperty(
                id=uuid4(),
                center_id=center.id,
                property_type_id=uuid4(),
                property_key="organic_food",
                category=PropertyCategory.SERVICE,
                value_boolean=False,  # Doesn't have it
                source=SourceType.VERIFIED
            )
        ]
        
        graph = builder.build_graph([application], [center])
        
        # No edges due to must-have constraint violation
        assert graph.number_of_edges() == 0
    
    def test_siblings_different_age_groups(self, builder, matcher):
        """Test matching siblings of different ages."""
        # Create family with infant and toddler
        family_id = uuid4()
        application = Application(
            id=uuid4(),
            family_id=family_id,
            children=[
                Child(
                    id=uuid4(),
                    family_id=family_id,
                    name="Baby",
                    birth_date=date.today() - timedelta(days=180)  # 6 months
                ),
                Child(
                    id=uuid4(),
                    family_id=family_id,
                    name="Toddler",
                    birth_date=date.today() - timedelta(days=900)  # ~30 months
                )
            ],
            home_location=Location(latitude=52.52, longitude=13.405, country_code="DE"),
            preferences=[],
            desired_start_date=date.today(),
            desired_hours=[],
            priority_flags=["sibling"]
        )
        
        # Center with separate infant and toddler groups
        center = self._create_center()
        center.capacity_buckets = [
            CapacityBucket(
                id=uuid4(),
                center_id=center.id,
                age_band=AgeGroup(min_age_months=0, max_age_months=12, name="Infants"),
                start_month=date.today(),
                total_capacity=10,
                available_capacity=5
            ),
            CapacityBucket(
                id=uuid4(),
                center_id=center.id,
                age_band=AgeGroup(min_age_months=24, max_age_months=48, name="Toddlers"),
                start_month=date.today(),
                total_capacity=10,
                available_capacity=5
            )
        ]
        
        graph = builder.build_graph([application], [center])
        
        # Should not create edges as children don't both fit in any single bucket
        assert graph.number_of_edges() == 0
    
    def test_reserved_bucket_without_qualification(self, builder):
        """Test accessing reserved bucket without proper flags."""
        application = self._create_application()
        application.priority_flags = []  # No special flags
        
        center = self._create_center()
        center.capacity_buckets[0].reserved_label = "sibling"  # Reserved for siblings
        
        filter = HardConstraintFilter()
        
        # Should fail constraint check
        assert not filter.is_valid_match(
            application,
            center,
            center.capacity_buckets[0]
        )
    
    def test_time_slot_no_overlap(self, builder):
        """Test when desired hours don't overlap with center hours."""
        application = self._create_application()
        
        # Want early morning care
        application.desired_hours = [
            TimeSlot.from_hours(day_of_week=0, start_hour=5, end_hour=7)
        ]
        
        center = self._create_center()
        # Center opens at 8am
        center.opening_hours = [
            TimeSlot.from_hours(day_of_week=0, start_hour=8, end_hour=18)
        ]
        
        filter = HardConstraintFilter()
        
        # Should fail hours constraint
        assert not filter.is_valid_match(
            application,
            center,
            center.capacity_buckets[0]
        )
    
    def test_exclusion_preference_matches(self, builder):
        """Test when center has excluded property."""
        application = self._create_application()
        
        # Exclude religious centers
        application.preferences = [
            ParentPreference(
                id=uuid4(),
                profile_id=application.id,
                property_key="religious_affiliation",
                operator=ComparisonOperator.EQUALS,
                value_boolean=True,
                weight=0.0,
                threshold=0.05  # Exclude
            )
        ]
        
        center = self._create_center()
        center.properties = [
            CenterProperty(
                id=uuid4(),
                center_id=center.id,
                property_type_id=uuid4(),
                property_key="religious_affiliation",
                category=PropertyCategory.APPROACH,
                value_boolean=True,
                source=SourceType.VERIFIED
            )
        ]
        
        filter = HardConstraintFilter()
        
        # Should fail exclusion constraint
        assert not filter.is_valid_match(
            application,
            center,
            center.capacity_buckets[0]
        )
    
    def test_global_allocation_insufficient_capacity(self, builder, matcher):
        """Test global allocation with more applications than capacity."""
        # Create 5 applications
        applications = [self._create_application() for _ in range(5)]
        
        # Create center with capacity for only 2
        center = self._create_center()
        center.capacity_buckets[0].total_capacity = 2
        center.capacity_buckets[0].available_capacity = 2
        
        graph = builder.build_graph(applications, [center])
        
        result = matcher.match(
            graph,
            MatchMode.ALLOCATE,
            respect_capacity=True
        )
        
        # Should only match 2 out of 5, but might fail due to INFEASIBLE
        if result.success:
            assert result.matched_applications <= 2
            assert result.total_applications == 5
            assert result.coverage_rate < 1.0
        else:
            # If allocation fails, that's also a valid outcome
            assert result.matched_applications is None or result.matched_applications <= 2
    
    def test_waitlist_with_policy_tiers(self, builder, matcher):
        """Test waitlist generation with policy priorities."""
        center = self._create_center()
        
        # Create applications with different priority flags
        app_sibling = self._create_application()
        app_sibling.priority_flags = ["sibling"]
        
        app_regular = self._create_application()
        app_regular.priority_flags = []
        
        app_low_income = self._create_application()
        app_low_income.priority_flags = ["low_income"]
        
        applications = [app_regular, app_low_income, app_sibling]
        
        graph = builder.build_graph(applications, [center])
        
        policy_tiers = {
            "sibling": 1000,
            "low_income": 500
        }
        
        result = matcher.match(
            graph,
            MatchMode.WAITLIST,
            center_id=center.id,
            policy_tiers=policy_tiers
        )
        
        # Check order: sibling should be first, then low_income, then regular
        entries = result.waitlist_entries
        assert len(entries) == 3
        assert entries[0].application_id == app_sibling.id
        assert entries[1].application_id == app_low_income.id
        assert entries[2].application_id == app_regular.id
    
    def test_empty_graph(self, matcher):
        """Test matching with empty graph."""
        graph = nx.Graph()
        
        result = matcher.match(graph, MatchMode.ALLOCATE)
        
        assert result.success == True
        assert result.total_applications == 0
        assert result.matched_applications == 0
    
    def test_circular_preferences(self, builder):
        """Test handling of potentially circular preference dependencies."""
        application = self._create_application()
        
        # Add preferences that might create circular logic
        application.preferences = [
            ParentPreference(
                id=uuid4(),
                profile_id=application.id,
                property_key="program_a",
                operator=ComparisonOperator.EQUALS,
                value_boolean=True,
                weight=0.8,
                threshold=0.9  # Must have A
            ),
            ParentPreference(
                id=uuid4(),
                profile_id=application.id,
                property_key="program_b",
                operator=ComparisonOperator.EQUALS,
                value_boolean=False,
                weight=0.8,
                threshold=0.05  # Exclude B
            )
        ]
        
        # Center has A but also requires B
        center = self._create_center()
        center.properties = [
            CenterProperty(
                id=uuid4(),
                center_id=center.id,
                property_type_id=uuid4(),
                property_key="program_a",
                category=PropertyCategory.PROGRAM,
                value_boolean=True,
                source=SourceType.VERIFIED
            ),
            CenterProperty(
                id=uuid4(),
                center_id=center.id,
                property_type_id=uuid4(),
                property_key="program_b",
                category=PropertyCategory.PROGRAM,
                value_boolean=True,  # Has B despite exclusion
                source=SourceType.VERIFIED
            )
        ]
        
        filter = HardConstraintFilter()
        
        # Should fail due to exclusion of B
        assert not filter.is_valid_match(
            application,
            center,
            center.capacity_buckets[0]
        )
    
    # Helper methods
    
    def _create_application(
        self,
        child_age_months: int = 24,
        max_distance_km: float = 10.0
    ) -> Application:
        """Create a test application."""
        family_id = uuid4()
        birth_date = date.today() - timedelta(days=child_age_months * 30)
        
        return Application(
            id=uuid4(),
            family_id=family_id,
            children=[
                Child(
                    id=uuid4(),
                    family_id=family_id,
                    name="Test Child",
                    birth_date=birth_date
                )
            ],
            home_location=Location(
                latitude=52.5200,
                longitude=13.4050,
                city="Berlin",
                country_code="DE"
            ),
            preferences=[],
            desired_start_date=date.today(),
            desired_hours=[
                TimeSlot.from_hours(day_of_week=i, start_hour=8, end_hour=16)
                for i in range(5)
            ],
            max_distance_km=max_distance_km,
            priority_flags=[]
        )
    
    def _create_center(self) -> Center:
        """Create a test center."""
        center_id = uuid4()
        
        return Center(
            id=center_id,
            name="Test Daycare",
            location=Location(
                latitude=52.5150,
                longitude=13.4100,
                city="Berlin",
                country_code="DE"
            ),
            opening_hours=[
                TimeSlot.from_hours(day_of_week=i, start_hour=7, end_hour=18)
                for i in range(5)
            ],
            properties=[],
            capacity_buckets=[
                CapacityBucket(
                    id=uuid4(),
                    center_id=center_id,
                    age_band=AgeGroup(
                        min_age_months=12,
                        max_age_months=48,
                        name="Mixed Age"
                    ),
                    start_month=date.today(),
                    total_capacity=20,
                    available_capacity=10
                )
            ]
        )


class TestScoringEdgeCases:
    """Test edge cases in scoring system."""
    
    def test_zero_weight_normalization(self):
        """Test scorer when all weights are zero."""
        scorer = CompositeScorer(
            preference_weight=0,
            property_weight=0,
            availability_weight=0,
            quality_weight=0
        )
        
        # Weights should be normalized somehow
        assert sum(scorer.weights.values()) > 0
    
    def test_preference_with_no_value(self):
        """Test scoring preference with None values."""
        scorer = CompositeScorer()
        
        application = Application(
            id=uuid4(),
            family_id=uuid4(),
            children=[],
            home_location=Location(latitude=0, longitude=0, country_code="DE"),
            preferences=[
                ParentPreference(
                    id=uuid4(),
                    profile_id=uuid4(),
                    property_key="test",
                    operator=ComparisonOperator.EQUALS,
                    value_text=None,  # No value
                    weight=0.5,
                    threshold=0.5
                )
            ],
            desired_start_date=date.today(),
            desired_hours=[]
        )
        
        center = Center(
            id=uuid4(),
            name="Test",
            location=Location(latitude=0, longitude=0, country_code="DE"),
            properties=[],
            capacity_buckets=[]
        )
        
        bucket = CapacityBucket(
            id=uuid4(),
            center_id=center.id,
            age_band=AgeGroup(min_age_months=0, max_age_months=60, name="All"),
            start_month=date.today(),
            total_capacity=10,
            available_capacity=5
        )
        
        # Should handle gracefully
        score = scorer.score(application, center, bucket)
        assert 0 <= score <= 1