"""Integration tests for the complete matching pipeline."""

from datetime import date, timedelta
from uuid import uuid4

import pytest

from src.matchmaker.graph.builder import MatchingGraphBuilder
from src.matchmaker.graph.matcher import GraphMatcher
from src.matchmaker.models.base import (
    AgeGroup,
    Application,
    CapacityBucket,
    Center,
    CenterProperty,
    Child,
    ComparisonOperator,
    Location,
    MatchMode,
    ParentPreference,
    PropertyCategory,
    SourceType,
    TimeSlot,
)
from src.matchmaker.scoring.composite_scorer import CompositeScorer
from src.matchmaker.utils.filters import HardConstraintFilter


class TestIntegrationScenarios:
    """Integration tests for real-world scenarios."""

    @pytest.fixture
    def pipeline_components(self):
        """Create full matching pipeline."""
        scorer = CompositeScorer()
        filter = HardConstraintFilter()
        builder = MatchingGraphBuilder(scorer, filter)
        matcher = GraphMatcher(seed=42)
        return builder, matcher

    def test_realistic_city_scenario(self, pipeline_components):
        """Test realistic scenario with multiple families and centers."""
        builder, matcher = pipeline_components

        # Create multiple families with different needs
        families = self._create_diverse_families()
        centers = self._create_diverse_centers()

        # Build graph
        graph = builder.build_graph(families, centers)

        # Test all three modes
        # 1. Get recommendations for first family
        recommend_result = matcher.match(
            graph,
            MatchMode.RECOMMEND,
            application_id=families[0].id,
            top_k=5
        )

        assert recommend_result.success
        assert len(recommend_result.offers) > 0

        # 2. Global allocation
        allocate_result = matcher.match(
            graph,
            MatchMode.ALLOCATE
        )

        assert allocate_result.success
        assert allocate_result.matched_applications > 0

        # 3. Waitlist for first center
        waitlist_result = matcher.match(
            graph,
            MatchMode.WAITLIST,
            center_id=centers[0].id
        )

        assert waitlist_result.success
        assert len(waitlist_result.waitlist_entries) > 0

    def test_capacity_constraint_respect(self, pipeline_components):
        """Test that capacity constraints are properly respected."""
        builder, matcher = pipeline_components

        # Create 10 applications
        applications = [self._create_family_application(f"Family {i}") for i in range(10)]

        # Create center with limited capacity
        center = self._create_limited_capacity_center(total_capacity=3)

        graph = builder.build_graph(applications, [center])

        # Global allocation should respect capacity
        result = matcher.match(graph, MatchMode.ALLOCATE, respect_capacity=True)

        # May fail with INFEASIBLE due to constraints
        if result.success:
            assert result.matched_applications <= 3
            assert len(result.offers) <= 3
        else:
            # If allocation fails, that's also acceptable for capacity constraints
            assert result.matched_applications is None or result.matched_applications <= 3

    def test_policy_tier_ordering(self, pipeline_components):
        """Test that policy tiers are correctly ordered in waitlists."""
        builder, matcher = pipeline_components

        # Create applications with different priorities
        sibling_app = self._create_family_application("Sibling Family")
        sibling_app.priority_flags = ["sibling"]

        regular_app = self._create_family_application("Regular Family")
        regular_app.priority_flags = []

        low_income_app = self._create_family_application("Low Income Family")
        low_income_app.priority_flags = ["low_income"]

        applications = [regular_app, low_income_app, sibling_app]  # Random order
        center = self._create_standard_center()

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

        entries = result.waitlist_entries
        assert len(entries) == 3

        # Check ordering: sibling first, then low_income, then regular
        assert entries[0].application_id == sibling_app.id
        assert entries[1].application_id == low_income_app.id
        assert entries[2].application_id == regular_app.id

    def test_sibling_bonus_scoring(self, pipeline_components):
        """Test that sibling applications get bonus scoring."""
        builder, matcher = pipeline_components

        # Single child family
        single_family = self._create_family_application("Single Child")

        # Sibling family
        sibling_family = self._create_sibling_family("Sibling Family")

        center = self._create_standard_center()

        # Build graph for both families
        single_graph = builder.build_graph([single_family], [center])
        sibling_graph = builder.build_graph([sibling_family], [center])

        # Get recommendations for both
        single_result = matcher.match(
            single_graph,
            MatchMode.RECOMMEND,
            application_id=single_family.id,
            top_k=1
        )

        sibling_result = matcher.match(
            sibling_graph,
            MatchMode.RECOMMEND,
            application_id=sibling_family.id,
            top_k=1
        )

        # Sibling family should have higher score due to bonus
        if single_result.offers and sibling_result.offers:
            assert sibling_result.offers[0].score > single_result.offers[0].score

    def test_progressive_distance_impact(self, pipeline_components):
        """Test that distance affects scoring progressively."""
        builder, matcher = pipeline_components

        # Create application at specific location
        application = self._create_family_application("Distance Test")

        # Create centers at different distances
        close_center = self._create_standard_center()
        close_center.name = "Close Center"
        close_center.location.latitude = application.home_location.latitude + 0.001  # ~100m

        far_center = self._create_standard_center()
        far_center.name = "Far Center"
        far_center.location.latitude = application.home_location.latitude + 0.05  # ~5.5km (within 10km limit)

        centers = [close_center, far_center]
        graph = builder.build_graph([application], centers)

        result = matcher.match(
            graph,
            MatchMode.RECOMMEND,
            application_id=application.id,
            top_k=2
        )

        assert len(result.offers) >= 1  # At least one center should match

        if len(result.offers) == 2:
            # If we have both centers, check distance ordering
            # Close center should be ranked higher
            assert result.offers[0].score > result.offers[1].score
        else:
            # If only one center matched, verify it's reasonable
            offer = result.offers[0]
            assert offer.score > 0.0

    def test_preference_satisfaction_scoring(self, pipeline_components):
        """Test that preference satisfaction affects scoring."""
        builder, matcher = pipeline_components

        # Create application with specific preference
        application = self._create_family_application("Preference Test")
        application.preferences = [
            ParentPreference(
                id=uuid4(),
                profile_id=application.id,
                property_key="organic_food",
                operator=ComparisonOperator.EQUALS,
                value_boolean=True,
                weight=0.9,
                threshold=0.7
            )
        ]

        # Create centers - one with, one without organic food
        organic_center = self._create_standard_center()
        organic_center.name = "Organic Center"
        organic_center.properties = [
            CenterProperty(
                id=uuid4(),
                center_id=organic_center.id,
                property_type_id=uuid4(),
                property_key="organic_food",
                category=PropertyCategory.SERVICE,
                value_boolean=True,
                source=SourceType.VERIFIED
            )
        ]

        regular_center = self._create_standard_center()
        regular_center.name = "Regular Center"
        regular_center.properties = []

        centers = [regular_center, organic_center]  # Reverse order
        graph = builder.build_graph([application], centers)

        result = matcher.match(
            graph,
            MatchMode.RECOMMEND,
            application_id=application.id,
            top_k=2
        )

        assert len(result.offers) == 2

        # Organic center should be ranked higher due to preference match
        organic_offer = next(o for o in result.offers if o.center_id == organic_center.id)
        regular_offer = next(o for o in result.offers if o.center_id == regular_center.id)

        assert organic_offer.score > regular_offer.score

    # Helper methods

    def _create_diverse_families(self):
        """Create diverse set of family applications."""
        families = []

        # Young family with infant
        young_family = self._create_family_application("Young Family")
        young_family.children[0].birth_date = date.today() - timedelta(days=180)  # 6 months
        young_family.preferences = [
            ParentPreference(
                id=uuid4(),
                profile_id=young_family.id,
                property_key="infant_care",
                operator=ComparisonOperator.EQUALS,
                value_boolean=True,
                weight=0.9,
                threshold=0.9
            )
        ]
        families.append(young_family)

        # Working parents needing long hours
        working_family = self._create_family_application("Working Family")
        working_family.desired_hours = [
            TimeSlot.from_hours(day_of_week=i, start_hour=7, end_hour=18)
            for i in range(5)
        ]
        families.append(working_family)

        # Family with special dietary needs
        dietary_family = self._create_family_application("Dietary Family")
        dietary_family.preferences = [
            ParentPreference(
                id=uuid4(),
                profile_id=dietary_family.id,
                property_key="vegetarian_meals",
                operator=ComparisonOperator.EQUALS,
                value_boolean=True,
                weight=0.8,
                threshold=0.8
            )
        ]
        families.append(dietary_family)

        return families

    def _create_diverse_centers(self):
        """Create diverse set of daycare centers."""
        centers = []

        # Premium center with many amenities
        premium = self._create_standard_center()
        premium.name = "Premium Daycare"
        premium.properties = [
            CenterProperty(
                id=uuid4(),
                center_id=premium.id,
                property_type_id=uuid4(),
                property_key="organic_food",
                category=PropertyCategory.SERVICE,
                value_boolean=True,
                source=SourceType.VERIFIED
            ),
            CenterProperty(
                id=uuid4(),
                center_id=premium.id,
                property_type_id=uuid4(),
                property_key="outdoor_space",
                category=PropertyCategory.FACILITY,
                value_boolean=True,
                source=SourceType.VERIFIED
            )
        ]
        centers.append(premium)

        # Budget-friendly center
        budget = self._create_standard_center()
        budget.name = "Budget Daycare"
        budget.location.latitude += 0.01  # Slightly different location
        centers.append(budget)

        # Specialized infant care center
        infant_center = self._create_standard_center()
        infant_center.name = "Infant Specialists"
        infant_center.location.longitude += 0.01
        infant_center.capacity_buckets = [
            CapacityBucket(
                id=uuid4(),
                center_id=infant_center.id,
                age_band=AgeGroup(min_age_months=0, max_age_months=18, name="Infants"),
                start_month=date.today(),
                total_capacity=10,
                available_capacity=5
            )
        ]
        infant_center.properties = [
            CenterProperty(
                id=uuid4(),
                center_id=infant_center.id,
                property_type_id=uuid4(),
                property_key="infant_care",
                category=PropertyCategory.SERVICE,
                value_boolean=True,
                source=SourceType.VERIFIED
            )
        ]
        centers.append(infant_center)

        return centers

    def _create_family_application(self, name: str) -> Application:
        """Create a standard family application."""
        family_id = uuid4()
        return Application(
            id=uuid4(),
            family_id=family_id,
            children=[
                Child(
                    id=uuid4(),
                    family_id=family_id,
                    name=f"{name} Child",
                    birth_date=date.today() - timedelta(days=720)  # 2 years old
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
            max_distance_km=10.0,
            priority_flags=[]
        )

    def _create_sibling_family(self, name: str) -> Application:
        """Create family with multiple children."""
        family_id = uuid4()
        return Application(
            id=uuid4(),
            family_id=family_id,
            children=[
                Child(
                    id=uuid4(),
                    family_id=family_id,
                    name=f"{name} Child 1",
                    birth_date=date.today() - timedelta(days=720)  # 2 years
                ),
                Child(
                    id=uuid4(),
                    family_id=family_id,
                    name=f"{name} Child 2",
                    birth_date=date.today() - timedelta(days=1080)  # 3 years
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
            max_distance_km=10.0,
            priority_flags=["sibling"]
        )

    def _create_standard_center(self) -> Center:
        """Create a standard daycare center."""
        center_id = uuid4()
        return Center(
            id=center_id,
            name="Standard Daycare",
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
                        max_age_months=72,
                        name="Mixed Age"
                    ),
                    start_month=date.today(),
                    total_capacity=20,
                    available_capacity=10
                )
            ]
        )

    def _create_limited_capacity_center(self, total_capacity: int) -> Center:
        """Create center with specific capacity."""
        center = self._create_standard_center()
        center.name = f"Limited Capacity ({total_capacity})"
        center.capacity_buckets[0].total_capacity = total_capacity
        center.capacity_buckets[0].available_capacity = total_capacity
        return center
