"""Tests for the time string format feature."""
import pytest
from src.matchmaker.models.base import TimeSlot
from src.matchmaker.models.requests import EnhancedRecommendationRequest
from src.matchmaker.graph.builder import MatchingGraphBuilder
from src.matchmaker.graph.matcher import GraphMatcher
from src.matchmaker.utils.filters import HardConstraintFilter
from src.matchmaker.scoring.composite_scorer import CompositeScorer
from src.matchmaker.models.base import MatchMode


class TestTimeStringFormat:
    """Test time string format functionality."""
    
    def test_time_slot_with_time_strings(self):
        """Test creating TimeSlot with HH:MM time strings."""
        slot = TimeSlot(day_of_week=0, start="07:30", end="16:30")
        
        assert slot.start_minute == 450  # 7*60 + 30
        assert slot.end_minute == 990    # 16*60 + 30
        assert slot.start_time_string == "07:30"
        assert slot.end_time_string == "16:30"
        
    def test_time_slot_with_minutes_backward_compatibility(self):
        """Test that existing minute format still works."""
        slot = TimeSlot(day_of_week=0, start_minute=450, end_minute=990)
        
        assert slot.start_minute == 450
        assert slot.end_minute == 990
        assert slot.start_time_string == "07:30"
        assert slot.end_time_string == "16:30"
        
    def test_time_slot_mixed_format(self):
        """Test mixed format (one time string, one minutes)."""
        slot = TimeSlot(day_of_week=0, start="09:15", end_minute=1035)
        
        assert slot.start_minute == 555  # 9*60 + 15
        assert slot.end_minute == 1035   # 17*60 + 15
        assert slot.start_time_string == "09:15"
        assert slot.end_time_string == "17:15"
        
    def test_from_time_strings_class_method(self):
        """Test the new from_time_strings class method."""
        slot = TimeSlot.from_time_strings(day_of_week=1, start="08:00", end="17:00")
        
        assert slot.day_of_week == 1
        assert slot.start_minute == 480
        assert slot.end_minute == 1020
        
    def test_time_validation(self):
        """Test that invalid time strings are rejected."""
        with pytest.raises(Exception):  # ValidationError from pydantic
            TimeSlot(day_of_week=0, start="25:00", end="16:00")  # Invalid hour
            
        with pytest.raises(Exception):
            TimeSlot(day_of_week=0, start="10:60", end="16:00")  # Invalid minute
            
    def test_missing_time_fields(self):
        """Test that missing time fields raise appropriate errors."""
        with pytest.raises(ValueError, match="requires start_minute or start field"):
            TimeSlot(day_of_week=0, end="16:00")  # Missing start
            
        with pytest.raises(ValueError, match="requires end_minute or end field"):
            TimeSlot(day_of_week=0, start="08:00")  # Missing end
            
    def test_time_string_properties(self):
        """Test time string properties."""
        slot = TimeSlot(day_of_week=2, start_minute=555, end_minute=1035)
        
        assert slot.start_time_string == "09:15"
        assert slot.end_time_string == "17:15"
        
        # Test edge cases
        slot_midnight = TimeSlot(day_of_week=3, start_minute=0, end_minute=60)
        assert slot_midnight.start_time_string == "00:00"
        assert slot_midnight.end_time_string == "01:00"
        
    def test_integration_with_matching_system(self):
        """Test that time strings work in the complete matching system."""
        # Create test data with time string format
        test_data = {
            "application": {
                "id": "550e8400-e29b-41d4-a716-446655440999",
                "family_id": "550e8400-e29b-41d4-a716-446655441099", 
                "children": [{
                    "id": "550e8400-e29b-41d4-a716-446655441199",
                    "family_id": "550e8400-e29b-41d4-a716-446655441099",
                    "name": "Test Child",
                    "birth_date": "2022-06-15"
                }],
                "home_location": {
                    "latitude": 52.52,
                    "longitude": 13.405,
                    "city": "Berlin",
                    "country_code": "DE"
                },
                "preferences": [],
                "desired_start_date": "2024-09-01",
                "desired_hours": [
                    {"day_of_week": 0, "start": "08:00", "end": "16:00"},
                    {"day_of_week": 1, "start": "08:30", "end": "16:30"}
                ],
                "max_distance_km": 5.0
            },
            "centers": [{
                "id": "550e8400-e29b-41d4-a716-446655441299",
                "name": "Time String Test Center",
                "location": {
                    "latitude": 52.521,
                    "longitude": 13.407,
                    "city": "Berlin",
                    "country_code": "DE"
                },
                "opening_hours": [
                    {"day_of_week": 0, "start": "07:00", "end": "18:00"},
                    {"day_of_week": 1, "start": "07:30", "end": "18:30"}
                ],
                "properties": [],
                "capacity_buckets": [{
                    "id": "550e8400-e29b-41d4-a716-446655441399",
                    "center_id": "550e8400-e29b-41d4-a716-446655441299",
                    "age_band": {
                        "min_age_months": 12,
                        "max_age_months": 48,
                        "name": "Test Group"
                    },
                    "start_month": "2024-09-01",
                    "total_capacity": 10,
                    "available_capacity": 5
                }]
            }],
            "top_k": 3,
            "include_full_centers": False
        }
        
        # Parse and match
        request = EnhancedRecommendationRequest(**test_data)
        
        # Verify time strings were converted correctly
        desired_hour = request.application.desired_hours[0]
        assert desired_hour.start_minute == 480  # 08:00
        assert desired_hour.end_minute == 960    # 16:00
        
        opening_hour = request.centers[0].opening_hours[0] 
        assert opening_hour.start_minute == 420  # 07:00
        assert opening_hour.end_minute == 1080   # 18:00
        
        # Run matching
        builder = MatchingGraphBuilder(
            scorer=CompositeScorer(),
            filter=HardConstraintFilter()
        )
        matcher = GraphMatcher(seed=42)
        
        graph = builder.build_graph(
            applications=[request.application],
            centers=request.centers,
            respect_capacity=True
        )
        
        result = matcher.match(
            graph,
            mode=MatchMode.RECOMMEND,
            application_id=request.application.id,
            top_k=request.top_k
        )
        
        # Should successfully match
        assert result.success
        assert len(result.offers) == 1
        assert result.offers[0].center_id == request.centers[0].id