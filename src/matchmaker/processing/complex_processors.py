"""Processors for complex preference types."""

import math
from abc import ABC, abstractmethod
from typing import Any

from ..models.base import Center, Location, ParentPreference


class ComplexPreferenceProcessor(ABC):
    """Base class for complex preference processors."""

    @abstractmethod
    def process(self, preference: ParentPreference, center: Center) -> tuple[float, str]:
        """
        Process a complex preference against a center.
        
        Returns:
            Tuple of (score, explanation)
        """
        pass

    @abstractmethod
    def get_type_name(self) -> str:
        """Get the complex type name this processor handles."""
        pass


class LocationDistanceProcessor(ComplexPreferenceProcessor):
    """Processor for location_distance complex preferences."""

    def get_type_name(self) -> str:
        return "location_distance"

    def process(self, preference: ParentPreference, center: Center) -> tuple[float, str]:
        """Process multi-location distance constraints."""
        if not preference.value_data:
            return 0.0, "No location data provided"

        locations = preference.value_data.get('locations', [])
        if not locations:
            return 0.0, "No locations specified"

        scores = []
        explanations = []

        for location in locations:
            distance = self._calculate_distance(location, center.location)
            max_dist = location.get('max_distance_km', 10.0)
            preferred_dist = location.get('preferred_distance_km', max_dist * 0.6)

            if distance > max_dist:
                scores.append(0.0)
                explanations.append(f"{location['name']}: too far ({distance:.1f}km > {max_dist}km)")
            elif distance <= preferred_dist:
                scores.append(1.0)
                explanations.append(f"{location['name']}: ideal distance ({distance:.1f}km)")
            else:
                # Linear decay between preferred and max
                score = 1.0 - (distance - preferred_dist) / (max_dist - preferred_dist)
                scores.append(score)
                explanations.append(f"{location['name']}: acceptable ({distance:.1f}km)")

        # Overall score is the minimum (all locations must be satisfied)
        overall_score = min(scores) if scores else 0.0
        explanation = "; ".join(explanations)

        return overall_score, explanation

    def _calculate_distance(self, location_data: dict[str, Any], center_location: Location) -> float:
        """Calculate distance between two locations using Haversine formula."""
        if 'latitude' in location_data and 'longitude' in location_data:
            lat1, lon1 = location_data['latitude'], location_data['longitude']
        else:
            # In production, would geocode the address
            # For now, return a default distance
            return 5.0

        lat2, lon2 = center_location.latitude, center_location.longitude

        # Haversine formula
        R = 6371  # Earth's radius in km
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        distance = R * c

        return distance


class ScheduleRangeProcessor(ComplexPreferenceProcessor):
    """Processor for schedule_range complex preferences."""

    def get_type_name(self) -> str:
        return "schedule_range"

    def process(self, preference: ParentPreference, center: Center) -> tuple[float, str]:
        """Process flexible schedule requirements."""
        if not preference.value_data:
            return 0.0, "No schedule data provided"

        start_time_str = preference.value_data.get('start_time', '08:00')
        end_time_str = preference.value_data.get('end_time', '18:00')
        flexibility_minutes = preference.value_data.get('flexibility_minutes', 0)
        days_of_week = preference.value_data.get('days_of_week',
                                                  ['monday', 'tuesday', 'wednesday', 'thursday', 'friday'])

        # Convert time strings to minutes
        start_minutes = self._time_to_minutes(start_time_str) - flexibility_minutes
        end_minutes = self._time_to_minutes(end_time_str) + flexibility_minutes

        # Check if center hours accommodate the schedule
        day_scores = []
        day_map = {'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
                   'friday': 4, 'saturday': 5, 'sunday': 6}

        for day in days_of_week:
            day_num = day_map.get(day.lower(), 0)
            day_slots = [slot for slot in center.opening_hours if slot.day_of_week == day_num]

            if not day_slots:
                day_scores.append(0.0)
                continue

            # Check if any slot covers the required time
            covered = False
            for slot in day_slots:
                if slot.start_minute <= start_minutes and slot.end_minute >= end_minutes:
                    covered = True
                    break

            day_scores.append(1.0 if covered else 0.0)

        # Overall score is the average across all required days
        overall_score = sum(day_scores) / len(day_scores) if day_scores else 0.0

        if overall_score == 1.0:
            explanation = f"Fully supports schedule {start_time_str}-{end_time_str}"
        elif overall_score > 0:
            explanation = f"Partially supports schedule ({overall_score*100:.0f}% of days)"
        else:
            explanation = f"Does not support required schedule {start_time_str}-{end_time_str}"

        if flexibility_minutes > 0:
            explanation += f" (with {flexibility_minutes}min flexibility)"

        return overall_score, explanation

    def _time_to_minutes(self, time_str: str) -> int:
        """Convert HH:MM to minutes from start of day."""
        hours, minutes = map(int, time_str.split(':'))
        return hours * 60 + minutes


class SocialConnectionProcessor(ComplexPreferenceProcessor):
    """Processor for social_connection complex preferences."""

    def __init__(self, db_manager=None):
        self.db_manager = db_manager

    def get_type_name(self) -> str:
        return "social_connection"

    async def process_async(self, preference: ParentPreference, center: Center) -> tuple[float, str]:
        """Process social connection requirements (async for database queries)."""
        if not preference.value_data:
            return 0.0, "No social connection data provided"

        connection_type = preference.value_data.get('connection_type', 'sibling')
        entity_ids = preference.value_data.get('entity_ids', [])
        same_center_required = preference.value_data.get('same_center_required', False)
        same_group_required = preference.value_data.get('same_group_required', False)

        if not entity_ids:
            return 0.0, "No connected entities specified"

        # In production, query database for entity placements
        # For now, simulate
        if connection_type == 'sibling':
            # Check if siblings are at this center
            if same_center_required:
                # Would query: SELECT * FROM enrollments WHERE child_id IN entity_ids AND center_id = center.id
                siblings_at_center = len(entity_ids) * 0.3  # Simulate 30% chance
                if siblings_at_center > 0:
                    return 1.0, "Siblings already enrolled at this center"
                else:
                    return 0.5, "Siblings not currently at this center (priority given)"
            else:
                return 0.8, "Sibling priority applied"

        return 0.5, f"{connection_type} connection considered"

    def process(self, preference: ParentPreference, center: Center) -> tuple[float, str]:
        """Synchronous version with limited functionality."""
        if not preference.value_data:
            return 0.0, "No social connection data provided"

        connection_type = preference.value_data.get('connection_type', 'sibling')
        priority_level = preference.value_data.get('priority_level', 'preferred')

        # Basic scoring based on connection type
        base_scores = {
            'sibling': 0.8,
            'friend': 0.5,
            'family': 0.6
        }

        score = base_scores.get(connection_type, 0.3)

        # Adjust based on priority level
        if priority_level == 'required':
            score = min(1.0, score * 1.2)
        elif priority_level == 'nice_to_have':
            score *= 0.7

        explanation = f"{connection_type} connection preference ({priority_level})"
        return score, explanation


class EducationalApproachProcessor(ComplexPreferenceProcessor):
    """Processor for educational_approach complex preferences."""

    def get_type_name(self) -> str:
        return "educational_approach"

    def process(self, preference: ParentPreference, center: Center) -> tuple[float, str]:
        """Process educational approach preferences."""
        if not preference.value_data:
            return 0.0, "No educational approach data provided"

        requested_approaches = preference.value_data.get('approaches', [])
        importance_weights = preference.value_data.get('importance_weights', {})
        must_have_all = preference.value_data.get('must_have_all', False)
        certification_required = preference.value_data.get('certification_required', False)

        if not requested_approaches:
            return 0.0, "No approaches specified"

        # Check which approaches the center offers
        # In production, this would check center properties
        center_approaches = self._get_center_approaches(center)

        matched_approaches = []
        scores = []

        for approach in requested_approaches:
            weight = importance_weights.get(approach, 1.0)
            if approach in center_approaches:
                matched_approaches.append(approach)
                scores.append(weight)
            elif must_have_all:
                return 0.0, f"Missing required approach: {approach}"
            else:
                scores.append(0.0)

        if not scores:
            return 0.0, "No approaches to evaluate"

        # Calculate weighted average
        total_weight = sum(importance_weights.get(a, 1.0) for a in requested_approaches)
        overall_score = sum(scores) / total_weight if total_weight > 0 else 0.0

        if matched_approaches:
            explanation = f"Offers: {', '.join(matched_approaches)}"
            if certification_required:
                # Check for certifications
                has_cert = self._check_certifications(center, matched_approaches)
                if not has_cert:
                    overall_score *= 0.5
                    explanation += " (no certification)"
        else:
            explanation = "None of the requested approaches available"

        return overall_score, explanation

    def _get_center_approaches(self, center: Center) -> list[str]:
        """Extract educational approaches from center properties."""
        approaches = []
        approach_keywords = ['montessori', 'waldorf', 'reggio', 'steiner', 'play-based', 'academic']

        for prop in center.properties:
            if prop.property_key and any(keyword in prop.property_key.lower() for keyword in approach_keywords):
                # Extract the approach name from the property
                for keyword in approach_keywords:
                    if keyword in prop.property_key.lower():
                        approaches.append(keyword)

        return approaches

    def _check_certifications(self, center: Center, approaches: list[str]) -> bool:
        """Check if center has certifications for the approaches."""
        # In production, would check specific certification properties
        for prop in center.properties:
            if 'cert' in prop.property_key.lower():
                for approach in approaches:
                    if approach in prop.property_key.lower():
                        return True
        return False


class RoutePreferenceProcessor(ComplexPreferenceProcessor):
    """Processor for route_preference complex preferences."""

    def get_type_name(self) -> str:
        return "route_preference"

    def process(self, preference: ParentPreference, center: Center) -> tuple[float, str]:
        """Process commute route preferences."""
        if not preference.value_data:
            return 0.0, "No route data provided"

        start_location = preference.value_data.get('start_location', {})
        end_location = preference.value_data.get('end_location', {})
        max_detour_minutes = preference.value_data.get('max_detour_minutes', 15)
        transport_mode = preference.value_data.get('transport_mode', 'car')

        if not start_location or not end_location:
            return 0.0, "Start or end location not specified"

        # Calculate detour time
        # In production, would use routing API (Google Maps, etc.)
        direct_time = self._estimate_travel_time(start_location, end_location, transport_mode)
        with_detour_time = (self._estimate_travel_time(start_location,
                                                       {'latitude': center.location.latitude,
                                                        'longitude': center.location.longitude},
                                                       transport_mode) +
                           self._estimate_travel_time({'latitude': center.location.latitude,
                                                      'longitude': center.location.longitude},
                                                      end_location, transport_mode))

        detour_minutes = with_detour_time - direct_time

        if detour_minutes <= max_detour_minutes:
            if detour_minutes <= max_detour_minutes * 0.5:
                score = 1.0
                explanation = f"Minimal detour ({detour_minutes:.0f}min) via {transport_mode}"
            else:
                score = 1.0 - (detour_minutes - max_detour_minutes * 0.5) / (max_detour_minutes * 0.5)
                explanation = f"Acceptable detour ({detour_minutes:.0f}min) via {transport_mode}"
        else:
            score = 0.0
            explanation = f"Detour too long ({detour_minutes:.0f}min > {max_detour_minutes}min)"

        return score, explanation

    def _estimate_travel_time(self, from_loc: dict[str, Any], to_loc: dict[str, Any], mode: str) -> float:
        """Estimate travel time between locations."""
        # Simple distance-based estimation
        # In production, use actual routing API

        if 'latitude' in from_loc and 'longitude' in from_loc and 'latitude' in to_loc and 'longitude' in to_loc:
            distance = self._calculate_distance(from_loc['latitude'], from_loc['longitude'],
                                              to_loc['latitude'], to_loc['longitude'])
        else:
            # Default estimate
            distance = 5.0

        # Speed estimates by transport mode (km/h)
        speeds = {
            'car': 30,
            'public_transport': 25,
            'bike': 15,
            'walk': 5
        }

        speed = speeds.get(mode, 30)
        return (distance / speed) * 60  # Convert to minutes

    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance using Haversine formula."""
        R = 6371  # Earth's radius in km
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        return R * c


class ComplexPreferenceProcessorFactory:
    """Factory for creating complex preference processors."""

    def __init__(self, db_manager=None):
        self.db_manager = db_manager
        self.processors = {
            'location_distance': LocationDistanceProcessor(),
            'schedule_range': ScheduleRangeProcessor(),
            'social_connection': SocialConnectionProcessor(db_manager),
            'educational_approach': EducationalApproachProcessor(),
            'route_preference': RoutePreferenceProcessor()
        }

    def get_processor(self, type_name: str) -> ComplexPreferenceProcessor | None:
        """Get processor for a specific complex type."""
        return self.processors.get(type_name)

    def process_preference(self, preference: ParentPreference, center: Center) -> tuple[float, str]:
        """Process any complex preference."""
        if not preference.value_data or not preference.complex_value_type_id:
            return 0.0, "Not a complex preference"

        # Determine type from value_data or lookup
        type_name = None
        if 'locations' in preference.value_data:
            type_name = 'location_distance'
        elif 'start_time' in preference.value_data and 'end_time' in preference.value_data:
            type_name = 'schedule_range'
        elif 'connection_type' in preference.value_data:
            type_name = 'social_connection'
        elif 'approaches' in preference.value_data:
            type_name = 'educational_approach'
        elif 'start_location' in preference.value_data:
            type_name = 'route_preference'

        if type_name:
            processor = self.get_processor(type_name)
            if processor:
                return processor.process(preference, center)

        return 0.0, "Unknown complex preference type"
