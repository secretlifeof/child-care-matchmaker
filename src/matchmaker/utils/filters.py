"""Hard constraint filters for matching."""

import logging

from ..models.base import (
    Application,
    CapacityBucket,
    Center,
    ComparisonOperator,
    PreferenceStrength,
    PropertyType,
)

logger = logging.getLogger(__name__)


class HardConstraintFilter:
    """Applies hard constraints to filter invalid matches."""

    def __init__(self):
        self.constraints_checked = 0
        self.constraints_violated = 0

    def is_valid_match(
        self,
        application: Application,
        center: Center,
        bucket: CapacityBucket
    ) -> bool:
        """
        Check if application-center-bucket combination satisfies all hard constraints.
        
        Returns:
            True if all hard constraints are satisfied
        """
        self.constraints_checked += 1

        # Age constraint
        if not self._check_age_constraint(application, bucket):
            self.constraints_violated += 1
            return False

        # Distance constraint
        if not self._check_distance_constraint(application, center):
            self.constraints_violated += 1
            return False

        # Opening hours constraint
        if not self._check_hours_constraint(application, center):
            self.constraints_violated += 1
            return False

        # Start date constraint
        if not self._check_start_date_constraint(application, bucket):
            self.constraints_violated += 1
            return False

        # Exclusion constraints
        if not self._check_exclusion_constraints(application, center):
            self.constraints_violated += 1
            return False

        # Must-have constraints
        if not self._check_must_have_constraints(application, center):
            self.constraints_violated += 1
            return False

        # Reserved bucket constraint
        if not self._check_reserved_bucket_constraint(application, bucket):
            self.constraints_violated += 1
            return False

        return True

    def _check_age_constraint(
        self,
        application: Application,
        bucket: CapacityBucket
    ) -> bool:
        """Check if children's ages match bucket age band."""
        for child in application.children:
            if not bucket.age_band.contains_age(child.age_months):
                logger.debug(
                    f"Age constraint violated: child {child.id} "
                    f"age {child.age_months} months not in range "
                    f"{bucket.age_band.min_age_months}-{bucket.age_band.max_age_months}"
                )
                return False
        return True

    def _check_distance_constraint(
        self,
        application: Application,
        center: Center
    ) -> bool:
        """Check if center is within max distance."""
        if not application.max_distance_km:
            return True

        from geopy.distance import geodesic
        distance = geodesic(
            (application.home_location.latitude, application.home_location.longitude),
            (center.location.latitude, center.location.longitude)
        ).kilometers

        if distance > application.max_distance_km:
            logger.debug(
                f"Distance constraint violated: {distance:.1f}km > "
                f"{application.max_distance_km}km max"
            )
            return False

        return True

    def _check_hours_constraint(
        self,
        application: Application,
        center: Center
    ) -> bool:
        """Check if center hours accommodate desired hours."""
        if not application.desired_hours:
            return True

        for app_slot in application.desired_hours:
            slot_covered = False
            for center_slot in center.opening_hours:
                if self._slot_contains(center_slot, app_slot):
                    slot_covered = True
                    break

            if not slot_covered:
                logger.debug(
                    f"Hours constraint violated: desired slot "
                    f"{app_slot.day_of_week} {app_slot.start_hour}:00-"
                    f"{app_slot.end_hour}:00 not covered"
                )
                return False

        return True

    def _slot_contains(self, container, contained):
        """Check if one time slot fully contains another."""
        if container.day_of_week != contained.day_of_week:
            return False

        container_start = container.start_hour * 60 + container.start_minute
        container_end = container.end_hour * 60 + container.end_minute
        contained_start = contained.start_hour * 60 + contained.start_minute
        contained_end = contained.end_hour * 60 + contained.end_minute

        return (container_start <= contained_start and
                container_end >= contained_end)

    def _check_start_date_constraint(
        self,
        application: Application,
        bucket: CapacityBucket
    ) -> bool:
        """Check if bucket start date is acceptable."""
        # Allow buckets starting on or after desired date
        if bucket.start_month < application.desired_start_date:
            logger.debug(
                f"Start date constraint violated: bucket starts "
                f"{bucket.start_month} before desired {application.desired_start_date}"
            )
            return False

        # Could add max delay constraint here
        return True

    def _check_exclusion_constraints(
        self,
        application: Application,
        center: Center
    ) -> bool:
        """Check exclusion preferences using new strength system."""
        for pref in application.preferences:
            # Check both old threshold system and new strength system
            is_exclusion = (pref.strength == PreferenceStrength.EXCLUDE or
                           pref.threshold <= 0.1)

            if is_exclusion:
                prop = center.get_property(pref.property_key)
                if prop and self._matches_preference_value(pref, prop):
                    logger.debug(
                        f"Exclusion constraint violated: center has "
                        f"{pref.property_key}={prop.get_value()}, which should be avoided"
                    )
                    return False
        return True

    def _check_must_have_constraints(
        self,
        application: Application,
        center: Center
    ) -> bool:
        """Check must-have preferences using new strength system."""
        for pref in application.preferences:
            # Check both old threshold system and new strength system
            is_required = (pref.strength == PreferenceStrength.REQUIRED or
                          pref.threshold >= 0.9)

            if is_required:
                prop = center.get_property(pref.property_key)
                if not prop or not self._matches_preference_value(pref, prop):
                    logger.debug(
                        f"Required constraint violated: center lacks "
                        f"required property {pref.property_key}"
                    )
                    return False
        return True

    def _check_reserved_bucket_constraint(
        self,
        application: Application,
        bucket: CapacityBucket
    ) -> bool:
        """Check if application qualifies for reserved bucket."""
        if not bucket.reserved_label:
            return True  # Unreserved bucket

        if bucket.reserved_label not in application.priority_flags:
            logger.debug(
                f"Reserved bucket constraint violated: application lacks "
                f"flag '{bucket.reserved_label}'"
            )
            return False

        return True

    def _matches_preference_value(self, preference, center_property) -> bool:
        """Check if property matches preference criteria with extended property types."""
        if preference.operator == ComparisonOperator.EQUALS:
            return center_property.get_value() == preference.get_value()

        elif preference.operator == ComparisonOperator.GREATER_THAN:
            if (center_property.value_numeric is not None and
                preference.value_numeric is not None):
                return center_property.value_numeric > preference.value_numeric

        elif preference.operator == ComparisonOperator.LESS_THAN:
            if (center_property.value_numeric is not None and
                preference.value_numeric is not None):
                return center_property.value_numeric < preference.value_numeric

        elif preference.operator == ComparisonOperator.BETWEEN:
            if (center_property.value_numeric is not None and
                preference.min_value is not None and
                preference.max_value is not None):
                return (preference.min_value <= center_property.value_numeric <=
                       preference.max_value)

        elif preference.operator == ComparisonOperator.CONTAINS:
            # Handle both text and list types
            if preference.property_type == PropertyType.LIST:
                if (center_property.value_list and preference.value_list):
                    return any(item in center_property.value_list
                              for item in preference.value_list)
            elif preference.property_type == PropertyType.TEXT:
                if (center_property.value_text and preference.value_text):
                    return preference.value_text.lower() in center_property.value_text.lower()

        elif preference.operator == ComparisonOperator.NOT_CONTAINS:
            if preference.property_type == PropertyType.LIST:
                if (center_property.value_list and preference.value_list):
                    return not any(item in center_property.value_list
                                  for item in preference.value_list)
            elif preference.property_type == PropertyType.TEXT:
                if (center_property.value_text and preference.value_text):
                    return preference.value_text.lower() not in center_property.value_text.lower()

        return False

    def get_statistics(self) -> dict:
        """Get filter statistics."""
        return {
            "constraints_checked": self.constraints_checked,
            "constraints_violated": self.constraints_violated,
            "violation_rate": (self.constraints_violated / self.constraints_checked
                              if self.constraints_checked > 0 else 0)
        }
