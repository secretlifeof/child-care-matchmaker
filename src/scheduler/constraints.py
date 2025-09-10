"""
Constraint definitions for the Schedule Optimization Service
"""

import logging
from collections import defaultdict
from datetime import date

from ortools.sat.python import cp_model

from .config import settings
from .models import *

logger = logging.getLogger(__name__)

class ConstraintBuilder:
    """Builds and manages constraints for the schedule optimization problem"""

    def __init__(self, model: cp_model.CpModel):
        self.model = model
        self.constraints = []

    def add_hard_constraints(
        self,
        assignments: dict[tuple[UUID, UUID, int, int], cp_model.IntVar],
        staff: list[Staff],
        groups: list[Group],
        requirements: list[StaffingRequirement],
        time_slots: list[tuple[int, int]],  # (day, hour)
    ):
        """Add all hard constraints that must be satisfied"""

        # 1. Staffing ratio constraints
        self._add_staffing_ratio_constraints(
            assignments, requirements, staff, groups, time_slots
        )

        # 2. Staff availability constraints
        self._add_availability_constraints(assignments, staff, time_slots)

        # 3. Qualification requirements
        self._add_qualification_constraints(
            assignments, staff, requirements, time_slots
        )

        # 4. Maximum working hours per day
        self._add_daily_hour_limits(assignments, staff, time_slots)

        # 5. Maximum working hours per week
        self._add_weekly_hour_limits(assignments, staff, time_slots)

        # 6. Maximum consecutive working hours
        self._add_consecutive_hour_limits(assignments, staff, time_slots)

        # 7. Minimum break between shifts
        self._add_break_requirements(assignments, staff, time_slots)

        # 8. One assignment per staff per time slot
        self._add_single_assignment_constraints(assignments, staff, groups, time_slots)

    def add_soft_constraints(
        self,
        assignments: dict[tuple[UUID, UUID, int, int], cp_model.IntVar],
        staff: list[Staff],
        time_slots: list[tuple[int, int]],
    ) -> list[cp_model.IntVar]:
        """Add soft constraints and return penalty variables for objective function"""

        penalty_vars = []

        # 1. Staff preference violations
        preference_penalties = self._add_preference_constraints(
            assignments, staff, time_slots
        )
        penalty_vars.extend(preference_penalties)

        # 2. Overtime penalties
        overtime_penalties = self._add_overtime_penalties(
            assignments, staff, time_slots
        )
        penalty_vars.extend(overtime_penalties)

        # 3. Fairness penalties (uneven hour distribution)
        fairness_penalties = self._add_fairness_constraints(
            assignments, staff, time_slots
        )
        penalty_vars.extend(fairness_penalties)

        # 4. Shift continuity preferences
        continuity_penalties = self._add_continuity_preferences(
            assignments, staff, time_slots
        )
        penalty_vars.extend(continuity_penalties)

        return penalty_vars

    def _add_staffing_ratio_constraints(
        self,
        assignments: dict[tuple[UUID, UUID, int, int], cp_model.IntVar],
        requirements: list[StaffingRequirement],
        staff: list[Staff],
        groups: list[Group],
        time_slots: list[tuple[int, int]],
    ):
        """
        For each time slot that has requirements, enforce proper staffing levels.
        Handle overlapping requirements by taking the maximum of min_staff and 
        minimum of max_staff across all applicable requirements.
        """

        # Build a mapping from (group_id, day_offset, hour) to list of requirements
        slot_requirements = defaultdict(list)

        # First, we need to map day_of_week from requirements to actual day_offsets
        # We'll iterate through all time slots and check which requirements apply
        for day_offset, hour in time_slots:
            # Convert day_offset back to day_of_week for requirement matching
            # This assumes we have a start date context - we'll need to pass it in
            # For now, let's assume day_offset maps to weekday cyclically
            actual_dow = day_offset % 7

            for req in requirements:
                req_dow = req.time_slot.day_of_week
                req_start_hour = req.time_slot.start_time.hour
                req_end_hour = req.time_slot.end_time.hour

                # Check if this requirement applies to this slot
                if (req_dow == actual_dow and
                    req_start_hour <= hour < req_end_hour):
                    slot_requirements[(req.group_id, day_offset, hour)].append(req)

        # Now enforce constraints for each slot that has requirements
        for (group_id, day_offset, hour), applicable_reqs in slot_requirements.items():
            if not applicable_reqs:
                continue

            # Collect all variables for this slot
            slot_vars = []
            for (sid, gid, d, h), var in assignments.items():
                if gid == group_id and d == day_offset and h == hour:
                    slot_vars.append(var)

            if not slot_vars:
                # No variables for this slot - check if this should cause infeasibility
                min_required = max(req.min_staff_count for req in applicable_reqs)
                if min_required > 0:
                    logger.warning(
                        f"No available staff variables for slot requiring {min_required} staff: "
                        f"group {group_id}, day {day_offset}, hour {hour}"
                    )
                    # Add an impossible constraint to make the problem infeasible
                    # This will help identify the root cause
                    impossible_var = self.model.NewBoolVar(f"impossible_{group_id}_{day_offset}_{hour}")
                    self.model.Add(impossible_var == 1)
                    self.model.Add(impossible_var == 0)  # Contradiction
                continue

            # Aggregate requirements for this slot
            min_staff_required = sum(req.min_staff_count for req in applicable_reqs)
            max_staff_allowed = sum(req.max_staff_count for req in applicable_reqs if req.max_staff_count is not None)

            # If some requirements don't have max_staff_count, use their min as max
            if len([req for req in applicable_reqs if req.max_staff_count is None]) > 0:
                # Add missing max values (use min_staff_count as max for requirements without explicit max)
                missing_max = sum(req.min_staff_count for req in applicable_reqs if req.max_staff_count is None)
                max_staff_allowed += missing_max

            # Ensure we don't exceed available staff
            max_staff_allowed = min(max_staff_allowed, len(slot_vars))

            # Sanity check
            if len(slot_vars) < min_staff_required:
                logger.warning(
                    f"Insufficient available staff: {len(slot_vars)} available but "
                    f"{min_staff_required} required for group {group_id} day {day_offset} hour {hour}"
                )
                # Still add the constraint - this will make the problem infeasible
                # which is the correct behavior

            # Add the constraints
            total_assigned = sum(slot_vars)

            # Minimum staffing constraint
            self.model.Add(total_assigned >= min_staff_required)

            # Maximum staffing constraint
            self.model.Add(total_assigned <= max_staff_allowed)

            logger.debug(
                f"Added staffing constraint for group {group_id}, day {day_offset}, hour {hour}: "
                f"{min_staff_required} <= staff <= {max_staff_allowed} "
                f"(from {len(applicable_reqs)} requirements, {len(slot_vars)} variables)"
            )

    def _add_availability_constraints(
        self,
        assignments: dict[tuple[UUID, UUID, int, int], cp_model.IntVar],
        staff: list[Staff],
        time_slots: list[tuple[int, int]],
    ):
        """Staff cannot be scheduled when unavailable"""

        for staff_member in staff:
            # Build availability map
            available_times = set()
            for avail in staff_member.availability:
                start_hour = avail.start_time.hour
                end_hour = avail.end_time.hour
                if end_hour <= start_hour:  # Overnight availability
                    end_hour += 24

                for hour in range(start_hour, end_hour):
                    if avail.is_available:
                        available_times.add((avail.day_of_week, hour % 24))

            # Add unavailability constraints
            for day, hour in time_slots:
                if (day, hour) not in available_times:
                    # Staff is not available at this time
                    for assignment_key in assignments:
                        if (
                            assignment_key[0] == staff_member.staff_id
                            and assignment_key[2] == day
                            and assignment_key[3] == hour
                        ):
                            self.model.Add(assignments[assignment_key] == 0)

    def _add_qualification_constraints(
        self,
        assignments: dict[tuple[UUID, UUID, int, int], cp_model.IntVar],
        staff: list[Staff],
        requirements: list[StaffingRequirement],
        time_slots: list[tuple[int, int]],
    ):
        """Disallow assignments for staff who lack required qualifications—
        applied over every hour of each explicit staffing requirement."""
        # build a map of each staff member’s VERIFIED qualifications
        staff_quals: dict[UUID, set[str]] = {}
        for sm in staff:
            quals = {
                q.qualification_name
                for q in sm.qualifications
                if q.is_verified and (not q.expiry_date or q.expiry_date > date.today())
            }
            staff_quals[sm.staff_id] = quals

        # for each requirement, for each hour in its window:
        for req in requirements:
            if not req.required_qualifications:
                continue

            day     = req.time_slot.day_of_week
            start_h = req.time_slot.start_time.hour
            end_h   = req.time_slot.end_time.hour

            for hour in range(start_h, end_h):
                for sm in staff:
                    # if staff member lacks one of the required quals:
                    if not set(req.required_qualifications).issubset(staff_quals[sm.staff_id]):
                        ak = (sm.staff_id, req.group_id, day, hour)
                        if ak in assignments:
                            # force them off this assignment
                            self.model.Add(assignments[ak] == 0)


    def _add_daily_hour_limits(
        self,
        assignments: dict[tuple[UUID, UUID, int, int], cp_model.IntVar],
        staff: list[Staff],
        time_slots: list[tuple[int, int]],
    ):
        """Limit daily working hours per staff member"""

        for staff_member in staff:
            max_daily_hours = min(settings.max_consecutive_hours, 12)  # Legal limit

            # Group by day
            days = set(day for day, hour in time_slots)
            for day in days:
                daily_assignments = []
                for assignment_key in assignments:
                    if (
                        assignment_key[0] == staff_member.staff_id
                        and assignment_key[2] == day
                    ):
                        daily_assignments.append(assignments[assignment_key])

                if daily_assignments:
                    self.model.Add(sum(daily_assignments) <= max_daily_hours)

    def _add_weekly_hour_limits(
        self,
        assignments: dict[tuple[UUID, UUID, int, int], cp_model.IntVar],
        staff: list[Staff],
        time_slots: list[tuple[int, int]],
    ):
        """Limit weekly working hours per staff member"""

        for staff_member in staff:
            weekly_assignments = []
            for assignment_key in assignments:
                if assignment_key[0] == staff_member.staff_id:
                    weekly_assignments.append(assignments[assignment_key])

            if weekly_assignments:
                self.model.Add(
                    sum(weekly_assignments) <= int(staff_member.max_weekly_hours)
                )

    def _add_consecutive_hour_limits(
        self,
        assignments: dict[tuple[UUID, UUID, int, int], cp_model.IntVar],
        staff: list[Staff],
        time_slots: list[tuple[int, int]],
    ):
        """Limit consecutive working hours"""

        max_consecutive = settings.max_consecutive_hours

        for staff_member in staff:
            # Sort time slots chronologically
            sorted_slots = sorted(time_slots, key=lambda x: (x[0], x[1]))

            # Check consecutive hours within each day
            for day in range(7):
                day_slots = [slot for slot in sorted_slots if slot[0] == day]

                if len(day_slots) <= max_consecutive:
                    continue

                # Create sliding window of consecutive hours
                for i in range(len(day_slots) - max_consecutive):
                    consecutive_assignments = []
                    for j in range(i, i + max_consecutive + 1):
                        assignment_key = (
                            staff_member.staff_id,
                            None,
                            day_slots[j][0],
                            day_slots[j][1],
                        )
                        # Find actual assignment keys for this staff/time
                        for key in assignments:
                            if (
                                key[0] == staff_member.staff_id
                                and key[2] == day_slots[j][0]
                                and key[3] == day_slots[j][1]
                            ):
                                consecutive_assignments.append(assignments[key])

                    if consecutive_assignments:
                        self.model.Add(sum(consecutive_assignments) <= max_consecutive)

    def _add_break_requirements(
        self,
        assignments: dict[tuple[UUID, UUID, int, int], cp_model.IntVar],
        staff: list[Staff],
        time_slots: list[tuple[int, int]],
    ):
        """Ensure minimum break between shifts"""

        min_break_hours = settings.min_break_between_shifts

        for staff_member in staff:
            # Check across consecutive days
            for day in range(6):  # Monday to Saturday
                today_assignments = []
                tomorrow_assignments = []

                for assignment_key in assignments:
                    if assignment_key[0] == staff_member.staff_id:
                        if assignment_key[2] == day:
                            today_assignments.append(
                                (assignment_key[3], assignments[assignment_key])
                            )
                        elif assignment_key[2] == day + 1:
                            tomorrow_assignments.append(
                                (assignment_key[3], assignments[assignment_key])
                            )

                # If working late today and early tomorrow, ensure break
                if today_assignments and tomorrow_assignments:
                    latest_today = max(hour for hour, _ in today_assignments)
                    earliest_tomorrow = min(hour for hour, _ in tomorrow_assignments)

                    if earliest_tomorrow + 24 - latest_today < min_break_hours:
                        # Create constraint to prevent this scenario
                        late_vars = [
                            var
                            for hour, var in today_assignments
                            if hour >= latest_today
                        ]
                        early_vars = [
                            var
                            for hour, var in tomorrow_assignments
                            if hour <= earliest_tomorrow
                        ]

                        if late_vars and early_vars:
                            # Can't work both late today and early tomorrow
                            self.model.Add(sum(late_vars) + sum(early_vars) <= 1)

    def _add_single_assignment_constraints(
        self,
        assignments: dict[tuple[UUID, UUID, int, int], cp_model.IntVar],
        staff: list[Staff],
        groups: list[Group],
        time_slots: list[tuple[int, int]],
    ):
        """Each staff member can only be assigned to one group per time slot"""

        for staff_member in staff:
            for day, hour in time_slots:
                time_slot_assignments = []
                for group in groups:
                    assignment_key = (staff_member.staff_id, group.group_id, day, hour)
                    if assignment_key in assignments:
                        time_slot_assignments.append(assignments[assignment_key])

                if len(time_slot_assignments) > 1:
                    self.model.Add(sum(time_slot_assignments) <= 1)

    def _add_preference_constraints(
        self,
        assignments: dict[tuple[UUID, UUID, int, int], cp_model.IntVar],
        staff: list[Staff],
        time_slots: list[tuple[int, int]],
    ) -> list[cp_model.IntVar]:
        """Add preference violation penalties"""

        penalty_vars = []

        for staff_member in staff:
            for preference in staff_member.preferences:
                if preference.preference_type == PreferenceType.UNAVAILABLE:
                    # Handle unavailable preferences (should be hard constraint)
                    continue
                elif preference.preference_type == PreferenceType.PREFERRED_TIME:
                    penalty_var = self._create_preference_penalty(
                        assignments,
                        staff_member,
                        preference,
                        time_slots,
                        is_positive=True,
                    )
                    penalty_vars.append(penalty_var)
                elif preference.preference_type == PreferenceType.EXCLUDE_DAYS:
                    penalty_var = self._create_preference_penalty(
                        assignments,
                        staff_member,
                        preference,
                        time_slots,
                        is_positive=False,
                    )
                    penalty_vars.append(penalty_var)

        return penalty_vars

    def _create_preference_penalty(
        self,
        assignments: dict[tuple[UUID, UUID, int, int], cp_model.IntVar],
        staff_member: Staff,
        preference: StaffPreference,
        time_slots: list[tuple[int, int]],
        is_positive: bool,
    ) -> cp_model.IntVar:
        """Create penalty variable for preference violations"""

        penalty_var = self.model.NewIntVar(
            0, 24 * 7, f"penalty_{staff_member.staff_id}_{preference.preference_type}"
        )

        relevant_assignments = []
        for assignment_key in assignments:
            if assignment_key[0] == staff_member.staff_id:
                day, hour = assignment_key[2], assignment_key[3]

                # Check if this assignment matches the preference criteria
                matches = True
                if preference.day_of_week is not None and day != preference.day_of_week:
                    matches = False
                if preference.time_range_start and preference.time_range_end:
                    if not (
                        preference.time_range_start.hour
                        <= hour
                        <= preference.time_range_end.hour
                    ):
                        matches = False

                if matches:
                    relevant_assignments.append(assignments[assignment_key])

        if relevant_assignments:
            if is_positive:
                # Penalty for NOT being assigned to preferred times
                max_preferred = len(relevant_assignments)
                actual_assigned = sum(relevant_assignments)
                self.model.Add(penalty_var == max_preferred - actual_assigned)
            else:
                # Penalty for being assigned to avoided times
                self.model.Add(penalty_var == sum(relevant_assignments))

        return penalty_var

    def _add_overtime_penalties(
        self,
        assignments: dict[tuple[UUID, UUID, int, int], cp_model.IntVar],
        staff: list[Staff],
        time_slots: list[tuple[int, int]],
    ) -> list[cp_model.IntVar]:
        """Add penalties for overtime hours"""

        penalty_vars = []

        for staff_member in staff:
            overtime_var = self.model.NewIntVar(
                0, 40, f"overtime_{staff_member.staff_id}"
            )

            total_assignments = []
            for assignment_key in assignments:
                if assignment_key[0] == staff_member.staff_id:
                    total_assignments.append(assignments[assignment_key])

            if total_assignments:
                total_hours = sum(total_assignments)
                regular_hours = int(min(staff_member.max_weekly_hours, 40))

                # Overtime is hours beyond regular hours
                self.model.Add(overtime_var >= total_hours - regular_hours)
                self.model.Add(overtime_var >= 0)

                penalty_vars.append(overtime_var)

        return penalty_vars

    def _add_fairness_constraints(
        self,
        assignments: dict[tuple[UUID, UUID, int, int], cp_model.IntVar],
        staff: list[Staff],
        time_slots: list[tuple[int, int]],
    ) -> list[cp_model.IntVar]:
        """Add penalties for unfair hour distribution"""

        penalty_vars = []

        # Calculate average hours per staff member
        total_possible_hours = len(time_slots)
        avg_hours_per_staff = total_possible_hours / len(staff)

        for staff_member in staff:
            fairness_penalty = self.model.NewIntVar(
                0, total_possible_hours, f"fairness_{staff_member.staff_id}"
            )

            staff_assignments = []
            for assignment_key in assignments:
                if assignment_key[0] == staff_member.staff_id:
                    staff_assignments.append(assignments[assignment_key])

            if staff_assignments:
                staff_hours = sum(staff_assignments)
                target_hours = int(avg_hours_per_staff * staff_member.priority_score)

                # Penalty for deviation from target hours
                deviation_pos = self.model.NewIntVar(
                    0, total_possible_hours, f"dev_pos_{staff_member.staff_id}"
                )
                deviation_neg = self.model.NewIntVar(
                    0, total_possible_hours, f"dev_neg_{staff_member.staff_id}"
                )

                self.model.Add(
                    staff_hours - target_hours == deviation_pos - deviation_neg
                )
                self.model.Add(fairness_penalty == deviation_pos + deviation_neg)

                penalty_vars.append(fairness_penalty)

        return penalty_vars

    def _add_continuity_preferences(
        self,
        assignments: dict[tuple[UUID, UUID, int, int], cp_model.IntVar],
        staff: list[Staff],
        time_slots: list[tuple[int, int]],
    ) -> list[cp_model.IntVar]:
        """Add penalties for fragmented schedules"""

        penalty_vars = []

        for staff_member in staff:
            continuity_penalty = self.model.NewIntVar(
                0, 50, f"continuity_{staff_member.staff_id}"
            )

            # Count shift changes (penalty for each separate shift block)
            shift_changes = 0

            for day in range(7):
                day_assignments = []
                for assignment_key in assignments:
                    if (
                        assignment_key[0] == staff_member.staff_id
                        and assignment_key[2] == day
                    ):
                        day_assignments.append(
                            (assignment_key[3], assignments[assignment_key])
                        )

                if len(day_assignments) > 1:
                    # Sort by hour
                    day_assignments.sort(key=lambda x: x[0])

                    # Count gaps in the schedule (separate shift blocks)
                    for i in range(len(day_assignments) - 1):
                        current_hour, current_var = day_assignments[i]
                        next_hour, next_var = day_assignments[i + 1]

                        if next_hour > current_hour + 1:
                            # There's a gap - penalty for discontinuity
                            gap_penalty = self.model.NewBoolVar(
                                f"gap_{staff_member.staff_id}_{day}_{i}"
                            )
                            self.model.Add(current_var + next_var - 1 <= gap_penalty)
                            shift_changes += gap_penalty

            penalty_vars.append(continuity_penalty)

        return penalty_vars

    def _calculate_group_ratios(self, groups: list[Group]) -> dict[AgeGroup, int]:
        """Calculate staff-to-child ratios by age group"""
        return {
            AgeGroup.INFANT: settings.infant_ratio,
            AgeGroup.TODDLER: settings.toddler_ratio,
            AgeGroup.PRESCHOOL: settings.preschool_ratio,
            AgeGroup.MIXED: settings.toddler_ratio,  # Use toddler ratio for mixed groups
        }
