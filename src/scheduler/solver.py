"""
OR-Tools CP-SAT solver for schedule optimization with flexible date ranges
"""

import time
from datetime import datetime, timedelta, time as datetime_time, date
from typing import List, Dict, Tuple, Optional, Set
from ortools.sat.python import cp_model
import logging
from uuid import UUID

from .models import *
from .constraints import ConstraintBuilder
from .config import settings


logger = logging.getLogger(__name__)


class ScheduleSolver:
    """Main solver class using OR-Tools CP-SAT with flexible date range support"""

    def __init__(self, optimization_config: OptimizationConfig):
        self.config = optimization_config
        self.model = cp_model.CpModel()
        self.solver = cp_model.CpSolver()
        self.constraint_builder = ConstraintBuilder(self.model)

        # Configure solver parameters
        self.solver.parameters.max_time_in_seconds = optimization_config.max_solver_time
        self.solver.parameters.num_search_workers = 4
        self.solver.parameters.log_search_progress = True

    def solve(
        self,
        staff: List[Staff],
        groups: List[Group],
        requirements: List[StaffingRequirement],
        constraints: List[ScheduleConstraint],
        schedule_start_date: date,
        schedule_end_date: date,
        existing_schedule: Optional[List[ScheduledShift]] = None,
    ) -> Tuple[List[ScheduledShift], OptimizationResult, List[ScheduleConflict]]:
        """
        Solve the schedule optimization problem for a flexible date range

        Args:
            schedule_start_date: Start date of the scheduling period
            schedule_end_date: End date of the scheduling period (inclusive)

        Returns:
            Tuple of (schedule, optimization_result, conflicts)
        """

        start_time = time.time()

        try:
            # Validate date range
            if schedule_end_date < schedule_start_date:
                raise ValueError("End date must be after start date")

            # Calculate scheduling period
            total_days = (schedule_end_date - schedule_start_date).days + 1
            if total_days > 365:  # Reasonable limit
                raise ValueError(f"Date range too large: {total_days} days (max 365)")

            logger.info(
                f"Solving schedule for {total_days} days: {schedule_start_date} to {schedule_end_date}"
            )

            # Generate time slots for the date range
            time_slots = self._generate_time_slots(
                schedule_start_date, schedule_end_date
            )

            # Create decision variables
            assignments = self._create_decision_variables(staff, groups, time_slots)

            # Add hard constraints
            self.constraint_builder.add_hard_constraints(
                assignments,
                staff,
                groups,
                requirements,
                time_slots,
                schedule_start_date,
                schedule_end_date,
            )

            # Add soft constraints and get penalty variables
            penalty_vars = []
            if self.config.consider_preferences:
                penalty_vars.extend(
                    self.constraint_builder.add_soft_constraints(
                        assignments, staff, time_slots, schedule_start_date
                    )
                )

            # Set objective function
            self._set_objective(assignments, staff, groups, penalty_vars, time_slots)

            # Handle existing schedule if provided
            if existing_schedule:
                self._add_existing_schedule_constraints(
                    assignments,
                    existing_schedule,
                    schedule_start_date,
                    schedule_end_date,
                )

            # Solve the model
            status = self.solver.Solve(self.model)
            solve_time = time.time() - start_time

            # Process results
            if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
                schedule = self._extract_schedule(
                    assignments, staff, groups, time_slots, schedule_start_date
                )
                conflicts = self._detect_conflicts(
                    schedule, staff, groups, requirements
                )

                optimization_result = OptimizationResult(
                    objective_value=self.solver.ObjectiveValue(),
                    solve_time_seconds=solve_time,
                    status="OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE",
                    iterations=self.solver.NumBranches(),
                    conflicts_resolved=len(conflicts),
                )

                logger.info(f"Schedule solved successfully in {solve_time:.2f} seconds")
                return schedule, optimization_result, conflicts

            else:
                # Problem is infeasible or timeout
                status_name = (
                    "INFEASIBLE" if status == cp_model.INFEASIBLE else "TIMEOUT"
                )

                optimization_result = OptimizationResult(
                    objective_value=0,
                    solve_time_seconds=solve_time,
                    status=status_name,
                    iterations=self.solver.NumBranches(),
                    conflicts_resolved=0,
                )

                conflicts = self._analyze_infeasibility(
                    staff, groups, requirements, time_slots
                )

                logger.warning(f"Schedule optimization failed: {status_name}")
                return [], optimization_result, conflicts

        except Exception as e:
            logger.error(f"Error in schedule optimization: {str(e)}")

            optimization_result = OptimizationResult(
                objective_value=0,
                solve_time_seconds=time.time() - start_time,
                status="ERROR",
                iterations=0,
                conflicts_resolved=0,
            )

            conflicts = [
                ScheduleConflict(
                    conflict_type="solver_error",
                    severity="error",
                    description=f"Solver error: {str(e)}",
                    suggested_solutions=[
                        "Check input data",
                        "Reduce problem size",
                        "Relax constraints",
                    ],
                )
            ]

            return [], optimization_result, conflicts

    def _generate_time_slots(
        self, start_date: date, end_date: date
    ) -> List[Tuple[int, int]]:
        """Generate all time slots for the specified date range (day_offset, hour pairs)"""
        time_slots = []

        # Operating hours: 6 AM to 8 PM (14 hours per day)
        start_hour = 6
        end_hour = 20

        # Calculate total days
        total_days = (end_date - start_date).days + 1

        for day_offset in range(total_days):
            for hour in range(start_hour, end_hour):
                time_slots.append((day_offset, hour))

        logger.info(f"Generated {len(time_slots)} time slots for {total_days} days")
        return time_slots

    def _create_decision_variables(
        self, staff: List[Staff], groups: List[Group], time_slots: List[Tuple[int, int]]
    ) -> Dict[Tuple[UUID, UUID, int, int], cp_model.IntVar]:
        """Create binary decision variables for staff assignments"""

        assignments = {}

        for staff_member in staff:
            for group in groups:
                for day_offset, hour in time_slots:
                    var_name = f"assign_{staff_member.staff_id}_{group.group_id}_{day_offset}_{hour}"
                    assignments[
                        (staff_member.staff_id, group.group_id, day_offset, hour)
                    ] = self.model.NewBoolVar(var_name)

        logger.info(f"Created {len(assignments)} decision variables")
        return assignments

    def _set_objective(
        self,
        assignments: Dict[Tuple[UUID, UUID, int, int], cp_model.IntVar],
        staff: List[Staff],
        groups: List[Group],
        penalty_vars: List[cp_model.IntVar],
        time_slots: List[Tuple[int, int]],
    ):
        """Set the objective function based on optimization goals"""

        objective_terms = []

        for goal in self.config.goals:
            if goal == OptimizationGoal.MINIMIZE_COST:
                cost_terms = self._create_cost_terms(assignments, staff, time_slots)
                objective_terms.extend([-term for term in cost_terms])  # Minimize cost

            elif goal == OptimizationGoal.MAXIMIZE_SATISFACTION:
                satisfaction_terms = self._create_satisfaction_terms(
                    assignments, staff, time_slots
                )
                objective_terms.extend(satisfaction_terms)  # Maximize satisfaction

            elif goal == OptimizationGoal.MINIMIZE_OVERTIME:
                overtime_terms = self._create_overtime_terms(
                    assignments, staff, time_slots
                )
                objective_terms.extend(
                    [-term for term in overtime_terms]
                )  # Minimize overtime

            elif goal == OptimizationGoal.MAXIMIZE_FAIRNESS:
                fairness_terms = self._create_fairness_terms(
                    assignments, staff, time_slots
                )
                objective_terms.extend(fairness_terms)  # Maximize fairness

            elif goal == OptimizationGoal.MAXIMIZE_CONTINUITY:
                continuity_terms = self._create_continuity_terms(
                    assignments, staff, time_slots
                )
                objective_terms.extend(continuity_terms)  # Maximize continuity

        # Add penalty terms (scaled by weights)
        for penalty_var in penalty_vars:
            objective_terms.append(-int(settings.preference_weight * 100) * penalty_var)

        if objective_terms:
            self.model.Maximize(sum(objective_terms))
            logger.info(f"Objective function created with {len(objective_terms)} terms")
        else:
            # Default objective: minimize total assignments (prefer fewer hours)
            total_assignments = sum(assignments.values())
            self.model.Minimize(total_assignments)

    def _create_cost_terms(
        self,
        assignments: Dict[Tuple[UUID, UUID, int, int], cp_model.IntVar],
        staff: List[Staff],
        time_slots: List[Tuple[int, int]],
    ) -> List[cp_model.IntVar]:
        """Create cost terms for the objective function"""

        cost_terms = []

        for staff_member in staff:
            if staff_member.hourly_rate:
                staff_cost = 0
                for assignment_key in assignments:
                    if assignment_key[0] == staff_member.staff_id:
                        # Scale hourly rate to integer (e.g., $15.50 -> 1550)
                        rate_scaled = int(staff_member.hourly_rate * 100)
                        staff_cost += rate_scaled * assignments[assignment_key]

                if staff_cost:
                    cost_var = self.model.NewIntVar(
                        0, 10000000, f"cost_{staff_member.staff_id}"
                    )
                    self.model.Add(cost_var == staff_cost)
                    cost_terms.append(cost_var)

        return cost_terms

    def _create_satisfaction_terms(
        self,
        assignments: Dict[Tuple[UUID, UUID, int, int], cp_model.IntVar],
        staff: List[Staff],
        time_slots: List[Tuple[int, int]],
    ) -> List[cp_model.IntVar]:
        """Create satisfaction terms based on staff preferences"""

        satisfaction_terms = []

        for staff_member in staff:
            total_satisfaction = 0

            for preference in staff_member.preferences:
                if preference.preference_type == PreferenceType.PREFERRED_TIME:
                    # Reward assignments to preferred times
                    pref_assignments = []

                    for assignment_key in assignments:
                        if assignment_key[0] == staff_member.staff_id:
                            day_offset, hour = assignment_key[2], assignment_key[3]

                            if self._matches_preference(preference, day_offset, hour):
                                weight_scaled = int(preference.weight * 100)
                                pref_assignments.append(
                                    weight_scaled * assignments[assignment_key]
                                )

                    if pref_assignments:
                        total_satisfaction += sum(pref_assignments)

            if total_satisfaction:
                satisfaction_var = self.model.NewIntVar(
                    0, 100000, f"satisfaction_{staff_member.staff_id}"
                )
                self.model.Add(satisfaction_var == total_satisfaction)
                satisfaction_terms.append(satisfaction_var)

        return satisfaction_terms

    def _create_overtime_terms(
        self,
        assignments: Dict[Tuple[UUID, UUID, int, int], cp_model.IntVar],
        staff: List[Staff],
        time_slots: List[Tuple[int, int]],
    ) -> List[cp_model.IntVar]:
        """Create overtime penalty terms"""

        overtime_terms = []

        # Calculate total days for weekly hour limits
        total_days = len(set(day_offset for day_offset, _ in time_slots))
        weeks = max(1, total_days / 7)  # Convert to weeks for overtime calculation

        for staff_member in staff:
            total_hours = 0
            for assignment_key in assignments:
                if assignment_key[0] == staff_member.staff_id:
                    total_hours += assignments[assignment_key]

            if total_hours:
                # Adjust weekly limit based on scheduling period
                weekly_limit = staff_member.max_weekly_hours * weeks
                regular_hours = int(min(weekly_limit, 40 * weeks))

                overtime_var = self.model.NewIntVar(
                    0, int(40 * weeks), f"overtime_{staff_member.staff_id}"
                )

                self.model.Add(overtime_var >= total_hours - regular_hours)
                self.model.Add(overtime_var >= 0)

                overtime_terms.append(overtime_var)

        return overtime_terms

    def _create_fairness_terms(
        self,
        assignments: Dict[Tuple[UUID, UUID, int, int], cp_model.IntVar],
        staff: List[Staff],
        time_slots: List[Tuple[int, int]],
    ) -> List[cp_model.IntVar]:
        """Create fairness terms to ensure even distribution"""

        fairness_terms = []

        # Calculate target hours per staff member
        total_slots = len(time_slots)
        avg_hours = total_slots / len(staff)

        for staff_member in staff:
            staff_hours = 0
            for assignment_key in assignments:
                if assignment_key[0] == staff_member.staff_id:
                    staff_hours += assignments[assignment_key]

            # Target hours adjusted by priority and availability
            target_hours = int(avg_hours * staff_member.priority_score)

            # Minimize deviation from target
            deviation_var = self.model.NewIntVar(
                0, total_slots, f"fairness_{staff_member.staff_id}"
            )

            # |staff_hours - target_hours|
            pos_dev = self.model.NewIntVar(
                0, total_slots, f"pos_dev_{staff_member.staff_id}"
            )
            neg_dev = self.model.NewIntVar(
                0, total_slots, f"neg_dev_{staff_member.staff_id}"
            )

            self.model.Add(staff_hours - target_hours == pos_dev - neg_dev)
            self.model.Add(deviation_var == pos_dev + neg_dev)

            fairness_terms.append(
                -deviation_var
            )  # Negative because we want to minimize deviation

        return fairness_terms

    def _create_continuity_terms(
        self,
        assignments: Dict[Tuple[UUID, UUID, int, int], cp_model.IntVar],
        staff: List[Staff],
        time_slots: List[Tuple[int, int]],
    ) -> List[cp_model.IntVar]:
        """Create continuity terms to prefer consecutive shifts"""

        continuity_terms = []

        # Group time slots by day
        days_slots = {}
        for day_offset, hour in time_slots:
            if day_offset not in days_slots:
                days_slots[day_offset] = []
            days_slots[day_offset].append(hour)

        for staff_member in staff:
            continuity_bonus = 0

            # Reward consecutive assignments within each day
            for day_offset, hours in days_slots.items():
                day_slots = sorted(hours)

                for i in range(len(day_slots) - 1):
                    current_hour = day_slots[i]
                    next_hour = day_slots[i + 1]

                    # Check if hours are consecutive
                    if next_hour == current_hour + 1:
                        current_key = None
                        next_key = None

                        for assignment_key in assignments:
                            if (
                                assignment_key[0] == staff_member.staff_id
                                and assignment_key[2] == day_offset
                                and assignment_key[3] == current_hour
                            ):
                                current_key = assignment_key
                            elif (
                                assignment_key[0] == staff_member.staff_id
                                and assignment_key[2] == day_offset
                                and assignment_key[3] == next_hour
                            ):
                                next_key = assignment_key

                        if current_key and next_key:
                            # Bonus for consecutive assignments
                            consecutive_bonus = self.model.NewBoolVar(
                                f"consecutive_{staff_member.staff_id}_{day_offset}_{i}"
                            )
                            self.model.Add(
                                assignments[current_key] + assignments[next_key] - 1
                                <= consecutive_bonus
                            )
                            continuity_bonus += consecutive_bonus

            if continuity_bonus:
                continuity_var = self.model.NewIntVar(
                    0, 1000, f"continuity_{staff_member.staff_id}"
                )
                self.model.Add(continuity_var == continuity_bonus)
                continuity_terms.append(continuity_var)

        return continuity_terms

    def _matches_preference(
        self,
        preference: StaffPreference,
        day_offset: int,
        hour: int,
        start_date: date = None,
    ) -> bool:
        """Check if a time slot matches a staff preference"""

        # If preference specifies a day of week, check it
        if preference.day_of_week is not None and start_date:
            actual_date = start_date + timedelta(days=day_offset)
            if actual_date.weekday() != preference.day_of_week:
                return False

        if preference.time_range_start and preference.time_range_end:
            start_hour = preference.time_range_start.hour
            end_hour = preference.time_range_end.hour

            if end_hour <= start_hour:  # Overnight range
                return hour >= start_hour or hour <= end_hour
            else:
                return start_hour <= hour <= end_hour

        return True

    def _add_existing_schedule_constraints(
        self,
        assignments: Dict[Tuple[UUID, UUID, int, int], cp_model.IntVar],
        existing_schedule: List[ScheduledShift],
        start_date: date,
        end_date: date,
    ):
        """Add constraints to preserve parts of existing schedule"""

        for shift in existing_schedule:
            if shift.status in [ShiftStatus.CONFIRMED, ShiftStatus.IN_PROGRESS]:
                # Check if shift falls within our date range
                if start_date <= shift.date <= end_date:
                    day_offset = (shift.date - start_date).days
                    hour = shift.start_time.hour

                    assignment_key = (shift.staff_id, shift.group_id, day_offset, hour)
                    if assignment_key in assignments:
                        # Force this assignment
                        self.model.Add(assignments[assignment_key] == 1)
                        logger.debug(
                            f"Preserved existing shift: {shift.staff_id} on {shift.date}"
                        )

    def _extract_schedule(
        self,
        assignments: Dict[Tuple[UUID, UUID, int, int], cp_model.IntVar],
        staff: List[Staff],
        groups: List[Group],
        time_slots: List[Tuple[int, int]],
        start_date: date,
    ) -> List[ScheduledShift]:
        """Extract the solved schedule from decision variables"""

        schedule = []

        for assignment_key, var in assignments.items():
            if self.solver.Value(var) == 1:
                staff_id, group_id, day_offset, hour = assignment_key

                shift_date = start_date + timedelta(days=day_offset)
                start_time = datetime_time(hour, 0)
                end_time = (
                    datetime_time(hour + 1, 0) if hour < 23 else datetime_time(23, 59)
                )

                shift = ScheduledShift(
                    staff_id=staff_id,
                    group_id=group_id,
                    date=shift_date,
                    start_time=start_time,
                    end_time=end_time,
                    scheduled_hours=1.0,
                    status=ShiftStatus.SCHEDULED,
                )

                schedule.append(shift)

        # Merge consecutive shifts for the same staff/group
        schedule = self._merge_consecutive_shifts(schedule)

        logger.info(f"Extracted {len(schedule)} shifts from solution")
        return schedule

    def _merge_consecutive_shifts(
        self, schedule: List[ScheduledShift]
    ) -> List[ScheduledShift]:
        """Merge consecutive hourly shifts into longer shifts"""

        if not schedule:
            return schedule

        # Group by staff and group and date
        grouped = {}
        for shift in schedule:
            key = (shift.staff_id, shift.group_id, shift.date)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(shift)

        merged_schedule = []

        for key, shifts in grouped.items():
            # Sort by start time
            shifts.sort(key=lambda s: s.start_time)

            merged_shifts = []
            current_shift = shifts[0]

            for next_shift in shifts[1:]:
                # Check if shifts are consecutive
                current_end_hour = current_shift.end_time.hour
                next_start_hour = next_shift.start_time.hour

                if current_end_hour == next_start_hour or (
                    current_end_hour == 23 and next_start_hour == 0
                ):
                    # Merge shifts
                    current_shift.end_time = next_shift.end_time
                    current_shift.scheduled_hours += next_shift.scheduled_hours
                else:
                    # Not consecutive, save current and start new
                    merged_shifts.append(current_shift)
                    current_shift = next_shift

            merged_shifts.append(current_shift)
            merged_schedule.extend(merged_shifts)

        return merged_schedule

    def _detect_conflicts(
        self,
        schedule: List[ScheduledShift],
        staff: List[Staff],
        groups: List[Group],
        requirements: List[StaffingRequirement],
    ) -> List[ScheduleConflict]:
        """Detect any remaining conflicts in the schedule"""

        conflicts = []

        # Check staffing requirements
        for req in requirements:
            conflicts.extend(self._check_staffing_requirement(schedule, req))

        # Check staff availability and constraints
        for staff_member in staff:
            conflicts.extend(self._check_staff_constraints(schedule, staff_member))

        return conflicts

    def _check_staffing_requirement(
        self, schedule: List[ScheduledShift], requirement: StaffingRequirement
    ) -> List[ScheduleConflict]:
        """Check if a staffing requirement is met"""

        conflicts = []

        # Find shifts that cover this requirement
        covering_shifts = []
        for shift in schedule:
            if shift.group_id == requirement.group_id and self._shift_covers_time_slot(
                shift, requirement.time_slot
            ):
                covering_shifts.append(shift)

        staff_count = len(covering_shifts)

        if staff_count < requirement.min_staff_count:
            conflicts.append(
                ScheduleConflict(
                    conflict_type="understaffed",
                    severity="error",
                    group_id=requirement.group_id,
                    time_slot=requirement.time_slot,
                    description=f"Understaffed: {staff_count}/{requirement.min_staff_count} staff assigned",
                    suggested_solutions=[
                        "Add more staff",
                        "Extend existing shifts",
                        "Call substitutes",
                    ],
                )
            )

        return conflicts

    def _check_staff_constraints(
        self, schedule: List[ScheduledShift], staff_member: Staff
    ) -> List[ScheduleConflict]:
        """Check if staff constraints are violated"""

        conflicts = []

        # Get staff shifts
        staff_shifts = [s for s in schedule if s.staff_id == staff_member.staff_id]

        # Calculate total hours for the period
        total_hours = sum(s.scheduled_hours for s in staff_shifts)

        # Calculate period duration in weeks for proper overtime calculation
        if staff_shifts:
            dates = [s.date for s in staff_shifts]
            period_days = (max(dates) - min(dates)).days + 1
            period_weeks = max(1, period_days / 7)
            max_hours_for_period = staff_member.max_weekly_hours * period_weeks

            if total_hours > max_hours_for_period:
                conflicts.append(
                    ScheduleConflict(
                        conflict_type="overtime_violation",
                        severity="warning",
                        staff_id=staff_member.staff_id,
                        description=f"Overtime: {total_hours:.1f}/{max_hours_for_period:.1f} hours for period",
                        suggested_solutions=[
                            "Reduce hours",
                            "Split shifts",
                            "Reassign to other staff",
                        ],
                    )
                )

        return conflicts

    def _shift_covers_time_slot(
        self, shift: ScheduledShift, time_slot: TimeSlot
    ) -> bool:
        """Check if a shift covers a specific time slot"""

        shift_day = shift.date.weekday()
        if shift_day != time_slot.day_of_week:
            return False

        shift_start_hour = shift.start_time.hour
        shift_end_hour = shift.end_time.hour
        slot_start_hour = time_slot.start_time.hour
        slot_end_hour = time_slot.end_time.hour

        return shift_start_hour <= slot_start_hour and shift_end_hour >= slot_end_hour

    def _analyze_infeasibility(
        self,
        staff: List[Staff],
        groups: List[Group],
        requirements: List[StaffingRequirement],
        time_slots: List[Tuple[int, int]],
    ) -> List[ScheduleConflict]:
        """Analyze why the problem is infeasible"""

        conflicts = []

        # Calculate total days from time slots
        total_days = len(set(day_offset for day_offset, _ in time_slots))
        period_weeks = max(1, total_days / 7)

        # Check if total required staff hours exceed available staff hours
        total_required_hours = (
            sum(
                req.min_staff_count * self._get_time_slot_duration(req.time_slot)
                for req in requirements
            )
            * period_weeks
        )

        total_available_hours = sum(
            staff_member.max_weekly_hours * period_weeks for staff_member in staff
        )

        if total_required_hours > total_available_hours:
            conflicts.append(
                ScheduleConflict(
                    conflict_type="insufficient_capacity",
                    severity="error",
                    description=f"Required hours ({total_required_hours:.1f}) exceed available hours ({total_available_hours:.1f}) for {total_days}-day period",
                    suggested_solutions=[
                        "Hire more staff",
                        "Reduce operating hours",
                        "Increase staff availability",
                    ],
                )
            )

        # Check for qualification mismatches
        for req in requirements:
            if req.required_qualifications:
                qualified_staff = []
                for staff_member in staff:
                    staff_quals = {
                        q.qualification_name
                        for q in staff_member.qualifications
                        if q.is_verified
                    }
                    if all(qual in staff_quals for qual in req.required_qualifications):
                        qualified_staff.append(staff_member)

                if len(qualified_staff) < req.min_staff_count:
                    conflicts.append(
                        ScheduleConflict(
                            conflict_type="qualification_shortage",
                            severity="error",
                            group_id=req.group_id,
                            description=f"Not enough qualified staff: {len(qualified_staff)}/{req.min_staff_count}",
                            suggested_solutions=[
                                "Train existing staff",
                                "Hire qualified staff",
                                "Relax requirements",
                            ],
                        )
                    )

        return conflicts

    def _get_time_slot_duration(self, time_slot: TimeSlot) -> float:
        """Calculate duration of a time slot in hours"""

        start_dt = datetime.combine(date.today(), time_slot.start_time)
        end_dt = datetime.combine(date.today(), time_slot.end_time)

        if end_dt <= start_dt:
            end_dt += timedelta(days=1)

        return (end_dt - start_dt).total_seconds() / 3600
