"""
OR-Tools CP-SAT solver for schedule optimization with flexible date ranges
"""

import logging
from datetime import datetime, timedelta, time as datetime_time, date
import time as _time_module
from typing import List, Dict, Tuple, Optional, Set
from ortools.sat.python import cp_model
from uuid import UUID
from collections import defaultdict


from .models import *
from .constraints import ConstraintBuilder  # Your existing constraint builder
from .config import settings  # Your existing config


logger = logging.getLogger(__name__)


class EnhancedScheduleSolver:
    """Enhanced solver with flexible date range support - extends your existing solver"""

    def __init__(self, optimization_config: OptimizationConfig):
        self.config = optimization_config
        self.model = cp_model.CpModel()
        self.solver = cp_model.CpSolver()

        # Use your existing constraint builder or create enhanced one
        if hasattr(self, "constraint_builder"):
            self.constraint_builder = self.constraint_builder
        else:
            self.constraint_builder = ConstraintBuilder(self.model)

        # Configure solver parameters
        self.solver.parameters.max_time_in_seconds = optimization_config.max_solver_time
        self.solver.parameters.num_search_workers = getattr(
            settings, "SOLVER_WORKERS", 4
        )
        self.solver.parameters.log_search_progress = getattr(
            settings, "LOG_SOLVER_PROGRESS", True
        )

    def solve_with_date_range(
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
        Enhanced solve method with flexible date range support
        """
        start_time = _time_module.time()
        try:
            # Store existing schedule for coverage counting
            self._existing_schedule = existing_schedule or []
            
            # Build coverage cache early to ensure consistency
            if self._existing_schedule:
                self._build_coverage_cache(schedule_start_date)
            # 1) Validate date range
            if schedule_end_date < schedule_start_date:
                raise ValueError("End date must be after start date")
            total_days = (schedule_end_date - schedule_start_date).days + 1
            if total_days > 365:
                raise ValueError(f"Date range too large: {total_days} days (max 365)")

            # 2) Generate time slots for the range (day_offset, hour)
            time_slots = self._generate_time_slots_for_range(
                schedule_start_date, schedule_end_date
            )
            
            logger.info(f"SOLVER SETUP: {len(staff)} staff, {len(groups)} groups, {len(requirements)} requirements")
            logger.info(f"DATE RANGE: {schedule_start_date} to {schedule_end_date} ({total_days} days)")
            logger.info(f"EXISTING SCHEDULES: {len(self._existing_schedule) if self._existing_schedule else 0} shifts")
            if self._existing_schedule:
                for i, shift in enumerate(self._existing_schedule):
                    logger.info(f"  Existing {i+1}: {shift.staff_id} -> {str(shift.group_id)[-8:]} on {shift.date} {shift.start_time}-{shift.end_time} ({shift.scheduled_hours}h)")
            
            self.debug_solver_state(staff, groups, requirements, schedule_start_date, schedule_end_date)

            # 3) Build assignment variables - FIXED VERSION
            assignments: Dict[Tuple[UUID, UUID, int, int], cp_model.IntVar] = {}
            
            # Create a mapping of day_of_week to day_offsets in our period
            dow_to_offsets = {}
            for day_offset in range(total_days):
                actual_date = schedule_start_date + timedelta(days=day_offset)
                dow = actual_date.weekday()
                if dow not in dow_to_offsets:
                    dow_to_offsets[dow] = []
                dow_to_offsets[dow].append(day_offset)

            # Collect all time slots that are covered by any requirement
            covered_slots = set()
            for req in requirements:
                req_dow = req.time_slot.day_of_week
                h0 = req.time_slot.start_time.hour
                h1 = req.time_slot.end_time.hour
                
                # Map to actual day_offsets in our period
                if req_dow in dow_to_offsets:
                    for day_offset in dow_to_offsets[req_dow]:
                        for h in range(h0, h1):
                            covered_slots.add((req.group_id, day_offset, h))

            # Create variables for all covered slots with qualified and available staff
            for gid, day_offset, hour in covered_slots:
                # Get actual date for availability checking
                actual_date = schedule_start_date + timedelta(days=day_offset)
                actual_dow = actual_date.weekday()
                
                # Find requirements that apply to this slot
                applicable_reqs = [
                    req for req in requirements
                    if req.group_id == gid
                    and req.time_slot.day_of_week == actual_dow
                    and req.time_slot.start_time.hour <= hour < req.time_slot.end_time.hour
                ]
                
                # CREATE variables for all groups (including over-capacity)
                # We'll handle over-capacity constraints separately
                
                # Get union of all required qualifications for this slot
                all_required_quals = set()
                for req in applicable_reqs:
                    if req.required_qualifications:
                        all_required_quals.update(req.required_qualifications)

                for sm in staff:
                    # Check qualifications
                    sm_quals = {
                        q.qualification_name
                        for q in sm.qualifications
                        if q.is_verified
                        and (not getattr(q, 'expiry_date', None)
                            or q.expiry_date > date.today())
                    }
                    if all_required_quals and not all_required_quals.issubset(sm_quals):
                        continue

                    # Check availability
                    is_available = False
                    for avail_slot in sm.availability:
                        if (avail_slot.day_of_week == actual_dow
                            and avail_slot.is_available
                            and avail_slot.start_time.hour <= hour < avail_slot.end_time.hour):
                            is_available = True
                            break
                    
                    if not is_available:
                        continue
                        
                    # Check if this staff member is already assigned to another group at this time
                    is_already_assigned = False
                    if hasattr(self, '_existing_schedule') and self._existing_schedule:
                        for shift in self._existing_schedule:
                            if (shift.staff_id == sm.staff_id and 
                                shift.date == actual_date and
                                shift.start_time.hour <= hour < shift.end_time.hour):
                                # Staff is already assigned during this hour
                                if shift.group_id != gid:
                                    # They're assigned to a different group, so can't be assigned here
                                    is_already_assigned = True
                                    logger.debug(f"Skipping variable creation: {sm.name} already assigned to group {str(shift.group_id)[-8:]} at {actual_date} hour {hour}")
                                break
                    
                    if not is_already_assigned:
                        var = self.model.NewBoolVar(
                            f"x_{sm.staff_id}_{gid}_{day_offset}_{hour}"
                        )
                        assignments[(sm.staff_id, gid, day_offset, hour)] = var

            logger.info(
                f"Built {len(assignments)} assignment variables from {len(requirements)} requirements"
            )
            
            # DEBUG: Log summary of variables by group
            var_summary = defaultdict(list)
            for (staff_id, group_id, day_offset, hour), var in assignments.items():
                var_summary[group_id].append(f"{str(staff_id)[-8:]}@h{hour}")
            
            for group_id, vars_list in var_summary.items():
                logger.info(f"Variables for group {str(group_id)[-8:]}: {len(vars_list)} vars - {vars_list[:10]}...")
            
            # DEBUG: Log detailed variable information
            if not assignments:
                logger.error("CRITICAL: No assignment variables created! This will cause INFEASIBLE.")
                logger.error(f"Debug info: staff={len(staff)}, groups={len(groups)}, requirements={len(requirements)}")
                logger.error(f"Time slots: {len(time_slots)}")
                for i, req in enumerate(requirements):
                    logger.error(f"Requirement {i}: group={req.group_id}, day={req.time_slot.day_of_week}, time={req.time_slot.start_time}-{req.time_slot.end_time}")
                return [], OptimizationResult(objective_value=0, solve_time_seconds=0, status="NO_VARIABLES", iterations=0, conflicts_resolved=0), []

            # 4) Add all constraints
            logger.info("CONSTRAINTS: Adding enhanced constraints...")
            self._add_enhanced_constraints(
                assignments,
                staff,
                groups,
                requirements,
                time_slots,
                schedule_start_date,
                schedule_end_date,
            )
            logger.info(f"CONSTRAINTS: Total model constraints so far: {len(list(self.model.Proto().constraints))}")
            
            # 4b) Temporarily disable shift continuity constraints for debugging
            # self._add_shift_continuity_constraints(
            #     assignments,
            #     requirements,
            #     time_slots,
            #     schedule_start_date,
            # )
            
            # 4c) Add group max staff constraints to prevent multiple staff per group
            logger.info("CONSTRAINTS: Adding group max staff constraints...")
            self._add_group_max_staff_constraints(
                assignments,
                requirements,
                time_slots,
                schedule_start_date,
            )
            logger.info(f"CONSTRAINTS: Total model constraints after group max: {len(list(self.model.Proto().constraints))}")

            # 5) Set simple objective for debugging
            # self._set_enhanced_objective(
            #     assignments, staff, groups, time_slots
            # )
            # Simple objective: minimize total assignments
            total_assignments = sum(assignments.values())
            self.model.Minimize(total_assignments)

            # 6) Preserve existing shifts as constraints
            if existing_schedule:
                logger.info(f"PRESERVATION: Processing {len(existing_schedule)} existing shifts for preservation")
                for shift in existing_schedule:
                    logger.info(f"EXISTING SHIFT: {shift.staff_id} -> {str(shift.group_id)[-8:]} on {shift.date} {shift.start_time}-{shift.end_time}")
                
                # Preserve confirmed shifts and block conflicting assignments
                logger.info("PRESERVATION: Adding existing shift preservation constraints...")
                self._preserve_existing_shifts(
                    assignments,
                    existing_schedule,
                    requirements,
                    schedule_start_date,
                    schedule_end_date,
                )
                logger.info(f"PRESERVATION: Total model constraints after preservation: {len(list(self.model.Proto().constraints))}")

            # 7) Solve
            logger.info(f"SOLVING: Starting solver with {len(assignments)} variables and {len(list(self.model.Proto().constraints))} constraints")
            status = self.solver.Solve(self.model)
            solve_time = _time_module.time() - start_time
            logger.info(f"SOLVING: Solver finished with status: {status} in {solve_time:.2f}s")

            # 8) Process results
            if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
                schedule = self._extract_enhanced_schedule(
                    assignments, staff, groups, time_slots, schedule_start_date
                )
                conflicts = self._detect_enhanced_conflicts(
                    schedule, staff, groups, requirements
                )
                optimization_result = OptimizationResult(
                    objective_value=self.solver.ObjectiveValue(),
                    solve_time_seconds=solve_time,
                    status="OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE",
                    iterations=self.solver.NumBranches(),
                    conflicts_resolved=len(conflicts),
                )
                logger.info(f"Schedule solved successfully in {solve_time:.2f}s")
                return schedule, optimization_result, conflicts

            # infeasible or timeout
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
            conflicts = self._analyze_infeasibility_for_range(
                staff, groups, requirements, time_slots, total_days, schedule_start_date, schedule_end_date
            )
            logger.warning(f"Schedule optimization failed: {status_name}")
            return [], optimization_result, conflicts

        except Exception as e:
            logger.error(f"Error in enhanced schedule optimization: {str(e)}")
            optimization_result = OptimizationResult(
                objective_value=0,
                solve_time_seconds=_time_module.time() - start_time,
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

    def _generate_time_slots_for_range(
        self, start_date: date, end_date: date
    ) -> List[Tuple[int, int]]:
        """Generate time slots for the specified date range with consistent day mapping"""

        time_slots = []

        # Get operating hours from settings or use defaults
        start_hour = getattr(settings, "OPERATING_START_HOUR", 6)
        end_hour = getattr(settings, "OPERATING_END_HOUR", 20)

        total_days = (end_date - start_date).days + 1

        # Log the date range and day mapping for debugging
        logger.info(f"Generating time slots for period {start_date} to {end_date}")
        for day_offset in range(min(7, total_days)):  # Log first week
            actual_date = start_date + timedelta(days=day_offset)
            logger.debug(f"Day offset {day_offset} = {actual_date} ({actual_date.strftime('%A')})")

        # Generate (day_offset, hour) pairs consistently
        for day_offset in range(total_days):
            for hour in range(start_hour, end_hour):
                time_slots.append((day_offset, hour))

        logger.info(
            f"Generated {len(time_slots)} time slots for {total_days} days "
            f"(hours {start_hour}-{end_hour} each day)"
        )
        
        # Log some example time slots for debugging
        if time_slots:
            logger.debug(f"Example time slots: {time_slots[:5]}...{time_slots[-5:]}")
        
        return time_slots

    def _create_assignment_variables(
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

    def _add_enhanced_constraints(
        self,
        assignments: Dict[Tuple[UUID, UUID, int, int], cp_model.IntVar],
        staff: List[Staff],
        groups: List[Group],
        requirements: List[StaffingRequirement],
        time_slots: List[Tuple[int, int]],
        start_date: date,
        end_date: date,
    ):
        """Add all of our hard CP‐SAT constraints with proper date mapping."""
        
        # DEBUGGING: Only add the most basic constraints to find the infeasibility source
        logger.info("Adding constraints in debug mode...")
        
        # 1) Enhanced staffing constraints with over-capacity handling
        self._add_enhanced_staffing_ratio_constraints(
            assignments, requirements, staff, groups, time_slots, start_date
        )
        
        # 2) Add time conflict constraints to ensure staff can only work one group at a time
        self._add_staff_time_conflict_constraints(
            assignments, staff, groups, time_slots, start_date
        )
        
        # 3) Add constraints to prevent overlapping shifts for same staff-group combination
        # DISABLED: This constraint was too restrictive and made the problem infeasible
        # self._add_no_overlapping_shifts_constraints(
        #     assignments, staff, groups, time_slots, start_date
        # )

        # 2) Skip qualification constraints for now
        # self._add_enhanced_qualification_constraints(
        #     assignments, staff, requirements, time_slots, start_date
        # )

        # 3) Skip minimum shift duration constraints
        # self._add_minimum_shift_duration_constraints(
        #     assignments, staff, groups, time_slots, start_date
        # )

        # 4) Skip working hours constraints for now
        # self._add_working_hours_constraints_for_period(
        #     assignments, staff, time_slots, start_date, end_date
        # )
        # self._add_consecutive_days_constraints(
        #     assignments, staff, time_slots, start_date, end_date
        # )
        # self._add_staff_availability_constraints_for_period(
        #     assignments, staff, time_slots, start_date
        # )

    def _add_enhanced_staffing_ratio_constraints(
        self,
        assignments: Dict[Tuple[UUID, UUID, int, int], cp_model.IntVar],
        requirements: List[StaffingRequirement],
        staff: List[Staff],
        groups: List[Group],
        time_slots: List[Tuple[int, int]],
        start_date: date,
    ):
        """
        Enhanced staffing ratio constraints that properly handle overlapping requirements
        and correctly map day_of_week to day_offsets.
        """
        
        logger.info("=== ENHANCED STAFFING CONSTRAINTS DEBUG ===")
        logger.info(f"Start date: {start_date}")
        logger.info(f"Requirements provided: {len(requirements)}")
        logger.info(f"Time slots to process: {len(time_slots)}")
        
        # Log all requirements for debugging
        for i, req in enumerate(requirements):
            logger.info(f"Requirement {i}: group={str(req.group_id)[-8:]}, dow={req.time_slot.day_of_week}, "
                       f"time={req.time_slot.start_time}-{req.time_slot.end_time}, "
                       f"staff={req.min_staff_count}-{req.max_staff_count}")
        
        # Build a mapping from (group_id, day_offset, hour) to applicable requirements
        slot_requirements = defaultdict(list)
        
        for day_offset, hour in time_slots:
            # Convert day_offset to actual day_of_week
            actual_date = start_date + timedelta(days=day_offset)
            actual_dow = actual_date.weekday()
            
            for req in requirements:
                req_dow = req.time_slot.day_of_week
                req_start_hour = req.time_slot.start_time.hour
                req_end_hour = req.time_slot.end_time.hour
                
                # Check if this requirement applies to this slot
                if (req_dow == actual_dow and 
                    req_start_hour <= hour < req_end_hour):
                    slot_requirements[(req.group_id, day_offset, hour)].append(req)
                    logger.debug(f"SLOT MATCH: {actual_date} (day_offset={day_offset}, dow={actual_dow}) "
                               f"hour {hour} matches req dow={req_dow} time={req_start_hour}-{req_end_hour} "
                               f"for group {str(req.group_id)[-8:]}")
        
        logger.info(f"Generated {len(slot_requirements)} slots with requirements")
        
        # DEBUG: Log the first few slots to see what's being generated
        for i, ((group_id, day_offset, hour), reqs) in enumerate(list(slot_requirements.items())[:10]):
            actual_date = start_date + timedelta(days=day_offset)
            logger.info(f"Slot {i}: {actual_date} (day_offset={day_offset}) hour {hour} "
                       f"group {str(group_id)[-8:]} needs {len(reqs)} requirements")

        # Enforce constraints for each slot that has requirements
        constraints_added = 0
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
                        f"group {group_id}, day {day_offset}, hour {hour} "
                        f"(requirements: {[req.time_slot.start_time.hour for req in applicable_reqs]})"
                    )
                    # Add an impossible constraint to make the problem infeasible
                    impossible_var = self.model.NewBoolVar(f"impossible_{group_id}_{day_offset}_{hour}")
                    self.model.Add(impossible_var == 1)
                    self.model.Add(impossible_var == 0)  # Contradiction
                continue

            # Aggregate requirements for this slot - FIXED: Use max instead of sum
            # Multiple overlapping requirements should take the MAXIMUM needed, not sum
            min_staff_required = max(req.min_staff_count for req in applicable_reqs)
            
            # For max staff, take the minimum of all max constraints (most restrictive)
            max_constraints = [req.max_staff_count for req in applicable_reqs if req.max_staff_count is not None]
            if max_constraints:
                max_staff_allowed = min(max_constraints)
            else:
                # If no max constraints, use min_staff_required as a reasonable default
                max_staff_allowed = min_staff_required
            
            # ADJUST requirements based on existing schedule coverage
            existing_staff_count = self._count_existing_coverage(group_id, day_offset, hour, start_date)
            
            # Reduce min requirement by existing coverage (can't go below 0)
            min_staff_required = max(0, min_staff_required - existing_staff_count)
            
            # Reduce max allowed by existing coverage (can't go below 0)
            max_staff_allowed = max(0, max_staff_allowed - existing_staff_count)
            
            # CRITICAL FIX: If group is already over capacity (existing coverage > original max),
            # OR if existing coverage already meets minimum requirements,
            # skip adding any constraints for this slot to avoid contradictions
            original_max = min(max_constraints) if max_constraints else max(req.min_staff_count for req in applicable_reqs)
            original_min = max(req.min_staff_count for req in applicable_reqs)
            
            logger.debug(f"STAFFING CHECK: Group {str(group_id)[-4:]} day {day_offset} hour {hour}: "
                        f"existing_coverage={existing_staff_count}, original_min={original_min}, original_max={original_max}, "
                        f"will_skip_over_capacity={existing_staff_count > original_max}, "
                        f"will_skip_satisfied={existing_staff_count >= original_min}")
            
            if existing_staff_count > original_max:
                logger.info(f"SKIPPING staffing constraints for over-capacity group {str(group_id)[-4:]} day {day_offset} hour {hour}: "
                           f"existing_coverage={existing_staff_count} > original_max={original_max}")
                continue
                
            if existing_staff_count >= original_min and max_staff_allowed == 0:
                logger.info(f"SKIPPING staffing constraints for satisfied group {str(group_id)[-4:]} day {day_offset} hour {hour}: "
                           f"existing_coverage={existing_staff_count} >= original_min={original_min} and no new staff allowed")
                continue
            
            logger.info(f"COVERAGE ANALYSIS: Group {str(group_id)[-4:]} day {day_offset} hour {hour}: "
                       f"existing_coverage={existing_staff_count}, "
                       f"original_min={max(req.min_staff_count for req in applicable_reqs)}, "
                       f"original_max={min(max_constraints) if max_constraints else 'default'}, "
                       f"adjusted_min={min_staff_required}, adjusted_max={max_staff_allowed}, "
                       f"available_vars={len(slot_vars)}")

            # Ensure we don't exceed available staff
            max_staff_allowed = min(max_staff_allowed, len(slot_vars))

            # Sanity check - adjust requirements to available staff to avoid infeasibility
            if len(slot_vars) < min_staff_required:
                logger.warning(
                    f"Insufficient available staff: {len(slot_vars)} available but "
                    f"{min_staff_required} required for group {group_id} day {day_offset} hour {hour}. "
                    f"Reducing requirement to {len(slot_vars)} to avoid infeasibility."
                )
                min_staff_required = len(slot_vars)  # Can't require more than available

            # Add the constraints
            total_assigned = sum(slot_vars)
            
            # Minimum staffing constraint (only if we have staff available)
            if min_staff_required > 0 and slot_vars:
                self.model.Add(total_assigned >= min_staff_required)
                constraints_added += 1
            
            # Maximum staffing constraint  
            if max_staff_allowed >= 0 and slot_vars:
                self.model.Add(total_assigned <= max_staff_allowed)
                constraints_added += 1
                logger.debug(f"CRITICAL: Added max staff constraint for group {str(group_id)[-4:]} day {day_offset} hour {hour}: "
                           f"total_assigned <= {max_staff_allowed} (from {len(slot_vars)} variables)")
            
            logger.debug(
                f"Added staffing constraint for group {group_id}, day {day_offset}, hour {hour}: "
                f"{min_staff_required} <= staff <= {max_staff_allowed} "
                f"(from {len(applicable_reqs)} requirements, {len(slot_vars)} variables)"
            )
        
        logger.info(f"Added {constraints_added} staffing ratio constraints")
    
    def _add_shift_continuity_constraints(
        self,
        assignments: Dict[Tuple[UUID, UUID, int, int], cp_model.IntVar],
        requirements: List[StaffingRequirement],
        time_slots: List[Tuple[int, int]],
        start_date: date,
    ):
        """Ensure that if a staff member is assigned to a group, they cover the entire shift period"""
        
        constraints_added = 0
        
        for req in requirements:
            req_dow = req.time_slot.day_of_week
            start_hour = req.time_slot.start_time.hour
            end_hour = req.time_slot.end_time.hour
            
            # Find day offsets that match this requirement's day_of_week
            matching_days = []
            for day_offset, _ in time_slots:
                actual_date = start_date + timedelta(days=day_offset)
                if actual_date.weekday() == req_dow:
                    matching_days.append(day_offset)
            
            # For each matching day, enforce shift continuity
            for day_offset in matching_days:
                # Get all staff variables for this group/day across all hours in the shift
                staff_hour_vars = {}  # staff_id -> list of hour variables
                
                for hour in range(start_hour, end_hour):
                    for (staff_id, group_id, d, h), var in assignments.items():
                        if group_id == req.group_id and d == day_offset and h == hour:
                            if staff_id not in staff_hour_vars:
                                staff_hour_vars[staff_id] = []
                            staff_hour_vars[staff_id].append(var)
                
                # For each staff member, if they're assigned to ANY hour, they must be assigned to ALL hours
                for staff_id, hour_vars in staff_hour_vars.items():
                    if len(hour_vars) > 1:  # Only add constraint if multiple hours exist
                        # All hour variables for this staff/group/day must be equal
                        # If any hour is 1, all hours must be 1; if any hour is 0, all hours must be 0
                        for i in range(1, len(hour_vars)):
                            self.model.Add(hour_vars[0] == hour_vars[i])
                            constraints_added += 1
                            
                        logger.debug(f"Added shift continuity for staff {staff_id} group {req.group_id} day {day_offset}: {len(hour_vars)} hours must be consistent")
        
        logger.info(f"Added {constraints_added} shift continuity constraints")
    
    def _build_coverage_cache(self, start_date: date):
        """Build cache of existing coverage for all shifts"""
        self._existing_coverage_cache = {}
        
        if hasattr(self, '_existing_schedule') and self._existing_schedule:
            for shift in self._existing_schedule:
                # Calculate day offset for this shift
                shift_day_offset = (shift.date - start_date).days
                
                # Count coverage for each hour in the shift
                for shift_hour in range(shift.start_time.hour, shift.end_time.hour):
                    key = (shift.group_id, shift_day_offset, shift_hour)
                    if key not in self._existing_coverage_cache:
                        self._existing_coverage_cache[key] = 0
                    self._existing_coverage_cache[key] += 1
        
        logger.info(f"Built coverage cache with {len(self._existing_coverage_cache)} entries")
    
    def _count_existing_coverage(self, group_id: UUID, day_offset: int, hour: int, start_date: date) -> int:
        """Count how many staff are already assigned to this group/day/hour via existing schedules"""
        if not hasattr(self, '_existing_coverage_cache'):
            self._build_coverage_cache(start_date)
        
        # Return count for this specific slot
        key = (group_id, day_offset, hour)
        return self._existing_coverage_cache.get(key, 0)
    
    def _add_group_max_staff_constraints(
        self,
        assignments: Dict[Tuple[UUID, UUID, int, int], cp_model.IntVar],
        requirements: List[StaffingRequirement],
        time_slots: List[Tuple[int, int]],
        start_date: date,
    ):
        """Add group-level constraints: total unique staff per group per requirement <= max_staff_count"""
        
        constraints_added = 0
        
        # Apply constraints per requirement (not per individual hour)
        for req in requirements:
            group_id = req.group_id
            max_staff_for_group = req.max_staff_count if req.max_staff_count is not None else req.min_staff_count
            req_dow = req.time_slot.day_of_week
            start_hour = req.time_slot.start_time.hour
            end_hour = req.time_slot.end_time.hour
            
            # Find day offsets that match this requirement's day_of_week
            matching_days = []
            for day_offset, _ in time_slots:
                actual_date = start_date + timedelta(days=day_offset)
                if actual_date.weekday() == req_dow:
                    matching_days.append(day_offset)
            
            # For each matching day, enforce max staff constraint per requirement period
            for day_offset in matching_days:
                # Count existing staff coverage for this requirement period
                existing_staff_for_period = set()
                if hasattr(self, '_existing_schedule') and self._existing_schedule:
                    actual_date = start_date + timedelta(days=day_offset)
                    for shift in self._existing_schedule:
                        if (shift.group_id == group_id and 
                            shift.date == actual_date and
                            # Check if existing shift overlaps with requirement period
                            shift.start_time.hour < end_hour and shift.end_time.hour > start_hour):
                            existing_staff_for_period.add(shift.staff_id)
                
                existing_coverage = len(existing_staff_for_period)
                max_new_staff_for_period = max(0, max_staff_for_group - existing_coverage)
                
                if max_new_staff_for_period == 0:
                    # This requirement period is at capacity - block all new assignments
                    for hour in range(start_hour, end_hour):
                        for (staff_id, gid, d, h), var in assignments.items():
                            if gid == group_id and d == day_offset and h == hour:
                                # Check if this is a preserved existing assignment
                                if staff_id not in existing_staff_for_period:
                                    self.model.Add(var == 0)
                                    constraints_added += 1
                    
                    logger.debug(f"BLOCKED new assignments to Group {str(group_id)[-4:]} on day {day_offset} {start_hour}:00-{end_hour}:00 (at capacity with {existing_coverage} existing staff)")
                
                elif max_new_staff_for_period > 0:
                    # Create constraint for unique staff assignment across the entire requirement period
                    # Collect all variables for this group/day/requirement period by staff
                    staff_period_vars = {}
                    for hour in range(start_hour, end_hour):
                        for (staff_id, gid, d, h), var in assignments.items():
                            if gid == group_id and d == day_offset and h == hour:
                                if staff_id not in existing_staff_for_period:  # Only count new assignments
                                    if staff_id not in staff_period_vars:
                                        staff_period_vars[staff_id] = []
                                    staff_period_vars[staff_id].append(var)
                    
                    if staff_period_vars:
                        # Create binary indicator variables for each staff (1 if staff is assigned to this period)
                        staff_assigned_indicators = []
                        for staff_id, staff_vars in staff_period_vars.items():
                            # Create indicator variable: 1 if this staff is assigned any hour in this period
                            indicator_var = self.model.NewBoolVar(f"staff_{staff_id}_assigned_to_group_{group_id}_day_{day_offset}_period_{start_hour}_{end_hour}")
                            staff_assigned_indicators.append(indicator_var)
                            
                            # Link indicator to staff assignment: indicator = 1 iff any hour variable = 1
                            self.model.Add(indicator_var <= sum(staff_vars))  # If any hour assigned, indicator = 1
                            self.model.Add(sum(staff_vars) <= len(staff_vars) * indicator_var)  # If indicator = 0, no hours assigned
                            constraints_added += 2
                        
                        # Constraint: total unique staff assigned <= max allowed
                        if staff_assigned_indicators:
                            self.model.Add(sum(staff_assigned_indicators) <= max_new_staff_for_period)
                            constraints_added += 1
                            
                            logger.debug(f"LIMITED Group {str(group_id)[-4:]} on day {day_offset} {start_hour}:00-{end_hour}:00 to max {max_new_staff_for_period} unique staff (from {len(staff_period_vars)} potential staff)")
        
        logger.info(f"Added {constraints_added} group-level max staff constraints (per requirement period)")
    
    def _add_basic_staffing_constraints_debug(
        self,
        assignments: Dict[Tuple[UUID, UUID, int, int], cp_model.IntVar],
        requirements: List[StaffingRequirement],
        staff: List[Staff],
        groups: List[Group],
        time_slots: List[Tuple[int, int]],
        start_date: date,
    ):
        """Very basic staffing constraints for debugging infeasibility"""
        
        constraints_added = 0
        total_slots_checked = 0
        slots_with_vars = 0
        
        # Only add constraints where we have at least one variable
        for req in requirements:
            req_dow = req.time_slot.day_of_week
            start_hour = req.time_slot.start_time.hour
            end_hour = req.time_slot.end_time.hour
            
            for day_offset, hour in time_slots:
                actual_date = start_date + timedelta(days=day_offset)
                actual_dow = actual_date.weekday()
                
                if actual_dow == req_dow and start_hour <= hour < end_hour:
                    total_slots_checked += 1
                    
                    # Find variables for this slot
                    slot_vars = []
                    for (sid, gid, d, h), var in assignments.items():
                        if gid == req.group_id and d == day_offset and h == hour:
                            slot_vars.append(var)
                    
                    if slot_vars:
                        slots_with_vars += 1
                        # Only add very relaxed constraints
                        total_assigned = sum(slot_vars)
                        
                        # Only require minimum 1 staff if we have at least 1 available
                        min_required = min(1, len(slot_vars))
                        
                        # EXTRA DEBUG: log detailed info before adding constraint
                        actual_date = start_date + timedelta(days=day_offset)
                        logger.info(f"CONSTRAINT: {actual_date} {hour}:00 group {str(req.group_id)[:8]}... needs >= {min_required} staff, have {len(slot_vars)} variables")
                        
                        if len(slot_vars) >= min_required:
                            self.model.Add(total_assigned >= min_required)
                            constraints_added += 1
                        else:
                            logger.error(f"INFEASIBLE REQUIREMENT: Need {min_required} staff but only {len(slot_vars)} available!")
                        
                        # Allow up to all available staff (no maximum constraint)
                        logger.debug(f"Added basic constraint for group {req.group_id}, day {day_offset}, hour {hour}: >= {min_required} staff (have {len(slot_vars)} vars)")
        
        logger.info(f"DEBUG: Added {constraints_added} basic staffing constraints"
                   f" - checked {total_slots_checked} slots, {slots_with_vars} had variables")
    
    def _add_staff_time_conflict_constraints(
        self,
        assignments: Dict[Tuple[UUID, UUID, int, int], cp_model.IntVar],
        staff: List[Staff],
        groups: List[Group],
        time_slots: List[Tuple[int, int]],
        start_date: date,
    ):
        """STRICT: Ensure each staff member can only be assigned to ONE group at any given time"""
        
        constraints_added = 0
        
        for staff_member in staff:
            # For each day and hour, ensure this staff member is assigned to at most one group
            for day_offset, hour in time_slots:
                # Find all assignments for this staff member at this specific time
                same_time_assignments = []
                for (sid, gid, d, h), var in assignments.items():
                    if sid == staff_member.staff_id and d == day_offset and h == hour:
                        same_time_assignments.append(var)
                
                # If there are multiple possible assignments at the same time, constrain them
                if len(same_time_assignments) > 1:
                    # Staff can be assigned to at most 1 group at any given hour
                    self.model.Add(sum(same_time_assignments) <= 1)
                    constraints_added += 1
                    
                    logger.debug(f"Added strict time conflict constraint for {staff_member.staff_id} on day {day_offset} hour {hour}: max 1 of {len(same_time_assignments)} groups")
        
        logger.info(f"STRICT: Added {constraints_added} time conflict constraints (1 group per staff per hour)")

    def _add_no_overlapping_shifts_constraints(
        self,
        assignments: Dict[Tuple[UUID, UUID, int, int], cp_model.IntVar],
        staff: List[Staff],
        groups: List[Group],
        time_slots: List[Tuple[int, int]],
        start_date: date,
    ):
        """Prevent the same staff from being assigned to overlapping time periods for the same group"""
        
        constraints_added = 0
        
        for staff_member in staff:
            for group in groups:
                # Get all time slots for this staff-group combination
                staff_group_assignments = []
                for (sid, gid, d, h), var in assignments.items():
                    if sid == staff_member.staff_id and gid == group.group_id:
                        staff_group_assignments.append(((d, h), var))
                
                if len(staff_group_assignments) <= 1:
                    continue  # No overlapping possible with 0 or 1 assignments
                
                # Sort by day and hour
                staff_group_assignments.sort(key=lambda x: x[0])
                
                # Group assignments by day to check for overlapping shifts within each day
                daily_assignments = {}
                for (day, hour), var in staff_group_assignments:
                    if day not in daily_assignments:
                        daily_assignments[day] = []
                    daily_assignments[day].append((hour, var))
                
                # For each day, ensure no overlapping shift patterns
                for day, day_assignments in daily_assignments.items():
                    if len(day_assignments) <= 1:
                        continue
                    
                    # Sort by hour
                    day_assignments.sort(key=lambda x: x[0])
                    
                    # Check for problematic overlapping patterns:
                    # Pattern: Two shifts that start at different times but both cover overlapping periods
                    # Example: 8-15 and 9-15 (both 7+ hour shifts starting 1 hour apart)
                    for i in range(len(day_assignments) - 1):
                        hour1, var1 = day_assignments[i]
                        hour2, var2 = day_assignments[i + 1]
                        
                        # If two shifts start close together (within 2 hours), it's likely overlapping
                        # This catches the 8-15 and 9-15 pattern without blocking legitimate consecutive hours
                        if hour2 - hour1 <= 2:
                            # Only add constraint if we detect a pattern of separate shift starts
                            # rather than consecutive hour progression
                            gap_suggests_separate_shifts = (hour2 - hour1 == 1 and 
                                                          len([h for h, _ in day_assignments if hour1 <= h <= hour2]) == 2)
                            
                            if gap_suggests_separate_shifts:
                                self.model.Add(var1 + var2 <= 1)
                                constraints_added += 1
                                
                                logger.debug(f"Added no-overlap constraint for {str(staff_member.staff_id)[-8:]} -> {str(group.group_id)[-8:]} "
                                           f"day {day} hours {hour1} and {hour2}: likely overlapping shifts")
        
        logger.info(f"Added {constraints_added} no-overlapping-shifts constraints")

    def _add_enhanced_qualification_constraints(
        self,
        assignments: Dict[Tuple[UUID, UUID, int, int], cp_model.IntVar],
        staff: List[Staff],
        requirements: List[StaffingRequirement],
        time_slots: List[Tuple[int, int]],
        start_date: date,
    ):
        """Enhanced qualification constraints with proper date mapping."""
        
        # Build a map of each staff member's VERIFIED qualifications
        staff_quals: Dict[UUID, Set[str]] = {}
        for sm in staff:
            quals = {
                q.qualification_name
                for q in sm.qualifications
                if q.is_verified and (not getattr(q, 'expiry_date', None) or q.expiry_date > date.today())
            }
            staff_quals[sm.staff_id] = quals

        # For each requirement, for each hour in its window:
        constraints_added = 0
        for req in requirements:
            if not req.required_qualifications:
                continue

            req_dow = req.time_slot.day_of_week
            start_h = req.time_slot.start_time.hour
            end_h = req.time_slot.end_time.hour

            for day_offset, hour in time_slots:
                # Convert day_offset to actual day_of_week
                actual_date = start_date + timedelta(days=day_offset)
                actual_dow = actual_date.weekday()
                
                # Check if this requirement applies to this time slot
                if actual_dow == req_dow and start_h <= hour < end_h:
                    for sm in staff:
                        # If staff member lacks one of the required quals:
                        if not set(req.required_qualifications).issubset(staff_quals[sm.staff_id]):
                            ak = (sm.staff_id, req.group_id, day_offset, hour)
                            if ak in assignments:
                                # Force them off this assignment
                                self.model.Add(assignments[ak] == 0)
                                constraints_added += 1

        logger.info(f"Added {constraints_added} qualification constraints")
        
    def _add_minimum_shift_duration_constraints(
        self,
        assignments: Dict[Tuple[UUID, UUID, int, int], cp_model.IntVar],
        staff: List[Staff],
        groups: List[Group],
        time_slots: List[Tuple[int, int]],
        start_date: date,
    ):
        """Add constraints to enforce minimum shift duration (prevent 1-hour shifts)"""
        
        min_shift_hours = getattr(settings, "MIN_SHIFT_HOURS", 2)  # Default 2 hours minimum
        constraints_added = 0
        
        # Group assignments by staff, group, and day
        daily_assignments = defaultdict(list)
        for (staff_id, group_id, day_offset, hour), var in assignments.items():
            daily_assignments[(staff_id, group_id, day_offset)].append((hour, var))
        
        for (staff_id, group_id, day_offset), hour_vars in daily_assignments.items():
            if len(hour_vars) < min_shift_hours:
                continue  # Can't have a minimum shift with fewer hours available
                
            # Sort by hour
            hour_vars.sort(key=lambda x: x[0])
            
            # For each possible starting hour, ensure consecutive hours or none
            for i in range(len(hour_vars) - min_shift_hours + 1):
                start_hour, start_var = hour_vars[i]
                consecutive_vars = [start_var]
                
                # Check if the next (min_shift_hours - 1) hours are consecutive
                is_consecutive = True
                for j in range(1, min_shift_hours):
                    if i + j < len(hour_vars):
                        next_hour, next_var = hour_vars[i + j]
                        if next_hour == start_hour + j:
                            consecutive_vars.append(next_var)
                        else:
                            is_consecutive = False
                            break
                    else:
                        is_consecutive = False
                        break
                
                if is_consecutive and len(consecutive_vars) == min_shift_hours:
                    # If first hour is assigned, all consecutive hours must be assigned
                    for var in consecutive_vars[1:]:
                        self.model.Add(var >= start_var)
                    constraints_added += 1
                    
                    # If last hour is assigned, all previous hours must be assigned
                    for var in consecutive_vars[:-1]:
                        self.model.Add(var >= consecutive_vars[-1])
                    constraints_added += 1
        
        logger.info(f"Added {constraints_added} minimum shift duration constraints (min {min_shift_hours} hours)")
        
    def _add_working_hours_constraints_for_period(
        self,
        assignments: Dict[Tuple[UUID, UUID, int, int], cp_model.IntVar],
        staff: List[Staff],
        time_slots: List[Tuple[int, int]],
        start_date: date,
        end_date: date,
    ):
        """Add working hours constraints adjusted for the scheduling period"""

        total_days = (end_date - start_date).days + 1
        period_weeks = max(1, total_days / 7)

        for staff_member in staff:
            # Get all NEW assignments for this staff member (exclude preserved ones)
            staff_assignments = []
            for assignment_key in assignments:
                if assignment_key[0] == staff_member.staff_id:
                    # Check if this assignment is from a preserved existing shift
                    is_preserved = False
                    if hasattr(self, '_existing_schedule') and self._existing_schedule:
                        staff_id, group_id, day_offset, hour = assignment_key
                        actual_date = start_date + timedelta(days=day_offset)
                        for shift in self._existing_schedule:
                            if (shift.staff_id == staff_id and 
                                shift.group_id == group_id and 
                                shift.date == actual_date and
                                shift.start_time.hour <= hour < shift.end_time.hour):
                                is_preserved = True
                                break
                    
                    # Only include non-preserved assignments in hours constraint
                    if not is_preserved:
                        staff_assignments.append(assignments[assignment_key])

            if staff_assignments:
                # Calculate existing hours for this staff member in the scheduling period
                existing_hours = 0
                if hasattr(self, '_existing_schedule') and self._existing_schedule:
                    for shift in self._existing_schedule:
                        if (shift.staff_id == staff_member.staff_id and 
                            start_date <= shift.date <= end_date):
                            existing_hours += shift.scheduled_hours
                
                # Adjust max hours for the period, subtracting existing hours
                max_hours_period = int(staff_member.max_weekly_hours * period_weeks)
                available_hours = max_hours_period - existing_hours
                
                logger.debug(f"HOURS CONSTRAINT: {staff_member.name} - max_period={max_hours_period}, existing={existing_hours}, available={available_hours}")
                
                if available_hours > 0:
                    self.model.Add(sum(staff_assignments) <= available_hours)
                else:
                    # Staff already at or over capacity - block all new assignments
                    logger.warning(f"Staff {staff_member.name} already at capacity ({existing_hours}h >= {max_hours_period}h) - blocking new assignments")
                    for var in staff_assignments:
                        self.model.Add(var == 0)

                # Daily maximum hours constraints - ENHANCED (exclude preserved assignments)
                daily_assignments = {}
                for assignment_key in assignments:
                    if assignment_key[0] == staff_member.staff_id:
                        # Check if this assignment is from a preserved existing shift
                        is_preserved = False
                        if hasattr(self, '_existing_schedule') and self._existing_schedule:
                            staff_id, group_id, day_offset, hour = assignment_key
                            actual_date = start_date + timedelta(days=day_offset)
                            for shift in self._existing_schedule:
                                if (shift.staff_id == staff_id and 
                                    shift.group_id == group_id and 
                                    shift.date == actual_date and
                                    shift.start_time.hour <= hour < shift.end_time.hour):
                                    is_preserved = True
                                    break
                        
                        # Only include non-preserved assignments in daily hours constraint
                        if not is_preserved:
                            day_offset = assignment_key[2]
                            if day_offset not in daily_assignments:
                                daily_assignments[day_offset] = []
                            daily_assignments[day_offset].append(
                                assignments[assignment_key]
                            )

                # Limit daily hours with proper enforcement (must be int for OR-Tools)
                # Temporarily increase the limit to avoid infeasibility during debugging
                max_daily_hours = int(min(
                    getattr(staff_member, "max_daily_hours", None) or 16,
                    getattr(settings, "MAX_DAILY_HOURS", 16),
                    16  # Increased limit during debugging
                ))
                
                for day_offset, day_assignments in daily_assignments.items():
                    # Calculate existing hours for this staff member on this specific day
                    existing_daily_hours = 0
                    if hasattr(self, '_existing_schedule') and self._existing_schedule:
                        for shift in self._existing_schedule:
                            if (shift.staff_id == staff_member.staff_id and 
                                shift.date == start_date + timedelta(days=day_offset)):
                                existing_daily_hours += shift.scheduled_hours
                    
                    available_daily_hours = max_daily_hours - existing_daily_hours
                    
                    actual_date = start_date + timedelta(days=day_offset)
                    logger.info(f"DAILY CONSTRAINT: {staff_member.name} on {actual_date} - max={max_daily_hours}, existing={existing_daily_hours}, available={available_daily_hours}, new_assignments={len(day_assignments)}")
                    
                    if available_daily_hours > 0:
                        self.model.Add(sum(day_assignments) <= available_daily_hours)
                        constraints_added += 1
                    else:
                        # Staff already at or over daily capacity - block all new assignments for this day
                        logger.warning(f"Staff {staff_member.name} already at daily capacity on {actual_date} ({existing_daily_hours}h >= {max_daily_hours}h) - blocking {len(day_assignments)} assignments")
                        for var in day_assignments:
                            self.model.Add(var == 0)
                            constraints_added += 1

    def _add_consecutive_days_constraints(
        self,
        assignments: Dict[Tuple[UUID, UUID, int, int], cp_model.IntVar],
        staff: List[Staff],
        time_slots: List[Tuple[int, int]],
        start_date: date,
        end_date: date,
    ):
        """Add constraints for maximum consecutive working days"""

        max_consecutive_days = getattr(
            self.config,
            "max_consecutive_days",
            getattr(settings, "MAX_CONSECUTIVE_DAYS", 6),
        )

        # Group time slots by day
        days = list(set(day_offset for day_offset, _ in time_slots))
        days.sort()

        for staff_member in staff:
            # Create binary variables for whether staff works each day
            daily_work_vars = {}
            for day_offset in days:
                day_var = self.model.NewBoolVar(
                    f"works_{staff_member.staff_id}_{day_offset}"
                )
                daily_work_vars[day_offset] = day_var

                # Link daily work variable to actual assignments
                day_assignments = []
                for assignment_key in assignments:
                    if (
                        assignment_key[0] == staff_member.staff_id
                        and assignment_key[2] == day_offset
                    ):
                        day_assignments.append(assignments[assignment_key])

                if day_assignments:
                    # If any assignment on this day, then works this day
                    for assignment in day_assignments:
                        self.model.Add(day_var >= assignment)
                    # If works this day, at least one assignment
                    self.model.Add(sum(day_assignments) >= day_var)

            # Consecutive days constraint
            for i in range(len(days) - max_consecutive_days):
                consecutive_days = []
                for j in range(max_consecutive_days + 1):
                    if days[i + j] in daily_work_vars:
                        consecutive_days.append(daily_work_vars[days[i + j]])

                if len(consecutive_days) == max_consecutive_days + 1:
                    self.model.Add(sum(consecutive_days) <= max_consecutive_days)

    def _add_staff_availability_constraints_for_period(
        self,
        assignments: Dict[Tuple[UUID, UUID, int, int], cp_model.IntVar],
        staff: List[Staff],
        time_slots: List[Tuple[int, int]],
        start_date: date,
    ):
        """Ensure staff are only assigned when available"""

        for staff_member in staff:
            # Create availability lookup
            availability_map = {}
            for avail in staff_member.availability:
                availability_map[avail.day_of_week] = avail

            for day_offset, hour in time_slots:
                actual_date = start_date + timedelta(days=day_offset)
                day_of_week = actual_date.weekday()

                # Check if staff is available on this day and time
                if day_of_week in availability_map:
                    avail = availability_map[day_of_week]

                    if not avail.is_available:
                        # Staff not available - prevent any assignments
                        for assignment_key in assignments:
                            if (
                                assignment_key[0] == staff_member.staff_id
                                and assignment_key[2] == day_offset
                                and assignment_key[3] == hour
                            ):
                                self.model.Add(assignments[assignment_key] == 0)

                    elif avail.start_time and avail.end_time:
                        # Check time availability
                        if hour < avail.start_time.hour or hour >= avail.end_time.hour:
                            # Outside available hours
                            for assignment_key in assignments:
                                if (
                                    assignment_key[0] == staff_member.staff_id
                                    and assignment_key[2] == day_offset
                                    and assignment_key[3] == hour
                                ):
                                    self.model.Add(assignments[assignment_key] == 0)

    def _set_enhanced_objective(
        self,
        assignments: Dict[Tuple[UUID, UUID, int, int], cp_model.IntVar],
        staff: List[Staff],
        groups: List[Group],
        time_slots: List[Tuple[int, int]],
    ):
        """Set enhanced objective function"""

        objective_terms = []

        # Use your existing optimization goals or enhance them
        for goal in self.config.goals:
            if goal == OptimizationGoal.MINIMIZE_COST:
                cost_terms = self._create_cost_terms(assignments, staff)
                objective_terms.extend([-term for term in cost_terms])

            elif goal == OptimizationGoal.MAXIMIZE_SATISFACTION:
                satisfaction_terms = self._create_satisfaction_terms(
                    assignments, staff, time_slots
                )
                objective_terms.extend(satisfaction_terms)

            elif goal == OptimizationGoal.MAXIMIZE_FAIRNESS:
                fairness_terms = self._create_fairness_terms(
                    assignments, staff, time_slots
                )
                objective_terms.extend(fairness_terms)

        if objective_terms:
            self.model.Maximize(sum(objective_terms))
        else:
            # Default objective
            total_assignments = sum(assignments.values())
            self.model.Minimize(total_assignments)

    def _create_cost_terms(
        self,
        assignments: Dict[Tuple[UUID, UUID, int, int], cp_model.IntVar],
        staff: List[Staff],
    ) -> List[cp_model.IntVar]:
        """Create cost terms for the objective function without boolean‐testing LinearExprs."""
        cost_terms: List[cp_model.IntVar] = []

        for staff_member in staff:
            # skip if no rate
            if not staff_member.hourly_rate:
                continue

            # collect each assignment’s cost expression
            rate_scaled = int(staff_member.hourly_rate * 100)
            cost_exprs: List[cp_model.LinearExpr] = []
            for (sid, _, _, _), var in assignments.items():
                if sid == staff_member.staff_id:
                    cost_exprs.append(rate_scaled * var)

            # only if we actually have cost expressions
            if cost_exprs:
                total_cost = sum(cost_exprs)  # now a single LinearExpr
                # create a var to hold that total
                cost_var = self.model.NewIntVar(
                    0, len(cost_exprs) * rate_scaled, f"cost_{staff_member.staff_id}"
                )
                self.model.Add(cost_var == total_cost)
                cost_terms.append(cost_var)

        return cost_terms

    def _create_satisfaction_terms(
        self,
        assignments: Dict[Tuple[UUID, UUID, int, int], cp_model.IntVar],
        staff: List[Staff],
        time_slots: List[Tuple[int, int]],
    ) -> List[cp_model.IntVar]:
        """Create satisfaction terms based on staff preferences without boolean‐testing LinearExprs."""
        satisfaction_terms: List[cp_model.IntVar] = []

        for staff_member in staff:
            # collect weighted assignment expressions matching preferences
            pref_exprs: List[cp_model.LinearExpr] = []

            for preference in staff_member.preferences:
                if preference.preference_type != PreferenceType.PREFERRED_TIME:
                    continue

                weight_scaled = int(preference.weight * 100)
                for (sid, _, day_offset, hour), var in assignments.items():
                    if sid != staff_member.staff_id:
                        continue
                    if self._matches_preference(preference, day_offset, hour):
                        pref_exprs.append(weight_scaled * var)

            # only if there are any matching slots
            if pref_exprs:
                total_satisfaction = sum(pref_exprs)  # a LinearExpr
                # bound it by max possible (handle empty preferences)
                if staff_member.preferences:
                    max_weight = max(int(p.weight * 100) for p in staff_member.preferences)
                else:
                    max_weight = 100  # Default weight
                upper = len(pref_exprs) * max_weight
                satisfaction_var = self.model.NewIntVar(
                    0, upper, f"satisfaction_{staff_member.staff_id}"
                )
                self.model.Add(satisfaction_var == total_satisfaction)
                satisfaction_terms.append(satisfaction_var)

        return satisfaction_terms


    def _create_fairness_terms(
        self,
        assignments: Dict[Tuple[UUID, UUID, int, int], cp_model.IntVar],
        staff: List[Staff],
        time_slots: List[Tuple[int, int]],
    ) -> List[cp_model.IntVar]:
        """Create fairness terms to ensure even distribution"""

        fairness_terms = []

        total_slots = len(time_slots)
        if len(staff) == 0:
            return fairness_terms  # Avoid division by zero
        
        avg_hours = total_slots / len(staff)

        for staff_member in staff:
            staff_hours = 0
            for assignment_key in assignments:
                if assignment_key[0] == staff_member.staff_id:
                    staff_hours += assignments[assignment_key]

            target_hours = int(avg_hours * staff_member.priority_score)

            deviation_var = self.model.NewIntVar(
                0, total_slots, f"fairness_{staff_member.staff_id}"
            )

            pos_dev = self.model.NewIntVar(
                0, total_slots, f"pos_dev_{staff_member.staff_id}"
            )
            neg_dev = self.model.NewIntVar(
                0, total_slots, f"neg_dev_{staff_member.staff_id}"
            )

            self.model.Add(staff_hours - target_hours == pos_dev - neg_dev)
            self.model.Add(deviation_var == pos_dev + neg_dev)

            fairness_terms.append(-deviation_var)

        return fairness_terms

    def _matches_preference(
        self, preference: StaffPreference, day_offset: int, hour: int
    ) -> bool:
        """Check if a time slot matches a staff preference"""

        if preference.time_range_start and preference.time_range_end:
            start_hour = preference.time_range_start.hour
            end_hour = preference.time_range_end.hour

            if end_hour <= start_hour:  # Overnight range
                return hour >= start_hour or hour <= end_hour
            else:
                return start_hour <= hour <= end_hour

        return True

    def _preserve_existing_shifts(
        self,
        assignments: Dict[Tuple[UUID, UUID, int, int], cp_model.IntVar],
        existing_schedule: List[ScheduledShift],
        requirements: List[StaffingRequirement],
        start_date: date,
        end_date: date,
    ):
        """Add constraints to preserve existing shifts and block staff during those times"""
        
        preserved_count = 0
        blocked_count = 0

        for shift in existing_schedule:
            # Check if shift falls within our date range
            if start_date <= shift.date <= end_date:
                day_offset = (shift.date - start_date).days
                start_hour = shift.start_time.hour
                end_hour = shift.end_time.hour
                
                # Handle overnight shifts
                if end_hour <= start_hour:
                    end_hour = 24
                
                # For confirmed and scheduled shifts, preserve them exactly
                # BUT only if the group is not over capacity
                if shift.status in [ShiftStatus.CONFIRMED, ShiftStatus.IN_PROGRESS, ShiftStatus.SCHEDULED]:
                    for hour in range(start_hour, min(end_hour, 24)):
                        assignment_key = (shift.staff_id, shift.group_id, day_offset, hour)
                        if assignment_key in assignments:
                            # Check if this group is over capacity before preserving
                            existing_coverage = self._count_existing_coverage(shift.group_id, day_offset, hour, start_date)
                            
                            # Find max staff count for this group/hour from requirements
                            applicable_reqs = []
                            actual_date = start_date + timedelta(days=day_offset)
                            actual_dow = actual_date.weekday()
                            
                            for req in requirements:
                                if (req.group_id == shift.group_id and 
                                    req.time_slot.day_of_week == actual_dow and
                                    req.time_slot.start_time.hour <= hour < req.time_slot.end_time.hour):
                                    applicable_reqs.append(req)
                            
                            if applicable_reqs:
                                max_constraints = [req.max_staff_count for req in applicable_reqs if req.max_staff_count is not None]
                                original_max = min(max_constraints) if max_constraints else max(req.min_staff_count for req in applicable_reqs)
                                
                                logger.debug(f"PRESERVATION CHECK: Group {str(shift.group_id)[-4:]} day {day_offset} hour {hour}: "
                                           f"existing_coverage={existing_coverage}, original_max={original_max}")
                                
                                if existing_coverage <= original_max:
                                    # Group is not over capacity, safe to preserve
                                    self.model.Add(assignments[assignment_key] == 1)
                                    preserved_count += 1
                                    logger.debug(f"Preserved existing shift: {shift.staff_id} on {shift.date} at {hour}:00")
                                else:
                                    logger.info(f"SKIPPED preserving over-capacity shift: {shift.staff_id} on {shift.date} at {hour}:00 "
                                               f"(coverage={existing_coverage} > max={original_max})")
                            else:
                                # No requirements found, preserve as normal
                                logger.debug(f"No requirements found for group {str(shift.group_id)[-4:]} day {day_offset} hour {hour}, preserving anyway")
                                self.model.Add(assignments[assignment_key] == 1)
                                preserved_count += 1
                                logger.debug(f"Preserved existing shift: {shift.staff_id} on {shift.date} at {hour}:00")
                
                # For ALL existing shifts (confirmed or not), block the staff from other assignments during this time
                # BUT only if the original shift was preserved (not skipped due to over-capacity)
                for hour in range(start_hour, min(end_hour, 24)):
                    # Check if this shift hour was actually preserved
                    was_preserved = False
                    assignment_key = (shift.staff_id, shift.group_id, day_offset, hour)
                    if assignment_key in assignments:
                        # Check if this shift was preserved (not skipped due to over-capacity)
                        existing_coverage = self._count_existing_coverage(shift.group_id, day_offset, hour, start_date)
                        applicable_reqs = []
                        actual_date = start_date + timedelta(days=day_offset)
                        actual_dow = actual_date.weekday()
                        
                        for req in requirements:
                            if (req.group_id == shift.group_id and 
                                req.time_slot.day_of_week == actual_dow and
                                req.time_slot.start_time.hour <= hour < req.time_slot.end_time.hour):
                                applicable_reqs.append(req)
                        
                        if applicable_reqs:
                            max_constraints = [req.max_staff_count for req in applicable_reqs if req.max_staff_count is not None]
                            original_max = min(max_constraints) if max_constraints else max(req.min_staff_count for req in applicable_reqs)
                            was_preserved = existing_coverage <= original_max
                        else:
                            was_preserved = True  # No requirements, so it was preserved
                    
                    # Only block other assignments if this shift was actually preserved
                    if was_preserved:
                        # Find all groups to block from
                        blocked_this_hour = 0
                        for assignment_key, var in assignments.items():
                            assigned_staff_id, assigned_group_id, assigned_day, assigned_hour = assignment_key
                            
                            # If this is the same staff member on the same day/hour but different group
                            if (assigned_staff_id == shift.staff_id and 
                                assigned_day == day_offset and 
                                assigned_hour == hour and 
                                assigned_group_id != shift.group_id):
                                
                                # Block this assignment - staff is already occupied
                                self.model.Add(var == 0)
                                blocked_this_hour += 1
                                logger.debug(f"BLOCKED: {str(shift.staff_id)[-8:]} from {str(assigned_group_id)[-8:]} on {shift.date} at {hour}:00")
                        blocked_count += blocked_this_hour
                    else:
                        logger.debug(f"NOT blocking other assignments for skipped over-capacity shift: {shift.staff_id} on {shift.date} at {hour}:00")
        
        logger.info(f"PRESERVATION SUMMARY: preserved={preserved_count}, blocked={blocked_count} existing shifts")
    
    def _preserve_existing_shifts_simple(
        self,
        assignments: Dict[Tuple[UUID, UUID, int, int], cp_model.IntVar],
        existing_schedule: List[ScheduledShift],
        start_date: date,
        end_date: date,
    ):
        """Simplified version that only preserves confirmed shifts without blocking"""
        
        preserved_count = 0

        for shift in existing_schedule:
            # Only preserve confirmed shifts
            if shift.status in [ShiftStatus.CONFIRMED, ShiftStatus.IN_PROGRESS]:
                # Check if shift falls within our date range
                if start_date <= shift.date <= end_date:
                    day_offset = (shift.date - start_date).days
                    start_hour = shift.start_time.hour
                    
                    # Only preserve the first hour for simplicity
                    assignment_key = (shift.staff_id, shift.group_id, day_offset, start_hour)
                    if assignment_key in assignments:
                        self.model.Add(assignments[assignment_key] == 1)
                        preserved_count += 1
                        logger.debug(f"Preserved existing shift: {shift.staff_id} on {shift.date} at {start_hour}:00")
        
        logger.info(f"Preserved {preserved_count} existing shift assignments (simplified version)")

    def _extract_enhanced_schedule(
        self,
        assignments: Dict[Tuple[UUID, UUID, int, int], cp_model.IntVar],
        staff: List[Staff],
        groups: List[Group],
        time_slots: List[Tuple[int, int]],
        start_date: date,
    ) -> List[ScheduledShift]:
        """Extract the solved schedule from decision variables"""

        # Extract newly generated shifts from solver
        new_shifts = []

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

                new_shifts.append(shift)

        # Combine new shifts with existing schedules for comprehensive merge
        all_shifts = new_shifts.copy()
        
        # Add existing schedules that fall within our date range
        if hasattr(self, '_existing_schedule') and self._existing_schedule:
            end_date = start_date + timedelta(days=max(day for day, _ in time_slots))
            for existing_shift in self._existing_schedule:
                if start_date <= existing_shift.date <= end_date:
                    all_shifts.append(existing_shift)
                    logger.debug(f"Including existing shift in merge: {str(existing_shift.staff_id)[-8:]} -> {str(existing_shift.group_id)[-8:]} on {existing_shift.date} {existing_shift.start_time}-{existing_shift.end_time}")

        # Merge consecutive shifts and handle overlapping assignments intelligently
        merged_schedule = self._merge_and_optimize_shifts(all_shifts)

        logger.info(f"Extracted {len(new_shifts)} new shifts + {len(all_shifts) - len(new_shifts)} existing shifts, merged into {len(merged_schedule)} final shifts")
        return merged_schedule

    def _merge_consecutive_shifts(
        self, schedule: List[ScheduledShift]
    ) -> List[ScheduledShift]:
        """Merge consecutive hourly shifts into longer shifts"""

        if not schedule:
            return schedule

        # Group by staff, group, and date
        grouped = {}
        for shift in schedule:
            key = (shift.staff_id, shift.group_id, shift.date)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(shift)

        merged_schedule = []

        for key, shifts in grouped.items():
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
                    merged_shifts.append(current_shift)
                    current_shift = next_shift

            merged_shifts.append(current_shift)
            merged_schedule.extend(merged_shifts)

        return merged_schedule
    
    def _merge_and_optimize_shifts(
        self, schedule: List[ScheduledShift]
    ) -> List[ScheduledShift]:
        """Smart merging that handles overlapping assignments and creates reasonable shifts"""

        if not schedule:
            return schedule

        # Group by staff and date
        staff_daily_shifts = defaultdict(lambda: defaultdict(list))
        for shift in schedule:
            staff_daily_shifts[shift.staff_id][shift.date].append(shift)

        optimized_schedule = []

        for staff_id, daily_shifts in staff_daily_shifts.items():
            for date, shifts in daily_shifts.items():
                if len(shifts) == 1:
                    # Single shift, keep as is
                    optimized_schedule.append(shifts[0])
                    continue

                # Multiple shifts for same staff on same day - optimize
                shifts.sort(key=lambda s: s.start_time)
                
                # Strategy: Create one consolidated shift per group (proper handling)
                group_shifts = defaultdict(list)
                for shift in shifts:
                    group_shifts[shift.group_id].append(shift)
                
                # Process each group separately and merge overlapping shifts within each group
                for group_id, group_shift_list in group_shifts.items():
                    if len(group_shift_list) == 1:
                        # Single shift for this group, keep as is
                        optimized_schedule.append(group_shift_list[0])
                    else:
                        # Multiple shifts for same group - merge overlapping/adjacent shifts
                        group_shift_list.sort(key=lambda s: s.start_time)
                        
                        merged_shifts = []
                        current_shift = group_shift_list[0]
                        
                        for next_shift in group_shift_list[1:]:
                            # Check if shifts overlap or are adjacent (within 1 hour gap)
                            current_end_hour = current_shift.end_time.hour
                            current_end_minute = current_shift.end_time.minute
                            next_start_hour = next_shift.start_time.hour
                            next_start_minute = next_shift.start_time.minute
                            
                            # Calculate time difference in minutes
                            current_end_minutes = current_end_hour * 60 + current_end_minute
                            next_start_minutes = next_start_hour * 60 + next_start_minute
                            gap_minutes = next_start_minutes - current_end_minutes
                            
                            # If gap is <= 60 minutes (1 hour), merge the shifts
                            if gap_minutes <= 60:
                                # Merge: extend current shift to cover both periods
                                earliest_start = min(current_shift.start_time, next_shift.start_time)
                                latest_end = max(current_shift.end_time, next_shift.end_time)
                                
                                # Calculate total hours properly based on time span
                                start_minutes = earliest_start.hour * 60 + earliest_start.minute
                                end_minutes = latest_end.hour * 60 + latest_end.minute
                                total_minutes = end_minutes - start_minutes
                                total_hours = min(12.0, total_minutes / 60.0)  # Cap at 12 hours
                                
                                current_shift = ScheduledShift(
                                    staff_id=current_shift.staff_id,
                                    group_id=current_shift.group_id,
                                    date=current_shift.date,
                                    start_time=earliest_start,
                                    end_time=latest_end,
                                    scheduled_hours=total_hours,
                                    status=ShiftStatus.SCHEDULED,
                                    schedule_id=None,
                                )
                                
                                logger.debug(f"Merged overlapping shifts for {str(staff_id)[-8:]} -> {str(group_id)[-8:]} on {date}: "
                                           f"{current_shift.start_time}-{current_shift.end_time} ({total_hours}h)")
                            else:
                                # No overlap, keep current shift and start processing next one
                                merged_shifts.append(current_shift)
                                current_shift = next_shift
                        
                        # Add the last processed shift
                        merged_shifts.append(current_shift)
                        
                        # Add all merged shifts for this group
                        optimized_schedule.extend(merged_shifts)
                        
                        logger.debug(f"Optimized {len(group_shift_list)} shifts for {str(staff_id)[-8:]} -> {str(group_id)[-8:]} on {date} into {len(merged_shifts)} consolidated shifts")

        logger.info(f"Optimized {len(schedule)} raw shifts into {len(optimized_schedule)} consolidated shifts")
        return optimized_schedule

    def _detect_enhanced_conflicts(
        self,
        schedule: List[ScheduledShift],
        staff: List[Staff],
        groups: List[Group],
        requirements: List[StaffingRequirement],
    ) -> List[ScheduleConflict]:
        """Detect conflicts in the enhanced schedule"""

        conflicts = []

        # Use your existing conflict detection or enhance it
        # Add period-specific conflict detection here

        return conflicts

    def _analyze_infeasibility_for_range(
        self,
        staff: List[Staff],
        groups: List[Group],
        requirements: List[StaffingRequirement],
        time_slots: List[Tuple[int, int]],
        total_days: int,
        schedule_start_date: date,
        schedule_end_date: date,
    ) -> List[ScheduleConflict]:
        """Enhanced infeasibility analysis with detailed conflicts"""
        
        # Use the provided schedule dates
        
        return generate_detailed_conflicts(
            staff, groups, requirements, schedule_start_date, schedule_end_date
        )  
  
    def _get_time_slot_duration(self, time_slot: TimeSlot) -> float:
        """Calculate duration of a time slot in hours"""

        start_dt = datetime.combine(date.today(), time_slot.start_time)
        end_dt = datetime.combine(date.today(), time_slot.end_time)

        if end_dt <= start_dt:
            end_dt += timedelta(days=1)

        return (end_dt - start_dt).total_seconds() / 3600

    # Backward compatibility method
    def solve(self, *args, **kwargs):
        """
        Backward compatibility wrapper for your existing solve method
        If called with new parameters, use enhanced solve, otherwise use original
        """

        # Check if this is a date range call
        if "schedule_start_date" in kwargs and "schedule_end_date" in kwargs:
            return self.solve_with_date_range(*args, **kwargs)
        elif len(args) >= 6 and isinstance(args[4], date) and isinstance(args[5], date):
            # Positional arguments with date range
            return self.solve_with_date_range(*args, **kwargs)
        else:
            # Fall back to your existing solve method
            # You'll need to implement this based on your current solver
            return self._solve_original(*args, **kwargs)

    def _solve_original(self, *args, **kwargs):
        """Your original solve method - implement based on your current solver"""
        # This should contain your existing solve logic
        # For now, I'll assume it's similar but with a week_start_date parameter

        if "week_start_date" in kwargs:
            week_start = kwargs["week_start_date"]
            week_end = week_start + timedelta(days=6)
            kwargs["schedule_start_date"] = week_start
            kwargs["schedule_end_date"] = week_end
            del kwargs["week_start_date"]

            return self.solve_with_date_range(*args, **kwargs)

        # If no date parameters, default to current week
        from datetime import date

        today = date.today()
        week_start = today - timedelta(days=today.weekday())
        week_end = week_start + timedelta(days=6)

        return self.solve_with_date_range(
            *args, schedule_start_date=week_start, schedule_end_date=week_end, **kwargs
        )
    
    def debug_solver_state(
        self,
        staff: List[Staff],
        groups: List[Group], 
        requirements: List[StaffingRequirement],
        schedule_start_date: date,
        schedule_end_date: date,
        assignments: Dict[Tuple[UUID, UUID, int, int], cp_model.IntVar] = None
    ):
        """
        Debug method to trace exactly what's happening in the solver
        Add this to your EnhancedScheduleSolver class
        """
        
        print("\n" + "="*80)
        print("🔍 DEBUGGING SOLVER STATE")
        print("="*80)
        
        total_days = (schedule_end_date - schedule_start_date).days + 1
        
        print(f"📅 Schedule period: {schedule_start_date} to {schedule_end_date} ({total_days} days)")
        print(f"Start day of week: {schedule_start_date.weekday()} ({schedule_start_date.strftime('%A')})")
        
        # Create day mapping 
        dow_to_offsets = {}
        print(f"\nDay mapping:")
        for day_offset in range(total_days):
            actual_date = schedule_start_date + timedelta(days=day_offset)
            actual_dow = actual_date.weekday()
            
            if actual_dow not in dow_to_offsets:
                dow_to_offsets[actual_dow] = []
            dow_to_offsets[actual_dow].append(day_offset)
            
            print(f"  Day offset {day_offset} = {actual_date.strftime('%A %Y-%m-%d')} (DOW {actual_dow})")
        
        print(f"\n👥 Staff analysis:")
        for i, staff_member in enumerate(staff):
            print(f"  {i+1}. {staff_member.name} ({str(staff_member.staff_id)[:8]}...)")
            
            # Check qualifications
            verified_quals = [
                q.qualification_name for q in staff_member.qualifications 
                if q.is_verified and (not hasattr(q, 'expiry_date') or not q.expiry_date or q.expiry_date > date.today())
            ]
            print(f"     Qualifications: {verified_quals}")
            
            # Check availability
            for avail in staff_member.availability:
                day_name = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][avail.day_of_week]
                status = "Available" if avail.is_available else "NOT available"
                print(f"     {status} on {day_name} (DOW {avail.day_of_week}) from {avail.start_time} to {avail.end_time}")
        
        print(f"\n📋 Requirements analysis:")
        for i, req in enumerate(requirements):
            print(f"  {i+1}. Group {str(req.group_id)[:8]}...")
            print(f"     Day: {req.time_slot.day_of_week} ({'Mon,Tue,Wed,Thu,Fri,Sat,Sun'.split(',')[req.time_slot.day_of_week]})")
            print(f"     Time: {req.time_slot.start_time} - {req.time_slot.end_time}")
            print(f"     Staff: {req.min_staff_count} - {req.max_staff_count}")
            print(f"     Quals: {req.required_qualifications}")
            
            # Check if this DOW exists in our period
            req_dow = req.time_slot.day_of_week
            if req_dow in dow_to_offsets:
                print(f"     ✓ DOW {req_dow} maps to day offsets: {dow_to_offsets[req_dow]}")
            else:
                print(f"     ❌ DOW {req_dow} NOT FOUND in scheduling period!")
        
        # If assignments are provided, analyze them
        if assignments:
            print(f"\n🔧 Variable analysis:")
            print(f"Total variables created: {len(assignments)}")
            
            # Group by slot
            slots = defaultdict(list)
            for (staff_id, group_id, day_offset, hour), var in assignments.items():
                slots[(group_id, day_offset, hour)].append((staff_id, var))
            
            print(f"Unique time slots with variables: {len(slots)}")
            
            # Check each requirement against available variables
            for i, req in enumerate(requirements):
                print(f"\nRequirement {i+1} coverage:")
                req_dow = req.time_slot.day_of_week
                start_hour = req.time_slot.start_time.hour
                end_hour = req.time_slot.end_time.hour
                
                if req_dow not in dow_to_offsets:
                    print(f"  ❌ No day offsets for DOW {req_dow}")
                    continue
                    
                for day_offset in dow_to_offsets[req_dow]:
                    print(f"  Day offset {day_offset}:")
                    for hour in range(start_hour, end_hour):
                        slot_key = (req.group_id, day_offset, hour)
                        if slot_key in slots:
                            var_count = len(slots[slot_key])
                            status = "✓" if var_count >= req.min_staff_count else "❌"
                            print(f"    Hour {hour}: {var_count} variables {status} (need {req.min_staff_count})")
                        else:
                            print(f"    Hour {hour}: 0 variables ❌ (need {req.min_staff_count})")
        
        else:
            print(f"\n🔧 Simulating variable creation:")
            
            # Simulate what variables would be created
            covered_slots = set()
            for req in requirements:
                req_dow = req.time_slot.day_of_week
                start_hour = req.time_slot.start_time.hour  
                end_hour = req.time_slot.end_time.hour
                
                if req_dow in dow_to_offsets:
                    for day_offset in dow_to_offsets[req_dow]:
                        for hour in range(start_hour, end_hour):
                            covered_slots.add((req.group_id, day_offset, hour))
            
            print(f"Slots that need variables: {len(covered_slots)}")
            
            variables_would_be_created = 0
            problem_slots = []
            
            for group_id, day_offset, hour in sorted(covered_slots):
                actual_date = schedule_start_date + timedelta(days=day_offset)
                actual_dow = actual_date.weekday()
                
                # Find applicable requirements
                applicable_reqs = [
                    req for req in requirements
                    if req.group_id == group_id
                    and req.time_slot.day_of_week == actual_dow  
                    and req.time_slot.start_time.hour <= hour < req.time_slot.end_time.hour
                ]
                
                # Get required qualifications
                all_required_quals = set()
                for req in applicable_reqs:
                    if req.required_qualifications:
                        all_required_quals.update(req.required_qualifications)
                
                # Check each staff member
                available_staff_count = 0
                for staff_member in staff:
                    # Check qualifications
                    staff_quals = {
                        q.qualification_name for q in staff_member.qualifications
                        if q.is_verified and (not hasattr(q, 'expiry_date') or not q.expiry_date or q.expiry_date > date.today())
                    }
                    
                    if all_required_quals and not all_required_quals.issubset(staff_quals):
                        continue
                    
                    # Check availability
                    is_available = any(
                        avail.day_of_week == actual_dow
                        and avail.is_available  
                        and avail.start_time.hour <= hour < avail.end_time.hour
                        for avail in staff_member.availability
                    )
                    
                    if is_available:
                        available_staff_count += 1
                        variables_would_be_created += 1
                
                min_needed = max(req.min_staff_count for req in applicable_reqs) if applicable_reqs else 0
                
                if available_staff_count < min_needed:
                    problem_slots.append({
                        'group_id': str(group_id)[:8],
                        'day_offset': day_offset,
                        'hour': hour,
                        'available': available_staff_count,
                        'needed': min_needed,
                        'date': actual_date.strftime('%A %Y-%m-%d')
                    })
            
            print(f"Variables that would be created: {variables_would_be_created}")
            
            if problem_slots:
                print(f"\n❌ PROBLEM SLOTS ({len(problem_slots)} found):")
                for slot in problem_slots:
                    print(f"  Group {slot['group_id']}..., {slot['date']}, hour {slot['hour']}: {slot['available']} available, {slot['needed']} needed")
            else:
                print(f"\n✅ All slots have sufficient staff")
        
        print("="*80 + "\n")
        
        # Return summary for programmatic use
        return {
            'total_days': total_days,
            'dow_mapping': dow_to_offsets,
            'requirements_count': len(requirements),
            'staff_count': len(staff),
            'variables_count': len(assignments) if assignments else variables_would_be_created
        }

def generate_detailed_conflicts(
    staff: List[Staff],
    groups: List[Group], 
    requirements: List[StaffingRequirement],
    schedule_start_date: date,
    schedule_end_date: date = None
) -> List[ScheduleConflict]:
    """
    Generate detailed, actionable conflict reports for scheduling issues
    """
    
    conflicts = []
    
    if schedule_end_date is None:
        schedule_end_date = schedule_start_date + timedelta(days=6)
    
    total_days = (schedule_end_date - schedule_start_date).days + 1
    
    # Build day mapping
    dow_to_offsets = {}
    for day_offset in range(total_days):
        actual_date = schedule_start_date + timedelta(days=day_offset)
        actual_dow = actual_date.weekday()
        if actual_dow not in dow_to_offsets:
            dow_to_offsets[actual_dow] = []
        dow_to_offsets[actual_dow].append((day_offset, actual_date))
    
    # Analyze each requirement
    for req_idx, req in enumerate(requirements):
        req_dow = req.time_slot.day_of_week
        start_hour = req.time_slot.start_time.hour
        end_hour = req.time_slot.end_time.hour
        group_id = req.group_id
        
        # Find the group
        group = next((g for g in groups if g.group_id == group_id), None)
        group_name = group.name if group else f"Group {str(group_id)[:8]}..."
        
        if req_dow not in dow_to_offsets:
            conflicts.append(ScheduleConflict(
                conflict_type="invalid_day",
                severity="error",
                group_id=group_id,
                time_slot=req.time_slot,
                description=f"Requirement {req_idx + 1} ({group_name}) needs coverage on {['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][req_dow]}, but this day is not in the scheduling period ({schedule_start_date} to {schedule_end_date})",
                suggested_solutions=[
                    "Adjust the scheduling period to include this day",
                    "Change the requirement to a different day",
                    "Remove this requirement"
                ]
            ))
            continue
        
        # Check each hour of this requirement
        for hour in range(start_hour, end_hour):
            for day_offset, actual_date in dow_to_offsets[req_dow]:
                
                # Find all requirements that apply to this specific slot
                overlapping_reqs = []
                for other_req in requirements:
                    if (other_req.group_id == group_id and 
                        other_req.time_slot.day_of_week == req_dow and
                        other_req.time_slot.start_time.hour <= hour < other_req.time_slot.end_time.hour):
                        overlapping_reqs.append(other_req)
                
                # Calculate total needed staff (using current logic)
                min_needed_current = max(r.min_staff_count for r in overlapping_reqs)
                max_allowed_current = min(r.max_staff_count for r in overlapping_reqs if r.max_staff_count is not None) if any(r.max_staff_count is not None for r in overlapping_reqs) else None
                
                # Calculate total needed staff (using additive logic)
                min_needed_additive = sum(r.min_staff_count for r in overlapping_reqs)
                
                # Get required qualifications
                all_required_quals = set()
                for r in overlapping_reqs:
                    if r.required_qualifications:
                        all_required_quals.update(r.required_qualifications)
                
                # Analyze staff availability for this slot
                available_staff = []
                unavailable_staff = []
                
                for staff_member in staff:
                    # Check qualifications
                    staff_quals = {
                        q.qualification_name for q in staff_member.qualifications
                        if q.is_verified and (not hasattr(q, 'expiry_date') or not q.expiry_date or q.expiry_date > date.today())
                    }
                    
                    has_required_quals = all_required_quals.issubset(staff_quals)
                    missing_quals = all_required_quals - staff_quals
                    
                    # Check availability
                    is_available = False
                    availability_reason = "No availability defined"
                    
                    for avail in staff_member.availability:
                        if (avail.day_of_week == actual_date.weekday() and 
                            avail.is_available):
                            
                            avail_start = avail.start_time.hour if hasattr(avail.start_time, 'hour') else int(str(avail.start_time).split(':')[0])
                            avail_end = avail.end_time.hour if hasattr(avail.end_time, 'hour') else int(str(avail.end_time).split(':')[0])
                            
                            if avail_start <= hour < avail_end:
                                is_available = True
                                availability_reason = "Available"
                                break
                            elif avail.day_of_week == actual_date.weekday():
                                availability_reason = f"Available {avail_start}:00-{avail_end}:00, but not at {hour}:00"
                        elif avail.day_of_week == actual_date.weekday():
                            availability_reason = "Marked as unavailable"
                    
                    if has_required_quals and is_available:
                        available_staff.append(staff_member)
                    else:
                        reasons = []
                        if not has_required_quals:
                            reasons.append(f"Missing qualifications: {list(missing_quals)}")
                        if not is_available:
                            reasons.append(availability_reason)
                        
                        unavailable_staff.append({
                            'staff': staff_member,
                            'reasons': reasons
                        })
                
                # Check for conflicts
                available_count = len(available_staff)
                
                # Check current logic conflicts
                if available_count < min_needed_current:
                    
                    # Generate detailed description
                    time_str = f"{hour:02d}:00-{hour+1:02d}:00"
                    day_str = actual_date.strftime('%A, %Y-%m-%d')
                    
                    description_parts = [
                        f"Insufficient staffing for {group_name} on {day_str} at {time_str}:",
                        f"• Available: {available_count} staff",
                        f"• Required: {min_needed_current} staff (from {len(overlapping_reqs)} overlapping requirement{'s' if len(overlapping_reqs) > 1 else ''})"
                    ]
                    
                    if len(overlapping_reqs) > 1:
                        description_parts.append("• Overlapping requirements:")
                        for i, r in enumerate(overlapping_reqs):
                            req_time = f"{r.time_slot.start_time}-{r.time_slot.end_time}"
                            description_parts.append(f"  - Req {i+1}: {req_time}, needs {r.min_staff_count}-{r.max_staff_count or 'unlimited'} staff")
                    
                    if all_required_quals:
                        description_parts.append(f"• Required qualifications: {list(all_required_quals)}")
                    
                    if available_staff:
                        description_parts.append(f"• Available staff: {[s.name for s in available_staff]}")
                    
                    if unavailable_staff:
                        description_parts.append("• Unavailable staff:")
                        for item in unavailable_staff:
                            reasons_str = "; ".join(item['reasons'])
                            description_parts.append(f"  - {item['staff'].name}: {reasons_str}")
                    
                    description = "\n".join(description_parts)
                    
                    # Generate specific solutions
                    solutions = []
                    
                    # Qualification-based solutions
                    qual_issues = [item for item in unavailable_staff if any('qualifications' in reason for reason in item['reasons'])]
                    if qual_issues:
                        solutions.append(f"Train staff in required qualifications: {list(all_required_quals)}")
                        solutions.append("Hire staff with required qualifications")
                    
                    # Availability-based solutions
                    avail_issues = [item for item in unavailable_staff if any('Available' in reason or 'unavailable' in reason for reason in item['reasons'])]
                    if avail_issues:
                        solutions.append(f"Extend availability for staff: {[item['staff'].name for item in avail_issues]}")
                        solutions.append(f"Adjust requirement time from {time_str} to match staff availability")
                    
                    # Scheduling solutions
                    if len(overlapping_reqs) > 1 and max_allowed_current is not None and min_needed_current > max_allowed_current:
                        solutions.append("Fix conflicting requirements: cannot need both minimum and maximum staff simultaneously")
                        solutions.append(f"Consider using additive staffing logic (would need {min_needed_additive} staff)")
                    
                    if available_count == 0:
                        solutions.append("Add more staff to the center")
                        solutions.append("Use substitute/temporary staff")
                    
                    if not solutions:
                        solutions = [
                            "Review staffing requirements",
                            "Adjust schedule parameters",
                            "Contact scheduling administrator"
                        ]
                    
                    conflicts.append(ScheduleConflict(
                        conflict_type="insufficient_staffing",
                        severity="error",
                        group_id=group_id,
                        time_slot=TimeSlot(
                            start_time=datetime_time(hour, 0),
                            end_time=datetime_time(hour + 1, 0),
                            day_of_week=actual_date.weekday()
                        ),
                        description=description,
                        suggested_solutions=solutions
                    ))
                
                # Check for impossible constraints (min > max)
                if max_allowed_current is not None and min_needed_current > max_allowed_current:
                    conflicts.append(ScheduleConflict(
                        conflict_type="impossible_constraint",
                        severity="error", 
                        group_id=group_id,
                        time_slot=TimeSlot(
                            start_time=datetime_time(hour, 0),
                            end_time=datetime_time(hour + 1, 0),
                            day_of_week=actual_date.weekday()
                        ),
                        description=f"Impossible staffing constraint for {group_name} on {actual_date.strftime('%A, %Y-%m-%d')} at {hour:02d}:00-{hour+1:02d}:00: Need at least {min_needed_current} staff but maximum allowed is {max_allowed_current} (from {len(overlapping_reqs)} conflicting requirements)",
                        suggested_solutions=[
                            "Fix conflicting requirements - they cannot overlap with incompatible min/max constraints",
                            f"Consider using additive staffing logic (would need {min_needed_additive} staff)",
                            "Split overlapping requirements into separate time periods",
                            "Increase max_staff_count for one or more requirements"
                        ]
                    ))
    
    return conflicts

