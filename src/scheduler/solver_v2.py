"""
Enhanced OR-Tools CP-SAT solver with shift templates and group assignment preferences
"""

import logging
from datetime import datetime, timedelta, time as datetime_time, date, time
import time as _time_module
from typing import List, Dict, Tuple, Optional, Set
from ortools.sat.python import cp_model
from uuid import UUID
from collections import defaultdict

from .models import *
from .constraints import ConstraintBuilder
from .config import settings

logger = logging.getLogger(__name__)


class ScheduleSolverV2:
    """Solver with shift templates and group assignment preferences"""

    def __init__(self, optimization_config: OptimizationConfig):
        self.config = optimization_config
        self.model = cp_model.CpModel()
        self.solver = cp_model.CpSolver()
        
        # Configure solver parameters
        self.solver.parameters.max_time_in_seconds = optimization_config.max_solver_time
        self.solver.parameters.num_search_workers = getattr(settings, "SOLVER_WORKERS", 4)
        self.solver.parameters.log_search_progress = getattr(settings, "LOG_SOLVER_PROGRESS", True)
        
        # Cache for shift templates
        self._shift_template_cache: Dict[UUID, ShiftTemplate] = {}
        
    def solve_with_templates(
        self,
        staff: List[Staff],
        groups: List[Group],
        requirements: List[StaffingRequirement],
        shift_templates: List[ShiftTemplate],
        shift_template_requirements: List[ShiftTemplateRequirement],
        constraints: List[ScheduleConstraint],
        schedule_start_date: date,
        schedule_end_date: date,
        existing_schedule: Optional[List[ScheduledShift]] = None,
    ) -> Tuple[List[ScheduledShift], OptimizationResult, List[ScheduleConflict]]:
        """
        Solve scheduling problem with shift templates and group assignment preferences
        """
        method_start_time = _time_module.time()
        try:
            # If no templates provided, use time-based scheduling with group preferences
            if not shift_templates:
                return self._solve_with_group_preferences(
                    staff, groups, requirements, constraints,
                    schedule_start_date, schedule_end_date, existing_schedule
                )
            
            # Store shift templates in cache
            self._shift_template_cache = {st.shift_template_id: st for st in shift_templates}
            
            # Validate date range
            if schedule_end_date < schedule_start_date:
                raise ValueError("End date must be after start date")
            total_days = (schedule_end_date - schedule_start_date).days + 1
            if total_days > 365:
                raise ValueError(f"Date range too large: {total_days} days (max 365)")
                
            logger.info(f"SOLVER SETUP: {len(staff)} staff, {len(groups)} groups")
            logger.info(f"SHIFT TEMPLATES: {len(shift_templates)} templates, {len(shift_template_requirements)} requirements")
            logger.info(f"DATE RANGE: {schedule_start_date} to {schedule_end_date} ({total_days} days)")
            
            # 1) Create shift template assignment variables
            template_assignments = self._create_template_assignment_variables(
                staff, groups, shift_templates, shift_template_requirements,
                schedule_start_date, schedule_end_date
            )
            
            # 2) Add shift template constraints
            self._add_shift_template_constraints(
                template_assignments, staff, groups, shift_templates,
                shift_template_requirements, schedule_start_date, schedule_end_date
            )
            
            # 3) Add group assignment preference constraints
            self._add_group_assignment_preference_constraints(
                template_assignments, staff, groups
            )
            
            # 4) Add group continuity constraints
            self._add_group_continuity_constraints(
                template_assignments, staff, groups, schedule_start_date, schedule_end_date
            )
            
            # 5) Add staff working hours constraints
            self._add_staff_hours_constraints(
                template_assignments, staff, shift_templates, schedule_start_date, schedule_end_date
            )
            
            # 6) Preserve existing schedules
            if existing_schedule:
                self._preserve_existing_template_shifts(
                    template_assignments, existing_schedule, schedule_start_date, schedule_end_date
                )
            
            # 7) Set objective function
            self._set_template_objective(
                template_assignments, staff, groups, shift_templates
            )
            
            # 8) Solve
            logger.info(f"SOLVING: Starting solver with {len(template_assignments)} variables")
            status = self.solver.Solve(self.model)
            solve_time = _time_module.time() - method_start_time
            logger.info(f"SOLVING: Solver finished with status: {status} in {solve_time:.2f}s")
            
            # 9) Process results
            if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
                schedule = self._extract_template_schedule(
                    template_assignments, staff, groups, shift_templates,
                    schedule_start_date
                )
                conflicts = self._detect_schedule_conflicts(
                    schedule, staff, groups, requirements
                )
                optimization_result = OptimizationResult(
                    objective_value=self.solver.ObjectiveValue(),
                    solve_time_seconds=solve_time,
                    status="OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE",
                    iterations=self.solver.NumBranches(),
                    conflicts_resolved=len(conflicts),
                )
                return schedule, optimization_result, conflicts
                
            # Handle infeasible or timeout
            status_name = "INFEASIBLE" if status == cp_model.INFEASIBLE else "TIMEOUT"
            optimization_result = OptimizationResult(
                objective_value=0,
                solve_time_seconds=solve_time,
                status=status_name,
                iterations=self.solver.NumBranches(),
                conflicts_resolved=0,
            )
            conflicts = self._analyze_template_infeasibility(
                staff, groups, shift_templates, shift_template_requirements
            )
            return [], optimization_result, conflicts
            
        except Exception as e:
            logger.error(f"Error in template schedule optimization: {str(e)}")
            optimization_result = OptimizationResult(
                objective_value=0,
                solve_time_seconds=_time_module.time() - method_start_time,
                status="ERROR",
                iterations=0,
                conflicts_resolved=0,
            )
            conflicts = [
                ScheduleConflict(
                    conflict_type="solver_error",
                    severity="error",
                    description=f"Solver error: {str(e)}",
                    suggested_solutions=["Check input data", "Reduce problem size"],
                )
            ]
            return [], optimization_result, conflicts
    
    def _create_template_assignment_variables(
        self,
        staff: List[Staff],
        groups: List[Group],
        shift_templates: List[ShiftTemplate],
        shift_template_requirements: List[ShiftTemplateRequirement],
        start_date: date,
        end_date: date,
    ) -> Dict[Tuple[UUID, UUID, UUID, date], cp_model.IntVar]:
        """Create binary variables for shift template assignments"""
        
        assignments = {}
        total_days = (end_date - start_date).days + 1
        
        # For each day in the range
        for day_offset in range(total_days):
            current_date = start_date + timedelta(days=day_offset)
            day_of_week = current_date.weekday()
            
            # For each shift template requirement
            for str_req in shift_template_requirements:
                # Check if this requirement applies to this day
                if str_req.day_of_week is None or str_req.day_of_week == day_of_week:
                    # Get the shift template
                    template = self._shift_template_cache.get(str_req.shift_template_id)
                    if not template or not template.is_active:
                        continue
                    
                    # Get the group
                    group = next((g for g in groups if g.group_id == str_req.group_id), None)
                    if not group:
                        continue
                    
                    # Check if this template is assigned to this group
                    if str_req.shift_template_id not in group.shift_template_ids:
                        continue
                    
                    # For each staff member
                    for staff_member in staff:
                        # Check if staff has required qualifications
                        staff_quals = {
                            q.qualification_name for q in staff_member.qualifications
                            if q.is_verified
                        }
                        if not all(rq in staff_quals for rq in template.required_qualifications):
                            continue
                        
                        # Check availability
                        is_available = self._check_staff_availability(
                            staff_member, current_date, template.start_time, template.end_time
                        )
                        if not is_available:
                            continue
                        
                        # Create variable
                        var_name = f"assign_{staff_member.staff_id}_{group.group_id}_{template.shift_template_id}_{current_date}"
                        var = self.model.NewBoolVar(var_name)
                        assignments[(staff_member.staff_id, group.group_id, template.shift_template_id, current_date)] = var
        
        logger.info(f"Created {len(assignments)} template assignment variables")
        return assignments
    
    def _add_shift_template_constraints(
        self,
        assignments: Dict[Tuple[UUID, UUID, UUID, date], cp_model.IntVar],
        staff: List[Staff],
        groups: List[Group],
        shift_templates: List[ShiftTemplate],
        shift_template_requirements: List[ShiftTemplateRequirement],
        start_date: date,
        end_date: date,
    ):
        """Add constraints for shift template requirements"""
        
        total_days = (end_date - start_date).days + 1
        
        # For each day
        for day_offset in range(total_days):
            current_date = start_date + timedelta(days=day_offset)
            day_of_week = current_date.weekday()
            
            # For each shift template requirement
            for str_req in shift_template_requirements:
                # Check if this requirement applies to this day
                if str_req.day_of_week is None or str_req.day_of_week == day_of_week:
                    # Get all assignments for this template/group/date
                    relevant_vars = []
                    for (staff_id, group_id, template_id, date), var in assignments.items():
                        if (group_id == str_req.group_id and 
                            template_id == str_req.shift_template_id and 
                            date == current_date):
                            relevant_vars.append(var)
                    
                    # Add constraint for required count
                    if relevant_vars and str_req.required_count > 0:
                        self.model.Add(sum(relevant_vars) >= str_req.required_count)
                    
                    # Add constraint for preferred count (soft constraint via objective)
                    if relevant_vars and str_req.preferred_count:
                        self.model.Add(sum(relevant_vars) <= str_req.preferred_count)
    
    def _add_group_assignment_preference_constraints(
        self,
        assignments: Dict[Tuple[UUID, UUID, UUID, date], cp_model.IntVar],
        staff: List[Staff],
        groups: List[Group],
    ):
        """Add soft constraints (via objective) for group assignment preferences"""
        
        # This will be handled in the objective function
        # We track which staff are assigned to their primary/secondary groups
        pass
    
    def _add_group_continuity_constraints(
        self,
        assignments: Dict[Tuple[UUID, UUID, UUID, date], cp_model.IntVar],
        staff: List[Staff],
        groups: List[Group],
        start_date: date,
        end_date: date,
    ):
        """Add constraints to encourage staff to work consecutive days in the same group"""
        
        total_days = (end_date - start_date).days + 1
        
        # For each staff member
        for staff_member in staff:
            # For each consecutive pair of days
            for day_offset in range(total_days - 1):
                current_date = start_date + timedelta(days=day_offset)
                next_date = current_date + timedelta(days=1)
                
                # Get all assignments for this staff on both days
                current_day_groups = defaultdict(list)
                next_day_groups = defaultdict(list)
                
                for (staff_id, group_id, template_id, date), var in assignments.items():
                    if staff_id == staff_member.staff_id:
                        if date == current_date:
                            current_day_groups[group_id].append(var)
                        elif date == next_date:
                            next_day_groups[group_id].append(var)
                
                # Soft constraint: if working both days, prefer same group
                # This will be handled in objective function
    
    def _add_staff_hours_constraints(
        self,
        assignments: Dict[Tuple[UUID, UUID, UUID, date], cp_model.IntVar],
        staff: List[Staff],
        shift_templates: List[ShiftTemplate],
        start_date: date,
        end_date: date,
    ):
        """Add constraints for staff working hours limits"""
        
        # Daily hours constraints
        total_days = (end_date - start_date).days + 1
        
        for staff_member in staff:
            # Daily hours
            for day_offset in range(total_days):
                current_date = start_date + timedelta(days=day_offset)
                
                daily_hours = []
                for (staff_id, group_id, template_id, date), var in assignments.items():
                    if staff_id == staff_member.staff_id and date == current_date:
                        template = self._shift_template_cache.get(template_id)
                        if template:
                            # Variable * hours
                            hours_var = self.model.NewIntVar(0, int(template.duration_hours * 10), 
                                                            f"hours_{staff_id}_{date}")
                            self.model.Add(hours_var == var * int(template.duration_hours * 10))
                            daily_hours.append(hours_var)
                
                if daily_hours:
                    max_daily = int((staff_member.max_daily_hours or 12) * 10)
                    self.model.Add(sum(daily_hours) <= max_daily)
            
            # Weekly hours
            for week_start in range(0, total_days, 7):
                week_hours = []
                for day_offset in range(min(7, total_days - week_start)):
                    current_date = start_date + timedelta(days=week_start + day_offset)
                    
                    for (staff_id, group_id, template_id, date), var in assignments.items():
                        if staff_id == staff_member.staff_id and date == current_date:
                            template = self._shift_template_cache.get(template_id)
                            if template:
                                hours_var = self.model.NewIntVar(0, int(template.duration_hours * 10), 
                                                                f"whours_{staff_id}_{date}")
                                self.model.Add(hours_var == var * int(template.duration_hours * 10))
                                week_hours.append(hours_var)
                
                if week_hours:
                    max_weekly = int(staff_member.max_weekly_hours * 10)
                    self.model.Add(sum(week_hours) <= max_weekly)
    
    def _preserve_existing_template_shifts(
        self,
        assignments: Dict[Tuple[UUID, UUID, UUID, date], cp_model.IntVar],
        existing_schedule: List[ScheduledShift],
        start_date: date,
        end_date: date,
    ):
        """Preserve existing scheduled shifts"""
        
        for shift in existing_schedule:
            # Check if shift is in our date range
            if start_date <= shift.date <= end_date:
                # Find matching assignment variable
                if shift.shift_template_id:
                    key = (shift.staff_id, shift.group_id, shift.shift_template_id, shift.date)
                    if key in assignments:
                        # Force this assignment to be true
                        self.model.Add(assignments[key] == 1)
    
    def _set_template_objective(
        self,
        assignments: Dict[Tuple[UUID, UUID, UUID, date], cp_model.IntVar],
        staff: List[Staff],
        groups: List[Group],
        shift_templates: List[ShiftTemplate],
    ):
        """Set objective function with group assignment preferences and continuity"""
        
        objective_terms = []
        
        # 1. Group assignment preferences
        group_preference_weight = 1000  # High weight for primary assignments
        secondary_weight = 500  # Medium weight for secondary assignments
        
        for (staff_id, group_id, template_id, date), var in assignments.items():
            staff_member = next((s for s in staff if s.staff_id == staff_id), None)
            if staff_member:
                # Check group assignment type
                assignment_type = None
                for ga in staff_member.group_assignments:
                    if ga.group_id == group_id:
                        assignment_type = ga.assignment_type
                        priority_weight = ga.priority_weight
                        break
                
                if assignment_type == GroupAssignmentType.PRIMARY:
                    # Reward primary assignments
                    objective_terms.append(var * int(group_preference_weight * priority_weight))
                elif assignment_type == GroupAssignmentType.SECONDARY:
                    # Reward secondary assignments (less than primary)
                    objective_terms.append(var * int(secondary_weight * priority_weight))
                else:
                    # Penalize assignments to non-assigned groups
                    objective_terms.append(var * (-100))
        
        # 2. Group continuity bonus
        continuity_weight = 200
        processed_pairs = set()
        
        for (staff_id1, group_id1, template_id1, date1), var1 in assignments.items():
            next_date = date1 + timedelta(days=1)
            key2 = (staff_id1, group_id1, template_id1, next_date)
            
            if key2 in assignments and (var1, assignments[key2]) not in processed_pairs:
                # Bonus for consecutive days in same group
                both_days = self.model.NewBoolVar(f"consec_{staff_id1}_{group_id1}_{date1}")
                self.model.Add(both_days == 1).OnlyEnforceIf([var1, assignments[key2]])
                self.model.Add(both_days == 0).OnlyEnforceIf(var1.Not())
                self.model.Add(both_days == 0).OnlyEnforceIf(assignments[key2].Not())
                
                objective_terms.append(both_days * continuity_weight)
                processed_pairs.add((var1, assignments[key2]))
        
        # 3. Minimize total cost (if applicable)
        if OptimizationGoal.MINIMIZE_COST in self.config.goals:
            for (staff_id, group_id, template_id, date), var in assignments.items():
                staff_member = next((s for s in staff if s.staff_id == staff_id), None)
                template = self._shift_template_cache.get(template_id)
                
                if staff_member and template and staff_member.hourly_rate:
                    cost = int(staff_member.hourly_rate * template.duration_hours * 10)
                    objective_terms.append(var * (-cost))  # Negative because we maximize
        
        # Maximize the objective
        self.model.Maximize(sum(objective_terms))
    
    def _extract_template_schedule(
        self,
        assignments: Dict[Tuple[UUID, UUID, UUID, date], cp_model.IntVar],
        staff: List[Staff],
        groups: List[Group],
        shift_templates: List[ShiftTemplate],
        start_date: date,
    ) -> List[ScheduledShift]:
        """Extract scheduled shifts from solved model"""
        
        schedule = []
        
        for (staff_id, group_id, template_id, shift_date), var in assignments.items():
            if self.solver.Value(var) == 1:
                template = self._shift_template_cache.get(template_id)
                if template:
                    # Determine assignment type
                    staff_member = next((s for s in staff if s.staff_id == staff_id), None)
                    assignment_type = None
                    if staff_member:
                        for ga in staff_member.group_assignments:
                            if ga.group_id == group_id:
                                assignment_type = ga.assignment_type
                                break
                    
                    shift = ScheduledShift(
                        staff_id=staff_id,
                        group_id=group_id,
                        shift_template_id=template_id,
                        date=shift_date,
                        start_time=template.start_time,
                        end_time=template.end_time,
                        scheduled_hours=template.duration_hours,
                        status=ShiftStatus.SCHEDULED,
                        assignment_type=assignment_type,
                    )
                    schedule.append(shift)
        
        # Sort by date and start time
        schedule.sort(key=lambda s: (s.date, s.start_time))
        return schedule
    
    def _check_staff_availability(
        self,
        staff_member: Staff,
        check_date: date,
        start_time: datetime_time,
        end_time: datetime_time,
    ) -> bool:
        """Check if staff is available for a shift"""
        
        day_of_week = check_date.weekday()
        
        # Check absences
        for absence in staff_member.absences:
            if absence.start_date <= check_date <= absence.end_date:
                if absence.is_full_day:
                    return False
                # TODO: Handle partial day absences
        
        # Check availability
        for avail in staff_member.availability:
            if avail.day_of_week == day_of_week and avail.is_available:
                # Check if shift time overlaps with availability
                if (avail.start_time <= start_time and 
                    avail.end_time >= end_time):
                    return True
        
        return False
    
    def _detect_schedule_conflicts(
        self,
        schedule: List[ScheduledShift],
        staff: List[Staff],
        groups: List[Group],
        requirements: List[StaffingRequirement],
    ) -> List[ScheduleConflict]:
        """Detect any conflicts in the generated schedule"""
        
        conflicts = []
        
        # Check for double-booking
        staff_schedule_map = defaultdict(list)
        for shift in schedule:
            staff_schedule_map[shift.staff_id].append(shift)
        
        for staff_id, shifts in staff_schedule_map.items():
            # Sort by date and time
            shifts.sort(key=lambda s: (s.date, s.start_time))
            
            # Check for overlaps
            for i in range(len(shifts) - 1):
                current = shifts[i]
                next_shift = shifts[i + 1]
                
                if current.date == next_shift.date:
                    current_end = datetime.combine(current.date, current.end_time)
                    next_start = datetime.combine(next_shift.date, next_shift.start_time)
                    
                    if current_end > next_start:
                        conflicts.append(ScheduleConflict(
                            conflict_type="double_booking",
                            severity="error",
                            staff_id=staff_id,
                            description=f"Staff double-booked on {current.date}",
                            suggested_solutions=["Adjust shift times", "Assign different staff"],
                        ))
        
        return conflicts
    
    def _analyze_template_infeasibility(
        self,
        staff: List[Staff],
        groups: List[Group],
        shift_templates: List[ShiftTemplate],
        shift_template_requirements: List[ShiftTemplateRequirement],
    ) -> List[ScheduleConflict]:
        """Analyze why the problem is infeasible"""
        
        conflicts = []
        
        # Check if there are enough qualified staff for requirements
        for str_req in shift_template_requirements:
            template = self._shift_template_cache.get(str_req.shift_template_id)
            if not template:
                continue
            
            # Count qualified staff
            qualified_count = 0
            for staff_member in staff:
                staff_quals = {
                    q.qualification_name for q in staff_member.qualifications
                    if q.is_verified
                }
                if all(rq in staff_quals for rq in template.required_qualifications):
                    qualified_count += 1
            
            if qualified_count < str_req.required_count:
                conflicts.append(ScheduleConflict(
                    conflict_type="insufficient_qualified_staff",
                    severity="error",
                    group_id=str_req.group_id,
                    description=f"Not enough qualified staff for {template.name}",
                    suggested_solutions=[
                        f"Need {str_req.required_count} staff, only {qualified_count} qualified",
                        "Reduce requirements or add qualified staff",
                    ],
                ))
        
        return conflicts
    
    def _solve_with_group_preferences(
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
        Solve scheduling problem without templates but with group assignment preferences
        """
        method_start_time = _time_module.time()
        
        try:
            import traceback
            # Validate date range
            if schedule_end_date < schedule_start_date:
                raise ValueError("End date must be after start date")
            total_days = (schedule_end_date - schedule_start_date).days + 1
            if total_days > 365:
                raise ValueError(f"Date range too large: {total_days} days (max 365)")
                
            logger.info(f"SOLVER SETUP (Group Preferences): {len(staff)} staff, {len(groups)} groups")
            logger.info(f"DATE RANGE: {schedule_start_date} to {schedule_end_date} ({total_days} days)")
            
            # Create assignment variables for ALL possible assignments (including existing ones)
            # This allows the solver to see the complete picture and make optimal decisions
            assignments = {}
            existing_assignments = set()  # Track existing assignments
            
            # First, identify all existing assignments
            if existing_schedule:
                for existing_shift in existing_schedule:
                    existing_key = (
                        existing_shift.staff_id,
                        existing_shift.group_id, 
                        existing_shift.date,
                        existing_shift.start_time,
                        existing_shift.end_time
                    )
                    existing_assignments.add(existing_key)
                    logger.debug(f"Existing assignment: {existing_shift.staff_id} -> {existing_shift.group_id} on {existing_shift.date} {existing_shift.start_time}-{existing_shift.end_time}")
            
            logger.info(f"Found {len(existing_assignments)} existing assignments to preserve as fixed constraints")
            
            # Don't create variables for existing schedules - they will be handled via hours accounting
            # This avoids the complex logic of trying to preserve existing assignments as fixed variables
            # which was causing the INFEASIBLE issues due to conflicts with new requirements
            
            # Create variables for all requirements (don't segment - let solver handle optimization)
            for req in requirements:
                for day_offset in range(total_days):
                    current_date = schedule_start_date + timedelta(days=day_offset)
                    day_of_week = current_date.weekday()
                    
                    # Check if this requirement applies to this day
                    if req.time_slot.day_of_week != day_of_week:
                        continue
                    
                    # Get the group for this requirement
                    group = next((g for g in groups if g.group_id == req.group_id), None)
                    if not group:
                        continue
                    
                    # For each staff member, create assignment variables
                    for staff_member in staff:
                        
                        # Check availability for this time slot
                        is_available = self._check_staff_availability_for_requirement(
                            staff_member, current_date, req.time_slot
                        )
                        if not is_available:
                            continue
                        
                        # Check qualifications
                        staff_quals = {
                            q.qualification_name for q in staff_member.qualifications
                            if q.is_verified
                        }
                        if not all(rq in staff_quals for rq in req.required_qualifications):
                            continue
                        
                        # Don't filter out variables based on existing schedules here
                        # Let the constraints and hours accounting handle the conflicts
                        # This ensures we have enough variables to satisfy minimum staffing requirements
                        
                        # Create variable (avoid duplicates)
                        assignment_key = (staff_member.staff_id, req.group_id, current_date, req.time_slot.start_time, req.time_slot.end_time)
                        if assignment_key not in assignments:
                            var_name = f"assign_{staff_member.staff_id}_{group.group_id}_{current_date}_{req.time_slot.start_time}_{req.time_slot.end_time}"
                            var = self.model.NewBoolVar(var_name)
                            assignments[assignment_key] = var
            
            logger.info(f"Created {len(assignments)} assignment variables")
            
            # Debug: Log first few assignment variables to see what's being created
            if assignments:
                logger.info("Sample assignment variables created:")
                for i, (key, var) in enumerate(list(assignments.items())[:5]):
                    staff_id, group_id, date, start_time, end_time = key
                    logger.info(f"  {i+1}: Staff {staff_id} -> Group {group_id} on {date} {start_time}-{end_time}")
            else:
                logger.warning("No assignment variables were created - this will cause INFEASIBLE")
            
            if not assignments:
                logger.warning("No valid assignments possible")
                return [], OptimizationResult(
                    objective_value=0,
                    solve_time_seconds=_time_module.time() - method_start_time,
                    status="INFEASIBLE",
                    iterations=0,
                    conflicts_resolved=0,
                ), []
            
            # Use Option 2: Hours Accounting Only (more flexible than fixed constraints)
            # This approach accounts for existing schedules in hours constraints without forcing
            # specific assignments, allowing the solver more flexibility to find feasible solutions
            logger.info(f"Using hours accounting approach for {len(existing_assignments)} existing assignments")
            logger.info("Existing schedules are accounted for in staff hours constraints, not as fixed assignments")
            
            # Add mutual exclusion constraints to prevent overlapping assignments for same staff/date
            logger.debug("Adding overlap prevention constraints")
            self._add_overlap_prevention_constraints(assignments, schedule_start_date, total_days)
            
            # Add staffing constraints
            logger.debug("Adding staffing constraints")
            self._add_staffing_constraints_for_time_slots(assignments, requirements, schedule_start_date, schedule_end_date, existing_schedule)
            
            # Add staff working hours constraints (including existing schedule hours)
            logger.debug("Adding hours constraints")
            self._add_staff_hours_constraints_for_time_slots(assignments, staff, schedule_start_date, schedule_end_date, existing_schedule)
            
            # Add group assignment preference constraints
            logger.debug("Adding group preference objectives")
            self._add_group_preference_objective_for_time_slots(assignments, staff, groups)
            
            # Solve
            logger.info(f"SOLVING: Starting solver with {len(assignments)} variables")
            status = self.solver.Solve(self.model)
            solve_time = _time_module.time() - method_start_time
            logger.info(f"SOLVING: Solver finished with status: {status} in {solve_time:.2f}s")
            
            # Process results
            if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
                # Extract all schedules (including existing ones that were fixed)
                all_schedules = self._extract_time_slot_schedule(assignments, staff, groups, schedule_start_date)
                
                # Filter to return only NEW schedules (not duplicating existing ones)
                new_schedules = self._filter_new_schedules_only(all_schedules, existing_schedule)
                
                conflicts = []  # Add conflict detection if needed
                optimization_result = OptimizationResult(
                    objective_value=self.solver.ObjectiveValue(),
                    solve_time_seconds=solve_time,
                    status="OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE",
                    iterations=self.solver.NumBranches(),
                    conflicts_resolved=len(conflicts),
                )
                return new_schedules, optimization_result, conflicts
            else:
                # Handle infeasible or timeout
                status_name = "INFEASIBLE" if status == cp_model.INFEASIBLE else "TIMEOUT"
                optimization_result = OptimizationResult(
                    objective_value=0,
                    solve_time_seconds=solve_time,
                    status=status_name,
                    iterations=self.solver.NumBranches(),
                    conflicts_resolved=0,
                )
                return [], optimization_result, []
            
        except Exception as e:
            logger.error(f"FULL TRACEBACK in _solve_with_group_preferences:")
            logger.error(traceback.format_exc())
            optimization_result = OptimizationResult(
                objective_value=0,
                solve_time_seconds=_time_module.time() - method_start_time,
                status="ERROR",
                iterations=0,
                conflicts_resolved=0,
            )
            conflicts = [
                ScheduleConflict(
                    conflict_type="solver_error",
                    severity="error",
                    description=f"Solver error: {str(e)}",
                    suggested_solutions=["Check input data", "Reduce problem size"],
                )
            ]
            return [], optimization_result, conflicts
            
        except Exception as e:
            logger.error(f"Error in group preference solver: {str(e)}")
            # If we already have a solution status from the solver, return that instead of ERROR
            if hasattr(self, 'solver') and hasattr(self.solver, 'StatusName'):
                try:
                    solver_status = self.solver.StatusName()
                    if solver_status in ['OPTIMAL', 'FEASIBLE']:
                        logger.warning(f"Post-processing error occurred but solver found {solver_status} solution")
                        # Return empty result to indicate post-processing failed but solving succeeded
                        optimization_result = OptimizationResult(
                            objective_value=0,
                            solve_time_seconds=_time_module.time() - method_start_time,
                            status=solver_status,
                            iterations=0,
                            conflicts_resolved=0,
                        )
                        return [], optimization_result, []
                except:
                    pass
            
            optimization_result = OptimizationResult(
                objective_value=0,
                solve_time_seconds=_time_module.time() - method_start_time,
                status="ERROR",
                iterations=0,
                conflicts_resolved=0,
            )
            conflicts = [
                ScheduleConflict(
                    conflict_type="solver_error",
                    severity="error",
                    description=f"Solver error: {str(e)}",
                    suggested_solutions=["Check input data", "Reduce problem size"],
                )
            ]
            return [], optimization_result, conflicts
    
    def _check_staff_availability_for_requirement(
        self, staff_member: Staff, date: date, time_slot
    ) -> bool:
        """Check if staff is available for a specific time slot"""
        day_of_week = date.weekday()
        
        # Check availability
        for availability in staff_member.availability:
            if (availability.day_of_week == day_of_week and 
                availability.is_available):
                
                # Convert time types for comparison if needed
                avail_start = availability.start_time
                avail_end = availability.end_time
                slot_start = time_slot.start_time
                slot_end = time_slot.end_time
                
                # Handle string time formats and normalize all types
                try:
                    # Normalize start time
                    if isinstance(slot_start, str):
                        from datetime import time as datetime_time
                        hour, minute = map(int, slot_start.split(':')[:2])
                        slot_start = datetime_time(hour, minute)
                    elif isinstance(slot_start, (int, float)):
                        from datetime import time as datetime_time
                        hours = int(slot_start)
                        minutes = int((slot_start - hours) * 60)
                        slot_start = datetime_time(hours, minutes)
                    
                    # Normalize end time
                    if isinstance(slot_end, str):
                        from datetime import time as datetime_time
                        hour, minute = map(int, slot_end.split(':')[:2])
                        slot_end = datetime_time(hour, minute)
                    elif isinstance(slot_end, (int, float)):
                        from datetime import time as datetime_time
                        hours = int(slot_end)
                        minutes = int((slot_end - hours) * 60)
                        slot_end = datetime_time(hours, minutes)
                    
                    if avail_start <= slot_start and avail_end >= slot_end:
                        return True
                except Exception as e:
                    logger.error(f"Error in availability check: {e}")
                    logger.error(f"Types: avail_start={type(avail_start)}, slot_start={type(slot_start)}, avail_end={type(avail_end)}, slot_end={type(slot_end)}")
                    logger.error(f"Values: avail_start={avail_start}, slot_start={slot_start}, avail_end={avail_end}, slot_end={slot_end}")
                    return False
        
        return False
    
    def _add_staffing_constraints_for_time_slots(
        self, assignments, requirements, start_date, end_date, existing_schedule=None
    ):
        """Add constraints to ensure staffing requirements are met"""
        total_days = (end_date - start_date).days + 1
        
        for req in requirements:
            for day_offset in range(total_days):
                current_date = start_date + timedelta(days=day_offset)
                day_of_week = current_date.weekday()
                
                if req.time_slot.day_of_week != day_of_week:
                    continue
                
                # Get all assignments for this requirement
                relevant_vars = []
                for (staff_id, group_id, date, start_time, end_time), var in assignments.items():
                    if (group_id == req.group_id and 
                        date == current_date and
                        start_time == req.time_slot.start_time and
                        end_time == req.time_slot.end_time):
                        relevant_vars.append(var)
                
                # Count existing assignments that satisfy this requirement (only exact matches)
                # This ensures we only count existing assignments that exactly match the requirement
                # to avoid over-counting and creating infeasible constraints
                existing_count = 0
                if existing_schedule:
                    for existing_shift in existing_schedule:
                        if (existing_shift.group_id == req.group_id and 
                            existing_shift.date == current_date):
                            # Only count exact time matches to avoid constraint conflicts
                            def time_to_minutes(time_val):
                                if isinstance(time_val, str):
                                    h, m = map(int, time_val.split(':')[:2])
                                    return h * 60 + m
                                elif hasattr(time_val, 'hour') and hasattr(time_val, 'minute'):
                                    return time_val.hour * 60 + time_val.minute
                                return 0
                            
                            existing_start = time_to_minutes(existing_shift.start_time)
                            existing_end = time_to_minutes(existing_shift.end_time)
                            req_start = time_to_minutes(req.time_slot.start_time)
                            req_end = time_to_minutes(req.time_slot.end_time)
                            
                            # Count if times overlap (not just exact match) since existing shift can cover requirement
                            overlaps = (req_start < existing_end and existing_start < req_end)
                            if overlaps:
                                existing_count += 1
                                logger.debug(f"Existing shift covers requirement: {existing_shift.staff_id} covers {req.group_id} on {current_date} {req_start//60:02d}:{req_start%60:02d}-{req_end//60:02d}:{req_end%60:02d}")
                            else:
                                logger.debug(f"Existing shift different times: {existing_shift.staff_id} has {existing_start//60:02d}:{existing_start%60:02d}-{existing_end//60:02d}:{existing_end%60:02d}, req wants {req_start//60:02d}:{req_start%60:02d}-{req_end//60:02d}:{req_end%60:02d}")
                
                # Adjust requirements based on existing assignments
                remaining_min = max(0, req.min_staff_count - existing_count)
                remaining_max = max(0, req.max_staff_count - existing_count)
                
                if relevant_vars and remaining_min > 0:
                    # Minimum staff constraint (only if more staff needed)
                    self.model.Add(sum(relevant_vars) >= remaining_min)
                if relevant_vars and remaining_max > 0:
                    # Maximum staff constraint
                    self.model.Add(sum(relevant_vars) <= remaining_max)
                elif existing_count >= req.min_staff_count:
                    # Requirement is already satisfied by existing schedules - prevent new assignments
                    if relevant_vars:
                        self.model.Add(sum(relevant_vars) == 0)
                    logger.debug(f"Requirement already satisfied by existing: {req.group_id} on {current_date} ({existing_count}/{req.min_staff_count}) - blocked new assignments")
    
    def _add_staff_hours_constraints_for_time_slots(
        self, assignments, staff, start_date, end_date, existing_schedule=None
    ):
        """Add constraints for staff working hours, including existing schedule hours"""
        total_days = (end_date - start_date).days + 1
        
        # Calculate existing hours per staff per day for accurate constraint calculation
        existing_daily_hours = defaultdict(lambda: defaultdict(int))  # staff_id -> date -> minutes
        existing_total_hours = defaultdict(int)  # staff_id -> total minutes
        
        if existing_schedule:
            for existing_shift in existing_schedule:
                staff_id = existing_shift.staff_id
                date = existing_shift.date
                hours = self._calculate_hours_from_times(existing_shift.start_time, existing_shift.end_time)
                minutes = int(hours * 60)
                existing_daily_hours[staff_id][date] += minutes
                existing_total_hours[staff_id] += minutes
                logger.debug(f"Existing hours: {staff_id} on {date} = {hours:.2f}h ({minutes}min)")
        
        for staff_member in staff:
            # Daily hours constraint (including existing hours)
            for day_offset in range(total_days):
                current_date = start_date + timedelta(days=day_offset)
                daily_vars = []
                
                # Add variables for new assignments on this date
                for (staff_id, group_id, date, start_time, end_time), var in assignments.items():
                    if staff_id == staff_member.staff_id and date == current_date:
                        # Calculate hours for this shift (convert to integer minutes)
                        hours = self._calculate_hours_from_times(start_time, end_time)
                        hours_int = int(hours * 60)  # Convert to minutes
                        daily_vars.append(var * hours_int)
                
                if daily_vars:
                    max_daily_minutes = int(staff_member.max_daily_hours * 60)
                    # Make daily constraint more lenient - allow up to 12 hours per day
                    max_daily_minutes = max(max_daily_minutes, 12 * 60)
                    
                    # Account for existing hours on this day
                    existing_minutes_today = existing_daily_hours[staff_member.staff_id][current_date]
                    available_minutes = max_daily_minutes - existing_minutes_today
                    
                    if available_minutes > 0:
                        self.model.Add(sum(daily_vars) <= available_minutes)
                        logger.debug(f"Daily constraint for {staff_member.staff_id} on {current_date}: "
                                   f"max {available_minutes}min (existing: {existing_minutes_today}min)")
                    else:
                        # Staff already at or over daily limit - don't add blocking constraint
                        # Let the solver handle this through other constraints
                        logger.debug(f"Daily constraint for {staff_member.staff_id} on {current_date}: "
                                   f"at capacity (existing: {existing_minutes_today}min >= {max_daily_minutes}min limit)")
            
            # Weekly/period hours constraint (including existing hours)
            all_vars = []
            for (staff_id, group_id, date, start_time, end_time), var in assignments.items():
                if staff_id == staff_member.staff_id:
                    hours = self._calculate_hours_from_times(start_time, end_time)
                    hours_int = int(hours * 60)  # Convert to minutes
                    all_vars.append(var * hours_int)
            
            if all_vars:
                max_period_minutes = int(min(staff_member.max_weekly_hours, total_days * staff_member.max_daily_hours) * 60)
                # Make weekly constraint more lenient - allow up to 50 hours per week
                max_period_minutes = max(max_period_minutes, 50 * 60)
                
                # Account for existing hours in the period
                existing_minutes_total = existing_total_hours[staff_member.staff_id]
                available_period_minutes = max_period_minutes - existing_minutes_total
                
                if available_period_minutes > 0:
                    self.model.Add(sum(all_vars) <= available_period_minutes)
                    logger.debug(f"Period constraint for {staff_member.staff_id}: "
                               f"max {available_period_minutes}min (existing: {existing_minutes_total}min)")
                else:
                    # Staff already at or over period limit - don't add blocking constraint
                    # Let the solver handle this through other constraints
                    logger.debug(f"Period constraint for {staff_member.staff_id}: "
                               f"at capacity (existing: {existing_minutes_total}min >= {max_period_minutes}min limit)")
    
    def _calculate_hours_from_times(self, start_time, end_time):
        """Calculate hours between two times"""
        from datetime import time as datetime_time
        
        try:
            # Convert all inputs to datetime.time objects first
            def normalize_time(time_input):
                if time_input is None:
                    raise ValueError("Time input cannot be None")
                
                # If already datetime.time, return as-is
                if isinstance(time_input, datetime_time):
                    return time_input
                    
                # If string, parse it
                if isinstance(time_input, str):
                    time_parts = time_input.split(':')
                    hour = int(time_parts[0])
                    minute = int(time_parts[1]) if len(time_parts) > 1 else 0
                    second = int(time_parts[2]) if len(time_parts) > 2 else 0
                    return datetime_time(hour, minute, second)
                
                # If float (representing hours), convert to time
                if isinstance(time_input, (int, float)):
                    total_seconds = int(time_input * 3600)
                    hours = total_seconds // 3600
                    minutes = (total_seconds % 3600) // 60
                    seconds = total_seconds % 60
                    return datetime_time(hours % 24, minutes, seconds)
                
                # Try to get hour/minute attributes (in case it's a different time-like object)
                if hasattr(time_input, 'hour') and hasattr(time_input, 'minute'):
                    return datetime_time(time_input.hour, time_input.minute, 
                                       getattr(time_input, 'second', 0))
                
                raise ValueError(f"Cannot convert {type(time_input)} to time: {time_input}")
            
            # Normalize both times
            start_time_obj = normalize_time(start_time)
            end_time_obj = normalize_time(end_time)
            
            # Convert to hours (float)
            start_hour = start_time_obj.hour + start_time_obj.minute / 60.0 + start_time_obj.second / 3600.0
            end_hour = end_time_obj.hour + end_time_obj.minute / 60.0 + end_time_obj.second / 3600.0
            
            # Calculate duration
            if end_hour <= start_hour:  # Next day
                return (24 - start_hour) + end_hour
            else:
                return end_hour - start_hour
                
        except Exception as e:
            logger.error(f"Error calculating hours from {start_time} ({type(start_time)}) to {end_time} ({type(end_time)}): {e}")
            # Return a default duration to prevent crashes
            return 8.0  # Default 8-hour shift
    
    def _add_group_preference_objective_for_time_slots(
        self, assignments, staff, groups
    ):
        """Add objective terms for group assignment preferences"""
        objective_terms = []
        
        for (staff_id, group_id, date, start_time, end_time), var in assignments.items():
            # Find staff member
            staff_member = next((s for s in staff if s.staff_id == staff_id), None)
            if not staff_member:
                continue
            
            # Get group assignment for this staff-group combination
            assignment_type = None
            priority_weight = 1.0
            
            for assignment in staff_member.group_assignments:
                if assignment.group_id == group_id:
                    assignment_type = assignment.assignment_type
                    priority_weight = assignment.priority_weight
                    break
            
            # Calculate preference weight
            if assignment_type == GroupAssignmentType.PRIMARY:
                # High weight for primary assignments
                objective_terms.append(var * int(2000 * priority_weight))
            elif assignment_type == GroupAssignmentType.SECONDARY:
                # Medium weight for secondary assignments
                objective_terms.append(var * int(500 * priority_weight))
            else:
                # Small penalty for unassigned groups (not -100 which causes infeasibility)
                # Allow assignments to unassigned groups but with lower priority
                objective_terms.append(var * 10)
        
        if objective_terms:
            self.model.Maximize(sum(objective_terms))
            logger.info(f"Added {len(objective_terms)} objective terms for group preferences")
    
    def _extract_time_slot_schedule(
        self, assignments, staff, groups, start_date
    ) -> List[ScheduledShift]:
        """Extract schedule from time slot assignments"""
        schedule = []
        
        for (staff_id, group_id, date, start_time, end_time), var in assignments.items():
            if self.solver.Value(var) == 1:
                # Find staff member to get assignment type
                staff_member = next((s for s in staff if s.staff_id == staff_id), None)
                assignment_type = None
                
                if staff_member:
                    for assignment in staff_member.group_assignments:
                        if assignment.group_id == group_id:
                            assignment_type = assignment.assignment_type
                            break
                
                # Calculate hours
                logger.debug(f"Calculating hours for extracted shift: {start_time} ({type(start_time)}) to {end_time} ({type(end_time)})")
                try:
                    hours = self._calculate_hours_from_times(start_time, end_time)
                except Exception as e:
                    logger.error(f"Error calculating hours in result extraction: {e}")
                    logger.error(f"start_time: {start_time} ({type(start_time)}), end_time: {end_time} ({type(end_time)})")
                    raise
                
                shift = ScheduledShift(
                    staff_id=staff_id,
                    group_id=group_id,
                    date=date,
                    start_time=start_time,
                    end_time=end_time,
                    scheduled_hours=hours,
                    assignment_type=assignment_type,
                    status=ShiftStatus.SCHEDULED,
                )
                schedule.append(shift)
        
        logger.info(f"Extracted {len(schedule)} scheduled shifts")
        return schedule

    def _segment_requirements_around_existing_schedules(
        self, 
        requirements: List[StaffingRequirement], 
        existing_schedule: Optional[List[ScheduledShift]], 
        schedule_start_date: date, 
        total_days: int
    ) -> List[StaffingRequirement]:
        """
        Segment requirements into non-overlapping time blocks to allow gap filling around existing schedules
        """
        if not existing_schedule:
            return requirements
        
        segmented_requirements = []
        
        for req in requirements:
            # Process this requirement for each applicable day
            requirement_processed = False
            
            for day_offset in range(total_days):
                current_date = schedule_start_date + timedelta(days=day_offset)
                day_of_week = current_date.weekday()
                
                # Check if this requirement applies to this day
                if req.time_slot.day_of_week != day_of_week:
                    continue
                
                requirement_processed = True
                
                # Find existing schedules for this group and date
                existing_shifts_for_date = [
                    shift for shift in existing_schedule 
                    if shift.date == current_date and shift.group_id == req.group_id
                ]
                
                # If no existing schedules for this group/date, keep the original requirement
                if not existing_shifts_for_date:
                    segmented_requirements.append(req)
                    continue
                
                # Check if the requirement is already fully satisfied
                req_start = req.time_slot.start_time
                req_end = req.time_slot.end_time
                
                # Normalize requirement times to datetime.time
                from datetime import time as datetime_time
                if isinstance(req_start, str):
                    hour, minute = map(int, req_start.split(':')[:2])
                    req_start = datetime_time(hour, minute)
                if isinstance(req_end, str):
                    hour, minute = map(int, req_end.split(':')[:2])
                    req_end = datetime_time(hour, minute)
                
                # Sort existing shifts by start time (normalize them too)
                existing_ranges = []
                for shift in existing_shifts_for_date:
                    shift_start = shift.start_time
                    shift_end = shift.end_time
                    
                    # Normalize shift times
                    if isinstance(shift_start, str):
                        hour, minute = map(int, shift_start.split(':')[:2])
                        shift_start = datetime_time(hour, minute)
                    if isinstance(shift_end, str):
                        hour, minute = map(int, shift_end.split(':')[:2])
                        shift_end = datetime_time(hour, minute)
                    
                    existing_ranges.append((shift_start, shift_end))
                
                existing_ranges.sort()
                
                # Check if existing schedules fully cover the requirement time range
                total_coverage = []
                for start_time, end_time in existing_ranges:
                    try:
                        # Only consider overlapping portions
                        overlap_start = max(req_start, start_time)
                        overlap_end = min(req_end, end_time)
                        if overlap_start < overlap_end:
                            total_coverage.append((overlap_start, overlap_end))
                    except Exception as e:
                        logger.error(f"Error in time comparison: {e}")
                        logger.error(f"req_start={req_start} ({type(req_start)}), start_time={start_time} ({type(start_time)})")
                        continue
                
                # Merge overlapping coverage ranges
                if total_coverage:
                    total_coverage.sort()
                    merged_coverage = [total_coverage[0]]
                    for current_start, current_end in total_coverage[1:]:
                        last_start, last_end = merged_coverage[-1]
                        if current_start <= last_end:
                            # Merge overlapping ranges
                            merged_coverage[-1] = (last_start, max(last_end, current_end))
                        else:
                            merged_coverage.append((current_start, current_end))
                    
                    # Check if merged coverage fully covers the requirement
                    if (len(merged_coverage) == 1 and 
                        merged_coverage[0][0] <= req_start and 
                        merged_coverage[0][1] >= req_end):
                        # Requirement is fully covered, skip it
                        logger.debug(f"Requirement fully covered: {req.group_id} on day {day_of_week} {req_start}-{req_end}")
                        continue
                
                # Create time segments around existing schedules
                # (req_start and req_end already defined above)
                # (existing_ranges already defined and sorted above)
                
                # Find gaps between requirement time and existing schedules
                current_time = req_start
                
                for existing_start, existing_end in existing_ranges:
                    # If there's a gap before this existing schedule
                    if current_time < existing_start:
                        gap_end = min(existing_start, req_end)
                        if current_time < gap_end:
                            # Create a segmented requirement for this gap
                            gap_req = StaffingRequirement(
                                group_id=req.group_id,
                                time_slot=TimeSlot(
                                    start_time=current_time,
                                    end_time=gap_end,
                                    day_of_week=req.time_slot.day_of_week
                                ),
                                min_staff_count=req.min_staff_count,
                                max_staff_count=req.max_staff_count,
                                required_qualifications=req.required_qualifications
                            )
                            segmented_requirements.append(gap_req)
                            logger.debug(f"Created gap segment: {req.group_id} on day {day_of_week} {current_time}-{gap_end}")
                    
                    # Move past the existing schedule
                    current_time = max(current_time, existing_end)
                    
                    # If we've covered the entire requirement time, stop
                    if current_time >= req_end:
                        break
                
                # Check if there's a gap after all existing schedules
                if current_time < req_end:
                    gap_req = StaffingRequirement(
                        group_id=req.group_id,
                        time_slot=TimeSlot(
                            start_time=current_time,
                            end_time=req_end,
                            day_of_week=req.time_slot.day_of_week
                        ),
                        min_staff_count=req.min_staff_count,
                        max_staff_count=req.max_staff_count,
                        required_qualifications=req.required_qualifications
                    )
                    segmented_requirements.append(gap_req)
                    logger.debug(f"Created final gap segment: {req.group_id} on day {day_of_week} {current_time}-{req_end}")
                
                # Exit the day loop since we've processed this requirement for its applicable day
                break
            
            # If the requirement had no applicable days in the date range, keep it as-is
            if not requirement_processed:
                segmented_requirements.append(req)
        
        return segmented_requirements

    def _add_overlap_prevention_constraints(
        self, 
        assignments: Dict[Tuple, any], 
        schedule_start_date: date, 
        total_days: int
    ):
        """
        Add constraints to prevent staff from being assigned to overlapping time slots on the same day.
        This is crucial for the hybrid model where we create variables for all possible assignments.
        """
        constraint_count = 0
        
        # Group assignments by staff and date
        staff_date_assignments = defaultdict(list)
        for (staff_id, group_id, assignment_date, start_time, end_time), var in assignments.items():
            staff_date_assignments[(staff_id, assignment_date)].append({
                'var': var,
                'start_time': start_time,
                'end_time': end_time,
                'group_id': group_id
            })
        
        # For each staff member on each date, ensure no overlapping assignments
        for (staff_id, assignment_date), date_assignments in staff_date_assignments.items():
            # Check all pairs of assignments for this staff/date combination
            for i, assignment1 in enumerate(date_assignments):
                for assignment2 in date_assignments[i + 1:]:
                    # Check if time ranges overlap
                    start1, end1 = assignment1['start_time'], assignment1['end_time']
                    start2, end2 = assignment2['start_time'], assignment2['end_time']
                    
                    # Normalize all times to datetime.time for comparison
                    from datetime import time as datetime_time
                    try:
                        def normalize_time_for_comparison(time_val):
                            if isinstance(time_val, str):
                                hour, minute = map(int, time_val.split(':')[:2])
                                return datetime_time(hour, minute)
                            elif isinstance(time_val, (int, float)):
                                hours = int(time_val)
                                minutes = int((time_val - hours) * 60)
                                return datetime_time(hours % 24, minutes)
                            elif isinstance(time_val, datetime_time):
                                return time_val
                            else:
                                logger.error(f"Unknown time type: {type(time_val)} = {time_val}")
                                return datetime_time(0, 0)
                        
                        start1_norm = normalize_time_for_comparison(start1)
                        end1_norm = normalize_time_for_comparison(end1)
                        start2_norm = normalize_time_for_comparison(start2)
                        end2_norm = normalize_time_for_comparison(end2)
                        
                        # Check if times are identical (same assignment)
                        identical = (start1_norm == start2_norm and end1_norm == end2_norm)
                        
                        # Times overlap if start1 < end2 and start2 < end1
                        # But allow adjacent shifts (e.g., 08:00-09:00 and 09:00-17:00)
                        overlaps = (start1_norm < end2_norm and start2_norm < end1_norm)
                        
                        # Check if they're adjacent (touching but not overlapping)
                        adjacent = (start1_norm == end2_norm or start2_norm == end1_norm)
                        
                    except Exception as e:
                        logger.error(f"Error in overlap detection: {e}")
                        logger.error(f"Times: start1={start1} ({type(start1)}), end1={end1} ({type(end1)}), start2={start2} ({type(start2)}), end2={end2} ({type(end2)})")
                        # Default to non-overlapping to avoid crashes
                        overlaps = False
                        adjacent = False
                        identical = False
                    
                    if identical:
                        # Don't add constraints for identical time slots - they represent the same assignment option
                        logger.debug(f"Skipping identical time slots for {staff_id} on {assignment_date}: "
                                   f"{start1}-{end1} same as {start2}-{end2}")
                    elif overlaps and not adjacent:
                        # Add constraint: at most one of these assignments can be selected
                        self.model.Add(assignment1['var'] + assignment2['var'] <= 1)
                        constraint_count += 1
                        logger.debug(f"Added overlap constraint for {staff_id} on {assignment_date}: "
                                   f"{start1}-{end1} vs {start2}-{end2}")
                    elif adjacent:
                        logger.debug(f"Allowing adjacent shifts for {staff_id} on {assignment_date}: "
                                   f"{start1}-{end1} adjacent to {start2}-{end2}")
        
        logger.info(f"Added {constraint_count} overlap prevention constraints")

    def _filter_new_schedules_only(
        self, 
        all_schedules: List[ScheduledShift], 
        existing_schedule: Optional[List[ScheduledShift]]
    ) -> List[ScheduledShift]:
        """
        Filter the generated schedules to return only NEW schedules that don't duplicate existing ones.
        This ensures we return only the incremental schedules that need to be created.
        """
        if not existing_schedule:
            return all_schedules
        
        # Create set of existing schedule keys for fast lookup
        existing_keys = set()
        for existing in existing_schedule:
            key = (
                existing.staff_id,
                existing.group_id,
                existing.date,
                existing.start_time,
                existing.end_time
            )
            existing_keys.add(key)
        
        # Filter out schedules that match existing ones
        new_schedules = []
        for schedule in all_schedules:
            key = (
                schedule.staff_id,
                schedule.group_id,
                schedule.date,
                schedule.start_time,
                schedule.end_time
            )
            if key not in existing_keys:
                new_schedules.append(schedule)
        
        logger.info(f"Filtered {len(all_schedules)} total schedules to {len(new_schedules)} new schedules")
        return new_schedules