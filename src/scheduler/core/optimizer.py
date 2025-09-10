"""
Main scheduler optimizer that coordinates the optimization process
"""

import json
import logging
from collections import defaultdict
from datetime import date, datetime, timedelta
from datetime import time as datetime_time
from uuid import UUID

from ..config import settings
from ..models import *
from ..solver import ScheduleSolver
from ..solver_v2 import ScheduleSolverV2
from .analyzer import ScheduleAnalyzer
from .cache import CacheManager
from .validator import ScheduleValidator

logger = logging.getLogger(__name__)


class ScheduleOptimizer:
    """Main scheduler class that coordinates the optimization process"""

    def __init__(self):
        self.cache_manager = CacheManager()
        self.validator = ScheduleValidator()
        self.analyzer = ScheduleAnalyzer()

    async def generate_schedule(
        self, request: ScheduleGenerationRequest
    ) -> ScheduleGenerationResponse:
        """
        Generate an optimized schedule for the given parameters
        """

        start_time = datetime.now()
        logger.info(
            f"Starting schedule generation for center {request.center_id}, week {request.week_start_date}"
        )

        try:
            print("🔍 DEBUG: Running constraint analysis...")
            request_dict = {
                "center_id": str(request.center_id),
                "week_start_date": request.week_start_date.isoformat(),
                "staff": [staff.dict() for staff in request.staff],
                "groups": [group.dict() for group in request.groups],
                "staffing_requirements": [req.dict() for req in request.staffing_requirements]
            }
            debug_schedule_constraints(request_dict)

            # Validate input data
            validation_result = await self._validate_input(request)
            if not validation_result.is_valid:
                return ScheduleGenerationResponse(
                    success=False,
                    conflicts=validation_result.conflicts,
                    optimization_result=OptimizationResult(
                        objective_value=0, solve_time_seconds=0, status="INVALID_INPUT"
                    ),
                    message="Input validation failed",
                )

            # Check cache for existing solution
            cache_key = self._generate_cache_key(request)
            cached_result = await self.cache_manager.get(cache_key)
            # if cached_result:
                # logger.info("Returning cached schedule solution")
                # return cached_result

            # Preprocess data
            processed_data = await self._preprocess_data(request)

            # Process staff availability with priority system
            processed_staff = self._process_staff_availability_priority(request)

            # Use period-wide optimization instead of daily optimization to fix excessive schedule generation
            logger.info(f"Using period-wide optimization for {request.total_days} days")

            # Check if we should use the enhanced solver V2
            # Use V2 if we have shift templates OR if any staff has group assignments
            has_group_assignments = any(
                staff_member.group_assignments and len(staff_member.group_assignments) > 0
                for staff_member in processed_staff
            )

            if request.shift_templates and request.shift_template_requirements:
                logger.info(f"Using template-based solver with {len(request.shift_templates)} templates")
                solver_v2 = ScheduleSolverV2(request.optimization_config)

                # Add optimization goals for group assignments and continuity
                if OptimizationGoal.RESPECT_GROUP_ASSIGNMENTS not in request.optimization_config.goals:
                    request.optimization_config.goals.append(OptimizationGoal.RESPECT_GROUP_ASSIGNMENTS)
                if OptimizationGoal.MAXIMIZE_GROUP_CONTINUITY not in request.optimization_config.goals:
                    request.optimization_config.goals.append(OptimizationGoal.MAXIMIZE_GROUP_CONTINUITY)

                schedule, optimization_result, conflicts = solver_v2.solve_with_templates(
                    staff=processed_staff,
                    groups=request.groups,
                    requirements=request.staffing_requirements,
                    shift_templates=request.shift_templates,
                    shift_template_requirements=request.shift_template_requirements,
                    constraints=request.constraints,
                    schedule_start_date=request.schedule_start_date,
                    schedule_end_date=request.effective_end_date,
                    existing_schedule=request.existing_schedules,
                )
            elif has_group_assignments:
                logger.info("Using solver V2 to respect group assignments (no templates)")
                solver_v2 = ScheduleSolverV2(request.optimization_config)

                # Add optimization goals for group assignments and continuity
                if OptimizationGoal.RESPECT_GROUP_ASSIGNMENTS not in request.optimization_config.goals:
                    request.optimization_config.goals.append(OptimizationGoal.RESPECT_GROUP_ASSIGNMENTS)
                if OptimizationGoal.MAXIMIZE_GROUP_CONTINUITY not in request.optimization_config.goals:
                    request.optimization_config.goals.append(OptimizationGoal.MAXIMIZE_GROUP_CONTINUITY)

                # Call solve_with_templates with empty templates - it will still respect group assignments
                schedule, optimization_result, conflicts = solver_v2.solve_with_templates(
                    staff=processed_staff,
                    groups=request.groups,
                    requirements=request.staffing_requirements,
                    shift_templates=[],  # Empty templates
                    shift_template_requirements=[],  # Empty requirements
                    constraints=request.constraints,
                    schedule_start_date=request.schedule_start_date,
                    schedule_end_date=request.effective_end_date,
                    existing_schedule=request.existing_schedules,
                )
            else:
                # Use original solver only when no group assignments exist
                logger.info("Using original solver (no templates, no group assignments)")
                solver = ScheduleSolver(request.optimization_config)

                # Solve for the entire period at once - this prevents 1-hour shifts and excessive schedules
                schedule, optimization_result, conflicts = solver.solve_with_date_range(
                    staff=processed_staff,
                    groups=request.groups,
                    requirements=request.staffing_requirements,
                    constraints=request.constraints,
                    schedule_start_date=request.schedule_start_date,
                    schedule_end_date=request.effective_end_date,
                    existing_schedule=request.existing_schedules,
                )

            # Post-process results
            final_schedule, additional_conflicts = await self._postprocess_schedule(
                schedule, request.staff, request.groups, request.staffing_requirements
            )

            all_conflicts = conflicts + additional_conflicts

            # Calculate metrics
            metrics = await self._calculate_metrics(
                final_schedule, request.staff, request.groups
            )

            # Separate existing vs new schedules
            existing_shifts = []
            new_shifts = []

            if request.existing_schedules:
                existing_shifts = [s for s in request.existing_schedules if s.date >= request.week_start_date]

            # Identify new vs existing shifts (don't auto-generate schedule_ids)
            for shift in final_schedule:
                # Check if this shift is new (not in existing_schedules)
                is_existing = any(
                    ex.staff_id == shift.staff_id and
                    ex.group_id == shift.group_id and
                    ex.date == shift.date and
                    ex.start_time == shift.start_time
                    for ex in existing_shifts
                )
                if not is_existing:
                    new_shifts.append(shift)
                # Note: schedule_id remains None if not provided in input

            # Create response
            response = ScheduleGenerationResponse(
                success=len(final_schedule) > 0
                and optimization_result.status in ["OPTIMAL", "FEASIBLE"],
                schedules=final_schedule,     # All schedules (existing + new)
                new_schedules=new_shifts,     # Only newly generated schedules
                conflicts=all_conflicts,
                optimization_result=optimization_result,
                total_cost=metrics["total_cost"],
                total_hours=metrics["total_hours"],
                staff_utilization=metrics["staff_utilization"],
                satisfaction_score=metrics["satisfaction_score"],
                message=self._generate_status_message(
                    optimization_result, len(all_conflicts)
                ),
            )

            # Cache successful results
            if response.success:
                await self.cache_manager.set(cache_key, response)

            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"Schedule generation completed in {duration:.2f} seconds")

            return response

        except Exception as e:
            logger.error(f"Error in schedule generation: {str(e)}", exc_info=True)

            return ScheduleGenerationResponse(
                success=False,
                conflicts=[
                    ScheduleConflict(
                        conflict_type="system_error",
                        severity="error",
                        description=f"System error: {str(e)}",
                        suggested_solutions=[
                            "Check input data",
                            "Try again",
                            "Contact support",
                        ],
                    )
                ],
                optimization_result=OptimizationResult(
                    objective_value=0,
                    solve_time_seconds=(datetime.now() - start_time).total_seconds(),
                    status="ERROR",
                ),
                message="Schedule generation failed due to system error",
            )

    async def validate_schedule(
          self, request: ScheduleValidationRequest
    ) -> ScheduleValidationResponse:
          """Validate an existing schedule against constraints"""

          try:
              # 1) collect raw conflicts from your three validators
              raw_conflicts = []
              raw_conflicts.extend(
                  self.validator.validate_basic_constraints(
                      request.schedule, request.staff
                  )
              )
              raw_conflicts.extend(
                  self.validator.validate_staffing_requirements(
                      request.schedule, request.groups, request.staffing_requirements
                  )
              )
              raw_conflicts.extend(
                  self.validator.validate_business_rules(
                      request.schedule, request.staff, request.constraints
                  )
              )

              # 2) normalize everything to ScheduleConflict instances
              conflicts: list[ScheduleConflict] = []
              for c in raw_conflicts:
                  if isinstance(c, ScheduleConflict):
                      conflicts.append(c)
                  elif isinstance(c, dict):
                      # assume keys match ScheduleConflict constructor
                      conflicts.append(ScheduleConflict(**c))
                  else:
                      logger.warning(f"Unknown conflict entry: {c!r}")

              # 3) warnings and suggestions as before
              warnings = self.analyzer.generate_warnings(request.schedule, request.staff)
              suggestions = self.analyzer.generate_suggestions(
                  request.schedule, request.staff, request.groups
              )

              # 4) now this no longer errors
              is_valid = len([c for c in conflicts if c.severity == "error"]) == 0

              return ScheduleValidationResponse(
                  is_valid=is_valid,
                  conflicts=conflicts,
                  warnings=warnings,
                  suggestions=suggestions,
              )

          except Exception as e:
              logger.error(f"Error in schedule validation: {str(e)}")

              return ScheduleValidationResponse(
                  is_valid=False,
                  conflicts=[
                      ScheduleConflict(
                          conflict_type="validation_error",
                          severity="error",
                          description=f"Validation error: {str(e)}",
                          suggested_solutions=["Check input data", "Try again"],
                      )
                  ],
                  warnings=[],
                  suggestions=[],
              )


    async def optimize_existing_schedules(
        self,
        current_schedule: list[ScheduledShift],
        request: ScheduleGenerationRequest,
        optimization_goals: list[OptimizationGoal] = None,
    ) -> ScheduleGenerationResponse:
        """Optimize an existing schedule with minimal changes"""

        if optimization_goals:
            request.optimization_config.goals = optimization_goals

        # Set existing schedule for constraints
        request.existing_schedules = current_schedule

        # Generate optimized version
        return await self.generate_schedule(request)

    async def _validate_input(
        self, request: ScheduleGenerationRequest
    ) -> ScheduleValidationResponse:
        """Validate input data before optimization"""

        conflicts = []

        # Check basic data validity
        if not request.staff:
            conflicts.append(
                ScheduleConflict(
                    conflict_type="missing_data",
                    severity="error",
                    description="No staff provided",
                    suggested_solutions=["Add staff members to the request"],
                )
            )

        if not request.groups:
            conflicts.append(
                ScheduleConflict(
                    conflict_type="missing_data",
                    severity="error",
                    description="No groups provided",
                    suggested_solutions=["Add groups to the request"],
                )
            )

        if not request.staffing_requirements:
            conflicts.append(
                ScheduleConflict(
                    conflict_type="missing_data",
                    severity="error",
                    description="No staffing requirements provided",
                    suggested_solutions=["Define staffing requirements for groups"],
                )
            )

        # Check data consistency
        group_ids = {g.group_id for g in request.groups}
        staff_ids = {s.staff_id for s in request.staff}

        for req in request.staffing_requirements:
            if req.group_id not in group_ids:
                conflicts.append(
                    ScheduleConflict(
                        conflict_type="data_inconsistency",
                        severity="error",
                        group_id=req.group_id,
                        description=f"Staffing requirement references unknown group {req.group_id}",
                        suggested_solutions=[
                            "Remove invalid requirement",
                            "Add missing group",
                        ],
                    )
                )

        if request.existing_schedules:
            for shift in request.existing_schedules:
                if shift.staff_id not in staff_ids:
                    conflicts.append(
                        ScheduleConflict(
                            conflict_type="data_inconsistency",
                            severity="warning",
                            staff_id=shift.staff_id,
                            description=f"Existing shift references unknown staff {shift.staff_id}",
                            suggested_solutions=[
                                "Remove invalid shift",
                                "Add missing staff",
                            ],
                        )
                    )

                if shift.group_id not in group_ids:
                    conflicts.append(
                        ScheduleConflict(
                            conflict_type="data_inconsistency",
                            severity="warning",
                            group_id=shift.group_id,
                            description=f"Existing shift references unknown group {shift.group_id}",
                            suggested_solutions=[
                                "Remove invalid shift",
                                "Add missing group",
                            ],
                        )
                    )

        # Check resource constraints
        total_required_hours = self._estimate_required_hours(
            request.staffing_requirements
        )
        total_available_hours = sum(s.max_weekly_hours for s in request.staff)

        if total_required_hours > total_available_hours * 1.2:  # 20% buffer
            conflicts.append(
                ScheduleConflict(
                    conflict_type="capacity_warning",
                    severity="warning",
                    description=f"High resource utilization: {total_required_hours:.1f} required vs {total_available_hours:.1f} available",
                    suggested_solutions=[
                        "Add more staff",
                        "Reduce requirements",
                        "Increase staff availability",
                    ],
                )
            )

        is_valid = len([c for c in conflicts if c.severity == "error"]) == 0

        return ScheduleValidationResponse(
            is_valid=is_valid, conflicts=conflicts, warnings=[], suggestions=[]
        )

    async def _preprocess_data(
        self, request: ScheduleGenerationRequest
    ) -> dict[str, any]:
        """Preprocess and enrich input data"""

        # Enrich staff data with computed metrics
        enriched_staff = []
        for staff_member in request.staff:
            # Calculate availability hours
            weekly_availability = self._calculate_weekly_availability(staff_member)

            # Adjust max hours based on availability
            adjusted_max_hours = min(staff_member.max_weekly_hours, weekly_availability)

            # Create enriched copy
            enriched_staff_member = staff_member.copy(deep=True)
            enriched_staff_member.max_weekly_hours = adjusted_max_hours

            enriched_staff.append(enriched_staff_member)

        # Enrich groups with computed ratios
        enriched_groups = []
        for group in request.groups:
            enriched_group = group.copy(deep=True)
            # Add computed fields if needed
            enriched_groups.append(enriched_group)

        # Expand time-based requirements to hourly slots
        expanded_requirements = self._expand_staffing_requirements(
            request.staffing_requirements
        )

        return {
            "staff": enriched_staff,
            "groups": enriched_groups,
            "requirements": expanded_requirements,
        }

    async def _postprocess_schedule(
        self,
        schedule: list[ScheduledShift],
        staff: list[Staff],
        groups: list[Group],
        requirements: list[StaffingRequirement],
    ) -> tuple[list[ScheduledShift], list[ScheduleConflict]]:
        """Post-process the generated schedule"""

        conflicts = []

        # Validate the schedule
        validation_result = await self.validate_schedule(
            ScheduleValidationRequest(
                schedule=schedule,
                staff=staff,
                groups=groups,
                staffing_requirements=requirements,
            )
        )

        conflicts.extend(validation_result.conflicts)

        # Apply business rules and adjustments
        adjusted_schedule = self._apply_business_rules(schedule, staff)

        # Optimize shift boundaries (merge/split as needed)
        optimized_schedule = self._optimize_shift_boundaries(adjusted_schedule)

        return optimized_schedule, conflicts

    async def _calculate_metrics(
        self, schedule: list[ScheduledShift], staff: list[Staff], groups: list[Group]
    ) -> dict[str, any]:
        """Calculate various metrics for the schedule"""

        metrics = {
            "total_cost": 0.0,
            "total_hours": 0.0,
            "staff_utilization": {},
            "satisfaction_score": 0.0,
        }

        # Calculate total hours and cost
        staff_hours = {}
        total_cost = 0.0

        for shift in schedule:
            metrics["total_hours"] += shift.scheduled_hours

            # Track hours per staff
            if shift.staff_id not in staff_hours:
                staff_hours[shift.staff_id] = 0
            staff_hours[shift.staff_id] += shift.scheduled_hours

            # Calculate cost if hourly rate available
            staff_member = next(
                (s for s in staff if s.staff_id == shift.staff_id), None
            )
            if staff_member and staff_member.hourly_rate:
                shift_cost = shift.scheduled_hours * staff_member.hourly_rate
                total_cost += shift_cost

        metrics["total_cost"] = total_cost

        # Calculate staff utilization
        for staff_member in staff:
            hours_worked = staff_hours.get(staff_member.staff_id, 0)
            utilization = (
                hours_worked / staff_member.max_weekly_hours
                if staff_member.max_weekly_hours > 0
                else 0
            )
            metrics["staff_utilization"][staff_member.staff_id] = min(utilization, 1.0)

        # Calculate satisfaction score based on preferences
        satisfaction_scores = []
        for staff_member in staff:
            staff_shifts = [s for s in schedule if s.staff_id == staff_member.staff_id]
            staff_satisfaction = self._calculate_staff_satisfaction(
                staff_member, staff_shifts
            )
            satisfaction_scores.append(staff_satisfaction)

        metrics["satisfaction_score"] = (
            sum(satisfaction_scores) / len(satisfaction_scores)
            if satisfaction_scores
            else 0.0
        )

        return metrics

    def _generate_cache_key(self, request: ScheduleGenerationRequest) -> str:
        """Generate a cache key for the request"""

        # Create a hash of the key parameters
        key_data = {
            "center_id": str(request.center_id),
            "week_start_date": request.week_start_date.isoformat(),
            "staff_count": len(request.staff),
            "groups_count": len(request.groups),
            "requirements_count": len(request.staffing_requirements),
            "config": request.optimization_config.dict(),
        }

        import hashlib

        key_str = json.dumps(key_data, sort_keys=True)
        return f"schedule:{hashlib.md5(key_str.encode()).hexdigest()}"

    def _generate_status_message(
        self, optimization_result: OptimizationResult, conflict_count: int
    ) -> str:
        """Generate a human-readable status message"""

        if optimization_result.status == "OPTIMAL":
            if conflict_count == 0:
                return "Optimal schedule generated successfully with no conflicts"
            else:
                return f"Optimal schedule generated with {conflict_count} conflicts to resolve"

        elif optimization_result.status == "FEASIBLE":
            return f"Feasible schedule generated in {optimization_result.solve_time_seconds:.1f}s with {conflict_count} conflicts"

        elif optimization_result.status == "INFEASIBLE":
            return "No feasible schedule found - please review constraints and requirements"

        elif optimization_result.status == "TIMEOUT":
            return "Schedule optimization timed out - try reducing problem size or increasing time limit"

        else:
            return f"Schedule generation failed: {optimization_result.status}"

    def _estimate_required_hours(
        self, requirements: list[StaffingRequirement]
    ) -> float:
        """Estimate total required staff hours from requirements"""

        total_hours = 0.0

        for req in requirements:
            # Calculate duration of time slot
            start_dt = datetime.combine(date.today(), req.time_slot.start_time)
            end_dt = datetime.combine(date.today(), req.time_slot.end_time)

            if end_dt <= start_dt:
                end_dt += timedelta(days=1)

            duration = (end_dt - start_dt).total_seconds() / 3600
            total_hours += req.min_staff_count * duration

        return total_hours

    def _calculate_weekly_availability(self, staff_member: Staff) -> float:
        """Calculate total weekly availability hours for staff member"""

        total_hours = 0.0

        for availability in staff_member.availability:
            if availability.is_available:
                start_dt = datetime.combine(date.today(), availability.start_time)
                end_dt = datetime.combine(date.today(), availability.end_time)

                if end_dt <= start_dt:
                    end_dt += timedelta(days=1)

                daily_hours = (end_dt - start_dt).total_seconds() / 3600
                total_hours += daily_hours

        return total_hours

    def _expand_staffing_requirements(
        self, requirements: list[StaffingRequirement]
    ) -> list[StaffingRequirement]:
        """Expand time-based requirements to hourly requirements"""

        expanded = []

        for req in requirements:
            start_hour = req.time_slot.start_time.hour
            end_hour = req.time_slot.end_time.hour

            if end_hour <= start_hour:
                end_hour += 24

            # Create hourly requirements
            for hour in range(start_hour, end_hour):
                hourly_req = StaffingRequirement(
                    group_id=req.group_id,
                    time_slot=TimeSlot(
                        start_time=datetime_time(hour % 24, 0),
                        end_time=datetime_time((hour + 1) % 24, 0),
                        day_of_week=req.time_slot.day_of_week,
                    ),
                    min_staff_count=req.min_staff_count,
                    max_staff_count=req.max_staff_count,
                    required_qualifications=req.required_qualifications,
                    preferred_qualifications=req.preferred_qualifications,
                )
                expanded.append(hourly_req)

        return expanded

    def _apply_business_rules(
        self, schedule: list[ScheduledShift], staff: list[Staff]
    ) -> list[ScheduledShift]:
        """Apply business rules to adjust the schedule"""

        # Apply minimum shift duration
        adjusted_schedule = []

        for shift in schedule:
            if shift.scheduled_hours < settings.min_shift_duration:
                # Try to extend shift or remove if too short
                # For now, just flag it
                shift.notes = f"Short shift: {shift.scheduled_hours} hours (min: {settings.min_shift_duration})"

            adjusted_schedule.append(shift)

        return adjusted_schedule

    def _optimize_shift_boundaries(
        self, schedule: list[ScheduledShift]
    ) -> list[ScheduledShift]:
        """Optimize shift start/end times for better boundaries"""

        # Group shifts by staff and date
        grouped_shifts = {}

        for shift in schedule:
            key = (shift.staff_id, shift.date)
            if key not in grouped_shifts:
                grouped_shifts[key] = []
            grouped_shifts[key].append(shift)

        optimized_schedule = []

        for key, shifts in grouped_shifts.items():
            if len(shifts) == 1:
                optimized_schedule.extend(shifts)
                continue

            # Sort shifts by start time
            shifts.sort(key=lambda s: s.start_time)

            # Merge adjacent shifts for same group
            merged_shifts = []
            current_shift = shifts[0]

            for next_shift in shifts[1:]:
                if (
                    current_shift.group_id == next_shift.group_id
                    and self._shifts_are_adjacent(current_shift, next_shift)
                ):
                    # Merge shifts
                    current_shift.end_time = next_shift.end_time
                    current_shift.scheduled_hours += next_shift.scheduled_hours
                else:
                    merged_shifts.append(current_shift)
                    current_shift = next_shift

            merged_shifts.append(current_shift)
            optimized_schedule.extend(merged_shifts)

        return optimized_schedule

    def _shifts_are_adjacent(
        self, shift1: ScheduledShift, shift2: ScheduledShift
    ) -> bool:
        """Check if two shifts are adjacent in time"""

        end_dt = datetime.combine(shift1.date, shift1.end_time)
        start_dt = datetime.combine(shift2.date, shift2.start_time)

        # Allow up to 1 hour gap for merging
        gap = (start_dt - end_dt).total_seconds() / 3600
        return 0 <= gap <= 1.0

    def _calculate_staff_satisfaction(
        self, staff_member: Staff, staff_shifts: list[ScheduledShift]
    ) -> float:
        """Calculate satisfaction score for a staff member's schedule"""

        if not staff_shifts:
            return 1.0  # Neutral if no shifts

        satisfaction_score = 0.0
        total_weight = 0.0

        for preference in staff_member.preferences:
            pref_satisfaction = self._evaluate_preference_satisfaction(
                preference, staff_shifts
            )
            satisfaction_score += pref_satisfaction * preference.weight
            total_weight += preference.weight

        if total_weight == 0:
            return 1.0

        return satisfaction_score / total_weight

    def _evaluate_preference_satisfaction(
        self, preference: StaffPreference, shifts: list[ScheduledShift]
    ) -> float:
        """Evaluate how well shifts satisfy a specific preference"""

        if preference.preference_type == PreferenceType.PREFERRED_TIME:
            # Count shifts that match preferred times
            matching_shifts = 0
            for shift in shifts:
                if self._shift_matches_preference(shift, preference):
                    matching_shifts += 1

            return matching_shifts / len(shifts) if shifts else 0.0

        elif preference.preference_type == PreferenceType.EXCLUDE_DAYS:
            # Penalize shifts on avoided days
            avoided_shifts = 0
            for shift in shifts:
                if (
                    preference.day_of_week is not None
                    and shift.date.weekday() == preference.day_of_week
                ):
                    avoided_shifts += 1

            return 1.0 - (avoided_shifts / len(shifts)) if shifts else 1.0

        elif preference.preference_type == PreferenceType.MAX_HOURS:
            # Check if daily hours don't exceed preference
            if preference.max_hours_per_day:
                daily_hours = {}
                for shift in shifts:
                    date_key = shift.date
                    if date_key not in daily_hours:
                        daily_hours[date_key] = 0
                    daily_hours[date_key] += shift.scheduled_hours

                violations = sum(
                    1
                    for hours in daily_hours.values()
                    if hours > preference.max_hours_per_day
                )
                return 1.0 - (violations / len(daily_hours)) if daily_hours else 1.0

        return 1.0  # Default neutral satisfaction

    def _shift_matches_preference(
        self, shift: ScheduledShift, preference: StaffPreference
    ) -> bool:
        """Check if a shift matches a preference criteria"""

        if preference.day_of_week is not None:
            if shift.date.weekday() != preference.day_of_week:
                return False

        if preference.time_range_start and preference.time_range_end:
            shift_start_hour = shift.start_time.hour
            pref_start_hour = preference.time_range_start.hour
            pref_end_hour = preference.time_range_end.hour

            if pref_end_hour <= pref_start_hour:  # Overnight range
                return (
                    shift_start_hour >= pref_start_hour
                    or shift_start_hour <= pref_end_hour
                )
            else:
                return pref_start_hour <= shift_start_hour <= pref_end_hour

        return True

    async def generate_enhanced_schedule(
        self, request: EnhancedScheduleGenerationRequest
    ) -> EnhancedScheduleResponse:
        """
        Generate an enhanced schedule with flexible date range and new constraints
        """
        start_time = datetime.now()
        logger.info(
            f"Starting enhanced schedule generation for center {request.center_id}, "
            f"period {request.schedule_start_date} to {request.effective_end_date} ({request.total_days} days)"
        )

        try:
            # Validate staff absences and availability
            self._validate_enhanced_constraints(request)

            # Process staff availability with priority system
            processed_staff = self._process_staff_availability_priority(request)

            # Use period-wide optimization instead of daily optimization to fix excessive schedule generation
            logger.info(f"Using period-wide optimization for {request.total_days} days")

            # Check if we should use the enhanced solver V2
            # Use V2 if we have shift templates OR if any staff has group assignments
            has_group_assignments = any(
                staff_member.group_assignments and len(staff_member.group_assignments) > 0
                for staff_member in processed_staff
            )

            if request.shift_templates and request.shift_template_requirements:
                logger.info(f"Using template-based solver with {len(request.shift_templates)} templates")
                solver_v2 = ScheduleSolverV2(request.optimization_config)

                # Add optimization goals for group assignments and continuity
                if OptimizationGoal.RESPECT_GROUP_ASSIGNMENTS not in request.optimization_config.goals:
                    request.optimization_config.goals.append(OptimizationGoal.RESPECT_GROUP_ASSIGNMENTS)
                if OptimizationGoal.MAXIMIZE_GROUP_CONTINUITY not in request.optimization_config.goals:
                    request.optimization_config.goals.append(OptimizationGoal.MAXIMIZE_GROUP_CONTINUITY)

                schedule, optimization_result, conflicts = solver_v2.solve_with_templates(
                    staff=processed_staff,
                    groups=request.groups,
                    requirements=request.staffing_requirements,
                    shift_templates=request.shift_templates,
                    shift_template_requirements=request.shift_template_requirements,
                    constraints=request.constraints,
                    schedule_start_date=request.schedule_start_date,
                    schedule_end_date=request.effective_end_date,
                    existing_schedule=request.existing_schedules,
                )
            elif has_group_assignments:
                logger.info("Using solver V2 to respect group assignments (no templates)")
                solver_v2 = ScheduleSolverV2(request.optimization_config)

                # Add optimization goals for group assignments and continuity
                if OptimizationGoal.RESPECT_GROUP_ASSIGNMENTS not in request.optimization_config.goals:
                    request.optimization_config.goals.append(OptimizationGoal.RESPECT_GROUP_ASSIGNMENTS)
                if OptimizationGoal.MAXIMIZE_GROUP_CONTINUITY not in request.optimization_config.goals:
                    request.optimization_config.goals.append(OptimizationGoal.MAXIMIZE_GROUP_CONTINUITY)

                # Call solve_with_templates with empty templates - it will still respect group assignments
                schedule, optimization_result, conflicts = solver_v2.solve_with_templates(
                    staff=processed_staff,
                    groups=request.groups,
                    requirements=request.staffing_requirements,
                    shift_templates=[],  # Empty templates
                    shift_template_requirements=[],  # Empty requirements
                    constraints=request.constraints,
                    schedule_start_date=request.schedule_start_date,
                    schedule_end_date=request.effective_end_date,
                    existing_schedule=request.existing_schedules,
                )
            else:
                # Use original solver only when no group assignments exist
                logger.info("Using original solver (no templates, no group assignments)")
                solver = ScheduleSolver(request.optimization_config)

                # Solve for the entire period at once - this prevents 1-hour shifts and excessive schedules
                schedule, optimization_result, conflicts = solver.solve_with_date_range(
                    staff=processed_staff,
                    groups=request.groups,
                    requirements=request.staffing_requirements,
                    constraints=request.constraints,
                    schedule_start_date=request.schedule_start_date,
                    schedule_end_date=request.effective_end_date,
                    existing_schedule=request.existing_schedules,
                )

            # Separate existing vs new schedules
            existing_shifts = []
            new_shifts = []
            all_daily_conflicts = conflicts
            total_cost = 0.0
            total_hours = 0.0

            # Separate existing vs new schedules from the single optimized result
            if request.existing_schedules:
                # Track which schedules are existing vs new
                existing_schedule_keys = set()
                for existing_shift in request.existing_schedules:
                    if request.schedule_start_date <= existing_shift.date <= request.effective_end_date:
                        existing_shifts.append(existing_shift)
                        # Create a key to identify this shift in the optimized schedule
                        key = (existing_shift.staff_id, existing_shift.group_id, existing_shift.date, existing_shift.start_time)
                        existing_schedule_keys.add(key)

                        # Count hours and cost from existing shifts
                        total_hours += existing_shift.scheduled_hours
                        staff_member = next((s for s in request.staff if s.staff_id == existing_shift.staff_id), None)
                        if staff_member and staff_member.hourly_rate:
                            rate = staff_member.overtime_rate if existing_shift.is_overtime else staff_member.hourly_rate
                            if existing_shift.is_overtime and not staff_member.overtime_rate:
                                rate = staff_member.hourly_rate * 1.5
                            total_cost += existing_shift.scheduled_hours * rate

                logger.info(f"Preserved {len(existing_shifts)} existing shifts")
            else:
                existing_schedule_keys = set()

            # Add newly generated shifts (those not in existing schedules)
            for shift in schedule:
                key = (shift.staff_id, shift.group_id, shift.date, shift.start_time)
                if key not in existing_schedule_keys:
                    # This is a newly generated shift
                    shift.schedule_id = None  # Ensure new shifts have no schedule_id
                    new_shifts.append(shift)

                    # Calculate cost for new shifts
                    staff_member = next((s for s in request.staff if s.staff_id == shift.staff_id), None)
                    if staff_member and staff_member.hourly_rate:
                        rate = staff_member.overtime_rate if shift.is_overtime else staff_member.hourly_rate
                        if shift.is_overtime and not staff_member.overtime_rate:
                            rate = staff_member.hourly_rate * 1.5
                        total_cost += shift.scheduled_hours * rate

                    total_hours += shift.scheduled_hours

            logger.info(f"Generated {len(new_shifts)} new shifts using period-wide optimization")

            # Combine all shifts for the main schedule fields
            combined_schedule = existing_shifts + new_shifts

            logger.info(f"SUMMARY: Total schedules={len(combined_schedule)}, New schedules={len(new_shifts)}, Existing schedules={len(existing_shifts)}")

            # Use the optimization result from the solver
            combined_conflicts = conflicts

            # Add infeasibility message if solver returned INFEASIBLE
            if optimization_result.status == "INFEASIBLE":
                combined_conflicts.append(ScheduleConflict(
                    conflict_type="infeasible_requirements",
                    severity="error",
                    description="Scheduling requirements are mathematically impossible to satisfy. Insufficient available staff to cover all required groups after accounting for existing schedule commitments.",
                    suggested_solutions=[
                        "Add more staff members",
                        "Reduce coverage requirements for some groups",
                        "Remove some existing schedule commitments",
                        "Allow group merging by increasing max_staff_count",
                        "Adjust shift times to reduce conflicts"
                    ]
                ))

            # Apply enhanced constraints validation
            enhanced_conflicts = self._validate_enhanced_constraints_post_generation(
                request, combined_schedule
            )
            combined_conflicts.extend(enhanced_conflicts)

            # Calculate enhanced metrics
            staff_utilization = self._calculate_enhanced_staff_utilization(
                combined_schedule, request.staff, request.total_days
            )

            # Update optimization result with enhanced metrics, but preserve solver status
            optimization_result.objective_value = len(combined_schedule)
            optimization_result.solve_time_seconds = (datetime.now() - start_time).total_seconds()

            # Don't override INFEASIBLE status from solver
            if optimization_result.status != "INFEASIBLE":
                optimization_result.status = "OPTIMAL" if not enhanced_conflicts else "FEASIBLE"

            optimization_result.conflicts_resolved = len([c for c in combined_conflicts if c.severity != "error"])

            # If solver says INFEASIBLE, treat as failure even if we have existing schedules
            if optimization_result.status == "INFEASIBLE":
                success = False
            else:
                success = len(combined_schedule) > 0 and not any(c.severity == "error" for c in combined_conflicts)

            response = EnhancedScheduleResponse(
                success=success,
                schedules=combined_schedule,  # All schedules (existing + new)
                new_schedules=new_shifts,    # Only newly generated schedules
                conflicts=combined_conflicts,
                optimization_result=optimization_result,
                total_cost=total_cost,
                total_hours=total_hours,
                staff_utilization=staff_utilization,
                satisfaction_score=0.8,  # Placeholder
                message="Enhanced schedule generated successfully" if success else "Schedule generation completed with conflicts",
                schedule_start_date=request.schedule_start_date,
                schedule_end_date=request.effective_end_date,
                # Add schedule counts
                total_schedules_count=len(combined_schedule),
                new_schedules_count=len(new_shifts),
                existing_schedules_count=len(existing_shifts),
                total_days=request.total_days,
                daily_averages={
                    "hours_per_day": total_hours / request.total_days if request.total_days > 0 else 0,
                    "shifts_per_day": len(combined_schedule) / request.total_days if request.total_days > 0 else 0,
                    "cost_per_day": total_cost / request.total_days if request.total_days > 0 else 0,
                },
                period_coverage=1.0 if success else 0.7,
            )

            logger.info(
                f"Enhanced schedule generation completed in {optimization_result.solve_time_seconds:.2f}s: "
                f"shifts={len(combined_schedule)}, conflicts={len(combined_conflicts)}, "
                f"total_days={request.total_days}"
            )

            return response

        except Exception as e:
            logger.error(f"Enhanced schedule generation failed: {str(e)}", exc_info=True)
            return EnhancedScheduleResponse(
                success=False,
                schedule=[],
                schedules=[],
                new_schedules=[],
                conflicts=[ScheduleConflict(
                    conflict_type="generation_error",
                    severity="error",
                    description=f"Schedule generation failed: {str(e)}",
                    suggested_solutions=["Check input data and try again"]
                )],
                optimization_result=OptimizationResult(
                    objective_value=0,
                    solve_time_seconds=(datetime.now() - start_time).total_seconds(),
                    status="INFEASIBLE",
                    iterations=0,
                    conflicts_resolved=0
                ),
                total_cost=0.0,
                total_hours=0.0,
                staff_utilization={},
                satisfaction_score=0.0,
                message=f"Enhanced schedule generation failed: {str(e)}",
                schedule_start_date=request.schedule_start_date,
                schedule_end_date=request.effective_end_date,
                total_days=request.total_days,
                daily_averages={},
                period_coverage=0.0,
            )

    def _validate_enhanced_constraints(self, request: EnhancedScheduleGenerationRequest):
        """Validate enhanced constraints before generation"""
        # Validate center configuration
        if not request.center_config.opening_hours:
            raise ValueError("Center must have opening hours defined")

        # Validate staff assignments
        for staff in request.staff:
            primary_assignments = [a for a in staff.group_assignments if a.assignment_type == GroupAssignmentType.PRIMARY]
            if len(primary_assignments) > 1:
                raise ValueError(f"Staff {staff.name} has multiple primary group assignments")

        # Validate existing schedule if provided
        if request.existing_schedules:
            self._validate_existing_schedules(request)

    def _validate_existing_schedules(self, request: EnhancedScheduleGenerationRequest):
        """Validate the existing schedule for consistency"""
        staff_ids = {s.staff_id for s in request.staff}
        group_ids = {g.group_id for g in request.groups}

        for shift in request.existing_schedules:
            # Validate staff exists
            if shift.staff_id not in staff_ids:
                raise ValueError(f"Existing shift references unknown staff member {shift.staff_id}")

            # Validate group exists
            if shift.group_id not in group_ids:
                raise ValueError(f"Existing shift references unknown group {shift.group_id}")

            # Validate shift is reasonable
            if shift.scheduled_hours <= 0:
                raise ValueError(f"Existing shift has invalid hours: {shift.scheduled_hours}")

            if shift.scheduled_hours > 16:
                raise ValueError(f"Existing shift has unrealistic hours: {shift.scheduled_hours} (max 16h)")

            # Check if shift falls within the requested date range
            if not (request.schedule_start_date <= shift.date <= request.effective_end_date):
                logger.warning(f"Existing shift on {shift.date} is outside requested date range {request.schedule_start_date} to {request.effective_end_date}")

        # Validate for conflicts between existing schedules
        conflicts = self._validate_existing_schedule_conflicts(request)
        if conflicts:
            # DON'T filter out existing schedules - keep them all but warn about violations
            logger.warning(f"Found {len(conflicts)} existing schedule conflicts - keeping schedules but they may affect optimization")
            for conflict in conflicts[:3]:  # Log first 3 conflicts
                logger.warning(f"Existing schedule conflict: {conflict.description}")

        logger.info(f"Validated {len(request.existing_schedules)} existing shifts (kept all including conflicts)")

    def _validate_existing_schedule_conflicts(self, request: EnhancedScheduleGenerationRequest) -> list[ScheduleConflict]:
        """Validate existing schedules for internal conflicts"""
        conflicts = []

        if not request.existing_schedules:
            return conflicts

        # Filter to only schedules within the requested period
        valid_schedules = [
            shift for shift in request.existing_schedules
            if request.schedule_start_date <= shift.date <= request.effective_end_date
        ]

        # Check for same staff assigned to multiple groups at same time
        staff_time_assignments = defaultdict(list)
        for shift in valid_schedules:
            key = (shift.staff_id, shift.date, shift.start_time, shift.end_time)
            staff_time_assignments[key].append(shift)

        for (staff_id, date, start_time, end_time), shifts in staff_time_assignments.items():
            if len(shifts) > 1:
                group_names = [f"Group-{str(shift.group_id)[-4:]}" for shift in shifts]
                conflicts.append(ScheduleConflict(
                    conflict_type="existing_schedule_time_conflict",
                    severity="error",
                    staff_id=staff_id,
                    description=f"Staff {str(staff_id)[-4:]} assigned to multiple groups ({', '.join(group_names)}) at same time {date} {start_time}-{end_time}",
                    suggested_solutions=["Remove conflicting assignments", "Adjust shift times", "Assign to single group only"]
                ))

        # Check for multiple staff assigned to same group beyond max capacity
        group_time_assignments = defaultdict(list)
        for shift in valid_schedules:
            # For time conflict checking, consider overlapping hours
            for hour in range(shift.start_time.hour, shift.end_time.hour):
                key = (shift.group_id, shift.date, hour)
                group_time_assignments[key].append(shift)

        # Find the max_staff_count for each group from requirements
        group_max_staff = {}
        for req in request.staffing_requirements:
            group_max_staff[req.group_id] = req.max_staff_count or req.min_staff_count

        for (group_id, date, hour), shifts in group_time_assignments.items():
            # Get unique staff for this group/time slot
            unique_staff = list(set(shift.staff_id for shift in shifts))
            max_allowed = group_max_staff.get(group_id, 999)  # Default to high number if not specified

            if len(unique_staff) > max_allowed:
                staff_names = [f"Staff-{str(staff_id)[-4:]}" for staff_id in unique_staff]
                conflicts.append(ScheduleConflict(
                    conflict_type="existing_schedule_capacity_conflict",
                    severity="error",
                    group_id=group_id,
                    description=f"Group {str(group_id)[-4:]} has {len(unique_staff)} staff assigned at {date} {hour}:00 but max allowed is {max_allowed}. Staff: {', '.join(staff_names)}",
                    suggested_solutions=["Remove excess staff assignments", "Increase max_staff_count", "Split shifts across different times"]
                ))

        return conflicts

    def _filter_conflicting_existing_schedules(self, existing_schedules: list[ScheduledShift]) -> tuple[list[ScheduledShift], int]:
        """Remove conflicting existing schedules, keeping the first occurrence of each conflict"""
        filtered_schedules = []
        seen_staff_times = set()
        group_staff_count = defaultdict(set)  # Track staff per group
        removed_count = 0

        for shift in existing_schedules:
            # Check 1: Staff-time conflicts (same staff, same time)
            staff_time_key = (shift.staff_id, shift.date, shift.start_time, shift.end_time)

            if staff_time_key in seen_staff_times:
                # This staff member already has an assignment at this time - skip this shift
                removed_count += 1
                logger.debug(f"Filtered staff-time conflict: Staff {str(shift.staff_id)[-4:]} already assigned at {shift.date} {shift.start_time}-{shift.end_time}")
                continue

            # Check 2: Group capacity conflicts (too many staff for same group)
            group_key = (shift.group_id, shift.date)
            current_staff_in_group = group_staff_count[group_key]

            # For this implementation, assume max_staff_count = 1 for all groups
            # TODO: Get actual max_staff_count from requirements
            max_staff_per_group = 1  # Hardcoded for now since all your groups have max=1

            if len(current_staff_in_group) >= max_staff_per_group:
                # This group already has max staff - skip this shift
                removed_count += 1
                existing_staff = [str(sid)[-4:] for sid in current_staff_in_group]
                logger.debug(f"Filtered group capacity conflict: Group {str(shift.group_id)[-4:]} already has {len(current_staff_in_group)} staff {existing_staff}, max={max_staff_per_group}")
                continue

            # Keep this shift
            seen_staff_times.add(staff_time_key)
            group_staff_count[group_key].add(shift.staff_id)
            filtered_schedules.append(shift)

        return filtered_schedules, removed_count

    def _process_staff_availability_priority(self, request: EnhancedScheduleGenerationRequest) -> list[Staff]:
        """Process staff availability with priority system: Absences > Preferences > Center opening hours"""
        processed_staff = []

        for staff in request.staff:
            # Create a copy of the staff member
            processed_staff_member = staff.copy(deep=True)

            # Process availability for each day in the date range
            new_availability = []
            current_date = request.schedule_start_date

            while current_date <= request.effective_end_date:
                day_of_week = current_date.weekday()

                # Check for absences (highest priority)
                is_absent = any(
                    absence.start_date <= current_date <= absence.end_date
                    for absence in staff.absences
                )

                if is_absent:
                    # Staff is absent, no availability
                    continue

                # Check for preferences
                day_preferences = [
                    av for av in staff.availability
                    if av.day_of_week == day_of_week and av.is_available
                ]

                if day_preferences:
                    # Use staff preferences
                    new_availability.extend(day_preferences)
                else:
                    # Use center opening hours as fallback
                    center_opening = [
                        oh for oh in request.center_config.opening_hours
                        if oh.day_of_week == day_of_week
                    ]

                    for opening in center_opening:
                        new_availability.append(StaffAvailability(
                            day_of_week=day_of_week,
                            start_time=opening.start_time,
                            end_time=opening.end_time,
                            is_available=True
                        ))

                current_date += timedelta(days=1)

            processed_staff_member.availability = new_availability
            processed_staff.append(processed_staff_member)

        return processed_staff

    def _create_daily_request(
        self,
        enhanced_request: EnhancedScheduleGenerationRequest,
        target_date: date,
        processed_staff: list[Staff]
    ) -> ScheduleGenerationRequest:
        """Create a daily schedule request from enhanced request"""

        # Find Monday of the week containing target_date
        days_since_monday = target_date.weekday()
        monday = target_date - timedelta(days=days_since_monday)

        # Filter staffing requirements for the target day
        day_of_week = target_date.weekday()
        daily_requirements = []

        for req in enhanced_request.staffing_requirements:
            # Include requirement if:
            # 1. day_of_week is None (applies to all days), OR
            # 2. day_of_week matches the current day
            if req.time_slot.day_of_week is None or req.time_slot.day_of_week == day_of_week:
                # For all-day requirements (day_of_week is None), set day_of_week for this specific day
                if req.time_slot.day_of_week is None:
                    # Create a copy with the specific day set
                    daily_req = req.copy(deep=True)
                    daily_req.time_slot.day_of_week = day_of_week
                    daily_requirements.append(daily_req)
                else:
                    daily_requirements.append(req)

        # Filter existing schedule for this specific date
        daily_existing_schedules = []
        if enhanced_request.existing_schedules:
            daily_existing_schedules = [
                shift for shift in enhanced_request.existing_schedules
                if shift.date == target_date
            ]
            if daily_existing_schedules:
                logger.info(f"Found {len(daily_existing_schedules)} existing shifts for {target_date}")

        return ScheduleGenerationRequest(
            center_id=enhanced_request.center_id,
            week_start_date=monday,
            staff=processed_staff,
            groups=enhanced_request.groups,
            staffing_requirements=daily_requirements,
            constraints=enhanced_request.constraints,
            optimization_config=enhanced_request.optimization_config,
            existing_schedules=daily_existing_schedules,
            extra_shift_eligible_staff_ids=enhanced_request.extra_shift_eligible_staff_ids,
        )

    def _validate_enhanced_constraints_post_generation(
        self,
        request: EnhancedScheduleGenerationRequest,
        schedule: list[ScheduledShift]
    ) -> list[ScheduleConflict]:
        """Validate enhanced constraints after schedule generation"""
        conflicts = []

        # Validate overtime limits
        staff_daily_hours = defaultdict(lambda: defaultdict(float))
        staff_weekly_hours = defaultdict(float)

        for shift in schedule:
            date_key = shift.date
            staff_daily_hours[shift.staff_id][date_key] += shift.scheduled_hours
            staff_weekly_hours[shift.staff_id] += shift.scheduled_hours

        # Check daily overtime limits
        for staff_id, daily_hours in staff_daily_hours.items():
            for date_key, hours in daily_hours.items():
                if hours > request.center_config.overtime_threshold_daily + request.center_config.max_daily_overtime_hours:
                    conflicts.append(ScheduleConflict(
                        conflict_type="daily_overtime_violation",
                        severity="error",
                        staff_id=staff_id,
                        description=f"Staff exceeds daily overtime limit: {hours}h > {request.center_config.overtime_threshold_daily + request.center_config.max_daily_overtime_hours}h on {date_key}",
                        suggested_solutions=["Reduce shift hours", "Assign additional staff"]
                    ))

        # Check weekly overtime limits (approximate based on date range)
        for staff_id, total_hours in staff_weekly_hours.items():
            weekly_limit = request.center_config.overtime_threshold_weekly + request.center_config.max_weekly_overtime_hours
            if total_hours > weekly_limit:
                conflicts.append(ScheduleConflict(
                    conflict_type="weekly_overtime_violation",
                    severity="error",
                    staff_id=staff_id,
                    description=f"Staff exceeds weekly overtime limit: {total_hours}h > {weekly_limit}h",
                    suggested_solutions=["Redistribute hours across staff", "Hire additional staff"]
                ))

        # Validate staff-to-child ratios
        self._validate_staff_child_ratios(request, schedule, conflicts)

        return conflicts

    def _validate_staff_child_ratios(
        self,
        request: EnhancedScheduleGenerationRequest,
        schedule: list[ScheduledShift],
        conflicts: list[ScheduleConflict]
    ):
        """Validate staff-to-child ratios"""
        # Group shifts by date and time
        shift_groups = defaultdict(list)

        for shift in schedule:
            key = (shift.date, shift.start_time, shift.end_time)
            shift_groups[key].append(shift)

        for (date_key, start_time, end_time), shifts in shift_groups.items():
            # Calculate staff count and required ratios
            staff_count = len(shifts)

            # Get groups and their enrollments
            total_children = 0
            required_staff = 0

            for group in request.groups:
                if any(shift.group_id == group.group_id for shift in shifts):
                    children_count = group.current_enrollment
                    total_children += children_count

                    # Get ratio for this age group
                    ratio = request.center_config.staff_child_ratios.get(group.age_group, 8.0)
                    required_staff += max(1, int(children_count / ratio))

            if staff_count < required_staff:
                conflicts.append(ScheduleConflict(
                    conflict_type="staff_child_ratio_violation",
                    severity="warning",
                    description=f"Insufficient staff for required ratios on {date_key} {start_time}-{end_time}: {staff_count} staff for {total_children} children (need {required_staff})",
                    suggested_solutions=["Add more staff to this time slot", "Reduce group sizes"]
                ))

    def _calculate_enhanced_staff_utilization(
        self,
        schedule: list[ScheduledShift],
        staff: list[Staff],
        total_days: int
    ) -> dict[UUID, float]:
        """Calculate staff utilization for enhanced scheduling"""
        utilization = {}

        # Calculate hours per staff
        staff_hours = defaultdict(float)
        for shift in schedule:
            staff_hours[shift.staff_id] += shift.scheduled_hours

        # Calculate utilization based on availability
        for staff_member in staff:
            total_hours = staff_hours.get(staff_member.staff_id, 0)

            # Estimate available hours (simplified)
            available_hours_per_day = 8  # Assume 8 hours available per day
            total_available_hours = available_hours_per_day * total_days

            utilization[staff_member.staff_id] = min(1.0, total_hours / total_available_hours) if total_available_hours > 0 else 0.0

        return utilization

    def _consolidate_conflicts(self, conflicts: list[ScheduleConflict]) -> list[ScheduleConflict]:
        """Consolidate similar conflicts to avoid duplicates and reduce noise"""
        if not conflicts:
            return []

        # Group conflicts by type and severity
        conflict_groups = defaultdict(lambda: defaultdict(list))

        for conflict in conflicts:
            # Create a grouping key based on conflict type and core description
            group_key = self._get_conflict_group_key(conflict)
            conflict_groups[conflict.conflict_type][group_key].append(conflict)

        consolidated = []

        for conflict_type, type_groups in conflict_groups.items():
            for group_key, group_conflicts in type_groups.items():
                if len(group_conflicts) == 1:
                    # Single conflict, add as-is
                    consolidated.append(group_conflicts[0])
                else:
                    # Multiple similar conflicts, consolidate them
                    consolidated_conflict = self._merge_conflicts(group_conflicts)
                    consolidated.append(consolidated_conflict)

        return consolidated

    def _get_conflict_group_key(self, conflict: ScheduleConflict) -> str:
        """Generate a grouping key for similar conflicts"""
        key_parts = [conflict.conflict_type, conflict.severity]

        # Add staff_id if present for staff-specific conflicts
        if conflict.staff_id:
            key_parts.append(f"staff_{conflict.staff_id}")

        # Add group_id if present for group-specific conflicts
        if conflict.group_id:
            key_parts.append(f"group_{conflict.group_id}")

        # Add time slot if present for time-specific conflicts
        if conflict.time_slot:
            day_part = conflict.time_slot.day_of_week if conflict.time_slot.day_of_week is not None else "all_days"
            key_parts.append(f"time_{day_part}_{conflict.time_slot.start_time}")

        # For conflicts without specific identifiers, group by description pattern
        if not conflict.staff_id and not conflict.group_id and not conflict.time_slot:
            # Extract pattern from description (e.g., "overtime violation", "ratio violation")
            desc_lower = conflict.description.lower()
            if "overtime" in desc_lower:
                key_parts.append("overtime_pattern")
            elif "ratio" in desc_lower:
                key_parts.append("ratio_pattern")
            elif "availability" in desc_lower:
                key_parts.append("availability_pattern")
            else:
                key_parts.append("general_pattern")

        return "_".join(key_parts)

    def _merge_conflicts(self, conflicts: list[ScheduleConflict]) -> ScheduleConflict:
        """Merge multiple similar conflicts into a single consolidated conflict"""
        if len(conflicts) == 1:
            return conflicts[0]

        # Take the first conflict as the base
        base_conflict = conflicts[0]
        count = len(conflicts)

        # Create consolidated description
        if count <= 3:
            # For small numbers, list specific instances
            descriptions = [c.description for c in conflicts]
            consolidated_description = f"{count} similar issues: " + "; ".join(descriptions[:3])
        else:
            # For larger numbers, provide summary
            consolidated_description = f"{count} similar {base_conflict.conflict_type} issues detected"

        # Collect all unique suggested solutions
        all_solutions = set()
        for conflict in conflicts:
            all_solutions.update(conflict.suggested_solutions)

        # Create consolidated conflict
        return ScheduleConflict(
            conflict_type=base_conflict.conflict_type,
            severity=base_conflict.severity,
            group_id=base_conflict.group_id,
            staff_id=base_conflict.staff_id,
            time_slot=base_conflict.time_slot,
            description=consolidated_description,
            suggested_solutions=list(all_solutions)
        )


def debug_schedule_constraints(request_data):
    """
    Comprehensive debugging function to trace what's happening
    Fixed to handle UUID objects properly
    """

    print("="*80)
    print("🔍 COMPREHENSIVE SCHEDULE DEBUGGING")
    print("="*80)

    # Helper function to safely convert UUID to string
    def safe_uuid_str(uuid_obj):
        if isinstance(uuid_obj, UUID):
            return str(uuid_obj)
        elif isinstance(uuid_obj, str):
            return uuid_obj
        else:
            return str(uuid_obj)

    # Helper function to safely slice UUID strings
    def safe_uuid_slice(uuid_obj, length=8):
        uuid_str = safe_uuid_str(uuid_obj)
        return uuid_str[:length] if len(uuid_str) >= length else uuid_str

    # Parse the data
    if isinstance(request_data.get("week_start_date"), str):
        week_start_date = datetime.strptime(request_data["week_start_date"], "%Y-%m-%d").date()
    else:
        week_start_date = request_data["week_start_date"]

    center_id = safe_uuid_str(request_data["center_id"])
    staff = request_data["staff"]
    groups = request_data["groups"]
    requirements = request_data["staffing_requirements"]

    print("\n📅 DATE ANALYSIS")
    print(f"Center ID: {center_id[:8]}...")
    print(f"Week start date: {week_start_date}")
    print(f"Day of week: {week_start_date.weekday()} ({week_start_date.strftime('%A')})")
    print("Expected: Monday (0)")

    if week_start_date.weekday() != 0:
        print("⚠️  WARNING: Week start is not Monday!")

    # Calculate end date
    week_end_date = week_start_date + timedelta(days=6)
    print(f"Week end date: {week_end_date}")

    print(f"\n👥 STAFF ANALYSIS ({len(staff)} staff members)")
    print("-" * 40)
    for i, s in enumerate(staff):
        staff_id = safe_uuid_slice(s['staff_id'])
        staff_name = s.get('name', 'Unknown')
        print(f"Staff {i+1}: {staff_name} ({staff_id}...)")

        # Check qualifications
        quals = []
        if 'qualifications' in s:
            for q in s['qualifications']:
                if isinstance(q, dict):
                    if q.get('is_verified', False):
                        quals.append(q.get('qualification_name', 'Unknown'))
                else:
                    # Handle Pydantic model objects
                    if hasattr(q, 'is_verified') and q.is_verified:
                        quals.append(getattr(q, 'qualification_name', 'Unknown'))

        print(f"  ✓ Verified qualifications: {quals}")

        # Check availability
        if 'availability' in s:
            for avail in s['availability']:
                if isinstance(avail, dict):
                    day_of_week = avail.get('day_of_week', 0)
                    start_time = avail.get('start_time', 'Unknown')
                    end_time = avail.get('end_time', 'Unknown')
                    is_available = avail.get('is_available', False)
                else:
                    # Handle Pydantic model objects
                    day_of_week = getattr(avail, 'day_of_week', 0)
                    start_time = getattr(avail, 'start_time', 'Unknown')
                    end_time = getattr(avail, 'end_time', 'Unknown')
                    is_available = getattr(avail, 'is_available', False)

                day_name = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][day_of_week]

                # Convert times to string for display
                if hasattr(start_time, 'hour'):
                    start_time_display = str(start_time)
                else:
                    start_time_display = str(start_time)

                if hasattr(end_time, 'hour'):
                    end_time_display = str(end_time)
                else:
                    end_time_display = str(end_time)

                status = "✓ Available" if is_available else "✗ Not available"
                print(f"  {status} on {day_name} ({day_of_week}) from {start_time_display} to {end_time_display}")

        max_hours = s.get('max_weekly_hours', 'Unknown')
        print(f"  Max weekly hours: {max_hours}")
        print()

    print(f"\n🏢 GROUPS ANALYSIS ({len(groups)} groups)")
    print("-" * 40)
    for group in groups:
        group_id = safe_uuid_slice(group['group_id'])
        group_name = group.get('name', 'Unknown')
        print(f"Group: {group_name} ({group_id}...)")

        required_quals = group.get('required_qualifications', [])
        print(f"  Required qualifications: {required_quals}")
        print()

    print(f"\n📋 REQUIREMENTS ANALYSIS ({len(requirements)} requirements)")
    print("-" * 40)
    group_requirements = defaultdict(list)

    for i, req in enumerate(requirements):
        print(f"Requirement {i+1}:")
        req_group_id = safe_uuid_slice(req['group_id'])
        print(f"  Group: {req_group_id}...")

        # Handle time_slot properly
        time_slot = req.get('time_slot', {})
        if isinstance(time_slot, dict):
            day_of_week = time_slot.get('day_of_week', 0)
            start_time = time_slot.get('start_time', 'Unknown')
            end_time = time_slot.get('end_time', 'Unknown')
        else:
            # Handle Pydantic model objects
            day_of_week = getattr(time_slot, 'day_of_week', 0)
            start_time = getattr(time_slot, 'start_time', 'Unknown')
            end_time = getattr(time_slot, 'end_time', 'Unknown')

        day_name = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][day_of_week]
        print(f"  Day: {day_of_week} ({day_name})")

        # Convert times to string for display
        if hasattr(start_time, 'hour'):
            start_time_display = str(start_time)
        else:
            start_time_display = str(start_time)

        if hasattr(end_time, 'hour'):
            end_time_display = str(end_time)
        else:
            end_time_display = str(end_time)

        print(f"  Time: {start_time_display} - {end_time_display}")

        min_staff = req.get('min_staff_count', 0)
        max_staff = req.get('max_staff_count', 'unlimited')
        print(f"  Staff needed: {min_staff} - {max_staff}")

        required_quals = req.get('required_qualifications', [])
        print(f"  Required qualifications: {required_quals}")

        # Use string version of group_id for grouping
        group_id_str = safe_uuid_str(req['group_id'])
        group_requirements[group_id_str].append(req)
        print()

    print("\n🔄 OVERLAP ANALYSIS")
    print("-" * 40)
    for group_id_str, group_reqs in group_requirements.items():
        if len(group_reqs) > 1:
            group_id_short = group_id_str[:8]
            print(f"Group {group_id_short}... has {len(group_reqs)} requirements:")

            # Analyze overlaps
            for i, req1 in enumerate(group_reqs):
                for j, req2 in enumerate(group_reqs[i+1:], i+1):
                    # Extract time slots
                    ts1 = req1.get('time_slot', {})
                    ts2 = req2.get('time_slot', {})

                    if isinstance(ts1, dict) and isinstance(ts2, dict):
                        dow1 = ts1.get('day_of_week', 0)
                        dow2 = ts2.get('day_of_week', 0)
                    else:
                        dow1 = getattr(ts1, 'day_of_week', 0)
                        dow2 = getattr(ts2, 'day_of_week', 0)

                    if dow1 == dow2:
                        overlap = find_time_overlap(ts1, ts2)
                        if overlap:
                            print(f"  📍 OVERLAP between req {i+1} and req {j+1}:")
                            print(f"     Time: {overlap['start']} - {overlap['end']}")
                            min1 = req1.get('min_staff_count', 0)
                            min2 = req2.get('min_staff_count', 0)
                            print(f"     Max min_staff: {max(min1, min2)}")

                            max1 = req1.get('max_staff_count')
                            max2 = req2.get('max_staff_count')
                            if max1 is not None and max2 is not None:
                                print(f"     Min max_staff: {min(max1, max2)}")
                        else:
                            print(f"  No overlap between req {i+1} and req {j+1}")

    print("\n🔧 VARIABLE CREATION SIMULATION")
    print("-" * 40)

    # Simulate the day mapping
    total_days = 7
    dow_to_offsets = {}

    print("Day mapping for the week:")
    for day_offset in range(total_days):
        actual_date = week_start_date + timedelta(days=day_offset)
        actual_dow = actual_date.weekday()

        if actual_dow not in dow_to_offsets:
            dow_to_offsets[actual_dow] = []
        dow_to_offsets[actual_dow].append(day_offset)

        print(f"  Day offset {day_offset} = {actual_date} = DOW {actual_dow} ({actual_date.strftime('%A')})")

    print(f"\nDOW to offsets mapping: {dow_to_offsets}")

    # Simulate variable creation
    print("\nSimulating variable creation:")
    variables_created = 0
    slots_needing_variables = set()

    # Collect all slots that need variables
    for req in requirements:
        time_slot = req.get('time_slot', {})
        group_id = safe_uuid_str(req['group_id'])

        if isinstance(time_slot, dict):
            req_dow = time_slot.get('day_of_week', 0)
            start_time_obj = time_slot.get('start_time', '00:00:00')
            end_time_obj = time_slot.get('end_time', '00:00:00')
        else:
            req_dow = getattr(time_slot, 'day_of_week', 0)
            start_time_obj = getattr(time_slot, 'start_time', '00:00:00')
            end_time_obj = getattr(time_slot, 'end_time', '00:00:00')

        # Parse hours - handle both string and datetime.time objects
        try:
            if hasattr(start_time_obj, 'hour'):
                # It's a datetime.time object
                start_hour = start_time_obj.hour
            else:
                # It's a string
                start_hour = int(str(start_time_obj).split(':')[0])

            if hasattr(end_time_obj, 'hour'):
                # It's a datetime.time object
                end_hour = end_time_obj.hour
            else:
                # It's a string
                end_hour = int(str(end_time_obj).split(':')[0])
        except (ValueError, IndexError, AttributeError):
            print(f"  ⚠️  WARNING: Could not parse times: {start_time_obj} - {end_time_obj}")
            continue

        print(f"\nRequirement on DOW {req_dow} from hour {start_hour} to {end_hour}:")

        if req_dow in dow_to_offsets:
            for day_offset in dow_to_offsets[req_dow]:
                for hour in range(start_hour, end_hour):
                    slots_needing_variables.add((group_id, day_offset, hour))
                    print(f"  Slot needed: group={group_id[:8]}..., day_offset={day_offset}, hour={hour}")
        else:
            print(f"  ⚠️  WARNING: DOW {req_dow} not found in mapping!")

    print(f"\nTotal unique slots needing variables: {len(slots_needing_variables)}")

    # Check which staff could have variables for each slot
    print("\nChecking staff availability for each slot:")
    problem_slots = []

    for group_id, day_offset, hour in sorted(slots_needing_variables):
        actual_date = week_start_date + timedelta(days=day_offset)
        actual_dow = actual_date.weekday()

        print(f"\nSlot: group={group_id[:8]}..., day_offset={day_offset}, hour={hour} (DOW {actual_dow})")

        # Find applicable requirements
        applicable_reqs = []
        for req in requirements:
            req_group_id = safe_uuid_str(req['group_id'])
            time_slot = req.get('time_slot', {})

            if isinstance(time_slot, dict):
                req_dow = time_slot.get('day_of_week', 0)
                start_time_obj = time_slot.get('start_time', '00:00:00')
                end_time_obj = time_slot.get('end_time', '00:00:00')
            else:
                req_dow = getattr(time_slot, 'day_of_week', 0)
                start_time_obj = getattr(time_slot, 'start_time', '00:00:00')
                end_time_obj = getattr(time_slot, 'end_time', '00:00:00')

            try:
                if hasattr(start_time_obj, 'hour'):
                    start_hour = start_time_obj.hour
                else:
                    start_hour = int(str(start_time_obj).split(':')[0])

                if hasattr(end_time_obj, 'hour'):
                    end_hour = end_time_obj.hour
                else:
                    end_hour = int(str(end_time_obj).split(':')[0])
            except (ValueError, IndexError, AttributeError):
                continue

            if (req_group_id == group_id and
                req_dow == actual_dow and
                start_hour <= hour < end_hour):
                applicable_reqs.append(req)

        print(f"  Applicable requirements: {len(applicable_reqs)}")

        # Check required qualifications
        all_required_quals = set()
        for req in applicable_reqs:
            required_quals = req.get('required_qualifications', [])
            all_required_quals.update(required_quals)
        print(f"  Required qualifications: {list(all_required_quals)}")

        # Check each staff member
        available_staff = []
        for s in staff:
            staff_name = s.get('name', 'Unknown')

            # Check qualifications
            staff_quals = set()
            if 'qualifications' in s:
                for q in s['qualifications']:
                    if isinstance(q, dict):
                        if q.get('is_verified', False):
                            staff_quals.add(q.get('qualification_name', ''))
                    else:
                        if hasattr(q, 'is_verified') and q.is_verified:
                            staff_quals.add(getattr(q, 'qualification_name', ''))

            has_quals = all_required_quals.issubset(staff_quals)

            # Check availability
            is_available = False
            if 'availability' in s:
                for avail in s['availability']:
                    if isinstance(avail, dict):
                        avail_dow = avail.get('day_of_week', 0)
                        avail_available = avail.get('is_available', False)
                        start_time_str = avail.get('start_time', '00:00:00')
                        end_time_str = avail.get('end_time', '00:00:00')
                    else:
                        avail_dow = getattr(avail, 'day_of_week', 0)
                        avail_available = getattr(avail, 'is_available', False)
                        start_time_str = str(getattr(avail, 'start_time', '00:00:00'))
                        end_time_str = str(getattr(avail, 'end_time', '00:00:00'))

                    try:
                        if hasattr(start_time_str, 'hour'):
                            avail_start_hour = start_time_str.hour
                        else:
                            avail_start_hour = int(str(start_time_str).split(':')[0])

                        if hasattr(end_time_str, 'hour'):
                            avail_end_hour = end_time_str.hour
                        else:
                            avail_end_hour = int(str(end_time_str).split(':')[0])
                    except (ValueError, IndexError, AttributeError):
                        continue

                    if (avail_dow == actual_dow and
                        avail_available and
                        avail_start_hour <= hour < avail_end_hour):
                        is_available = True
                        break

            if has_quals and is_available:
                available_staff.append(staff_name)
                variables_created += 1
                print(f"    ✓ {staff_name} - variable would be created")
            else:
                reason = []
                if not has_quals:
                    missing = all_required_quals - staff_quals
                    reason.append(f"missing quals: {list(missing)}")
                if not is_available:
                    reason.append("not available")
                print(f"    ✗ {staff_name} - {'; '.join(reason)}")

        print(f"  Available staff count: {len(available_staff)}")

        # Check if this meets requirements
        min_needed = 0
        if applicable_reqs:
            min_needed = max(req.get('min_staff_count', 0) for req in applicable_reqs)

        if len(available_staff) < min_needed:
            print(f"  ⚠️  PROBLEM: Only {len(available_staff)} available but {min_needed} needed!")
            problem_slots.append({
                'group_id': group_id[:8],
                'day_offset': day_offset,
                'hour': hour,
                'available': len(available_staff),
                'needed': min_needed
            })
        else:
            print(f"  ✓ OK: {len(available_staff)} available, {min_needed} needed")

    print("\n📊 SUMMARY")
    print("-" * 40)
    print(f"Total variables that would be created: {variables_created}")
    print(f"Total unique time slots: {len(slots_needing_variables)}")
    print(f"Staff members: {len(staff)}")
    print(f"Groups: {len(groups)}")
    print(f"Requirements: {len(requirements)}")

    if problem_slots:
        print(f"\n❌ PROBLEM SLOTS FOUND ({len(problem_slots)}):")
        for slot in problem_slots:
            print(f"  Group {slot['group_id']}..., day {slot['day_offset']}, hour {slot['hour']}: {slot['available']} available, {slot['needed']} needed")
    else:
        print("\n✅ All slots have sufficient staff availability")

    print("="*80)

    return {
        'variables_would_be_created': variables_created,
        'unique_slots': len(slots_needing_variables),
        'dow_mapping': dow_to_offsets,
        'problem_slots': problem_slots
    }

def find_time_overlap(slot1, slot2):
    """Find overlap between two time slots (handles both dict and object formats)"""
    def time_to_minutes(time_obj):
        if hasattr(time_obj, 'hour'):
            # It's a datetime.time object
            return time_obj.hour * 60 + time_obj.minute
        elif isinstance(time_obj, str):
            time_str = time_obj
        else:
            time_str = str(time_obj)

        try:
            hours, minutes = map(int, time_str.split(':')[:2])
            return hours * 60 + minutes
        except (ValueError, IndexError, AttributeError):
            return 0

    def get_time_field(slot, field):
        if isinstance(slot, dict):
            time_obj = slot.get(field, '00:00:00')
        else:
            time_obj = getattr(slot, field, '00:00:00')

        # Return the time object as-is (will be handled by time_to_minutes)
        return time_obj

    start1 = time_to_minutes(get_time_field(slot1, 'start_time'))
    end1 = time_to_minutes(get_time_field(slot1, 'end_time'))
    start2 = time_to_minutes(get_time_field(slot2, 'start_time'))
    end2 = time_to_minutes(get_time_field(slot2, 'end_time'))

    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)

    if overlap_start < overlap_end:
        def minutes_to_time(minutes):
            hours = minutes // 60
            mins = minutes % 60
            return f"{hours:02d}:{mins:02d}:00"

        return {
            'start': minutes_to_time(overlap_start),
            'end': minutes_to_time(overlap_end)
        }

    return None
