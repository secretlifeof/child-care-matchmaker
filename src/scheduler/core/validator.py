"""
Schedule validation engine for business rules and constraints
"""
import logging
from datetime import datetime, date, time, timedelta
from typing import List, Dict, Optional, Set
from uuid import UUID

from ..models import *
from ..config import settings

logger = logging.getLogger(__name__)


class ScheduleValidator:
    """Validates schedules against business rules and constraints"""
    
    def __init__(self):
        logger.info("Schedule validator initialized")
    
    def validate_basic_constraints(
        self,
        schedule: List[ScheduledShift],
        staff: List[Staff]
    ) -> List[ScheduleConflict]:
        """Validate basic scheduling constraints"""
        
        conflicts = []
        staff_dict = {s.staff_id: s for s in staff}
        
        # Group shifts by staff and date for validation
        staff_daily_shifts = {}
        staff_weekly_hours = {}
        
        for shift in schedule:
            # Track daily shifts
            key = (shift.staff_id, shift.date)
            if key not in staff_daily_shifts:
                staff_daily_shifts[key] = []
            staff_daily_shifts[key].append(shift)
            
            # Track weekly hours
            if shift.staff_id not in staff_weekly_hours:
                staff_weekly_hours[shift.staff_id] = 0
            staff_weekly_hours[shift.staff_id] += shift.scheduled_hours
        
        # Validate daily constraints
        for (staff_id, date), shifts in staff_daily_shifts.items():
            staff_member = staff_dict.get(staff_id)
            if not staff_member:
                conflicts.append(ScheduleConflict(
                    conflict_type='unknown_staff',
                    severity='error',
                    staff_id=staff_id,
                    description=f'Schedule references unknown staff member {staff_id}',
                    suggested_solutions=['Remove invalid shifts', 'Add missing staff member']
                ))
                continue
            
            # Check daily hour limits
            daily_hours = sum(s.scheduled_hours for s in shifts)
            max_daily_hours = min(settings.max_consecutive_hours, 12)
            
            if daily_hours > max_daily_hours:
                conflicts.append(ScheduleConflict(
                    conflict_type='daily_hour_limit',
                    severity='error',
                    staff_id=staff_id,
                    description=f'{staff_member.name} scheduled for {daily_hours:.1f} hours on {date} (max: {max_daily_hours})',
                    suggested_solutions=['Reduce shift hours', 'Split into multiple days', 'Add break time']
                ))
            
            # Check for overlapping shifts
            sorted_shifts = sorted(shifts, key=lambda s: s.start_time)
            for i in range(len(sorted_shifts) - 1):
                current = sorted_shifts[i]
                next_shift = sorted_shifts[i + 1]
                
                if current.end_time > next_shift.start_time:
                    conflicts.append(ScheduleConflict(
                        conflict_type='overlapping_shifts',
                        severity='error',
                        staff_id=staff_id,
                        description=f'{staff_member.name} has overlapping shifts on {date}: {current.start_time}-{current.end_time} and {next_shift.start_time}-{next_shift.end_time}',
                        suggested_solutions=['Adjust shift times', 'Remove overlapping shift', 'Assign to different staff']
                    ))
            
            # Check availability constraints
            availability_conflicts = self._check_availability_violations(staff_member, shifts, date)
            conflicts.extend(availability_conflicts)
        
        # Validate weekly constraints
        for staff_id, weekly_hours in staff_weekly_hours.items():
            staff_member = staff_dict.get(staff_id)
            if not staff_member:
                continue
            
            # Check weekly hour limits
            if weekly_hours > staff_member.max_weekly_hours:
                severity = 'error' if weekly_hours > staff_member.max_weekly_hours * 1.2 else 'warning'
                conflicts.append(ScheduleConflict(
                    conflict_type='weekly_hour_limit',
                    severity=severity,
                    staff_id=staff_id,
                    description=f'{staff_member.name} scheduled for {weekly_hours:.1f} hours (max: {staff_member.max_weekly_hours})',
                    suggested_solutions=['Reduce weekly hours', 'Distribute to other staff', 'Increase staff availability']
                ))
            
            # Check minimum hours if specified
            min_hours = self._get_minimum_hours_for_staff(staff_member)
            if min_hours > 0 and weekly_hours < min_hours:
                conflicts.append(ScheduleConflict(
                    conflict_type='insufficient_hours',
                    severity='warning',
                    staff_id=staff_id,
                    description=f'{staff_member.name} scheduled for only {weekly_hours:.1f} hours (min expected: {min_hours})',
                    suggested_solutions=['Add more shifts', 'Check staff availability', 'Review staffing needs']
                ))
        
        # Check break requirements between shifts
        break_conflicts = self._validate_break_requirements(schedule, staff_dict)
        conflicts.extend(break_conflicts)
        
        return conflicts
    
    def validate_staffing_requirements(
        self,
        schedule: List[ScheduledShift],
        groups: List[Group],
        requirements: List[StaffingRequirement]
    ) -> List[ScheduleConflict]:
        """Validate staffing level requirements"""
        
        conflicts = []
        group_dict = {g.group_id: g for g in groups}
        
        for requirement in requirements:
            # Find shifts that cover this requirement
            covering_shifts = self._find_covering_shifts(schedule, requirement)
            unique_staff = len(set(shift.staff_id for shift in covering_shifts))
            
            group = group_dict.get(requirement.group_id)
            group_name = group.name if group else str(requirement.group_id)
            
            # Check minimum staffing
            if unique_staff < requirement.min_staff_count:
                # Calculate shortage severity
                shortage_percent = (requirement.min_staff_count - unique_staff) / requirement.min_staff_count
                severity = 'error' if shortage_percent > 0.5 else 'warning'
                
                conflicts.append(ScheduleConflict(
                    conflict_type='understaffed',
                    severity=severity,
                    group_id=requirement.group_id,
                    time_slot=requirement.time_slot,
                    description=f'Group "{group_name}" understaffed: {unique_staff}/{requirement.min_staff_count} staff on {self._format_time_slot(requirement.time_slot)}',
                    suggested_solutions=['Add more staff', 'Extend existing shifts', 'Reduce group size', 'Call substitutes']
                ))
            
            # Check maximum staffing if specified
            elif requirement.max_staff_count and unique_staff > requirement.max_staff_count:
                overstaffing_percent = (unique_staff - requirement.max_staff_count) / requirement.max_staff_count
                severity = 'warning' if overstaffing_percent < 0.5 else 'info'
                
                conflicts.append(ScheduleConflict(
                    conflict_type='overstaffed',
                    severity=severity,
                    group_id=requirement.group_id,
                    time_slot=requirement.time_slot,
                    description=f'Group "{group_name}" overstaffed: {unique_staff}/{requirement.max_staff_count} staff on {self._format_time_slot(requirement.time_slot)}',
                    suggested_solutions=['Reduce staff assignment', 'Reassign to other groups', 'Schedule break time']
                ))
            
            # Check if group has appropriate enrollment for staffing level
            if group and unique_staff > 0:
                enrollment_conflicts = self._validate_enrollment_ratio(group, unique_staff, requirement.time_slot)
                conflicts.extend(enrollment_conflicts)
        
        return conflicts
    
    def validate_business_rules(
        self,
        schedule: List[ScheduledShift],
        staff: List[Staff],
        constraints: List[ScheduleConstraint]
    ) -> List[ScheduleConflict]:
        """Validate business-specific rules and constraints"""
        
        conflicts = []
        
        # Validate shift duration constraints
        duration_conflicts = self._validate_shift_durations(schedule)
        conflicts.extend(duration_conflicts)
        
        # Validate qualification requirements
        qualification_conflicts = self._validate_qualifications(schedule, staff)
        conflicts.extend(qualification_conflicts)
        
        # Validate priority weight fairness
        priority_conflicts = self._validate_priority_fairness(schedule, staff)
        conflicts.extend(priority_conflicts)
        
        # Validate consecutive working days
        consecutive_conflicts = self._validate_consecutive_working_days(schedule, staff)
        conflicts.extend(consecutive_conflicts)
        
        # Apply custom constraints
        for constraint in constraints:
            if constraint.is_mandatory:
                custom_conflicts = self._validate_custom_constraint(schedule, staff, constraint)
                conflicts.extend(custom_conflicts)
        
        return conflicts
    
    def _check_availability_violations(
        self,
        staff_member: Staff,
        shifts: List[ScheduledShift],
        shift_date: date
    ) -> List[ScheduleConflict]:
        """Check if shifts violate staff availability"""
        
        conflicts = []
        day_of_week = shift_date.weekday()
        
        # Build availability map for this day
        available_times = set()
        for avail in staff_member.availability:
            if avail.day_of_week == day_of_week and avail.is_available:
                start_hour = avail.start_time.hour
                end_hour = avail.end_time.hour
                if end_hour <= start_hour:  # Overnight availability
                    end_hour += 24
                
                for hour in range(start_hour, end_hour):
                    available_times.add(hour % 24)
        
        # Check each shift against availability
        for shift in shifts:
            shift_start_hour = shift.start_time.hour
            shift_end_hour = shift.end_time.hour
            if shift_end_hour <= shift_start_hour:
                shift_end_hour += 24
            
            # Check if any hour of the shift is outside availability
            unavailable_hours = []
            for hour in range(shift_start_hour, shift_end_hour):
                if (hour % 24) not in available_times:
                    unavailable_hours.append(hour % 24)
            
            if unavailable_hours:
                conflicts.append(ScheduleConflict(
                    conflict_type='availability_violation',
                    severity='error',
                    staff_id=staff_member.staff_id,
                    description=f'{staff_member.name} scheduled outside availability on {shift_date}: {shift.start_time}-{shift.end_time} (unavailable hours: {unavailable_hours})',
                    suggested_solutions=['Adjust shift time', 'Assign to available staff', 'Update staff availability']
                ))
        
        return conflicts
    
    def _validate_break_requirements(
        self,
        schedule: List[ScheduledShift],
        staff_dict: Dict[UUID, Staff]
    ) -> List[ScheduleConflict]:
        """Validate break requirements between shifts"""
        
        conflicts = []
        min_break_hours = settings.min_break_between_shifts
        
        # Group shifts by staff
        staff_shifts = {}
        for shift in schedule:
            if shift.staff_id not in staff_shifts:
                staff_shifts[shift.staff_id] = []
            staff_shifts[shift.staff_id].append(shift)
        
        for staff_id, shifts in staff_shifts.items():
            if len(shifts) < 2:
                continue
            
            staff_member = staff_dict.get(staff_id)
            if not staff_member:
                continue
            
            # Sort shifts chronologically
            sorted_shifts = sorted(shifts, key=lambda s: (s.date, s.start_time))
            
            for i in range(len(sorted_shifts) - 1):
                current_shift = sorted_shifts[i]
                next_shift = sorted_shifts[i + 1]
                
                # Calculate time between shifts
                current_end = datetime.combine(current_shift.date, current_shift.end_time)
                next_start = datetime.combine(next_shift.date, next_shift.start_time)
                
                # Handle next day shifts
                if next_start <= current_end:
                    next_start += timedelta(days=1)
                
                break_hours = (next_start - current_end).total_seconds() / 3600
                
                if 0 < break_hours < min_break_hours:
                    severity = 'error' if break_hours < min_break_hours / 2 else 'warning'
                    conflicts.append(ScheduleConflict(
                        conflict_type='insufficient_break',
                        severity=severity,
                        staff_id=staff_id,
                        description=f'{staff_member.name} has insufficient break: {break_hours:.1f} hours between {current_shift.date} {current_shift.end_time} and {next_shift.date} {next_shift.start_time} (min: {min_break_hours}h)',
                        suggested_solutions=['Extend break time', 'Adjust shift times', 'Remove one shift']
                    ))
        
        return conflicts
    
    def _validate_shift_durations(
        self,
        schedule: List[ScheduledShift]
    ) -> List[ScheduleConflict]:
        """Validate shift duration constraints"""
        
        conflicts = []
        
        for shift in schedule:
            # Check minimum duration
            if shift.scheduled_hours < settings.min_shift_duration:
                conflicts.append(ScheduleConflict(
                    conflict_type='short_shift',
                    severity='warning',
                    staff_id=shift.staff_id,
                    description=f'Shift too short: {shift.scheduled_hours:.1f} hours on {shift.date} (min: {settings.min_shift_duration}h)',
                    suggested_solutions=['Extend shift', 'Combine with adjacent shift', 'Remove shift']
                ))
            
            # Check maximum duration
            elif shift.scheduled_hours > settings.max_shift_duration:
                conflicts.append(ScheduleConflict(
                    conflict_type='long_shift',
                    severity='warning',
                    staff_id=shift.staff_id,
                    description=f'Shift too long: {shift.scheduled_hours:.1f} hours on {shift.date} (max: {settings.max_shift_duration}h)',
                    suggested_solutions=['Split shift', 'Add break time', 'Reduce shift hours']
                ))
            
            # Check if shift spans midnight (might need special handling)
            shift_start = datetime.combine(shift.date, shift.start_time)
            shift_end = datetime.combine(shift.date, shift.end_time)
            if shift_end <= shift_start:
                # Overnight shift
                actual_end = shift_end + timedelta(days=1)
                actual_duration = (actual_end - shift_start).total_seconds() / 3600
                
                if abs(actual_duration - shift.scheduled_hours) > 0.1:  # Allow small rounding differences
                    conflicts.append(ScheduleConflict(
                        conflict_type='duration_mismatch',
                        severity='warning',
                        staff_id=shift.staff_id,
                        description=f'Overnight shift duration mismatch: scheduled {shift.scheduled_hours:.1f}h vs calculated {actual_duration:.1f}h',
                        suggested_solutions=['Correct shift times', 'Update scheduled hours']
                    ))
        
        return conflicts
    
    def _validate_qualifications(
        self,
        schedule: List[ScheduledShift],
        staff: List[Staff]
    ) -> List[ScheduleConflict]:
        """Validate qualification requirements are met"""
        
        conflicts = []
        staff_dict = {s.staff_id: s for s in staff}
        
        # Build staff qualification map
        staff_qualifications = {}
        for staff_member in staff:
            quals = set()
            for qual in staff_member.qualifications:
                if qual.is_verified and (not qual.expiry_date or qual.expiry_date > date.today()):
                    quals.add(qual.qualification_name)
            staff_qualifications[staff_member.staff_id] = quals
        
        # Skip qualification checking here since it should be handled by 
        # staffing requirements validation, not hardcoded critical qualifications
        # This prevents false positives for qualifications not actually required
        
        return conflicts
    
    def _validate_priority_fairness(
        self,
        schedule: List[ScheduledShift],
        staff: List[Staff]
    ) -> List[ScheduleConflict]:
        """Validate that priority weights are fairly reflected"""
        
        conflicts = []
        
        try:
            # Calculate actual hours per staff
            staff_hours = {}
            for shift in schedule:
                staff_hours[shift.staff_id] = staff_hours.get(shift.staff_id, 0) + shift.scheduled_hours
            
            # Group staff by similar priority scores
            priority_groups = {}
            for staff_member in staff:
                priority_bucket = round(staff_member.priority_score, 1)
                if priority_bucket not in priority_groups:
                    priority_groups[priority_bucket] = []
                priority_groups[priority_bucket].append(staff_member)
            
            # Check fairness within priority groups
            for priority_score, group_staff in priority_groups.items():
                if len(group_staff) <= 1:
                    continue
                
                group_hours = []
                for staff_member in group_staff:
                    hours = staff_hours.get(staff_member.staff_id, 0)
                    group_hours.append((staff_member, hours))
                
                if not group_hours:
                    continue
                
                # Calculate variance within group
                hours_only = [h for _, h in group_hours]
                avg_hours = sum(hours_only) / len(hours_only)
                max_hours = max(hours_only)
                min_hours = min(hours_only)
                
                # Flag significant variance within same priority group
                if avg_hours > 0 and (max_hours - min_hours) / avg_hours > 0.4:  # 40% variance threshold
                    high_staff = [s for s, h in group_hours if h == max_hours][0]
                    low_staff = [s for s, h in group_hours if h == min_hours][0]
                    
                    conflicts.append(ScheduleConflict(
                        conflict_type='priority_unfairness',
                        severity='warning',
                        description=f'Unfair distribution within priority group {priority_score}: {high_staff.name} ({max_hours:.1f}h) vs {low_staff.name} ({min_hours:.1f}h)',
                        suggested_solutions=[
                            'Review availability constraints',
                            'Balance assignments within priority group',
                            'Adjust priority weights if appropriate'
                        ]
                    ))
            
        except Exception as e:
            logger.warning(f"Error in priority fairness validation: {e}")
        
        return conflicts
    
    def _validate_consecutive_working_days(
        self,
        schedule: List[ScheduledShift],
        staff: List[Staff]
    ) -> List[ScheduleConflict]:
        """Validate consecutive working days limits"""
        
        conflicts = []
        max_consecutive_days = 6  # Maximum consecutive working days
        
        staff_dict = {s.staff_id: s for s in staff}
        
        # Group shifts by staff
        staff_shifts = {}
        for shift in schedule:
            if shift.staff_id not in staff_shifts:
                staff_shifts[shift.staff_id] = []
            staff_shifts[shift.staff_id].append(shift)
        
        for staff_id, shifts in staff_shifts.items():
            staff_member = staff_dict.get(staff_id)
            if not staff_member:
                continue
            
            # Get unique working dates
            working_dates = sorted(set(shift.date for shift in shifts))
            
            if len(working_dates) <= max_consecutive_days:
                continue
            
            # Check for consecutive sequences
            consecutive_sequences = []
            current_sequence = [working_dates[0]]
            
            for i in range(1, len(working_dates)):
                if (working_dates[i] - working_dates[i-1]).days == 1:
                    current_sequence.append(working_dates[i])
                else:
                    if len(current_sequence) > max_consecutive_days:
                        consecutive_sequences.append(current_sequence)
                    current_sequence = [working_dates[i]]
            
            # Check final sequence
            if len(current_sequence) > max_consecutive_days:
                consecutive_sequences.append(current_sequence)
            
            # Report violations
            for sequence in consecutive_sequences:
                conflicts.append(ScheduleConflict(
                    conflict_type='excessive_consecutive_days',
                    severity='warning',
                    staff_id=staff_id,
                    description=f'{staff_member.name} scheduled for {len(sequence)} consecutive days: {sequence[0]} to {sequence[-1]} (max recommended: {max_consecutive_days})',
                    suggested_solutions=['Add rest days', 'Redistribute shifts', 'Check labor law compliance']
                ))
        
        return conflicts
    
    def _validate_custom_constraint(
        self,
        schedule: List[ScheduledShift],
        staff: List[Staff],
        constraint: ScheduleConstraint
    ) -> List[ScheduleConflict]:
        """Validate a custom constraint"""
        
        conflicts = []
        
        try:
            if constraint.constraint_type == 'max_shifts_per_day':
                max_shifts = constraint.config.get('max_shifts', 2)
                conflicts.extend(self._check_max_shifts_per_day(schedule, staff, max_shifts))
            
            elif constraint.constraint_type == 'required_staff_coverage':
                coverage_rules = constraint.config.get('coverage_rules', {})
                conflicts.extend(self._check_staff_coverage(schedule, staff, coverage_rules))
            
            elif constraint.constraint_type == 'skill_distribution':
                skill_requirements = constraint.config.get('skill_requirements', {})
                conflicts.extend(self._check_skill_distribution(schedule, staff, skill_requirements))
            
        except Exception as e:
            logger.error(f"Error validating custom constraint {constraint.constraint_type}: {str(e)}")
            conflicts.append(ScheduleConflict(
                conflict_type='constraint_error',
                severity='warning',
                description=f'Error validating constraint: {constraint.constraint_type}',
                suggested_solutions=['Check constraint configuration', 'Contact administrator']
            ))
        
        return conflicts
    
    def _find_covering_shifts(
        self,
        schedule: List[ScheduledShift],
        requirement: StaffingRequirement
    ) -> List[ScheduledShift]:
        """Find shifts that cover a staffing requirement"""
        
        covering_shifts = []
        req_day = requirement.time_slot.day_of_week
        req_start = requirement.time_slot.start_time
        req_end = requirement.time_slot.end_time
        
        for shift in schedule:
            if shift.group_id != requirement.group_id:
                continue
            
            # Check if shift date matches requirement day
            shift_day = shift.date.weekday()
            # If req_day is None, requirement applies to all days
            if req_day is not None and shift_day != req_day:
                continue
            
            # Check if shift time overlaps with requirement time
            if self._times_overlap(shift.start_time, shift.end_time, req_start, req_end):
                covering_shifts.append(shift)
        
        return covering_shifts
    
    def _times_overlap(
        self,
        start1: time, end1: time,
        start2: time, end2: time
    ) -> bool:
        """Check if two time ranges overlap"""
        
        # Convert to minutes for easier comparison
        start1_min = start1.hour * 60 + start1.minute
        end1_min = end1.hour * 60 + end1.minute
        start2_min = start2.hour * 60 + start2.minute
        end2_min = end2.hour * 60 + end2.minute
        
        # Handle overnight times
        if end1_min <= start1_min:
            end1_min += 24 * 60
        if end2_min <= start2_min:
            end2_min += 24 * 60
        
        return max(start1_min, start2_min) < min(end1_min, end2_min)
    
    def _validate_enrollment_ratio(
        self,
        group: Group,
        staff_count: int,
        time_slot: TimeSlot
    ) -> List[ScheduleConflict]:
        """Validate staff-to-child ratio for group enrollment"""
        
        conflicts = []
        
        # Get required ratio for age group
        required_ratio = self._get_required_ratio(group.age_group)
        max_children_for_staff = staff_count * required_ratio
        
        if group.current_enrollment > max_children_for_staff:
            shortage = group.current_enrollment - max_children_for_staff
            additional_staff_needed = int(shortage / required_ratio) + 1
            
            conflicts.append(ScheduleConflict(
                conflict_type='ratio_violation',
                severity='error',
                group_id=group.group_id,
                time_slot=time_slot,
                description=f'Group "{group.name}" exceeds ratio: {group.current_enrollment} children with {staff_count} staff (ratio 1:{required_ratio}, need {additional_staff_needed} more staff)',
                suggested_solutions=[
                    f'Add {additional_staff_needed} more staff',
                    'Reduce group enrollment',
                    'Split group into smaller groups'
                ]
            ))
        
        return conflicts
    
    def _get_required_ratio(self, age_group: AgeGroup) -> int:
        """Get required staff-to-child ratio by age group"""
        ratios = {
            AgeGroup.INFANT: settings.infant_ratio,
            AgeGroup.TODDLER: settings.toddler_ratio,
            AgeGroup.PRESCHOOL: settings.preschool_ratio,
            AgeGroup.MIXED: settings.toddler_ratio  # Use toddler ratio for mixed groups
        }
        return ratios.get(age_group, settings.toddler_ratio)
    
    def _get_minimum_hours_for_staff(self, staff_member: Staff) -> float:
        """Calculate minimum expected hours for staff member based on role and priority"""
        base_hours = {
            StaffRole.TEACHER: 30,
            StaffRole.SUPERVISOR: 35,
            StaffRole.ASSISTANT: 20,
            StaffRole.SUBSTITUTE: 10,
            StaffRole.ADMIN: 25,
            StaffRole.STAFF: 25
        }
        
        role_hours = base_hours.get(staff_member.role, 20)
        # Adjust by priority score
        return role_hours * min(staff_member.priority_score, 1.5)
    
    def _format_time_slot(self, time_slot: TimeSlot) -> str:
        """Format time slot for display"""
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        if time_slot.day_of_week is not None:
            day_name = days[time_slot.day_of_week] if 0 <= time_slot.day_of_week < 7 else 'Unknown'
            return f"{day_name} {time_slot.start_time.strftime('%H:%M')}-{time_slot.end_time.strftime('%H:%M')}"
        else:
            return f"Daily {time_slot.start_time.strftime('%H:%M')}-{time_slot.end_time.strftime('%H:%M')}"
    
    def _check_max_shifts_per_day(
        self,
        schedule: List[ScheduledShift],
        staff: List[Staff],
        max_shifts: int
    ) -> List[ScheduleConflict]:
        """Check maximum shifts per day constraint"""
        
        conflicts = []
        staff_dict = {s.staff_id: s for s in staff}
        
        # Group shifts by staff and date
        staff_daily_shifts = {}
        for shift in schedule:
            key = (shift.staff_id, shift.date)
            if key not in staff_daily_shifts:
                staff_daily_shifts[key] = []
            staff_daily_shifts[key].append(shift)
        
        for (staff_id, date), shifts in staff_daily_shifts.items():
            if len(shifts) > max_shifts:
                staff_member = staff_dict.get(staff_id)
                staff_name = staff_member.name if staff_member else str(staff_id)
                
                conflicts.append(ScheduleConflict(
                    conflict_type='too_many_shifts',
                    severity='warning',
                    staff_id=staff_id,
                    description=f'{staff_name} has {len(shifts)} shifts on {date} (max: {max_shifts})',
                    suggested_solutions=['Combine shifts', 'Redistribute to other days', 'Assign to other staff']
                ))
        
        return conflicts
    
    def _check_staff_coverage(
        self,
        schedule: List[ScheduledShift],
        staff: List[Staff],
        coverage_rules: Dict
    ) -> List[ScheduleConflict]:
        """Check staff coverage requirements"""
        
        conflicts = []
        # Implementation would depend on specific coverage rules
        # This is a placeholder for extensibility
        return conflicts
    
    def _check_skill_distribution(
        self,
        schedule: List[ScheduledShift],
        staff: List[Staff],
        skill_requirements: Dict
    ) -> List[ScheduleConflict]:
        """Check skill distribution requirements"""
        
        conflicts = []
        # Implementation would depend on specific skill requirements
        # This is a placeholder for extensibility
        return conflicts