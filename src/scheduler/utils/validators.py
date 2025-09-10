"""
Schedule validation utilities
"""
import logging
from datetime import date, datetime, time
from typing import Any

logger = logging.getLogger(__name__)


class ScheduleValidator:
    """Validates schedules against various constraints and business rules"""

    def __init__(self):
        self.validation_rules = {}

    def validate_basic_constraints(self, schedule: list[Any], staff: list[Any]) -> list[Any]:
        """Validate basic schedule constraints"""
        conflicts = []

        # Check for overlapping shifts for same staff member
        staff_shifts = {}
        for shift in schedule:
            staff_id = getattr(shift, 'staff_id', None)
            if staff_id not in staff_shifts:
                staff_shifts[staff_id] = []
            staff_shifts[staff_id].append(shift)

        for staff_id, shifts in staff_shifts.items():
            conflicts.extend(self._check_shift_overlaps(staff_id, shifts))

        # Check staff availability
        conflicts.extend(self._check_staff_availability(schedule, staff))

        # Check maximum hours constraints
        conflicts.extend(self._check_max_hours(schedule, staff))

        return conflicts

    def validate_staffing_requirements(
        self,
        schedule: list[Any],
        groups: list[Any],
        requirements: list[Any]
    ) -> list[Any]:
        """Enhanced validation of staffing requirements with better error context"""
        conflicts = []

        # Create group lookup for better error messages
        group_lookup = {getattr(group, 'group_id', None): getattr(group, 'name', 'Unknown Group')
                      for group in groups}

        # Group schedule by detailed time slots (date, hour, group)
        time_slot_coverage = {}
        for shift in schedule:
            shift_date = getattr(shift, 'date', None)
            start_time = getattr(shift, 'start_time', None)
            end_time = getattr(shift, 'end_time', None)
            group_id = getattr(shift, 'group_id', None)

            if shift_date and start_time and end_time:
                # Create hourly coverage entries
                start_hour = start_time.hour if hasattr(start_time, 'hour') else int(str(start_time).split(':')[0])
                end_hour = end_time.hour if hasattr(end_time, 'hour') else int(str(end_time).split(':')[0])

                for hour in range(start_hour, end_hour):
                    hour_time = datetime_time(hour, 0)
                    key = (shift_date, hour_time, group_id)
                    if key not in time_slot_coverage:
                        time_slot_coverage[key] = []
                    time_slot_coverage[key].append(shift)

        # Store group names in the validator for use in error messages
        self.group_lookup = group_lookup

        # Check each requirement with enhanced validation
        for req_idx, req in enumerate(requirements):
            logger.debug(f"Validating requirement {req_idx + 1}: {req}")
            req_conflicts = self._validate_requirement_coverage(req, time_slot_coverage)
            conflicts.extend(req_conflicts)

        return conflicts


    def validate_business_rules(
        self,
        schedule: list[Any],
        staff: list[Any],
        constraints: Any
    ) -> list[Any]:
        """Validate business rules and policies"""
        conflicts = []

        # Check minimum shift duration
        conflicts.extend(self._check_min_shift_duration(schedule))

        # Check break requirements
        conflicts.extend(self._check_break_requirements(schedule))

        # Check qualification requirements
        conflicts.extend(self._check_qualifications(schedule, staff))

        return conflicts

    def _check_shift_overlaps(self, staff_id: str, shifts: list[Any]) -> list[Any]:
        """Check for overlapping shifts for a staff member"""
        conflicts = []

        # Sort shifts by date and start time
        sorted_shifts = sorted(shifts, key=lambda s: (
            getattr(s, 'date', date.min),
            getattr(s, 'start_time', time.min)
        ))

        for i in range(len(sorted_shifts) - 1):
            current_shift = sorted_shifts[i]
            next_shift = sorted_shifts[i + 1]

            # Check if shifts overlap
            if (getattr(current_shift, 'date', None) == getattr(next_shift, 'date', None) and
                getattr(current_shift, 'end_time', None) > getattr(next_shift, 'start_time', None)):

                conflicts.append(self._create_conflict(
                    'shift_overlap',
                    'error',
                    f'Overlapping shifts for staff {staff_id}',
                    staff_id=staff_id
                ))

        return conflicts

    def _check_staff_availability(self, schedule: list[Any], staff: list[Any]) -> list[Any]:
        """Check if scheduled shifts match staff availability"""
        conflicts = []

        # Create staff availability lookup
        staff_availability = {}
        for staff_member in staff:
            staff_id = getattr(staff_member, 'staff_id', None)
            availability = getattr(staff_member, 'availability', [])
            staff_availability[staff_id] = availability

        for shift in schedule:
            staff_id = getattr(shift, 'staff_id', None)
            shift_date = getattr(shift, 'date', None)
            shift_start = getattr(shift, 'start_time', None)
            shift_end = getattr(shift, 'end_time', None)

            if not self._is_staff_available(staff_id, shift_date, shift_start, shift_end, staff_availability):
                conflicts.append(self._create_conflict(
                    'availability_conflict',
                    'error',
                    f'Staff {staff_id} not available for scheduled shift',
                    staff_id=staff_id
                ))

        return conflicts

    def _check_max_hours(self, schedule: list[Any], staff: list[Any]) -> list[Any]:
        """Check maximum hours constraints"""
        conflicts = []

        # Calculate hours per staff member
        staff_hours = {}
        for shift in schedule:
            staff_id = getattr(shift, 'staff_id', None)
            hours = getattr(shift, 'scheduled_hours', 0)

            if staff_id not in staff_hours:
                staff_hours[staff_id] = 0
            staff_hours[staff_id] += hours

        # Check against maximum hours
        staff_max_hours = {}
        for staff_member in staff:
            staff_id = getattr(staff_member, 'staff_id', None)
            max_hours = getattr(staff_member, 'max_weekly_hours', 40)
            staff_max_hours[staff_id] = max_hours

        for staff_id, actual_hours in staff_hours.items():
            max_hours = staff_max_hours.get(staff_id, 40)
            if actual_hours > max_hours:
                conflicts.append(self._create_conflict(
                    'max_hours_exceeded',
                    'warning',
                    f'Staff {staff_id} scheduled for {actual_hours} hours (max: {max_hours})',
                    staff_id=staff_id
                ))

        return conflicts

    def _validate_requirement_coverage(self, requirement: Any, time_slot_coverage: dict) -> list[Any]:
        """Validate that a specific requirement is covered with detailed error reporting"""
        conflicts = []

        group_id = getattr(requirement, 'group_id', None)
        time_slot = getattr(requirement, 'time_slot', None)
        min_staff = getattr(requirement, 'min_staff_count', 1)
        max_staff = getattr(requirement, 'max_staff_count', None)
        required_quals = getattr(requirement, 'required_qualifications', [])

        if not time_slot:
            return conflicts

        # Get time slot details
        day_of_week = getattr(time_slot, 'day_of_week', None)
        start_time = getattr(time_slot, 'start_time', None)
        end_time = getattr(time_slot, 'end_time', None)

        # Parse time range
        if start_time and end_time:
            start_hour = start_time.hour if hasattr(start_time, 'hour') else int(str(start_time).split(':')[0])
            end_hour = end_time.hour if hasattr(end_time, 'hour') else int(str(end_time).split(':')[0])
        else:
            return conflicts

        # Find the group name for better error messages
        group_name = f"Group {str(group_id)[:8]}..."  # You might want to look this up from groups list

        # Check coverage for each hour in the requirement window
        for hour in range(start_hour, end_hour):
            # Find matching shifts for this specific hour and group
            matching_shifts = []

            for key, shifts in time_slot_coverage.items():
                slot_date, slot_time, slot_group = key

                # Check if this shift matches our requirement
                if slot_group == group_id and slot_date and slot_time:
                    # Check if the shift covers this hour
                    shift_hour = slot_time.hour if hasattr(slot_time, 'hour') else int(str(slot_time).split(':')[0])
                    shift_date_dow = slot_date.weekday() if hasattr(slot_date, 'weekday') else None

                    # Match if the shift is at this hour on the right day
                    if shift_hour == hour and shift_date_dow == day_of_week:
                        matching_shifts.extend(shifts)

            # Check if we have sufficient staffing for this hour
            if len(matching_shifts) < min_staff:
                # Create detailed error message
                day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                day_name = day_names[day_of_week] if day_of_week is not None else f"Day {day_of_week}"
                time_str = f"{hour:02d}:00-{hour+1:02d}:00"

                description_parts = [
                    f"Insufficient staffing for {group_name} on {day_name} at {time_str}:",
                    f"• Found: {len(matching_shifts)} staff assigned",
                    f"• Required: {min_staff} staff minimum"
                ]

                if max_staff is not None:
                    description_parts.append(f"• Maximum allowed: {max_staff} staff")

                if required_quals:
                    description_parts.append(f"• Required qualifications: {required_quals}")

                # Add information about scheduled staff if any
                if matching_shifts:
                    staff_names = [getattr(shift, 'staff_id', 'Unknown')[:8] + "..." for shift in matching_shifts]
                    description_parts.append(f"• Currently scheduled: {staff_names}")
                else:
                    description_parts.append("• No staff currently scheduled for this time slot")

                description = "\n".join(description_parts)

                # Generate specific solutions
                solutions = []

                if len(matching_shifts) == 0:
                    solutions.extend([
                        f"Schedule staff to work on {day_name} from {time_str}",
                        "Check staff availability for this time slot",
                        "Verify staff have required qualifications",
                        "Consider hiring additional staff"
                    ])
                else:
                    shortage = min_staff - len(matching_shifts)
                    solutions.extend([
                        f"Schedule {shortage} additional staff for {day_name} {time_str}",
                        "Extend existing staff shifts to cover this time",
                        "Split longer shifts to provide coverage",
                        "Use substitute or part-time staff"
                    ])

                if required_quals:
                    solutions.append(f"Ensure staff have qualifications: {', '.join(required_quals)}")

                # Create time slot object for the conflict
                conflict_time_slot = {
                    'start_time': start_time,
                    'end_time': end_time,
                    'day_of_week': day_of_week
                }

                conflict = self._create_conflict(
                    'insufficient_staffing',
                    'error',
                    description,
                    group_id=group_id,
                    time_slot=conflict_time_slot,
                    suggested_solutions=solutions
                )
                conflicts.append(conflict)

        return conflicts

    def _check_min_shift_duration(self, schedule: list[Any]) -> list[Any]:
        """Check minimum shift duration requirements"""
        conflicts = []
        min_duration = 2.0  # Minimum 2 hours

        for shift in schedule:
            hours = getattr(shift, 'scheduled_hours', 0)
            if hours < min_duration:
                conflicts.append(self._create_conflict(
                    'short_shift',
                    'warning',
                    f'Shift duration {hours}h is below minimum {min_duration}h',
                    staff_id=getattr(shift, 'staff_id', None)
                ))

        return conflicts

    def _check_break_requirements(self, schedule: list[Any]) -> list[Any]:
        """Check break and rest period requirements"""
        conflicts = []

        # Group shifts by staff and date
        daily_shifts = {}
        for shift in schedule:
            staff_id = getattr(shift, 'staff_id', None)
            shift_date = getattr(shift, 'date', None)
            key = (staff_id, shift_date)

            if key not in daily_shifts:
                daily_shifts[key] = []
            daily_shifts[key].append(shift)

        for (staff_id, shift_date), shifts in daily_shifts.items():
            total_hours = sum(getattr(shift, 'scheduled_hours', 0) for shift in shifts)

            # Check if break is required (shifts > 6 hours)
            if total_hours > 6 and len(shifts) == 1:  # Single long shift
                conflicts.append(self._create_conflict(
                    'break_required',
                    'warning',
                    f'Staff {staff_id} needs break for {total_hours}h shift on {shift_date}',
                    staff_id=staff_id
                ))

        return conflicts

    def _check_qualifications(self, schedule: list[Any], staff: list[Any]) -> list[Any]:
        """Check staff qualification requirements"""
        conflicts = []

        # Create staff qualifications lookup
        staff_qualifications = {}
        for staff_member in staff:
            staff_id = getattr(staff_member, 'staff_id', None)
            qualifications = getattr(staff_member, 'qualifications', [])
            staff_qualifications[staff_id] = qualifications

        # Check each shift (this is simplified - would need group requirements)
        for shift in schedule:
            staff_id = getattr(shift, 'staff_id', None)
            # For now, just ensure staff has some qualifications
            if staff_id not in staff_qualifications or not staff_qualifications[staff_id]:
                conflicts.append(self._create_conflict(
                    'missing_qualifications',
                    'warning',
                    f'Staff {staff_id} may lack required qualifications',
                    staff_id=staff_id
                ))

        return conflicts

    def _is_staff_available(
        self,
        staff_id: str,
        shift_date: date,
        start_time: time,
        end_time: time,
        staff_availability: dict
    ) -> bool:
        """Check if staff is available for the given time slot"""
        if staff_id not in staff_availability:
            return False

        availability_slots = staff_availability[staff_id]
        day_of_week = shift_date.weekday()

        for availability in availability_slots:
            avail_day = getattr(availability, 'day_of_week', None)
            avail_start = getattr(availability, 'start_time', None)
            avail_end = getattr(availability, 'end_time', None)
            is_available = getattr(availability, 'is_available', True)

            if (avail_day == day_of_week and is_available and
                avail_start <= start_time and end_time <= avail_end):
                return True

        return False

    def _create_conflict(
        self,
        conflict_type: str,
        severity: str,
        description: str,
        **kwargs
    ) -> Any:
        """Enhanced conflict creation with better structure"""
        conflict = {
            'conflict_type': conflict_type,
            'severity': severity,
            'description': description,
            'suggested_solutions': kwargs.pop('suggested_solutions', []),
            'group_id': kwargs.get('group_id'),
            'staff_id': kwargs.get('staff_id'),
            'time_slot': kwargs.get('time_slot')
        }

        # Add any additional fields
        for key, value in kwargs.items():
            if key not in conflict and value is not None:
                conflict[key] = value

        return conflict

class ScheduleAnalyzer:
    """Analyzes schedules to generate insights, warnings, and suggestions"""

    def __init__(self):
        self.analysis_cache = {}

    def generate_warnings(self, schedule: list[Any], staff: list[Any]) -> list[Any]:
        """Generate warnings about potential schedule issues"""
        warnings = []

        # Check for workload imbalance
        warnings.extend(self._check_workload_balance(schedule, staff))

        # Check for scheduling patterns that might cause fatigue
        warnings.extend(self._check_fatigue_patterns(schedule))

        # Check for inefficient shift patterns
        warnings.extend(self._check_efficiency_patterns(schedule))

        return warnings

    def generate_suggestions(
        self,
        schedule: list[Any],
        staff: list[Any],
        groups: list[Any]
    ) -> list[Any]:
        """Generate suggestions for schedule improvements"""
        suggestions = []

        # Suggest shift consolidations
        suggestions.extend(self._suggest_consolidations(schedule))

        # Suggest better coverage patterns
        suggestions.extend(self._suggest_coverage_improvements(schedule, groups))

        # Suggest staff utilization improvements
        suggestions.extend(self._suggest_utilization_improvements(schedule, staff))

        return suggestions

    def analyze_schedule_metrics(self, schedule: list[Any], staff: list[Any]) -> dict[str, Any]:
        """Analyze various schedule metrics"""
        metrics = {}

        # Calculate coverage metrics
        metrics['coverage'] = self._analyze_coverage(schedule)

        # Calculate efficiency metrics
        metrics['efficiency'] = self._analyze_efficiency(schedule)

        # Calculate staff satisfaction metrics
        metrics['satisfaction'] = self._analyze_satisfaction(schedule, staff)

        return metrics

    def _check_workload_balance(self, schedule: list[Any], staff: list[Any]) -> list[Any]:
        """Check for workload imbalance among staff"""
        warnings = []

        # Calculate hours per staff member
        staff_hours = {}
        for shift in schedule:
            staff_id = getattr(shift, 'staff_id', None)
            hours = getattr(shift, 'scheduled_hours', 0)

            if staff_id not in staff_hours:
                staff_hours[staff_id] = 0
            staff_hours[staff_id] += hours

        if len(staff_hours) > 1:
            hours_values = list(staff_hours.values())
            avg_hours = sum(hours_values) / len(hours_values)
            max_hours = max(hours_values)
            min_hours = min(hours_values)

            # Check for significant imbalance
            if max_hours - min_hours > avg_hours * 0.5:  # 50% variance
                warnings.append({
                    'type': 'workload_imbalance',
                    'severity': 'warning',
                    'description': f'Significant workload imbalance: {min_hours:.1f}h - {max_hours:.1f}h range',
                    'suggestion': 'Consider redistributing shifts for better balance'
                })

        return warnings

    def _check_fatigue_patterns(self, schedule: list[Any]) -> list[Any]:
        """Check for patterns that might cause staff fatigue"""
        warnings = []

        # Group shifts by staff
        staff_shifts = {}
        for shift in schedule:
            staff_id = getattr(shift, 'staff_id', None)
            if staff_id not in staff_shifts:
                staff_shifts[staff_id] = []
            staff_shifts[staff_id].append(shift)

        for staff_id, shifts in staff_shifts.items():
            # Check for consecutive long days
            daily_hours = {}
            for shift in shifts:
                shift_date = getattr(shift, 'date', None)
                hours = getattr(shift, 'scheduled_hours', 0)

                if shift_date not in daily_hours:
                    daily_hours[shift_date] = 0
                daily_hours[shift_date] += hours

            # Check for multiple consecutive days with >8 hours
            consecutive_long_days = 0
            for date_obj in sorted(daily_hours.keys()):
                if daily_hours[date_obj] > 8:
                    consecutive_long_days += 1
                    if consecutive_long_days >= 3:
                        warnings.append({
                            'type': 'fatigue_risk',
                            'severity': 'warning',
                            'description': f'Staff {staff_id} has {consecutive_long_days} consecutive long days',
                            'suggestion': 'Consider adding rest days or shorter shifts'
                        })
                        break
                else:
                    consecutive_long_days = 0

        return warnings

    def _check_efficiency_patterns(self, schedule: list[Any]) -> list[Any]:
        """Check for inefficient scheduling patterns"""
        warnings = []

        # Check for fragmented shifts
        staff_shifts = {}
        for shift in schedule:
            staff_id = getattr(shift, 'staff_id', None)
            shift_date = getattr(shift, 'date', None)
            key = (staff_id, shift_date)

            if key not in staff_shifts:
                staff_shifts[key] = []
            staff_shifts[key].append(shift)

        for (staff_id, shift_date), shifts in staff_shifts.items():
            if len(shifts) > 2:  # More than 2 shifts per day
                total_hours = sum(getattr(shift, 'scheduled_hours', 0) for shift in shifts)
                warnings.append({
                    'type': 'fragmented_schedule',
                    'severity': 'info',
                    'description': f'Staff {staff_id} has {len(shifts)} shifts on {shift_date} ({total_hours:.1f}h total)',
                    'suggestion': 'Consider consolidating into fewer shifts'
                })

        return warnings

    def _suggest_consolidations(self, schedule: list[Any]) -> list[str]:
        """Suggest shift consolidations"""
        suggestions = []

        # Group shifts by staff and date
        daily_shifts = {}
        for shift in schedule:
            staff_id = getattr(shift, 'staff_id', None)
            shift_date = getattr(shift, 'date', None)
            key = (staff_id, shift_date)

            if key not in daily_shifts:
                daily_shifts[key] = []
            daily_shifts[key].append(shift)

        for (staff_id, shift_date), shifts in daily_shifts.items():
            if len(shifts) > 1:
                # Check if shifts are close together
                sorted_shifts = sorted(shifts, key=lambda s: getattr(s, 'start_time', time.min))

                for i in range(len(sorted_shifts) - 1):
                    current_end = getattr(sorted_shifts[i], 'end_time', None)
                    next_start = getattr(sorted_shifts[i + 1], 'start_time', None)

                    if current_end and next_start:
                        # Calculate gap
                        current_end_dt = datetime.combine(shift_date, current_end)
                        next_start_dt = datetime.combine(shift_date, next_start)
                        gap_hours = (next_start_dt - current_end_dt).total_seconds() / 3600

                        if gap_hours <= 2:  # Less than 2 hour gap
                            suggestions.append(
                                f"Consider consolidating shifts for {staff_id} on {shift_date} "
                                f"(gap: {gap_hours:.1f}h)"
                            )
                            break

        return suggestions

    def _suggest_coverage_improvements(self, schedule: list[Any], groups: list[Any]) -> list[str]:
        """Suggest coverage improvements"""
        suggestions = []

        # Analyze coverage patterns
        group_coverage = {}
        for shift in schedule:
            group_id = getattr(shift, 'group_id', None)
            shift_date = getattr(shift, 'date', None)
            start_time = getattr(shift, 'start_time', None)

            if group_id not in group_coverage:
                group_coverage[group_id] = {}
            if shift_date not in group_coverage[group_id]:
                group_coverage[group_id][shift_date] = []

            group_coverage[group_id][shift_date].append(start_time)

        # Check for gaps in coverage
        for group_id, daily_coverage in group_coverage.items():
            for shift_date, start_times in daily_coverage.items():
                if len(start_times) < 2:  # Less than 2 shifts per day
                    suggestions.append(
                        f"Consider adding coverage for group {group_id} on {shift_date}"
                    )

        return suggestions

    def _suggest_utilization_improvements(self, schedule: list[Any], staff: list[Any]) -> list[str]:
        """Suggest staff utilization improvements"""
        suggestions = []

        # Calculate utilization rates
        staff_hours = {}
        staff_max_hours = {}

        for shift in schedule:
            staff_id = getattr(shift, 'staff_id', None)
            hours = getattr(shift, 'scheduled_hours', 0)

            if staff_id not in staff_hours:
                staff_hours[staff_id] = 0
            staff_hours[staff_id] += hours

        for staff_member in staff:
            staff_id = getattr(staff_member, 'staff_id', None)
            max_hours = getattr(staff_member, 'max_weekly_hours', 40)
            staff_max_hours[staff_id] = max_hours

        # Find underutilized staff
        for staff_id, max_hours in staff_max_hours.items():
            actual_hours = staff_hours.get(staff_id, 0)
            utilization = actual_hours / max_hours if max_hours > 0 else 0

            if utilization < 0.5:  # Less than 50% utilization
                suggestions.append(
                    f"Staff {staff_id} is underutilized ({utilization:.1%}) - "
                    f"consider adding more shifts"
                )
            elif utilization > 0.95:  # Over 95% utilization
                suggestions.append(
                    f"Staff {staff_id} is highly utilized ({utilization:.1%}) - "
                    f"monitor for burnout risk"
                )

        return suggestions

    def _analyze_coverage(self, schedule: list[Any]) -> dict[str, Any]:
        """Analyze schedule coverage metrics"""
        coverage_metrics = {
            'total_shifts': len(schedule),
            'total_hours': sum(getattr(shift, 'scheduled_hours', 0) for shift in schedule),
            'unique_dates': len(set(getattr(shift, 'date', None) for shift in schedule)),
            'unique_staff': len(set(getattr(shift, 'staff_id', None) for shift in schedule)),
            'unique_groups': len(set(getattr(shift, 'group_id', None) for shift in schedule))
        }

        return coverage_metrics

    def _analyze_efficiency(self, schedule: list[Any]) -> dict[str, Any]:
        """Analyze schedule efficiency metrics"""
        # Calculate average shift duration
        shift_durations = [getattr(shift, 'scheduled_hours', 0) for shift in schedule]
        avg_shift_duration = sum(shift_durations) / len(shift_durations) if shift_durations else 0

        # Count fragmented schedules
        daily_shift_counts = {}
        for shift in schedule:
            staff_id = getattr(shift, 'staff_id', None)
            shift_date = getattr(shift, 'date', None)
            key = (staff_id, shift_date)

            if key not in daily_shift_counts:
                daily_shift_counts[key] = 0
            daily_shift_counts[key] += 1

        fragmented_days = sum(1 for count in daily_shift_counts.values() if count > 1)

        efficiency_metrics = {
            'avg_shift_duration': avg_shift_duration,
            'fragmented_days': fragmented_days,
            'total_staff_days': len(daily_shift_counts),
            'fragmentation_rate': fragmented_days / len(daily_shift_counts) if daily_shift_counts else 0
        }

        return efficiency_metrics

    def _analyze_satisfaction(self, schedule: list[Any], staff: list[Any]) -> dict[str, Any]:
        """Analyze staff satisfaction metrics"""
        # This is a simplified satisfaction analysis
        # In practice, would incorporate staff preferences, etc.

        staff_hours = {}
        staff_max_hours = {}

        for shift in schedule:
            staff_id = getattr(shift, 'staff_id', None)
            hours = getattr(shift, 'scheduled_hours', 0)

            if staff_id not in staff_hours:
                staff_hours[staff_id] = 0
            staff_hours[staff_id] += hours

        for staff_member in staff:
            staff_id = getattr(staff_member, 'staff_id', None)
            max_hours = getattr(staff_member, 'max_weekly_hours', 40)
            staff_max_hours[staff_id] = max_hours

        # Calculate utilization distribution
        utilizations = []
        for staff_id in staff_max_hours:
            actual_hours = staff_hours.get(staff_id, 0)
            max_hours = staff_max_hours[staff_id]
            utilization = actual_hours / max_hours if max_hours > 0 else 0
            utilizations.append(utilization)

        avg_utilization = sum(utilizations) / len(utilizations) if utilizations else 0

        satisfaction_metrics = {
            'avg_utilization': avg_utilization,
            'utilization_variance': sum((u - avg_utilization) ** 2 for u in utilizations) / len(utilizations) if utilizations else 0,
            'staff_with_shifts': len(staff_hours),
            'total_staff': len(staff_max_hours)
        }

        return satisfaction_metrics


class CacheManager:
    """Manages caching for schedule optimization results"""

    def __init__(self):
        self.cache = {}
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0
        }
        self.max_cache_size = 100

    async def get(self, key: str) -> Any:
        """Get cached value"""
        if key in self.cache:
            self.cache_stats['hits'] += 1
            cache_entry = self.cache[key]

            # Check if cache entry is still valid (simple time-based expiry)
            if self._is_cache_valid(cache_entry):
                return cache_entry['value']
            else:
                del self.cache[key]

        self.cache_stats['misses'] += 1
        return None

    async def set(self, key: str, value: Any, ttl_seconds: int = 3600):
        """Set cached value with TTL"""
        # Implement LRU eviction if cache is full
        if len(self.cache) >= self.max_cache_size:
            self._evict_oldest()

        self.cache[key] = {
            'value': value,
            'timestamp': datetime.now(),
            'ttl_seconds': ttl_seconds
        }
        self.cache_stats['sets'] += 1

    async def delete(self, key: str):
        """Delete cached value"""
        if key in self.cache:
            del self.cache[key]

    async def clear(self):
        """Clear all cached values"""
        self.cache.clear()
        self.cache_stats = {'hits': 0, 'misses': 0, 'sets': 0}

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = self.cache_stats['hits'] / total_requests if total_requests > 0 else 0

        return {
            'cache_size': len(self.cache),
            'max_cache_size': self.max_cache_size,
            'hit_rate': hit_rate,
            'stats': self.cache_stats.copy()
        }

    def _is_cache_valid(self, cache_entry: dict) -> bool:
        """Check if cache entry is still valid"""
        timestamp = cache_entry['timestamp']
        ttl_seconds = cache_entry['ttl_seconds']

        age_seconds = (datetime.now() - timestamp).total_seconds()
        return age_seconds < ttl_seconds

    def _evict_oldest(self):
        """Evict oldest cache entry (LRU)"""
        if not self.cache:
            return

        oldest_key = min(self.cache.keys(),
                        key=lambda k: self.cache[k]['timestamp'])
        del self.cache[oldest_key]
