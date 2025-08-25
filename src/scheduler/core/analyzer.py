"""
Schedule analysis engine for insights and recommendations
"""
import logging
from datetime import datetime, date, time, timedelta
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict, Counter
import statistics

from ..models import *
from ..config import settings

logger = logging.getLogger(__name__)


class ScheduleAnalyzer:
    """Analyzes schedules and provides insights and suggestions"""
    
    def __init__(self):
        logger.info("Schedule analyzer initialized")
    
    def generate_warnings(
        self,
        schedule: List[ScheduledShift],
        staff: List[Staff]
    ) -> List[str]:
        """Generate warnings about potential schedule issues"""
        
        warnings = []
        
        try:
            # Analyze hour distribution fairness
            distribution_warnings = self._analyze_hour_distribution(schedule, staff)
            warnings.extend(distribution_warnings)
            
            # Check weekend coverage patterns
            weekend_warnings = self._analyze_weekend_coverage(schedule, staff)
            warnings.extend(weekend_warnings)
            
            # Analyze rush hour coverage
            rush_hour_warnings = self._analyze_rush_hour_coverage(schedule)
            warnings.extend(rush_hour_warnings)
            
            # Check schedule fragmentation
            fragmentation_warnings = self._analyze_schedule_fragmentation(schedule, staff)
            warnings.extend(fragmentation_warnings)
            
            # Analyze priority weight effectiveness
            priority_warnings = self._analyze_priority_effectiveness(schedule, staff)
            warnings.extend(priority_warnings)
            
        except Exception as e:
            logger.error(f"Error generating warnings: {e}")
            warnings.append("Unable to complete schedule analysis due to system error")
        
        return warnings
    
    def generate_suggestions(
        self,
        schedule: List[ScheduledShift],
        staff: List[Staff],
        groups: List[Group]
    ) -> List[str]:
        """Generate suggestions for schedule improvement"""
        
        suggestions = []
        
        try:
            # Cost optimization suggestions
            cost_suggestions = self._generate_cost_optimization_suggestions(schedule, staff)
            suggestions.extend(cost_suggestions)
            
            # Scheduling pattern suggestions
            pattern_suggestions = self._generate_pattern_suggestions(schedule, staff)
            suggestions.extend(pattern_suggestions)
            
            # Coverage optimization suggestions
            coverage_suggestions = self._generate_coverage_suggestions(schedule, groups)
            suggestions.extend(coverage_suggestions)
            
            # Staff satisfaction suggestions
            satisfaction_suggestions = self._generate_satisfaction_suggestions(schedule, staff)
            suggestions.extend(satisfaction_suggestions)
            
            # Priority weight optimization suggestions
            priority_suggestions = self._generate_priority_suggestions(schedule, staff)
            suggestions.extend(priority_suggestions)
            
        except Exception as e:
            logger.error(f"Error generating suggestions: {e}")
            suggestions.append("Unable to generate optimization suggestions due to system error")
        
        return suggestions
    
    def analyze_schedule_metrics(
        self,
        schedule: List[ScheduledShift],
        staff: List[Staff],
        groups: List[Group]
    ) -> Dict[str, any]:
        """Generate comprehensive schedule metrics and analysis"""
        
        metrics = {
            'efficiency_metrics': {},
            'fairness_metrics': {},
            'satisfaction_metrics': {},
            'cost_metrics': {},
            'coverage_metrics': {},
            'priority_metrics': {}
        }
        
        try:
            # Calculate efficiency metrics
            metrics['efficiency_metrics'] = self._calculate_efficiency_metrics(schedule, staff)
            
            # Calculate fairness metrics
            metrics['fairness_metrics'] = self._calculate_fairness_metrics(schedule, staff)
            
            # Calculate satisfaction metrics
            metrics['satisfaction_metrics'] = self._calculate_satisfaction_metrics(schedule, staff)
            
            # Calculate cost metrics
            metrics['cost_metrics'] = self._calculate_cost_metrics(schedule, staff)
            
            # Calculate coverage metrics
            metrics['coverage_metrics'] = self._calculate_coverage_metrics(schedule, groups)
            
            # Calculate priority metrics
            metrics['priority_metrics'] = self._calculate_priority_metrics(schedule, staff)
            
        except Exception as e:
            logger.error(f"Error analyzing schedule metrics: {e}")
            metrics['error'] = str(e)
        
        return metrics
    
    def _analyze_hour_distribution(
        self,
        schedule: List[ScheduledShift],
        staff: List[Staff]
    ) -> List[str]:
        """Analyze fairness of hour distribution"""
        
        warnings = []
        
        # Calculate hours per staff
        staff_hours = {}
        staff_dict = {s.staff_id: s for s in staff}
        
        for shift in schedule:
            staff_hours[shift.staff_id] = staff_hours.get(shift.staff_id, 0) + shift.scheduled_hours
        
        if len(staff_hours) > 1:
            hours_values = list(staff_hours.values())
            
            # Calculate distribution statistics
            avg_hours = statistics.mean(hours_values)
            stdev_hours = statistics.stdev(hours_values) if len(hours_values) > 1 else 0
            max_hours = max(hours_values)
            min_hours = min(hours_values)
            
            # Check for large variance
            coefficient_of_variation = stdev_hours / avg_hours if avg_hours > 0 else 0
            
            if coefficient_of_variation > 0.3:  # 30% coefficient of variation
                warnings.append(
                    f"Uneven hour distribution detected - some staff may be overworked "
                    f"(range: {min_hours:.1f}-{max_hours:.1f}h, avg: {avg_hours:.1f}h, CV: {coefficient_of_variation:.1%})"
                )
            
            # Check if priority weights are properly reflected
            priority_hour_correlation = self._calculate_priority_hour_correlation(staff_hours, staff_dict)
            if priority_hour_correlation < 0.3:  # Low correlation
                warnings.append(
                    f"Priority weights may not be properly reflected in hour distribution "
                    f"(correlation: {priority_hour_correlation:.2f})"
                )
            
            # Identify significantly under/over-allocated staff
            for staff_id, hours in staff_hours.items():
                staff_member = staff_dict.get(staff_id)
                if not staff_member:
                    continue
                
                expected_hours = avg_hours * staff_member.priority_score
                deviation = abs(hours - expected_hours) / expected_hours if expected_hours > 0 else 0
                
                if deviation > 0.4:  # 40% deviation from expected
                    status = "under-allocated" if hours < expected_hours else "over-allocated"
                    warnings.append(
                        f"{staff_member.name} appears {status} based on priority weights "
                        f"({hours:.1f}h actual vs {expected_hours:.1f}h expected)"
                    )
        
        return warnings
    
    def _analyze_weekend_coverage(
        self,
        schedule: List[ScheduledShift],
        staff: List[Staff]
    ) -> List[str]:
        """Analyze weekend coverage patterns"""
        
        warnings = []
        
        # Count weekend workers
        weekend_workers = set()
        weekday_workers = set()
        
        for shift in schedule:
            day_of_week = shift.date.weekday()
            if day_of_week >= 5:  # Saturday or Sunday
                weekend_workers.add(shift.staff_id)
            else:
                weekday_workers.add(shift.staff_id)
        
        total_staff = len(set(shift.staff_id for shift in schedule))
        
        if total_staff > 0:
            weekend_coverage_ratio = len(weekend_workers) / total_staff
            
            if weekend_coverage_ratio < 0.3:
                warnings.append(
                    f"Limited weekend coverage - only {len(weekend_workers)}/{total_staff} "
                    f"staff scheduled for weekends ({weekend_coverage_ratio:.1%})"
                )
            
            # Check for weekend equity
            only_weekend_workers = weekend_workers - weekday_workers
            if len(only_weekend_workers) > len(weekend_workers) * 0.5:
                warnings.append(
                    f"Weekend burden may be unequally distributed - "
                    f"{len(only_weekend_workers)} staff work only weekends"
                )
        
        return warnings
    
    def _analyze_rush_hour_coverage(
        self,
        schedule: List[ScheduledShift]
    ) -> List[str]:
        """Analyze coverage during rush hours"""
        
        warnings = []
        
        # Define rush hours (typically drop-off and pick-up times)
        rush_hours = [
            (7, 9),   # Morning drop-off
            (17, 19)  # Evening pick-up
        ]
        
        # Count staff coverage for each rush hour
        for start_hour, end_hour in rush_hours:
            rush_period = f"{start_hour:02d}:00-{end_hour:02d}:00"
            
            # Count unique staff working during this period
            rush_hour_staff = set()
            total_rush_shifts = 0
            
            for shift in schedule:
                shift_start = shift.start_time.hour
                shift_end = shift.end_time.hour
                if shift_end <= shift_start:  # Overnight shift
                    shift_end += 24
                
                # Check if shift overlaps with rush hour
                if shift_start < end_hour and shift_end > start_hour:
                    rush_hour_staff.add(shift.staff_id)
                    total_rush_shifts += 1
            
            total_staff = len(set(shift.staff_id for shift in schedule))
            
            if total_staff > 0:
                coverage_ratio = len(rush_hour_staff) / total_staff
                
                if coverage_ratio < 0.4:  # Less than 40% of staff during rush hours
                    warnings.append(
                        f"Potential understaffing during rush hour {rush_period} - "
                        f"only {len(rush_hour_staff)}/{total_staff} staff available "
                        f"({coverage_ratio:.1%} coverage)"
                    )
        
        return warnings
    
    def _analyze_schedule_fragmentation(
        self,
        schedule: List[ScheduledShift],
        staff: List[Staff]
    ) -> List[str]:
        """Analyze schedule fragmentation and continuity"""
        
        warnings = []
        staff_dict = {s.staff_id: s for s in staff}
        
        # Group shifts by staff
        staff_shifts = defaultdict(list)
        for shift in schedule:
            staff_shifts[shift.staff_id].append(shift)
        
        fragmented_staff = []
        
        for staff_id, shifts in staff_shifts.items():
            staff_member = staff_dict.get(staff_id)
            if not staff_member or len(shifts) <= 1:
                continue
            
            # Analyze fragmentation for this staff member
            daily_shifts = defaultdict(list)
            for shift in shifts:
                daily_shifts[shift.date].append(shift)
            
            fragmentation_score = 0
            total_shift_blocks = 0
            
            for date, day_shifts in daily_shifts.items():
                if len(day_shifts) <= 1:
                    total_shift_blocks += 1
                    continue
                
                # Sort shifts by start time
                day_shifts.sort(key=lambda s: s.start_time)
                
                # Count gaps between shifts
                shift_blocks = 1
                for i in range(len(day_shifts) - 1):
                    current_end = day_shifts[i].end_time
                    next_start = day_shifts[i + 1].start_time
                    
                    # Check for gap (more than 30 minutes)
                    gap_minutes = (
                        datetime.combine(date, next_start) - 
                        datetime.combine(date, current_end)
                    ).total_seconds() / 60
                    
                    if gap_minutes > 30:  # 30+ minute gap
                        shift_blocks += 1
                        fragmentation_score += gap_minutes / 60  # Convert to hours
                
                total_shift_blocks += shift_blocks
            
            # Calculate average fragmentation
            working_days = len(daily_shifts)
            avg_blocks_per_day = total_shift_blocks / working_days if working_days > 0 else 0
            
            if avg_blocks_per_day > 2 or fragmentation_score > 10:  # Highly fragmented
                fragmented_staff.append((staff_member.name, avg_blocks_per_day, fragmentation_score))
        
        if fragmented_staff:
            for name, avg_blocks, frag_score in fragmented_staff[:3]:  # Top 3 most fragmented
                warnings.append(
                    f"{name} has a fragmented schedule with {avg_blocks:.1f} shift blocks per day "
                    f"and {frag_score:.1f} hours of gaps"
                )
        
        return warnings
    
    def _analyze_priority_effectiveness(
        self,
        schedule: List[ScheduledShift],
        staff: List[Staff]
    ) -> List[str]:
        """Analyze how effectively priority weights are being used"""
        
        warnings = []
        
        # Calculate actual vs expected hour distribution based on priorities
        staff_dict = {s.staff_id: s for s in staff}
        staff_hours = {}
        
        for shift in schedule:
            staff_hours[shift.staff_id] = staff_hours.get(shift.staff_id, 0) + shift.scheduled_hours
        
        if not staff_hours:
            return warnings
        
        # Calculate priority-weighted expected distribution
        total_actual_hours = sum(staff_hours.values())
        total_priority_weight = sum(s.priority_score for s in staff if s.staff_id in staff_hours)
        
        if total_priority_weight == 0:
            return warnings
        
        priority_deviations = []
        
        for staff_member in staff:
            if staff_member.staff_id not in staff_hours:
                continue
            
            actual_hours = staff_hours[staff_member.staff_id]
            expected_hours = (staff_member.priority_score / total_priority_weight) * total_actual_hours
            
            if expected_hours > 0:
                deviation = abs(actual_hours - expected_hours) / expected_hours
                priority_deviations.append((staff_member.name, deviation, actual_hours, expected_hours))
        
        # Check for large deviations
        high_deviation_staff = [
            (name, dev, actual, expected) for name, dev, actual, expected in priority_deviations 
            if dev > 0.3  # 30% deviation
        ]
        
        if high_deviation_staff:
            warnings.append(
                f"Priority weights may not be effectively applied - "
                f"{len(high_deviation_staff)} staff have significant hour deviations from expected"
            )
            
            # Show top deviations
            high_deviation_staff.sort(key=lambda x: x[1], reverse=True)
            for name, deviation, actual, expected in high_deviation_staff[:2]:
                warnings.append(
                    f"  • {name}: {actual:.1f}h actual vs {expected:.1f}h expected "
                    f"({deviation:.1%} deviation)"
                )
        
        return warnings
    
    def _generate_cost_optimization_suggestions(
        self,
        schedule: List[ScheduledShift],
        staff: List[Staff]
    ) -> List[str]:
        """Generate cost optimization suggestions"""
        
        suggestions = []
        staff_dict = {s.staff_id: s for s in staff}
        
        # Calculate cost metrics
        total_cost = 0
        total_hours = 0
        staff_costs = {}
        
        for shift in schedule:
            staff_member = staff_dict.get(shift.staff_id)
            if staff_member and staff_member.hourly_rate:
                shift_cost = shift.scheduled_hours * staff_member.hourly_rate
                total_cost += shift_cost
                total_hours += shift.scheduled_hours
                
                if shift.staff_id not in staff_costs:
                    staff_costs[shift.staff_id] = {'cost': 0, 'hours': 0}
                staff_costs[shift.staff_id]['cost'] += shift_cost
                staff_costs[shift.staff_id]['hours'] += shift.scheduled_hours
        
        if total_cost > 0 and total_hours > 0:
            avg_hourly_cost = total_cost / total_hours
            
            # Identify high-cost staff utilization
            high_cost_staff = []
            for staff_id, cost_data in staff_costs.items():
                staff_member = staff_dict.get(staff_id)
                if staff_member and staff_member.hourly_rate:
                    if staff_member.hourly_rate > avg_hourly_cost * 1.3:  # 30% above average
                        efficiency_score = cost_data['hours'] * staff_member.priority_score
                        high_cost_staff.append((staff_member.name, staff_member.hourly_rate, efficiency_score))
            
            if high_cost_staff:
                suggestions.append(
                    f"Consider optimizing high-cost staff usage - average hourly cost: ${avg_hourly_cost:.2f}"
                )
                
                # Suggest efficiency improvements
                high_cost_staff.sort(key=lambda x: x[2])  # Sort by efficiency score
                for name, rate, efficiency in high_cost_staff[:2]:
                    suggestions.append(
                        f"  • Review {name}'s schedule (${rate:.2f}/hr) for efficiency opportunities"
                    )
            
            # Suggest overtime reduction if applicable
            overtime_staff = []
            for staff_member in staff:
                if staff_member.staff_id in staff_costs:
                    hours = staff_costs[staff_member.staff_id]['hours']
                    if hours > staff_member.max_weekly_hours:
                        overtime_hours = hours - staff_member.max_weekly_hours
                        overtime_cost = overtime_hours * (staff_member.hourly_rate or 0) * 1.5  # 1.5x overtime rate
                        overtime_staff.append((staff_member.name, overtime_hours, overtime_cost))
            
            if overtime_staff:
                total_overtime_cost = sum(cost for _, _, cost in overtime_staff)
                suggestions.append(
                    f"Reduce overtime to save ${total_overtime_cost:.2f} weekly - "
                    f"{len(overtime_staff)} staff working overtime"
                )
        
        return suggestions
    
    def _generate_pattern_suggestions(
        self,
        schedule: List[ScheduledShift],
        staff: List[Staff]
    ) -> List[str]:
        """Generate scheduling pattern suggestions"""
        
        suggestions = []
        staff_dict = {s.staff_id: s for s in staff}
        
        # Analyze shift patterns
        staff_patterns = defaultdict(lambda: {'fragmented_days': 0, 'long_shifts': 0, 'short_shifts': 0})
        
        staff_shifts = defaultdict(list)
        for shift in schedule:
            staff_shifts[shift.staff_id].append(shift)
        
        for staff_id, shifts in staff_shifts.items():
            staff_member = staff_dict.get(staff_id)
            if not staff_member:
                continue
            
            # Group by date
            daily_shifts = defaultdict(list)
            for shift in shifts:
                daily_shifts[shift.date].append(shift)
            
            for date, day_shifts in daily_shifts.items():
                day_shifts.sort(key=lambda s: s.start_time)
                
                # Check for fragmentation
                if len(day_shifts) > 2:
                    staff_patterns[staff_id]['fragmented_days'] += 1
                
                # Check shift lengths
                total_day_hours = sum(s.scheduled_hours for s in day_shifts)
                if total_day_hours > 8:
                    staff_patterns[staff_id]['long_shifts'] += 1
                elif total_day_hours < 4:
                    staff_patterns[staff_id]['short_shifts'] += 1
        
        # Generate suggestions based on patterns
        for staff_id, patterns in staff_patterns.items():
            staff_member = staff_dict.get(staff_id)
            if not staff_member:
                continue
            
            if patterns['fragmented_days'] > 2:
                suggestions.append(
                    f"Consider consolidating {staff_member.name}'s shifts - "
                    f"{patterns['fragmented_days']} fragmented days detected"
                )
            
            if patterns['short_shifts'] > 3:
                suggestions.append(
                    f"Consider extending {staff_member.name}'s short shifts for better efficiency"
                )
        
        return suggestions
    
    def _generate_coverage_suggestions(
        self,
        schedule: List[ScheduledShift],
        groups: List[Group]
    ) -> List[str]:
        """Generate coverage optimization suggestions"""
        
        suggestions = []
        
        # Analyze coverage by group and time
        group_coverage = defaultdict(lambda: defaultdict(int))
        
        for shift in schedule:
            hour = shift.start_time.hour
            group_coverage[shift.group_id][hour] += 1
        
        for group in groups:
            coverage = group_coverage[group.group_id]
            if not coverage:
                suggestions.append(f"No staff assigned to group '{group.name}' - verify if this is intentional")
                continue
            
            # Check coverage consistency
            if coverage:
                hours = list(coverage.keys())
                coverage_values = list(coverage.values())
                
                min_coverage = min(coverage_values)
                max_coverage = max(coverage_values)
                
                if max_coverage > min_coverage * 2:
                    suggestions.append(
                        f"Consider balancing staff coverage for group '{group.name}' "
                        f"(range: {min_coverage}-{max_coverage} staff per hour)"
                    )
        
        return suggestions
    
    def _generate_satisfaction_suggestions(
        self,
        schedule: List[ScheduledShift],
        staff: List[Staff]
    ) -> List[str]:
        """Generate staff satisfaction improvement suggestions"""
        
        suggestions = []
        
        # Analyze preference satisfaction
        low_satisfaction_staff = []
        
        for staff_member in staff:
            staff_shifts = [s for s in schedule if s.staff_id == staff_member.staff_id]
            satisfaction_score = self._calculate_individual_satisfaction(staff_member, staff_shifts)
            
            if satisfaction_score < 0.6:  # Low satisfaction threshold
                low_satisfaction_staff.append((staff_member.name, satisfaction_score))
        
        if low_satisfaction_staff:
            suggestions.append(
                f"Improve schedule satisfaction for {len(low_satisfaction_staff)} staff members "
                f"with low preference alignment"
            )
            
            # Show specific staff with lowest satisfaction
            low_satisfaction_staff.sort(key=lambda x: x[1])
            for name, score in low_satisfaction_staff[:2]:
                suggestions.append(f"  • Review {name}'s preferences (satisfaction: {score:.1%})")
        
        return suggestions
    
    def _generate_priority_suggestions(
        self,
        schedule: List[ScheduledShift],
        staff: List[Staff]
    ) -> List[str]:
        """Generate priority weight optimization suggestions"""
        
        suggestions = []
        
        # Analyze priority effectiveness
        staff_hours = {}
        for shift in schedule:
            staff_hours[shift.staff_id] = staff_hours.get(shift.staff_id, 0) + shift.scheduled_hours
        
        # Check if high-priority staff are getting appropriate hours
        high_priority_staff = [s for s in staff if s.priority_score > 1.3]
        low_utilization_high_priority = []
        
        for staff_member in high_priority_staff:
            hours = staff_hours.get(staff_member.staff_id, 0)
            expected_hours = 35 * staff_member.priority_score  # Base expectation
            
            if hours < expected_hours * 0.7:  # Less than 70% of expected
                low_utilization_high_priority.append(staff_member.name)
        
        if low_utilization_high_priority:
            suggestions.append(
                f"High-priority staff may be under-utilized: {', '.join(low_utilization_high_priority)}"
            )
            suggestions.append("Consider increasing availability or adjusting priority weights")
        
        return suggestions
    
    # Helper methods for metrics calculation
    def _calculate_efficiency_metrics(self, schedule: List[ScheduledShift], staff: List[Staff]) -> Dict:
        """Calculate efficiency-related metrics"""
        
        metrics = {
            'shift_utilization': 0.0,
            'schedule_density': 0.0,
            'fragmentation_index': 0.0
        }
        
        if not schedule:
            return metrics
        
        # Calculate shift utilization (hours per shift)
        total_hours = sum(s.scheduled_hours for s in schedule)
        avg_shift_hours = total_hours / len(schedule)
        metrics['shift_utilization'] = avg_shift_hours / 8.0  # Normalize to 8-hour standard
        
        # Calculate schedule density (coverage across time)
        time_coverage = set()
        for shift in schedule:
            for hour in range(shift.start_time.hour, shift.end_time.hour):
                time_coverage.add((shift.date, hour))
        
        possible_coverage = len(schedule) * 8  # Assume 8-hour potential per shift
        metrics['schedule_density'] = len(time_coverage) / possible_coverage if possible_coverage > 0 else 0
        
        return metrics
    
    def _calculate_fairness_metrics(self, schedule: List[ScheduledShift], staff: List[Staff]) -> Dict:
        """Calculate fairness-related metrics"""
        
        metrics = {
            'hour_distribution_cv': 0.0,
            'priority_fairness_score': 0.0,
            'weekend_equity': 0.0
        }
        
        staff_hours = {}
        for shift in schedule:
            staff_hours[shift.staff_id] = staff_hours.get(shift.staff_id, 0) + shift.scheduled_hours
        
        if len(staff_hours) > 1:
            hours_values = list(staff_hours.values())
            avg_hours = statistics.mean(hours_values)
            stdev_hours = statistics.stdev(hours_values)
            metrics['hour_distribution_cv'] = stdev_hours / avg_hours if avg_hours > 0 else 0
        
        return metrics
    
    def _calculate_satisfaction_metrics(self, schedule: List[ScheduledShift], staff: List[Staff]) -> Dict:
        """Calculate satisfaction-related metrics"""
        
        metrics = {
            'avg_satisfaction': 0.0,
            'preference_compliance': 0.0
        }
        
        satisfaction_scores = []
        for staff_member in staff:
            staff_shifts = [s for s in schedule if s.staff_id == staff_member.staff_id]
            satisfaction = self._calculate_individual_satisfaction(staff_member, staff_shifts)
            satisfaction_scores.append(satisfaction)
        
        if satisfaction_scores:
            metrics['avg_satisfaction'] = statistics.mean(satisfaction_scores)
        
        return metrics
    
    def _calculate_cost_metrics(self, schedule: List[ScheduledShift], staff: List[Staff]) -> Dict:
        """Calculate cost-related metrics"""
        
        metrics = {
            'total_cost': 0.0,
            'avg_hourly_cost': 0.0,
            'overtime_cost': 0.0
        }
        
        staff_dict = {s.staff_id: s for s in staff}
        total_cost = 0
        total_hours = 0
        
        for shift in schedule:
            staff_member = staff_dict.get(shift.staff_id)
            if staff_member and staff_member.hourly_rate:
                cost = shift.scheduled_hours * staff_member.hourly_rate
                total_cost += cost
                total_hours += shift.scheduled_hours
        
        metrics['total_cost'] = total_cost
        metrics['avg_hourly_cost'] = total_cost / total_hours if total_hours > 0 else 0
        
        return metrics
    
    def _calculate_coverage_metrics(self, schedule: List[ScheduledShift], groups: List[Group]) -> Dict:
        """Calculate coverage-related metrics"""
        
        metrics = {
            'group_coverage_balance': 0.0,
            'time_coverage_consistency': 0.0
        }
        
        # Calculate coverage balance across groups
        group_hours = defaultdict(float)
        for shift in schedule:
            group_hours[shift.group_id] += shift.scheduled_hours
        
        if len(group_hours) > 1:
            hours_values = list(group_hours.values())
            avg_hours = statistics.mean(hours_values)
            stdev_hours = statistics.stdev(hours_values)
            cv = stdev_hours / avg_hours if avg_hours > 0 else 0
            metrics['group_coverage_balance'] = 1 - cv  # Higher is better
        
        return metrics
    
    def _calculate_priority_metrics(self, schedule: List[ScheduledShift], staff: List[Staff]) -> Dict:
        """Calculate priority-related metrics"""
        
        metrics = {
            'priority_hour_correlation': 0.0,
            'high_priority_utilization': 0.0
        }
        
        staff_dict = {s.staff_id: s for s in staff}
        staff_hours = {}
        
        for shift in schedule:
            staff_hours[shift.staff_id] = staff_hours.get(shift.staff_id, 0) + shift.scheduled_hours
        
        # Calculate correlation between priority and hours
        metrics['priority_hour_correlation'] = self._calculate_priority_hour_correlation(staff_hours, staff_dict)
        
        return metrics
    
    def _calculate_priority_hour_correlation(self, staff_hours: Dict, staff_dict: Dict) -> float:
        """Calculate correlation between priority scores and assigned hours"""
        
        try:
            priorities = []
            hours = []
            
            for staff_id, staff_hours_val in staff_hours.items():
                staff_member = staff_dict.get(staff_id)
                if staff_member:
                    priorities.append(staff_member.priority_score)
                    hours.append(staff_hours_val)
            
            if len(priorities) < 2:
                return 0.0
            
            # Calculate Pearson correlation coefficient
            n = len(priorities)
            sum_p = sum(priorities)
            sum_h = sum(hours)
            sum_p2 = sum(p * p for p in priorities)
            sum_h2 = sum(h * h for h in hours)
            sum_ph = sum(p * h for p, h in zip(priorities, hours))
            
            numerator = n * sum_ph - sum_p * sum_h
            denominator = ((n * sum_p2 - sum_p * sum_p) * (n * sum_h2 - sum_h * sum_h)) ** 0.5
            
            return numerator / denominator if denominator != 0 else 0.0
            
        except Exception as e:
            logger.warning(f"Error calculating priority-hour correlation: {e}")
            return 0.0
    
    def _calculate_individual_satisfaction(self, staff_member: Staff, staff_shifts: List[ScheduledShift]) -> float:
        """Calculate satisfaction score for individual staff member"""
        
        if not staff_shifts:
            return 1.0
        
        if not staff_member.preferences:
            return 1.0
        
        satisfaction_score = 0.0
        total_weight = 0.0
        
        for preference in staff_member.preferences:
            pref_satisfaction = self._evaluate_preference_satisfaction(preference, staff_shifts)
            satisfaction_score += pref_satisfaction * preference.weight
            total_weight += preference.weight
        
        return satisfaction_score / total_weight if total_weight > 0 else 1.0
    
    def _evaluate_preference_satisfaction(self, preference: StaffPreference, shifts: List[ScheduledShift]) -> float:
        """Evaluate how well shifts satisfy a specific preference"""
        
        if preference.preference_type == PreferenceType.PREFERRED_TIME:
            matching_shifts = sum(1 for shift in shifts if self._shift_matches_preference(shift, preference))
            return matching_shifts / len(shifts) if shifts else 0.0
        
        elif preference.preference_type == PreferenceType.AVOID_DAYS:
            avoided_shifts = sum(1 for shift in shifts 
                               if preference.day_of_week is not None and 
                               shift.date.weekday() == preference.day_of_week)
            return 1.0 - (avoided_shifts / len(shifts)) if shifts else 1.0
        
        elif preference.preference_type == PreferenceType.MAX_HOURS:
            if preference.max_hours_per_day:
                daily_hours = defaultdict(float)
                for shift in shifts:
                    daily_hours[shift.date] += shift.scheduled_hours
                
                violations = sum(1 for hours in daily_hours.values() 
                               if hours > preference.max_hours_per_day)
                return 1.0 - (violations / len(daily_hours)) if daily_hours else 1.0
        
        return 1.0
    
    def _shift_matches_preference(self, shift: ScheduledShift, preference: StaffPreference) -> bool:
        """Check if a shift matches preference criteria"""
        
        if preference.day_of_week is not None:
            if shift.date.weekday() != preference.day_of_week:
                return False
        
        if preference.time_range_start and preference.time_range_end:
            shift_start_hour = shift.start_time.hour
            pref_start_hour = preference.time_range_start.hour
            pref_end_hour = preference.time_range_end.hour
            
            if pref_end_hour <= pref_start_hour:  # Overnight range
                return shift_start_hour >= pref_start_hour or shift_start_hour <= pref_end_hour
            else:
                return pref_start_hour <= shift_start_hour <= pref_end_hour
        
        return True