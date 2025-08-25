"""
Utility helper functions for the scheduler service
"""

import re
import hashlib
import json
import logging
from datetime import datetime, date, time, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple
from uuid import UUID
import calendar
from datetime import date, datetime, timedelta, time
from dataclasses import dataclass
import json
import csv
from io import StringIO
from calendar import monthrange

from ..models import *
from ..config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def generate_request_id() -> str:
    """Generate a unique request ID for tracking"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
    return f"req_{timestamp}"


def generate_hash(data: Any) -> str:
    """Generate MD5 hash of data for caching/comparison"""
    if isinstance(data, (dict, list)):
        json_str = json.dumps(data, sort_keys=True, default=str)
    else:
        json_str = str(data)

    return hashlib.md5(json_str.encode()).hexdigest()


def safe_division(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero"""
    try:
        return numerator / denominator if denominator != 0 else default
    except (TypeError, ValueError):
        return default


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value between min and max bounds"""
    return max(min_val, min(value, max_val))


def normalize_name(name: str) -> str:
    """Normalize a name for consistent formatting"""
    if not name:
        return ""

    # Remove extra whitespace and convert to title case
    normalized = re.sub(r"\s+", " ", name.strip()).title()
    return normalized


def validate_email(email: str) -> bool:
    """Validate email format"""
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email))


def validate_phone(phone: str) -> bool:
    """Validate phone number format (flexible)"""
    # Remove all non-digit characters
    digits_only = re.sub(r"\D", "", phone)
    # Check if it's a reasonable length (7-15 digits)
    return 7 <= len(digits_only) <= 15


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human readable string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_percentage(value: float, decimal_places: int = 1) -> str:
    """Format a decimal as percentage"""
    return f"{value * 100:.{decimal_places}f}%"


def get_week_dates(start_date: date) -> List[date]:
    """Get all dates in a week starting from given date"""
    dates = []
    for i in range(7):
        dates.append(start_date + timedelta(days=i))
    return dates


def get_monday_of_week(date_obj: date) -> date:
    """Get the Monday of the week containing the given date"""
    days_since_monday = date_obj.weekday()
    monday = date_obj - timedelta(days=days_since_monday)
    return monday


def time_to_minutes(time_obj: time) -> int:
    """Convert time object to minutes since midnight"""
    return time_obj.hour * 60 + time_obj.minute


def minutes_to_time(minutes: int) -> time:
    """Convert minutes since midnight to time object"""
    hours = minutes // 60
    mins = minutes % 60
    return time(hours % 24, mins)


def calculate_time_overlap(start1: time, end1: time, start2: time, end2: time) -> float:
    """Calculate overlap between two time ranges in hours"""

    # Convert to minutes for easier calculation
    start1_min = time_to_minutes(start1)
    end1_min = time_to_minutes(end1)
    start2_min = time_to_minutes(start2)
    end2_min = time_to_minutes(end2)

    # Handle overnight times
    if end1_min <= start1_min:
        end1_min += 24 * 60
    if end2_min <= start2_min:
        end2_min += 24 * 60

    # Calculate overlap
    overlap_start = max(start1_min, start2_min)
    overlap_end = min(end1_min, end2_min)

    if overlap_end <= overlap_start:
        return 0.0

    return (overlap_end - overlap_start) / 60.0  # Convert back to hours


def is_weekend(date_obj: date) -> bool:
    """Check if date is weekend (Saturday or Sunday)"""
    return date_obj.weekday() >= 5


def get_age_group_from_age(age_months: int) -> str:
    """Determine age group from age in months"""
    if age_months < 18:
        return "infant"
    elif age_months < 36:
        return "toddler"
    elif age_months < 60:
        return "preschool"
    else:
        return "school_age"


def calculate_staff_child_ratio(staff_count: int, child_count: int) -> str:
    """Calculate and format staff-to-child ratio"""
    if staff_count == 0:
        return "0:0"

    # Simplify ratio by finding GCD
    import math

    gcd = math.gcd(staff_count, child_count)
    simplified_staff = staff_count // gcd
    simplified_children = child_count // gcd

    return f"{simplified_staff}:{simplified_children}"


def parse_time_range(time_range_str: str) -> Tuple[Optional[time], Optional[time]]:
    """Parse time range string like '09:00-17:00' to time objects"""
    try:
        if "-" not in time_range_str:
            return None, None

        start_str, end_str = time_range_str.split("-", 1)
        start_time = datetime.strptime(start_str.strip(), "%H:%M").time()
        end_time = datetime.strptime(end_str.strip(), "%H:%M").time()

        return start_time, end_time

    except (ValueError, AttributeError):
        return None, None


def format_time_range(start_time: time, end_time: time) -> str:
    """Format time range to string"""
    return f"{start_time.strftime('%H:%M')}-{end_time.strftime('%H:%M')}"


def get_business_days_in_month(year: int, month: int) -> int:
    """Get number of business days (Mon-Fri) in a month"""
    cal = calendar.monthcalendar(year, month)
    business_days = 0

    for week in cal:
        for day in week[:5]:  # Monday to Friday
            if day != 0:  # 0 means day belongs to another month
                business_days += 1

    return business_days


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split a list into chunks of specified size"""
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def flatten_list(nested_list: List[List[Any]]) -> List[Any]:
    """Flatten a nested list"""
    return [item for sublist in nested_list for item in sublist]


def deep_merge_dicts(dict1: Dict, dict2: Dict) -> Dict:
    """Deep merge two dictionaries"""
    result = dict1.copy()

    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value

    return result


def sanitize_filename(filename: str) -> str:
    """Sanitize filename by removing invalid characters"""
    # Remove invalid filename characters
    sanitized = re.sub(r'[<>:"/\\|?*]', "_", filename)
    # Remove leading/trailing dots and spaces
    sanitized = sanitized.strip(". ")
    # Limit length
    return sanitized[:255]


def convert_to_serializable(obj: Any) -> Any:
    """Convert object to JSON serializable format"""
    if isinstance(obj, (UUID, date, datetime, time)):
        return str(obj)
    elif isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


def calculate_priority_score(
    seniority_weight: float,
    performance_weight: float,
    flexibility_score: float,
    overall_priority: str,
) -> float:
    """Calculate combined priority score from individual weights"""

    priority_multipliers = {"low": 0.5, "medium": 1.0, "high": 1.5, "critical": 2.0}

    base_score = priority_multipliers.get(overall_priority.lower(), 1.0)
    return base_score * seniority_weight * performance_weight * flexibility_score


def validate_priority_weights(
    seniority_weight: float, performance_weight: float, flexibility_score: float
) -> List[str]:
    """Validate priority weight values and return any errors"""

    errors = []

    # Check ranges
    if not 0.1 <= seniority_weight <= 3.0:
        errors.append("Seniority weight must be between 0.1 and 3.0")

    if not 0.1 <= performance_weight <= 3.0:
        errors.append("Performance weight must be between 0.1 and 3.0")

    if not 0.1 <= flexibility_score <= 3.0:
        errors.append("Flexibility score must be between 0.1 and 3.0")

    # Check for extreme combinations
    total_multiplier = seniority_weight * performance_weight * flexibility_score
    if total_multiplier > 10.0:
        errors.append(
            "Combined weight multiplier too high (> 10.0) - may cause unfair distribution"
        )
    elif total_multiplier < 0.1:
        errors.append(
            "Combined weight multiplier too low (< 0.1) - staff may not receive adequate hours"
        )

    return errors


def format_money(amount: float, currency: str = "$") -> str:
    """Format monetary amount"""
    return f"{currency}{amount:,.2f}"


def parse_csv_line(line: str, delimiter: str = ",") -> List[str]:
    """Parse CSV line handling quoted fields"""
    import csv
    import io

    reader = csv.reader(io.StringIO(line), delimiter=delimiter)
    return next(reader, [])


def generate_color_for_staff(staff_id: str) -> str:
    """Generate consistent color for staff member based on ID"""
    # Use hash to generate consistent color
    hash_value = hash(staff_id)

    # Generate HSL color for better distinctiveness
    hue = hash_value % 360
    saturation = 70 + (hash_value % 30)  # 70-100%
    lightness = 40 + (hash_value % 20)  # 40-60%

    # Convert HSL to hex (simplified)
    return f"hsl({hue}, {saturation}%, {lightness}%)"


def calculate_schedule_density(
    schedule: List[Dict], total_possible_hours: int
) -> float:
    """Calculate how densely packed a schedule is"""
    if total_possible_hours == 0:
        return 0.0

    total_scheduled_hours = sum(shift.get("scheduled_hours", 0) for shift in schedule)
    return total_scheduled_hours / total_possible_hours


def find_schedule_gaps(shifts: List[Dict], date_obj: date) -> List[Tuple[time, time]]:
    """Find gaps in a day's schedule"""
    if not shifts:
        return []

    # Sort shifts by start time
    sorted_shifts = sorted(shifts, key=lambda s: s.get("start_time", time(0)))

    gaps = []
    for i in range(len(sorted_shifts) - 1):
        current_end = sorted_shifts[i].get("end_time")
        next_start = sorted_shifts[i + 1].get("start_time")

        if current_end and next_start and current_end < next_start:
            # Calculate gap duration
            gap_duration = (
                datetime.combine(date_obj, next_start)
                - datetime.combine(date_obj, current_end)
            ).total_seconds() / 60  # minutes

            # Only include significant gaps (> 15 minutes)
            if gap_duration > 15:
                gaps.append((current_end, next_start))

    return gaps


def suggest_shift_consolidation(shifts: List[Dict]) -> List[str]:
    """Suggest ways to consolidate fragmented shifts"""
    suggestions = []

    if len(shifts) <= 1:
        return suggestions

    # Group shifts by date
    daily_shifts = {}
    for shift in shifts:
        shift_date = shift.get("date")
        if shift_date not in daily_shifts:
            daily_shifts[shift_date] = []
        daily_shifts[shift_date].append(shift)

    for date_obj, day_shifts in daily_shifts.items():
        if len(day_shifts) <= 1:
            continue

        gaps = find_schedule_gaps(day_shifts, date_obj)

        if gaps:
            total_gap_minutes = sum(
                (
                    datetime.combine(date_obj, end) - datetime.combine(date_obj, start)
                ).total_seconds()
                / 60
                for start, end in gaps
            )

            if total_gap_minutes < 120:  # Less than 2 hours of gaps
                suggestions.append(
                    f"Consider consolidating {len(day_shifts)} shifts on {date_obj} "
                    f"(total gaps: {total_gap_minutes:.0f} minutes)"
                )

    return suggestions


def calculate_workload_balance(staff_hours: Dict[str, float]) -> Dict[str, float]:
    """Calculate workload balance metrics"""
    if not staff_hours:
        return {}

    hours_list = list(staff_hours.values())

    if len(hours_list) == 1:
        return {
            "balance_score": 1.0,
            "coefficient_of_variation": 0.0,
            "range_ratio": 0.0,
        }

    mean_hours = sum(hours_list) / len(hours_list)
    variance = sum((h - mean_hours) ** 2 for h in hours_list) / len(hours_list)
    std_dev = variance**0.5

    coefficient_of_variation = std_dev / mean_hours if mean_hours > 0 else 0
    range_ratio = (
        (max(hours_list) - min(hours_list)) / mean_hours if mean_hours > 0 else 0
    )

    # Balance score: 1.0 is perfect balance, lower is worse
    balance_score = max(0, 1 - coefficient_of_variation)

    return {
        "balance_score": balance_score,
        "coefficient_of_variation": coefficient_of_variation,
        "range_ratio": range_ratio,
        "mean_hours": mean_hours,
        "std_dev": std_dev,
    }


def estimate_optimization_complexity(
    staff_count: int, group_count: int, time_slots: int, constraints_count: int
) -> str:
    """Estimate optimization problem complexity"""

    # Calculate problem size factors
    decision_variables = staff_count * group_count * time_slots
    constraint_density = (
        constraints_count / decision_variables if decision_variables > 0 else 0
    )

    if decision_variables < 1000:
        size_category = "small"
    elif decision_variables < 10000:
        size_category = "medium"
    elif decision_variables < 50000:
        size_category = "large"
    else:
        size_category = "very_large"

    if constraint_density < 0.1:
        constraint_category = "sparse"
    elif constraint_density < 0.5:
        constraint_category = "moderate"
    else:
        constraint_category = "dense"

    return f"{size_category}_{constraint_category}"


def log_performance_warning(operation: str, duration: float, threshold: float):
    """Log performance warning if operation exceeds threshold"""
    if duration > threshold:
        logger.warning(
            f"Performance warning: {operation} took {duration:.2f}s "
            f"(threshold: {threshold:.2f}s)"
        )


@dataclass
class ScheduleTemplate:
    """Template for common schedule patterns"""

    name: str
    description: str
    pattern: Dict[str, Any]
    recommended_for: List[str]


class DateRangeHelper:
    """Helper utilities for date range operations"""

    @staticmethod
    def get_week_dates(year: int, week: int) -> Tuple[date, date]:
        """Get start and end dates for a specific week"""

        # Get the first day of the year
        jan_1 = date(year, 1, 1)

        # Calculate the start of the first week
        days_to_monday = jan_1.weekday()
        week_1_start = jan_1 - timedelta(days=days_to_monday)

        # Calculate the target week start
        week_start = week_1_start + timedelta(weeks=week - 1)
        week_end = week_start + timedelta(days=6)

        return week_start, week_end

    @staticmethod
    def get_month_dates(year: int, month: int) -> Tuple[date, date]:
        """Get start and end dates for a specific month"""

        start_date = date(year, month, 1)
        _, last_day = monthrange(year, month)
        end_date = date(year, month, last_day)

        return start_date, end_date

    @staticmethod
    def get_quarter_dates(year: int, quarter: int) -> Tuple[date, date]:
        """Get start and end dates for a specific quarter"""

        if not 1 <= quarter <= 4:
            raise ValueError("Quarter must be between 1 and 4")

        start_month = (quarter - 1) * 3 + 1
        start_date = date(year, start_month, 1)

        end_month = start_month + 2
        _, last_day = monthrange(year, end_month)
        end_date = date(year, end_month, last_day)

        return start_date, end_date

    @staticmethod
    def split_date_range(
        start_date: date, end_date: date, chunk_size: int
    ) -> List[Tuple[date, date]]:
        """Split a date range into smaller chunks"""

        chunks = []
        current_start = start_date

        while current_start <= end_date:
            current_end = min(current_start + timedelta(days=chunk_size - 1), end_date)
            chunks.append((current_start, current_end))
            current_start = current_end + timedelta(days=1)

        return chunks

    @staticmethod
    def get_business_days(start_date: date, end_date: date) -> List[date]:
        """Get list of business days (Monday-Friday) in date range"""

        business_days = []
        current_date = start_date

        while current_date <= end_date:
            if current_date.weekday() < 5:  # Monday = 0, Friday = 4
                business_days.append(current_date)
            current_date += timedelta(days=1)

        return business_days

    @staticmethod
    def get_weekends(start_date: date, end_date: date) -> List[date]:
        """Get list of weekend days in date range"""

        weekends = []
        current_date = start_date

        while current_date <= end_date:
            if current_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
                weekends.append(current_date)
            current_date += timedelta(days=1)

        return weekends


class SchedulePatternGenerator:
    """Generate common scheduling patterns"""

    @staticmethod
    def create_shift_rotation(
        staff_ids: List[UUID], start_date: date, days: int, shifts_per_day: int = 3
    ) -> List[Dict[str, Any]]:
        """Create a rotating shift pattern"""

        patterns = []
        shift_types = [
            {"name": "Morning", "start": time(6, 0), "end": time(14, 0)},
            {"name": "Afternoon", "start": time(14, 0), "end": time(22, 0)},
            {"name": "Night", "start": time(22, 0), "end": time(6, 0)},
        ]

        for day_offset in range(days):
            current_date = start_date + timedelta(days=day_offset)

            for shift_idx in range(shifts_per_day):
                staff_idx = (day_offset * shifts_per_day + shift_idx) % len(staff_ids)
                shift_type = shift_types[shift_idx % len(shift_types)]

                pattern = {
                    "staff_id": staff_ids[staff_idx],
                    "date": current_date,
                    "shift_type": shift_type["name"],
                    "start_time": shift_type["start"],
                    "end_time": shift_type["end"],
                }
                patterns.append(pattern)

        return patterns

    @staticmethod
    def create_weekly_template(template_type: str = "standard") -> ScheduleTemplate:
        """Create a weekly schedule template"""

        templates = {
            "standard": {
                "name": "Standard Business Week",
                "description": "Monday-Friday 9-5 coverage",
                "pattern": {
                    "days": [0, 1, 2, 3, 4],  # Monday-Friday
                    "hours": {"start": 9, "end": 17},
                    "min_staff": 2,
                    "preferred_staff": 3,
                },
                "recommended_for": ["offices", "administration", "customer_service"],
            },
            "24x7": {
                "name": "24/7 Operations",
                "description": "Round-the-clock coverage",
                "pattern": {
                    "days": [0, 1, 2, 3, 4, 5, 6],  # All week
                    "hours": {"start": 0, "end": 24},
                    "shifts": [
                        {"start": 0, "end": 8, "min_staff": 1},
                        {"start": 8, "end": 16, "min_staff": 2},
                        {"start": 16, "end": 24, "min_staff": 1},
                    ],
                },
                "recommended_for": ["healthcare", "security", "manufacturing"],
            },
            "retail": {
                "name": "Retail Schedule",
                "description": "Extended hours with weekend coverage",
                "pattern": {
                    "days": [0, 1, 2, 3, 4, 5, 6],
                    "hours": {"start": 8, "end": 22},
                    "peak_hours": [11, 12, 13, 14, 17, 18, 19],
                    "min_staff": 2,
                    "peak_staff": 4,
                },
                "recommended_for": ["retail", "restaurants", "hospitality"],
            },
        }

        if template_type not in templates:
            raise ValueError(f"Unknown template type: {template_type}")

        template_data = templates[template_type]
        return ScheduleTemplate(**template_data)


class ScheduleAnalyzer:
    """Analyze schedule patterns and provide insights"""

    @staticmethod
    def analyze_coverage(
        schedule: List[ScheduledShift], requirements: List[StaffingRequirement]
    ) -> Dict[str, Any]:
        """Analyze schedule coverage against requirements"""

        coverage_analysis = {
            "total_shifts": len(schedule),
            "coverage_by_day": {},
            "coverage_by_hour": {},
            "understaffed_periods": [],
            "overstaffed_periods": [],
            "coverage_percentage": 0.0,
        }

        # Group shifts by day and hour
        shifts_by_time = {}
        for shift in schedule:
            date_key = shift.date.isoformat()

            # Calculate hourly coverage
            start_hour = shift.start_time.hour
            end_hour = shift.end_time.hour

            for hour in range(start_hour, end_hour):
                time_key = f"{date_key}_{hour:02d}"
                if time_key not in shifts_by_time:
                    shifts_by_time[time_key] = 0
                shifts_by_time[time_key] += 1

        # Compare against requirements
        total_periods = 0
        covered_periods = 0

        for req in requirements:
            req_key = f"{req.time_slot.day_of_week}_{req.time_slot.start_time.hour:02d}"
            actual_staff = shifts_by_time.get(req_key, 0)

            total_periods += 1

            if actual_staff >= req.min_staff_count:
                covered_periods += 1
            elif actual_staff < req.min_staff_count:
                coverage_analysis["understaffed_periods"].append(
                    {
                        "period": req_key,
                        "required": req.min_staff_count,
                        "actual": actual_staff,
                        "shortage": req.min_staff_count - actual_staff,
                    }
                )
            elif req.max_staff_count and actual_staff > req.max_staff_count:
                coverage_analysis["overstaffed_periods"].append(
                    {
                        "period": req_key,
                        "max_allowed": req.max_staff_count,
                        "actual": actual_staff,
                        "excess": actual_staff - req.max_staff_count,
                    }
                )

        coverage_analysis["coverage_percentage"] = (
            (covered_periods / total_periods * 100) if total_periods > 0 else 0
        )

        return coverage_analysis

    @staticmethod
    def analyze_staff_workload(
        schedule: List[ScheduledShift], staff: List[Staff]
    ) -> Dict[UUID, Dict[str, Any]]:
        """Analyze workload distribution across staff"""

        staff_analysis = {}

        # Initialize analysis for each staff member
        for staff_member in staff:
            staff_analysis[staff_member.staff_id] = {
                "name": staff_member.name,
                "total_hours": 0.0,
                "shifts_count": 0,
                "avg_shift_length": 0.0,
                "days_worked": set(),
                "utilization": 0.0,
                "overtime_hours": 0.0,
                "consecutive_days": 0,
                "schedule_pattern": [],
            }

        # Analyze each shift
        for shift in schedule:
            if shift.staff_id in staff_analysis:
                analysis = staff_analysis[shift.staff_id]

                analysis["total_hours"] += shift.scheduled_hours
                analysis["shifts_count"] += 1
                analysis["days_worked"].add(shift.date)
                analysis["schedule_pattern"].append(
                    {
                        "date": shift.date,
                        "start": shift.start_time,
                        "end": shift.end_time,
                        "hours": shift.scheduled_hours,
                    }
                )

        # Calculate derived metrics
        for staff_id, analysis in staff_analysis.items():
            if analysis["shifts_count"] > 0:
                analysis["avg_shift_length"] = (
                    analysis["total_hours"] / analysis["shifts_count"]
                )

            # Find corresponding staff member for max hours
            staff_member = next((s for s in staff if s.staff_id == staff_id), None)
            if staff_member:
                analysis["utilization"] = (
                    analysis["total_hours"] / staff_member.max_weekly_hours
                )
                analysis["overtime_hours"] = max(
                    0, analysis["total_hours"] - 40
                )  # Assuming 40h standard

            # Calculate consecutive working days
            if analysis["days_worked"]:
                sorted_days = sorted(analysis["days_worked"])
                current_consecutive = 1
                max_consecutive = 1

                for i in range(1, len(sorted_days)):
                    if (sorted_days[i] - sorted_days[i - 1]).days == 1:
                        current_consecutive += 1
                        max_consecutive = max(max_consecutive, current_consecutive)
                    else:
                        current_consecutive = 1

                analysis["consecutive_days"] = max_consecutive

            # Convert days_worked set to count
            analysis["days_worked"] = len(analysis["days_worked"])

        return staff_analysis

    @staticmethod
    def identify_schedule_conflicts(
        schedule: List[ScheduledShift],
    ) -> List[Dict[str, Any]]:
        """Identify conflicts in the schedule"""

        conflicts = []

        # Group shifts by staff
        staff_shifts = {}
        for shift in schedule:
            if shift.staff_id not in staff_shifts:
                staff_shifts[shift.staff_id] = []
            staff_shifts[shift.staff_id].append(shift)

        # Check for overlapping shifts
        for staff_id, shifts in staff_shifts.items():
            shifts.sort(key=lambda s: (s.date, s.start_time))

            for i in range(len(shifts) - 1):
                current = shifts[i]
                next_shift = shifts[i + 1]

                # Check for same-day overlaps
                if current.date == next_shift.date:
                    current_end = datetime.combine(current.date, current.end_time)
                    next_start = datetime.combine(
                        next_shift.date, next_shift.start_time
                    )

                    if current_end > next_start:
                        conflicts.append(
                            {
                                "type": "overlapping_shifts",
                                "staff_id": staff_id,
                                "shift_1": {
                                    "date": current.date,
                                    "time": f"{current.start_time}-{current.end_time}",
                                },
                                "shift_2": {
                                    "date": next_shift.date,
                                    "time": f"{next_shift.start_time}-{next_shift.end_time}",
                                },
                                "overlap_minutes": int(
                                    (current_end - next_start).total_seconds() / 60
                                ),
                            }
                        )

        return conflicts


class ScheduleExportHelper:
    """Helper for exporting schedules in various formats"""

    @staticmethod
    def export_to_csv(
        schedule: List[ScheduledShift],
        staff: List[Staff] = None,
        groups: List[Group] = None,
    ) -> str:
        """Export schedule to CSV format"""

        # Create lookup dictionaries
        staff_lookup = {s.staff_id: s for s in (staff or [])}
        group_lookup = {g.group_id: g for g in (groups or [])}

        output = StringIO()
        writer = csv.writer(output)

        # Write header
        headers = [
            "Date",
            "Day of Week",
            "Staff ID",
            "Staff Name",
            "Group ID",
            "Group Name",
            "Start Time",
            "End Time",
            "Hours",
            "Status",
        ]
        writer.writerow(headers)

        # Write data
        for shift in sorted(schedule, key=lambda s: (s.date, s.start_time)):
            staff_name = staff_lookup.get(
                shift.staff_id, type("obj", (object,), {"name": "Unknown"})
            ).name
            group_name = group_lookup.get(
                shift.group_id, type("obj", (object,), {"name": "Unknown"})
            ).name

            row = [
                shift.date.strftime("%Y-%m-%d"),
                shift.date.strftime("%A"),
                str(shift.staff_id),
                staff_name,
                str(shift.group_id),
                group_name,
                shift.start_time.strftime("%H:%M"),
                shift.end_time.strftime("%H:%M"),
                shift.scheduled_hours,
                shift.status.value if shift.status else "SCHEDULED",
            ]
            writer.writerow(row)

        content = output.getvalue()
        output.close()
        return content

    @staticmethod
    def export_to_calendar_json(
        schedule: List[ScheduledShift],
        staff: List[Staff] = None,
        groups: List[Group] = None,
    ) -> str:
        """Export schedule to calendar JSON format"""

        staff_lookup = {s.staff_id: s for s in (staff or [])}
        group_lookup = {g.group_id: g for g in (groups or [])}

        events = []

        for shift in schedule:
            staff_name = staff_lookup.get(
                shift.staff_id, type("obj", (object,), {"name": "Unknown"})
            ).name
            group_name = group_lookup.get(
                shift.group_id, type("obj", (object,), {"name": "Unknown"})
            ).name

            event = {
                "id": f"{shift.staff_id}_{shift.date}_{shift.start_time}",
                "title": f"{staff_name} - {group_name}",
                "start": f"{shift.date}T{shift.start_time}",
                "end": f"{shift.date}T{shift.end_time}",
                "allDay": False,
                "extendedProps": {
                    "staffId": str(shift.staff_id),
                    "staffName": staff_name,
                    "groupId": str(shift.group_id),
                    "groupName": group_name,
                    "hours": shift.scheduled_hours,
                    "status": shift.status.value if shift.status else "SCHEDULED",
                },
            }
            events.append(event)

        calendar_data = {
            "events": events,
            "metadata": {
                "total_shifts": len(schedule),
                "date_range": {
                    "start": min(s.date for s in schedule).isoformat()
                    if schedule
                    else None,
                    "end": max(s.date for s in schedule).isoformat()
                    if schedule
                    else None,
                },
                "export_time": datetime.now().isoformat(),
            },
        }

        return json.dumps(calendar_data, indent=2)


class ScheduleImportHelper:
    """Helper for importing schedules from various formats"""

    @staticmethod
    def import_from_csv(csv_content: str) -> Tuple[List[ScheduledShift], List[str]]:
        """Import schedule from CSV content"""

        shifts = []
        errors = []

        try:
            csv_reader = csv.DictReader(StringIO(csv_content))

            for row_num, row in enumerate(csv_reader, 2):
                try:
                    shift = ScheduledShift(
                        staff_id=UUID(row["Staff ID"]),
                        group_id=UUID(row["Group ID"]),
                        date=datetime.strptime(row["Date"], "%Y-%m-%d").date(),
                        start_time=datetime.strptime(row["Start Time"], "%H:%M").time(),
                        end_time=datetime.strptime(row["End Time"], "%H:%M").time(),
                        scheduled_hours=float(row["Hours"]),
                        status=ShiftStatus(row.get("Status", "SCHEDULED")),
                    )
                    shifts.append(shift)

                except Exception as e:
                    errors.append(f"Row {row_num}: {str(e)}")

        except Exception as e:
            errors.append(f"CSV parsing error: {str(e)}")

        return shifts, errors


class ScheduleValidationHelper:
    """Helper for schedule validation operations"""

    @staticmethod
    def validate_date_range(start_date: date, end_date: date) -> List[str]:
        """Validate a date range"""

        errors = []

        if end_date < start_date:
            errors.append("End date must be after start date")

        total_days = (end_date - start_date).days + 1

        if total_days > settings.MAX_DATE_RANGE_DAYS:
            errors.append(
                f"Date range too large: {total_days} days (max: {settings.MAX_DATE_RANGE_DAYS})"
            )

        if start_date < date.today() - timedelta(days=365):
            errors.append("Start date is too far in the past")

        if end_date > date.today() + timedelta(days=730):
            errors.append("End date is too far in the future")

        return errors

    @staticmethod
    def validate_staff_data(staff: List[Staff]) -> List[str]:
        """Validate staff data"""

        errors = []

        if not staff:
            errors.append("No staff provided")
            return errors

        staff_ids = set()
        for i, staff_member in enumerate(staff):
            if staff_member.staff_id in staff_ids:
                errors.append(f"Duplicate staff ID: {staff_member.staff_id}")
            staff_ids.add(staff_member.staff_id)

            if staff_member.max_weekly_hours <= 0:
                errors.append(
                    f"Staff {staff_member.name}: max_weekly_hours must be positive"
                )

            if staff_member.max_weekly_hours > 168:  # 24 * 7
                errors.append(
                    f"Staff {staff_member.name}: max_weekly_hours cannot exceed 168"
                )

            # Validate availability
            total_available_hours = 0
            for avail in staff_member.availability:
                if avail.is_available and avail.start_time and avail.end_time:
                    if avail.end_time <= avail.start_time:
                        errors.append(
                            f"Staff {staff_member.name}: invalid availability time range"
                        )
                    else:
                        daily_hours = (
                            datetime.combine(date.today(), avail.end_time)
                            - datetime.combine(date.today(), avail.start_time)
                        ).total_seconds() / 3600
                        total_available_hours += daily_hours

            if (
                total_available_hours > 0
                and staff_member.max_weekly_hours > total_available_hours
            ):
                errors.append(
                    f"Staff {staff_member.name}: max_weekly_hours exceeds available hours"
                )

        return errors

    @staticmethod
    def validate_requirements(requirements: List[StaffingRequirement]) -> List[str]:
        """Validate staffing requirements"""

        errors = []

        if not requirements:
            errors.append("No staffing requirements provided")
            return errors

        for i, req in enumerate(requirements):
            if req.min_staff_count <= 0:
                errors.append(f"Requirement {i}: min_staff_count must be positive")

            if req.max_staff_count and req.max_staff_count < req.min_staff_count:
                errors.append(
                    f"Requirement {i}: max_staff_count cannot be less than min_staff_count"
                )

            if req.time_slot.end_time <= req.time_slot.start_time:
                errors.append(f"Requirement {i}: invalid time slot range")

        return errors


class ScheduleOptimizationHelper:
    """Helper for optimization-related operations"""

    @staticmethod
    def estimate_solve_time(
        total_days: int, staff_count: int, groups_count: int, requirements_count: int
    ) -> Dict[str, Any]:
        """Estimate solve time and complexity"""

        # Calculate problem size metrics
        hours_per_day = settings.total_operating_hours
        total_time_slots = total_days * hours_per_day
        estimated_variables = staff_count * groups_count * total_time_slots

        # Estimate complexity
        complexity_score = (
            (total_days / 30) * 0.3
            + (staff_count / 20) * 0.25
            + (groups_count / 10) * 0.2
            + (estimated_variables / 100000) * 0.25
        )

        # Estimate solve time
        if complexity_score < 0.3:
            estimated_time = "30 seconds - 2 minutes"
            complexity = "Low"
        elif complexity_score < 0.7:
            estimated_time = "2 - 10 minutes"
            complexity = "Medium"
        elif complexity_score < 1.2:
            estimated_time = "10 - 30 minutes"
            complexity = "High"
        else:
            estimated_time = "30+ minutes"
            complexity = "Very High"

        # Suggest strategy
        if total_days > 21:
            suggested_strategy = "chunked"
        elif groups_count > 5 and staff_count > 10:
            suggested_strategy = "parallel"
        else:
            suggested_strategy = "standard"

        return {
            "complexity": complexity,
            "complexity_score": complexity_score,
            "estimated_solve_time": estimated_time,
            "estimated_variables": estimated_variables,
            "total_time_slots": total_time_slots,
            "suggested_strategy": suggested_strategy,
            "recommendations": [
                f"Use {suggested_strategy} optimization strategy",
                "Ensure all input data is validated",
                "Consider chunking if solve time exceeds expectations",
            ],
        }

    @staticmethod
    def get_optimization_config_recommendations(
        total_days: int, staff_count: int, problem_type: str = "general"
    ) -> Dict[str, Any]:
        """Get optimization configuration recommendations"""

        base_time = 300  # 5 minutes

        # Adjust time based on problem size
        if total_days > 14:
            solver_time = min(base_time * (total_days // 7), 1800)  # Max 30 minutes
        else:
            solver_time = base_time

        # Goal recommendations by problem type
        goal_recommendations = {
            "general": [
                OptimizationGoal.MAXIMIZE_SATISFACTION,
                OptimizationGoal.MINIMIZE_COST,
                OptimizationGoal.MAXIMIZE_FAIRNESS,
            ],
            "healthcare": [
                OptimizationGoal.MAXIMIZE_SATISFACTION,
                OptimizationGoal.MAXIMIZE_CONTINUITY,
                OptimizationGoal.MINIMIZE_OVERTIME,
            ],
            "retail": [
                OptimizationGoal.MINIMIZE_COST,
                OptimizationGoal.MAXIMIZE_FAIRNESS,
                OptimizationGoal.MAXIMIZE_SATISFACTION,
            ],
            "manufacturing": [
                OptimizationGoal.MAXIMIZE_CONTINUITY,
                OptimizationGoal.MINIMIZE_COST,
                OptimizationGoal.MAXIMIZE_FAIRNESS,
            ],
        }

        return {
            "max_solver_time": solver_time,
            "recommended_goals": goal_recommendations.get(
                problem_type, goal_recommendations["general"]
            ),
            "consider_preferences": True,
            "balance_workload_over_period": total_days > 7,
            "max_consecutive_days": min(6, total_days),
            "min_days_off_per_week": 1 if total_days >= 7 else 0,
        }


# Convenience functions for common operations


def create_quick_weekly_schedule(
    center_id: UUID, week_start: date, staff_names: List[str], group_names: List[str]
) -> EnhancedScheduleGenerationRequest:
    """Quickly create a basic weekly schedule request"""

    # Create staff objects
    staff = [
        Staff(
            staff_id=uuid4(),
            name=name,
            max_weekly_hours=40,
            availability=[
                StaffAvailability(
                    day_of_week=i,
                    start_time=time(9, 0),
                    end_time=time(17, 0),
                    is_available=True,
                )
                for i in range(5)  # Monday-Friday
            ],
        )
        for name in staff_names
    ]

    # Create groups
    groups = [Group(group_id=uuid4(), name=name, capacity=10) for name in group_names]

    # Create basic requirements
    requirements = [
        StaffingRequirement(
            group_id=group.group_id,
            time_slot=TimeSlot(
                start_time=time(9, 0), end_time=time(17, 0), day_of_week=None
            ),
            min_staff_count=1,
            max_staff_count=3,
        )
        for group in groups
    ]

    # Create basic center configuration
    center_config = CenterConfiguration(
        center_id=center_id,
        name="Default Center",
        opening_hours=[
            TimeSlot(day_of_week=i, start_time=time(7, 0), end_time=time(18, 0))
            for i in range(7)  # Monday to Sunday
        ]
    )

    return EnhancedScheduleGenerationRequest(
        center_id=center_id,
        schedule_start_date=week_start,
        schedule_end_date=week_start + timedelta(days=6),
        staff=staff,
        groups=groups,
        staffing_requirements=requirements,
        center_config=center_config,
        optimization_config=EnhancedOptimizationConfig(),
    )


def validate_schedule_request(request: EnhancedScheduleGenerationRequest) -> List[str]:
    """Validate a complete schedule request"""

    all_errors = []

    # Date range validation
    date_errors = ScheduleValidationHelper.validate_date_range(
        request.schedule_start_date, request.effective_end_date
    )
    all_errors.extend(date_errors)

    # Staff validation
    staff_errors = ScheduleValidationHelper.validate_staff_data(request.staff)
    all_errors.extend(staff_errors)

    # Requirements validation
    req_errors = ScheduleValidationHelper.validate_requirements(
        request.staffing_requirements
    )
    all_errors.extend(req_errors)

    return all_errors


def get_schedule_summary(schedule: List[ScheduledShift]) -> Dict[str, Any]:
    """Get a summary of a schedule"""

    if not schedule:
        return {"message": "No shifts in schedule"}

    dates = [shift.date for shift in schedule]
    total_hours = sum(shift.scheduled_hours for shift in schedule)

    return {
        "total_shifts": len(schedule),
        "total_hours": total_hours,
        "date_range": {
            "start": min(dates),
            "end": max(dates),
            "days": (max(dates) - min(dates)).days + 1,
        },
        "average_shift_length": total_hours / len(schedule),
        "unique_staff": len(set(shift.staff_id for shift in schedule)),
        "unique_groups": len(set(shift.group_id for shift in schedule)),
    }
