#!/usr/bin/env python3
"""
Debug script to identify validation issues
"""

import json
from pydantic import ValidationError
from src.scheduler.models import EnhancedScheduleGenerationRequest

# Sample problematic data (simplified)
test_data = {
    "center_id": "7a17528b-0497-43a7-8824-6bfa5e238da3",
    "schedule_start_date": "2025-06-23",
    "schedule_end_date": "2025-06-29",
    "staff": [
        {
            "staff_id": "ecb88ef0-cefa-41e9-92de-d5009007e02b",
            "name": "Camilla Gumprich",
            "role": "staff",  # INVALID: should be teacher/assistant/etc
            "qualifications": [],
            "availability": [],
            "preferences": [],
            "absences": [],
            "group_assignments": [],
            "seniority_weight": "1.00",  # INVALID: should be float
            "performance_weight": "1.00",  # INVALID: should be float
            "flexibility_score": "1.00",  # INVALID: should be float
            "overall_priority": "medium",
            "max_weekly_hours": 40,
            "hourly_rate": 25,
            "max_daily_hours": 10,
            "max_shift_hours": 8
        }
    ],
    "groups": [
        {
            "group_id": "26b54028-b2c8-4fe3-b0ee-0b121f5309dd",
            "name": "Gruppe Hellelb",
            "age_group": "preschool",
            "capacity": 12,
            "current_enrollment": "1",  # INVALID: should be int
            "required_qualifications": [],
            "preferred_qualifications": []
        }
    ],
    "staffing_requirements": [
        {
            "group_id": "26b54028-b2c8-4fe3-b0ee-0b121f5309dd",
            "time_slot": {
                "start_time": "09:00:00",
                "end_time": "17:00:00"
            },
            "min_staff_count": 2,
            "max_staff_count": 3,
            "required_qualifications": []
        }
    ],
    "center_config": {
        "center_id": "7a17528b-0497-43a7-8824-6bfa5e238da3",
        "name": "Kita Tschentscher",
        "opening_hours": [
            {
                "day_of_week": 0,
                "start_time": "07:00:00",
                "end_time": "18:00:00"
            }
        ],
        "staff_child_ratios": {
            "infant": 4,
            "toddler": 6,
            "preschool": 10,
            "mixed": 8
        },
        "max_daily_overtime_hours": 2,
        "max_weekly_overtime_hours": 10,
        "overtime_threshold_daily": 8,
        "overtime_threshold_weekly": 40,
        "min_break_between_shifts_hours": 10
    },
    "optimization_config": {
        "goals": ["maximize_satisfaction", "minimize_overtime"],
        "max_solver_time": 300,
        "consider_preferences": True
    },
    "existing_schedules": [
        {
            "schedule_id": "e778f90e-c5f3-4fd5-bd74-dfda313b01d4",
            "staff_id": "f62bc776-4c86-4f05-b69c-810a3c65026c",
            "group_id": "a7cf684c-5612-455a-98b2-b7852ac39293",
            "date": "2025-06-22",  # INVALID: outside date range (2025-06-23 to 2025-06-29)
            "start_time": "09:00:00",
            "end_time": "17:00:00",
            "scheduled_hours": "7.03",  # INVALID: should be float
            "status": "scheduled"
        }
    ]
}

def debug_validation():
    """Test validation and show specific errors"""
    
    try:
        # Try to create the request object
        request = EnhancedScheduleGenerationRequest(**test_data)
        print("✅ Validation passed!")
        
    except ValidationError as e:
        print("❌ Validation failed:")
        print("\nSpecific errors:")
        
        for error in e.errors():
            print(f"- Field: {' -> '.join(str(x) for x in error['loc'])}")
            print(f"  Error: {error['msg']}")
            print(f"  Input: {error['input']}")
            print(f"  Type: {error['type']}")
            print()
            
    except Exception as e:
        print(f"❌ Other error: {e}")

if __name__ == "__main__":
    debug_validation()