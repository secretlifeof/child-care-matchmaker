#!/usr/bin/env python3

import logging
import json
from datetime import date, datetime, time
from typing import List, Dict, Tuple
import sys
import os

# Add the src directory to the path  
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

# Import what we need
from scheduler.enhanced_solver import EnhancedScheduleSolver
from scheduler.models import *

def test_failing_case():
    print("=== TESTING FAILING CASE: 2-day schedule with existing schedules ===")
    
    with open('test_more_staff_2days.json', 'r') as f:
        data = json.load(f)
    
    # Convert the JSON data to our models
    staff = [Staff(**s) for s in data['staff']]
    groups = [Group(**g) for g in data['groups']]
    requirements = [StaffingRequirement(**r) for r in data['staffing_requirements']]
    existing_schedules = [ScheduledShift(**s) for s in data.get('existing_schedules', [])]
    
    # Create config
    config = OptimizationConfig(**data['optimization_config'])
    
    # Create dates
    start_date = datetime.fromisoformat(data['schedule_start_date']).date()
    end_date = datetime.fromisoformat(data['schedule_end_date']).date()
    
    print(f"Staff: {len(staff)}, Groups: {len(groups)}, Requirements: {len(requirements)}")
    print(f"Existing schedules: {len(existing_schedules)}")
    print(f"Date range: {start_date} to {end_date}")
    
    # Create solver
    solver = EnhancedScheduleSolver(config)
    
    # Solve
    try:
        schedule, result, conflicts = solver.solve_with_date_range(
            staff=staff,
            groups=groups,
            requirements=requirements,
            constraints=[],
            schedule_start_date=start_date,
            schedule_end_date=end_date,
            existing_schedule=existing_schedules
        )
        
        print(f"RESULT: {result.status}")
        print(f"Generated {len(schedule)} shifts")
        if conflicts:
            print(f"Conflicts: {[c.description for c in conflicts]}")
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_failing_case()