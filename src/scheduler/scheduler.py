"""
Scheduler module - imports from core.optimizer for backward compatibility
"""

# Import everything from the properly organized core.optimizer module
from .core.optimizer import (
    ScheduleOptimizer,
    debug_schedule_constraints,
    find_time_overlap,
)

# Export for backward compatibility
__all__ = ['ScheduleOptimizer', 'debug_schedule_constraints', 'find_time_overlap']
