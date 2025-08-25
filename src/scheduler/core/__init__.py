"""
Core business logic for the Daycare Schedule Optimizer

This module contains the main business logic components:
- Schedule optimization engine
- Validation and analysis engines  
- Caching and performance utilities
- Core algorithms and data processing
"""

from .optimizer import ScheduleOptimizer
from .validator import ScheduleValidator
from .analyzer import ScheduleAnalyzer
from .cache import CacheManager, ScheduleCacheHelper

__all__ = [
    "ScheduleOptimizer",
    "ScheduleValidator", 
    "ScheduleAnalyzer",
    "CacheManager",
    "ScheduleCacheHelper"
]

# Version information for core components
CORE_VERSION = "1.0.0"

# Core component registry
CORE_COMPONENTS = {
    "optimizer": ScheduleOptimizer,
    "validator": ScheduleValidator,
    "analyzer": ScheduleAnalyzer,
    "cache_manager": CacheManager,
    "cache_helper": ScheduleCacheHelper
}

def get_core_component(component_name: str):
    """Get a core component class by name"""
    return CORE_COMPONENTS.get(component_name)

def list_core_components():
    """List all available core components"""
    return list(CORE_COMPONENTS.keys())