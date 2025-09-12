"""Plugin system for complex preference processing."""

import asyncio
import logging
import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from uuid import UUID

from ..models.base import Center, Location, ParentPreference
from ..services.map_service import get_map_service

logger = logging.getLogger(__name__)


class PluginResult(Enum):
    """Plugin processing result types."""
    SUCCESS = "success"
    USER_INTERACTION_REQUIRED = "user_interaction_required"
    ERROR = "error"
    SKIP = "skip"


@dataclass
class InteractionRequired:
    """Represents a user interaction requirement."""
    interaction_type: str  # "location_input", "confirmation", "choice"
    question: str
    field_name: str
    options: Optional[List[Dict[str, Any]]] = None
    context: Optional[Dict[str, Any]] = None


@dataclass
class PluginProcessingResult:
    """Result of plugin processing."""
    status: PluginResult
    data: Optional[Dict[str, Any]] = None
    interaction: Optional[InteractionRequired] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ProcessingPlugin(ABC):
    """Base class for processing plugins."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.name = self.__class__.__name__
    
    @abstractmethod
    async def process(
        self, 
        data: Dict[str, Any], 
        preference: ParentPreference,
        center: Center,
        context: Dict[str, Any]
    ) -> PluginProcessingResult:
        """Process the data and return result."""
        pass
    
    @abstractmethod
    def get_plugin_name(self) -> str:
        """Get the plugin identifier."""
        pass
    
    def validate_input(self, data: Dict[str, Any]) -> bool:
        """Validate input data format."""
        return True
    
    def get_required_context(self) -> List[str]:
        """Get list of required context keys."""
        return []


class LocationResolverPlugin(ProcessingPlugin):
    """Plugin for resolving location references to coordinates."""
    
    def get_plugin_name(self) -> str:
        return "location_resolver"
    
    async def process(
        self, 
        data: Dict[str, Any], 
        preference: ParentPreference,
        center: Center,
        context: Dict[str, Any]
    ) -> PluginProcessingResult:
        """Resolve location references to lat/lng coordinates."""
        
        locations = data.get('locations', [])
        updated_locations = []
        
        for location in locations:
            resolved_location = await self._resolve_single_location(
                location, preference, context
            )
            
            if isinstance(resolved_location, InteractionRequired):
                return PluginProcessingResult(
                    status=PluginResult.USER_INTERACTION_REQUIRED,
                    interaction=resolved_location
                )
            
            updated_locations.append(resolved_location)
        
        return PluginProcessingResult(
            status=PluginResult.SUCCESS,
            data={**data, 'locations': updated_locations}
        )
    
    async def _resolve_single_location(
        self, 
        location: Dict[str, Any], 
        preference: ParentPreference,
        context: Dict[str, Any]
    ) -> Dict[str, Any] | InteractionRequired:
        """Resolve a single location to coordinates."""
        
        # Already has coordinates
        if 'latitude' in location and 'longitude' in location:
            return location
        
        location_name = location.get('name', '').lower()
        
        # Handle home location
        if location_name == 'home':
            parent_location = await self._get_parent_home_location(
                preference.profile_id, context
            )
            if parent_location:
                location.update({
                    'latitude': parent_location.latitude,
                    'longitude': parent_location.longitude,
                    'resolved_address': parent_location.address
                })
                return location
        
        # Handle work location
        elif location_name == 'work':
            work_location = await self._get_parent_work_location(
                preference.profile_id, context
            )
            if work_location:
                location.update({
                    'latitude': work_location.latitude,
                    'longitude': work_location.longitude,
                    'resolved_address': work_location.address
                })
                return location
            else:
                # Need user input for work location
                return InteractionRequired(
                    interaction_type="location_input",
                    question="Where is your work located?",
                    field_name="work_location",
                    context={
                        'preference_id': str(preference.id),
                        'location_reference': location
                    }
                )
        
        # Handle specific address
        elif 'address' in location:
            coordinates = await self._geocode_address(location['address'], context)
            if coordinates:
                location.update(coordinates)
                return location
            else:
                return InteractionRequired(
                    interaction_type="confirmation",
                    question=f"Could not find location '{location['address']}'. Please verify or provide a different address.",
                    field_name="corrected_address",
                    context={'original_address': location['address']}
                )
        
        # Unknown location format
        return InteractionRequired(
            interaction_type="location_input",
            question=f"Please provide coordinates or address for '{location_name}'",
            field_name="location_details",
            context={'location_reference': location}
        )
    
    async def _get_parent_home_location(
        self, 
        parent_id: UUID, 
        context: Dict[str, Any]
    ) -> Optional[Location]:
        """Get parent's home location from database."""
        if 'db_repository' not in context:
            return None
        
        # This would query the parent's profile for home address
        # For now, return None to simulate missing data
        return None
    
    async def _get_parent_work_location(
        self, 
        parent_id: UUID, 
        context: Dict[str, Any]
    ) -> Optional[Location]:
        """Get parent's work location from database."""
        if 'db_repository' not in context:
            return None
        
        # This would query for stored work location
        return None
    
    async def _geocode_address(
        self, 
        address: str, 
        context: Dict[str, Any]
    ) -> Optional[Dict[str, float]]:
        """Geocode an address to coordinates using MapBox."""
        try:
            # Get MapBox API key from environment
            mapbox_api_key = os.getenv('MAPBOX_API_KEY')
            if not mapbox_api_key:
                logger.error("MapBox API key not configured (MAPBOX_API_KEY environment variable)")
                return None
            
            # Get map service instance
            map_service = get_map_service(mapbox_api_key)
            
            # Geocode the address
            result = await map_service.geocode_address(
                address=address,
                country_code='de'  # Focus on Germany, adjust as needed
            )
            
            if result:
                return {
                    'latitude': result.latitude,
                    'longitude': result.longitude,
                    'formatted_address': result.formatted_address,
                    'confidence': result.confidence
                }
            else:
                logger.warning(f"No geocoding results for address: {address}")
                return None
                        
        except Exception as e:
            logger.error(f"MapBox geocoding failed for {address}: {e}")
            return None


class DistanceCalculatorPlugin(ProcessingPlugin):
    """Plugin for calculating distances with different transport modes."""
    
    def get_plugin_name(self) -> str:
        return "distance_calculator"
    
    async def process(
        self, 
        data: Dict[str, Any], 
        preference: ParentPreference,
        center: Center,
        context: Dict[str, Any]
    ) -> PluginProcessingResult:
        """Calculate distances and travel times for all locations."""
        
        locations = data.get('locations', [])
        
        for location in locations:
            if 'latitude' not in location or 'longitude' not in location:
                return PluginProcessingResult(
                    status=PluginResult.ERROR,
                    error_message=f"Location {location.get('name', 'unknown')} missing coordinates"
                )
            
            # Calculate straight-line distance
            straight_distance = self._calculate_haversine_distance(
                location['latitude'], location['longitude'],
                center.location.latitude, center.location.longitude
            )
            
            location['straight_distance_km'] = straight_distance
            
            # Calculate travel time based on transport mode
            transport_mode = location.get('transport_mode', 'walking')
            travel_time = await self._calculate_travel_time(
                location, center.location, transport_mode, context
            )
            
            location['travel_time_minutes'] = travel_time
            location['transport_mode'] = transport_mode
        
        return PluginProcessingResult(
            status=PluginResult.SUCCESS,
            data={**data, 'locations': locations}
        )
    
    def _calculate_haversine_distance(
        self, 
        lat1: float, lon1: float, 
        lat2: float, lon2: float
    ) -> float:
        """Calculate straight-line distance using Haversine formula."""
        import math
        
        R = 6371  # Earth's radius in km
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat/2)**2 + 
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
             math.sin(dlon/2)**2)
        c = 2 * math.asin(math.sqrt(a))
        return R * c
    
    async def _calculate_travel_time(
        self, 
        location: Dict[str, Any], 
        center_location: Location,
        transport_mode: str,
        context: Dict[str, Any]
    ) -> float:
        """Calculate travel time based on transport mode."""
        
        distance_km = location['straight_distance_km']
        
        # Speed estimates by transport mode (km/h)
        speed_estimates = {
            'walking': 5,
            'bike': 15,
            'cycling': 15,
            'car': 30,
            'driving': 30,
            'public_transport': 25,
            'transit': 25
        }
        
        speed_kmh = speed_estimates.get(transport_mode, 5)  # Default to walking
        travel_time_hours = distance_km / speed_kmh
        travel_time_minutes = travel_time_hours * 60
        
        # Add buffer for non-walking modes (traffic, waiting, etc.)
        if transport_mode in ['car', 'driving']:
            travel_time_minutes *= 1.3  # 30% buffer for traffic
        elif transport_mode in ['public_transport', 'transit']:
            travel_time_minutes *= 1.5  # 50% buffer for connections/waiting
        
        return travel_time_minutes


class LocationScoringPlugin(ProcessingPlugin):
    """Plugin for scoring location constraints."""
    
    def get_plugin_name(self) -> str:
        return "location_scorer"
    
    async def process(
        self, 
        data: Dict[str, Any], 
        preference: ParentPreference,
        center: Center,
        context: Dict[str, Any]
    ) -> PluginProcessingResult:
        """Score the location constraints."""
        
        locations = data.get('locations', [])
        if not locations:
            return PluginProcessingResult(
                status=PluginResult.ERROR,
                error_message="No locations to score"
            )
        
        location_scores = []
        explanations = []
        
        for location in locations:
            score, explanation = self._score_single_location(location)
            location_scores.append(score)
            explanations.append(explanation)
            location['score'] = score
            location['explanation'] = explanation
        
        # Overall score is minimum (all constraints must be satisfied)
        overall_score = min(location_scores) if location_scores else 0.0
        overall_explanation = " | ".join(explanations)
        
        return PluginProcessingResult(
            status=PluginResult.SUCCESS,
            data={
                'score': overall_score,
                'explanation': overall_explanation,
                'locations': locations,
                'metadata': {
                    'location_scores': location_scores,
                    'scoring_method': 'minimum_constraint'
                }
            }
        )
    
    def _score_single_location(self, location: Dict[str, Any]) -> tuple[float, str]:
        """Score a single location constraint."""
        
        constraint_type = location.get('constraint_type', 'distance')
        name = location.get('name', 'location')
        
        if constraint_type == 'travel_time':
            return self._score_travel_time_constraint(location, name)
        else:
            return self._score_distance_constraint(location, name)
    
    def _score_distance_constraint(self, location: Dict[str, Any], name: str) -> tuple[float, str]:
        """Score distance-based constraint."""
        
        actual_distance = location.get('straight_distance_km', 0)
        max_distance = location.get('max_distance_km', 10.0)
        preferred_distance = location.get('preferred_distance_km', max_distance * 0.6)
        
        if actual_distance <= preferred_distance:
            return 1.0, f"{name}: ideal distance ({actual_distance:.1f}km)"
        elif actual_distance <= max_distance:
            score = 1.0 - (actual_distance - preferred_distance) / (max_distance - preferred_distance)
            return score, f"{name}: acceptable distance ({actual_distance:.1f}km)"
        else:
            return 0.0, f"{name}: too far ({actual_distance:.1f}km > {max_distance}km)"
    
    def _score_travel_time_constraint(self, location: Dict[str, Any], name: str) -> tuple[float, str]:
        """Score travel time-based constraint."""
        
        actual_time = location.get('travel_time_minutes', 0)
        max_time = location.get('max_travel_minutes', 30)
        preferred_time = location.get('preferred_travel_minutes', max_time * 0.7)
        transport_mode = location.get('transport_mode', 'walking')
        
        if actual_time <= preferred_time:
            return 1.0, f"{name}: ideal {transport_mode} time ({actual_time:.0f}min)"
        elif actual_time <= max_time:
            score = 1.0 - (actual_time - preferred_time) / (max_time - preferred_time)
            return score, f"{name}: acceptable {transport_mode} time ({actual_time:.0f}min)"
        else:
            return 0.0, f"{name}: too long {transport_mode} time ({actual_time:.0f}min > {max_time}min)"


class PluginPipeline:
    """Manages and executes a pipeline of processing plugins."""
    
    def __init__(self, plugins: List[ProcessingPlugin]):
        self.plugins = plugins
        self.name = "->".join([p.get_plugin_name() for p in plugins])
    
    async def execute(
        self,
        initial_data: Dict[str, Any],
        preference: ParentPreference,
        center: Center,
        context: Dict[str, Any]
    ) -> PluginProcessingResult:
        """Execute the plugin pipeline."""
        
        current_data = initial_data.copy()
        pipeline_metadata = {
            'executed_plugins': [],
            'execution_times': [],
            'pipeline_name': self.name
        }
        
        for plugin in self.plugins:
            start_time = asyncio.get_event_loop().time()
            
            try:
                result = await plugin.process(current_data, preference, center, context)
                
                execution_time = asyncio.get_event_loop().time() - start_time
                pipeline_metadata['executed_plugins'].append(plugin.get_plugin_name())
                pipeline_metadata['execution_times'].append(execution_time)
                
                if result.status != PluginResult.SUCCESS:
                    # Pipeline stopped - return the result with metadata
                    if result.metadata:
                        result.metadata.update(pipeline_metadata)
                    else:
                        result.metadata = pipeline_metadata
                    return result
                
                # Continue with updated data
                current_data = result.data or current_data
                
            except Exception as e:
                logger.error(f"Plugin {plugin.get_plugin_name()} failed: {e}")
                return PluginProcessingResult(
                    status=PluginResult.ERROR,
                    error_message=f"Plugin {plugin.get_plugin_name()} failed: {str(e)}",
                    metadata=pipeline_metadata
                )
        
        # All plugins succeeded
        return PluginProcessingResult(
            status=PluginResult.SUCCESS,
            data=current_data,
            metadata=pipeline_metadata
        )


class PluginFactory:
    """Factory for creating plugins from configuration."""
    
    _plugin_registry = {
        'location_resolver': LocationResolverPlugin,
        'distance_calculator': DistanceCalculatorPlugin,
        'location_scorer': LocationScoringPlugin,
    }
    
    @classmethod
    def create_plugin(cls, plugin_name: str, config: Dict[str, Any] = None) -> ProcessingPlugin:
        """Create a plugin instance."""
        if plugin_name not in cls._plugin_registry:
            raise ValueError(f"Unknown plugin: {plugin_name}")
        
        plugin_class = cls._plugin_registry[plugin_name]
        return plugin_class(config)
    
    @classmethod
    def create_pipeline(cls, plugin_configs: List[Dict[str, Any]]) -> PluginPipeline:
        """Create a pipeline from plugin configurations."""
        plugins = []
        for config in plugin_configs:
            plugin_name = config.get('name')
            plugin_config = config.get('config', {})
            plugin = cls.create_plugin(plugin_name, plugin_config)
            plugins.append(plugin)
        
        return PluginPipeline(plugins)
    
    @classmethod
    def register_plugin(cls, plugin_name: str, plugin_class: type):
        """Register a new plugin type."""
        cls._plugin_registry[plugin_name] = plugin_class