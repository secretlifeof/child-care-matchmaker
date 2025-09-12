"""Dynamic agent registry for complex preference processing."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from uuid import UUID
import json

from ..models.base import Center, ParentPreference
from ..database.complex_queries import ComplexPreferenceRepository
from ..processing.plugins import PluginPipeline, PluginFactory, PluginProcessingResult, PluginResult

logger = logging.getLogger(__name__)


class ComplexTypeAgent(ABC):
    """Base class for complex type processing agents."""
    
    def __init__(self, pipeline_config: List[Dict[str, Any]] = None):
        """Initialize agent with optional plugin pipeline."""
        self.pipeline = None
        if pipeline_config:
            self.pipeline = PluginFactory.create_pipeline(pipeline_config)
    
    @abstractmethod
    def get_type_name(self) -> str:
        """Return the complex type name this agent handles."""
        pass
    
    async def process_preference(
        self, 
        preference: ParentPreference, 
        center: Center,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Process a complex preference against a center using plugin pipeline.
        
        Returns:
            Dict with: score, explanation, metadata, semantic_enhancements, user_interaction
        """
        if not self.pipeline:
            return {
                "score": 0.0,
                "explanation": f"No pipeline configured for agent {self.__class__.__name__}",
                "metadata": {"error": "missing_pipeline"},
                "semantic_enhancements": []
            }
        
        # Extract initial data from preference
        initial_data = self._extract_initial_data(preference)
        
        # Execute pipeline
        result = await self.pipeline.execute(initial_data, preference, center, context or {})
        
        # Convert plugin result to agent result format
        return self._convert_plugin_result(result)
    
    def _extract_initial_data(self, preference: ParentPreference) -> Dict[str, Any]:
        """Extract initial data from preference for pipeline processing."""
        return preference.value_data or {}
    
    def _convert_plugin_result(self, result: PluginProcessingResult) -> Dict[str, Any]:
        """Convert plugin processing result to agent result format."""
        
        base_result = {
            "score": 0.0,
            "explanation": "Processing incomplete",
            "metadata": result.metadata or {},
            "semantic_enhancements": []
        }
        
        if result.status == PluginResult.SUCCESS:
            data = result.data or {}
            base_result.update({
                "score": data.get('score', 0.0),
                "explanation": data.get('explanation', 'Processed successfully'),
                "metadata": {**(result.metadata or {}), **data.get('metadata', {})},
            })
        
        elif result.status == PluginResult.USER_INTERACTION_REQUIRED:
            base_result.update({
                "user_interaction": {
                    "required": True,
                    "type": result.interaction.interaction_type,
                    "question": result.interaction.question,
                    "field_name": result.interaction.field_name,
                    "options": result.interaction.options,
                    "context": result.interaction.context
                }
            })
        
        elif result.status == PluginResult.ERROR:
            base_result.update({
                "explanation": result.error_message or "Processing failed",
                "metadata": {**(result.metadata or {}), "error": True}
            })
        
        return base_result
    
    @abstractmethod
    def get_processing_config(self) -> Dict[str, Any]:
        """Return agent configuration for database storage."""
        pass


class LocationDistanceAgent(ComplexTypeAgent):
    """Agent for processing location distance constraints."""
    
    def __init__(self, pipeline_config: List[Dict[str, Any]] = None):
        """Initialize with default or custom pipeline configuration."""
        if pipeline_config is None:
            # Default pipeline for location distance processing
            pipeline_config = [
                {"name": "location_resolver", "config": {}},
                {"name": "distance_calculator", "config": {}},
                {"name": "location_scorer", "config": {}}
            ]
        super().__init__(pipeline_config)
    
    def get_type_name(self) -> str:
        return "location_distance"
    
    
    def _extract_initial_data(self, preference: ParentPreference) -> Dict[str, Any]:
        """Extract location data and enhance with constraint information."""
        data = super()._extract_initial_data(preference)
        
        # Enhance locations with constraint type information
        locations = data.get('locations', [])
        for location in locations:
            # Detect constraint type from the data structure
            if 'max_travel_minutes' in location or 'travel_mode' in location:
                location['constraint_type'] = 'travel_time'
                location['transport_mode'] = location.get('transport_mode', 'walking')
            else:
                location['constraint_type'] = 'distance'
        
        return data
    
    def get_processing_config(self) -> Dict[str, Any]:
        """Configuration for database storage."""
        return {
            "requires_geolocation": True,
            "requires_semantic_service": False,  # Optional
            "cache_duration_minutes": 60,
            "priority": 100  # High priority for location constraints
        }
    
    async def _calculate_distance(self, location: Dict, center_location) -> float:
        """Calculate distance between location and center."""
        # Implementation would use actual geolocation service
        # For now, return mock distance
        return 2.5
    
    async def _find_location_semantic_matches(self, location: Dict, semantic_service) -> List[Dict]:
        """Find semantic enhancements for location context."""
        context = location.get('context', '')
        if context:
            # Use semantic service to find related features
            matches = await semantic_service.find_semantic_matches(
                query_text=context,
                similarity_threshold=0.4,
                limit=5
            )
            return [{"source": "location_context", "matches": matches}]
        return []


class ScheduleRangeAgent(ComplexTypeAgent):
    """Agent for processing schedule/time range constraints."""
    
    def __init__(self, pipeline_config: List[Dict[str, Any]] = None):
        """Initialize with default or custom pipeline configuration."""
        if pipeline_config is None:
            # Default pipeline for schedule processing
            pipeline_config = [
                {"name": "schedule_matcher", "config": {}},
                {"name": "schedule_scorer", "config": {}}
            ]
        super().__init__(pipeline_config)
    
    def get_type_name(self) -> str:
        return "schedule_range"
    
    def get_processing_config(self) -> Dict[str, Any]:
        return {
            "requires_geolocation": False,
            "requires_semantic_service": False,
            "cache_duration_minutes": 1440,  # 24 hours
            "priority": 80
        }
    
    async def _match_schedule(self, start, end, flexibility, days, center) -> float:
        """Match schedule against center's hours."""
        # Implementation would check center.opening_hours
        return 0.85  # Mock score


class SemanticEnhancedAgent(ComplexTypeAgent):
    """Agent that uses semantic matching for complex types."""
    
    def __init__(self, pipeline_config: List[Dict[str, Any]] = None):
        """Initialize with default or custom pipeline configuration."""
        if pipeline_config is None:
            # Default pipeline for semantic matching
            pipeline_config = [
                {"name": "semantic_matcher", "config": {"threshold": 0.3}},
                {"name": "semantic_scorer", "config": {}}
            ]
        super().__init__(pipeline_config)
    
    def get_type_name(self) -> str:
        return "educational_approach"
    
    def get_processing_config(self) -> Dict[str, Any]:
        return {
            "requires_geolocation": False,
            "requires_semantic_service": True,  # Required for this agent
            "cache_duration_minutes": 120,
            "priority": 70
        }
    
    def _center_has_feature(self, center: Center, feature_key: str) -> bool:
        """Check if center has a specific feature."""
        # Implementation would check center.properties
        return True  # Mock


class ComplexTypeAgentRegistry:
    """Registry for dynamic complex type agent management."""
    
    def __init__(self, db_repository: ComplexPreferenceRepository):
        self.db_repository = db_repository
        self.agents: Dict[str, ComplexTypeAgent] = {}
        self._agent_classes = {
            'LocationDistanceAgent': LocationDistanceAgent,
            'ScheduleRangeAgent': ScheduleRangeAgent, 
            'SemanticEnhancedAgent': SemanticEnhancedAgent
        }
        
    async def initialize(self):
        """Initialize registry by loading configurations from database."""
        await self._load_agents_from_database()
        await self._register_fallback_agents()
    
    async def _load_agents_from_database(self):
        """Load agent configurations from database."""
        complex_types = await self.db_repository.get_complex_value_types()
        
        for complex_type in complex_types:
            agent_class_name = complex_type.agent_class
            pipeline_config = complex_type.plugin_pipeline
            
            if agent_class_name and agent_class_name in self._agent_classes:
                # Create agent with database configuration
                agent_class = self._agent_classes[agent_class_name]
                
                # Parse pipeline configuration from database
                if pipeline_config and isinstance(pipeline_config, list):
                    agent = agent_class(pipeline_config)
                else:
                    agent = agent_class()  # Use default configuration
                
                self.register_agent(agent)
                logger.info(f"Loaded agent {agent_class_name} for {complex_type.type_name} with pipeline: {pipeline_config}")
            else:
                logger.warning(f"Unknown agent class: {agent_class_name} for type {complex_type.type_name}")
    
    async def _register_fallback_agents(self):
        """Register built-in agents for types not configured in database."""
        fallback_agents = [
            LocationDistanceAgent(),
            ScheduleRangeAgent(),
            SemanticEnhancedAgent()
        ]
        
        for agent in fallback_agents:
            type_name = agent.get_type_name()
            if type_name not in self.agents:
                self.register_agent(agent)
                logger.info(f"Registered fallback agent: {type_name}")
    
    def register_agent(self, agent: ComplexTypeAgent):
        """Register a new agent."""
        type_name = agent.get_type_name()
        self.agents[type_name] = agent
        logger.debug(f"Registered complex type agent: {type_name}")
    
    def register_agent_class(self, class_name: str, agent_class: type):
        """Register a new agent class for database instantiation."""
        self._agent_classes[class_name] = agent_class
        logger.info(f"Registered agent class: {class_name}")
    
    async def process_complex_preference(
        self, 
        preference: ParentPreference, 
        center: Center,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Process a complex preference using the appropriate agent."""
        
        if not preference.complex_value_type_id:
            return {"score": 0.0, "explanation": "No complex type specified"}
        
        # Get the complex type configuration from database
        complex_type = await self.db_repository.get_complex_value_type_by_id(
            preference.complex_value_type_id
        )
        
        if not complex_type:
            return {"score": 0.0, "explanation": "Unknown complex type"}
        
        # Get the appropriate agent
        agent = self.agents.get(complex_type.type_name)
        if not agent:
            logger.warning(f"No agent found for complex type: {complex_type.type_name}")
            return {"score": 0.0, "explanation": f"No processor for {complex_type.type_name}"}
        
        # Process with the agent
        result = await agent.process_preference(preference, center, context)
        
        # Add metadata about the processing
        result["processing_metadata"] = {
            "agent_type": complex_type.type_name,
            "agent_config": agent.get_processing_config(),
            "complex_type_version": complex_type.version,
            "supports_user_interaction": complex_type.supports_user_interaction,
            "pipeline_used": hasattr(agent, 'pipeline') and agent.pipeline is not None
        }
        
        return result
    
    async def get_available_agents(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all available agents."""
        agent_info = {}
        for type_name, agent in self.agents.items():
            agent_info[type_name] = {
                "type_name": type_name,
                "config": agent.get_processing_config(),
                "class_name": agent.__class__.__name__,
                "has_pipeline": hasattr(agent, 'pipeline') and agent.pipeline is not None,
                "pipeline_name": agent.pipeline.name if hasattr(agent, 'pipeline') and agent.pipeline else None
            }
        return agent_info