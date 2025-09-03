"""Base graph database client interface."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)


class GraphNode(BaseModel):
    """Generic graph node representation."""
    id: str
    labels: List[str]
    properties: Dict[str, Any] = {}


class GraphEdge(BaseModel):
    """Generic graph edge representation."""
    from_id: str
    to_id: str
    relationship_type: str
    properties: Dict[str, Any] = {}


class FeatureMatchResult(BaseModel):
    """Result of feature matching for a center."""
    center_id: str
    feature_key: str
    preference: str  # "must", "nice_to_have", "exclude"
    center_confidence: float
    parent_confidence: float
    feature_satisfied: bool
    center_value: Optional[Any] = None
    parent_value: Optional[Any] = None
    similarity_score: float = 1.0  # For semantic/embedding matches


class GraphQueryResult(BaseModel):
    """Generic query result container."""
    nodes: List[GraphNode] = []
    edges: List[GraphEdge] = []
    feature_matches: List[FeatureMatchResult] = []
    metadata: Dict[str, Any] = {}


class GraphClient(ABC):
    """Abstract base class for graph database clients."""
    
    def __init__(self):
        self.connected = False
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to graph database."""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Close connection to graph database.""" 
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if graph database is healthy."""
        pass
    
    # Node operations
    @abstractmethod
    async def create_center_node(self, center_id: str, **properties) -> bool:
        """
        Create or update a center node.
        
        Args:
            center_id: Unique center identifier
            **properties: Additional node properties (name, lat, lon, etc.)
            
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    async def create_parent_node(self, parent_id: str, **properties) -> bool:
        """
        Create or update a parent node.
        
        Args:
            parent_id: Unique parent identifier  
            **properties: Additional node properties (home_lat, home_lon, etc.)
            
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    async def create_feature_node(self, feature_key: str, category: str, **properties) -> bool:
        """
        Create or update a feature node.
        
        Args:
            feature_key: Unique feature identifier
            category: Feature category
            **properties: Additional properties
            
        Returns:
            True if successful
        """
        pass
    
    # Edge operations  
    @abstractmethod
    async def create_has_feature_edge(
        self, 
        center_id: str, 
        feature_key: str,
        confidence: float = 1.0,
        source: str = "extraction",
        raw_phrase: Optional[str] = None,
        value_bool: Optional[bool] = None,
        value_num: Optional[float] = None,
        unit: Optional[str] = None,
        **properties
    ) -> bool:
        """
        Create HAS_FEATURE edge from center to feature.
        
        Args:
            center_id: Center node ID
            feature_key: Feature node ID
            confidence: Extraction confidence (0-1)
            source: Source of information ("extraction", "manual", "import")
            raw_phrase: Original text that led to this feature
            value_bool: Boolean value if applicable
            value_num: Numeric value if applicable  
            unit: Unit for numeric values
            **properties: Additional edge properties
            
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    async def create_wants_feature_edge(
        self,
        parent_id: str,
        feature_key: str, 
        preference: str,  # "must", "nice_to_have", "exclude"
        confidence: float,
        value_bool: Optional[bool] = None,
        value_num: Optional[float] = None,
        value_text: Optional[str] = None,
        unit: Optional[str] = None,
        **properties
    ) -> bool:
        """
        Create WANTS edge from parent to feature.
        
        Args:
            parent_id: Parent node ID
            feature_key: Feature node ID
            preference: Preference level
            confidence: Confidence in preference extraction
            value_bool: Boolean value if applicable
            value_num: Numeric value if applicable
            value_text: Text value if applicable
            unit: Unit for numeric values
            **properties: Additional edge properties
            
        Returns:
            True if successful
        """
        pass
    
    # Query operations
    @abstractmethod
    async def query_feature_matches(
        self, 
        parent_id: str,
        candidate_center_ids: List[str]
    ) -> List[FeatureMatchResult]:
        """
        Query feature matches between parent and candidate centers.
        
        Args:
            parent_id: Parent node ID
            candidate_center_ids: List of center IDs to check
            
        Returns:
            List of feature match results
        """
        pass
    
    @abstractmethod
    async def query_centers_with_feature(
        self, 
        feature_key: str, 
        value_filter: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Find centers that have a specific feature.
        
        Args:
            feature_key: Feature to search for
            value_filter: Optional filter on feature values
            
        Returns:
            List of center IDs
        """
        pass
    
    @abstractmethod
    async def get_center_features(self, center_id: str) -> Dict[str, Any]:
        """
        Get all features for a center.
        
        Args:
            center_id: Center node ID
            
        Returns:
            Dictionary mapping feature_key to feature data
        """
        pass
    
    @abstractmethod
    async def get_parent_preferences(self, parent_id: str) -> Dict[str, Any]:
        """
        Get all preferences for a parent.
        
        Args:
            parent_id: Parent node ID
            
        Returns:
            Dictionary mapping feature_key to preference data
        """
        pass
    
    # Bulk operations
    async def bulk_create_center_features(
        self, 
        center_features: List[Dict[str, Any]]
    ) -> int:
        """
        Bulk create center features.
        
        Args:
            center_features: List of center feature dictionaries
            
        Returns:
            Number of features created
        """
        created_count = 0
        for feature in center_features:
            success = await self.create_has_feature_edge(**feature)
            if success:
                created_count += 1
        return created_count
    
    async def bulk_create_parent_preferences(
        self,
        parent_preferences: List[Dict[str, Any]]
    ) -> int:
        """
        Bulk create parent preferences.
        
        Args:
            parent_preferences: List of parent preference dictionaries
            
        Returns:
            Number of preferences created
        """
        created_count = 0
        for preference in parent_preferences:
            success = await self.create_wants_feature_edge(**preference)
            if success:
                created_count += 1
        return created_count
    
    # Utility methods
    @abstractmethod
    async def delete_center_features(self, center_id: str) -> bool:
        """Delete all features for a center."""
        pass
    
    @abstractmethod
    async def delete_parent_preferences(self, parent_id: str) -> bool:
        """Delete all preferences for a parent."""
        pass
    
    @abstractmethod
    async def execute_raw_query(self, query: str, parameters: Dict[str, Any] = None) -> Any:
        """Execute raw database-specific query."""
        pass
    
    def __str__(self):
        return f"{self.__class__.__name__}(connected={self.connected})"