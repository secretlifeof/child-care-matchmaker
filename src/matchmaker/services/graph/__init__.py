"""Graph database services for the matchmaker."""

from .base import GraphClient, FeatureMatchResult, GraphNode, GraphEdge, GraphQueryResult
from .tigergraph_client import TigerGraphClient
from .neo4j_client import Neo4jClient
from .factory import GraphClientFactory, GraphDBType, create_graph_client, get_graph_client, close_global_client

__all__ = [
    # Base classes
    "GraphClient",
    "FeatureMatchResult", 
    "GraphNode",
    "GraphEdge",
    "GraphQueryResult",
    
    # Client implementations
    "TigerGraphClient",
    "Neo4jClient",
    
    # Factory
    "GraphClientFactory",
    "GraphDBType",
    "create_graph_client",
    "get_graph_client",
    "close_global_client"
]