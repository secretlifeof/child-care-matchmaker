"""Graph database services for the matchmaker."""

from .base import (
    FeatureMatchResult,
    GraphClient,
    GraphEdge,
    GraphNode,
    GraphQueryResult,
)
from .factory import (
    GraphClientFactory,
    GraphDBType,
    close_global_client,
    create_graph_client,
    get_graph_client,
)
from .neo4j_client import Neo4jClient
from .tigergraph_client import TigerGraphClient

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
