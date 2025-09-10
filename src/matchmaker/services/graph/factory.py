"""Graph client factory for creating database-specific clients."""

import logging
import os
from enum import Enum

from .base import GraphClient
from .neo4j_client import Neo4jClient
from .tigergraph_client import TigerGraphClient

logger = logging.getLogger(__name__)


class GraphDBType(str, Enum):
    """Supported graph database types."""
    TIGERGRAPH = "tigergraph"
    NEO4J = "neo4j"


class GraphClientFactory:
    """Factory for creating graph database clients."""

    @staticmethod
    def create_client(graph_type: str | None = None) -> GraphClient:
        """
        Create a graph client based on configuration.
        
        Args:
            graph_type: Override graph type (otherwise uses environment)
            
        Returns:
            Configured graph client
            
        Raises:
            ValueError: If graph type is unsupported
            RuntimeError: If required environment variables are missing
        """
        # Determine graph type
        db_type = graph_type or os.getenv('GRAPH_DB_TYPE', 'tigergraph')
        db_type = db_type.lower().strip()

        logger.info(f"Creating graph client for: {db_type}")

        if db_type == GraphDBType.TIGERGRAPH:
            return GraphClientFactory._create_tigergraph_client()
        elif db_type == GraphDBType.NEO4J:
            return GraphClientFactory._create_neo4j_client()
        else:
            raise ValueError(f"Unsupported graph database type: {db_type}. "
                           f"Supported types: {[t.value for t in GraphDBType]}")

    @staticmethod
    def _create_tigergraph_client() -> TigerGraphClient:
        """Create TigerGraph client from environment variables."""
        required_vars = [
            'TIGERGRAPH_HOST',
            'TIGERGRAPH_USERNAME',
            'TIGERGRAPH_PASSWORD'
        ]

        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise RuntimeError(f"Missing required environment variables for TigerGraph: {missing_vars}")

        config = {
            'host': os.getenv('TIGERGRAPH_HOST'),
            'username': os.getenv('TIGERGRAPH_USERNAME'),
            'password': os.getenv('TIGERGRAPH_PASSWORD'),
            'graph_name': os.getenv('TIGERGRAPH_GRAPH_NAME', 'childcare'),
            'version': os.getenv('TIGERGRAPH_VERSION', '3.9.0')
        }

        logger.info(f"Creating TigerGraph client: {config['host']}, graph: {config['graph_name']}")
        return TigerGraphClient(**config)

    @staticmethod
    def _create_neo4j_client() -> Neo4jClient:
        """Create Neo4j client from environment variables."""
        required_vars = [
            'NEO4J_URI',
            'NEO4J_USER',
            'NEO4J_PASSWORD'
        ]

        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise RuntimeError(f"Missing required environment variables for Neo4j: {missing_vars}")

        config = {
            'uri': os.getenv('NEO4J_URI'),
            'user': os.getenv('NEO4J_USER'),
            'password': os.getenv('NEO4J_PASSWORD'),
            'database': os.getenv('NEO4J_DATABASE', 'neo4j'),
            'max_connection_lifetime': int(os.getenv('NEO4J_MAX_CONNECTION_LIFETIME', '3600')),
            'max_connection_pool_size': int(os.getenv('NEO4J_MAX_CONNECTION_POOL_SIZE', '50'))
        }

        logger.info(f"Creating Neo4j client: {config['uri']}, database: {config['database']}")
        return Neo4jClient(**config)

    @staticmethod
    def get_supported_types() -> list[str]:
        """Get list of supported graph database types."""
        return [t.value for t in GraphDBType]

    @staticmethod
    def validate_environment(graph_type: str | None = None) -> tuple[bool, list[str]]:
        """
        Validate that required environment variables are set.
        
        Args:
            graph_type: Graph type to validate (otherwise uses GRAPH_DB_TYPE)
            
        Returns:
            Tuple of (is_valid, missing_variables)
        """
        db_type = graph_type or os.getenv('GRAPH_DB_TYPE', 'tigergraph')
        db_type = db_type.lower().strip()

        if db_type == GraphDBType.TIGERGRAPH:
            required_vars = [
                'TIGERGRAPH_HOST',
                'TIGERGRAPH_USERNAME',
                'TIGERGRAPH_PASSWORD'
            ]
        elif db_type == GraphDBType.NEO4J:
            required_vars = [
                'NEO4J_URI',
                'NEO4J_USER',
                'NEO4J_PASSWORD'
            ]
        else:
            return False, [f"Unsupported graph type: {db_type}"]

        missing_vars = [var for var in required_vars if not os.getenv(var)]
        return len(missing_vars) == 0, missing_vars


# Convenience function for creating a client
def create_graph_client(graph_type: str | None = None) -> GraphClient:
    """
    Convenience function to create a graph client.
    
    Args:
        graph_type: Override graph type
        
    Returns:
        Configured graph client
    """
    return GraphClientFactory.create_client(graph_type)


# Global client instance (lazy initialization)
_global_client: GraphClient | None = None


async def get_graph_client() -> GraphClient:
    """
    Get global graph client instance (singleton pattern).
    
    Returns:
        Connected graph client
    """
    global _global_client

    if _global_client is None:
        _global_client = create_graph_client()

        # Connect if not already connected
        if not _global_client.connected:
            success = await _global_client.connect()
            if not success:
                raise RuntimeError("Failed to connect to graph database")

    return _global_client


async def close_global_client():
    """Close the global graph client connection."""
    global _global_client

    if _global_client and _global_client.connected:
        await _global_client.disconnect()
        _global_client = None
