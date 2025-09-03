"""Neo4j client implementation."""

import logging
from typing import Dict, List, Optional, Any
import asyncio

try:
    from neo4j import GraphDatabase, AsyncGraphDatabase
    from neo4j.exceptions import ServiceUnavailable, ClientError
except ImportError:
    GraphDatabase = None
    AsyncGraphDatabase = None

from .base import GraphClient, FeatureMatchResult, GraphNode, GraphEdge

logger = logging.getLogger(__name__)


class Neo4jClient(GraphClient):
    """Neo4j implementation of GraphClient interface."""
    
    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str = "neo4j",
        max_connection_lifetime: int = 3600,
        max_connection_pool_size: int = 50
    ):
        super().__init__()
        
        if GraphDatabase is None:
            raise ImportError("neo4j is required for Neo4j client. Install with: pip install neo4j")
        
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        
        self.driver = None
        self.async_driver = None
        
        # Connection settings
        self.max_connection_lifetime = max_connection_lifetime
        self.max_connection_pool_size = max_connection_pool_size
    
    async def connect(self) -> bool:
        """Establish connection to Neo4j."""
        try:
            # Create both sync and async drivers
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
                max_connection_lifetime=self.max_connection_lifetime,
                max_connection_pool_size=self.max_connection_pool_size
            )
            
            # Test connection
            with self.driver.session(database=self.database) as session:
                session.run("RETURN 1")
            
            self.connected = True
            logger.info(f"Connected to Neo4j at {self.uri}")
            
            # Create indexes and constraints
            await self._create_schema()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            self.connected = False
            return False
    
    async def disconnect(self):
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()
            self.driver = None
        
        if self.async_driver:
            await self.async_driver.close()
            self.async_driver = None
        
        self.connected = False
        logger.info("Disconnected from Neo4j")
    
    async def health_check(self) -> bool:
        """Check Neo4j health."""
        if not self.driver:
            return False
        
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN 1 as health")
                return result.single()["health"] == 1
        except:
            return False
    
    async def _create_schema(self):
        """Create necessary indexes and constraints."""
        schema_queries = [
            # Constraints (unique IDs)
            "CREATE CONSTRAINT center_id IF NOT EXISTS FOR (c:Center) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT parent_id IF NOT EXISTS FOR (p:Parent) REQUIRE p.id IS UNIQUE", 
            "CREATE CONSTRAINT feature_key IF NOT EXISTS FOR (f:Feature) REQUIRE f.key IS UNIQUE",
            
            # Indexes for performance
            "CREATE INDEX center_location IF NOT EXISTS FOR (c:Center) ON (c.lat, c.lon)",
            "CREATE INDEX feature_category IF NOT EXISTS FOR (f:Feature) ON f.category",
            "CREATE INDEX has_feature_confidence IF NOT EXISTS FOR ()-[r:HAS_FEATURE]-() ON r.confidence",
            "CREATE INDEX wants_preference IF NOT EXISTS FOR ()-[r:WANTS]-() ON r.preference"
        ]
        
        if not self.driver:
            return
        
        try:
            with self.driver.session(database=self.database) as session:
                for query in schema_queries:
                    try:
                        session.run(query)
                    except ClientError as e:
                        # Ignore if constraint/index already exists
                        if "already exists" not in str(e):
                            logger.warning(f"Schema query failed: {query}, Error: {e}")
            
            logger.info("Neo4j schema setup completed")
            
        except Exception as e:
            logger.error(f"Error creating Neo4j schema: {e}")
    
    async def create_center_node(self, center_id: str, **properties) -> bool:
        """Create center node in Neo4j."""
        if not self.driver:
            logger.error("Not connected to Neo4j")
            return False
        
        try:
            query = """
            MERGE (c:Center {id: $center_id})
            SET c.name = $name,
                c.lat = $lat,
                c.lon = $lon,
                c.h3_9 = $h3_9,
                c.updated_at = datetime()
            """
            
            with self.driver.session(database=self.database) as session:
                session.run(query, {
                    "center_id": center_id,
                    "name": properties.get("name", ""),
                    "lat": properties.get("lat"),
                    "lon": properties.get("lon"),
                    "h3_9": properties.get("h3_9")
                })
            
            logger.debug(f"Created center node: {center_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating center node {center_id}: {e}")
            return False
    
    async def create_parent_node(self, parent_id: str, **properties) -> bool:
        """Create parent node in Neo4j."""
        if not self.driver:
            logger.error("Not connected to Neo4j")
            return False
        
        try:
            query = """
            MERGE (p:Parent {id: $parent_id})
            SET p.home_lat = $home_lat,
                p.home_lon = $home_lon,
                p.updated_at = datetime()
            """
            
            with self.driver.session(database=self.database) as session:
                session.run(query, {
                    "parent_id": parent_id,
                    "home_lat": properties.get("home_lat"),
                    "home_lon": properties.get("home_lon")
                })
            
            logger.debug(f"Created parent node: {parent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating parent node {parent_id}: {e}")
            return False
    
    async def create_feature_node(self, feature_key: str, category: str, **properties) -> bool:
        """Create feature node in Neo4j."""
        if not self.driver:
            logger.error("Not connected to Neo4j")
            return False
        
        try:
            query = """
            MERGE (f:Feature {key: $feature_key})
            SET f.category = $category,
                f.updated_at = datetime()
            """
            
            # Add any additional properties
            for key, value in properties.items():
                query += f", f.{key} = ${key}"
            
            params = {
                "feature_key": feature_key,
                "category": category,
                **properties
            }
            
            with self.driver.session(database=self.database) as session:
                session.run(query, params)
            
            logger.debug(f"Created feature node: {feature_key}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating feature node {feature_key}: {e}")
            return False
    
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
        """Create HAS_FEATURE edge in Neo4j."""
        if not self.driver:
            logger.error("Not connected to Neo4j")
            return False
        
        try:
            query = """
            MATCH (c:Center {id: $center_id})
            MATCH (f:Feature {key: $feature_key})
            MERGE (c)-[r:HAS_FEATURE]->(f)
            SET r.confidence = $confidence,
                r.source = $source,
                r.raw_phrase = $raw_phrase,
                r.value_bool = $value_bool,
                r.value_num = $value_num,
                r.unit = $unit,
                r.updated_at = datetime()
            """
            
            # Add additional properties
            params = {
                "center_id": center_id,
                "feature_key": feature_key,
                "confidence": confidence,
                "source": source,
                "raw_phrase": raw_phrase,
                "value_bool": value_bool,
                "value_num": value_num,
                "unit": unit,
                **properties
            }
            
            with self.driver.session(database=self.database) as session:
                session.run(query, params)
            
            logger.debug(f"Created HAS_FEATURE edge: {center_id} -> {feature_key}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating HAS_FEATURE edge {center_id}->{feature_key}: {e}")
            return False
    
    async def create_wants_feature_edge(
        self,
        parent_id: str,
        feature_key: str,
        preference: str,
        confidence: float,
        value_bool: Optional[bool] = None,
        value_num: Optional[float] = None,
        value_text: Optional[str] = None,
        unit: Optional[str] = None,
        **properties
    ) -> bool:
        """Create WANTS edge in Neo4j."""
        if not self.driver:
            logger.error("Not connected to Neo4j")
            return False
        
        try:
            query = """
            MATCH (p:Parent {id: $parent_id})
            MATCH (f:Feature {key: $feature_key})
            MERGE (p)-[r:WANTS]->(f)
            SET r.preference = $preference,
                r.confidence = $confidence,
                r.value_bool = $value_bool,
                r.value_num = $value_num,
                r.value_text = $value_text,
                r.unit = $unit,
                r.updated_at = datetime()
            """
            
            params = {
                "parent_id": parent_id,
                "feature_key": feature_key,
                "preference": preference,
                "confidence": confidence,
                "value_bool": value_bool,
                "value_num": value_num,
                "value_text": value_text,
                "unit": unit,
                **properties
            }
            
            with self.driver.session(database=self.database) as session:
                session.run(query, params)
            
            logger.debug(f"Created WANTS edge: {parent_id} -> {feature_key}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating WANTS edge {parent_id}->{feature_key}: {e}")
            return False
    
    async def query_feature_matches(
        self,
        parent_id: str,
        candidate_center_ids: List[str]
    ) -> List[FeatureMatchResult]:
        """Query feature matches using Neo4j."""
        if not self.driver:
            logger.error("Not connected to Neo4j")
            return []
        
        try:
            query = """
            MATCH (p:Parent {id: $parent_id})-[wants:WANTS]->(f:Feature)<-[has:HAS_FEATURE]-(c:Center)
            WHERE c.id IN $center_ids
            RETURN c.id as center_id,
                   f.key as feature_key,
                   wants.preference as preference,
                   has.confidence as center_confidence,
                   wants.confidence as parent_confidence,
                   CASE 
                     WHEN wants.value_bool IS NOT NULL AND has.value_bool IS NOT NULL 
                     THEN wants.value_bool = has.value_bool
                     WHEN wants.value_num IS NOT NULL AND has.value_num IS NOT NULL
                     THEN abs(wants.value_num - has.value_num) <= 0.1
                     ELSE true
                   END as feature_satisfied,
                   has.value_bool as center_value_bool,
                   has.value_num as center_value_num,
                   wants.value_bool as parent_value_bool,
                   wants.value_num as parent_value_num,
                   1.0 as similarity_score
            """
            
            with self.driver.session(database=self.database) as session:
                result = session.run(query, {
                    "parent_id": parent_id,
                    "center_ids": candidate_center_ids
                })
                
                matches = []
                for record in result:
                    # Determine center and parent values
                    center_value = record["center_value_bool"]
                    if center_value is None:
                        center_value = record["center_value_num"]
                    
                    parent_value = record["parent_value_bool"] 
                    if parent_value is None:
                        parent_value = record["parent_value_num"]
                    
                    matches.append(FeatureMatchResult(
                        center_id=record["center_id"],
                        feature_key=record["feature_key"],
                        preference=record["preference"],
                        center_confidence=record["center_confidence"],
                        parent_confidence=record["parent_confidence"],
                        feature_satisfied=record["feature_satisfied"],
                        center_value=center_value,
                        parent_value=parent_value,
                        similarity_score=record["similarity_score"]
                    ))
                
                logger.debug(f"Found {len(matches)} feature matches for parent {parent_id}")
                return matches
            
        except Exception as e:
            logger.error(f"Error querying feature matches for parent {parent_id}: {e}")
            return []
    
    async def query_centers_with_feature(
        self,
        feature_key: str,
        value_filter: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Find centers with specific feature."""
        if not self.driver:
            logger.error("Not connected to Neo4j")
            return []
        
        try:
            base_query = """
            MATCH (c:Center)-[r:HAS_FEATURE]->(f:Feature {key: $feature_key})
            """
            
            # Add value filters if provided
            where_conditions = []
            params = {"feature_key": feature_key}
            
            if value_filter:
                for key, value in value_filter.items():
                    if key == "value_bool":
                        where_conditions.append("r.value_bool = $value_bool")
                        params["value_bool"] = value
                    elif key == "value_num_min":
                        where_conditions.append("r.value_num >= $value_num_min")
                        params["value_num_min"] = value
                    elif key == "value_num_max":
                        where_conditions.append("r.value_num <= $value_num_max")
                        params["value_num_max"] = value
            
            if where_conditions:
                base_query += " WHERE " + " AND ".join(where_conditions)
            
            base_query += " RETURN DISTINCT c.id as center_id"
            
            with self.driver.session(database=self.database) as session:
                result = session.run(base_query, params)
                center_ids = [record["center_id"] for record in result]
                
                logger.debug(f"Found {len(center_ids)} centers with feature {feature_key}")
                return center_ids
                
        except Exception as e:
            logger.error(f"Error querying centers with feature {feature_key}: {e}")
            return []
    
    async def get_center_features(self, center_id: str) -> Dict[str, Any]:
        """Get all features for a center."""
        if not self.driver:
            logger.error("Not connected to Neo4j")
            return {}
        
        try:
            query = """
            MATCH (c:Center {id: $center_id})-[r:HAS_FEATURE]->(f:Feature)
            RETURN f.key as feature_key,
                   r.confidence as confidence,
                   r.source as source,
                   r.value_bool as value_bool,
                   r.value_num as value_num,
                   r.unit as unit,
                   r.raw_phrase as raw_phrase
            """
            
            with self.driver.session(database=self.database) as session:
                result = session.run(query, {"center_id": center_id})
                
                features = {}
                for record in result:
                    feature_key = record["feature_key"]
                    features[feature_key] = {
                        "confidence": record["confidence"],
                        "source": record["source"],
                        "value_bool": record["value_bool"],
                        "value_num": record["value_num"],
                        "unit": record["unit"],
                        "raw_phrase": record["raw_phrase"]
                    }
                
                logger.debug(f"Retrieved {len(features)} features for center {center_id}")
                return features
                
        except Exception as e:
            logger.error(f"Error getting features for center {center_id}: {e}")
            return {}
    
    async def get_parent_preferences(self, parent_id: str) -> Dict[str, Any]:
        """Get all preferences for a parent."""
        if not self.driver:
            logger.error("Not connected to Neo4j")
            return {}
        
        try:
            query = """
            MATCH (p:Parent {id: $parent_id})-[r:WANTS]->(f:Feature)
            RETURN f.key as feature_key,
                   r.preference as preference,
                   r.confidence as confidence,
                   r.value_bool as value_bool,
                   r.value_num as value_num,
                   r.value_text as value_text,
                   r.unit as unit
            """
            
            with self.driver.session(database=self.database) as session:
                result = session.run(query, {"parent_id": parent_id})
                
                preferences = {}
                for record in result:
                    feature_key = record["feature_key"]
                    preferences[feature_key] = {
                        "preference": record["preference"],
                        "confidence": record["confidence"],
                        "value_bool": record["value_bool"],
                        "value_num": record["value_num"],
                        "value_text": record["value_text"],
                        "unit": record["unit"]
                    }
                
                logger.debug(f"Retrieved {len(preferences)} preferences for parent {parent_id}")
                return preferences
                
        except Exception as e:
            logger.error(f"Error getting preferences for parent {parent_id}: {e}")
            return {}
    
    async def delete_center_features(self, center_id: str) -> bool:
        """Delete all features for a center."""
        if not self.driver:
            logger.error("Not connected to Neo4j")
            return False
        
        try:
            query = """
            MATCH (c:Center {id: $center_id})-[r:HAS_FEATURE]->()
            DELETE r
            """
            
            with self.driver.session(database=self.database) as session:
                session.run(query, {"center_id": center_id})
                
            logger.debug(f"Deleted features for center {center_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting features for center {center_id}: {e}")
            return False
    
    async def delete_parent_preferences(self, parent_id: str) -> bool:
        """Delete all preferences for a parent."""
        if not self.driver:
            logger.error("Not connected to Neo4j")
            return False
        
        try:
            query = """
            MATCH (p:Parent {id: $parent_id})-[r:WANTS]->()
            DELETE r
            """
            
            with self.driver.session(database=self.database) as session:
                session.run(query, {"parent_id": parent_id})
                
            logger.debug(f"Deleted preferences for parent {parent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting preferences for parent {parent_id}: {e}")
            return False
    
    async def execute_raw_query(self, query: str, parameters: Dict[str, Any] = None) -> Any:
        """Execute raw Cypher query."""
        if not self.driver:
            logger.error("Not connected to Neo4j")
            return None
        
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
                
        except Exception as e:
            logger.error(f"Error executing raw query: {e}")
            return None