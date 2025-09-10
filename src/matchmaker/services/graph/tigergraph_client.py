"""TigerGraph client implementation."""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

try:
    import pyTigerGraph as tg
except ImportError:
    tg = None

from .base import FeatureMatchResult, GraphClient

logger = logging.getLogger(__name__)


class TigerGraphClient(GraphClient):
    """TigerGraph implementation of GraphClient interface."""

    def __init__(
        self,
        host: str,
        username: str,
        password: str,
        graph_name: str = "childcare",
        version: str = "3.9.0"
    ):
        super().__init__()

        if tg is None:
            raise ImportError("pyTigerGraph is required for TigerGraph client. Install with: pip install pyTigerGraph")

        self.host = host
        self.username = username
        self.password = password
        self.graph_name = graph_name
        self.version = version

        self.conn = None
        self._token = None
        self._executor = ThreadPoolExecutor(max_workers=4)

    async def connect(self) -> bool:
        """Establish connection to TigerGraph."""
        try:
            loop = asyncio.get_event_loop()
            self.conn = await loop.run_in_executor(
                self._executor,
                self._create_connection
            )

            # Get authentication token
            self._token = await loop.run_in_executor(
                self._executor,
                self.conn.getToken
            )

            self.connected = True
            logger.info(f"Connected to TigerGraph at {self.host}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to TigerGraph: {e}")
            self.connected = False
            return False

    def _create_connection(self):
        """Create TigerGraph connection (blocking operation)."""
        return tg.TigerGraphConnection(
            host=self.host,
            username=self.username,
            password=self.password,
            graphname=self.graph_name,
            version=self.version
        )

    async def disconnect(self):
        """Close TigerGraph connection."""
        if self._executor:
            self._executor.shutdown(wait=True)
        self.connected = False
        logger.info("Disconnected from TigerGraph")

    async def health_check(self) -> bool:
        """Check TigerGraph health."""
        if not self.conn:
            return False

        try:
            loop = asyncio.get_event_loop()
            # Simple query to test connection
            result = await loop.run_in_executor(
                self._executor,
                lambda: self.conn.runInstalledQuery("health_check", {})
            )
            return True
        except:
            return False

    async def create_center_node(self, center_id: str, **properties) -> bool:
        """Create center node in TigerGraph."""
        if not self.conn:
            logger.error("Not connected to TigerGraph")
            return False

        try:
            vertex_data = {
                "id": center_id,
                "name": properties.get("name", ""),
                "lat": properties.get("lat"),
                "lon": properties.get("lon"),
                "h3_9": properties.get("h3_9")
            }

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._executor,
                lambda: self.conn.upsertVertex("Center", center_id, vertex_data)
            )

            logger.debug(f"Created center node: {center_id}")
            return True

        except Exception as e:
            logger.error(f"Error creating center node {center_id}: {e}")
            return False

    async def create_parent_node(self, parent_id: str, **properties) -> bool:
        """Create parent node in TigerGraph."""
        if not self.conn:
            logger.error("Not connected to TigerGraph")
            return False

        try:
            vertex_data = {
                "id": parent_id,
                "home_lat": properties.get("home_lat"),
                "home_lon": properties.get("home_lon")
            }

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._executor,
                lambda: self.conn.upsertVertex("Parent", parent_id, vertex_data)
            )

            logger.debug(f"Created parent node: {parent_id}")
            return True

        except Exception as e:
            logger.error(f"Error creating parent node {parent_id}: {e}")
            return False

    async def create_feature_node(self, feature_key: str, category: str, **properties) -> bool:
        """Create feature node in TigerGraph."""
        if not self.conn:
            logger.error("Not connected to TigerGraph")
            return False

        try:
            vertex_data = {
                "key": feature_key,
                "category": category,
                **properties
            }

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._executor,
                lambda: self.conn.upsertVertex("Feature", feature_key, vertex_data)
            )

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
        raw_phrase: str | None = None,
        value_bool: bool | None = None,
        value_num: float | None = None,
        unit: str | None = None,
        **properties
    ) -> bool:
        """Create HAS_FEATURE edge in TigerGraph."""
        if not self.conn:
            logger.error("Not connected to TigerGraph")
            return False

        try:
            edge_data = {
                "confidence": confidence,
                "source": source,
                "raw": raw_phrase or "",
                "value_bool": value_bool,
                "value_num": value_num,
                "unit": unit or "",
                **properties
            }

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._executor,
                lambda: self.conn.upsertEdge("Center", center_id, "HAS_FEATURE", "Feature", feature_key, edge_data)
            )

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
        value_bool: bool | None = None,
        value_num: float | None = None,
        value_text: str | None = None,
        unit: str | None = None,
        **properties
    ) -> bool:
        """Create WANTS edge in TigerGraph."""
        if not self.conn:
            logger.error("Not connected to TigerGraph")
            return False

        try:
            edge_data = {
                "preference": preference,
                "value_bool": value_bool,
                "value_num": value_num,
                "value_text": value_text or "",
                "unit": unit or "",
                "confidence": confidence,
                **properties
            }

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._executor,
                lambda: self.conn.upsertEdge("Parent", parent_id, "WANTS", "Feature", feature_key, edge_data)
            )

            logger.debug(f"Created WANTS edge: {parent_id} -> {feature_key}")
            return True

        except Exception as e:
            logger.error(f"Error creating WANTS edge {parent_id}->{feature_key}: {e}")
            return False

    async def query_feature_matches(
        self,
        parent_id: str,
        candidate_center_ids: list[str]
    ) -> list[FeatureMatchResult]:
        """Query feature matches using TigerGraph."""
        if not self.conn:
            logger.error("Not connected to TigerGraph")
            return []

        try:
            loop = asyncio.get_event_loop()

            # Run installed query for feature matching
            result = await loop.run_in_executor(
                self._executor,
                lambda: self.conn.runInstalledQuery(
                    "match_parent_centers",
                    {
                        "parent_id": parent_id,
                        "center_ids": candidate_center_ids
                    }
                )
            )

            matches = []
            for item in result:
                matches.append(FeatureMatchResult(
                    center_id=item.get("center_id", ""),
                    feature_key=item.get("feature_key", ""),
                    preference=item.get("preference", "nice_to_have"),
                    center_confidence=item.get("center_confidence", 1.0),
                    parent_confidence=item.get("parent_confidence", 1.0),
                    feature_satisfied=item.get("feature_satisfied", False),
                    center_value=item.get("center_value"),
                    parent_value=item.get("parent_value"),
                    similarity_score=item.get("similarity_score", 1.0)
                ))

            logger.debug(f"Found {len(matches)} feature matches for parent {parent_id}")
            return matches

        except Exception as e:
            logger.error(f"Error querying feature matches for parent {parent_id}: {e}")
            return []

    async def query_centers_with_feature(
        self,
        feature_key: str,
        value_filter: dict[str, Any] | None = None
    ) -> list[str]:
        """Find centers with specific feature."""
        if not self.conn:
            logger.error("Not connected to TigerGraph")
            return []

        try:
            loop = asyncio.get_event_loop()

            result = await loop.run_in_executor(
                self._executor,
                lambda: self.conn.runInstalledQuery(
                    "centers_with_feature",
                    {
                        "feature_key": feature_key,
                        "value_filter": value_filter or {}
                    }
                )
            )

            center_ids = [item.get("center_id") for item in result]
            logger.debug(f"Found {len(center_ids)} centers with feature {feature_key}")
            return center_ids

        except Exception as e:
            logger.error(f"Error querying centers with feature {feature_key}: {e}")
            return []

    async def get_center_features(self, center_id: str) -> dict[str, Any]:
        """Get all features for a center."""
        if not self.conn:
            logger.error("Not connected to TigerGraph")
            return {}

        try:
            loop = asyncio.get_event_loop()

            result = await loop.run_in_executor(
                self._executor,
                lambda: self.conn.runInstalledQuery(
                    "get_center_features",
                    {"center_id": center_id}
                )
            )

            features = {}
            for item in result:
                feature_key = item.get("feature_key")
                if feature_key:
                    features[feature_key] = {
                        "confidence": item.get("confidence", 1.0),
                        "source": item.get("source", ""),
                        "value_bool": item.get("value_bool"),
                        "value_num": item.get("value_num"),
                        "unit": item.get("unit", ""),
                        "raw": item.get("raw", "")
                    }

            logger.debug(f"Retrieved {len(features)} features for center {center_id}")
            return features

        except Exception as e:
            logger.error(f"Error getting features for center {center_id}: {e}")
            return {}

    async def get_parent_preferences(self, parent_id: str) -> dict[str, Any]:
        """Get all preferences for a parent."""
        if not self.conn:
            logger.error("Not connected to TigerGraph")
            return {}

        try:
            loop = asyncio.get_event_loop()

            result = await loop.run_in_executor(
                self._executor,
                lambda: self.conn.runInstalledQuery(
                    "get_parent_preferences",
                    {"parent_id": parent_id}
                )
            )

            preferences = {}
            for item in result:
                feature_key = item.get("feature_key")
                if feature_key:
                    preferences[feature_key] = {
                        "preference": item.get("preference", "nice_to_have"),
                        "confidence": item.get("confidence", 1.0),
                        "value_bool": item.get("value_bool"),
                        "value_num": item.get("value_num"),
                        "value_text": item.get("value_text", ""),
                        "unit": item.get("unit", "")
                    }

            logger.debug(f"Retrieved {len(preferences)} preferences for parent {parent_id}")
            return preferences

        except Exception as e:
            logger.error(f"Error getting preferences for parent {parent_id}: {e}")
            return {}

    async def delete_center_features(self, center_id: str) -> bool:
        """Delete all features for a center."""
        if not self.conn:
            logger.error("Not connected to TigerGraph")
            return False

        try:
            loop = asyncio.get_event_loop()

            result = await loop.run_in_executor(
                self._executor,
                lambda: self.conn.runInstalledQuery(
                    "delete_center_features",
                    {"center_id": center_id}
                )
            )

            logger.debug(f"Deleted features for center {center_id}")
            return True

        except Exception as e:
            logger.error(f"Error deleting features for center {center_id}: {e}")
            return False

    async def delete_parent_preferences(self, parent_id: str) -> bool:
        """Delete all preferences for a parent."""
        if not self.conn:
            logger.error("Not connected to TigerGraph")
            return False

        try:
            loop = asyncio.get_event_loop()

            result = await loop.run_in_executor(
                self._executor,
                lambda: self.conn.runInstalledQuery(
                    "delete_parent_preferences",
                    {"parent_id": parent_id}
                )
            )

            logger.debug(f"Deleted preferences for parent {parent_id}")
            return True

        except Exception as e:
            logger.error(f"Error deleting preferences for parent {parent_id}: {e}")
            return False

    async def execute_raw_query(self, query: str, parameters: dict[str, Any] = None) -> Any:
        """Execute raw GSQL query."""
        if not self.conn:
            logger.error("Not connected to TigerGraph")
            return None

        try:
            loop = asyncio.get_event_loop()

            result = await loop.run_in_executor(
                self._executor,
                lambda: self.conn.gsql(query)
            )

            return result

        except Exception as e:
            logger.error(f"Error executing raw query: {e}")
            return None

    async def install_required_queries(self):
        """Install required GSQL queries for matching operations."""
        queries = {
            "match_parent_centers": """
            CREATE QUERY match_parent_centers(STRING parent_id, SET<STRING> center_ids) FOR GRAPH childcare {
                start = {Parent.*};
                parent = SELECT p FROM start:p WHERE p.id == parent_id;
                
                matched = SELECT c, f, p_edge, c_edge FROM parent:p -(WANTS:p_edge)- Feature:f -(HAS_FEATURE:c_edge)- Center:c
                         WHERE c.id IN center_ids
                         ACCUM @@results += FeatureMatch(c.id, f.key, p_edge.preference, 
                                                        c_edge.confidence, p_edge.confidence,
                                                        TRUE, c_edge.value_bool, p_edge.value_bool, 1.0);
                                                        
                PRINT @@results;
            }
            """,

            "centers_with_feature": """
            CREATE QUERY centers_with_feature(STRING feature_key, MAP<STRING,STRING> value_filter) FOR GRAPH childcare {
                centers = SELECT c FROM Center:c -(HAS_FEATURE)- Feature:f
                         WHERE f.key == feature_key;
                PRINT centers[centers.id AS center_id];
            }
            """,

            "get_center_features": """
            CREATE QUERY get_center_features(STRING center_id) FOR GRAPH childcare {
                features = SELECT f, e FROM Center:c -(HAS_FEATURE:e)- Feature:f
                          WHERE c.id == center_id
                          ACCUM @@results += FeatureData(f.key, e.confidence, e.source, 
                                                        e.value_bool, e.value_num, e.unit, e.raw);
                PRINT @@results;
            }
            """,

            "get_parent_preferences": """
            CREATE QUERY get_parent_preferences(STRING parent_id) FOR GRAPH childcare {
                preferences = SELECT f, e FROM Parent:p -(WANTS:e)- Feature:f
                             WHERE p.id == parent_id
                             ACCUM @@results += PreferenceData(f.key, e.preference, e.confidence,
                                                              e.value_bool, e.value_num, e.value_text, e.unit);
                PRINT @@results;
            }
            """,

            "delete_center_features": """
            CREATE QUERY delete_center_features(STRING center_id) FOR GRAPH childcare {
                DELETE e FROM Center:c -(HAS_FEATURE:e)- Feature:f WHERE c.id == center_id;
            }
            """,

            "delete_parent_preferences": """
            CREATE QUERY delete_parent_preferences(STRING parent_id) FOR GRAPH childcare {
                DELETE e FROM Parent:p -(WANTS:e)- Feature:f WHERE p.id == parent_id;
            }
            """
        }

        if not self.conn:
            logger.error("Not connected to TigerGraph")
            return False

        try:
            loop = asyncio.get_event_loop()

            for query_name, query_body in queries.items():
                # Install the query
                await loop.run_in_executor(
                    self._executor,
                    lambda: self.conn.gsql(query_body)
                )

                await loop.run_in_executor(
                    self._executor,
                    lambda: self.conn.gsql(f"INSTALL QUERY {query_name}")
                )

                logger.info(f"Installed TigerGraph query: {query_name}")

            return True

        except Exception as e:
            logger.error(f"Error installing queries: {e}")
            return False
