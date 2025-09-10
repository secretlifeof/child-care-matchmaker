"""TigerGraph client for semantic property relationships."""

import logging

import pyTigerGraph as tg
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class RelatedProperty(BaseModel):
    """Related property from TigerGraph."""
    name: str
    similarity: float
    relationship_type: str
    confidence: float
    source: str  # "conceptnet", "embedding", "learned"


class PropertyRelationship(BaseModel):
    """Property relationship data."""
    source_property: str
    target_property: str
    similarity: float
    relationship_type: str
    confidence: float


class TigerGraphClient:
    """Client for querying TigerGraph semantic relationships."""

    def __init__(self, host: str, username: str, password: str, graph_name: str = "childcare"):
        """Initialize TigerGraph connection."""
        self.host = host
        self.graph_name = graph_name

        try:
            self.conn = tg.TigerGraphConnection(
                host=host,
                username=username,
                password=password,
                graphname=graph_name
            )
            self._token = self.conn.getToken()
            logger.info(f"Connected to TigerGraph at {host}")
        except Exception as e:
            logger.error(f"Failed to connect to TigerGraph: {e}")
            self.conn = None

    async def get_related_properties(self, property_name: str,
                                   max_depth: int = 2,
                                   min_similarity: float = 0.7) -> list[RelatedProperty]:
        """
        Get semantically related properties from TigerGraph.
        
        Args:
            property_name: The property to find relationships for
            max_depth: Maximum traversal depth in graph
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of related properties with similarity scores
        """
        if not self.conn:
            logger.warning("TigerGraph not connected, returning empty relationships")
            return []

        try:
            # Run the installed query to find related properties
            result = self.conn.runInstalledQuery(
                "find_related_properties",
                {
                    "property_name": property_name,
                    "max_depth": max_depth,
                    "min_similarity": min_similarity
                }
            )

            related = []
            for item in result:
                related.append(RelatedProperty(
                    name=item.get("name", ""),
                    similarity=item.get("similarity", 0.0),
                    relationship_type=item.get("relationship_type", "related"),
                    confidence=item.get("confidence", 0.0),
                    source=item.get("source", "unknown")
                ))

            logger.debug(f"Found {len(related)} related properties for '{property_name}'")
            return related

        except Exception as e:
            logger.error(f"Error querying TigerGraph for '{property_name}': {e}")
            return []

    async def property_exists(self, property_name: str) -> bool:
        """Check if property exists in TigerGraph."""
        if not self.conn:
            return False

        try:
            result = self.conn.runInstalledQuery(
                "check_property_exists",
                {"property_name": property_name}
            )
            return len(result) > 0
        except Exception as e:
            logger.error(f"Error checking property existence: {e}")
            return False

    async def get_property_hierarchy(self, property_name: str) -> dict[str, list[str]]:
        """
        Get property hierarchy (broader/narrower terms).
        
        Returns:
            Dict with 'broader', 'narrower', and 'related' property lists
        """
        if not self.conn:
            return {"broader": [], "narrower": [], "related": []}

        try:
            result = self.conn.runInstalledQuery(
                "get_property_hierarchy",
                {"property_name": property_name}
            )

            hierarchy = {"broader": [], "narrower": [], "related": []}
            for item in result:
                rel_type = item.get("relationship_type", "related")
                prop_name = item.get("name", "")

                if rel_type == "broader":
                    hierarchy["broader"].append(prop_name)
                elif rel_type == "narrower":
                    hierarchy["narrower"].append(prop_name)
                else:
                    hierarchy["related"].append(prop_name)

            return hierarchy

        except Exception as e:
            logger.error(f"Error getting property hierarchy: {e}")
            return {"broader": [], "narrower": [], "related": []}

    async def get_center_properties(self, center_id: str) -> list[dict]:
        """Get all properties for a center from TigerGraph."""
        if not self.conn:
            return []

        try:
            result = self.conn.runInstalledQuery(
                "get_center_properties",
                {"center_id": center_id}
            )

            return result

        except Exception as e:
            logger.error(f"Error getting center properties: {e}")
            return []

    def install_queries(self):
        """Install required GSQL queries in TigerGraph."""
        if not self.conn:
            logger.error("Cannot install queries - no TigerGraph connection")
            return

        queries = {
            "find_related_properties": """
            CREATE QUERY find_related_properties(STRING property_name, INT max_depth, DOUBLE min_similarity) FOR GRAPH childcare {
                start = {Property.*};
                related = SELECT t FROM start:s -(RELATES_TO>){1,max_depth} Property:t
                         WHERE s.name == property_name 
                         AND t.similarity > min_similarity
                         ACCUM t.@similarity = t.similarity,
                               t.@relationship_type = t.relationship_type,
                               t.@confidence = t.confidence,
                               t.@source = t.source
                         ORDER BY t.similarity DESC
                         LIMIT 20;
                         
                PRINT related[related.name, related.@similarity AS similarity, 
                             related.@relationship_type AS relationship_type,
                             related.@confidence AS confidence,
                             related.@source AS source];
            }
            """,

            "check_property_exists": """
            CREATE QUERY check_property_exists(STRING property_name) FOR GRAPH childcare {
                result = SELECT s FROM Property:s 
                        WHERE s.name == property_name;
                PRINT result[result.name];
            }
            """,

            "get_property_hierarchy": """
            CREATE QUERY get_property_hierarchy(STRING property_name) FOR GRAPH childcare {
                start = {Property.*};
                
                broader = SELECT t FROM start:s -(RELATES_TO>)- Property:t
                         WHERE s.name == property_name 
                         AND t.relationship_type == "broader";
                         
                narrower = SELECT t FROM start:s -(RELATES_TO>)- Property:t
                          WHERE s.name == property_name 
                          AND t.relationship_type == "narrower";
                          
                related = SELECT t FROM start:s -(RELATES_TO>)- Property:t
                         WHERE s.name == property_name 
                         AND t.relationship_type == "related";
                
                PRINT broader[broader.name, broader.relationship_type];
                PRINT narrower[narrower.name, narrower.relationship_type];  
                PRINT related[related.name, related.relationship_type];
            }
            """,

            "get_center_properties": """
            CREATE QUERY get_center_properties(STRING center_id) FOR GRAPH childcare {
                center_props = SELECT p FROM Center:c -(HAS_PROPERTY>)- Property:p
                              WHERE c.center_id == center_id
                              ACCUM p.@value_boolean = p.value_boolean,
                                    p.@value_number = p.value_number,
                                    p.@value_text = p.value_text,
                                    p.@value_list = p.value_list,
                                    p.@confidence = p.confidence,
                                    p.@source = p.source;
                                    
                PRINT center_props[center_props.name, center_props.property_type,
                                  center_props.@value_boolean AS value_boolean,
                                  center_props.@value_number AS value_number, 
                                  center_props.@value_text AS value_text,
                                  center_props.@value_list AS value_list,
                                  center_props.@confidence AS confidence,
                                  center_props.@source AS source];
            }
            """
        }

        for query_name, query_body in queries.items():
            try:
                self.conn.gsql(query_body)
                self.conn.gsql(f"INSTALL QUERY {query_name}")
                logger.info(f"Installed TigerGraph query: {query_name}")
            except Exception as e:
                logger.error(f"Failed to install query {query_name}: {e}")


class SemanticPropertyExpander:
    """Expands parent preferences using TigerGraph semantic relationships."""

    def __init__(self, tigergraph_client: TigerGraphClient):
        self.tg_client = tigergraph_client
        self._cache = {}  # Simple in-memory cache

    async def expand_preferences(self, preferences: list['ParentPreference']) -> list['ParentPreference']:
        """
        Expand preferences with semantically related properties.
        
        Args:
            preferences: Original parent preferences
            
        Returns:
            Expanded list including semantic matches with reduced weights
        """
        from ..models.base import ParentPreference

        expanded = list(preferences)  # Start with original preferences

        for pref in preferences:
            # Skip if this is an absolute requirement (don't dilute)
            if pref.is_absolute:
                continue

            # Check cache first
            cache_key = f"{pref.property_key}_{pref.strength.value}"
            if cache_key in self._cache:
                related = self._cache[cache_key]
            else:
                related = await self.tg_client.get_related_properties(
                    pref.property_key,
                    max_depth=2,
                    min_similarity=0.7
                )
                self._cache[cache_key] = related

            # Create semantic preferences with reduced strength
            for rel_prop in related[:5]:  # Limit to top 5 to avoid noise
                semantic_strength = self._derive_semantic_strength(pref.strength, rel_prop.similarity)

                semantic_pref = ParentPreference(
                    id=pref.id,  # Same ID for tracking
                    profile_id=pref.profile_id,
                    property_key=rel_prop.name,
                    property_type=pref.property_type,
                    operator=pref.operator,
                    strength=semantic_strength,
                    value_boolean=pref.value_boolean,
                    value_numeric=pref.value_numeric,
                    value_text=pref.value_text,
                    value_list=pref.value_list,
                    min_value=pref.min_value,
                    max_value=pref.max_value
                )

                expanded.append(semantic_pref)

        logger.info(f"Expanded {len(preferences)} preferences to {len(expanded)} with semantic relationships")
        return expanded

    def _derive_semantic_strength(self, original_strength: 'PreferenceStrength', similarity: float) -> 'PreferenceStrength':
        """
        Derive semantic preference strength based on original strength and similarity.
        
        Args:
            original_strength: Original preference strength  
            similarity: Similarity score (0.0-1.0)
            
        Returns:
            Reduced strength for semantic match
        """
        from ..models.base import PreferenceStrength

        # High similarity maintains strength better
        if similarity >= 0.9:
            strength_mapping = {
                PreferenceStrength.PREFERRED: PreferenceStrength.PREFERRED,
                PreferenceStrength.NICE_TO_HAVE: PreferenceStrength.NICE_TO_HAVE
            }
        elif similarity >= 0.8:
            strength_mapping = {
                PreferenceStrength.PREFERRED: PreferenceStrength.NICE_TO_HAVE,
                PreferenceStrength.NICE_TO_HAVE: PreferenceStrength.NICE_TO_HAVE
            }
        else:  # similarity >= 0.7
            strength_mapping = {
                PreferenceStrength.PREFERRED: PreferenceStrength.NICE_TO_HAVE,
                PreferenceStrength.NICE_TO_HAVE: PreferenceStrength.NICE_TO_HAVE
            }

        return strength_mapping.get(original_strength, PreferenceStrength.NICE_TO_HAVE)
