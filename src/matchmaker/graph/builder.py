"""Graph builder for matching problem."""

import logging
from typing import List, Tuple, Dict, Optional, Set
from uuid import UUID
import networkx as nx
from geopy.distance import geodesic

from ..models.base import (
    Application, Center, CapacityBucket, 
    Location, ComparisonOperator
)
from ..scoring.composite_scorer import CompositeScorer
from ..utils.filters import HardConstraintFilter

logger = logging.getLogger(__name__)


class MatchingGraphBuilder:
    """Builds bipartite graph for matching problem."""
    
    def __init__(
        self,
        scorer: CompositeScorer,
        filter: HardConstraintFilter,
        max_candidates_per_application: int = 50,
        distance_decay_factor: float = 0.1
    ):
        self.scorer = scorer
        self.filter = filter
        self.max_candidates = max_candidates_per_application
        self.distance_decay = distance_decay_factor
    
    def build_graph(
        self,
        applications: List[Application],
        centers: List[Center],
        respect_capacity: bool = True
    ) -> nx.Graph:
        """
        Build bipartite graph with applications on one side and capacity buckets on other.
        
        Args:
            applications: List of applications to match
            centers: List of available centers
            respect_capacity: Whether to respect capacity constraints
            
        Returns:
            NetworkX bipartite graph with edge weights
        """
        G = nx.Graph()
        
        # Add application nodes
        for app in applications:
            G.add_node(
                f"app_{app.id}",
                bipartite=0,
                type="application",
                application=app
            )
        
        # Add capacity bucket nodes
        bucket_map = {}
        for center in centers:
            for bucket in center.capacity_buckets:
                if not respect_capacity or bucket.is_available:
                    bucket_id = f"bucket_{bucket.id}"
                    G.add_node(
                        bucket_id,
                        bipartite=1,
                        type="bucket",
                        bucket=bucket,
                        center=center
                    )
                    bucket_map[bucket_id] = (center, bucket)
        
        # Create edges with weights
        edges = self._create_edges(applications, centers, bucket_map, respect_capacity)
        
        for app_id, bucket_id, weight, metadata in edges:
            G.add_edge(
                app_id,
                bucket_id,
                weight=weight,
                metadata=metadata
            )
        
        logger.info(
            f"Built graph with {len(applications)} applications, "
            f"{len(bucket_map)} buckets, {G.number_of_edges()} edges"
        )
        
        return G
    
    def _create_edges(
        self,
        applications: List[Application],
        centers: List[Center],
        bucket_map: Dict[str, Tuple[Center, CapacityBucket]],
        respect_capacity: bool
    ) -> List[Tuple[str, str, float, Dict]]:
        """Create edges between applications and buckets."""
        edges = []
        
        for app in applications:
            # Get candidate centers for this application
            candidates = self._get_candidate_centers(app, centers)
            
            for center in candidates[:self.max_candidates]:
                # Check each bucket in the center
                for bucket in center.capacity_buckets:
                    bucket_id = f"bucket_{bucket.id}"
                    
                    if bucket_id not in bucket_map:
                        continue
                    
                    # Apply hard filters
                    if not self._passes_hard_filters(app, center, bucket):
                        continue
                    
                    # Calculate weight
                    weight, metadata = self._calculate_edge_weight(app, center, bucket)
                    
                    if weight > 0:
                        edges.append((
                            f"app_{app.id}",
                            bucket_id,
                            weight,
                            metadata
                        ))
        
        return edges
    
    def _get_candidate_centers(
        self,
        application: Application,
        centers: List[Center]
    ) -> List[Center]:
        """Get candidate centers for an application, sorted by distance."""
        candidates = []
        
        for center in centers:
            # Calculate distance
            distance_km = self._calculate_distance(
                application.home_location,
                center.location
            )
            
            # Check max distance constraint
            if application.max_distance_km and distance_km > application.max_distance_km:
                continue
            
            candidates.append((distance_km, center))
        
        # Sort by distance and return centers only
        candidates.sort(key=lambda x: x[0])
        return [c for _, c in candidates]
    
    def _passes_hard_filters(
        self,
        application: Application,
        center: Center,
        bucket: CapacityBucket
    ) -> bool:
        """Check if application-center-bucket combination passes hard filters."""
        # Age compatibility
        for child in application.children:
            if not bucket.age_band.contains_age(child.age_months):
                return False
        
        # Opening hours compatibility
        if not self._check_hours_overlap(application, center):
            return False
        
        # Start date compatibility
        if bucket.start_month < application.desired_start_date:
            return False
        
        # Check EXCLUDE preferences
        for pref in application.preferences:
            if pref.constraint_type == "exclude":
                if self._matches_exclusion(pref, center):
                    return False
        
        # Check MUST_HAVE preferences
        for pref in application.preferences:
            if pref.threshold >= 0.9:  # Must have
                if not self._satisfies_requirement(pref, center):
                    return False
        
        # Reserved bucket compatibility
        if bucket.reserved_label:
            if bucket.reserved_label not in application.priority_flags:
                return False
        
        return True
    
    def _check_hours_overlap(
        self,
        application: Application,
        center: Center
    ) -> bool:
        """Check if desired hours overlap with center hours."""
        if not application.desired_hours or not center.opening_hours:
            return True  # No constraint if not specified
        
        for app_slot in application.desired_hours:
            has_overlap = False
            for center_slot in center.opening_hours:
                if app_slot.overlaps_with(center_slot):
                    has_overlap = True
                    break
            if not has_overlap:
                return False
        
        return True
    
    def _matches_exclusion(
        self,
        preference,
        center: Center
    ) -> bool:
        """Check if center matches an exclusion preference."""
        prop = center.get_property(preference.property_key)
        if not prop:
            return False
        
        if preference.operator == ComparisonOperator.EQUALS:
            return prop.get_value() == preference.get_value()
        elif preference.operator == ComparisonOperator.CONTAINS:
            return preference.get_value() in str(prop.get_value())
        
        return False
    
    def _satisfies_requirement(
        self,
        preference,
        center: Center
    ) -> bool:
        """Check if center satisfies a must-have requirement."""
        prop = center.get_property(preference.property_key)
        if not prop:
            return False
        
        if preference.operator == ComparisonOperator.EQUALS:
            return prop.get_value() == preference.get_value()
        elif preference.operator == ComparisonOperator.GREATER_THAN:
            return float(prop.get_value()) > float(preference.get_value())
        elif preference.operator == ComparisonOperator.LESS_THAN:
            return float(prop.get_value()) < float(preference.get_value())
        elif preference.operator == ComparisonOperator.CONTAINS:
            return preference.get_value() in str(prop.get_value())
        elif preference.operator == ComparisonOperator.BETWEEN:
            if isinstance(preference.get_value(), list) and len(preference.get_value()) == 2:
                val = float(prop.get_value())
                return preference.get_value()[0] <= val <= preference.get_value()[1]
        
        return True
    
    def _calculate_edge_weight(
        self,
        application: Application,
        center: Center,
        bucket: CapacityBucket
    ) -> Tuple[float, Dict]:
        """Calculate weight for edge between application and bucket."""
        metadata = {
            "distance_km": 0,
            "preference_score": 0,
            "sibling_bonus": 0,
            "priority_bonus": 0,
            "components": {}
        }
        
        # Distance component
        distance_km = self._calculate_distance(
            application.home_location,
            center.location
        )
        metadata["distance_km"] = distance_km
        distance_penalty = distance_km * self.distance_decay
        
        # Preference scoring
        pref_score = self.scorer.score(application, center, bucket)
        metadata["preference_score"] = pref_score
        metadata["components"] = self.scorer.get_score_breakdown(application, center, bucket)
        
        # Sibling bonus (if multiple children in same center)
        sibling_bonus = 0
        if application.has_siblings():
            # Check if this center already has siblings from this family
            sibling_bonus = 0.2  # Configurable
            metadata["sibling_bonus"] = sibling_bonus
        
        # Priority bonus (for reserved buckets)
        priority_bonus = 0
        if bucket.reserved_label in application.priority_flags:
            priority_bonus = 0.3  # Configurable
            metadata["priority_bonus"] = priority_bonus
        
        # Final weight calculation
        weight = max(0, pref_score - distance_penalty + sibling_bonus + priority_bonus)
        
        return weight, metadata
    
    def _calculate_distance(self, loc1: Location, loc2: Location) -> float:
        """Calculate distance in kilometers between two locations."""
        return geodesic(
            (loc1.latitude, loc1.longitude),
            (loc2.latitude, loc2.longitude)
        ).kilometers
    
    def prune_edges(
        self,
        graph: nx.Graph,
        keep_top_k_per_application: Optional[int] = None,
        min_weight_threshold: float = 0.0
    ) -> nx.Graph:
        """
        Prune edges to reduce graph size.
        
        Args:
            graph: Input graph
            keep_top_k_per_application: Keep only top K edges per application
            min_weight_threshold: Remove edges below this weight
            
        Returns:
            Pruned graph
        """
        if not keep_top_k_per_application and min_weight_threshold <= 0:
            return graph
        
        pruned = graph.copy()
        
        # Get application nodes
        app_nodes = [n for n, d in pruned.nodes(data=True) if d.get("type") == "application"]
        
        for app_node in app_nodes:
            # Get all edges for this application
            edges = [(app_node, nbr, pruned[app_node][nbr]["weight"]) 
                    for nbr in pruned.neighbors(app_node)]
            
            if not edges:
                continue
            
            # Sort by weight descending
            edges.sort(key=lambda x: x[2], reverse=True)
            
            # Determine which edges to remove
            edges_to_remove = []
            
            if keep_top_k_per_application and len(edges) > keep_top_k_per_application:
                edges_to_remove.extend(edges[keep_top_k_per_application:])
            
            for source, target, weight in edges:
                if weight < min_weight_threshold and (source, target) not in edges_to_remove:
                    edges_to_remove.append((source, target))
            
            # Remove edges
            for source, target, _ in edges_to_remove:
                pruned.remove_edge(source, target)
        
        logger.info(
            f"Pruned graph from {graph.number_of_edges()} to "
            f"{pruned.number_of_edges()} edges"
        )
        
        return pruned