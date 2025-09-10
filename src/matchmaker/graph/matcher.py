"""Graph matching algorithms for different modes."""

import logging
from uuid import UUID

import networkx as nx
import numpy as np
from ortools.graph.python import min_cost_flow

from ..models.base import MatchMode
from ..models.results import (
    MatchExplanation,
    MatchOffer,
    MatchResult,
    WaitlistEntry,
)

logger = logging.getLogger(__name__)


class GraphMatcher:
    """Implements different matching algorithms for various modes."""

    def __init__(self, seed: int | None = None):
        self.seed = seed
        if seed:
            np.random.seed(seed)

    def match(
        self,
        graph: nx.Graph,
        mode: MatchMode,
        **kwargs
    ) -> MatchResult:
        """
        Execute matching based on mode.
        
        Args:
            graph: Bipartite graph with applications and buckets
            mode: Matching mode (recommend, allocate, waitlist)
            **kwargs: Mode-specific parameters
            
        Returns:
            MatchResult with offers or rankings
        """
        if mode == MatchMode.RECOMMEND:
            return self._recommend_mode(graph, **kwargs)
        elif mode == MatchMode.ALLOCATE:
            return self._allocate_mode(graph, **kwargs)
        elif mode == MatchMode.WAITLIST:
            return self._waitlist_mode(graph, **kwargs)
        else:
            raise ValueError(f"Unknown matching mode: {mode}")

    def _recommend_mode(
        self,
        graph: nx.Graph,
        application_id: UUID,
        top_k: int = 10,
        include_full: bool = False
    ) -> MatchResult:
        """
        Generate top-K recommendations for a specific application.
        
        Args:
            graph: Matching graph
            application_id: ID of application to get recommendations for
            top_k: Number of recommendations
            include_full: Whether to include full buckets
            
        Returns:
            MatchResult with recommended centers
        """
        app_node = f"app_{application_id}"

        if app_node not in graph:
            return MatchResult(
                mode=MatchMode.RECOMMEND,
                offers=[],
                success=False,
                message=f"Application {application_id} not found in graph"
            )

        # Get all edges for this application
        edges = []
        for neighbor in graph.neighbors(app_node):
            edge_data = graph[app_node][neighbor]
            node_data = graph.nodes[neighbor]

            bucket = node_data["bucket"]
            center = node_data["center"]

            # Skip full buckets unless requested
            if not include_full and not bucket.is_available:
                continue

            edges.append({
                "bucket": bucket,
                "center": center,
                "weight": edge_data["weight"],
                "metadata": edge_data.get("metadata", {})
            })

        # Sort by weight descending
        edges.sort(key=lambda x: x["weight"], reverse=True)

        # Create offers for top-K
        offers = []
        for i, edge in enumerate(edges[:top_k]):
            offer = MatchOffer(
                application_id=application_id,
                center_id=edge["center"].id,
                bucket_id=edge["bucket"].id,
                rank=i + 1,
                score=edge["weight"],
                is_available=edge["bucket"].is_available,
                explanation=self._create_explanation(edge["metadata"])
            )
            offers.append(offer)

        return MatchResult(
            mode=MatchMode.RECOMMEND,
            offers=offers,
            success=True,
            total_applications=1,
            matched_applications=1 if offers else 0
        )

    def _allocate_mode(
        self,
        graph: nx.Graph,
        respect_capacity: bool = True,
        prioritize_siblings: bool = True
    ) -> MatchResult:
        """
        Global optimal allocation using min-cost flow.
        
        Args:
            graph: Matching graph
            respect_capacity: Whether to respect capacity constraints
            prioritize_siblings: Whether to prioritize sibling co-assignment
            
        Returns:
            MatchResult with allocations
        """
        # Build min-cost flow problem
        flow_graph, node_map = self._build_flow_graph(graph, respect_capacity)

        # Solve using OR-Tools
        solution = self._solve_min_cost_flow(flow_graph, node_map)

        if solution is None:
            # Count total applications in the graph
            total_apps = sum(1 for node, data in graph.nodes(data=True)
                           if data.get("type") == "application")
            return MatchResult(
                mode=MatchMode.ALLOCATE,
                offers=[],
                success=False,
                message="No feasible allocation found",
                total_applications=total_apps,
                matched_applications=0,
                coverage_rate=0.0
            )

        # Extract offers from solution
        offers = self._extract_offers_from_flow(solution, graph, node_map)

        # Handle sibling constraints if needed
        if prioritize_siblings:
            offers = self._adjust_for_siblings(offers, graph)

        matched_apps = len(set(o.application_id for o in offers))
        total_apps = len([n for n in graph.nodes() if n.startswith("app_")])

        return MatchResult(
            mode=MatchMode.ALLOCATE,
            offers=offers,
            success=True,
            total_applications=total_apps,
            matched_applications=matched_apps,
            coverage_rate=matched_apps / total_apps if total_apps > 0 else 0
        )

    def _waitlist_mode(
        self,
        graph: nx.Graph,
        center_id: UUID,
        bucket_id: UUID | None = None,
        policy_tiers: dict[str, int] | None = None
    ) -> MatchResult:
        """
        Generate waitlist ranking for a specific center.
        
        Args:
            graph: Matching graph
            center_id: Center to generate waitlist for
            bucket_id: Specific bucket (optional)
            policy_tiers: Priority tiers for policy compliance
            
        Returns:
            MatchResult with waitlist entries
        """
        # Find all applications interested in this center
        candidates = []

        for node, data in graph.nodes(data=True):
            if data.get("type") != "bucket":
                continue

            bucket = data["bucket"]
            center = data["center"]

            if center.id != center_id:
                continue

            if bucket_id and bucket.id != bucket_id:
                continue

            # Get all applications connected to this bucket
            for app_node in graph.neighbors(node):
                if not app_node.startswith("app_"):
                    continue

                edge_data = graph[app_node][node]
                app_data = graph.nodes[app_node]
                application = app_data["application"]

                candidates.append({
                    "application": application,
                    "bucket": bucket,
                    "weight": edge_data["weight"],
                    "metadata": edge_data.get("metadata", {})
                })

        # Apply policy tiers if provided
        if policy_tiers:
            candidates = self._apply_policy_tiers(candidates, policy_tiers)

        # Sort by weight descending (after tier adjustment)
        candidates.sort(key=lambda x: x["weight"], reverse=True)

        # Create waitlist entries
        entries = []
        for i, candidate in enumerate(candidates):
            entry = WaitlistEntry(
                center_id=center_id,
                bucket_id=candidate["bucket"].id,
                application_id=candidate["application"].id,
                rank=i + 1,
                score=candidate.get("original_score", candidate["weight"]),
                tier=candidate.get("tier", "standard"),
                explanation=self._create_explanation(candidate["metadata"])
            )
            entries.append(entry)

        return MatchResult(
            mode=MatchMode.WAITLIST,
            waitlist_entries=entries,
            success=True,
            message=f"Generated waitlist for center {center_id} with {len(entries)} entries"
        )

    def _build_flow_graph(
        self,
        graph: nx.Graph,
        respect_capacity: bool
    ) -> tuple[min_cost_flow.SimpleMinCostFlow, dict]:
        """Build min-cost flow problem from bipartite graph."""
        mcf = min_cost_flow.SimpleMinCostFlow()

        # Create node mapping
        node_map = {"source": 0, "sink": 1}
        node_counter = 2

        # Map graph nodes to flow nodes
        for node in graph.nodes():
            node_map[node] = node_counter
            node_counter += 1

        # Add arcs from source to applications (capacity 1)
        for node, data in graph.nodes(data=True):
            if data.get("type") == "application":
                mcf.add_arc_with_capacity_and_unit_cost(
                    node_map["source"],
                    node_map[node],
                    1,  # capacity
                    0   # cost
                )

        # Add arcs from applications to buckets
        for app_node, bucket_node in graph.edges():
            edge_data = graph[app_node][bucket_node]
            # Convert weight to cost (negative for maximization)
            cost = int(-edge_data["weight"] * 1000)  # Scale for integer costs

            mcf.add_arc_with_capacity_and_unit_cost(
                node_map[app_node],
                node_map[bucket_node],
                1,  # capacity
                cost
            )

        # Add arcs from buckets to sink
        for node, data in graph.nodes(data=True):
            if data.get("type") == "bucket":
                bucket = data["bucket"]
                capacity = bucket.available_capacity if respect_capacity else 1000

                mcf.add_arc_with_capacity_and_unit_cost(
                    node_map[node],
                    node_map["sink"],
                    capacity,
                    0  # cost
                )

        # Set supply/demand
        num_apps = len([n for n in graph.nodes() if n.startswith("app_")])
        mcf.set_node_supply(node_map["source"], num_apps)
        mcf.set_node_supply(node_map["sink"], -num_apps)

        return mcf, node_map

    def _solve_min_cost_flow(
        self,
        mcf: min_cost_flow.SimpleMinCostFlow,
        node_map: dict
    ) -> dict | None:
        """Solve min-cost flow problem."""
        status = mcf.solve()

        if status != mcf.OPTIMAL:
            logger.warning(f"Flow solver returned status: {status}")
            return None

        # Extract solution
        solution = {
            "flows": [],
            "total_cost": mcf.optimal_cost()
        }

        reverse_map = {v: k for k, v in node_map.items()}

        for arc in range(mcf.num_arcs()):
            if mcf.flow(arc) > 0:
                tail = mcf.tail(arc)
                head = mcf.head(arc)

                tail_node = reverse_map.get(tail, f"node_{tail}")
                head_node = reverse_map.get(head, f"node_{head}")

                if tail_node.startswith("app_") and head_node.startswith("bucket_"):
                    solution["flows"].append({
                        "application": tail_node,
                        "bucket": head_node,
                        "flow": mcf.flow(arc)
                    })

        return solution

    def _extract_offers_from_flow(
        self,
        solution: dict,
        graph: nx.Graph,
        node_map: dict
    ) -> list[MatchOffer]:
        """Extract match offers from flow solution."""
        offers = []

        for flow in solution["flows"]:
            app_node = flow["application"]
            bucket_node = flow["bucket"]

            # Get application and bucket data
            app_data = graph.nodes[app_node]
            bucket_data = graph.nodes[bucket_node]
            edge_data = graph[app_node][bucket_node]

            application = app_data["application"]
            bucket = bucket_data["bucket"]
            center = bucket_data["center"]

            offer = MatchOffer(
                application_id=application.id,
                center_id=center.id,
                bucket_id=bucket.id,
                score=edge_data["weight"],
                is_allocated=True,
                explanation=self._create_explanation(edge_data.get("metadata", {}))
            )
            offers.append(offer)

        return offers

    def _adjust_for_siblings(
        self,
        offers: list[MatchOffer],
        graph: nx.Graph
    ) -> list[MatchOffer]:
        """Adjust allocations to keep siblings together when possible."""
        # Group offers by family
        family_offers = {}
        for offer in offers:
            # Get application from graph
            app_node = f"app_{offer.application_id}"
            if app_node in graph:
                app = graph.nodes[app_node]["application"]
                family_id = app.family_id

                if family_id not in family_offers:
                    family_offers[family_id] = []
                family_offers[family_id].append(offer)

        # Check families with multiple children
        adjusted_offers = []
        for family_id, family_offer_list in family_offers.items():
            if len(family_offer_list) > 1:
                # Check if siblings are in different centers
                centers = set(o.center_id for o in family_offer_list)
                if len(centers) > 1:
                    # Try to consolidate to one center
                    # (Simplified - in practice would need more sophisticated logic)
                    best_center = max(centers, key=lambda c:
                        sum(1 for o in family_offer_list if o.center_id == c))

                    for offer in family_offer_list:
                        if offer.center_id != best_center:
                            # Mark for potential reallocation
                            offer.is_tentative = True

            adjusted_offers.extend(family_offer_list)

        return adjusted_offers

    def _apply_policy_tiers(
        self,
        candidates: list[dict],
        policy_tiers: dict[str, int]
    ) -> list[dict]:
        """Apply policy tier priorities to candidates."""
        for candidate in candidates:
            application = candidate["application"]

            # Store original score for WaitlistEntry validation
            candidate["original_score"] = candidate["weight"]

            # Determine tier based on priority flags
            tier = "standard"
            tier_boost = 0

            for flag in application.priority_flags:
                if flag in policy_tiers:
                    flag_tier = policy_tiers[flag]
                    if flag_tier > tier_boost:
                        tier = flag
                        tier_boost = flag_tier

            candidate["tier"] = tier
            # Add tier boost to weight for sorting
            candidate["weight"] += tier_boost

        return candidates

    def _create_explanation(self, metadata: dict) -> MatchExplanation:
        """Create explanation from edge metadata."""
        return MatchExplanation(
            distance_km=metadata.get("distance_km", 0),
            preference_score=metadata.get("preference_score", 0),
            components=metadata.get("components", {}),
            constraints_satisfied=metadata.get("constraints_satisfied", []),
            constraints_violated=metadata.get("constraints_violated", [])
        )
