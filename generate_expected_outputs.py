#!/usr/bin/env python3

import json
import sys
from pathlib import Path
from uuid import UUID

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.matchmaker.graph.builder import MatchingGraphBuilder
from src.matchmaker.graph.matcher import GraphMatcher
from src.matchmaker.models.base import MatchMode
from src.matchmaker.models.requests import EnhancedRecommendationRequest
from src.matchmaker.scoring.composite_scorer import CompositeScorer
from src.matchmaker.utils.filters import HardConstraintFilter


def serialize_result(result):
    """Convert result object to serializable dict."""
    if hasattr(result, 'model_dump'):
        return result.model_dump()
    elif isinstance(result, UUID):
        return str(result)
    elif hasattr(result, '__dict__'):
        return {k: serialize_result(v) for k, v in result.__dict__.items()}
    elif isinstance(result, list):
        return [serialize_result(item) for item in result]
    elif isinstance(result, dict):
        return {k: serialize_result(v) for k, v in result.items()}
    else:
        return result

def generate_expected_output(case_name: str):
    """Generate expected output for a test case."""
    try:
        # Load test case
        test_data_path = Path("tests/data") / f"{case_name}.json"
        with open(test_data_path) as f:
            test_case = json.load(f)

        # Create matching components
        scorer = CompositeScorer()
        filter = HardConstraintFilter()
        builder = MatchingGraphBuilder(scorer, filter)
        matcher = GraphMatcher(seed=42)

        # Convert to request object
        request_data = test_case["request"]
        request = EnhancedRecommendationRequest(**request_data)

        # Build graph and match
        graph = builder.build_graph(
            applications=[request.application],
            centers=request.centers,
            respect_capacity=not request.include_full_centers
        )
        result = matcher.match(
            graph,
            mode=MatchMode.RECOMMEND,
            application_id=request.application.id,
            top_k=request.top_k or 10,
            include_full=getattr(request, 'include_full_centers', False)
        )

        # Create expected output structure
        expected_output = {
            "test_name": test_case["name"],
            "description": test_case["description"],
            "output": serialize_result(result)
        }

        # Write output file
        output_path = Path("tests/expected_outputs") / f"{case_name}_output.json"
        with open(output_path, 'w') as f:
            json.dump(expected_output, f, indent=2, default=str)

        print(f"✅ Generated {output_path}")

        # Print summary for verification
        print(f"   - Success: {result.success}")
        print(f"   - Offers: {len(result.offers)}")
        if result.offers:
            print(f"   - Top center: {result.offers[0].center_id}")
            print(f"   - Top score: {result.offers[0].score:.4f}")

        return True

    except Exception as e:
        print(f"❌ Error generating {case_name}: {e}")
        return False

def main():
    """Generate expected outputs for new test cases."""
    cases = [
        "case_11_comprehensive_preferences_test",
        "case_12_mixed_preferences_test",
        "case_13_strict_requirements_test"
    ]

    print("Generating expected outputs for new test cases...")

    success_count = 0
    for case in cases:
        if generate_expected_output(case):
            success_count += 1

    print(f"\n✨ Generated {success_count}/{len(cases)} expected output files")

if __name__ == "__main__":
    main()
