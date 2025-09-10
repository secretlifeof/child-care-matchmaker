#!/usr/bin/env python3
"""
Startup script for the Daycare Schedule Optimization Service
"""
import argparse
import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import uvicorn
from logging_config import setup_logging

from config import settings


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Daycare Schedule Optimization Service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                     # Run in production mode
  python run.py --dev               # Run in development mode
  python run.py --port 8001         # Run on different port
  python run.py --workers 2         # Run with 2 worker processes
  python run.py --test              # Run tests
  python run.py --check             # Check configuration
        """
    )

    parser.add_argument(
        '--dev', '--development',
        action='store_true',
        help='Run in development mode with auto-reload'
    )

    parser.add_argument(
        '--host',
        default=settings.host,
        help=f'Host to bind to (default: {settings.host})'
    )

    parser.add_argument(
        '--port', '-p',
        type=int,
        default=settings.port,
        help=f'Port to listen on (default: {settings.port})'
    )

    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=1 if settings.DEBUG else 4,
        help='Number of worker processes'
    )

    parser.add_argument(
        '--log-level',
        choices=['debug', 'info', 'warning', 'error'],
        default='debug' if settings.DEBUG else 'info',
        help='Log level'
    )

    parser.add_argument(
        '--test', '-t',
        action='store_true',
        help='Run tests instead of starting server'
    )

    parser.add_argument(
        '--check', '-c',
        action='store_true',
        help='Check configuration and dependencies'
    )

    parser.add_argument(
        '--generate-example',
        action='store_true',
        help='Generate example request and test the API'
    )

    return parser.parse_args()


def check_dependencies():
    """Check if all required dependencies are available"""
    print("Checking dependencies...")

    missing_deps = []

    try:
        import fastapi
        print(f"✓ FastAPI {fastapi.__version__}")
    except ImportError:
        missing_deps.append("fastapi")

    try:
        import uvicorn
        print(f"✓ Uvicorn {uvicorn.__version__}")
    except ImportError:
        missing_deps.append("uvicorn")

    try:
        import ortools
        print(f"✓ OR-Tools {ortools.__version__}")
    except ImportError:
        missing_deps.append("ortools")

    try:
        import redis
        print("✓ Redis client available")
    except ImportError:
        print("! Redis client not available (will use in-memory cache)")

    try:
        import pydantic
        print(f"✓ Pydantic {pydantic.__version__}")
    except ImportError:
        missing_deps.append("pydantic")

    if missing_deps:
        print(f"\n❌ Missing dependencies: {', '.join(missing_deps)}")
        print("Install with: pip install -r requirements.txt")
        return False

    print("\n✅ All dependencies satisfied")
    return True


def check_configuration():
    """Check configuration settings"""
    print("\nConfiguration check:")
    print(f"  App name: {settings.app_name}")
    print(f"  Version: {settings.app_version}")
    print(f"  Debug mode: {settings.DEBUG}")
    print(f"  Host: {settings.host}")
    print(f"  Port: {settings.port}")
    print(f"  Redis host: {settings.redis_host}:{settings.redis_port}")
    print(f"  Max solver time: {settings.max_solver_time_seconds}s")
    print(f"  Cache TTL: {settings.cache_ttl}s")

    # Check if logs directory exists
    logs_dir = Path("logs")
    if not logs_dir.exists():
        print(f"  Creating logs directory: {logs_dir}")
        logs_dir.mkdir(exist_ok=True)
    else:
        print(f"  Logs directory: {logs_dir} ✓")

    return True


def run_tests():
    """Run the test suite"""
    print("Running tests...")

    try:
        import pytest
        exit_code = pytest.main([
            "test_scheduler.py",
            "-v",
            "--tb=short",
            "--color=yes"
        ])
        return exit_code == 0
    except ImportError:
        print("❌ pytest not installed. Install with: pip install pytest")
        return False


async def generate_example():
    """Generate an example API request and test the service"""
    print("Generating example request...")

    import json
    import sys
    from datetime import date, time
    from uuid import uuid4
    sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
    from scheduler.models import (
        AgeGroup,
        Group,
        OptimizationConfig,
        OptimizationGoal,
        Qualification,
        Staff,
        StaffAvailability,
        StaffingRequirement,
        StaffRole,
        TimeSlot,
    )

    # Create example data
    staff_id_1 = uuid4()
    staff_id_2 = uuid4()
    group_id_1 = uuid4()
    group_id_2 = uuid4()

    example_request = ScheduleGenerationRequest(
        center_id=uuid4(),
        week_start_date=date(2024, 1, 15),  # Monday
        staff=[
            Staff(
                staff_id=staff_id_1,
                name="Alice Johnson",
                role=StaffRole.TEACHER,
                qualifications=[
                    Qualification(
                        qualification_type="certification",
                        qualification_name="Early Childhood Education",
                        is_verified=True
                    )
                ],
                availability=[
                    StaffAvailability(
                        day_of_week=0,
                        start_time=time(8, 0),
                        end_time=time(17, 0),
                        is_available=True
                    )
                ],
                preferences=[
                    StaffPreference(
                        preference_type=PreferenceType.PREFERRED_TIME,
                        day_of_week=0,
                        time_range_start=time(9, 0),
                        time_range_end=time(15, 0),
                        weight=0.8
                    )
                ],
                max_weekly_hours=40.0,
                hourly_rate=25.0
            ),
            Staff(
                staff_id=staff_id_2,
                name="Bob Smith",
                role=StaffRole.ASSISTANT,
                availability=[
                    StaffAvailability(
                        day_of_week=0,
                        start_time=time(7, 0),
                        end_time=time(16, 0),
                        is_available=True
                    )
                ],
                max_weekly_hours=30.0,
                hourly_rate=18.0
            )
        ],
        groups=[
            Group(
                group_id=group_id_1,
                name="Toddlers Room A",
                age_group=AgeGroup.TODDLER,
                capacity=12,
                current_enrollment=10,
                required_qualifications=["Early Childhood Education"]
            ),
            Group(
                group_id=group_id_2,
                name="Infants Room B",
                age_group=AgeGroup.INFANT,
                capacity=8,
                current_enrollment=6
            )
        ],
        staffing_requirements=[
            StaffingRequirement(
                group_id=group_id_1,
                time_slot=TimeSlot(
                    start_time=time(9, 0),
                    end_time=time(12, 0),
                    day_of_week=0
                ),
                min_staff_count=2,
                required_qualifications=["Early Childhood Education"]
            ),
            StaffingRequirement(
                group_id=group_id_2,
                time_slot=TimeSlot(
                    start_time=time(8, 0),
                    end_time=time(11, 0),
                    day_of_week=0
                ),
                min_staff_count=1
            )
        ],
        optimization_config=OptimizationConfig(
            goals=[OptimizationGoal.MINIMIZE_COST, OptimizationGoal.MAXIMIZE_SATISFACTION],
            max_solver_time=30
        )
    )

    # Save example to file
    example_file = Path("example_request.json")
    with open(example_file, 'w') as f:
        # Convert to dict for JSON serialization
        request_dict = example_request.dict()
        # Convert UUIDs to strings
        def convert_uuids(obj):
            if isinstance(obj, dict):
                return {k: convert_uuids(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_uuids(item) for item in obj]
            elif hasattr(obj, '__str__') and 'uuid' in str(type(obj)).lower():
                return str(obj)
            else:
                return obj

        request_dict = convert_uuids(request_dict)
        json.dump(request_dict, f, indent=2, default=str)

    print(f"✅ Example request saved to {example_file}")
    print(f"   Center ID: {example_request.center_id}")
    print(f"   Staff: {len(example_request.staff)} members")
    print(f"   Groups: {len(example_request.groups)} groups")
    print(f"   Requirements: {len(example_request.staffing_requirements)} requirements")

    # Test with the API if it's running
    try:
        import httpx

        print("\nTesting API connection...")
        async with httpx.AsyncClient() as client:
            # Test health endpoint
            try:
                response = await client.get("http://localhost:8000/health", timeout=5)
                if response.status_code == 200:
                    print("✅ API is running and healthy")

                    # Test the optimization endpoint
                    print("Testing schedule generation...")
                    response = await client.post(
                        "http://localhost:8000/api/schedule/generate",
                        json=request_dict,
                        timeout=60
                    )

                    if response.status_code == 200:
                        result = response.json()
                        print("✅ Schedule generated successfully!")
                        print(f"   Status: {result.get('optimization_result', {}).get('status')}")
                        print(f"   Shifts: {len(result.get('schedule', []))}")
                        print(f"   Conflicts: {len(result.get('conflicts', []))}")
                        print(f"   Solve time: {result.get('optimization_result', {}).get('solve_time_seconds', 0):.2f}s")
                    else:
                        print(f"❌ API error: {response.status_code}")
                        print(response.text[:500])
                else:
                    print(f"❌ API health check failed: {response.status_code}")
            except Exception as e:
                print(f"! API not running or not accessible: {e}")
                print("  Start the service first with: python run.py")

    except ImportError:
        print("! httpx not available for API testing")


def main():
    """Main entry point"""
    args = parse_args()

    # Setup logging first
    setup_logging()

    if args.check:
        success = check_dependencies() and check_configuration()
        sys.exit(0 if success else 1)

    if args.test:
        success = run_tests()
        sys.exit(0 if success else 1)

    if args.generate_example:
        asyncio.run(generate_example())
        sys.exit(0)

    # Check dependencies before starting
    if not check_dependencies():
        sys.exit(1)

    check_configuration()

    # Determine if we're running in development mode
    is_dev = args.dev or settings.DEBUG

    print(f"\n🚀 Starting {settings.app_name}")
    print(f"   Mode: {'Development' if is_dev else 'Production'}")
    print(f"   URL: http://{args.host}:{args.port}")
    if is_dev:
        print(f"   Docs: http://{args.host}:{args.port}/docs")
        print(f"   ReDoc: http://{args.host}:{args.port}/redoc")
    print(f"   Workers: {args.workers}")
    print()

    # Start the server
    try:
        uvicorn.run(
            "main:app",
            host=args.host,
            port=args.port,
            reload=is_dev,
            workers=1 if is_dev else args.workers,
            log_level=args.log_level,
            access_log=True,
            server_header=False,
            date_header=False
        )
    except KeyboardInterrupt:
        print("\n👋 Service stopped")
    except Exception as e:
        print(f"\n❌ Error starting service: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
