"""
Schedule validation routes
"""

import logging
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends

from ...models import ScheduleValidationRequest, ScheduleValidationResponse
from ...scheduler import ScheduleOptimizer
from ...utils import PerformanceProfiler
from ..dependencies import get_optimizer, get_profiler, rate_limit

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/schedule", tags=["Schedule Validation"])


@router.post(
    "/validate",
    response_model=ScheduleValidationResponse,
    summary="Validate existing schedule",
    description="""
    Validate an existing schedule against business rules and constraints.
    
    Checks for:
    - Staff availability violations
    - Qualification requirements
    - Labor law compliance (max hours, breaks)
    - Staffing ratio requirements
    - Priority weight fairness
    """,
)
async def validate_schedule(
    request: ScheduleValidationRequest,
    optimizer: ScheduleOptimizer = Depends(get_optimizer),
    profiler: PerformanceProfiler = Depends(get_profiler),
    _: bool = Depends(rate_limit),
) -> ScheduleValidationResponse:
    """Validate a schedule against business rules and constraints"""

    request_id = f"val_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger.info(
        f"[{request_id}] Schedule validation request for {len(request.schedule)} shifts"
    )

    try:
        profiler.start_timer(f"schedule_validation_{request_id}")
        profiler.increment_counter("validation_requests")

        response = await optimizer.validate_schedule(request)

        validation_time = profiler.end_timer(f"schedule_validation_{request_id}")

        if response.is_valid:
            profiler.increment_counter("valid_schedules")
            logger.info(
                f"[{request_id}] Schedule validation passed in {validation_time:.2f}s"
            )
        else:
            profiler.increment_counter("invalid_schedules")
            logger.info(
                f"[{request_id}] Schedule validation failed in {validation_time:.2f}s: "
                f"conflicts={len(response.conflicts)}"
            )

        return response

    except Exception as e:
        profiler.increment_counter("validation_errors")
        logger.error(f"[{request_id}] Validation error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Validation failed")