"""
Schedule generation and optimization API routes
"""

import asyncio
import logging
from datetime import datetime, date, timedelta
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query
from fastapi.responses import JSONResponse

from ...models import (
    ScheduleGenerationRequest,
    ScheduleGenerationResponse,
    ScheduleValidationRequest,
    ScheduleValidationResponse,
    ScheduledShift,
    OptimizationGoal,
    ScheduleConflict,
    EnhancedScheduleGenerationRequest,
    EnhancedScheduleResponse,
    CenterConfiguration,
    Staff,
    Group,
    StaffingRequirement,
    ScheduleConstraint,
    EnhancedOptimizationConfig,
)
from ...core.optimizer import ScheduleOptimizer
from ...utils.profiler import PerformanceProfiler
from ...utils.exceptions import OptimizationError, ValidationError
from ..dependencies import get_optimizer, get_profiler, rate_limit

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/schedule", tags=["Schedule Generation"])


@router.post(
    "/generate",
    response_model=EnhancedScheduleResponse,
    summary="Generate optimal schedule with flexible date range",
    description="""
    Generate an optimized schedule with flexible date range and advanced features.
    
    **New Features:**
    - Flexible date ranges (start_date to end_date) instead of just weekly
    - Center configuration with opening hours and staff-to-child ratios
    - Staff availability priority: Absences > Preferences > Center opening hours
    - Group assignments (primary/secondary with preference boost)
    - Overtime limits (daily and weekly, center-wide)
    - Enhanced constraint system with detailed validation
    
    **Priority Weight Effects:**
    - `seniority_weight > 1.0`: Staff gets more hours and better time slots
    - `performance_weight > 1.0`: Increased preference satisfaction priority
    - `flexibility_score > 1.0`: Higher consideration in fair distribution
    
    **Optimization Process:**
    1. Validate enhanced constraints and staff availability
    2. Process staff availability with priority system
    3. Generate daily schedules for the date range
    4. Apply overtime limits and ratio validations
    5. Combine results with enhanced analytics
    """,
    responses={
        200: {"description": "Schedule generated successfully"},
        400: {"description": "Invalid request data"},
        422: {"description": "Validation error"},
        500: {"description": "Optimization failed"},
        503: {"description": "Service temporarily unavailable"},
    },
)
async def generate_schedule(
    request: EnhancedScheduleGenerationRequest,
    background_tasks: BackgroundTasks,
    optimizer: ScheduleOptimizer = Depends(get_optimizer),
    profiler: PerformanceProfiler = Depends(get_profiler),
    _: bool = Depends(rate_limit),
) -> EnhancedScheduleResponse:
    """Generate an enhanced schedule with flexible date range support"""

    request_id = f"gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger.info(
        f"[{request_id}] Enhanced schedule generation request for center {request.center_id}"
    )

    try:
        # Start performance tracking
        profiler.start_timer(f"enhanced_generation_{request_id}")
        profiler.increment_counter("enhanced_generation_requests")

        # Log request details
        logger.info(
            f"[{request_id}] Request details: "
            f"staff={len(request.staff)}, groups={len(request.groups)}, "
            f"requirements={len(request.staffing_requirements)}, "
            f"period={request.schedule_start_date} to {request.effective_end_date} ({request.total_days} days)"
        )

        # Log priority weights for analysis
        _log_priority_weights(request_id, request.staff)

        # Validate request size limits
        if len(request.staff) > 100:
            raise HTTPException(
                status_code=400,
                detail="Too many staff members. Maximum 100 allowed per request.",
            )

        if len(request.groups) > 20:
            raise HTTPException(
                status_code=400,
                detail="Too many groups. Maximum 20 allowed per request.",
            )

        if request.total_days > 90:
            raise HTTPException(
                status_code=400,
                detail="Date range too large. Maximum 90 days allowed per request.",
            )

        # Generate the enhanced schedule
        response = await optimizer.generate_enhanced_schedule(request)

        # Track performance metrics
        generation_time = profiler.end_timer(f"enhanced_generation_{request_id}")

        # Log results
        if response.success:
            profiler.increment_counter("successful_enhanced_generations")
            logger.info(
                f"[{request_id}] Enhanced schedule generated successfully in {generation_time:.2f}s: "
                f"total_schedules={len(response.schedules)}, new_schedules={len(response.new_schedules)}, conflicts={len(response.conflicts)}, "
                f"status={response.optimization_result.status}"
            )

            # Log staff hour distribution for monitoring
            _log_hour_distribution(request_id, response.schedules, request.staff)
        else:
            profiler.increment_counter("failed_enhanced_generations")
            logger.warning(
                f"[{request_id}] Enhanced schedule generation failed: {response.message}, "
                f"total_schedules={len(response.schedules)}, new_schedules={len(response.new_schedules)}, conflicts={len(response.conflicts)}"
            )

        # Schedule background tasks
        background_tasks.add_task(
            _update_enhanced_analytics,
            request.center_id,
            response,
            generation_time,
            request_id,
        )

        return response

    except OptimizationError as e:
        profiler.increment_counter("enhanced_optimization_errors")
        logger.error(f"[{request_id}] Enhanced optimization error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")

    except ValidationError as e:
        profiler.increment_counter("enhanced_validation_errors")
        logger.error(f"[{request_id}] Enhanced validation error: {str(e)}")
        raise HTTPException(status_code=422, detail=f"Validation error: {str(e)}")

    except Exception as e:
        profiler.increment_counter("enhanced_generation_errors")
        logger.error(
            f"[{request_id}] Unexpected enhanced generation error: {str(e)}",
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Internal server error")


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


@router.post(
    "/optimize-existing",
    response_model=ScheduleGenerationResponse,
    summary="Optimize existing schedule",
    description="""
    Improve an existing schedule with minimal changes.
    
    This endpoint takes an existing schedule and applies optimization to improve:
    - Staff satisfaction (better preference matching)
    - Cost efficiency (reduce overtime, optimize staff mix)
    - Fairness (better hour distribution based on priority weights)
    - Continuity (reduce fragmented shifts)
    """,
)
async def optimize_existing_schedule(
    base_request: ScheduleGenerationRequest,
    current_schedule: List[ScheduledShift],
    optimization_goals: Optional[List[OptimizationGoal]] = None,
    preserve_confirmed: bool = Query(True, description="Preserve confirmed shifts"),
    max_changes: int = Query(
        10, ge=0, le=50, description="Maximum number of changes allowed"
    ),
    optimizer: ScheduleOptimizer = Depends(get_optimizer),
    profiler: PerformanceProfiler = Depends(get_profiler),
    _: bool = Depends(rate_limit),
) -> ScheduleGenerationResponse:
    """Optimize an existing schedule with constraints on changes"""

    request_id = f"opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger.info(
        f"[{request_id}] Schedule optimization request for {len(current_schedule)} shifts"
    )

    try:
        profiler.start_timer(f"schedule_optimization_{request_id}")
        profiler.increment_counter("optimization_requests")

        # Set optimization goals if provided
        if optimization_goals:
            base_request.optimization_config.goals = optimization_goals

        # Add existing schedule constraint
        base_request.existing_schedule = current_schedule

        # Add change limitation constraint
        if max_changes < len(current_schedule):
            change_constraint = {
                "constraint_type": "max_changes",
                "is_mandatory": True,
                "config": {
                    "max_changes": max_changes,
                    "preserve_confirmed": preserve_confirmed,
                },
            }
            base_request.constraints.append(change_constraint)

        response = await optimizer.optimize_existing_schedule(
            current_schedule, base_request, optimization_goals
        )

        optimization_time = profiler.end_timer(f"schedule_optimization_{request_id}")

        # Calculate change metrics
        changes_made = _calculate_schedule_changes(current_schedule, response.schedules)

        logger.info(
            f"[{request_id}] Schedule optimization completed in {optimization_time:.2f}s: "
            f"changes={changes_made}, status={response.optimization_result.status}"
        )

        return response

    except Exception as e:
        profiler.increment_counter("optimization_errors")
        logger.error(f"[{request_id}] Optimization error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Optimization failed")


@router.post(
    "/batch-generate",
    response_model=List[ScheduleGenerationResponse],
    summary="Generate schedules for multiple weeks",
    description="""
    Generate optimized schedules for multiple weeks in batch.
    
    **Benefits:**
    - Consistent staff allocation across weeks
    - Better long-term fairness based on priority weights
    - Reduced overall computation time
    - Cross-week constraint consideration
    
    **Limitations:**
    - Maximum 8 weeks per batch
    - Higher memory usage
    - Longer processing time
    """,
)
async def generate_batch_schedules(
    requests: List[ScheduleGenerationRequest],
    background_tasks: BackgroundTasks,
    ensure_fairness_across_weeks: bool = Query(
        True, description="Ensure fair distribution across all weeks"
    ),
    max_total_time: int = Query(
        300, ge=60, le=600, description="Maximum total processing time in seconds"
    ),
    optimizer: ScheduleOptimizer = Depends(get_optimizer),
    profiler: PerformanceProfiler = Depends(get_profiler),
    _: bool = Depends(rate_limit),
) -> List[ScheduleGenerationResponse]:
    """Generate schedules for multiple weeks in batch"""

    request_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if len(requests) > 8:
        raise HTTPException(
            status_code=400, detail="Maximum 8 weeks allowed per batch request"
        )

    if not requests:
        raise HTTPException(
            status_code=400, detail="At least one schedule request required"
        )

    logger.info(f"[{request_id}] Batch schedule generation for {len(requests)} weeks")

    try:
        profiler.start_timer(f"batch_generation_{request_id}")
        profiler.increment_counter("batch_requests")

        # Validate all requests have same center_id and staff
        center_id = requests[0].center_id
        if not all(req.center_id == center_id for req in requests):
            raise HTTPException(
                status_code=400, detail="All requests must be for the same center"
            )

        # Process requests with time budget per week
        time_per_week = max_total_time // len(requests)
        responses = []

        for i, request in enumerate(requests):
            week_start = datetime.now()
            request.optimization_config.max_solver_time = min(
                request.optimization_config.max_solver_time, time_per_week
            )

            logger.info(
                f"[{request_id}] Processing week {i + 1}/{len(requests)}: {request.week_start_date}"
            )

            response = await optimizer.generate_schedule(request)
            responses.append(response)

            week_time = (datetime.now() - week_start).total_seconds()
            logger.info(f"[{request_id}] Week {i + 1} completed in {week_time:.1f}s")

        batch_time = profiler.end_timer(f"batch_generation_{request_id}")
        successful_weeks = sum(1 for r in responses if r.success)

        logger.info(
            f"[{request_id}] Batch generation completed in {batch_time:.2f}s: "
            f"successful={successful_weeks}/{len(requests)}"
        )

        # Schedule analytics update
        background_tasks.add_task(
            _update_batch_analytics, center_id, responses, batch_time, request_id
        )

        return responses

    except Exception as e:
        profiler.increment_counter("batch_errors")
        logger.error(f"[{request_id}] Batch generation error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Batch generation failed")


@router.get(
    "/status/{request_id}",
    response_model=dict,
    summary="Get request status",
    description="Get the status of a long-running schedule generation request",
)
async def get_request_status(request_id: str, _: bool = Depends(rate_limit)) -> dict:
    """Get status of a schedule generation request"""

    # This would typically query a job queue or database
    # For now, return a simple response
    return {
        "request_id": request_id,
        "status": "completed",  # or "running", "failed", "queued"
        "progress": 100,
        "message": "Schedule generation completed",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    }


# Helper functions
def _log_priority_weights(request_id: str, staff: List) -> None:
    """Log priority weight distribution for analysis"""
    try:
        weights = [
            s.seniority_weight * s.performance_weight * s.flexibility_score
            for s in staff
        ]
        if weights:
            avg_weight = sum(weights) / len(weights)
            max_weight = max(weights)
            min_weight = min(weights)

            logger.info(
                f"[{request_id}] Priority weights - "
                f"avg: {avg_weight:.2f}, min: {min_weight:.2f}, max: {max_weight:.2f}, "
                f"spread: {max_weight - min_weight:.2f}"
            )
    except Exception as e:
        logger.warning(f"[{request_id}] Failed to log priority weights: {e}")


def _log_hour_distribution(
    request_id: str, schedule: List[ScheduledShift], staff: List
) -> None:
    """Log hour distribution for monitoring fairness"""
    try:
        if not schedule:
            return

        # Calculate hours per staff
        staff_hours = {}
        for shift in schedule:
            staff_hours[shift.staff_id] = (
                staff_hours.get(shift.staff_id, 0) + shift.scheduled_hours
            )

        if staff_hours:
            hours = list(staff_hours.values())
            avg_hours = sum(hours) / len(hours)
            max_hours = max(hours)
            min_hours = min(hours)

            logger.info(
                f"[{request_id}] Hour distribution - "
                f"avg: {avg_hours:.1f}h, min: {min_hours:.1f}h, max: {max_hours:.1f}h, "
                f"variance: {max_hours - min_hours:.1f}h"
            )

    except Exception as e:
        logger.warning(f"[{request_id}] Failed to log hour distribution: {e}")


def _calculate_schedule_changes(
    old_schedule: List[ScheduledShift], new_schedule: List[ScheduledShift]
) -> int:
    """Calculate number of changes between two schedules"""
    try:
        old_shifts = {
            (s.staff_id, s.group_id, s.date, s.start_time): s for s in old_schedule
        }
        new_shifts = {
            (s.staff_id, s.group_id, s.date, s.start_time): s for s in new_schedule
        }

        changes = 0

        # Count removed shifts
        changes += len(old_shifts.keys() - new_shifts.keys())

        # Count added shifts
        changes += len(new_shifts.keys() - old_shifts.keys())

        # Count modified shifts
        for key in old_shifts.keys() & new_shifts.keys():
            old_shift = old_shifts[key]
            new_shift = new_shifts[key]
            if (
                old_shift.end_time != new_shift.end_time
                or old_shift.scheduled_hours != new_shift.scheduled_hours
            ):
                changes += 1

        return changes

    except Exception:
        return 0


async def _update_analytics(
    center_id: UUID,
    response: ScheduleGenerationResponse,
    generation_time: float,
    request_id: str,
) -> None:
    """Update analytics in background"""
    try:
        # This would typically update a database or send to analytics service
        analytics_data = {
            "request_id": request_id,
            "center_id": str(center_id),
            "timestamp": datetime.now().isoformat(),
            "success": response.success,
            "shift_count": len(response.schedules),
            "conflict_count": len(response.conflicts),
            "total_hours": response.total_hours,
            "total_cost": response.total_cost,
            "solve_time": response.optimization_result.solve_time_seconds,
            "generation_time": generation_time,
            "status": response.optimization_result.status,
            "satisfaction_score": response.satisfaction_score,
        }

        logger.debug(f"[{request_id}] Analytics updated: {analytics_data}")

    except Exception as e:
        logger.error(f"[{request_id}] Failed to update analytics: {e}")


async def _update_batch_analytics(
    center_id: UUID,
    responses: List[ScheduleGenerationResponse],
    batch_time: float,
    request_id: str,
) -> None:
    """Update batch analytics in background"""
    try:
        successful_count = sum(1 for r in responses if r.success)
        total_shifts = sum(len(r.schedule) for r in responses)
        total_conflicts = sum(len(r.conflicts) for r in responses)

        batch_analytics = {
            "request_id": request_id,
            "center_id": str(center_id),
            "timestamp": datetime.now().isoformat(),
            "weeks_processed": len(responses),
            "successful_weeks": successful_count,
            "total_shifts": total_shifts,
            "total_conflicts": total_conflicts,
            "batch_time": batch_time,
            "avg_time_per_week": batch_time / len(responses) if responses else 0,
        }

        logger.info(f"[{request_id}] Batch analytics: {batch_analytics}")

    except Exception as e:
        logger.error(f"[{request_id}] Failed to update batch analytics: {e}")


async def _update_enhanced_analytics(
    center_id: UUID,
    response: EnhancedScheduleResponse,
    generation_time: float,
    request_id: str,
) -> None:
    """Update enhanced analytics in background"""
    try:
        analytics_data = {
            "request_id": request_id,
            "center_id": str(center_id),
            "timestamp": datetime.now().isoformat(),
            "success": response.success,
            "shift_count": len(response.schedules),
            "conflict_count": len(response.conflicts),
            "total_hours": response.total_hours,
            "total_cost": response.total_cost,
            "solve_time": response.optimization_result.solve_time_seconds,
            "generation_time": generation_time,
            "status": response.optimization_result.status,
            "satisfaction_score": response.satisfaction_score,
            "period_days": response.total_days,
            "start_date": response.schedule_start_date.isoformat()
            if response.schedule_start_date
            else None,
            "end_date": response.schedule_end_date.isoformat()
            if response.schedule_end_date
            else None,
        }

        logger.debug(f"[{request_id}] Enhanced analytics updated: {analytics_data}")

    except Exception as e:
        logger.error(f"[{request_id}] Failed to update enhanced analytics: {e}")


@router.get(
    "/preview-complexity",
    summary="Preview schedule complexity",
    description="Preview the complexity and estimated solve time for a date range",
)
async def preview_schedule_complexity(
    center_id: UUID,
    schedule_start_date: date,
    schedule_end_date: Optional[date] = None,
    staff_count: int = Query(..., description="Number of staff members"),
    groups_count: int = Query(..., description="Number of groups"),
    requirements_count: int = Query(..., description="Number of staffing requirements"),
    _: bool = Depends(rate_limit),
):
    """Preview the complexity and estimated solve time for a date range"""

    if schedule_end_date is None:
        schedule_end_date = schedule_start_date + timedelta(days=6)

    total_days = (schedule_end_date - schedule_start_date).days + 1

    # Estimate complexity
    hours_per_day = 14  # Operating hours
    total_time_slots = total_days * hours_per_day
    estimated_variables = staff_count * groups_count * total_time_slots

    # Estimate solve time based on problem size
    if estimated_variables < 10000:
        estimated_solve_time = "30 seconds - 2 minutes"
        complexity = "Low"
    elif estimated_variables < 50000:
        estimated_solve_time = "2 - 10 minutes"
        complexity = "Medium"
    elif estimated_variables < 200000:
        estimated_solve_time = "10 - 30 minutes"
        complexity = "High"
    else:
        estimated_solve_time = "30+ minutes"
        complexity = "Very High"

    # Suggest optimization strategy
    if total_days > 21:
        suggested_strategy = "chunked"
        strategy_description = "Large date range will be split into smaller chunks"
    elif groups_count > 5 and staff_count > 10:
        suggested_strategy = "parallel"
        strategy_description = "Independent groups will be optimized in parallel"
    else:
        suggested_strategy = "standard"
        strategy_description = "Standard optimization approach"

    return {
        "center_id": center_id,
        "period": {
            "start_date": schedule_start_date,
            "end_date": schedule_end_date,
            "total_days": total_days,
        },
        "complexity": {
            "level": complexity,
            "estimated_variables": estimated_variables,
            "estimated_solve_time": estimated_solve_time,
        },
        "strategy": {
            "recommended": suggested_strategy,
            "description": strategy_description,
        },
        "recommendations": [
            "Consider chunking if date range exceeds 21 days",
            "Ensure staff availability data is accurate",
            "Review staffing requirements for feasibility",
            "Use existing schedule data for better continuity",
        ],
    }
