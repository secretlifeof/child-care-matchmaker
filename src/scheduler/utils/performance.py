"""
Performance monitoring and analytics utility
File: scheduler/utils/performance.py
"""
import json
import logging
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any

import psutil

from ..models import OptimizationResult

logger = logging.getLogger(__name__)


@dataclass
class PerformanceSnapshot:
    """Single performance measurement snapshot"""

    timestamp: datetime
    operation: str
    duration_seconds: float
    memory_usage_mb: float
    cpu_percent: float

    # Problem characteristics
    total_days: int
    staff_count: int
    groups_count: int
    requirements_count: int

    # Results
    success: bool
    optimization_status: str
    conflicts_count: int
    solution_quality: float

    # Resource usage
    peak_memory_mb: float
    variables_count: int | None = None
    constraints_count: int | None = None


@dataclass
class PerformanceStats:
    """Aggregated performance statistics"""

    operation: str
    total_operations: int
    success_rate: float

    # Timing statistics
    avg_duration: float
    min_duration: float
    max_duration: float
    p95_duration: float

    # Resource statistics
    avg_memory_usage: float
    peak_memory_usage: float
    avg_cpu_usage: float

    # Problem size statistics
    avg_problem_size: float
    largest_problem_size: float

    # Quality statistics
    avg_solution_quality: float
    avg_conflicts: float


class PerformanceMonitor:
    """Monitor and track performance of schedule optimization operations"""

    def __init__(self, max_snapshots: int = 1000):
        self.max_snapshots = max_snapshots
        self.snapshots: deque = deque(maxlen=max_snapshots)
        self.operation_stats: dict[str, list[PerformanceSnapshot]] = defaultdict(list)
        self.lock = threading.Lock()

        # Active monitoring
        self.active_operations: dict[str, dict[str, Any]] = {}

    @contextmanager
    def monitor_operation(
        self,
        operation_name: str,
        total_days: int = 0,
        staff_count: int = 0,
        groups_count: int = 0,
        requirements_count: int = 0
    ):
        """Context manager to monitor a complete operation"""

        operation_id = f"{operation_name}_{int(time.time() * 1000)}"
        start_time = time.time()
        start_memory = self._get_memory_usage()
        start_cpu = psutil.cpu_percent()

        # Track operation start
        self.active_operations[operation_id] = {
            'operation': operation_name,
            'start_time': start_time,
            'start_memory': start_memory,
            'peak_memory': start_memory,
            'total_days': total_days,
            'staff_count': staff_count,
            'groups_count': groups_count,
            'requirements_count': requirements_count
        }

        try:
            yield operation_id
            success = True
            optimization_status = 'COMPLETED'
        except Exception as e:
            success = False
            optimization_status = f'ERROR: {type(e).__name__}'
            logger.error(f"Operation {operation_name} failed: {str(e)}")
            raise
        finally:
            # Calculate final metrics
            end_time = time.time()
            duration = end_time - start_time
            end_memory = self._get_memory_usage()
            end_cpu = psutil.cpu_percent()

            # Create snapshot
            snapshot = PerformanceSnapshot(
                timestamp=datetime.now(),
                operation=operation_name,
                duration_seconds=duration,
                memory_usage_mb=end_memory,
                cpu_percent=(start_cpu + end_cpu) / 2,
                total_days=total_days,
                staff_count=staff_count,
                groups_count=groups_count,
                requirements_count=requirements_count,
                success=success,
                optimization_status=optimization_status,
                conflicts_count=0,  # Will be updated if provided
                solution_quality=0.0,  # Will be updated if provided
                peak_memory_mb=self.active_operations[operation_id]['peak_memory']
            )

            # Store snapshot
            self._record_snapshot(snapshot)

            # Clean up
            del self.active_operations[operation_id]

    def update_operation_results(
        self,
        operation_id: str,
        optimization_result: OptimizationResult,
        conflicts_count: int = 0,
        solution_quality: float = 0.0,
        variables_count: int | None = None,
        constraints_count: int | None = None
    ):
        """Update operation with final results"""

        if operation_id in self.active_operations:
            op_data = self.active_operations[operation_id]

            # Update the most recent snapshot for this operation
            with self.lock:
                for snapshot in reversed(self.snapshots):
                    if (snapshot.operation == op_data['operation'] and
                        abs((snapshot.timestamp - datetime.now()).total_seconds()) < 60):

                        snapshot.optimization_status = optimization_result.status
                        snapshot.conflicts_count = conflicts_count
                        snapshot.solution_quality = solution_quality
                        snapshot.variables_count = variables_count
                        snapshot.constraints_count = constraints_count
                        break

    def record_memory_peak(self, operation_id: str):
        """Record a memory peak for an active operation"""

        if operation_id in self.active_operations:
            current_memory = self._get_memory_usage()
            if current_memory > self.active_operations[operation_id]['peak_memory']:
                self.active_operations[operation_id]['peak_memory'] = current_memory

    def _record_snapshot(self, snapshot: PerformanceSnapshot):
        """Record a performance snapshot"""

        with self.lock:
            self.snapshots.append(snapshot)
            self.operation_stats[snapshot.operation].append(snapshot)

            # Trim operation-specific stats if needed
            if len(self.operation_stats[snapshot.operation]) > self.max_snapshots:
                self.operation_stats[snapshot.operation].pop(0)

    def get_operation_stats(self, operation: str, hours: int = 24) -> PerformanceStats | None:
        """Get performance statistics for a specific operation"""

        cutoff_time = datetime.now() - timedelta(hours=hours)

        with self.lock:
            recent_snapshots = [
                s for s in self.operation_stats[operation]
                if s.timestamp >= cutoff_time
            ]

        if not recent_snapshots:
            return None

        # Calculate statistics
        durations = [s.duration_seconds for s in recent_snapshots]
        memory_usages = [s.memory_usage_mb for s in recent_snapshots]
        cpu_usages = [s.cpu_percent for s in recent_snapshots]
        problem_sizes = [s.total_days * s.staff_count * s.groups_count for s in recent_snapshots]
        solution_qualities = [s.solution_quality for s in recent_snapshots if s.success]
        conflicts = [s.conflicts_count for s in recent_snapshots]

        success_count = sum(1 for s in recent_snapshots if s.success)

        return PerformanceStats(
            operation=operation,
            total_operations=len(recent_snapshots),
            success_rate=success_count / len(recent_snapshots),

            avg_duration=sum(durations) / len(durations),
            min_duration=min(durations),
            max_duration=max(durations),
            p95_duration=self._percentile(durations, 95),

            avg_memory_usage=sum(memory_usages) / len(memory_usages),
            peak_memory_usage=max(s.peak_memory_mb for s in recent_snapshots),
            avg_cpu_usage=sum(cpu_usages) / len(cpu_usages),

            avg_problem_size=sum(problem_sizes) / len(problem_sizes) if problem_sizes else 0,
            largest_problem_size=max(problem_sizes) if problem_sizes else 0,

            avg_solution_quality=sum(solution_qualities) / len(solution_qualities) if solution_qualities else 0,
            avg_conflicts=sum(conflicts) / len(conflicts) if conflicts else 0
        )

    def get_system_health(self) -> dict[str, Any]:
        """Get current system health metrics"""

        # Recent performance (last hour)
        recent_operations = []
        cutoff_time = datetime.now() - timedelta(hours=1)

        with self.lock:
            recent_operations = [s for s in self.snapshots if s.timestamp >= cutoff_time]

        if not recent_operations:
            return {
                "status": "idle",
                "message": "No recent operations",
                "system": self._get_system_info()
            }

        # Calculate health metrics
        success_rate = sum(1 for op in recent_operations if op.success) / len(recent_operations)
        avg_duration = sum(op.duration_seconds for op in recent_operations) / len(recent_operations)
        peak_memory = max(op.peak_memory_mb for op in recent_operations)

        # Determine health status
        if success_rate < 0.8:
            status = "degraded"
            message = f"Low success rate: {success_rate:.1%}"
        elif avg_duration > 300:  # 5 minutes
            status = "slow"
            message = f"High average duration: {avg_duration:.1f}s"
        elif peak_memory > 8000:  # 8GB
            status = "high_memory"
            message = f"High memory usage: {peak_memory:.1f}MB"
        else:
            status = "healthy"
            message = "System operating normally"

        return {
            "status": status,
            "message": message,
            "metrics": {
                "recent_operations": len(recent_operations),
                "success_rate": success_rate,
                "avg_duration_seconds": avg_duration,
                "peak_memory_mb": peak_memory
            },
            "system": self._get_system_info()
        }

    def get_performance_recommendations(self) -> list[dict[str, str]]:
        """Get performance improvement recommendations"""

        recommendations = []

        # Analyze recent performance
        recent_stats = {}
        for operation in self.operation_stats.keys():
            stats = self.get_operation_stats(operation, hours=24)
            if stats:
                recent_stats[operation] = stats

        for operation, stats in recent_stats.items():
            # Check for slow operations
            if stats.avg_duration > 180:  # 3 minutes
                recommendations.append({
                    "type": "performance",
                    "operation": operation,
                    "issue": "Slow operation",
                    "recommendation": f"Consider chunking for {operation} (avg: {stats.avg_duration:.1f}s)",
                    "priority": "medium"
                })

            # Check for high memory usage
            if stats.peak_memory_usage > 4000:  # 4GB
                recommendations.append({
                    "type": "memory",
                    "operation": operation,
                    "issue": "High memory usage",
                    "recommendation": f"Optimize memory usage for {operation} (peak: {stats.peak_memory_usage:.1f}MB)",
                    "priority": "high"
                })

            # Check for low success rate
            if stats.success_rate < 0.9:
                recommendations.append({
                    "type": "reliability",
                    "operation": operation,
                    "issue": "Low success rate",
                    "recommendation": f"Investigate failures in {operation} ({stats.success_rate:.1%} success rate)",
                    "priority": "high"
                })

            # Check for large problem sizes
            if stats.largest_problem_size > 500000:
                recommendations.append({
                    "type": "scalability",
                    "operation": operation,
                    "issue": "Large problem size",
                    "recommendation": f"Use parallel or chunked optimization for {operation}",
                    "priority": "medium"
                })

        return recommendations

    def export_performance_data(
        self,
        operation: str | None = None,
        hours: int = 24,
        format: str = "json"
    ) -> str:
        """Export performance data for analysis"""

        cutoff_time = datetime.now() - timedelta(hours=hours)

        with self.lock:
            if operation:
                snapshots = [
                    s for s in self.operation_stats[operation]
                    if s.timestamp >= cutoff_time
                ]
            else:
                snapshots = [s for s in self.snapshots if s.timestamp >= cutoff_time]

        if format == "json":
            data = {
                "export_time": datetime.now().isoformat(),
                "period_hours": hours,
                "operation_filter": operation,
                "snapshots": [asdict(s) for s in snapshots]
            }
            return json.dumps(data, indent=2, default=str)

        elif format == "csv":
            # Convert to CSV format
            if not snapshots:
                return "No data available"

            headers = [
                "timestamp", "operation", "duration_seconds", "memory_usage_mb",
                "cpu_percent", "total_days", "staff_count", "groups_count",
                "success", "optimization_status", "conflicts_count", "solution_quality"
            ]

            lines = [",".join(headers)]

            for snapshot in snapshots:
                row = [
                    snapshot.timestamp.isoformat(),
                    snapshot.operation,
                    str(snapshot.duration_seconds),
                    str(snapshot.memory_usage_mb),
                    str(snapshot.cpu_percent),
                    str(snapshot.total_days),
                    str(snapshot.staff_count),
                    str(snapshot.groups_count),
                    str(snapshot.success),
                    snapshot.optimization_status,
                    str(snapshot.conflicts_count),
                    str(snapshot.solution_quality)
                ]
                lines.append(",".join(row))

            return "\n".join(lines)

        else:
            raise ValueError(f"Unsupported format: {format}")

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""

        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

    def _get_system_info(self) -> dict[str, Any]:
        """Get current system information"""

        memory = psutil.virtual_memory()
        cpu_count = psutil.cpu_count()

        return {
            "cpu_count": cpu_count,
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_total_gb": memory.total / 1024 / 1024 / 1024,
            "memory_available_gb": memory.available / 1024 / 1024 / 1024,
            "memory_percent": memory.percent,
            "active_operations": len(self.active_operations)
        }

    @staticmethod
    def _percentile(data: list[float], percentile: int) -> float:
        """Calculate percentile of a list of numbers"""

        if not data:
            return 0.0

        sorted_data = sorted(data)
        index = (len(sorted_data) - 1) * percentile / 100

        if index == int(index):
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


# Decorator for automatic performance monitoring
def monitor_performance(
    operation_name: str | None = None,
    track_memory_peaks: bool = True
):
    """Decorator to automatically monitor function performance"""

    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"

            # Extract problem size info if available
            total_days = kwargs.get('total_days', 0)
            staff_count = len(kwargs.get('staff', []))
            groups_count = len(kwargs.get('groups', []))
            requirements_count = len(kwargs.get('requirements', []))

            with performance_monitor.monitor_operation(
                op_name, total_days, staff_count, groups_count, requirements_count
            ) as operation_id:

                if track_memory_peaks:
                    # Record memory usage periodically during execution
                    import threading

                    def track_memory():
                        while operation_id in performance_monitor.active_operations:
                            performance_monitor.record_memory_peak(operation_id)
                            time.sleep(1)

                    memory_thread = threading.Thread(target=track_memory, daemon=True)
                    memory_thread.start()

                result = func(*args, **kwargs)

                # Update with results if available
                if hasattr(result, 'optimization_result'):
                    performance_monitor.update_operation_results(
                        operation_id,
                        result.optimization_result,
                        len(getattr(result, 'conflicts', [])),
                        getattr(result, 'satisfaction_score', 0.0)
                    )

                return result

        return wrapper
    return decorator


# Usage examples:

"""
# Using context manager
with performance_monitor.monitor_operation(
    'generate_monthly_schedule', 
    total_days=31, 
    staff_count=15, 
    groups_count=5
) as operation_id:
    # Your optimization code here
    result = optimizer.generate_schedule(...)
    
    # Update with results
    performance_monitor.update_operation_results(
        operation_id,
        result.optimization_result,
        len(result.conflicts),
        result.satisfaction_score
    )

# Using decorator
@monitor_performance("weekly_schedule_generation")
def generate_weekly_schedule(staff, groups, requirements):
    # Your optimization code
    return result

# Getting performance stats
stats = performance_monitor.get_operation_stats('generate_monthly_schedule', hours=24)
print(f"Average duration: {stats.avg_duration:.2f}s")
print(f"Success rate: {stats.success_rate:.1%}")

# System health check
health = performance_monitor.get_system_health()
print(f"System status: {health['status']}")

# Get recommendations
recommendations = performance_monitor.get_performance_recommendations()
for rec in recommendations:
    print(f"{rec['priority'].upper()}: {rec['recommendation']}")

# Export data for analysis
data = performance_monitor.export_performance_data(format="csv", hours=48)
with open("performance_data.csv", "w") as f:
    f.write(data)
"""
