"""
Performance profiler for optimization debugging and monitoring
"""
import time
import logging
import threading
from datetime import datetime
from typing import Dict, Any, Optional, List
from collections import defaultdict, deque
import statistics
import psutil
import os

logger = logging.getLogger(__name__)


class PerformanceProfiler:
    """Performance profiler for tracking optimization metrics"""
    
    def __init__(self):
        self.timers = {}
        self.counters = defaultdict(int)
        self.measurements = defaultdict(list)
        self.system_metrics = {}
        self.lock = threading.Lock()
        self.start_time = time.time()
        
        # Initialize process handle for system monitoring
        try:
            self.process = psutil.Process(os.getpid())
        except Exception:
            self.process = None
        
        logger.info("Performance profiler initialized")
    
    def start_timer(self, name: str):
        """Start a named timer"""
        with self.lock:
            self.timers[name] = {
                'start_time': time.time(),
                'start_cpu': time.process_time(),
                'start_memory': self._get_memory_usage()
            }
    
    def end_timer(self, name: str) -> float:
        """End a named timer and return duration in seconds"""
        end_time = time.time()
        end_cpu = time.process_time()
        end_memory = self._get_memory_usage()
        
        with self.lock:
            if name not in self.timers:
                logger.warning(f"Timer {name} was not started")
                return 0.0
            
            timer_data = self.timers[name]
            duration = end_time - timer_data['start_time']
            cpu_time = end_cpu - timer_data['start_cpu']
            memory_delta = end_memory - timer_data['start_memory']
            
            # Store measurement
            self.measurements[name].append({
                'duration': duration,
                'cpu_time': cpu_time,
                'memory_delta': memory_delta,
                'timestamp': datetime.now()
            })
            
            # Keep only recent measurements (last 100)
            if len(self.measurements[name]) > 100:
                self.measurements[name] = self.measurements[name][-100:]
            
            del self.timers[name]
            return duration
    
    def increment_counter(self, name: str, value: int = 1):
        """Increment a named counter"""
        with self.lock:
            self.counters[name] += value
    
    def set_gauge(self, name: str, value: float):
        """Set a gauge value (current state measurement)"""
        with self.lock:
            self.system_metrics[name] = {
                'value': value,
                'timestamp': datetime.now()
            }
    
    def record_measurement(self, name: str, value: float, metadata: Optional[Dict] = None):
        """Record a custom measurement"""
        with self.lock:
            measurement = {
                'value': value,
                'timestamp': datetime.now()
            }
            if metadata:
                measurement['metadata'] = metadata
            
            self.measurements[name].append(measurement)
            
            # Keep only recent measurements
            if len(self.measurements[name]) > 100:
                self.measurements[name] = self.measurements[name][-100:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        with self.lock:
            stats = {
                'uptime_seconds': time.time() - self.start_time,
                'active_timers': list(self.timers.keys()),
                'counters': dict(self.counters),
                'timer_stats': self._calculate_timer_stats(),
                'system_metrics': dict(self.system_metrics),
                'current_system_state': self._get_current_system_state()
            }
            return stats
    
    def get_timer_summary(self, name: str) -> Optional[Dict[str, Any]]:
        """Get summary statistics for a specific timer"""
        with self.lock:
            if name not in self.measurements:
                return None
            
            measurements = self.measurements[name]
            if not measurements:
                return None
            
            durations = [m['duration'] for m in measurements]
            cpu_times = [m['cpu_time'] for m in measurements]
            memory_deltas = [m['memory_delta'] for m in measurements]
            
            return {
                'count': len(measurements),
                'duration': {
                    'mean': statistics.mean(durations),
                    'median': statistics.median(durations),
                    'min': min(durations),
                    'max': max(durations),
                    'stdev': statistics.stdev(durations) if len(durations) > 1 else 0
                },
                'cpu_time': {
                    'mean': statistics.mean(cpu_times),
                    'total': sum(cpu_times)
                },
                'memory_delta': {
                    'mean': statistics.mean(memory_deltas),
                    'max': max(memory_deltas),
                    'min': min(memory_deltas)
                },
                'last_execution': measurements[-1]['timestamp'].isoformat()
            }
    
    def get_performance_report(self) -> str:
        """Generate a human-readable performance report"""
        stats = self.get_stats()
        
        report = []
        report.append("=== Performance Report ===")
        report.append(f"Uptime: {stats['uptime_seconds']:.1f} seconds")
        report.append("")
        
        # Counter summary
        if stats['counters']:
            report.append("Counters:")
            for name, value in sorted(stats['counters'].items()):
                report.append(f"  {name}: {value}")
            report.append("")
        
        # Timer summary
        if stats['timer_stats']:
            report.append("Timer Statistics:")
            for name, timer_stats in sorted(stats['timer_stats'].items()):
                if timer_stats:
                    duration_stats = timer_stats['duration']
                    report.append(f"  {name}:")
                    report.append(f"    Count: {timer_stats['count']}")
                    report.append(f"    Avg: {duration_stats['mean']:.3f}s")
                    report.append(f"    Range: {duration_stats['min']:.3f}s - {duration_stats['max']:.3f}s")
                    if duration_stats['stdev'] > 0:
                        report.append(f"    StdDev: {duration_stats['stdev']:.3f}s")
            report.append("")
        
        # System state
        system_state = stats['current_system_state']
        if system_state:
            report.append("Current System State:")
            report.append(f"  CPU Usage: {system_state.get('cpu_percent', 'N/A')}%")
            report.append(f"  Memory Usage: {system_state.get('memory_percent', 'N/A')}%")
            report.append(f"  Memory RSS: {system_state.get('memory_rss_mb', 'N/A')} MB")
        
        return "\n".join(report)
    
    def reset(self):
        """Reset all performance data"""
        with self.lock:
            self.timers.clear()
            self.counters.clear()
            self.measurements.clear()
            self.system_metrics.clear()
            self.start_time = time.time()
            logger.info("Performance profiler reset")
    
    def export_metrics(self, format: str = 'json') -> str:
        """Export metrics in specified format"""
        stats = self.get_stats()
        
        if format.lower() == 'json':
            import json
            return json.dumps(stats, indent=2, default=str)
        
        elif format.lower() == 'prometheus':
            return self._export_prometheus_format(stats)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _calculate_timer_stats(self) -> Dict[str, Dict[str, Any]]:
        """Calculate statistics for all timers"""
        timer_stats = {}
        
        for name, measurements in self.measurements.items():
            if not measurements:
                continue
            
            # Filter to only timer measurements (those with duration)
            timer_measurements = [m for m in measurements if 'duration' in m]
            if not timer_measurements:
                continue
            
            timer_stats[name] = self.get_timer_summary(name)
        
        return timer_stats
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            if self.process:
                return self.process.memory_info().rss / 1024 / 1024  # MB
        except Exception:
            pass
        return 0.0
    
    def _get_current_system_state(self) -> Dict[str, Any]:
        """Get current system resource usage"""
        try:
            state = {}
            
            if self.process:
                # Process-specific metrics
                memory_info = self.process.memory_info()
                state['memory_rss_mb'] = memory_info.rss / 1024 / 1024
                state['memory_vms_mb'] = memory_info.vms / 1024 / 1024
                state['cpu_percent'] = self.process.cpu_percent()
                
                # System-wide metrics
                state['system_memory_percent'] = psutil.virtual_memory().percent
                state['system_cpu_percent'] = psutil.cpu_percent()
                
            return state
            
        except Exception as e:
            logger.warning(f"Failed to get system state: {e}")
            return {}
    
    def _export_prometheus_format(self, stats: Dict[str, Any]) -> str:
        """Export metrics in Prometheus format"""
        lines = []
        
        # Counters
        for name, value in stats.get('counters', {}).items():
            prometheus_name = name.replace('-', '_').replace(' ', '_')
            lines.append(f"# TYPE scheduler_{prometheus_name}_total counter")
            lines.append(f"scheduler_{prometheus_name}_total {value}")
        
        # Timer statistics
        for name, timer_stats in stats.get('timer_stats', {}).items():
            if not timer_stats:
                continue
            
            prometheus_name = name.replace('-', '_').replace(' ', '_')
            duration_stats = timer_stats['duration']
            
            lines.append(f"# TYPE scheduler_{prometheus_name}_duration_seconds histogram")
            lines.append(f"scheduler_{prometheus_name}_duration_seconds_count {timer_stats['count']}")
            lines.append(f"scheduler_{prometheus_name}_duration_seconds_sum {duration_stats['mean'] * timer_stats['count']}")
            
            # Add quantiles
            lines.append(f"scheduler_{prometheus_name}_duration_seconds{{quantile=\"0.5\"}} {duration_stats['median']}")
            lines.append(f"scheduler_{prometheus_name}_duration_seconds{{quantile=\"0.95\"}} {duration_stats['max']}")
        
        # System metrics
        system_state = stats.get('current_system_state', {})
        for metric_name, value in system_state.items():
            if isinstance(value, (int, float)):
                prometheus_name = metric_name.replace('-', '_').replace(' ', '_')
                lines.append(f"# TYPE scheduler_system_{prometheus_name} gauge")
                lines.append(f"scheduler_system_{prometheus_name} {value}")
        
        return '\n'.join(lines)


class OptimizationProfiler(PerformanceProfiler):
    """Specialized profiler for optimization operations"""
    
    def __init__(self):
        super().__init__()
        self.optimization_metrics = defaultdict(list)
    
    def profile_optimization_start(
        self,
        request_id: str,
        staff_count: int,
        group_count: int,
        requirements_count: int
    ):
        """Profile the start of an optimization"""
        self.start_timer(f"optimization_{request_id}")
        self.record_measurement('optimization_problem_size', staff_count * group_count)
        self.record_measurement('optimization_staff_count', staff_count)
        self.record_measurement('optimization_group_count', group_count)
        self.record_measurement('optimization_requirements_count', requirements_count)
        
        logger.debug(f"Started profiling optimization {request_id}")
    
    def profile_optimization_end(
        self,
        request_id: str,
        status: str,
        objective_value: float,
        iterations: int,
        conflicts_count: int
    ):
        """Profile the end of an optimization"""
        duration = self.end_timer(f"optimization_{request_id}")
        
        self.record_measurement('optimization_duration', duration)
        self.record_measurement('optimization_objective_value', objective_value)
        self.record_measurement('optimization_iterations', iterations)
        self.record_measurement('optimization_conflicts', conflicts_count)
        
        self.increment_counter(f'optimization_status_{status.lower()}')
        
        # Store detailed optimization metrics
        optimization_data = {
            'request_id': request_id,
            'duration': duration,
            'status': status,
            'objective_value': objective_value,
            'iterations': iterations,
            'conflicts_count': conflicts_count,
            'timestamp': datetime.now()
        }
        
        self.optimization_metrics['completed_optimizations'].append(optimization_data)
        
        # Keep only recent optimizations
        if len(self.optimization_metrics['completed_optimizations']) > 50:
            self.optimization_metrics['completed_optimizations'] = \
                self.optimization_metrics['completed_optimizations'][-50:]
        
        logger.debug(f"Completed profiling optimization {request_id} in {duration:.2f}s")
    
    def profile_constraint_building(self, constraint_type: str, count: int):
        """Profile constraint building phase"""
        self.record_measurement(f'constraints_{constraint_type}_count', count)
        self.increment_counter('constraints_built', count)
    
    def profile_solver_progress(self, iteration: int, objective_value: float, time_elapsed: float):
        """Profile solver progress (called periodically during solving)"""
        self.record_measurement('solver_iteration', iteration)
        self.record_measurement('solver_objective_value', objective_value)
        self.record_measurement('solver_time_elapsed', time_elapsed)
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization performance"""
        completed = self.optimization_metrics['completed_optimizations']
        
        if not completed:
            return {'message': 'No completed optimizations'}
        
        durations = [opt['duration'] for opt in completed]
        objective_values = [opt['objective_value'] for opt in completed if opt['objective_value'] > 0]
        iterations = [opt['iterations'] for opt in completed]
        
        # Count by status
        status_counts = defaultdict(int)
        for opt in completed:
            status_counts[opt['status']] += 1
        
        summary = {
            'total_optimizations': len(completed),
            'success_rate': status_counts.get('OPTIMAL', 0) / len(completed) if completed else 0,
            'duration_stats': {
                'mean': statistics.mean(durations),
                'median': statistics.median(durations),
                'min': min(durations),
                'max': max(durations)
            } if durations else {},
            'iterations_stats': {
                'mean': statistics.mean(iterations),
                'max': max(iterations)
            } if iterations else {},
            'status_distribution': dict(status_counts),
            'recent_optimizations': completed[-5:]  # Last 5 optimizations
        }
        
        if objective_values:
            summary['objective_value_stats'] = {
                'mean': statistics.mean(objective_values),
                'min': min(objective_values),
                'max': max(objective_values)
            }
        
        return summary


class RequestProfiler:
    """Profiler for HTTP request performance"""
    
    def __init__(self):
        self.request_timers = {}
        self.request_stats = defaultdict(list)
    
    def start_request(self, request_id: str, method: str, path: str):
        """Start profiling an HTTP request"""
        self.request_timers[request_id] = {
            'start_time': time.time(),
            'method': method,
            'path': path
        }
    
    def end_request(self, request_id: str, status_code: int, response_size: int = 0):
        """End profiling an HTTP request"""
        if request_id not in self.request_timers:
            return
        
        timer_data = self.request_timers[request_id]
        duration = time.time() - timer_data['start_time']
        
        request_data = {
            'method': timer_data['method'],
            'path': timer_data['path'],
            'duration': duration,
            'status_code': status_code,
            'response_size': response_size,
            'timestamp': datetime.now()
        }
        
        self.request_stats[timer_data['path']].append(request_data)
        
        # Keep only recent requests per endpoint
        if len(self.request_stats[timer_data['path']]) > 100:
            self.request_stats[timer_data['path']] = \
                self.request_stats[timer_data['path']][-100:]
        
        del self.request_timers[request_id]
    
    def get_endpoint_stats(self, path: str) -> Dict[str, Any]:
        """Get statistics for a specific endpoint"""
        if path not in self.request_stats:
            return {}
        
        requests = self.request_stats[path]
        durations = [r['duration'] for r in requests]
        status_codes = [r['status_code'] for r in requests]
        
        return {
            'total_requests': len(requests),
            'avg_duration': statistics.mean(durations),
            'p95_duration': sorted(durations)[int(len(durations) * 0.95)] if durations else 0,
            'error_rate': len([s for s in status_codes if s >= 400]) / len(status_codes),
            'recent_requests': requests[-10:]
        }