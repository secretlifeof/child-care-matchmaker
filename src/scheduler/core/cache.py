"""
Caching layer for optimization results and frequently accessed data
"""

import asyncio
import hashlib
import json
import logging
import pickle
from datetime import datetime
from typing import Any, Optional

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from ..config import settings

logger = logging.getLogger(__name__)


class CacheManager:
    """Manages caching of optimization results and frequently accessed data"""

    def __init__(self):
        self.redis_client = None
        self.memory_cache = {}
        self.cache_timestamps = {}
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "errors": 0,
        }

        self._initialize_redis()
        logger.info(
            f"Cache manager initialized (Redis: {'available' if self.redis_client else 'unavailable'})"
        )

    def _initialize_redis(self):
        """Initialize Redis connection if available"""
        if not REDIS_AVAILABLE:
            logger.info("Redis not available, using in-memory cache only")
            return

        try:
            self.redis_client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db,
                password=settings.redis_password,
                decode_responses=False,  # We'll handle encoding ourselves
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True,
                max_connections=20,
            )

            # Test connection
            self.redis_client.ping()
            logger.info(
                f"Redis connected successfully to {settings.redis_host}:{settings.redis_port}"
            )

        except Exception as e:
            logger.warning(
                f"Redis connection failed: {str(e)}. Using in-memory cache only."
            )
            self.redis_client = None

    async def get(self, key: str) -> Any | None:
        """Get cached value"""

        try:
            # Try Redis first if available
            if self.redis_client:
                cached_data = await self._redis_get(key)
                if cached_data is not None:
                    self.cache_stats["hits"] += 1
                    return cached_data

            # Fallback to memory cache
            if key in self.memory_cache:
                timestamp = self.cache_timestamps.get(key, 0)
                if datetime.now().timestamp() - timestamp < settings.cache_ttl:
                    self.cache_stats["hits"] += 1
                    return self.memory_cache[key]
                else:
                    # Expired, remove from cache
                    await self._remove_from_memory_cache(key)

            self.cache_stats["misses"] += 1
            return None

        except Exception as e:
            logger.error(f"Cache get error for key {key}: {str(e)}")
            self.cache_stats["errors"] += 1
            return None

    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set cached value"""

        if ttl is None:
            ttl = settings.cache_ttl

        try:
            success = False

            # Try Redis first if available
            if self.redis_client:
                success = await self._redis_set(key, value, ttl)

            # Always store in memory cache as backup
            await self._set_in_memory_cache(key, value, ttl)

            if success or not self.redis_client:  # Success if Redis worked or no Redis
                self.cache_stats["sets"] += 1
                return True

            return False

        except Exception as e:
            logger.error(f"Cache set error for key {key}: {str(e)}")
            self.cache_stats["errors"] += 1
            return False

    async def delete(self, key: str) -> bool:
        """Delete cached value"""

        try:
            success = True

            # Delete from Redis if available
            if self.redis_client:
                try:
                    await asyncio.get_event_loop().run_in_executor(
                        None, self.redis_client.delete, key
                    )
                except Exception as e:
                    logger.warning(f"Redis delete failed for key {key}: {e}")
                    success = False

            # Delete from memory cache
            await self._remove_from_memory_cache(key)

            if success:
                self.cache_stats["deletes"] += 1

            return success

        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {str(e)}")
            self.cache_stats["errors"] += 1
            return False

    async def clear(self) -> bool:
        """Clear all cached values"""

        try:
            # Clear Redis if available
            if self.redis_client:
                try:
                    await asyncio.get_event_loop().run_in_executor(
                        None, self.redis_client.flushdb
                    )
                except Exception as e:
                    logger.warning(f"Redis clear failed: {e}")

            # Clear memory cache
            self.memory_cache.clear()
            self.cache_timestamps.clear()

            logger.info("Cache cleared successfully")
            return True

        except Exception as e:
            logger.error(f"Cache clear error: {str(e)}")
            self.cache_stats["errors"] += 1
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""

        try:
            # Check Redis first
            if self.redis_client:
                exists = await asyncio.get_event_loop().run_in_executor(
                    None, self.redis_client.exists, key
                )
                if exists:
                    return True

            # Check memory cache
            if key in self.memory_cache:
                timestamp = self.cache_timestamps.get(key, 0)
                if datetime.now().timestamp() - timestamp < settings.cache_ttl:
                    return True
                else:
                    # Expired
                    await self._remove_from_memory_cache(key)

            return False

        except Exception as e:
            logger.error(f"Cache exists check error for key {key}: {str(e)}")
            return False

    async def get_stats(self) -> dict[str, Any]:
        """Get cache statistics"""

        stats = self.cache_stats.copy()
        stats["memory_cache_size"] = len(self.memory_cache)
        stats["redis_available"] = self.redis_client is not None

        # Add Redis-specific stats if available
        if self.redis_client:
            try:
                redis_info = await asyncio.get_event_loop().run_in_executor(
                    None, self.redis_client.info
                )
                stats["redis_memory_usage"] = redis_info.get(
                    "used_memory_human", "unknown"
                )
                stats["redis_connected_clients"] = redis_info.get(
                    "connected_clients", 0
                )
                stats["redis_keyspace_hits"] = redis_info.get("keyspace_hits", 0)
                stats["redis_keyspace_misses"] = redis_info.get("keyspace_misses", 0)
            except Exception as e:
                logger.warning(f"Failed to get Redis stats: {e}")

        return stats

    async def get_keys_by_pattern(self, pattern: str) -> list[str]:
        """Get keys matching a pattern"""

        try:
            keys = []

            # Get from Redis if available
            if self.redis_client:
                redis_keys = await asyncio.get_event_loop().run_in_executor(
                    None, self.redis_client.keys, pattern
                )
                keys.extend(
                    [
                        key.decode() if isinstance(key, bytes) else key
                        for key in redis_keys
                    ]
                )

            # Get from memory cache
            import fnmatch

            memory_keys = [
                key for key in self.memory_cache.keys() if fnmatch.fnmatch(key, pattern)
            ]
            keys.extend(memory_keys)

            return list(set(keys))  # Remove duplicates

        except Exception as e:
            logger.error(f"Error getting keys by pattern {pattern}: {str(e)}")
            return []

    async def set_with_expiry(self, key: str, value: Any, expire_at: datetime) -> bool:
        """Set value with specific expiry time"""

        now = datetime.now()
        if expire_at <= now:
            return False

        ttl = int((expire_at - now).total_seconds())
        return await self.set(key, value, ttl)

    async def increment(self, key: str, amount: int = 1) -> int | None:
        """Increment a numeric value in cache"""

        try:
            # Try Redis first for atomic operation
            if self.redis_client:
                try:
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, self.redis_client.incrby, key, amount
                    )
                    return result
                except Exception as e:
                    logger.warning(f"Redis increment failed for key {key}: {e}")

            # Fallback to get-set operation
            current_value = await self.get(key)
            if current_value is None:
                current_value = 0

            new_value = int(current_value) + amount
            success = await self.set(key, new_value)
            return new_value if success else None

        except Exception as e:
            logger.error(f"Cache increment error for key {key}: {str(e)}")
            return None

    async def expire(self, key: str, ttl: int) -> bool:
        """Set expiry time for existing key"""

        try:
            # Update Redis expiry if available
            if self.redis_client:
                try:
                    await asyncio.get_event_loop().run_in_executor(
                        None, self.redis_client.expire, key, ttl
                    )
                except Exception as e:
                    logger.warning(f"Redis expire failed for key {key}: {e}")

            # Update memory cache timestamp
            if key in self.memory_cache:
                self.cache_timestamps[key] = datetime.now().timestamp()

            return True

        except Exception as e:
            logger.error(f"Cache expire error for key {key}: {str(e)}")
            return False

    # Redis-specific operations
    async def _redis_get(self, key: str) -> Any | None:
        """Get value from Redis"""

        try:
            cached_data = await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.get, key
            )

            if cached_data:
                return pickle.loads(cached_data)
            return None

        except Exception as e:
            logger.warning(f"Redis get failed for key {key}: {e}")
            return None

    async def _redis_set(self, key: str, value: Any, ttl: int) -> bool:
        """Set value in Redis"""

        try:
            serialized_data = pickle.dumps(value)
            result = await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.setex, key, ttl, serialized_data
            )
            return bool(result)

        except Exception as e:
            logger.warning(f"Redis set failed for key {key}: {e}")
            return False

    # Memory cache operations
    async def _set_in_memory_cache(self, key: str, value: Any, ttl: int):
        """Set value in memory cache"""

        self.memory_cache[key] = value
        self.cache_timestamps[key] = datetime.now().timestamp()

        # Simple cleanup of expired entries
        await self._cleanup_memory_cache()

    async def _remove_from_memory_cache(self, key: str):
        """Remove key from memory cache"""

        self.memory_cache.pop(key, None)
        self.cache_timestamps.pop(key, None)

    async def _cleanup_memory_cache(self):
        """Clean up expired entries from memory cache"""

        if len(self.memory_cache) < 100:  # Only cleanup if cache is getting large
            return

        current_time = datetime.now().timestamp()
        expired_keys = []

        for key, timestamp in self.cache_timestamps.items():
            if current_time - timestamp > settings.cache_ttl:
                expired_keys.append(key)

        for key in expired_keys:
            await self._remove_from_memory_cache(key)

        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")


class ScheduleCacheHelper:
    """Helper class for schedule-specific caching operations"""

    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager

    async def cache_schedule_result(
        self,
        request_hash: str,
        response: "ScheduleGenerationResponse",
        ttl: int | None = None,
    ) -> bool:
        """Cache a schedule generation result"""

        cache_key = f"schedule:result:{request_hash}"
        return await self.cache.set(cache_key, response, ttl)

    async def get_cached_schedule_result(
        self, request_hash: str
    ) -> Optional["ScheduleGenerationResponse"]:
        """Get cached schedule generation result"""

        cache_key = f"schedule:result:{request_hash}"
        return await self.cache.get(cache_key)

    async def cache_staff_availability(
        self, staff_id: str, week_start: str, availability_data: dict
    ) -> bool:
        """Cache staff availability data"""

        cache_key = f"staff:availability:{staff_id}:{week_start}"
        return await self.cache.set(cache_key, availability_data, ttl=3600)  # 1 hour

    async def get_cached_staff_availability(
        self, staff_id: str, week_start: str
    ) -> dict | None:
        """Get cached staff availability data"""

        cache_key = f"staff:availability:{staff_id}:{week_start}"
        return await self.cache.get(cache_key)

    async def cache_optimization_metrics(
        self, center_id: str, metrics_data: dict
    ) -> bool:
        """Cache optimization metrics for analytics"""

        cache_key = f"metrics:{center_id}:{datetime.now().strftime('%Y-%m-%d')}"
        return await self.cache.set(cache_key, metrics_data, ttl=86400)  # 24 hours

    async def invalidate_schedule_cache(self, center_id: str, week_start: str) -> bool:
        """Invalidate cached schedules for a specific center and week"""

        pattern = f"schedule:result:*{center_id}*{week_start}*"
        keys = await self.cache.get_keys_by_pattern(pattern)

        success = True
        for key in keys:
            if not await self.cache.delete(key):
                success = False

        return success

    def generate_request_hash(self, request: "ScheduleGenerationRequest") -> str:
        """Generate a hash for a schedule request for caching"""

        # Create a simplified representation for hashing
        hash_data = {
            "center_id": str(request.center_id),
            "week_start_date": request.week_start_date.isoformat(),
            "staff_count": len(request.staff),
            "groups_count": len(request.groups),
            "requirements_count": len(request.staffing_requirements),
            "optimization_goals": [
                goal.value for goal in request.optimization_config.goals
            ],
            "max_solver_time": request.optimization_config.max_solver_time,
        }

        # Add staff priority weights to hash
        staff_priorities = {}
        for staff_member in request.staff:
            staff_priorities[str(staff_member.staff_id)] = {
                "seniority_weight": staff_member.seniority_weight,
                "performance_weight": staff_member.performance_weight,
                "flexibility_score": staff_member.flexibility_score,
                "overall_priority": staff_member.overall_priority.value,
                "max_weekly_hours": staff_member.max_weekly_hours,
            }
        hash_data["staff_priorities"] = staff_priorities

        # Create hash
        hash_string = json.dumps(hash_data, sort_keys=True)
        return hashlib.md5(hash_string.encode()).hexdigest()


class CacheWarmer:
    """Background cache warming service"""

    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager
        self.is_running = False

    async def start_warming(self):
        """Start cache warming process"""

        if self.is_running:
            return

        self.is_running = True
        logger.info("Starting cache warming process")

        # This would run in background to pre-populate frequently accessed data
        asyncio.create_task(self._warm_cache_background())

    async def stop_warming(self):
        """Stop cache warming process"""

        self.is_running = False
        logger.info("Stopping cache warming process")

    async def _warm_cache_background(self):
        """Background process to warm cache with frequently accessed data"""

        while self.is_running:
            try:
                # Warm up common configuration data
                await self._warm_configuration_cache()

                # Warm up staff data that hasn't been accessed recently
                await self._warm_staff_cache()

                # Sleep for a while before next warming cycle
                await asyncio.sleep(300)  # 5 minutes

            except Exception as e:
                logger.error(f"Error in cache warming: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error

    async def _warm_configuration_cache(self):
        """Warm cache with configuration data"""

        # Cache common configuration that doesn't change often
        config_data = {
            "staff_ratios": {
                "infant": settings.infant_ratio,
                "toddler": settings.toddler_ratio,
                "preschool": settings.preschool_ratio,
            },
            "business_rules": {
                "max_consecutive_hours": settings.max_consecutive_hours,
                "min_break_between_shifts": settings.min_break_between_shifts,
                "max_weekly_hours": settings.max_weekly_hours,
            },
        }

        await self.cache.set("config:ratios_and_rules", config_data, ttl=3600)

    async def _warm_staff_cache(self):
        """Warm cache with staff-related data"""

        # This would typically query database for staff that might be needed soon
        # For now, just ensure cache health
        await self.cache.get_stats()
