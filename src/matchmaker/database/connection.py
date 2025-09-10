"""PostgreSQL database connection management."""

import logging
import os
from contextlib import asynccontextmanager

import asyncpg

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages PostgreSQL database connections."""

    def __init__(self):
        self.pool: asyncpg.Pool | None = None
        self._initialized = False

    async def initialize(self, database_url: str | None = None):
        """Initialize database connection pool."""
        if self._initialized:
            logger.warning("Database already initialized")
            return

        db_url = database_url or os.getenv('DATABASE_URL')
        if not db_url:
            raise ValueError("DATABASE_URL environment variable is required")

        try:
            # Create connection pool
            self.pool = await asyncpg.create_pool(
                db_url,
                min_size=5,
                max_size=20,
                command_timeout=30,
                server_settings={
                    'application_name': 'matchmaker-service'
                }
            )

            # Test connection
            async with self.pool.acquire() as conn:
                await conn.fetchval('SELECT 1')

            self._initialized = True
            logger.info("PostgreSQL connection pool initialized")

        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            raise

    async def close(self):
        """Close database connection pool."""
        if self.pool:
            await self.pool.close()
            self._initialized = False
            logger.info("PostgreSQL connection pool closed")

    @asynccontextmanager
    async def get_connection(self):
        """Get a database connection from the pool."""
        if not self._initialized or not self.pool:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        async with self.pool.acquire() as connection:
            yield connection

    async def execute_query(self, query: str, *args):
        """Execute a query and return results."""
        async with self.get_connection() as conn:
            return await conn.fetch(query, *args)

    async def execute_one(self, query: str, *args):
        """Execute a query and return single result."""
        async with self.get_connection() as conn:
            return await conn.fetchrow(query, *args)

    async def execute_scalar(self, query: str, *args):
        """Execute a query and return scalar value."""
        async with self.get_connection() as conn:
            return await conn.fetchval(query, *args)

    async def execute_command(self, query: str, *args):
        """Execute a command (INSERT, UPDATE, DELETE)."""
        async with self.get_connection() as conn:
            return await conn.execute(query, *args)

    @property
    def is_initialized(self) -> bool:
        """Check if database is initialized."""
        return self._initialized and self.pool is not None

    async def health_check(self) -> bool:
        """Check database health."""
        try:
            async with self.get_connection() as conn:
                result = await conn.fetchval('SELECT 1')
                return result == 1
        except:
            return False


# Global database manager instance
_db_manager: DatabaseManager | None = None


async def get_database_manager() -> DatabaseManager:
    """Get global database manager instance."""
    global _db_manager

    if _db_manager is None:
        _db_manager = DatabaseManager()
        await _db_manager.initialize()

    return _db_manager


async def close_database_manager():
    """Close global database manager."""
    global _db_manager

    if _db_manager:
        await _db_manager.close()
        _db_manager = None


@asynccontextmanager
async def get_db_connection():
    """Convenience function to get database connection."""
    db_manager = await get_database_manager()
    async with db_manager.get_connection() as conn:
        yield conn
