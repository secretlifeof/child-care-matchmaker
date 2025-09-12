"""Main FastAPI application for the matchmaker service."""

import os
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.middleware import LoggingMiddleware
from .api.routes import matches, semantic, streaming, preferences
from .database import close_database_manager, get_database_manager
from .services.graph import close_global_client, get_graph_client
from .services.graph.factory import GraphClientFactory
from .services.map_service import close_map_service
from .utils.logger import get_logger

# Get the matchmaker logger instead of basic logging
logger = get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    logger.log_info("Starting Parent-Daycare Matchmaker Service")

    # Initialize PostgreSQL database connection
    try:
        db_manager = await get_database_manager()
        if await db_manager.health_check():
            logger.log_info("PostgreSQL database connected successfully")
        else:
            raise RuntimeError("PostgreSQL health check failed")
    except Exception as e:
        logger.log_error(f"Failed to connect to PostgreSQL: {e}")
        raise

    # Initialize graph database connection
    try:
        # Validate graph database configuration
        graph_type = os.getenv('GRAPH_DB_TYPE', 'tigergraph')
        is_valid, missing_vars = GraphClientFactory.validate_environment()

        if is_valid:
            graph_client = await get_graph_client()
            logger.log_info(f"Connected to {graph_type} graph database")

            # Install required queries for TigerGraph
            if hasattr(graph_client, 'install_required_queries'):
                await graph_client.install_required_queries()
                logger.log_info("Installed required graph queries")
        else:
            logger.log_warning(f"Graph database configuration invalid: missing {missing_vars}")
            logger.log_warning("Graph features will be disabled")

    except Exception as e:
        logger.log_error(f"Failed to initialize graph database: {e}")
        logger.log_warning("Graph features will be disabled")

    # Initialize semantic matching embeddings if OpenAI key is available
    try:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            from .services.semantic_matching import SemanticMatchingService
            db_manager = await get_database_manager()
            semantic_service = SemanticMatchingService(db_manager.pool, openai_api_key)
            await semantic_service._ensure_embeddings_exist()
            logger.log_info("Semantic matching embeddings ready")
        else:
            logger.log_warning("OpenAI API key not configured, semantic matching disabled")
    except Exception as e:
        logger.log_error(f"Failed to initialize semantic matching: {e}")
        logger.log_warning("Semantic matching features will be disabled")

    # Initialize MapBox service if API key is available
    try:
        mapbox_api_key = os.getenv("MAPBOX_API_KEY")
        if mapbox_api_key:
            from .services.map_service import get_map_service
            # Initialize the service (this will create the global instance)
            get_map_service(mapbox_api_key)
            logger.log_info("MapBox geocoding service initialized")
        else:
            logger.log_warning("MapBox API key not configured, geocoding features may be limited")
    except Exception as e:
        logger.log_error(f"Failed to initialize MapBox service: {e}")
        logger.log_warning("MapBox geocoding features will be disabled")

    yield

    # Shutdown
    logger.log_info("Shutting down Parent-Daycare Matchmaker Service")
    await close_global_client()
    await close_database_manager()
    await close_map_service()


# Create FastAPI app
app = FastAPI(
    title="Parent-Daycare Matchmaker",
    description="Enhanced graph-based matching service with TigerGraph/Neo4j support",
    version="2.0.0",
    lifespan=lifespan
)

# Add logging middleware first
app.add_middleware(LoggingMiddleware)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(matches.router)
app.include_router(semantic.router)
app.include_router(streaming.router)
app.include_router(preferences.router)


@app.get("/")
async def root():
    """Root endpoint."""
    # Get graph database info
    graph_type = os.getenv('GRAPH_DB_TYPE', 'tigergraph')
    is_valid, missing_vars = GraphClientFactory.validate_environment()

    return {
        "service": "Parent-Daycare Matchmaker",
        "version": "2.0.0",
        "status": "healthy",
        "graph_database": {
            "type": graph_type,
            "configured": is_valid,
            "missing_config": missing_vars if not is_valid else []
        },
        "endpoints": {
            "/api/matches/recommend": "Get personalized recommendations (legacy)",
            "/api/matches/enhanced-recommend": "Enhanced recommendations with graph DB",
            "/api/matches/allocate": "Perform global allocation",
            "/api/matches/waitlist": "Generate center waitlist",
            "/api/matches/batch": "Batch matching operations",
            "/api/matches/stats": "Service statistics and configuration",
            "/api/preferences/process/interactive": "Process parent preferences with user interaction streaming",
            "/api/preferences/interactions/respond": "Respond to preference processing interactions",
            "/api/streaming/matches/interactive": "Interactive matching with user input streaming (deprecated)",
            "/api/streaming/interactions/respond": "Respond to user interaction requests (deprecated)",
            "/docs": "Interactive API documentation"
        },
        "features": {
            "graph_databases": ["TigerGraph", "Neo4j"],
            "capacity_checking": True,
            "semantic_matching": True,
            "explainable_results": True,
            "geocoding": ["MapBox"],
            "streaming_interactions": True
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint with graph database status."""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.0"
    }

    # Check PostgreSQL database health
    try:
        db_manager = await get_database_manager()
        db_healthy = await db_manager.health_check()
        health_status["postgresql"] = {
            "status": "healthy" if db_healthy else "unhealthy",
            "connected": db_manager.is_initialized
        }
        if not db_healthy:
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["postgresql"] = {
            "status": "error",
            "error": str(e)
        }
        health_status["status"] = "degraded"

    # Check graph database health
    try:
        graph_client = await get_graph_client()
        graph_healthy = await graph_client.health_check()
        health_status["graph_database"] = {
            "type": os.getenv('GRAPH_DB_TYPE', 'tigergraph'),
            "status": "healthy" if graph_healthy else "unhealthy",
            "connected": graph_client.connected
        }
        if not graph_healthy:
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["graph_database"] = {
            "type": os.getenv('GRAPH_DB_TYPE', 'tigergraph'),
            "status": "error",
            "error": str(e)
        }
        health_status["status"] = "degraded"

    # Check MapBox service health  
    try:
        mapbox_api_key = os.getenv("MAPBOX_API_KEY")
        if mapbox_api_key:
            from .services.map_service import get_map_service
            map_service = get_map_service(mapbox_api_key)
            # Simple test geocoding to check if service is working
            test_result = await map_service.geocode_address("Berlin", country_code="de")
            health_status["mapbox"] = {
                "status": "healthy" if test_result else "degraded",
                "configured": True,
                "test_geocoding": test_result is not None
            }
            if not test_result:
                health_status["status"] = "degraded"
        else:
            health_status["mapbox"] = {
                "status": "not_configured",
                "configured": False,
                "message": "MapBox API key not provided"
            }
    except Exception as e:
        health_status["mapbox"] = {
            "status": "error",
            "error": str(e),
            "configured": mapbox_api_key is not None
        }
        health_status["status"] = "degraded"

    return health_status


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
