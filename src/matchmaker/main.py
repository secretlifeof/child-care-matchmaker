"""Main FastAPI application for the matchmaker service."""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import matches
from .api.middleware import LoggingMiddleware
from .utils.logger import get_logger

# Get the matchmaker logger instead of basic logging
logger = get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    logger.log_info("Starting Parent-Daycare Matchmaker Service")
    yield
    # Shutdown
    logger.log_info("Shutting down Parent-Daycare Matchmaker Service")


# Create FastAPI app
app = FastAPI(
    title="Parent-Daycare Matchmaker",
    description="Graph-based matching service for parents and daycare centers",
    version="1.0.0",
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


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Parent-Daycare Matchmaker",
        "version": "1.0.0",
        "status": "healthy",
        "endpoints": {
            "/api/matches/recommend": "Get personalized recommendations",
            "/api/matches/allocate": "Perform global allocation",
            "/api/matches/waitlist": "Generate center waitlist",
            "/api/matches/batch": "Batch matching operations",
            "/docs": "API documentation"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)