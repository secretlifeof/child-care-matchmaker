FROM python:3.11-slim AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create application user
RUN groupadd -r matchmaker && useradd -r -g matchmaker matchmaker

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/

# Create logs directory
RUN mkdir -p logs && chown -R matchmaker:matchmaker logs

# Change to non-root user
USER matchmaker

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Production stage
FROM base as production

# Set production environment
ENV ENVIRONMENT=production
ENV DEBUG=false

# Run the application
CMD ["uvicorn", "src.matchmaker.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "4"]

# Development stage
FROM base as development

# Set development environment
ENV ENVIRONMENT=development
ENV DEBUG=true

# Run with auto-reload for development
CMD ["uvicorn", "src.matchmaker.main:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]

# Default to production
FROM production