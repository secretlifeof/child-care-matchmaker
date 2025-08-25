"""Logging middleware for request/response tracking."""

import time
import uuid
import json
from typing import Callable, Dict, Any
from fastapi import Request, Response
from fastapi.routing import APIRoute
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import Message

from src.matchmaker.utils.logger import get_logger


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log all requests and responses."""
    
    def __init__(self, app):
        super().__init__(app)
        self.logger = get_logger()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process and log each request/response."""
        # Generate request ID
        request_id = str(uuid.uuid4())
        
        # Extract request details
        method = request.method
        path = request.url.path
        query_params = dict(request.query_params)
        headers = dict(request.headers)
        
        # Read body (if present)
        body = None
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                body_bytes = await request.body()
                if body_bytes:
                    body = json.loads(body_bytes)
                # Need to recreate request with body for downstream
                async def receive() -> Message:
                    return {"type": "http.request", "body": body_bytes}
                request._receive = receive
            except Exception as e:
                self.logger.log_debug(f"Could not parse request body: {e}")
        
        # Log the request
        self.logger.log_request(
            request_id=request_id,
            method=method,
            endpoint=path,
            headers=headers,
            body=body,
            query_params=query_params
        )
        
        # Track timing
        start_time = time.time()
        
        try:
            # Process the request
            response = await call_next(request)
            
            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000
            
            # Capture response body
            response_body = b""
            async for chunk in response.body_iterator:
                response_body += chunk
            
            # Parse response body if JSON
            try:
                response_data = json.loads(response_body) if response_body else None
            except:
                response_data = response_body.decode('utf-8') if response_body else None
            
            # Log the response
            self.logger.log_response(
                request_id=request_id,
                status_code=response.status_code,
                headers=dict(response.headers),
                body=response_data,
                duration_ms=duration_ms
            )
            
            # Return response with body
            return Response(
                content=response_body,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type
            )
            
        except Exception as e:
            # Log error
            duration_ms = (time.time() - start_time) * 1000
            self.logger.log_error(
                f"Request failed: {str(e)}",
                error=e,
                request_id=request_id,
                duration_ms=duration_ms
            )
            raise


class LoggingRoute(APIRoute):
    """Custom route class for request/response logging."""
    
    def get_route_handler(self) -> Callable:
        original_route_handler = super().get_route_handler()
        logger = get_logger()
        
        async def custom_route_handler(request: Request) -> Response:
            # Generate request ID
            request_id = str(uuid.uuid4())
            request.state.request_id = request_id
            
            # Set request context
            logger.set_request_context(
                request_id=request_id,
                endpoint=request.url.path,
                method=request.method
            )
            
            try:
                response = await original_route_handler(request)
                return response
            finally:
                # Clear context
                logger.clear_request_context()
        
        return custom_route_handler