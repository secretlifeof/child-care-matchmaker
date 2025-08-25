"""Loguru-based logger configuration for the matchmaker service."""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger


# Create logs directory if it doesn't exist
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

# Generate log filename with timestamp
SERVICE_START_TIME = datetime.now()
LOG_FILENAME = SERVICE_START_TIME.strftime("matchmaker_%Y%m%d_%H%M%S.log")
LOG_FILEPATH = LOGS_DIR / LOG_FILENAME

# Configuration from environment
DEBUG_MODE = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG" if DEBUG_MODE else "INFO")
VERBOSE_LOGGING = os.getenv("VERBOSE_LOGGING", "false").lower() in ("true", "1", "yes")


class MatchmakerLoguru:
    """Loguru-based logger for the matchmaker service."""
    
    _instance = None
    _configured = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._configured:
            self._setup_logger()
            self._configured = True
    
    def _setup_logger(self):
        """Configure Loguru logger with file and console handlers."""
        # Remove default handler
        logger.remove()
        
        # Console handler for development (pretty printing)
        if DEBUG_MODE:
            console_format = (
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "<level>{message}</level>"
            )
            logger.add(
                sys.stdout,
                format=console_format,
                level=LOG_LEVEL,
                colorize=True,
                backtrace=True,
                diagnose=True
            )
        
        # File handler with conditional JSON formatting
        logger.add(
            LOG_FILEPATH,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
            level="DEBUG",
            rotation="50 MB",
            retention="10 days",
            compression="gz",
            serialize=not DEBUG_MODE,  # Only serialize to JSON in production
            backtrace=True,
            diagnose=True,
            enqueue=True
        )
        
        # Log service start
        logger.bind(
            log_file=str(LOG_FILEPATH),
            debug_mode=DEBUG_MODE,
            log_level=LOG_LEVEL,
            verbose_logging=VERBOSE_LOGGING
        ).info("Matchmaker service started")
    
    def log_request(
        self,
        request_id: str,
        method: str,
        endpoint: str,
        headers: Dict[str, str],
        body: Any,
        query_params: Optional[Dict[str, str]] = None
    ):
        """Log incoming request."""
        # Sanitize sensitive data
        safe_headers = self._sanitize_headers(headers) if VERBOSE_LOGGING else {}
        safe_body = self._sanitize_body(body)
        
        log_data = {
            "request_id": request_id,
            "request_type": "incoming",
            "method": method,
            "endpoint": endpoint,
            "body": safe_body,
        }
        
        if VERBOSE_LOGGING:
            log_data.update({
                "headers": safe_headers,
                "query_params": query_params or {},
                "body_size": len(json.dumps(body)) if body else 0
            })
        
        if DEBUG_MODE:
            # Pretty print for console
            body_preview = json.dumps(safe_body, indent=2) if safe_body else "None"
            message = f"📥 Request {method} {endpoint}\n{body_preview}"
        else:
            message = f"Request received: {method} {endpoint}"
        
        logger.bind(**log_data).info(message)
    
    def log_response(
        self,
        request_id: str,
        status_code: int,
        headers: Dict[str, str],
        body: Any,
        duration_ms: float
    ):
        """Log outgoing response."""
        safe_body = self._sanitize_body(body)
        
        log_data = {
            "request_id": request_id,
            "request_type": "outgoing",
            "status_code": status_code,
            "duration_ms": duration_ms,
            "body": safe_body,
        }
        
        if VERBOSE_LOGGING:
            log_data.update({
                "headers": dict(headers),
                "body_size": len(json.dumps(body)) if body else 0
            })
        
        # Determine log level based on status code
        if status_code >= 500:
            log_level = "error"
        elif status_code >= 400:
            log_level = "warning"
        else:
            log_level = "info"
        
        if DEBUG_MODE:
            # Pretty print for console
            status_emoji = "✅" if status_code < 400 else "❌" if status_code >= 500 else "⚠️"
            body_preview = json.dumps(safe_body, indent=2) if safe_body else "None"
            message = f"{status_emoji} Response {status_code} ({duration_ms:.1f}ms)\n{body_preview}"
        else:
            message = f"Response sent - Status: {status_code}"
        
        getattr(logger.bind(**log_data), log_level)(message)
    
    def log_error(
        self,
        message: str,
        error: Optional[Exception] = None,
        **kwargs
    ):
        """Log an error."""
        if error:
            logger.bind(**kwargs).exception(f"❌ {message}")
        else:
            logger.bind(**kwargs).error(f"❌ {message}")
    
    def log_warning(self, message: str, **kwargs):
        """Log a warning."""
        logger.bind(**kwargs).warning(f"⚠️  {message}")
    
    def log_info(self, message: str, **kwargs):
        """Log info message."""
        logger.bind(**kwargs).info(f"ℹ️  {message}")
    
    def log_debug(self, message: str, **kwargs):
        """Log debug message."""
        logger.bind(**kwargs).debug(f"🔍 {message}")
    
    def log_matching_operation(
        self,
        operation: str,
        mode: str,
        input_data: Dict[str, Any],
        result: Dict[str, Any],
        duration_ms: float
    ):
        """Log matching operation details."""
        log_data = {
            "operation": operation,
            "mode": mode,
            "duration_ms": duration_ms,
            "input_summary": self._summarize_input(input_data),
            "result_summary": self._summarize_result(result)
        }
        
        if DEBUG_MODE:
            message = f"🎯 Matching operation completed: {operation} ({mode}) - {duration_ms:.1f}ms"
        else:
            message = f"Matching operation completed: {operation}"
        
        logger.bind(**log_data).info(message)
    
    def _sanitize_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Remove sensitive information from headers."""
        sensitive_headers = {
            'authorization', 'cookie', 'x-api-key', 
            'x-auth-token', 'x-csrf-token'
        }
        
        safe_headers = {}
        for key, value in headers.items():
            if key.lower() in sensitive_headers:
                safe_headers[key] = "***REDACTED***"
            else:
                safe_headers[key] = value
        
        return safe_headers
    
    def _sanitize_body(self, body: Any) -> Any:
        """Remove sensitive information from request/response body."""
        if not body:
            return body
        
        # List of sensitive field names
        sensitive_fields = {
            'password', 'token', 'secret', 'api_key',
            'access_token', 'refresh_token', 'ssn',
            'credit_card', 'bank_account'
        }
        
        if isinstance(body, dict):
            safe_body = {}
            for key, value in body.items():
                if any(sensitive in key.lower() for sensitive in sensitive_fields):
                    safe_body[key] = "***REDACTED***"
                elif isinstance(value, (dict, list)):
                    safe_body[key] = self._sanitize_body(value)
                else:
                    safe_body[key] = value
            return safe_body
        
        elif isinstance(body, list):
            return [self._sanitize_body(item) for item in body]
        
        return body
    
    def _summarize_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of input data for logging."""
        summary = {}
        
        if 'applications' in input_data:
            summary['application_count'] = len(input_data['applications'])
        elif 'application' in input_data:
            summary['application_id'] = str(input_data.get('application', {}).get('id', 'unknown'))
        
        if 'centers' in input_data:
            summary['center_count'] = len(input_data['centers'])
        
        if 'top_k' in input_data:
            summary['requested_recommendations'] = input_data['top_k']
        
        return summary
    
    def _summarize_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of result data for logging."""
        summary = {}
        
        if 'offers' in result:
            summary['offers_count'] = len(result['offers'])
        
        if 'waitlist_entries' in result:
            summary['waitlist_count'] = len(result['waitlist_entries'])
        
        if 'success' in result:
            summary['success'] = result['success']
        
        if 'matched_applications' in result:
            summary['matched_applications'] = result['matched_applications']
        
        if 'coverage_rate' in result:
            summary['coverage_rate'] = result['coverage_rate']
        
        return summary


# Singleton instance
matchmaker_logger = MatchmakerLoguru()

# Convenience functions
def get_logger() -> MatchmakerLoguru:
    """Get the matchmaker logger instance."""
    return matchmaker_logger

def log_request(*args, **kwargs):
    """Log a request."""
    matchmaker_logger.log_request(*args, **kwargs)

def log_response(*args, **kwargs):
    """Log a response."""
    matchmaker_logger.log_response(*args, **kwargs)

def log_error(*args, **kwargs):
    """Log an error."""
    matchmaker_logger.log_error(*args, **kwargs)

def log_warning(*args, **kwargs):
    """Log a warning."""
    matchmaker_logger.log_warning(*args, **kwargs)

def log_info(*args, **kwargs):
    """Log info."""
    matchmaker_logger.log_info(*args, **kwargs)

def log_debug(*args, **kwargs):
    """Log debug."""
    matchmaker_logger.log_debug(*args, **kwargs)