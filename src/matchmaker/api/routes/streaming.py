"""Streaming API endpoints for user interactions during complex preference processing."""

import asyncio
import json
import logging
from typing import Any, Dict, List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ...services.complex_type_agent_registry import ComplexTypeAgentRegistry
from ...database.complex_queries import ComplexPreferenceRepository
from ...models.base import ParentPreference, Center

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/streaming", tags=["streaming"])


class InteractiveMatchRequest(BaseModel):
    """Request for interactive matching with streaming responses."""
    parent_id: UUID
    limit: int = 10
    max_distance_km: float = 10.0
    include_full_centers: bool = False
    include_explanations: bool = True
    session_id: str | None = None


class UserInteractionResponse(BaseModel):
    """User interaction response from client."""
    session_id: str
    interaction_id: str
    response_type: str  # "location_input", "confirmation", "choice"
    response_data: Dict[str, Any]


class StreamMessage(BaseModel):
    """Base class for streaming messages."""
    type: str
    session_id: str
    timestamp: int


class ProcessingUpdateMessage(StreamMessage):
    """Processing progress update."""
    type: str = "processing_update"
    stage: str
    progress: float  # 0.0 to 1.0
    details: str


class UserInteractionMessage(StreamMessage):
    """User interaction required."""
    type: str = "user_interaction"
    interaction_id: str
    interaction_type: str  # "location_input", "confirmation", "choice"
    question: str
    field_name: str
    options: List[Dict[str, Any]] | None = None
    context: Dict[str, Any]


class MatchResultMessage(StreamMessage):
    """Final match results."""
    type: str = "match_results"
    offers: List[Dict[str, Any]]
    processing_details: Dict[str, Any]
    success: bool
    message: str | None = None


class ErrorMessage(StreamMessage):
    """Error during processing."""
    type: str = "error"
    error_code: str
    message: str
    details: Dict[str, Any] | None = None


# Global session storage (in production, use Redis)
active_sessions: Dict[str, Dict[str, Any]] = {}


@router.post("/matches/interactive")
async def start_interactive_matching(
    request: InteractiveMatchRequest,
    background_tasks: BackgroundTasks,
    # agent_registry: ComplexTypeAgentRegistry = Depends(get_agent_registry),
    # db_repository: ComplexPreferenceRepository = Depends(get_db_repository)
):
    """Start interactive matching with streaming responses for user interactions."""
    
    session_id = request.session_id or f"session_{request.parent_id}_{asyncio.get_event_loop().time()}"
    
    # Initialize session storage
    active_sessions[session_id] = {
        "parent_id": request.parent_id,
        "request": request.dict(),
        "interactions": {},
        "queue": asyncio.Queue(),
        "completed": False,
        "results": None
    }
    
    # Start background processing
    background_tasks.add_task(
        process_interactive_matching,
        session_id,
        request
    )
    
    # Return streaming response
    return StreamingResponse(
        stream_responses(session_id),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",  # Configure appropriately for production
            "Access-Control-Allow-Headers": "Content-Type",
        }
    )


async def stream_responses(session_id: str):
    """Stream responses for a session."""
    session = active_sessions.get(session_id)
    if not session:
        yield f"data: {json.dumps({'type': 'error', 'message': 'Session not found'})}\n\n"
        return
    
    queue = session["queue"]
    
    try:
        while not session["completed"]:
            try:
                # Wait for message with timeout
                message = await asyncio.wait_for(queue.get(), timeout=30.0)
                yield f"data: {json.dumps(message.dict())}\n\n"
                
                # If this is a final result, mark as completed
                if message.type in ["match_results", "error"]:
                    session["completed"] = True
                    break
                    
            except asyncio.TimeoutError:
                # Send keepalive
                yield f"data: {json.dumps({'type': 'keepalive', 'session_id': session_id})}\n\n"
                
    except Exception as e:
        logger.error(f"Streaming error for session {session_id}: {e}")
        error_msg = ErrorMessage(
            session_id=session_id,
            timestamp=int(asyncio.get_event_loop().time()),
            error_code="streaming_error",
            message=str(e)
        )
        yield f"data: {json.dumps(error_msg.dict())}\n\n"
    
    finally:
        # Clean up session after delay
        await asyncio.sleep(300)  # Keep session for 5 minutes
        if session_id in active_sessions:
            del active_sessions[session_id]


async def process_interactive_matching(
    session_id: str,
    request: InteractiveMatchRequest
):
    """Process matching with interactive user input collection."""
    session = active_sessions[session_id]
    queue = session["queue"]
    
    try:
        # Send processing start
        await queue.put(ProcessingUpdateMessage(
            session_id=session_id,
            timestamp=int(asyncio.get_event_loop().time()),
            stage="loading_preferences",
            progress=0.1,
            details="Loading parent preferences from database"
        ))
        
        # Load preferences (mock implementation)
        preferences = await load_parent_preferences(request.parent_id)
        
        await queue.put(ProcessingUpdateMessage(
            session_id=session_id,
            timestamp=int(asyncio.get_event_loop().time()),
            stage="finding_centers",
            progress=0.3,
            details=f"Found {len(preferences)} preferences, searching for nearby centers"
        ))
        
        # Find nearby centers (mock)
        centers = await find_nearby_centers(request.parent_id, request.max_distance_km)
        
        await queue.put(ProcessingUpdateMessage(
            session_id=session_id,
            timestamp=int(asyncio.get_event_loop().time()),
            stage="processing_complex_preferences",
            progress=0.5,
            details=f"Processing complex preferences against {len(centers)} centers"
        ))
        
        # Process complex preferences - this is where interactions may be needed
        offers = []
        for i, center in enumerate(centers[:request.limit]):
            
            # Process each complex preference
            for preference in preferences:
                if preference.get("complex_value_type_id"):
                    # This would use the actual agent registry
                    interaction_needed = await check_if_interaction_needed(preference, center)
                    
                    if interaction_needed:
                        # Send user interaction request
                        interaction_id = f"interaction_{len(session['interactions'])}"
                        
                        await queue.put(UserInteractionMessage(
                            session_id=session_id,
                            timestamp=int(asyncio.get_event_loop().time()),
                            interaction_id=interaction_id,
                            interaction_type="location_input",
                            question="Where is your work location?",
                            field_name="work_location",
                            context={
                                "preference_id": preference.get("id"),
                                "center_id": center.get("id"),
                                "required_for": "distance_calculation"
                            }
                        ))
                        
                        # Wait for user response
                        response = await wait_for_user_response(session_id, interaction_id)
                        if not response:
                            raise Exception("User interaction timeout")
                        
                        # Continue processing with user response
                        preference["user_responses"] = response
            
            # Create mock offer
            offers.append({
                "center_id": center["id"],
                "center_name": center["name"],
                "score": 0.85,  # Mock score
                "preference_matches": {
                    "complex_preferences": {
                        "location_distance": {
                            "satisfied": True,
                            "details": "home: ideal walking time (8min); work: acceptable bike time (18min)",
                            "score": 0.95
                        }
                    }
                }
            })
            
            # Update progress
            progress = 0.5 + (i / len(centers)) * 0.4
            await queue.put(ProcessingUpdateMessage(
                session_id=session_id,
                timestamp=int(asyncio.get_event_loop().time()),
                stage="processing_complex_preferences",
                progress=progress,
                details=f"Processed {i+1}/{len(centers)} centers"
            ))
        
        # Send final results
        await queue.put(MatchResultMessage(
            session_id=session_id,
            timestamp=int(asyncio.get_event_loop().time()),
            offers=offers,
            processing_details={
                "centers_evaluated": len(centers),
                "complex_types_processed": ["location_distance"],
                "processing_time_ms": int((asyncio.get_event_loop().time() - session["start_time"]) * 1000),
                "interactions_required": len(session["interactions"])
            },
            success=True
        ))
        
    except Exception as e:
        logger.error(f"Processing error for session {session_id}: {e}")
        await queue.put(ErrorMessage(
            session_id=session_id,
            timestamp=int(asyncio.get_event_loop().time()),
            error_code="processing_error",
            message=str(e),
            details={"parent_id": str(request.parent_id)}
        ))


@router.post("/interactions/respond")
async def respond_to_interaction(response: UserInteractionResponse):
    """Handle user response to an interaction."""
    
    session = active_sessions.get(response.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Store the response
    session["interactions"][response.interaction_id] = {
        "response_type": response.response_type,
        "response_data": response.response_data,
        "timestamp": asyncio.get_event_loop().time()
    }
    
    return {"status": "received", "interaction_id": response.interaction_id}


@router.get("/sessions/{session_id}/status")
async def get_session_status(session_id: str):
    """Get current status of a session."""
    session = active_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session_id,
        "completed": session["completed"],
        "interactions_count": len(session["interactions"]),
        "has_results": session["results"] is not None
    }


# Mock implementation functions
async def load_parent_preferences(parent_id: UUID) -> List[Dict[str, Any]]:
    """Mock: Load parent preferences from database."""
    await asyncio.sleep(0.5)  # Simulate DB query
    return [
        {
            "id": "pref_1",
            "complex_value_type_id": "location_distance_type",
            "value_data": {
                "locations": [
                    {"name": "home", "max_travel_minutes": 15, "transport_mode": "walking"},
                    {"name": "work", "max_travel_minutes": 20, "transport_mode": "bike"}
                ]
            }
        }
    ]


async def find_nearby_centers(parent_id: UUID, max_distance_km: float) -> List[Dict[str, Any]]:
    """Mock: Find nearby centers."""
    await asyncio.sleep(0.3)
    return [
        {"id": "center_1", "name": "Green Garden Daycare"},
        {"id": "center_2", "name": "Happy Kids Center"},
        {"id": "center_3", "name": "Little Explorers"}
    ]


async def check_if_interaction_needed(preference: Dict[str, Any], center: Dict[str, Any]) -> bool:
    """Mock: Check if user interaction is needed."""
    # Simulate that work location is missing
    locations = preference.get("value_data", {}).get("locations", [])
    for location in locations:
        if location.get("name") == "work" and "latitude" not in location and "address" not in location:
            return True
    return False


async def wait_for_user_response(session_id: str, interaction_id: str, timeout: int = 300) -> Dict[str, Any] | None:
    """Wait for user response to interaction."""
    session = active_sessions[session_id]
    
    start_time = asyncio.get_event_loop().time()
    while asyncio.get_event_loop().time() - start_time < timeout:
        if interaction_id in session["interactions"]:
            return session["interactions"][interaction_id]["response_data"]
        await asyncio.sleep(1)
    
    return None