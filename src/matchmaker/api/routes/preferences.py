"""API routes for interactive parent preference processing."""

import asyncio
import json
import logging
from typing import Any, Dict, List
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ...services.complex_type_agent_registry import ComplexTypeAgentRegistry
from ...services.semantic_matching import SemanticMatchingService
from ...database.complex_queries import ComplexPreferenceRepository
from ...models.base import ParentPreference

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/preferences", tags=["preferences"])


class ExtractedPreference(BaseModel):
    """Extracted simple preference from user text."""
    feature_key: str
    confidence: float
    preference: str  # "required", "preferred", "nice_to_have", "avoid"
    raw_phrase: str
    value_type: str = "text"


class ExtractedComplexPreference(BaseModel):
    """Pre-extracted complex preference with LLM-generated value_data."""
    feature_key: str
    type_name: str  # e.g., "location_distance", "schedule_range"
    value_data: Dict[str, Any]  # LLM-generated structured data
    preference: str  # "required", "preferred", "nice_to_have", "avoid"
    confidence: float
    rationale: str  # LLM explanation of why it extracted this


class ProcessPreferencesRequest(BaseModel):
    """Request for processing parent preferences with optional pre-extracted complex features."""
    parent_id: UUID
    original_text: str
    extracted_features: List[ExtractedPreference] = []
    complex_features: List[ExtractedComplexPreference] = []  # Pre-extracted by LLM
    session_id: str | None = None


class UserInteractionResponse(BaseModel):
    """User interaction response."""
    session_id: str
    interaction_id: str
    response_type: str  # "location_input", "confirmation", "choice"
    response_data: Dict[str, Any]


class StreamMessage(BaseModel):
    """Base stream message."""
    type: str
    session_id: str
    timestamp: int


class ProcessingUpdateMessage(StreamMessage):
    """Processing progress update."""
    type: str = "processing_update"
    stage: str
    progress: float
    details: str


class ComplexFeatureDetectedMessage(StreamMessage):
    """Complex feature detected that needs processing."""
    type: str = "complex_feature_detected"
    feature_key: str
    complex_type: str
    description: str
    requires_user_input: bool


class UserInteractionMessage(StreamMessage):
    """User interaction required."""
    type: str = "user_interaction"
    interaction_id: str
    feature_key: str
    interaction_type: str  # "location_input", "confirmation", "choice", "clarification"
    question: str
    field_name: str
    options: List[Dict[str, Any]] | None = None
    context: Dict[str, Any]


class PreferenceProcessingResultMessage(StreamMessage):
    """Final preference processing results."""
    type: str = "preference_processing_complete"
    processed_preferences: List[Dict[str, Any]]
    complex_features_processed: List[str]
    success: bool
    message: str | None = None


class ErrorMessage(StreamMessage):
    """Error during processing."""
    type: str = "error"
    error_code: str
    message: str


# Global session storage (use Redis in production)
preference_sessions: Dict[str, Dict[str, Any]] = {}


@router.post("/process/interactive")
async def process_preferences_interactive(
    request: ProcessPreferencesRequest,
    background_tasks: BackgroundTasks,
    # semantic_service: SemanticMatchingService = Depends(get_semantic_service),
    # db_repository: ComplexPreferenceRepository = Depends(get_db_repository),
    # agent_registry: ComplexTypeAgentRegistry = Depends(get_agent_registry)
):
    """Process parent preferences with interactive complex feature resolution."""
    
    session_id = request.session_id or f"pref_session_{request.parent_id}_{asyncio.get_event_loop().time()}"
    
    # Initialize session
    preference_sessions[session_id] = {
        "parent_id": request.parent_id,
        "request": request.dict(),
        "interactions": {},
        "queue": asyncio.Queue(),
        "completed": False,
        "results": None,
        "start_time": asyncio.get_event_loop().time()
    }
    
    # Start background processing
    background_tasks.add_task(
        process_preferences_with_interactions,
        session_id,
        request
    )
    
    return StreamingResponse(
        stream_preference_responses(session_id),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type",
        }
    )


async def stream_preference_responses(session_id: str):
    """Stream preference processing responses."""
    session = preference_sessions.get(session_id)
    if not session:
        yield f"data: {json.dumps({'type': 'error', 'message': 'Session not found'})}\n\n"
        return
    
    queue = session["queue"]
    
    try:
        while not session["completed"]:
            try:
                message = await asyncio.wait_for(queue.get(), timeout=30.0)
                yield f"data: {json.dumps(message.dict())}\n\n"
                
                if message.type in ["preference_processing_complete", "error"]:
                    session["completed"] = True
                    break
                    
            except asyncio.TimeoutError:
                yield f"data: {json.dumps({'type': 'keepalive', 'session_id': session_id})}\n\n"
                
    except Exception as e:
        logger.error(f"Preference streaming error for session {session_id}: {e}")
        error_msg = ErrorMessage(
            session_id=session_id,
            timestamp=int(asyncio.get_event_loop().time()),
            error_code="streaming_error",
            message=str(e)
        )
        yield f"data: {json.dumps(error_msg.dict())}\n\n"
    
    finally:
        # Cleanup after delay
        await asyncio.sleep(300)
        if session_id in preference_sessions:
            del preference_sessions[session_id]


async def process_preferences_with_interactions(
    session_id: str,
    request: ProcessPreferencesRequest
):
    """Process preferences with interactive complex feature resolution."""
    session = preference_sessions[session_id]
    queue = session["queue"]
    
    try:
        # Stage 1: Handle pre-extracted complex features vs detect from simple features
        await queue.put(ProcessingUpdateMessage(
            session_id=session_id,
            timestamp=int(asyncio.get_event_loop().time()),
            stage="analyzing_features",
            progress=0.1,
            details=f"Processing {len(request.complex_features)} pre-extracted complex features and {len(request.extracted_features)} simple features"
        ))
        
        complex_features = []
        processed_preferences = []
        
        # Handle pre-extracted complex features (from LLM)
        for complex_feature in request.complex_features:
            complex_features.append({
                "feature": complex_feature,
                "complex_type": complex_feature.type_name,
                "pre_extracted": True
            })
            
            await queue.put(ComplexFeatureDetectedMessage(
                session_id=session_id,
                timestamp=int(asyncio.get_event_loop().time()),
                feature_key=complex_feature.feature_key,
                complex_type=complex_feature.type_name,
                description=f"Pre-extracted {complex_feature.type_name}: {complex_feature.rationale}",
                requires_user_input=await requires_user_input_for_complex_feature(complex_feature)
            ))
        
        # Handle simple features that might be complex types
        for feature in request.extracted_features:
            complex_type = await detect_complex_type(feature)
            
            if complex_type:
                complex_features.append({
                    "feature": feature,
                    "complex_type": complex_type,
                    "pre_extracted": False
                })
                
                await queue.put(ComplexFeatureDetectedMessage(
                    session_id=session_id,
                    timestamp=int(asyncio.get_event_loop().time()),
                    feature_key=feature.feature_key,
                    complex_type=complex_type,
                    description=f"Detected {complex_type} constraint: {feature.raw_phrase}",
                    requires_user_input=await requires_user_input(feature, complex_type)
                ))
        
        # Stage 2: Process complex features with user interaction
        await queue.put(ProcessingUpdateMessage(
            session_id=session_id,
            timestamp=int(asyncio.get_event_loop().time()),
            stage="processing_complex_features",
            progress=0.3,
            details=f"Processing {len(complex_features)} complex preference types"
        ))
        
        for i, complex_feature in enumerate(complex_features):
            feature = complex_feature["feature"]
            complex_type = complex_feature["complex_type"]
            is_pre_extracted = complex_feature.get("pre_extracted", False)
            
            # Process this complex feature
            if is_pre_extracted:
                # Handle LLM pre-extracted complex preference
                processed_feature = await process_pre_extracted_complex_feature(
                    session_id, feature, queue
                )
            else:
                # Handle detected complex feature from simple text
                processed_feature = await process_complex_feature_interactive(
                    session_id, feature, complex_type, queue
                )
            processed_preferences.append(processed_feature)
            
            # Update progress
            progress = 0.3 + (i / len(complex_features)) * 0.5
            await queue.put(ProcessingUpdateMessage(
                session_id=session_id,
                timestamp=int(asyncio.get_event_loop().time()),
                stage="processing_complex_features",
                progress=progress,
                details=f"Processed {i+1}/{len(complex_features)} complex features"
            ))
        
        # Stage 3: Process simple features
        await queue.put(ProcessingUpdateMessage(
            session_id=session_id,
            timestamp=int(asyncio.get_event_loop().time()),
            stage="processing_simple_features",
            progress=0.8,
            details="Processing simple preference features"
        ))
        
        # Add simple features (non-complex)
        for feature in request.extracted_features:
            if not await detect_complex_type(feature):
                processed_preferences.append({
                    "feature_key": feature.feature_key,
                    "value_type": feature.value_type,
                    "preference": feature.preference,
                    "confidence": feature.confidence,
                    "raw_phrase": feature.raw_phrase,
                    "value_data": None,
                    "complex_value_type_id": None
                })
        
        # Stage 4: Save to database
        await queue.put(ProcessingUpdateMessage(
            session_id=session_id,
            timestamp=int(asyncio.get_event_loop().time()),
            stage="saving_preferences",
            progress=0.9,
            details="Saving processed preferences to database"
        ))
        
        await save_processed_preferences(request.parent_id, processed_preferences)
        
        # Send final results
        await queue.put(PreferenceProcessingResultMessage(
            session_id=session_id,
            timestamp=int(asyncio.get_event_loop().time()),
            processed_preferences=processed_preferences,
            complex_features_processed=[cf["complex_type"] for cf in complex_features],
            success=True
        ))
        
    except Exception as e:
        logger.error(f"Preference processing error for session {session_id}: {e}")
        await queue.put(ErrorMessage(
            session_id=session_id,
            timestamp=int(asyncio.get_event_loop().time()),
            error_code="processing_error",
            message=str(e)
        ))


async def process_pre_extracted_complex_feature(
    session_id: str,
    complex_feature: ExtractedComplexPreference,
    queue: asyncio.Queue
) -> Dict[str, Any]:
    """Process a pre-extracted complex feature (from LLM) with potential user interaction."""
    
    # Check if the LLM-generated value_data needs user input for completion
    needs_interaction = await check_pre_extracted_needs_interaction(complex_feature)
    
    if needs_interaction:
        # Even though LLM extracted it, we might need user input for missing data
        return await enhance_pre_extracted_with_user_input(session_id, complex_feature, queue)
    else:
        # LLM extraction is complete, just format for database storage
        complex_type_id = await get_complex_type_id(complex_feature.type_name)
        
        return {
            "feature_key": complex_feature.feature_key,
            "value_type": "complex",
            "preference": complex_feature.preference,
            "confidence": complex_feature.confidence,
            "raw_phrase": complex_feature.rationale,  # Use rationale as raw_phrase
            "value_data": complex_feature.value_data,
            "complex_value_type_id": complex_type_id
        }


async def enhance_pre_extracted_with_user_input(
    session_id: str,
    complex_feature: ExtractedComplexPreference,
    queue: asyncio.Queue
) -> Dict[str, Any]:
    """Enhance LLM pre-extracted feature with user input where needed."""
    
    value_data = complex_feature.value_data.copy()
    
    if complex_feature.type_name == "location_distance":
        # Check if locations need geocoding
        locations = value_data.get("locations", [])
        
        for location in locations:
            if location.get("name") in ["work", "office"] and not location.get("latitude"):
                # Need work location
                interaction_id = f"work_location_{len(preference_sessions[session_id]['interactions'])}"
                
                await queue.put(UserInteractionMessage(
                    session_id=session_id,
                    timestamp=int(asyncio.get_event_loop().time()),
                    interaction_id=interaction_id,
                    feature_key=complex_feature.feature_key,
                    interaction_type="location_input",
                    question=f"Please provide the address for '{location.get('name', 'location')}'",
                    field_name=f"{location.get('name', 'location')}_address",
                    context={
                        "feature_key": complex_feature.feature_key,
                        "location_name": location.get("name"),
                        "llm_rationale": complex_feature.rationale
                    }
                ))
                
                # Wait for user response
                response = await wait_for_user_response(session_id, interaction_id)
                if response:
                    if "address" in response:
                        coords = await geocode_address(response["address"])
                        if coords:
                            location.update(coords)
                            location["address"] = response["address"]
    
    complex_type_id = await get_complex_type_id(complex_feature.type_name)
    
    return {
        "feature_key": complex_feature.feature_key,
        "value_type": "complex",
        "preference": complex_feature.preference,
        "confidence": complex_feature.confidence,
        "raw_phrase": complex_feature.rationale,
        "value_data": value_data,
        "complex_value_type_id": complex_type_id
    }


async def process_complex_feature_interactive(
    session_id: str,
    feature: ExtractedPreference,
    complex_type: str,
    queue: asyncio.Queue
) -> Dict[str, Any]:
    """Process a single complex feature with user interaction (detected from simple text)."""
    
    if complex_type == "location_distance":
        return await process_location_distance_feature(session_id, feature, queue)
    elif complex_type == "schedule_range":
        return await process_schedule_range_feature(session_id, feature, queue)
    elif complex_type == "educational_approach":
        return await process_educational_approach_feature(session_id, feature, queue)
    else:
        # Unknown complex type, treat as simple
        return {
            "feature_key": feature.feature_key,
            "value_type": "text",
            "preference": feature.preference,
            "confidence": feature.confidence,
            "raw_phrase": feature.raw_phrase,
            "value_data": None,
            "complex_value_type_id": None
        }


async def process_location_distance_feature(
    session_id: str,
    feature: ExtractedPreference,
    queue: asyncio.Queue
) -> Dict[str, Any]:
    """Process location distance feature with user interaction."""
    
    # Parse the raw phrase to extract location requirements
    # Example: "5 km from my home" or "15 minutes walking from work"
    locations = await parse_location_requirements(feature.raw_phrase)
    
    # Check if we need additional user input
    for location in locations:
        if location.get("name") == "work" and "latitude" not in location and "address" not in location:
            # Need work location
            interaction_id = f"work_location_{len(preference_sessions[session_id]['interactions'])}"
            
            await queue.put(UserInteractionMessage(
                session_id=session_id,
                timestamp=int(asyncio.get_event_loop().time()),
                interaction_id=interaction_id,
                feature_key=feature.feature_key,
                interaction_type="location_input",
                question="Where is your work location?",
                field_name="work_address",
                context={
                    "feature_key": feature.feature_key,
                    "location_name": "work",
                    "required_for": "distance_calculation"
                }
            ))
            
            # Wait for user response
            response = await wait_for_user_response(session_id, interaction_id)
            if response:
                if "address" in response:
                    # Geocode the address
                    coords = await geocode_address(response["address"])
                    if coords:
                        location.update(coords)
                        location["address"] = response["address"]
                elif "latitude" in response and "longitude" in response:
                    location["latitude"] = response["latitude"]
                    location["longitude"] = response["longitude"]
    
    # Get the location distance complex type ID
    complex_type_id = await get_complex_type_id("location_distance")
    
    return {
        "feature_key": feature.feature_key,
        "value_type": "complex",
        "preference": feature.preference,
        "confidence": feature.confidence,
        "raw_phrase": feature.raw_phrase,
        "value_data": {"locations": locations},
        "complex_value_type_id": complex_type_id
    }


async def process_schedule_range_feature(
    session_id: str,
    feature: ExtractedPreference,
    queue: asyncio.Queue
) -> Dict[str, Any]:
    """Process schedule range feature."""
    
    # Parse schedule from raw phrase
    schedule_data = await parse_schedule_requirements(feature.raw_phrase)
    
    # May need clarification for ambiguous times
    if schedule_data.get("needs_clarification"):
        interaction_id = f"schedule_clarify_{len(preference_sessions[session_id]['interactions'])}"
        
        await queue.put(UserInteractionMessage(
            session_id=session_id,
            timestamp=int(asyncio.get_event_loop().time()),
            interaction_id=interaction_id,
            feature_key=feature.feature_key,
            interaction_type="clarification",
            question=f"Did you mean {schedule_data['interpretation']}?",
            field_name="schedule_confirmation",
            options=[
                {"value": "yes", "label": "Yes, that's correct"},
                {"value": "no", "label": "No, let me specify differently"}
            ],
            context={"schedule_data": schedule_data}
        ))
        
        response = await wait_for_user_response(session_id, interaction_id)
        if response and response.get("selected_option", {}).get("value") == "no":
            # Handle re-specification (simplified for example)
            schedule_data = {"start_time": "16:00", "end_time": "18:00", "flexibility_minutes": 30}
    
    complex_type_id = await get_complex_type_id("schedule_range")
    
    return {
        "feature_key": feature.feature_key,
        "value_type": "complex",
        "preference": feature.preference,
        "confidence": feature.confidence,
        "raw_phrase": feature.raw_phrase,
        "value_data": schedule_data,
        "complex_value_type_id": complex_type_id
    }


async def process_educational_approach_feature(
    session_id: str,
    feature: ExtractedPreference,
    queue: asyncio.Queue
) -> Dict[str, Any]:
    """Process educational approach feature."""
    
    # Extract approaches from raw phrase
    approaches = await extract_educational_approaches(feature.raw_phrase)
    
    complex_type_id = await get_complex_type_id("educational_approach")
    
    return {
        "feature_key": feature.feature_key,
        "value_type": "complex",
        "preference": feature.preference,
        "confidence": feature.confidence,
        "raw_phrase": feature.raw_phrase,
        "value_data": {
            "approaches": approaches,
            "must_have_all": False,
            "certification_required": "certification" in feature.raw_phrase.lower()
        },
        "complex_value_type_id": complex_type_id
    }


@router.post("/interactions/respond")
async def respond_to_preference_interaction(response: UserInteractionResponse):
    """Handle user response to preference processing interaction."""
    
    session = preference_sessions.get(response.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session["interactions"][response.interaction_id] = {
        "response_type": response.response_type,
        "response_data": response.response_data,
        "timestamp": asyncio.get_event_loop().time()
    }
    
    return {"status": "received", "interaction_id": response.interaction_id}


# Helper functions (mock implementations)
async def detect_complex_type(feature: ExtractedPreference) -> str | None:
    """Detect if this feature represents a complex preference type."""
    phrase = feature.raw_phrase.lower()
    
    if any(word in phrase for word in ["km", "distance", "minutes", "walking", "driving", "home", "work"]):
        return "location_distance"
    elif any(word in phrase for word in ["time", "schedule", "pickup", "drop", "hours"]):
        return "schedule_range"
    elif any(word in phrase for word in ["montessori", "waldorf", "reggio", "approach", "pedagogy"]):
        return "educational_approach"
    
    return None


async def requires_user_input(feature: ExtractedPreference, complex_type: str) -> bool:
    """Check if this complex feature requires user input."""
    if complex_type == "location_distance":
        return "work" in feature.raw_phrase.lower() or "office" in feature.raw_phrase.lower()
    return False


async def requires_user_input_for_complex_feature(complex_feature: ExtractedComplexPreference) -> bool:
    """Check if a pre-extracted complex feature requires user input."""
    return await check_pre_extracted_needs_interaction(complex_feature)


async def check_pre_extracted_needs_interaction(complex_feature: ExtractedComplexPreference) -> bool:
    """Check if LLM pre-extracted complex feature needs user interaction."""
    
    if complex_feature.type_name == "location_distance":
        # Check if any locations are missing coordinates or addresses
        locations = complex_feature.value_data.get("locations", [])
        for location in locations:
            # If location name is "work" or similar but no coordinates/address provided
            if (location.get("name", "").lower() in ["work", "office", "job"] and 
                not location.get("latitude") and not location.get("address")):
                return True
            # If location has vague reference
            if (location.get("name", "").lower() in ["there", "that place", "where"] and
                not location.get("latitude") and not location.get("address")):
                return True
    
    elif complex_feature.type_name == "schedule_range":
        # Check if schedule has ambiguous or missing time information
        schedule_data = complex_feature.value_data
        if (not schedule_data.get("start_time") or not schedule_data.get("end_time") or
            "flexible" in str(schedule_data).lower() or "varies" in str(schedule_data).lower()):
            return True
    
    elif complex_feature.type_name == "social_connection":
        # Check if entity_ids are missing or placeholder
        entity_ids = complex_feature.value_data.get("entity_ids", [])
        if not entity_ids or any(str(id).startswith("placeholder") for id in entity_ids):
            return True
    
    return False


async def parse_location_requirements(raw_phrase: str) -> List[Dict[str, Any]]:
    """Parse location requirements from raw phrase."""
    locations = []
    phrase = raw_phrase.lower()
    
    if "home" in phrase:
        locations.append({
            "name": "home",
            "max_distance_km": 5.0,  # Default, would extract from phrase
            "preferred_distance_km": 3.0
        })
    
    if "work" in phrase or "office" in phrase:
        locations.append({
            "name": "work",
            "max_travel_minutes": 20,  # Default, would extract from phrase
            "transport_mode": "walking"
        })
    
    return locations


async def parse_schedule_requirements(raw_phrase: str) -> Dict[str, Any]:
    """Parse schedule requirements from raw phrase."""
    return {
        "start_time": "16:00",
        "end_time": "18:00",
        "flexibility_minutes": 30,
        "days_of_week": ["monday", "tuesday", "wednesday", "thursday", "friday"]
    }


async def extract_educational_approaches(raw_phrase: str) -> List[str]:
    """Extract educational approaches from raw phrase."""
    approaches = []
    phrase = raw_phrase.lower()
    
    if "montessori" in phrase:
        approaches.append("montessori")
    if "waldorf" in phrase:
        approaches.append("waldorf")
    if "reggio" in phrase:
        approaches.append("reggio_emilia")
    
    return approaches


async def geocode_address(address: str) -> Dict[str, float] | None:
    """Geocode address using OpenStreetMap."""
    # Implementation would use the same OpenStreetMap logic from plugins.py
    return {"latitude": 52.5096, "longitude": 13.3765}  # Mock


async def get_complex_type_id(type_name: str) -> UUID:
    """Get complex value type ID from database."""
    # Mock implementation
    from uuid import uuid4
    return uuid4()


async def save_processed_preferences(parent_id: UUID, preferences: List[Dict[str, Any]]):
    """Save processed preferences to database."""
    # Implementation would save to matching.Parent_Preferences
    pass


async def wait_for_user_response(session_id: str, interaction_id: str, timeout: int = 300) -> Dict[str, Any] | None:
    """Wait for user response to interaction."""
    session = preference_sessions[session_id]
    
    start_time = asyncio.get_event_loop().time()
    while asyncio.get_event_loop().time() - start_time < timeout:
        if interaction_id in session["interactions"]:
            return session["interactions"][interaction_id]["response_data"]
        await asyncio.sleep(1)
    
    return None