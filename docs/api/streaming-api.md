# Streaming API Documentation

The streaming API enables real-time, interactive complex preference processing with user input collection during matching.

## Overview

Traditional matching APIs process all data synchronously and return final results. The streaming API allows the system to:

1. **Start processing immediately** with available data
2. **Request user input** when needed (e.g., missing work location)
3. **Continue processing** with user-provided data
4. **Stream progress updates** in real-time
5. **Return results** when complete

This is especially useful for complex preferences that may require additional user data.

## Architecture

```
Client (Node.js v22)     →    FastAPI Streaming Endpoint
        ↓                              ↓
   Start Request         →    Background Processing
        ↓                              ↓
  Stream Listener       ←    Progress Updates
        ↓                              ↓
  User Interaction      ←    Interaction Request
        ↓                              ↓
  Send Response         →    Continue Processing
        ↓                              ↓  
  Receive Results       ←    Final Results
```

## API Endpoints

### 1. Start Interactive Matching

**POST** `/api/streaming/matches/interactive`

Starts an interactive matching session with streaming responses.

**Request Body:**
```json
{
  "parent_id": "550e8400-e29b-41d4-a716-446655440001",
  "limit": 10,
  "max_distance_km": 15.0,
  "include_full_centers": false,
  "include_explanations": true,
  "session_id": "optional-custom-session-id"
}
```

**Response:** 
- **Content-Type:** `text/event-stream`
- **Format:** Server-Sent Events (SSE)

### 2. Respond to User Interaction

**POST** `/api/streaming/interactions/respond`

Sends user response to an interaction request.

**Request Body:**
```json
{
  "session_id": "session_uuid_123456",
  "interaction_id": "interaction_0",
  "response_type": "location_input",
  "response_data": {
    "address": "Potsdamer Platz 1, Berlin",
    "source": "user_address"
  }
}
```

**Response:**
```json
{
  "status": "received",
  "interaction_id": "interaction_0"
}
```

### 3. Get Session Status

**GET** `/api/streaming/sessions/{session_id}/status`

Gets current status of an active session.

**Response:**
```json
{
  "session_id": "session_uuid_123456",
  "completed": false,
  "interactions_count": 1,
  "has_results": false
}
```

## Stream Message Types

All messages follow this base structure:
```json
{
  "type": "message_type",
  "session_id": "session_uuid_123456", 
  "timestamp": 1699123456789
}
```

### 1. Processing Update

Indicates processing progress:

```json
{
  "type": "processing_update",
  "session_id": "session_uuid_123456",
  "timestamp": 1699123456789,
  "stage": "loading_preferences",
  "progress": 0.25,
  "details": "Loading parent preferences from database"
}
```

**Stages:**
- `loading_preferences` - Loading parent data from database
- `finding_centers` - Searching for nearby centers  
- `processing_complex_preferences` - Processing complex constraints
- `generating_results` - Finalizing match results

### 2. User Interaction Required

Requests user input to continue processing:

```json
{
  "type": "user_interaction",
  "session_id": "session_uuid_123456", 
  "timestamp": 1699123456789,
  "interaction_id": "interaction_0",
  "interaction_type": "location_input",
  "question": "Where is your work located?",
  "field_name": "work_location",
  "context": {
    "preference_id": "pref_uuid_123",
    "required_for": "distance_calculation"
  }
}
```

**Interaction Types:**

#### Location Input (`location_input`)
- **Question:** Request for location data
- **Expected Response:** Address or coordinates
- **Response Data:**
  ```json
  {
    "address": "Potsdamer Platz 1, Berlin",
    "source": "user_address"
  }
  ```
  OR
  ```json
  {
    "latitude": 52.5096,
    "longitude": 13.3765,
    "source": "user_coordinates"
  }
  ```

#### Confirmation (`confirmation`)
- **Question:** Yes/No confirmation
- **Expected Response:** Boolean confirmation
- **Response Data:**
  ```json
  {
    "confirmed": true
  }
  ```

#### Choice Selection (`choice`)
- **Question:** Multiple choice selection
- **Options:** Array of choices provided in message
- **Response Data:**
  ```json
  {
    "selected_option": {
      "value": "option_1",
      "label": "First Option"
    }
  }
  ```

### 3. Match Results

Final matching results:

```json
{
  "type": "match_results",
  "session_id": "session_uuid_123456",
  "timestamp": 1699123456789,
  "success": true,
  "offers": [
    {
      "center_id": "center_uuid_456",
      "center_name": "Green Garden Daycare",
      "score": 0.87,
      "preference_matches": {
        "complex_preferences": {
          "location_distance": {
            "satisfied": true,
            "details": "home: ideal walking time (8min); work: acceptable bike time (18min)",
            "score": 0.95,
            "processing_metadata": {
              "pipeline_used": true,
              "pipeline_name": "location_resolver->distance_calculator->location_scorer",
              "execution_times": [120, 45, 30]
            }
          }
        }
      }
    }
  ],
  "processing_details": {
    "centers_evaluated": 45,
    "complex_types_processed": ["location_distance"],
    "processing_time_ms": 2340,
    "interactions_required": 1
  }
}
```

### 4. Error Message

Processing errors:

```json
{
  "type": "error",
  "session_id": "session_uuid_123456",
  "timestamp": 1699123456789,
  "error_code": "processing_error",
  "message": "Failed to process location constraint",
  "details": {
    "parent_id": "550e8400-e29b-41d4-a716-446655440001",
    "failed_stage": "processing_complex_preferences"
  }
}
```

### 5. Keepalive

Connection keepalive (no action required):

```json
{
  "type": "keepalive",
  "session_id": "session_uuid_123456"
}
```

## Node.js v22 Client Integration

### Basic Usage

```javascript
import { MatchingStreamClient } from './nodejs-streaming-client.js';

const client = new MatchingStreamClient('http://localhost:8001');

// Start interactive matching
await client.startMatching('550e8400-e29b-41d4-a716-446655440001', {
    limit: 5,
    maxDistanceKm: 15.0,
    includeExplanations: true
});
```

### Advanced Usage with Custom Handlers

```javascript
class CustomMatchingClient extends MatchingStreamClient {
    
    async handleLocationInput(message) {
        // Custom location input logic
        if (message.field_name === 'work_location') {
            // Check local cache first
            const cachedLocation = await this.getCachedWorkLocation();
            if (cachedLocation) {
                return cachedLocation;
            }
        }
        
        // Fall back to parent implementation
        return super.handleLocationInput(message);
    }
    
    async handleMatchResults(message) {
        // Custom result processing
        await this.saveResultsToDatabase(message.offers);
        await this.notifyUser(message.offers);
        
        // Call parent for display
        await super.handleMatchResults(message);
    }
}
```

## Complex Preference Examples

### Travel Time Constraints

**Input:**
```json
{
  "locations": [
    {
      "name": "home",
      "max_travel_minutes": 15,
      "transport_mode": "walking"
    },
    {
      "name": "work", 
      "max_travel_minutes": 20,
      "transport_mode": "bike"
    }
  ]
}
```

**User Interaction if work location unknown:**
- System asks: "Where is your work located?"
- User provides: "Potsdamer Platz 1, Berlin"
- System geocodes via MapBox
- Processing continues with coordinates

### Multiple Location Constraints

**Input:**
```json
{
  "locations": [
    {
      "name": "home",
      "max_distance_km": 5
    },
    {
      "name": "grandparents",
      "address": "Unter den Linden 1, Berlin",
      "max_distance_km": 8
    },
    {
      "name": "work",
      "max_travel_minutes": 25,
      "transport_mode": "public_transport"
    }
  ]
}
```

## Error Handling

### Client-Side Error Handling

```javascript
try {
    await client.startMatching(parentId, options);
} catch (error) {
    if (error.code === 'SESSION_TIMEOUT') {
        console.log('Session timed out, please try again');
    } else if (error.code === 'NETWORK_ERROR') {
        console.log('Network error, retrying...');
        await client.startMatching(parentId, options);
    }
}
```

### Server-Side Error Codes

- `session_not_found` - Session ID not found or expired
- `processing_error` - Error during complex preference processing
- `geocoding_failed` - MapBox geocoding failed
- `user_interaction_timeout` - User didn't respond within timeout
- `invalid_response` - User provided invalid interaction response

## Rate Limiting & Performance

### MapBox Geocoding
- **Rate Limit:** 10 requests per second (configurable)
- **Monthly Limit:** 100,000 requests on free tier
- **Timeout:** 10 seconds per request
- **Fallback:** Graceful degradation if geocoding fails

### Session Management
- **Session Timeout:** 5 minutes after completion
- **Interaction Timeout:** 5 minutes for user response
- **Keepalive:** Every 30 seconds during processing

### Scalability
- **Production:** Use Redis for session storage
- **Load Balancing:** Session-aware routing required
- **Cleanup:** Automatic session cleanup after timeout

## Security Considerations

### CORS Configuration
```python
# Production CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)
```

### Session Security
- Use UUID4 for session IDs
- Implement session expiry
- Validate parent_id ownership
- Rate limit interaction endpoints

### Data Privacy
- Don't log user location data
- Clean up session data promptly  
- Use HTTPS in production
- Implement request authentication