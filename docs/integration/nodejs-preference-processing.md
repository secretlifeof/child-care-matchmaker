# Node.js Integration Guide: Enhanced Preference Processing

This guide shows how to integrate your Node.js application with the enhanced preference processing system that handles complex preferences with user interactions.

## Architecture Overview

```
Main App (Node.js) → Preference Processing → Matching
     ↓                      ↓                   ↓
  Extract Text      → Complex Detection  → Use Processed Data
  Basic Prefs       → User Interaction   → Return Results
                    → Store in DB        
```

The key change: **Complex preference processing happens during preference extraction, not during matching**.

## Updated processParentPreferences Method

Here's how to update your existing method:

```typescript
interface ExtractedPreference {
    feature_key: string;
    confidence: number;
    preference: 'required' | 'preferred' | 'nice_to_have' | 'avoid';
    raw_phrase: string;
}

interface ExtractedComplexPreference {
    feature_key: string;
    type_name: string; // e.g., "location_distance", "schedule_range"
    value_data: Record<string, any>; // LLM-generated structured data
    preference: 'required' | 'preferred' | 'nice_to_have' | 'avoid';
    confidence: number;
    rationale: string; // LLM explanation
}

interface ComplexPreferenceDetection {
    success: boolean;
    message?: string;
    complex_features_detected: Array<{
        feature_key: string;
        complex_type: string;
        raw_phrase: string;
        requires_interaction: boolean;
    }>;
    recommendation?: {
        use_endpoint: string;
        method: string;
        reason: string;
        example_usage: any;
    };
}

class PreferenceProcessor {
    
    async processParentPreferences(
        parentId: string, 
        originalText: string, 
        extractedPreferences: ExtractedPreference[],
        complexPreferences: ExtractedComplexPreference[]  // Pre-extracted by LLM (required)
    ) {
        console.log(`🤖 Processing ${complexPreferences.length} LLM-extracted complex preferences`);
        
        return await this.processComplexPreferencesInteractively(
            parentId, originalText, extractedPreferences, complexPreferences
        );
    }
    
    
    private async processComplexPreferencesInteractively(
        parentId: string,
        originalText: string,
        extractedPreferences: ExtractedPreference[],
        complexPreferences: ExtractedComplexPreference[]
    ) {
        
        const client = new PreferenceProcessingStreamClient(this.config.baseUrl);
        
        try {
            const result = await client.processPreferences(parentId, {
                original_text: originalText,
                extracted_features: extractedPreferences,
                complex_features: complexPreferences  // LLM pre-extracted complex features
            });
            
            this.logger.info(`✅ Complex preferences processed successfully`);
            return {
                success: true,
                processed_preferences: result.processed_preferences,
                complex_features_processed: result.complex_features_processed,
                processing_type: 'interactive_complex',
                llm_extracted_count: complexPreferences.length
            };
            
        } catch (error) {
            this.logger.error('❌ Complex preference processing failed:', error);
            throw error;
        }
    }
}
```

## Streaming Client for Complex Preferences

```typescript
interface ProcessPreferencesRequest {
    parent_id: string;
    original_text: string;
    extracted_features: ExtractedPreference[];
    complex_features: ExtractedComplexPreference[];  // LLM pre-extracted (required)
    session_id?: string;
}

interface ProcessingResult {
    processed_preferences: Array<{
        feature_key: string;
        value_type: string;
        preference: string;
        confidence: number;
        raw_phrase: string;
        value_data?: any;
        complex_value_type_id?: string;
    }>;
    complex_features_processed: string[];
}

class PreferenceProcessingStreamClient {
    constructor(private baseUrl: string) {
        this.rl = readline.createInterface({
            input: process.stdin,
            output: process.stdout
        });
    }

    async processPreferences(
        parentId: string, 
        request: Omit<ProcessPreferencesRequest, 'parent_id'>
    ): Promise<ProcessingResult> {
        
        const requestData: ProcessPreferencesRequest = {
            parent_id: parentId,
            ...request,
            session_id: `pref_${parentId}_${Date.now()}`
        };

        console.log(`🔄 Starting preference processing for parent ${parentId}`);
        
        const response = await fetch(`${this.baseUrl}/api/preferences/process/interactive`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'text/event-stream',
            },
            body: JSON.stringify(requestData)
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        return await this.handleStreamingResponse(response, requestData.session_id);
    }

    private async handleStreamingResponse(
        response: Response, 
        sessionId: string
    ): Promise<ProcessingResult> {
        
        const reader = response.body!.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let result: ProcessingResult | null = null;

        try {
            while (true) {
                const { value, done } = await reader.read();
                
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const data = line.slice(6);
                        if (data.trim()) {
                            try {
                                const message = JSON.parse(data);
                                const messageResult = await this.handleMessage(message, sessionId);
                                if (messageResult) {
                                    result = messageResult;
                                }
                            } catch (e) {
                                console.error('Failed to parse message:', data);
                            }
                        }
                    }
                }
            }
        } finally {
            reader.releaseLock();
        }

        if (!result) {
            throw new Error('No processing result received');
        }

        return result;
    }

    private async handleMessage(message: any, sessionId: string): Promise<ProcessingResult | null> {
        switch (message.type) {
            case 'processing_update':
                console.log(`📊 ${message.stage}: ${(message.progress * 100).toFixed(1)}%`);
                console.log(`   ${message.details}`);
                break;
            
            case 'complex_feature_detected':
                console.log(`🔍 Complex feature detected: ${message.complex_type}`);
                console.log(`   "${message.description}"`);
                if (message.requires_user_input) {
                    console.log(`   ⚠️  Will require user input`);
                }
                break;
            
            case 'user_interaction':
                await this.handleUserInteraction(message, sessionId);
                break;
            
            case 'preference_processing_complete':
                console.log(`✅ Processing complete!`);
                console.log(`   Processed ${message.processed_preferences.length} preferences`);
                console.log(`   Complex types: ${message.complex_features_processed.join(', ')}`);
                return {
                    processed_preferences: message.processed_preferences,
                    complex_features_processed: message.complex_features_processed
                };
            
            case 'error':
                console.error(`❌ Processing error: ${message.message}`);
                throw new Error(message.message);
        }
        
        return null;
    }

    private async handleUserInteraction(message: any, sessionId: string): Promise<void> {
        console.log(`\n❓ ${message.question}`);

        let responseData = {};

        switch (message.interaction_type) {
            case 'location_input':
                responseData = await this.getLocationInput(message);
                break;
            case 'clarification':
                responseData = await this.getConfirmation(message);
                break;
            default:
                console.log('Unknown interaction type:', message.interaction_type);
                return;
        }

        // Send response
        await this.sendInteractionResponse(sessionId, message.interaction_id, message.interaction_type, responseData);
    }

    private async getLocationInput(message: any): Promise<any> {
        console.log('\n📍 Please provide the location:');
        console.log('   Examples: "Potsdamer Platz 1, Berlin" or "52.5096,13.3765"');
        
        const input = await this.rl.question('Enter location: ');

        // Check if it's coordinates
        if (input.includes(',') && input.split(',').length === 2) {
            const [lat, lng] = input.split(',').map(s => parseFloat(s.trim()));
            if (!isNaN(lat) && !isNaN(lng)) {
                return { latitude: lat, longitude: lng, source: 'user_coordinates' };
            }
        }

        // Treat as address
        return { address: input.trim(), source: 'user_address' };
    }

    private async getConfirmation(message: any): Promise<any> {
        if (message.options) {
            console.log('\nOptions:');
            message.options.forEach((option: any, index: number) => {
                console.log(`   ${index + 1}. ${option.label}`);
            });
            
            const choice = await this.rl.question('Select option (number): ');
            const choiceIndex = parseInt(choice) - 1;
            
            if (choiceIndex >= 0 && choiceIndex < message.options.length) {
                return { selected_option: message.options[choiceIndex] };
            }
        }
        
        return { selected_option: null };
    }

    private async sendInteractionResponse(
        sessionId: string, 
        interactionId: string, 
        responseType: string, 
        responseData: any
    ): Promise<void> {
        
        const response = await fetch(`${this.baseUrl}/api/preferences/interactions/respond`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: sessionId,
                interaction_id: interactionId,
                response_type: responseType,
                response_data: responseData
            })
        });

        if (!response.ok) {
            throw new Error(`Failed to send interaction response: ${response.statusText}`);
        }

        console.log('✅ Response sent successfully\n');
    }
}
```

## Usage Examples

### Processing LLM Pre-Extracted Complex Preferences

```typescript
// Your existing LLM extraction
const complexExtractionResult = await context.llmClient.generateObject({
    promptPath: 'system/shared/complex-feature-extraction',
    schema: complexExtractionSchema,
    templateValues: {
        input_role: 'parent_preference',
        input_text: context.preferenceText,
        complex_types: context.complexTypes.map((t) => ({ 
            type_name: t.type_name, 
            description: t.description 
        })),
        schemas: context.complexTypes.reduce(
            (acc, t) => ({ ...acc, [t.type_name]: { schema: t.schema, examples: t.examples } }),
            {},
        ),
    },
});

// Convert LLM result to our format
const complexPreferences: ExtractedComplexPreference[] = complexExtractionResult.object.complex_items.map(item => ({
    feature_key: item.feature_key,
    type_name: item.type_name,
    value_data: item.value_data,
    preference: item.preference,
    confidence: item.confidence,
    rationale: item.rationale
}));

// Process with pre-extracted complex preferences
const result = await processor.processParentPreferences(
    'parent-uuid',
    'I need a daycare within 10 minutes walking from home and 15 minutes by bike from my work',
    [], // Simple preferences (if any)
    complexPreferences // LLM pre-extracted complex preferences
);

// Console output:
// 🤖 Processing 1 LLM-extracted complex preferences
// 🔍 Complex feature detected: location_distance
//    Pre-extracted location_distance: Extracted location constraints for home and work with specific travel times and transport modes
//    ⚠️  Will require user input
// ❓ Please provide the address for 'work'
// [User provides work address]
// ✅ Processing complete!

console.log(result.llm_extracted_count); // 1
console.log(result.processed_preferences[0].value_data);
// {
//   "locations": [
//     {
//       "name": "home",
//       "max_travel_minutes": 10,
//       "transport_mode": "walking"
//     },
//     {
//       "name": "work",
//       "max_travel_minutes": 15,
//       "transport_mode": "bike",
//       "latitude": 52.5096,
//       "longitude": 13.3765,
//       "address": "Potsdamer Platz 1, Berlin"
//     }
//   ]
// }
```

### Example LLM Extraction Result

Your LLM might extract something like this:

```javascript
// LLM extracts this from: "I need it close to home and work, max 10 min walk from home, 15 min bike from work"
{
    complex_items: [
        {
            feature_key: "location_proximity",
            type_name: "location_distance",
            value_data: {
                locations: [
                    {
                        name: "home",
                        max_travel_minutes: 10,
                        preferred_travel_minutes: 7,
                        transport_mode: "walking"
                    },
                    {
                        name: "work",
                        max_travel_minutes: 15,
                        preferred_travel_minutes: 12,
                        transport_mode: "bike"
                        // Note: no latitude/longitude - will trigger user interaction
                    }
                ]
            },
            preference: "required",
            confidence: 0.92,
            rationale: "User specified specific travel time constraints from home (10min walk) and work (15min bike)"
        }
    ]
}
```

The service will:
1. **Accept the LLM extraction** as-is
2. **Detect missing data** (work location has no coordinates)
3. **Stream user interaction** to get work address
4. **Geocode with MapBox** 
5. **Store complete preference** with all data filled in

## Integration with Matching

After preferences are processed, matching becomes much simpler:

```typescript
// Matching now just uses the pre-processed complex preferences
const matches = await fetch(`${baseUrl}/api/matches/recommend`, {
    method: 'POST',
    body: JSON.stringify({
        parent_id: parentId,
        limit: 10,
        max_distance_km: 15.0
    })
});

const results = await matches.json();
// Results include scores from complex preferences that were already processed and stored
```

## Benefits

1. **Clean Separation**: Preference processing vs matching are separate concerns
2. **Better UX**: User interactions happen during preference setting, not during search
3. **Performance**: Matching is faster since complex processing is already done
4. **Reliability**: User data is collected once and stored, not requested repeatedly
5. **Flexibility**: Can process preferences offline, batch process, etc.

## Error Handling

```typescript
try {
    const result = await processor.processParentPreferences(parentId, text, prefs);
    return result;
} catch (error) {
    if (error.message.includes('user_interaction_timeout')) {
        // Handle timeout - maybe retry or use defaults
        console.log('User interaction timed out, using default preferences');
    } else if (error.message.includes('geocoding_failed')) {
        // Handle geocoding failure
        console.log('Could not find location, please provide coordinates');
    }
    throw error;
}
```

This architecture provides a much cleaner separation between preference collection/processing and the actual matching algorithm.