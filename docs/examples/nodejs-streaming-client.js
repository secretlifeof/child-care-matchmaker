/**
 * Node.js v22 client for streaming complex preference matching
 * Uses Server-Sent Events (SSE) for real-time user interactions
 */

import { EventSource } from 'eventsource';
import readline from 'readline/promises';

class MatchingStreamClient {
    constructor(baseUrl = 'http://localhost:8001') {
        this.baseUrl = baseUrl;
        this.sessionId = null;
        this.eventSource = null;
        this.pendingInteractions = new Map();
        
        // Setup readline interface for user input
        this.rl = readline.createInterface({
            input: process.stdin,
            output: process.stdout
        });
    }

    /**
     * Start interactive matching session
     */
    async startMatching(parentId, options = {}) {
        const requestData = {
            parent_id: parentId,
            limit: options.limit || 10,
            max_distance_km: options.maxDistanceKm || 10.0,
            include_full_centers: options.includeFullCenters || false,
            include_explanations: options.includeExplanations || true,
            session_id: options.sessionId || `session_${parentId}_${Date.now()}`
        };

        this.sessionId = requestData.session_id;

        console.log(`🚀 Starting interactive matching for parent ${parentId}`);
        console.log(`📡 Session ID: ${this.sessionId}\n`);

        try {
            // Start streaming endpoint
            const response = await fetch(`${this.baseUrl}/api/streaming/matches/interactive`, {
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

            // Process streaming response
            await this.handleStreamingResponse(response);

        } catch (error) {
            console.error('❌ Matching failed:', error.message);
            return null;
        }
    }

    /**
     * Handle streaming response using fetch API streams (Node.js v22)
     */
    async handleStreamingResponse(response) {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        try {
            while (true) {
                const { value, done } = await reader.read();
                
                if (done) {
                    console.log('✅ Stream completed');
                    break;
                }

                // Decode chunk and add to buffer
                buffer += decoder.decode(value, { stream: true });

                // Process complete messages
                const lines = buffer.split('\n');
                buffer = lines.pop() || ''; // Keep incomplete line in buffer

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const data = line.slice(6); // Remove 'data: ' prefix
                        if (data.trim()) {
                            try {
                                const message = JSON.parse(data);
                                await this.handleMessage(message);
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
    }

    /**
     * Handle individual stream messages
     */
    async handleMessage(message) {
        switch (message.type) {
            case 'processing_update':
                await this.handleProcessingUpdate(message);
                break;
            
            case 'user_interaction':
                await this.handleUserInteraction(message);
                break;
            
            case 'match_results':
                await this.handleMatchResults(message);
                break;
            
            case 'error':
                await this.handleError(message);
                break;
            
            case 'keepalive':
                // Silent keepalive
                break;
            
            default:
                console.log('Unknown message type:', message.type);
        }
    }

    /**
     * Handle processing progress updates
     */
    async handleProcessingUpdate(message) {
        const progressBar = this.createProgressBar(message.progress);
        console.log(`📊 ${message.stage}: ${progressBar} ${(message.progress * 100).toFixed(1)}%`);
        console.log(`   ${message.details}\n`);
    }

    /**
     * Handle user interaction requests
     */
    async handleUserInteraction(message) {
        console.log('\n❓ User input required:');
        console.log(`   ${message.question}`);

        let responseData = {};

        switch (message.interaction_type) {
            case 'location_input':
                responseData = await this.handleLocationInput(message);
                break;
            
            case 'confirmation':
                responseData = await this.handleConfirmation(message);
                break;
            
            case 'choice':
                responseData = await this.handleChoice(message);
                break;
            
            default:
                console.log('Unknown interaction type:', message.interaction_type);
                return;
        }

        // Send response back to server
        await this.sendInteractionResponse(message.interaction_id, message.interaction_type, responseData);
    }

    /**
     * Handle location input
     */
    async handleLocationInput(message) {
        const fieldName = message.field_name;
        
        console.log(`\n📍 Please provide ${fieldName}:`);
        console.log('   You can enter:');
        console.log('   - Full address: "Potsdamer Platz 1, 10785 Berlin"');
        console.log('   - Coordinates: "52.5096,13.3765"');
        console.log('   - Partial address: "Potsdamer Platz, Berlin"');

        const input = await this.rl.question('Enter location: ');

        // Parse input
        if (input.includes(',') && input.split(',').length === 2) {
            // Assume coordinates
            const [lat, lng] = input.split(',').map(s => parseFloat(s.trim()));
            if (!isNaN(lat) && !isNaN(lng)) {
                return {
                    latitude: lat,
                    longitude: lng,
                    source: 'user_coordinates'
                };
            }
        }

        // Treat as address
        return {
            address: input.trim(),
            source: 'user_address'
        };
    }

    /**
     * Handle confirmation requests
     */
    async handleConfirmation(message) {
        const answer = await this.rl.question('Confirm? (y/n): ');
        return {
            confirmed: answer.toLowerCase().startsWith('y')
        };
    }

    /**
     * Handle choice requests
     */
    async handleChoice(message) {
        if (message.options && message.options.length > 0) {
            console.log('\nAvailable options:');
            message.options.forEach((option, index) => {
                console.log(`   ${index + 1}. ${option.label || option.value}`);
            });

            const choice = await this.rl.question('Select option (number): ');
            const choiceIndex = parseInt(choice) - 1;
            
            if (choiceIndex >= 0 && choiceIndex < message.options.length) {
                return {
                    selected_option: message.options[choiceIndex]
                };
            }
        }

        return { selected_option: null };
    }

    /**
     * Send interaction response to server
     */
    async sendInteractionResponse(interactionId, responseType, responseData) {
        try {
            const response = await fetch(`${this.baseUrl}/api/streaming/interactions/respond`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    session_id: this.sessionId,
                    interaction_id: interactionId,
                    response_type: responseType,
                    response_data: responseData
                })
            });

            if (!response.ok) {
                console.error('Failed to send interaction response:', response.statusText);
            } else {
                console.log('✅ Response sent successfully\n');
            }

        } catch (error) {
            console.error('Error sending interaction response:', error.message);
        }
    }

    /**
     * Handle final match results
     */
    async handleMatchResults(message) {
        console.log('\n🎯 Matching Results:');
        console.log(`   Found ${message.offers.length} matching centers`);
        console.log(`   Processing time: ${message.processing_details.processing_time_ms}ms`);
        
        if (message.processing_details.interactions_required) {
            console.log(`   User interactions required: ${message.processing_details.interactions_required}`);
        }

        console.log('\n📋 Top matches:');
        message.offers.forEach((offer, index) => {
            console.log(`\n${index + 1}. ${offer.center_name} (Score: ${(offer.score * 100).toFixed(1)}%)`);
            
            if (offer.preference_matches?.complex_preferences) {
                Object.entries(offer.preference_matches.complex_preferences).forEach(([type, match]) => {
                    const status = match.satisfied ? '✅' : '❌';
                    console.log(`   ${status} ${type}: ${match.details}`);
                });
            }
        });

        console.log('\n🏁 Matching completed!');
        this.cleanup();
    }

    /**
     * Handle error messages
     */
    async handleError(message) {
        console.error('\n❌ Error occurred:');
        console.error(`   Code: ${message.error_code}`);
        console.error(`   Message: ${message.message}`);
        
        if (message.details) {
            console.error(`   Details:`, message.details);
        }

        this.cleanup();
    }

    /**
     * Create simple progress bar
     */
    createProgressBar(progress, width = 20) {
        const filled = Math.round(progress * width);
        const empty = width - filled;
        return '█'.repeat(filled) + '░'.repeat(empty);
    }

    /**
     * Cleanup resources
     */
    cleanup() {
        if (this.rl) {
            this.rl.close();
        }
    }
}

// Example usage
async function main() {
    const client = new MatchingStreamClient('http://localhost:8001');
    
    // Example parent ID (UUID)
    const parentId = '550e8400-e29b-41d4-a716-446655440001';
    
    await client.startMatching(parentId, {
        limit: 5,
        maxDistanceKm: 15.0,
        includeExplanations: true
    });
}

// Run if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
    main().catch(console.error);
}

export { MatchingStreamClient };