# MapBox Integration Setup

This service uses MapBox for geocoding addresses to latitude/longitude coordinates during complex preference processing.

## Getting a MapBox API Key

1. **Create a MapBox Account**:
   - Visit [mapbox.com](https://www.mapbox.com/) 
   - Sign up for a free account

2. **Access Your API Keys**:
   - Go to your [MapBox Account Dashboard](https://account.mapbox.com/)
   - Navigate to "Access tokens"
   - Copy your "Default public token" or create a new token

3. **Configure Token Scopes** (Optional):
   - For geocoding, you need the `styles:read` and `geocoding:read` scopes
   - The default public token includes these scopes

## Environment Configuration

Add your MapBox API key to your environment variables:

```env
MAPBOX_API_KEY=pk.eyJ1IjoieW91cnVzZXJuYW1lIiwiYSI6ImNsb3JlczEyMzQ1NiJ9.example_token_here
```

### Docker Compose

Add the API key to your `docker-compose.yml`:

```yaml
services:
  matchmaker:
    environment:
      - MAPBOX_API_KEY=${MAPBOX_API_KEY}
```

### Local Development

Create a `.env` file in the project root:

```env
MAPBOX_API_KEY=pk.eyJ1IjoieW91cnVzZXJuYW1lIiwiYSI6ImNsb3JlczEyMzQ1NiJ9.example_token_here
```

## API Usage Limits

### Free Tier Limits
- **Geocoding**: 100,000 requests per month
- **Rate Limiting**: Our service implements 10 requests per second max
- **Geographic Coverage**: Global

### Paid Tiers
- Higher request limits
- Premium features like batch geocoding
- Better rate limits

## Service Features

### Geocoding Support
- **Forward Geocoding**: Address → Coordinates
- **Reverse Geocoding**: Coordinates → Address  
- **Country Filtering**: Focus on specific countries (default: Germany)
- **Confidence Scores**: Quality assessment of geocoding results

### Error Handling
- Automatic retry for rate limit errors
- Graceful fallback when geocoding fails
- Detailed logging for debugging

### Performance Optimizations
- Built-in rate limiting
- Connection pooling with aiohttp
- Configurable timeout handling
- Memory-efficient async processing

## Testing the Integration

### Using the Service Directly

```python
from src.matchmaker.services.map_service import get_map_service

# Initialize service
map_service = get_map_service("your_mapbox_api_key")

# Geocode an address
result = await map_service.geocode_address(
    address="Potsdamer Platz 1",
    city="Berlin", 
    country_code="de"
)

if result:
    print(f"Coordinates: {result.latitude}, {result.longitude}")
    print(f"Formatted: {result.formatted_address}")
    print(f"Confidence: {result.confidence}")
```

### Via API Endpoint

The geocoding happens automatically during preference processing:

```bash
# This will trigger geocoding when work location is missing
curl -X POST "http://localhost:8001/api/preferences/process/interactive" \
  -H "Content-Type: application/json" \
  -d '{
    "parent_id": "uuid-here",
    "original_text": "I need daycare close to my work at Potsdamer Platz",
    "extracted_features": [],
    "complex_features": [
      {
        "feature_key": "location_proximity",
        "type_name": "location_distance", 
        "value_data": {
          "locations": [
            {
              "name": "work",
              "max_travel_minutes": 15,
              "transport_mode": "bike"
            }
          ]
        },
        "preference": "required",
        "confidence": 0.9,
        "rationale": "User specified work location constraint"
      }
    ]
  }'
```

## Troubleshooting

### Common Issues

**1. "MapBox API key not configured" Error**
```
Solution: Set MAPBOX_API_KEY environment variable
```

**2. "Authentication failed" (401 Error)**
```
Solution: Check that your API key is valid and not expired
```

**3. "Rate limit exceeded" (429 Error)** 
```
Solution: Wait or upgrade to a higher tier plan
Our service includes automatic rate limiting to prevent this
```

**4. "No results found" for Address**
```
Solution: 
- Check address spelling
- Try different address formats
- Ensure the location exists in MapBox's database
```

### Debugging Tips

**Enable Debug Logging**:
```python
import logging
logging.getLogger('matchmaker.services.map_service').setLevel(logging.DEBUG)
```

**Test Geocoding Manually**:
```bash
# Test via MapBox API directly
curl "https://api.mapbox.com/geocoding/v5/mapbox.places/Potsdamer%20Platz%201%2C%20Berlin.json?access_token=YOUR_TOKEN&country=de&limit=1"
```

**Check Service Status**:
```bash
# Check if MapBox is initialized
curl "http://localhost:8001/health"
# Look for mapbox_service status in response
```

## Alternative Providers

If MapBox is not available, you can implement alternative providers by extending the `MapServiceProvider` base class:

```python
from src.matchmaker.services.map_service import MapServiceProvider

class CustomGeocodingProvider(MapServiceProvider):
    async def geocode_address(self, address: str, city: str = None, country_code: str = "de"):
        # Your implementation here
        pass
    
    async def reverse_geocode(self, latitude: float, longitude: float):
        # Your implementation here  
        pass
```

## Security Best Practices

1. **API Key Security**:
   - Never commit API keys to source control
   - Use environment variables only
   - Rotate keys regularly
   - Use different keys for different environments

2. **Request Limiting**:
   - Monitor usage to avoid unexpected charges
   - Implement caching where possible
   - Use country/region filtering to improve performance

3. **Error Handling**:
   - Always handle geocoding failures gracefully
   - Provide fallback options for users
   - Log errors for monitoring but not sensitive data

The MapBox integration provides reliable, fast geocoding for the complex preference processing system while maintaining good performance and error handling characteristics.