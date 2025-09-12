"""Tests for MapBox integration."""

import os
import pytest
from unittest.mock import AsyncMock, patch

from src.matchmaker.services.map_service import (
    MapBoxProvider, 
    MapService, 
    GeocodeResult,
    get_map_service
)


@pytest.fixture
def mock_api_key():
    """Mock MapBox API key."""
    return "pk.test_key_12345"


@pytest.fixture
def mapbox_provider(mock_api_key):
    """Create MapBox provider for testing."""
    return MapBoxProvider(mock_api_key)


@pytest.fixture 
def map_service(mapbox_provider):
    """Create map service for testing."""
    return MapService(mapbox_provider)


class TestGeocodeResult:
    """Test GeocodeResult class."""
    
    def test_geocode_result_creation(self):
        """Test creating a GeocodeResult."""
        result = GeocodeResult(
            latitude=52.5200,
            longitude=13.4050,
            formatted_address="Berlin, Germany",
            confidence=0.95
        )
        
        assert result.latitude == 52.5200
        assert result.longitude == 13.4050
        assert result.formatted_address == "Berlin, Germany"
        assert result.confidence == 0.95
    
    def test_geocode_result_to_dict(self):
        """Test converting GeocodeResult to dictionary."""
        result = GeocodeResult(
            latitude=52.5200,
            longitude=13.4050,
            formatted_address="Berlin, Germany"
        )
        
        result_dict = result.to_dict()
        
        assert result_dict == {
            'latitude': 52.5200,
            'longitude': 13.4050,
            'formatted_address': "Berlin, Germany",
            'confidence': 1.0
        }


class TestMapBoxProvider:
    """Test MapBox provider."""
    
    def test_provider_initialization(self, mock_api_key):
        """Test MapBox provider initialization."""
        provider = MapBoxProvider(mock_api_key)
        
        assert provider.api_key == mock_api_key
        assert provider.base_url == "https://api.mapbox.com/geocoding/v5/mapbox.places"
        assert provider._min_request_interval == 0.1
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, mapbox_provider):
        """Test rate limiting functionality."""
        import time
        
        start_time = time.time()
        await mapbox_provider._rate_limit()
        await mapbox_provider._rate_limit()
        end_time = time.time()
        
        # Second call should be delayed by at least the interval
        assert end_time - start_time >= mapbox_provider._min_request_interval
    
    @pytest.mark.asyncio 
    async def test_geocode_success(self, mapbox_provider):
        """Test successful geocoding."""
        mock_response_data = {
            'features': [
                {
                    'geometry': {
                        'coordinates': [13.4050, 52.5200]  # [lng, lat]
                    },
                    'place_name': 'Berlin, Germany',
                    'relevance': 0.95
                }
            ]
        }
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = mock_response_data
            mock_get.return_value.__aenter__.return_value = mock_response
            
            result = await mapbox_provider.geocode_address("Berlin")
            
            assert result is not None
            assert result.latitude == 52.5200
            assert result.longitude == 13.4050
            assert result.formatted_address == "Berlin, Germany"
            assert result.confidence == 0.95
    
    @pytest.mark.asyncio
    async def test_geocode_no_results(self, mapbox_provider):
        """Test geocoding with no results."""
        mock_response_data = {'features': []}
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = mock_response_data
            mock_get.return_value.__aenter__.return_value = mock_response
            
            result = await mapbox_provider.geocode_address("NonexistentPlace12345")
            
            assert result is None
    
    @pytest.mark.asyncio
    async def test_geocode_auth_error(self, mapbox_provider):
        """Test geocoding with authentication error."""
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 401
            mock_get.return_value.__aenter__.return_value = mock_response
            
            result = await mapbox_provider.geocode_address("Berlin")
            
            assert result is None
    
    @pytest.mark.asyncio
    async def test_geocode_rate_limit_error(self, mapbox_provider):
        """Test geocoding with rate limit error."""
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 429
            mock_get.return_value.__aenter__.return_value = mock_response
            
            result = await mapbox_provider.geocode_address("Berlin")
            
            assert result is None
    
    @pytest.mark.asyncio
    async def test_reverse_geocode_success(self, mapbox_provider):
        """Test successful reverse geocoding."""
        mock_response_data = {
            'features': [
                {
                    'geometry': {
                        'coordinates': [13.4050, 52.5200]
                    },
                    'place_name': 'Berlin, Germany',
                    'relevance': 0.95
                }
            ]
        }
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = mock_response_data
            mock_get.return_value.__aenter__.return_value = mock_response
            
            result = await mapbox_provider.reverse_geocode(52.5200, 13.4050)
            
            assert result is not None
            assert result.latitude == 52.5200
            assert result.longitude == 13.4050
            assert result.formatted_address == "Berlin, Germany"


class TestMapService:
    """Test MapService class."""
    
    @pytest.mark.asyncio
    async def test_geocode_address_delegates_to_provider(self, map_service):
        """Test that MapService delegates to provider."""
        expected_result = GeocodeResult(52.5200, 13.4050, "Berlin, Germany")
        
        with patch.object(map_service.provider, 'geocode_address', return_value=expected_result) as mock_geocode:
            result = await map_service.geocode_address("Berlin")
            
            mock_geocode.assert_called_once_with("Berlin", None, "de")
            assert result == expected_result
    
    @pytest.mark.asyncio
    async def test_reverse_geocode_delegates_to_provider(self, map_service):
        """Test that MapService delegates reverse geocoding to provider."""
        expected_result = GeocodeResult(52.5200, 13.4050, "Berlin, Germany")
        
        with patch.object(map_service.provider, 'reverse_geocode', return_value=expected_result) as mock_reverse:
            result = await map_service.reverse_geocode(52.5200, 13.4050)
            
            mock_reverse.assert_called_once_with(52.5200, 13.4050)
            assert result == expected_result


class TestMapServiceFactory:
    """Test global map service factory."""
    
    def test_get_map_service_requires_api_key_first_time(self):
        """Test that get_map_service requires API key on first call."""
        # Reset global state
        import src.matchmaker.services.map_service
        src.matchmaker.services.map_service._map_service = None
        
        with pytest.raises(ValueError, match="MapBox API key required"):
            get_map_service()
    
    def test_get_map_service_creates_instance(self, mock_api_key):
        """Test that get_map_service creates instance with API key."""
        # Reset global state
        import src.matchmaker.services.map_service
        src.matchmaker.services.map_service._map_service = None
        
        service = get_map_service(mock_api_key)
        
        assert service is not None
        assert isinstance(service, MapService)
        assert isinstance(service.provider, MapBoxProvider)
        assert service.provider.api_key == mock_api_key
    
    def test_get_map_service_returns_singleton(self, mock_api_key):
        """Test that subsequent calls return the same instance."""
        # Reset global state
        import src.matchmaker.services.map_service
        src.matchmaker.services.map_service._map_service = None
        
        service1 = get_map_service(mock_api_key)
        service2 = get_map_service()  # No API key needed for subsequent calls
        
        assert service1 is service2


@pytest.mark.integration
class TestMapBoxIntegration:
    """Integration tests for MapBox (require real API key)."""
    
    @pytest.mark.skipif(
        not os.getenv('MAPBOX_API_KEY'), 
        reason="MAPBOX_API_KEY environment variable not set"
    )
    @pytest.mark.asyncio
    async def test_real_geocoding(self):
        """Test real geocoding with MapBox API."""
        api_key = os.getenv('MAPBOX_API_KEY')
        provider = MapBoxProvider(api_key)
        
        # Test geocoding Berlin
        result = await provider.geocode_address("Berlin", country_code="de")
        
        assert result is not None
        assert 52.0 <= result.latitude <= 53.0  # Rough Berlin latitude range
        assert 13.0 <= result.longitude <= 14.0  # Rough Berlin longitude range
        assert "Berlin" in result.formatted_address
        
        # Clean up
        await provider.close()
    
    @pytest.mark.skipif(
        not os.getenv('MAPBOX_API_KEY'),
        reason="MAPBOX_API_KEY environment variable not set"
    )
    @pytest.mark.asyncio
    async def test_real_reverse_geocoding(self):
        """Test real reverse geocoding with MapBox API."""
        api_key = os.getenv('MAPBOX_API_KEY')
        provider = MapBoxProvider(api_key)
        
        # Test reverse geocoding of Berlin coordinates
        result = await provider.reverse_geocode(52.5200, 13.4050)
        
        assert result is not None
        assert result.formatted_address is not None
        assert len(result.formatted_address) > 0
        
        # Clean up
        await provider.close()