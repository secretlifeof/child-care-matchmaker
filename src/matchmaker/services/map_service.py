"""Map service for geocoding addresses using various providers."""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple

import aiohttp

logger = logging.getLogger(__name__)


class GeocodeResult:
    """Standardized geocoding result."""
    
    def __init__(self, latitude: float, longitude: float, formatted_address: str, confidence: float = 1.0):
        self.latitude = latitude
        self.longitude = longitude
        self.formatted_address = formatted_address
        self.confidence = confidence
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'latitude': self.latitude,
            'longitude': self.longitude,
            'formatted_address': self.formatted_address,
            'confidence': self.confidence
        }


class MapServiceProvider(ABC):
    """Abstract base class for map service providers."""
    
    @abstractmethod
    async def geocode_address(self, address: str, city: str = None, country_code: str = "de") -> Optional[GeocodeResult]:
        """Geocode an address to latitude/longitude coordinates."""
        pass
    
    @abstractmethod
    async def reverse_geocode(self, latitude: float, longitude: float) -> Optional[GeocodeResult]:
        """Reverse geocode coordinates to an address."""
        pass


class MapBoxProvider(MapServiceProvider):
    """MapBox geocoding provider."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.mapbox.com/geocoding/v5/mapbox.places"
        self._session: Optional[aiohttp.ClientSession] = None
        # Rate limiting: MapBox allows 100,000 requests per month on free tier
        # We'll implement basic rate limiting to stay within reasonable bounds
        self._last_request_time = 0
        self._min_request_interval = 0.1  # 10 requests per second max
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10),
                headers={'User-Agent': 'ChildCareMatchmaker/1.0'}
            )
        return self._session
    
    async def _rate_limit(self):
        """Simple rate limiting."""
        import time
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        
        if time_since_last < self._min_request_interval:
            await asyncio.sleep(self._min_request_interval - time_since_last)
        
        self._last_request_time = time.time()
    
    async def geocode_address(self, address: str, city: str = None, country_code: str = "de") -> Optional[GeocodeResult]:
        """Geocode address using MapBox Geocoding API."""
        try:
            await self._rate_limit()
            
            # Construct search query
            query = address
            if city:
                query = f"{address}, {city}"
            
            # MapBox geocoding parameters
            params = {
                'access_token': self.api_key,
                'country': country_code,
                'limit': 1,
                'types': 'address,poi'  # Prefer addresses and points of interest
            }
            
            session = await self._get_session()
            url = f"{self.base_url}/{query}.json"
            
            logger.debug(f"MapBox geocoding request: {query}")
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get('features') and len(data['features']) > 0:
                        feature = data['features'][0]
                        coordinates = feature['geometry']['coordinates']  # [longitude, latitude]
                        
                        result = GeocodeResult(
                            latitude=coordinates[1],
                            longitude=coordinates[0],
                            formatted_address=feature['place_name'],
                            confidence=feature.get('relevance', 1.0)
                        )
                        
                        logger.info(f"MapBox geocoded '{query}' -> {result.latitude}, {result.longitude}")
                        return result
                    else:
                        logger.warning(f"MapBox: No results found for '{query}'")
                        return None
                        
                elif response.status == 401:
                    logger.error("MapBox API authentication failed - check API key")
                    return None
                elif response.status == 429:
                    logger.warning("MapBox API rate limit exceeded")
                    return None
                else:
                    logger.error(f"MapBox API error: {response.status}")
                    return None
                    
        except asyncio.TimeoutError:
            logger.error(f"MapBox geocoding timeout for '{query}'")
            return None
        except Exception as e:
            logger.error(f"MapBox geocoding error for '{query}': {e}")
            return None
    
    async def reverse_geocode(self, latitude: float, longitude: float) -> Optional[GeocodeResult]:
        """Reverse geocode coordinates using MapBox."""
        try:
            await self._rate_limit()
            
            params = {
                'access_token': self.api_key,
                'types': 'address'
            }
            
            session = await self._get_session()
            url = f"{self.base_url}/{longitude},{latitude}.json"
            
            logger.debug(f"MapBox reverse geocoding: {latitude}, {longitude}")
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get('features') and len(data['features']) > 0:
                        feature = data['features'][0]
                        coordinates = feature['geometry']['coordinates']
                        
                        result = GeocodeResult(
                            latitude=coordinates[1],
                            longitude=coordinates[0],
                            formatted_address=feature['place_name'],
                            confidence=feature.get('relevance', 1.0)
                        )
                        
                        logger.info(f"MapBox reverse geocoded {latitude}, {longitude} -> '{result.formatted_address}'")
                        return result
                    else:
                        logger.warning(f"MapBox: No reverse geocoding results for {latitude}, {longitude}")
                        return None
                else:
                    logger.error(f"MapBox reverse geocoding error: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"MapBox reverse geocoding error: {e}")
            return None
    
    async def close(self):
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()


class MapService:
    """Main map service that can use different providers."""
    
    def __init__(self, provider: MapServiceProvider):
        self.provider = provider
    
    async def geocode_address(self, address: str, city: str = None, country_code: str = "de") -> Optional[GeocodeResult]:
        """Geocode an address to coordinates."""
        return await self.provider.geocode_address(address, city, country_code)
    
    async def reverse_geocode(self, latitude: float, longitude: float) -> Optional[GeocodeResult]:
        """Reverse geocode coordinates to address."""
        return await self.provider.reverse_geocode(latitude, longitude)
    
    async def close(self):
        """Close provider resources."""
        if hasattr(self.provider, 'close'):
            await self.provider.close()


# Global service instance
_map_service: Optional[MapService] = None


def get_map_service(api_key: str = None) -> MapService:
    """Get global map service instance."""
    global _map_service
    
    if _map_service is None:
        if not api_key:
            raise ValueError("MapBox API key required for first initialization")
        
        provider = MapBoxProvider(api_key)
        _map_service = MapService(provider)
    
    return _map_service


async def close_map_service():
    """Close global map service."""
    global _map_service
    if _map_service:
        await _map_service.close()
        _map_service = None