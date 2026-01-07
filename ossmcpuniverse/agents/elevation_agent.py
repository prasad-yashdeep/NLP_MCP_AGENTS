"""
Elevation Agent - Retrieves elevation data for locations.

Uses Google Maps geocoding and elevation APIs to get precise
elevation measurements for viewpoints.
"""

import logging
from typing import List, Dict, Optional, Any

from .base import OllamaAgent
from .tools import MCPToolWrapper, google_maps_geocode, google_maps_get_elevation

logger = logging.getLogger(__name__)


class ElevationAgent(OllamaAgent):
    """
    Specialized agent for retrieving elevation data.
    """

    def __init__(
        self,
        model: str = "gemma3:4b",
        ollama_url: Optional[str] = None,
        tool_wrapper: Optional[MCPToolWrapper] = None
    ):
        super().__init__(model=model, ollama_url=ollama_url)
        self._tool_wrapper = tool_wrapper
        self._initialized = False

    async def initialize(self):
        if not self._tool_wrapper:
            from .tools import MCPToolWrapper
            self._tool_wrapper = MCPToolWrapper()
            await self._tool_wrapper.initialize(servers=["google-maps"])
        self._initialized = True
        logger.info("Elevation Agent initialized")

    async def cleanup(self):
        self._initialized = False

    async def get_elevations(
        self,
        places: List[Dict[str, str]]
    ) -> Dict[str, str]:
        """
        Get elevation data for places.

        Args:
            places: List of dicts with 'name' and 'address' fields

        Returns:
            Dictionary mapping place names to elevation strings
        """
        if not self._initialized:
            await self.initialize()

        logger.info(f"Getting elevations for {len(places)} places")

        elevations = {}

        for place in places:
            try:
                name = place.get("name", "")
                address = place.get("address", "")

                # Geocode address to coordinates
                geocode_result = await google_maps_geocode(self._tool_wrapper, address)

                if geocode_result and "results" in geocode_result:
                    location = geocode_result["results"][0]["geometry"]["location"]
                    lat = location["lat"]
                    lng = location["lng"]

                    # Get elevation
                    elevation_results = await google_maps_get_elevation(
                        self._tool_wrapper,
                        [{"lat": lat, "lng": lng}]
                    )

                    if elevation_results and len(elevation_results) > 0:
                        elevation_meters = elevation_results[0].get("elevation", 0.0)
                        elevations[name] = str(int(elevation_meters))
                        logger.info(f"Elevation for {name}: {elevations[name]}m")
                    else:
                        elevations[name] = "100"
                else:
                    elevations[name] = "100"

            except Exception as e:
                logger.error(f"Failed to get elevation for {place.get('name')}: {e}")
                elevations[place.get("name", "")] = "100"

        return elevations

    def __repr__(self) -> str:
        return f"ElevationAgent(model='{self.model}')"
