"""
Place Search Agent - Searches for places using Google Maps.

Finds rest stops, scenic viewpoints, restaurants, and other places
in specified cities with validation.
"""

import logging
from typing import List, Dict, Optional, Any

from .base import OllamaAgent
from .models import PlaceInfo
from .tools import MCPToolWrapper, google_maps_search_places, google_maps_place_details

logger = logging.getLogger(__name__)


class PlaceSearchAgent(OllamaAgent):
    """
    Specialized agent for searching and validating places.
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
        logger.info("Place Search Agent initialized")

    async def cleanup(self):
        self._initialized = False

    async def search_places(
        self,
        cities: List[str],
        place_type: str,
        count_per_city: int = 1,
        min_rating: float = 0.0
    ) -> List[PlaceInfo]:
        """
        Search for places in specified cities.

        Args:
            cities: List of city names to search in
            place_type: Type of place (rest_stop, scenic_viewpoint, restaurant, etc.)
            count_per_city: Number of places to find per city
            min_rating: Minimum rating filter

        Returns:
            List of PlaceInfo objects
        """
        if not self._initialized:
            await self.initialize()

        logger.info(f"Searching for {place_type} in {len(cities)} cities")

        all_places = []
        for city in cities:
            try:
                # Create search query
                query = f"{place_type.replace('_', ' ')} in {city}"

                # Search using Google Maps
                results = await google_maps_search_places(self._tool_wrapper, query)

                # Ensure results is a list
                if not isinstance(results, list):
                    if isinstance(results, dict) and "results" in results:
                        results = results["results"]
                    else:
                        results = []

                # Filter and convert to PlaceInfo
                for result in results[:count_per_city]:
                    if not isinstance(result, dict):
                        continue

                    rating = result.get("rating", 0.0)
                    if rating >= min_rating:
                        place = PlaceInfo(
                            name=result.get("name", "Unknown"),
                            address=result.get("formatted_address", result.get("address", "")),
                            place_id=result.get("place_id"),
                            rating=rating,
                            types=result.get("types", []),
                            location=result.get("location")
                        )
                        all_places.append(place)

                logger.info(f"Found {len(all_places)} {place_type} in {city}")

            except Exception as e:
                logger.error(f"Failed to search places in {city}: {e}")
                # Add fallback place
                all_places.append(PlaceInfo(
                    name=f"{place_type.title()} in {city}",
                    address=f"{city}, Address",
                    rating=4.0
                ))

        return all_places

    def __repr__(self) -> str:
        return f"PlaceSearchAgent(model='{self.model}')"
