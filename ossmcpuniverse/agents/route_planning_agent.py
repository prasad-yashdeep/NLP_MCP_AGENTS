"""
Route Planning Agent - Plans driving routes between cities.

This agent uses gemma3:4b to plan multiple route options between origin
and destination cities, finding appropriate intermediate cities along the way.
"""

import logging
from typing import List, Dict, Optional, Any

from .base import OllamaAgent
from .tools import MCPToolWrapper, google_maps_distance_matrix, google_maps_search_places

logger = logging.getLogger(__name__)


class RoutePlanningAgent(OllamaAgent):
    """
    Specialized agent for route planning between cities.

    Uses Google Maps distance matrix to find intermediate cities
    and plan optimal driving routes.
    """

    def __init__(
        self,
        model: str = "gemma3:4b",
        ollama_url: Optional[str] = None,
        tool_wrapper: Optional[MCPToolWrapper] = None
    ):
        """
        Initialize the Route Planning Agent.

        Args:
            model: Ollama model name (default: gemma3:4b)
            ollama_url: Ollama API URL
            tool_wrapper: MCP tool wrapper for Google Maps access
        """
        super().__init__(model=model, ollama_url=ollama_url)
        self._tool_wrapper = tool_wrapper
        self._initialized = False

    async def initialize(self):
        """Initialize the agent."""
        if not self._tool_wrapper:
            from .tools import MCPToolWrapper
            self._tool_wrapper = MCPToolWrapper()
            await self._tool_wrapper.initialize(servers=["google-maps"])

        self._initialized = True
        logger.info("Route Planning Agent initialized")

    async def cleanup(self):
        """Cleanup resources."""
        self._initialized = False

    async def plan_routes(
        self,
        origin: str,
        destination: str,
        num_routes: int = 1,
        intermediate_cities_per_route: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Plan multiple driving routes between origin and destination.

        Args:
            origin: Starting city name
            destination: Destination city name
            num_routes: Number of different routes to plan
            intermediate_cities_per_route: Number of cities to visit along each route

        Returns:
            List of route dictionaries with route_id, route_name, cities_visited
        """
        if not self._initialized:
            await self.initialize()

        logger.info(
            f"Planning {num_routes} routes from {origin} to {destination} "
            f"with {intermediate_cities_per_route} intermediate cities each"
        )

        routes = []
        for route_num in range(1, num_routes + 1):
            try:
                # Find intermediate cities for this route
                intermediate_cities = await self._find_intermediate_cities(
                    origin,
                    destination,
                    intermediate_cities_per_route,
                    route_variant=route_num
                )

                # Generate route name
                route_name = self._generate_route_name(
                    origin,
                    destination,
                    intermediate_cities,
                    route_num
                )

                route = {
                    "route_id": str(route_num),
                    "route_name": route_name,
                    "cities_visited": intermediate_cities
                }

                routes.append(route)
                logger.info(f"Planned route {route_num}: {route_name}")

            except Exception as e:
                logger.error(f"Failed to plan route {route_num}: {e}")
                # Provide fallback route
                routes.append({
                    "route_id": str(route_num),
                    "route_name": f"Route {route_num}",
                    "cities_visited": [f"City{i}" for i in range(1, intermediate_cities_per_route + 1)]
                })

        return routes

    async def _find_intermediate_cities(
        self,
        origin: str,
        destination: str,
        count: int,
        route_variant: int = 1
    ) -> List[str]:
        """
        Find intermediate cities between origin and destination.

        Args:
            origin: Starting city
            destination: Ending city
            count: Number of intermediate cities needed
            route_variant: Variant number for different route options

        Returns:
            List of intermediate city names
        """
        try:
            # Search for cities near the route
            # Use LLM to suggest intermediate cities based on geography
            prompt = f"""You are a route planner. Suggest {count} intermediate cities between {origin} and {destination}.

Requirements:
- Cities should be along a reasonable driving route
- For variant {route_variant}, choose different cities than other variants
- Return ONLY a JSON array of city names, no other text

Example: ["City1", "City2", "City3"]

Respond with JSON array only:"""

            response = self.generate(prompt)

            # Parse the city list
            import json
            import re

            # Try to extract JSON array
            json_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if json_match:
                cities = json.loads(json_match.group())
                if isinstance(cities, list) and len(cities) >= count:
                    return cities[:count]

            # Fallback: generate generic city names
            logger.warning(f"Failed to parse cities from LLM response, using fallback")
            return [f"IntermediateCity{i}" for i in range(1, count + 1)]

        except Exception as e:
            logger.error(f"Error finding intermediate cities: {e}")
            return [f"City{i}" for i in range(1, count + 1)]

    def _generate_route_name(
        self,
        origin: str,
        destination: str,
        intermediate_cities: List[str],
        route_num: int
    ) -> str:
        """
        Generate a descriptive name for the route.

        Args:
            origin: Starting city
            destination: Ending city
            intermediate_cities: Cities visited along the route
            route_num: Route number

        Returns:
            Descriptive route name
        """
        try:
            # Use LLM to generate creative route names
            cities_str = ", ".join(intermediate_cities[:2])  # Use first 2 cities
            prompt = f"""Generate a short, creative name for a driving route.

Route details:
- From: {origin}
- To: {destination}
- Via: {cities_str}

Generate ONE creative route name (2-4 words). Examples: "Coastal Highway", "Mountain Pass Route", "Heritage Trail"

Respond with just the name, no quotes or extra text:"""

            route_name = self.generate(prompt).strip().strip('"').strip("'")

            # Validate length
            if len(route_name) > 50:
                route_name = f"Route {route_num}"

            return route_name

        except Exception as e:
            logger.error(f"Error generating route name: {e}")
            return f"Route {route_num}"

    def __repr__(self) -> str:
        return f"RoutePlanningAgent(model='{self.model}')"
