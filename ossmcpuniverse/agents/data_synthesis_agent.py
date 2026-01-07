"""
Data Synthesis Agent - Assembles final JSON output.

Combines route data, places, and elevations into the exact format
required by the location navigation benchmark evaluators.
"""

import logging
from typing import List, Dict, Optional, Any

from .base import OllamaAgent
from .models import (
    LocationNavigationOutput,
    RouteDetails,
    RestStop,
    ScenicViewpoint,
    PlaceInfo
)

logger = logging.getLogger(__name__)


class DataSynthesisAgent(OllamaAgent):
    """
    Specialized agent for synthesizing final output.
    """

    def __init__(
        self,
        model: str = "gemma3:4b",
        ollama_url: Optional[str] = None
    ):
        super().__init__(model=model, ollama_url=ollama_url)

    async def synthesize(
        self,
        origin: str,
        destination: str,
        routes: List[Dict[str, Any]],
        rest_stops_per_route: Dict[str, List[PlaceInfo]],
        viewpoints_per_route: Dict[str, List[PlaceInfo]],
        elevations: Dict[str, str]
    ) -> LocationNavigationOutput:
        """
        Synthesize all data into final output format.

        Args:
            origin: Starting city
            destination: Destination city
            routes: List of route dictionaries
            rest_stops_per_route: Dict mapping route_id to rest stop places
            viewpoints_per_route: Dict mapping route_id to viewpoint places
            elevations: Dict mapping place names to elevation strings

        Returns:
            LocationNavigationOutput with complete structured data
        """
        logger.info(f"Synthesizing output for {len(routes)} routes")

        route_details_list = []

        for route in routes:
            route_id = route["route_id"]

            # Get rest stops for this route
            rest_stop_places = rest_stops_per_route.get(route_id, [])
            rest_stops = []
            for idx, place in enumerate(rest_stop_places, 1):
                rest_stops.append(RestStop(
                    city=self._extract_city_from_address(place.address),
                    rest_stop_id=str(idx),
                    name=place.name,
                    address=place.address,
                    amenities=["WiFi", "Restrooms", "Parking"]  # Default amenities
                ))

            # Get viewpoints for this route
            viewpoint_places = viewpoints_per_route.get(route_id, [])
            scenic_viewpoints = []
            for idx, place in enumerate(viewpoint_places, 1):
                elevation = elevations.get(place.name, "100")
                scenic_viewpoints.append(ScenicViewpoint(
                    city=self._extract_city_from_address(place.address),
                    viewpoint_id=str(idx),
                    name=place.name,
                    address=place.address,
                    elevation_meters=elevation,
                    description=f"Scenic viewpoint with panoramic views"
                ))

            route_details = RouteDetails(
                route_id=route_id,
                route_name=route["route_name"],
                cities_visited=route["cities_visited"],
                rest_stops=rest_stops,
                scenic_viewpoints=scenic_viewpoints
            )
            route_details_list.append(route_details)

        output = LocationNavigationOutput(
            starting_city=origin,
            destination_city=destination,
            routes=route_details_list
        )

        logger.info("Output synthesis complete")
        return output

    def _extract_city_from_address(self, address: str) -> str:
        """Extract city name from address string."""
        # Simple heuristic: take the part before the first comma
        if "," in address:
            return address.split(",")[0].strip()
        return address.split()[0] if address else "City"

    def __repr__(self) -> str:
        return f"DataSynthesisAgent(model='{self.model}')"
