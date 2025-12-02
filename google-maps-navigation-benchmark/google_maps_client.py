"""
Google Maps API Client for Places, Directions, Geocoding, and Distance Matrix APIs.
This module provides a wrapper around Google Maps APIs for use with CrewAI agents.
"""
import googlemaps
from typing import List, Dict, Any, Optional
from config import GOOGLE_MAPS_API_KEY

class GoogleMapsClient:
    """Client for interacting with Google Maps APIs."""
    
    def __init__(self, api_key: str = None):
        """
        Initialize Google Maps client.
        
        Args:
            api_key: Google Maps API key. If None, uses GOOGLE_MAPS_API_KEY from config.
        """
        self.api_key = api_key or GOOGLE_MAPS_API_KEY
        if not self.api_key or self.api_key == "YOUR_GOOGLE_MAPS_API_KEY_HERE":
            raise ValueError(
                "Google Maps API key not set. Please set GOOGLE_MAPS_API_KEY in config.py or as an environment variable.\n"
                "Required APIs: Places API, Directions API, Geocoding API, Distance Matrix API"
            )
        self.client = googlemaps.Client(key=self.api_key)
    
    def geocode(self, address: str) -> List[Dict[str, Any]]:
        """
        Geocode an address to get coordinates.
        
        Args:
            address: Address string to geocode
            
        Returns:
            List of geocoding results
        """
        try:
            return self.client.geocode(address)
        except Exception as e:
            raise Exception(f"Geocoding failed for '{address}': {str(e)}")
    
    def reverse_geocode(self, lat: float, lng: float) -> List[Dict[str, Any]]:
        """
        Reverse geocode coordinates to get address.
        
        Args:
            lat: Latitude
            lng: Longitude
            
        Returns:
            List of reverse geocoding results
        """
        try:
            return self.client.reverse_geocode((lat, lng))
        except Exception as e:
            raise Exception(f"Reverse geocoding failed for ({lat}, {lng}): {str(e)}")
    
    def directions(
        self,
        origin: str,
        destination: str,
        waypoints: Optional[List[str]] = None,
        mode: str = "driving",
        alternatives: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get directions between origin and destination.
        
        Args:
            origin: Starting location
            destination: Ending location
            waypoints: Optional list of waypoints
            mode: Travel mode (driving, walking, bicycling, transit)
            alternatives: Whether to return alternative routes
            
        Returns:
            List of route options
        """
        try:
            return self.client.directions(
                origin=origin,
                destination=destination,
                waypoints=waypoints,
                mode=mode,
                alternatives=alternatives
            )
        except Exception as e:
            raise Exception(f"Directions failed: {str(e)}")
    
    def places_nearby(
        self,
        location: str,
        radius: int = 5000,
        place_type: Optional[str] = None,
        keyword: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Find places near a location.
        
        Args:
            location: Location string or coordinates
            radius: Search radius in meters
            place_type: Place type (e.g., 'restaurant', 'gas_station', 'park')
            keyword: Keyword search
            
        Returns:
            List of nearby places
        """
        try:
            # First geocode the location if it's a string
            if isinstance(location, str):
                geocode_result = self.geocode(location)
                if not geocode_result:
                    return []
                location = geocode_result[0]['geometry']['location']
            
            params = {
                'location': (location['lat'], location['lng']),
                'radius': radius
            }
            
            if place_type:
                params['type'] = place_type
            if keyword:
                params['keyword'] = keyword
            
            result = self.client.places_nearby(**params)
            return result.get('results', [])
        except Exception as e:
            raise Exception(f"Places nearby search failed: {str(e)}")
    
    def place_search(
        self,
        query: str,
        location: Optional[str] = None,
        radius: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for places using text query.
        
        Args:
            query: Search query
            location: Optional location bias
            radius: Optional search radius
            
        Returns:
            List of places matching the query
        """
        try:
            params = {'query': query}
            if location:
                params['location'] = location
            if radius:
                params['radius'] = radius
            
            result = self.client.places(query=query, location=location, radius=radius)
            return result.get('results', [])
        except Exception as e:
            raise Exception(f"Place search failed: {str(e)}")
    
    def place_details(self, place_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a place.
        
        Args:
            place_id: Google Place ID
            
        Returns:
            Place details
        """
        try:
            result = self.client.place(place_id=place_id)
            return result.get('result', {})
        except Exception as e:
            raise Exception(f"Place details failed for {place_id}: {str(e)}")
    
    def distance_matrix(
        self,
        origins: List[str],
        destinations: List[str],
        mode: str = "driving"
    ) -> Dict[str, Any]:
        """
        Calculate distance and duration between multiple origins and destinations.
        
        Args:
            origins: List of origin locations
            destinations: List of destination locations
            mode: Travel mode
            
        Returns:
            Distance matrix results
        """
        try:
            return self.client.distance_matrix(origins, destinations, mode=mode)
        except Exception as e:
            raise Exception(f"Distance matrix failed: {str(e)}")
    
    def elevation(self, locations: List[tuple]) -> List[Dict[str, Any]]:
        """
        Get elevation data for locations.
        
        Args:
            locations: List of (lat, lng) tuples
            
        Returns:
            List of elevation results
        """
        try:
            return self.client.elevation(locations)
        except Exception as e:
            raise Exception(f"Elevation lookup failed: {str(e)}")

