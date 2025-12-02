"""
CrewAI Agents for Google Maps Navigation Benchmark.
Hierarchical orchestrator architecture with specialist agents.
"""
import os
import json
from typing import Dict, Any, List, Union, Optional
from pydantic import Field

# IMPORTANT: Configure environment BEFORE importing CrewAI
from config import CREWAI_MODEL, CREWAI_TEMPERATURE, OLLAMA_BASE_URL
from llm_config import configure_crewai_for_ollama

# Configure CrewAI to use Ollama via litellm BEFORE importing CrewAI
# CrewAI uses litellm internally, which supports Ollama
ollama_model_name = configure_crewai_for_ollama()

# Set environment variables that CrewAI will use
os.environ["OPENAI_API_BASE"] = f"{OLLAMA_BASE_URL}/v1"
os.environ["OPENAI_API_KEY"] = "ollama"
os.environ["LITELLM_MODEL"] = ollama_model_name

# Now import CrewAI (after environment is configured)
from crewai import Agent, Task, Crew
from crewai.tools import tool
from google_maps_client import GoogleMapsClient

# Use the configured model name for agents
# Format: "ollama/gemma2:9b" or "ollama/gemma2:2b"
AGENT_MODEL = ollama_model_name if ollama_model_name else f"ollama/{CREWAI_MODEL}"

# Print model configuration for debugging
print(f"Using model: {AGENT_MODEL}")
print(f"Ollama base URL: {OLLAMA_BASE_URL}")

# Initialize Google Maps client
try:
    maps_client = GoogleMapsClient()
except ValueError as e:
    print(f"Warning: {e}")
    maps_client = None


# Google Maps Tools using CrewAI 1.6.1 @tool decorator
# Format: @tool("tool_name") with description in docstring
maps_tools = []
if maps_client:
    @tool("geocode")
    def geocode(address: Union[str, Dict[str, Any]]) -> str:
        """Geocode an address to get coordinates. Input: address string. Returns: coordinates and formatted address."""
        # Handle case where agent passes dict instead of string
        if isinstance(address, dict):
            address = address.get("description") or address.get("address") or str(address)
        address_str = str(address) if address else ""
        result = maps_client.geocode(address_str)
        return str(result) if result else "No results found"
    
    @tool("reverse_geocode")
    def reverse_geocode(lat: Union[float, str, Dict[str, Any]], lng: Union[float, str, Dict[str, Any]]) -> str:
        """Reverse geocode coordinates to get address. Input: latitude and longitude. Returns: formatted address."""
        # Handle case where agent passes dict or string
        if isinstance(lat, dict):
            lat = lat.get("lat") or lat.get("latitude") or lat.get("description")
        if isinstance(lng, dict):
            lng = lng.get("lng") or lng.get("longitude") or lng.get("description")
        try:
            lat_float = float(lat)
            lng_float = float(lng)
        except (ValueError, TypeError):
            return "Invalid coordinates"
        result = maps_client.reverse_geocode(lat_float, lng_float)
        return str(result) if result else "No results found"
    
    @tool("get_directions")
    def get_directions(origin: Union[str, Dict[str, Any]], destination: Union[str, Dict[str, Any]], waypoints: Union[str, List, None] = None, mode: str = "driving") -> str:
        """Get driving directions between locations with optional waypoints. Input: origin, destination, waypoints (list), mode. Returns: route information."""
        # Handle dict inputs
        if isinstance(origin, dict):
            origin = origin.get("description") or origin.get("origin") or str(origin)
        if isinstance(destination, dict):
            destination = destination.get("description") or destination.get("destination") or str(destination)
        origin_str = str(origin)
        destination_str = str(destination)
        waypoints_list = eval(waypoints) if waypoints and isinstance(waypoints, str) else waypoints
        result = maps_client.directions(origin_str, destination_str, waypoints_list, mode)
        return str(result) if result else "No route found"
    
    @tool("find_places_nearby")
    def find_places_nearby(
        location: Union[str, Dict[str, Any]], 
        radius: Optional[Union[str, int]] = "5000", 
        place_type: Optional[str] = None, 
        keyword: Optional[str] = None
    ) -> str:
        """Find places near a location. Input: location, radius (meters as string, default 5000), place_type (optional), keyword (optional). Returns: list of nearby places."""
        # Handle dict input for location
        if isinstance(location, dict):
            location = location.get("description") or location.get("location") or str(location)
        location_str = str(location)
        # Convert radius string to int
        try:
            radius_int = int(radius) if radius and str(radius).strip() else 5000
        except (ValueError, TypeError):
            radius_int = 5000
        result = maps_client.places_nearby(location_str, radius_int, place_type, keyword)
        return str(result) if result else "No places found"
    
    @tool("search_places")
    def search_places(
        query: Union[str, Dict[str, Any]], 
        location: Optional[Union[str, Dict[str, Any]]] = Field(default=None, description="Optional location string"),
        radius: Optional[Union[str, int]] = Field(default=None, description="Optional radius as string or int")
    ) -> str:
        """Search for places using text query. Input: query string, location (optional), radius (optional integer as string). Returns: matching places."""
        # Handle dict inputs
        if isinstance(query, dict):
            query = query.get("description") or query.get("query") or str(query)
        if isinstance(location, dict):
            location = location.get("description") or location.get("location") or str(location)
        query_str = str(query)
        location_str = str(location) if location else None
        # Handle optional radius - convert string to int if provided
        radius_int = None
        if radius is not None and str(radius).strip():
            try:
                radius_int = int(radius)
            except (ValueError, TypeError):
                radius_int = None
        result = maps_client.place_search(query_str, location_str, radius_int)
        return str(result) if result else "No places found"
    
    @tool("get_place_details")
    def get_place_details(place_id: Union[str, Dict[str, Any]]) -> str:
        """Get detailed information about a place using its Place ID. Input: place_id. Returns: place details including rating, address, amenities."""
        # Handle dict input
        if isinstance(place_id, dict):
            place_id = place_id.get("place_id") or place_id.get("description") or place_id.get("id") or str(place_id)
        place_id_str = str(place_id)
        result = maps_client.place_details(place_id_str)
        return str(result) if result else "Place not found"
    
    @tool("calculate_distance_matrix")
    def calculate_distance_matrix(origins: Union[str, List, Dict[str, Any]], destinations: Union[str, List, Dict[str, Any]], mode: str = "driving") -> str:
        """Calculate distance and duration between multiple origins and destinations. Input: origins (list), destinations (list), mode. Returns: distance matrix."""
        # Handle dict inputs
        if isinstance(origins, dict):
            origins = origins.get("origins") or origins.get("description") or str(origins)
        if isinstance(destinations, dict):
            destinations = destinations.get("destinations") or destinations.get("description") or str(destinations)
        origins_list = eval(origins) if isinstance(origins, str) else origins
        destinations_list = eval(destinations) if isinstance(destinations, str) else destinations
        result = maps_client.distance_matrix(origins_list, destinations_list, mode)
        return str(result) if result else "Calculation failed"
    
    @tool("get_elevation")
    def get_elevation(locations: Union[str, List, Dict[str, Any]]) -> str:
        """Get elevation data for locations. Input: list of (latitude, longitude) tuples. Returns: elevation in meters."""
        # Handle dict input
        if isinstance(locations, dict):
            locations = locations.get("locations") or locations.get("description") or str(locations)
        # Handle string or list input
        if isinstance(locations, str):
            try:
                locations_list = eval(locations)
            except:
                return "Invalid locations format"
        elif isinstance(locations, dict):
            # Try to extract from dict
            locations_list = locations.get("locations") or locations.get("description")
            if locations_list:
                try:
                    locations_list = eval(locations_list) if isinstance(locations_list, str) else locations_list
                except:
                    return "Invalid locations format"
            else:
                return "Invalid locations format"
        else:
            locations_list = locations
        
        if not isinstance(locations_list, list):
            return "Locations must be a list"
        
        result = maps_client.elevation(locations_list)
        return str(result) if result else "Elevation data not found"
    
    maps_tools = [
        geocode,
        reverse_geocode,
        get_directions,
        find_places_nearby,
        search_places,
        get_place_details,
        calculate_distance_matrix,
        get_elevation
    ]


def create_orchestrator_agent() -> Agent:
    """Create the hierarchical orchestrator agent."""
    return Agent(
        role="Navigation Orchestrator",
        goal="Analyze navigation questions and delegate to appropriate specialist agents",
        backstory="""You are an expert navigation orchestrator with deep understanding of 
        different types of navigation tasks. Your role is to analyze user questions and 
        determine which specialist agent should handle the task:
        - Route Planning: Multi-city routes with stops and viewpoints
        - Distance Optimization: Finding optimal stops along routes
        - Time Optimization: Equidistant meeting points and time-based routing
        - Place Finding: Location discovery with geographic constraints
        
        After delegation, you coordinate the response and ensure it meets all requirements.""",
        verbose=True,
        allow_delegation=True,
        tools=maps_tools,
        llm=AGENT_MODEL  # Explicitly set Ollama model
    )


def create_route_planning_agent() -> Agent:
    """Create specialist agent for route planning tasks."""
    return Agent(
        role="Route Planning Specialist",
        goal="Plan multi-city routes with stops, rest areas, and scenic viewpoints",
        backstory="""You are an expert in route planning and road trip optimization. 
        You excel at:
        - Finding multiple route options between cities
        - Identifying convenient rest stops along routes
        - Locating scenic viewpoints with elevation data
        - Ensuring routes visit required cities
        - Balancing route efficiency with interesting stops
        
        You use Google Maps APIs to find routes, places, and elevation data.""",
        verbose=True,
        tools=maps_tools,
        llm=AGENT_MODEL,  # Explicitly set Ollama model
        max_iter=15  # Limit iterations to prevent recursion issues
    )


def create_distance_optimization_agent() -> Agent:
    """Create specialist agent for distance optimization tasks."""
    return Agent(
        role="Distance Optimization Specialist",
        goal="Find optimal stops along routes based on distance criteria",
        backstory="""You are an expert in distance-based route optimization. 
        You specialize in:
        - Finding stops at specific points along routes (quarter points, midpoints, etc.)
        - Calculating distances using route polylines
        - Optimizing stop locations based on distance constraints
        - Validating that stops meet distance requirements
        - Working with Place IDs and route geometry
        
        You use Google Maps Directions and Places APIs extensively.""",
        verbose=True,
        tools=maps_tools,
        llm=AGENT_MODEL,  # Explicitly set Ollama model
        max_iter=15  # Limit iterations to prevent recursion issues
    )


def create_time_optimization_agent() -> Agent:
    """Create specialist agent for time optimization tasks."""
    return Agent(
        role="Time Optimization Specialist",
        goal="Find equidistant meeting points and optimize routes based on travel time",
        backstory="""You are an expert in time-based route optimization. 
        You excel at:
        - Finding equidistant meeting points from multiple locations
        - Optimizing routes based on travel time
        - Calculating time-based distances
        - Finding locations that minimize travel time for all parties
        - Working with Distance Matrix API for time calculations
        
        You use Google Maps Distance Matrix and Directions APIs.""",
        verbose=True,
        tools=maps_tools,
        llm=AGENT_MODEL,  # Explicitly set Ollama model
        max_iter=15  # Limit iterations to prevent recursion issues
    )


def create_place_finding_agent() -> Agent:
    """Create specialist agent for place finding tasks."""
    return Agent(
        role="Place Finding Specialist",
        goal="Find locations with specific geographic and categorical constraints",
        backstory="""You are an expert in location discovery and geographic constraints. 
        You specialize in:
        - Finding places with specific types (restaurants, hotels, parks, etc.)
        - Applying geographic constraints (latitude/longitude boundaries)
        - Filtering by ratings and amenities
        - Finding places within countries or regions
        - Validating location relationships (north/south/east/west)
        
        You use Google Maps Places API and Geocoding API extensively.""",
        verbose=True,
        tools=maps_tools,
        llm=AGENT_MODEL,  # Explicitly set Ollama model
        max_iter=15  # Limit iterations to prevent recursion issues
    )


def create_task_for_agent(
    question: str,
    output_format: Dict[str, Any],
    agent: Agent,
    category: str
) -> Task:
    """Create a CrewAI task for a specific agent."""
    return Task(
        description=f"""
        Category: {category}
        
        Question: {question}
        
        Output Format (JSON):
        {json.dumps(output_format, indent=2)}
        
        Please analyze the question carefully and use Google Maps APIs to generate 
        a response that exactly matches the required output format. Ensure all 
        criteria are met and provide accurate Place IDs, addresses, and other 
        required information.
        """,
        agent=agent,
        expected_output="JSON response matching the specified output format with all required fields populated"
    )


def create_orchestration_task(
    question: str,
    output_format: Dict[str, Any],
    category: str
) -> Task:
    """Create orchestration task that delegates to specialists."""
    orchestrator = create_orchestrator_agent()
    
    # Create specialist agents
    route_agent = create_route_planning_agent()
    distance_agent = create_distance_optimization_agent()
    time_agent = create_time_optimization_agent()
    place_agent = create_place_finding_agent()
    
    # Map category to specialist
    category_to_agent = {
        "Route Planning": route_agent,
        "Distance Optimization": distance_agent,
        "Time Optimization": time_agent,
        "Place Finding": place_agent
    }
    
    specialist = category_to_agent.get(category, route_agent)
    
    return Task(
        description=f"""
        Category: {category}
        
        Question: {question}
        
        Output Format (JSON):
        {json.dumps(output_format, indent=2)}
        
        Analyze this navigation question and delegate to the appropriate specialist 
        agent. The specialist should use Google Maps APIs to generate a response 
        that exactly matches the required output format.
        """,
        agent=orchestrator,
        expected_output="JSON response matching the specified output format",
        tools=maps_tools
    )

