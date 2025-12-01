"""
Pydantic models for structured communication between agents.

These models ensure type-safe data flow between the orchestrator
and specialized sub-agents in the multi-agent financial analysis system.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class TaskPlan(BaseModel):
    """Output from Orchestrator's planning phase."""

    tickers: List[str] = Field(
        description="Stock ticker symbols (e.g., ['MSFT', 'AAPL'])"
    )
    start_date: str = Field(
        description="Start date in YYYY-MM-DD format"
    )
    end_date: str = Field(
        description="End date in YYYY-MM-DD format"
    )
    investment: float = Field(
        description="Total investment amount in dollars"
    )
    splits: List[float] = Field(
        description="Allocation percentages for each ticker (should sum to 1.0)"
    )
    output_format: Optional[Dict[str, str]] = Field(
        default=None,
        description="Expected output format from the task"
    )

    def validate_splits(self) -> bool:
        """Validate that splits sum to approximately 1.0."""
        return abs(sum(self.splits) - 1.0) < 0.01


class StockData(BaseModel):
    """Output from Data Agent - stock price information."""

    ticker: str = Field(description="Stock ticker symbol")
    start_price: float = Field(description="Stock price at start date")
    end_price: float = Field(description="Stock price at end date")
    start_date: str = Field(description="Actual start date used")
    end_date: str = Field(description="Actual end date used")


class CalculationResult(BaseModel):
    """Output from Calculator Agent."""

    total_value: float = Field(
        description="Final portfolio value"
    )
    percentage_return: float = Field(
        description="Total percentage return"
    )
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Detailed breakdown by ticker"
    )


class FinalOutput(BaseModel):
    """Final formatted output matching benchmark expected format."""

    total_value: str = Field(
        description="Formatted total value as string"
    )
    total_percentage_return: str = Field(
        description="Formatted percentage return as string"
    )

    def to_benchmark_format(self) -> Dict[str, str]:
        """Convert to the format expected by the benchmark."""
        return {
            "total value": self.total_value,
            "total percentage return": self.total_percentage_return
        }


class AgentMessage(BaseModel):
    """Message passed between agents."""

    sender: str = Field(description="Name of the sending agent")
    receiver: str = Field(description="Name of the receiving agent")
    content: Any = Field(description="Message content (varies by agent)")
    message_type: str = Field(
        default="data",
        description="Type of message: 'data', 'instruction', 'result', 'error'"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata"
    )


# ============================================================================
# Web Search Multi-Agent Models
# ============================================================================

class SearchPlan(BaseModel):
    """Output from Orchestrator's planning phase for web search."""

    main_query: str = Field(description="Main question to answer")
    search_queries: List[str] = Field(description="Decomposed search queries")
    strategy: str = Field(description="Search strategy description")
    expected_answer_type: Optional[str] = Field(
        default=None,
        description="Type of answer expected (e.g., 'person_name', 'number', 'date')"
    )


class SearchResult(BaseModel):
    """Output from Search Agent - search result for a single query."""

    query: str = Field(description="The search query executed")
    results: List[Dict[str, Any]] = Field(
        description="List of search results with title, url, snippet, position"
    )
    total_results: int = Field(
        default=0,
        description="Total number of results found"
    )


class FetchedContent(BaseModel):
    """Output from Fetch Agent - content from a webpage."""

    url: str = Field(description="URL of the fetched page")
    content: str = Field(description="Extracted text content")
    title: Optional[str] = Field(default=None, description="Page title")
    status: str = Field(
        default="success",
        description="Fetch status: 'success', 'error', 'timeout'"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if fetch failed"
    )


class ExtractedInfo(BaseModel):
    """Information extracted from sources by Synthesis Agent."""

    facts: List[str] = Field(description="Relevant facts extracted from sources")
    sources: List[str] = Field(description="URLs of sources used")
    confidence: float = Field(
        description="Confidence score (0.0 to 1.0) in the extracted information"
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Reasoning behind the extracted information"
    )


class WebSearchOutput(BaseModel):
    """Final formatted output for web search tasks."""

    answer: str = Field(description="The final answer to the query")
    confidence: Optional[float] = Field(
        default=None,
        description="Confidence in the answer"
    )
    sources: Optional[List[str]] = Field(
        default=None,
        description="Source URLs used to generate the answer"
    )

    def to_benchmark_format(self) -> Dict[str, str]:
        """Convert to the format expected by the benchmark."""
        return {"answer": self.answer}


# ============================================================================
# Location Navigation Multi-Agent Models
# ============================================================================

class PlaceInfo(BaseModel):
    """Information about a place (rest stop, viewpoint, restaurant, etc.)"""

    name: str = Field(description="Place name")
    address: str = Field(description="Place address")
    place_id: Optional[str] = Field(
        default=None,
        description="Google Maps place ID"
    )
    rating: Optional[float] = Field(
        default=None,
        description="Place rating (0.0-5.0)"
    )
    types: Optional[List[str]] = Field(
        default=None,
        description="Place types from Google Maps"
    )
    location: Optional[Dict[str, float]] = Field(
        default=None,
        description="Lat/lng coordinates"
    )


class RestStop(BaseModel):
    """Rest stop with amenities for a route."""

    city: str = Field(description="City where rest stop is located")
    rest_stop_id: str = Field(description="Unique ID for this rest stop")
    name: str = Field(description="Rest stop name")
    address: str = Field(description="Rest stop address")
    amenities: List[str] = Field(
        description="Available amenities (WiFi, Restrooms, etc.)"
    )


class ScenicViewpoint(BaseModel):
    """Scenic viewpoint with elevation data."""

    city: str = Field(description="City where viewpoint is located")
    viewpoint_id: str = Field(description="Unique ID for this viewpoint")
    name: str = Field(description="Viewpoint name")
    address: str = Field(description="Viewpoint address")
    elevation_meters: str = Field(
        description="Elevation in meters (as string for exact format)"
    )
    description: str = Field(description="Brief description of the viewpoint")


class RouteDetails(BaseModel):
    """Complete details for a single route."""

    route_id: str = Field(description="Unique route identifier")
    route_name: str = Field(description="Descriptive name for the route")
    cities_visited: List[str] = Field(
        description="List of intermediate cities visited on this route"
    )
    rest_stops: List[RestStop] = Field(
        description="Rest stops along this route"
    )
    scenic_viewpoints: List[ScenicViewpoint] = Field(
        description="Scenic viewpoints along this route"
    )


class LocationNavigationOutput(BaseModel):
    """Complete output for location navigation tasks."""

    starting_city: str = Field(description="Origin city")
    destination_city: str = Field(description="Destination city")
    routes: List[RouteDetails] = Field(
        description="List of route options with details"
    )

    def to_benchmark_format(self) -> Dict[str, Any]:
        """Convert to exact format required by evaluators."""
        return self.model_dump()
