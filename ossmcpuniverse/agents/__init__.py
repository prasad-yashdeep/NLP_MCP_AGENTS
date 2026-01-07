"""
Multi-Agent System for OSS MCP Universe.

This package provides multi-agent architectures using Ollama models
with an orchestrator-worker pattern for:
- Financial analysis (using yfinance and calculator MCP servers)
- Web search (using google-search and fetch MCP servers)
- Location navigation (using google-maps MCP server)
"""

from .base import OllamaAgent
from .models import (
    TaskPlan,
    StockData,
    CalculationResult,
    FinalOutput,
    SearchPlan,
    SearchResult,
    FetchedContent,
    ExtractedInfo,
    WebSearchOutput,
    PlaceInfo,
    RestStop,
    ScenicViewpoint,
    RouteDetails,
    LocationNavigationOutput,
)
from .data_agent import DataAgent
from .calculator_agent import CalculatorAgent
from .formatter_agent import FormatterAgent
from .orchestrator import OrchestratorAgent
from .financial_manager import FinancialAnalysisManager
from .search_agent import SearchAgent
from .fetch_agent import FetchAgent
from .synthesis_agent import SynthesisAgent
from .web_search_orchestrator import WebSearchOrchestrator
from .web_search_manager import WebSearchManager
from .web_search_react import WebSearchReAct
from .route_planning_agent import RoutePlanningAgent
from .place_search_agent import PlaceSearchAgent
from .elevation_agent import ElevationAgent
from .data_synthesis_agent import DataSynthesisAgent
from .location_navigation_react import LocationNavigationReAct

__all__ = [
    # Base
    "OllamaAgent",
    # Financial Analysis
    "TaskPlan",
    "StockData",
    "CalculationResult",
    "FinalOutput",
    "DataAgent",
    "CalculatorAgent",
    "FormatterAgent",
    "OrchestratorAgent",
    "FinancialAnalysisManager",
    # Web Search
    "SearchPlan",
    "SearchResult",
    "FetchedContent",
    "ExtractedInfo",
    "WebSearchOutput",
    "SearchAgent",
    "FetchAgent",
    "SynthesisAgent",
    "WebSearchOrchestrator",
    "WebSearchManager",
    "WebSearchReAct",
    # Location Navigation
    "PlaceInfo",
    "RestStop",
    "ScenicViewpoint",
    "RouteDetails",
    "LocationNavigationOutput",
    "RoutePlanningAgent",
    "PlaceSearchAgent",
    "ElevationAgent",
    "DataSynthesisAgent",
    "LocationNavigationReAct",
]
