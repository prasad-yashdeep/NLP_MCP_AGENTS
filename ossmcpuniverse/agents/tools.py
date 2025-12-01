"""
MCP Tool wrappers for the multi-agent system.

This module provides simplified interfaces to MCP servers (yfinance, calculator)
that can be used by the specialized agents.
"""

import os
import json
import asyncio
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

# Import MCP client from MCP-Universe
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'MCP-Universe'))

from mcpuniverse.mcp.manager import MCPManager
from mcpuniverse.mcp.client import MCPClient

logger = logging.getLogger(__name__)


class MCPToolWrapper:
    """
    Wrapper class for MCP tools that provides a simplified interface.

    This class manages MCP server connections and provides easy-to-use
    methods for calling MCP tools.
    """

    def __init__(self, server_configs_path: Optional[str] = None):
        """
        Initialize the MCP tool wrapper.

        Args:
            server_configs_path: Path to MCP server configurations (optional)
        """
        self._mcp_manager: Optional[MCPManager] = None
        self._clients: Dict[str, MCPClient] = {}
        self._initialized = False
        self._server_configs_path = server_configs_path

    async def initialize(self, servers: List[str] = None):
        """
        Initialize MCP clients for the specified servers.

        Args:
            servers: List of server names to connect to (default: yfinance, calculator)
        """
        if self._initialized:
            return

        if servers is None:
            servers = ["yfinance", "calculator"]

        # Create MCP manager - use default config if no path specified
        self._mcp_manager = MCPManager(config=self._server_configs_path)

        # Connect to servers
        for server_name in servers:
            try:
                client = await self._mcp_manager.build_client(server_name)
                self._clients[server_name] = client
                logger.info(f"Connected to MCP server: {server_name}")
            except Exception as e:
                logger.error(f"Failed to connect to {server_name}: {e}")

        self._initialized = True

    async def cleanup(self):
        """Cleanup MCP connections."""
        for name, client in list(self._clients.items()):
            try:
                await client.cleanup()
            except asyncio.CancelledError:
                # Expected during cleanup, silently ignore
                logger.debug(f"Cleanup cancelled for {name} (expected during shutdown)")
            except Exception as e:
                # Check if it's the known cancel scope error from MCP library
                error_msg = str(e)
                if "cancel scope" in error_msg.lower() or "exit" in error_msg.lower():
                    # This is expected during MCP client cleanup, log as debug
                    logger.debug(f"MCP client cleanup for {name}: {e}")
                else:
                    # Unexpected error, log as warning
                    logger.warning(f"Error during cleanup of {name}: {e}")
        self._clients = {}
        self._initialized = False

    async def call_tool(
        self,
        server: str,
        tool: str,
        arguments: Dict[str, Any]
    ) -> str:
        """
        Call an MCP tool.

        Args:
            server: Server name (e.g., "yfinance", "calculator")
            tool: Tool name
            arguments: Tool arguments

        Returns:
            Tool response as string
        """
        if not self._initialized:
            raise RuntimeError("MCPToolWrapper not initialized. Call initialize() first.")

        if server not in self._clients:
            raise ValueError(f"Server '{server}' not connected")

        try:
            result = await self._clients[server].execute_tool(tool, arguments)

            # Extract text content from result
            if hasattr(result, 'content'):
                if result.content and len(result.content) > 0:
                    content = result.content[0]
                    if hasattr(content, 'text'):
                        return content.text
                    if hasattr(content, 'data'):
                        return str(content.data)
            return str(result)

        except Exception as e:
            logger.error(f"Error calling {server}.{tool}: {e}")
            raise RuntimeError(f"Tool call failed: {e}") from e

    async def list_tools(self, server: str) -> List[str]:
        """List available tools for a server."""
        if server not in self._clients:
            return []
        tools = await self._clients[server].list_tools()
        return [tool.name for tool in tools]


# Convenience functions for common operations

async def get_stock_price(
    tool_wrapper: MCPToolWrapper,
    ticker: str,
    date: str
) -> Optional[float]:
    """
    Get stock price for a given ticker and date.

    Uses get_historical_stock_prices to fetch data for a range around the date,
    then extracts the closing price for the specific date.

    Args:
        tool_wrapper: Initialized MCPToolWrapper
        ticker: Stock ticker symbol (e.g., "MSFT")
        date: Date in YYYY-MM-DD format

    Returns:
        Stock price (Close) as float, or None if not available
    """
    try:
        # Fetch historical data for a small range around the date
        # This handles weekends/holidays by getting nearby data
        from datetime import datetime, timedelta
        target_date = datetime.strptime(date, "%Y-%m-%d")
        start_date = (target_date - timedelta(days=5)).strftime("%Y-%m-%d")
        end_date = (target_date + timedelta(days=5)).strftime("%Y-%m-%d")

        result = await tool_wrapper.call_tool(
            server="yfinance",
            tool="get_historical_stock_prices",
            arguments={
                "ticker": ticker,
                "start_date": start_date,
                "end_date": end_date,
                "interval": "1d"
            }
        )

        # Parse the result - it returns JSON data
        if isinstance(result, str):
            try:
                data = json.loads(result)
                if isinstance(data, list):
                    prices = {}
                    for entry in data:
                        # Date format: "2023-01-09T05:00:00.000Z"
                        entry_date = entry.get("Date", "")[:10]  # Extract YYYY-MM-DD
                        close_price = entry.get("Close")
                        if entry_date and close_price is not None:
                            prices[entry_date] = float(close_price)

                    # Find exact date or closest date
                    if date in prices:
                        return prices[date]

                    # Find closest available date
                    available_dates = sorted(prices.keys())
                    for d in available_dates:
                        if d >= date:
                            return prices[d]
                    if available_dates:
                        return prices[available_dates[-1]]
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON response for {ticker}")

        return None

    except Exception as e:
        logger.error(f"Failed to get stock price for {ticker} on {date}: {e}")
        return None


async def calculate(
    tool_wrapper: MCPToolWrapper,
    expression: str
) -> Optional[float]:
    """
    Perform a calculation using the calculator MCP server.

    Args:
        tool_wrapper: Initialized MCPToolWrapper
        expression: Mathematical expression to evaluate

    Returns:
        Calculation result as float
    """
    try:
        result = await tool_wrapper.call_tool(
            server="calculator",
            tool="calculate",
            arguments={"expression": expression}
        )

        # Parse the result
        if isinstance(result, str):
            try:
                data = json.loads(result)
                return float(data.get("result", 0))
            except json.JSONDecodeError:
                # Try to extract number
                import re
                numbers = re.findall(r'-?[\d.]+', result)
                if numbers:
                    return float(numbers[-1])  # Take last number as result
        return float(result) if result else None

    except Exception as e:
        logger.error(f"Calculation failed for '{expression}': {e}")
        return None


async def get_stock_data_range(
    tool_wrapper: MCPToolWrapper,
    ticker: str,
    start_date: str,
    end_date: str
) -> Dict[str, Any]:
    """
    Get stock data for a date range.

    Args:
        tool_wrapper: Initialized MCPToolWrapper
        ticker: Stock ticker symbol
        start_date: Start date YYYY-MM-DD
        end_date: End date YYYY-MM-DD

    Returns:
        Dictionary with start_price, end_price, and metadata
    """
    start_price = await get_stock_price(tool_wrapper, ticker, start_date)
    end_price = await get_stock_price(tool_wrapper, ticker, end_date)

    return {
        "ticker": ticker,
        "start_date": start_date,
        "end_date": end_date,
        "start_price": start_price,
        "end_price": end_price
    }


# ============================================================================
# Web Search Tool Wrappers
# ============================================================================

async def google_search(
    tool_wrapper: MCPToolWrapper,
    query: str,
    num_results: int = 10
) -> List[Dict[str, str]]:
    """
    Execute a Google search query via the google-search MCP server.

    Args:
        tool_wrapper: Initialized MCPToolWrapper
        query: Search query string
        num_results: Number of results to return (default: 10)

    Returns:
        List of search results with 'title', 'url', 'snippet' fields
    """
    try:
        result = await tool_wrapper.call_tool(
            server="google-search",
            tool="search",  # Correct tool name is "search"
            arguments={
                "query": query,
                "num_results": num_results
            }
        )

        # Parse the result - google-search returns newline-separated JSON objects with indentation
        if isinstance(result, str):
            # Check for error response
            try:
                error_check = json.loads(result)
                if isinstance(error_check, dict) and "error" in error_check:
                    logger.error(f"Search error: {error_check['error']}")
                    return []
            except json.JSONDecodeError:
                pass  # Not a single JSON object, continue with parsing

            # Parse multiple JSON objects separated by "}\n{"
            # Add brackets to make it a valid JSON array
            array_str = result.strip()
            if not array_str.startswith('['):
                # Replace "}\n{" with "},\n{" to separate objects
                array_str = array_str.replace('}\n{', '},\n{')
                array_str = '[' + array_str + ']'

            try:
                items = json.loads(array_str)
                results = []
                for item in items:
                    # Convert to expected format with 'url' instead of 'link'
                    results.append({
                        "title": item.get("title", ""),
                        "url": item.get("link", ""),
                        "snippet": item.get("snippet", ""),
                        "position": item.get("position", 0)
                    })
                return results
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse search results: {str(e)}")
                return []

        return []

    except Exception as e:
        logger.error(f"Google search failed for '{query}': {e}")
        return []


async def fetch_url(
    tool_wrapper: MCPToolWrapper,
    url: str,
    max_length: int = 10000
) -> Dict[str, Any]:
    """
    Fetch webpage content via the fetch MCP server.

    Args:
        tool_wrapper: Initialized MCPToolWrapper
        url: URL to fetch
        max_length: Maximum content length to retrieve (default: 10000)

    Returns:
        Dictionary with 'url', 'content', 'title', 'status' fields
    """
    try:
        result = await tool_wrapper.call_tool(
            server="fetch",
            tool="fetch",
            arguments={
                "url": url,
                "max_length": max_length
            }
        )

        # Parse the result
        if isinstance(result, str):
            try:
                data = json.loads(result)
                return {
                    "url": url,
                    "content": data.get("content", result),
                    "title": data.get("title"),
                    "status": "success"
                }
            except json.JSONDecodeError:
                # If not JSON, treat the whole result as content
                return {
                    "url": url,
                    "content": result[:max_length],
                    "title": None,
                    "status": "success"
                }

        return {
            "url": url,
            "content": "",
            "title": None,
            "status": "error",
            "error_message": "Empty response"
        }

    except Exception as e:
        logger.error(f"Failed to fetch URL '{url}': {e}")
        return {
            "url": url,
            "content": "",
            "title": None,
            "status": "error",
            "error_message": str(e)
        }


async def fetch_multiple_urls(
    tool_wrapper: MCPToolWrapper,
    urls: List[str],
    max_length: int = 10000
) -> List[Dict[str, Any]]:
    """
    Fetch content from multiple URLs in parallel.

    Args:
        tool_wrapper: Initialized MCPToolWrapper
        urls: List of URLs to fetch
        max_length: Maximum content length per URL

    Returns:
        List of dictionaries with fetched content
    """
    tasks = [fetch_url(tool_wrapper, url, max_length) for url in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out exceptions and return successful results
    fetched = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Failed to fetch {urls[i]}: {result}")
            fetched.append({
                "url": urls[i],
                "content": "",
                "title": None,
                "status": "error",
                "error_message": str(result)
            })
        else:
            fetched.append(result)

    return fetched


# ============================================================================
# Google Maps MCP Tool Wrappers
# ============================================================================

async def google_maps_search_places(
    tool_wrapper: MCPToolWrapper,
    query: str
) -> List[Dict[str, Any]]:
    """
    Search for places using Google Maps.

    Args:
        tool_wrapper: Initialized MCPToolWrapper
        query: Search query (e.g., "restaurants in Tokyo")

    Returns:
        List of place dictionaries with name, address, place_id, rating, types
    """
    try:
        result = await tool_wrapper.call_tool(
            server="google-maps",
            tool="maps_search_places",
            arguments={"query": query}
        )

        places = json.loads(result) if isinstance(result, str) else result
        logger.debug(f"Found {len(places)} places for query: {query}")
        return places

    except Exception as e:
        logger.error(f"Google Maps search failed for '{query}': {e}")
        return []


async def google_maps_get_elevation(
    tool_wrapper: MCPToolWrapper,
    locations: List[Dict[str, float]]
) -> List[Dict[str, Any]]:
    """
    Get elevation data for coordinates.

    Args:
        tool_wrapper: Initialized MCPToolWrapper
        locations: List of {lat, lng} coordinate dictionaries

    Returns:
        List of elevation dictionaries with elevation in meters
    """
    try:
        result = await tool_wrapper.call_tool(
            server="google-maps",
            tool="maps_elevation",
            arguments={"locations": locations}
        )

        elevations = json.loads(result) if isinstance(result, str) else result
        logger.debug(f"Retrieved elevation data for {len(locations)} locations")
        return elevations

    except Exception as e:
        logger.error(f"Google Maps elevation failed: {e}")
        return []


async def google_maps_distance_matrix(
    tool_wrapper: MCPToolWrapper,
    origins: List[str],
    destinations: List[str],
    mode: str = "driving"
) -> Dict[str, Any]:
    """
    Calculate distances and travel times between origins and destinations.

    Args:
        tool_wrapper: Initialized MCPToolWrapper
        origins: List of origin addresses
        destinations: List of destination addresses
        mode: Travel mode ('driving', 'walking', 'transit', 'bicycling')

    Returns:
        Distance matrix with distances and durations
    """
    try:
        result = await tool_wrapper.call_tool(
            server="google-maps",
            tool="maps_distance_matrix",
            arguments={
                "origins": origins,
                "destinations": destinations,
                "mode": mode
            }
        )

        matrix = json.loads(result) if isinstance(result, str) else result
        logger.debug(f"Distance matrix: {len(origins)} origins x {len(destinations)} destinations")
        return matrix

    except Exception as e:
        logger.error(f"Google Maps distance matrix failed: {e}")
        return {"rows": []}


async def google_maps_geocode(
    tool_wrapper: MCPToolWrapper,
    address: str
) -> Dict[str, Any]:
    """
    Convert address to coordinates.

    Args:
        tool_wrapper: Initialized MCPToolWrapper
        address: Address string to geocode

    Returns:
        Dictionary with lat, lng, formatted_address
    """
    try:
        result = await tool_wrapper.call_tool(
            server="google-maps",
            tool="maps_geocode",
            arguments={"address": address}
        )

        geocode_data = json.loads(result) if isinstance(result, str) else result
        logger.debug(f"Geocoded address: {address}")
        return geocode_data

    except Exception as e:
        logger.error(f"Google Maps geocode failed for '{address}': {e}")
        return {}


async def google_maps_place_details(
    tool_wrapper: MCPToolWrapper,
    place_id: str
) -> Dict[str, Any]:
    """
    Get detailed information about a place.

    Args:
        tool_wrapper: Initialized MCPToolWrapper
        place_id: Google Maps place ID

    Returns:
        Dictionary with detailed place information
    """
    try:
        result = await tool_wrapper.call_tool(
            server="google-maps",
            tool="maps_place_details",
            arguments={"place_id": place_id}
        )

        details = json.loads(result) if isinstance(result, str) else result
        logger.debug(f"Retrieved details for place_id: {place_id}")
        return details

    except Exception as e:
        logger.error(f"Google Maps place details failed for '{place_id}': {e}")
        return {}


async def google_maps_reverse_geocode(
    tool_wrapper: MCPToolWrapper,
    lat: float,
    lng: float
) -> Dict[str, Any]:
    """
    Convert coordinates to address.

    Args:
        tool_wrapper: Initialized MCPToolWrapper
        lat: Latitude
        lng: Longitude

    Returns:
        Dictionary with address information
    """
    try:
        result = await tool_wrapper.call_tool(
            server="google-maps",
            tool="maps_reverse_geocode",
            arguments={"lat": lat, "lng": lng}
        )

        address_data = json.loads(result) if isinstance(result, str) else result
        logger.debug(f"Reverse geocoded: ({lat}, {lng})")
        return address_data

    except Exception as e:
        logger.error(f"Google Maps reverse geocode failed for ({lat}, {lng}): {e}")
        return {}
