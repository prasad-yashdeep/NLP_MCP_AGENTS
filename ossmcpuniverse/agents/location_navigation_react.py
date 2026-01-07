"""
ReAct-style Location Navigation Agent.

Implements Thought-Action-Observation loop for location navigation tasks,
following the pattern from MCP-Universe/mcpuniverse/agent/react.py.
"""

import logging
import json
import re
from typing import Optional, Dict, List, Any

from .base import OllamaAgent
from .models import LocationNavigationOutput
from .route_planning_agent import RoutePlanningAgent
from .place_search_agent import PlaceSearchAgent
from .elevation_agent import ElevationAgent
from .data_synthesis_agent import DataSynthesisAgent

logger = logging.getLogger(__name__)


REACT_LOCATION_NAVIGATION_PROMPT = """You are a ReAct agent for location navigation and route planning tasks.

Available Actions:
1. "plan_routes": Plan driving routes between cities with intermediate stops
2. "search_places": Find specific types of places in given cities
3. "get_elevations": Retrieve elevation data for viewpoints

When All Data Collected:
- Provide final answer using the "answer" key (NOT as an action)

Your task is to answer location planning questions by iteratively:
- THINKING about what information you need next
- ACTING to gather route, place, or elevation data
- OBSERVING results and updating your plan

For complex multi-route tasks:
- Plan all routes first to understand the geography
- Search for places systematically (all rest stops, then all viewpoints)
- Get elevations for all viewpoints together
- Provide final answer when all data is collected

Response Format - JSON only:

Plan routes:
{{
  "thought": "Reasoning about route planning",
  "action": {{
    "type": "plan_routes",
    "origin": "City A",
    "destination": "City B",
    "num_routes": 3,
    "intermediate_cities": 4
  }}
}}

Search places:
{{
  "thought": "Need to find rest stops",
  "action": {{
    "type": "search_places",
    "place_type": "rest_stop",
    "count": 2
  }}
}}

Get elevations:
{{
  "thought": "Need elevation data",
  "action": {{
    "type": "get_elevations"
  }}
}}

Final answer (use "answer" key, NOT action type):
{{
  "thought": "All data collected, ready to provide final output",
  "answer": {{
    "starting_city": "Origin",
    "destination_city": "Destination",
    "routes": [...]
  }}
}}

Maximum {max_steps} iterations."""


class LocationNavigationReAct(OllamaAgent):
    """
    ReAct-style location navigation agent.
    """

    def __init__(
        self,
        orchestrator_model: str = "gemma3:27b",
        worker_model: str = "gemma3:4b",
        ollama_url: Optional[str] = None,
        max_iterations: int = 10
    ):
        super().__init__(model=orchestrator_model, ollama_url=ollama_url)

        self._worker_model = worker_model
        self._max_iterations = max_iterations
        self._history: List[str] = []
        self._ollama_url = ollama_url

        # Sub-agents
        self.route_agent: Optional[RoutePlanningAgent] = None
        self.place_agent: Optional[PlaceSearchAgent] = None
        self.elevation_agent: Optional[ElevationAgent] = None
        self.synthesis_agent: Optional[DataSynthesisAgent] = None

        self._initialized = False

        # State tracking
        self._origin = None
        self._destination = None
        self._planned_routes = []
        self._rest_stops = {}
        self._viewpoints = {}
        self._elevations = {}

    async def initialize(self):
        if self._initialized:
            return

        from .tools import MCPToolWrapper

        self._tool_wrapper = MCPToolWrapper()
        await self._tool_wrapper.initialize(servers=["google-maps"])

        self.route_agent = RoutePlanningAgent(
            model=self._worker_model,
            ollama_url=self._ollama_url,
            tool_wrapper=self._tool_wrapper
        )
        await self.route_agent.initialize()

        self.place_agent = PlaceSearchAgent(
            model=self._worker_model,
            ollama_url=self._ollama_url,
            tool_wrapper=self._tool_wrapper
        )
        await self.place_agent.initialize()

        self.elevation_agent = ElevationAgent(
            model=self._worker_model,
            ollama_url=self._ollama_url,
            tool_wrapper=self._tool_wrapper
        )
        await self.elevation_agent.initialize()

        self.synthesis_agent = DataSynthesisAgent(
            model=self._worker_model,
            ollama_url=self._ollama_url
        )

        self._initialized = True
        logger.info("Location Navigation ReAct Agent initialized")

    async def cleanup(self):
        if self.route_agent:
            await self.route_agent.cleanup()
        if self.place_agent:
            await self.place_agent.cleanup()
        if self.elevation_agent:
            await self.elevation_agent.cleanup()
        if self._tool_wrapper:
            await self._tool_wrapper.cleanup()
        self._initialized = False

    def _build_prompt(self, question: str) -> str:
        prompt_parts = [
            REACT_LOCATION_NAVIGATION_PROMPT.format(max_steps=self._max_iterations),
            f"\nQuestion: {question}",
        ]

        if self._history:
            prompt_parts.append("\nPrevious Steps:")
            prompt_parts.append("\n".join(self._history))

        prompt_parts.append("\nWhat is your next step? Respond with JSON only.")
        return "\n".join(prompt_parts)

    def _add_history(self, step_num: int, thought: str = "", action: str = "", observation: str = ""):
        entry_parts = [f"Step {step_num}:"]
        if thought:
            entry_parts.append(f"Thought: {thought}")
        if action:
            entry_parts.append(f"Action: {action}")
        if observation:
            entry_parts.append(f"Observation: {observation}")
        self._history.append("\n".join(entry_parts))

    async def search(
        self,
        question: str,
        output_format: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if not self._initialized:
            await self.initialize()

        logger.info(f"=== ReAct Location Navigation: {question[:100]}... ===")

        self._history = []
        self._planned_routes = []
        self._rest_stops = {}
        self._viewpoints = {}
        self._elevations = {}

        for iteration in range(self._max_iterations):
            logger.info(f"\n=== Iteration {iteration + 1}/{self._max_iterations} ===")

            prompt = self._build_prompt(question)
            response = self.generate(prompt)
            logger.info(f"LLM Response: {response[:200]}...")

            try:
                response = response.strip().strip('`').strip()
                if response.startswith("json"):
                    response = response[4:].strip()
                parsed = json.loads(response)

                if "thought" not in parsed:
                    raise ValueError("No 'thought' in response")

                thought = parsed["thought"]
                logger.info(f"Thought: {thought}")

                if "answer" in parsed:
                    # LLM wants to provide final answer - synthesize from collected data
                    logger.info(f"Final Answer requested - synthesizing from collected data")
                    self._add_history(iteration + 1, thought=thought, action="ANSWER", observation="Synthesizing output")

                    if self._planned_routes:
                        output = await self.synthesis_agent.synthesize(
                            self._origin or "Origin",
                            self._destination or "Destination",
                            self._planned_routes,
                            self._rest_stops,
                            self._viewpoints,
                            self._elevations
                        )
                        return output.to_benchmark_format()
                    else:
                        # No routes planned - try to use LLM's answer if it has correct structure
                        llm_answer = parsed["answer"]
                        if isinstance(llm_answer, dict) and "routes" in llm_answer:
                            return llm_answer
                        else:
                            return {"error": "No routes planned and invalid answer format"}

                if "action" not in parsed:
                    raise ValueError("No 'action' or 'answer'")

                action = parsed["action"]
                action_type = action.get("type", "")

                if action_type == "plan_routes":
                    logger.info("Action: PLAN_ROUTES")
                    self._origin = action.get("origin", "")
                    self._destination = action.get("destination", "")
                    num_routes = action.get("num_routes", 1)
                    intermediate = action.get("intermediate_cities", 2)

                    self._planned_routes = await self.route_agent.plan_routes(
                        self._origin,
                        self._destination,
                        num_routes,
                        intermediate
                    )

                    observation = f"Planned {len(self._planned_routes)} routes with {intermediate} intermediate cities each"
                    logger.info(f"Observation: {observation}")
                    self._add_history(iteration + 1, thought=thought, action="PLAN_ROUTES", observation=observation)

                elif action_type == "search_places":
                    place_type = action.get("place_type", "rest_stop")
                    count = action.get("count", 2)
                    logger.info(f"Action: SEARCH_PLACES ({place_type})")

                    all_cities = []
                    for route in self._planned_routes:
                        all_cities.extend(route["cities_visited"])

                    places = await self.place_agent.search_places(
                        all_cities,
                        place_type,
                        count_per_city=1
                    )

                    # Distribute places across routes
                    for idx, route in enumerate(self._planned_routes):
                        route_id = route["route_id"]
                        route_places = places[idx * count:(idx + 1) * count] if places else []

                        if place_type == "rest_stop":
                            self._rest_stops[route_id] = route_places
                        elif place_type == "scenic_viewpoint":
                            self._viewpoints[route_id] = route_places

                    observation = f"Found {len(places)} {place_type}s"
                    logger.info(f"Observation: {observation}")
                    self._add_history(iteration + 1, thought=thought, action=f"SEARCH_PLACES: {place_type}", observation=observation)

                elif action_type == "get_elevations":
                    logger.info("Action: GET_ELEVATIONS")

                    all_viewpoints = []
                    for viewpoint_list in self._viewpoints.values():
                        all_viewpoints.extend(viewpoint_list)

                    places_for_elevation = [
                        {"name": vp.name, "address": vp.address}
                        for vp in all_viewpoints
                    ]

                    self._elevations = await self.elevation_agent.get_elevations(places_for_elevation)

                    observation = f"Retrieved elevations for {len(self._elevations)} viewpoints"
                    logger.info(f"Observation: {observation}")
                    self._add_history(iteration + 1, thought=thought, action="GET_ELEVATIONS", observation=observation)

                elif action_type == "answer":
                    # Fallback: LLM mistakenly used "answer" as action type
                    logger.info("Action: ANSWER (synthesizing from collected data)")
                    if self._planned_routes:
                        output = await self.synthesis_agent.synthesize(
                            self._origin or "Origin",
                            self._destination or "Destination",
                            self._planned_routes,
                            self._rest_stops,
                            self._viewpoints,
                            self._elevations
                        )
                        return output.to_benchmark_format()
                    else:
                        observation = "Cannot answer: no routes planned yet"
                        logger.warning(observation)
                        self._add_history(iteration + 1, thought=thought, action="ANSWER", observation=observation)

                else:
                    observation = f"Unknown action: {action_type}. Valid actions: plan_routes, search_places, get_elevations. To provide final answer, use 'answer' key (not action type)."
                    logger.warning(observation)
                    self._add_history(iteration + 1, thought=thought, action=str(action), observation=observation)

            except (json.JSONDecodeError, ValueError, KeyError) as e:
                error_msg = f"Failed to parse response: {e}"
                logger.error(error_msg)
                self._add_history(iteration + 1, thought="", action="ERROR", observation=error_msg)
                continue

        # Max iterations reached - synthesize best answer
        logger.warning("Max iterations reached, synthesizing output")

        if self._planned_routes:
            output = await self.synthesis_agent.synthesize(
                self._origin or "Origin",
                self._destination or "Destination",
                self._planned_routes,
                self._rest_stops,
                self._viewpoints,
                self._elevations
            )
            return output.to_benchmark_format()

        return {"error": "Unable to complete within iteration limit"}
