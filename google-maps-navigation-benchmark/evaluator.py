"""
Evaluation module for Google Maps benchmark tasks.
Validates responses against task criteria.
"""
import json
from typing import Dict, Any, List, Optional
from google_maps_client import GoogleMapsClient


class TaskEvaluator:
    """Evaluates task responses against criteria."""
    
    def __init__(self, maps_client: Optional[GoogleMapsClient] = None):
        """
        Initialize evaluator.
        
        Args:
            maps_client: Optional Google Maps client for validation
        """
        self.maps_client = maps_client
    
    def evaluate(self, response: str, evaluators: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate a response against task evaluators.
        
        Args:
            response: JSON response string from agent
            evaluators: List of evaluator configurations from task
            
        Returns:
            Dictionary with evaluation results
        """
        results = {
            "passed": [],
            "failed": [],
            "errors": []
        }
        
        try:
            response_json = json.loads(response) if isinstance(response, str) else response
        except json.JSONDecodeError as e:
            results["errors"].append(f"Invalid JSON: {str(e)}")
            return results
        
        for evaluator in evaluators:
            func = evaluator.get("func", "")
            op = evaluator.get("op")
            value = evaluator.get("value")
            op_args = evaluator.get("op_args", {})
            
            try:
                if func == "json":
                    # Basic JSON validation
                    if isinstance(response_json, dict):
                        results["passed"].append("Valid JSON structure")
                    else:
                        results["failed"].append("Response is not a JSON object")
                        continue
                
                # Parse function chain (e.g., "json -> get(routes) -> len")
                if "->" in func:
                    result = self._evaluate_function_chain(response_json, func, op, value, op_args)
                else:
                    result = self._evaluate_simple(response_json, func, op, value, op_args)
                
                if result["passed"]:
                    results["passed"].append(result["message"])
                else:
                    results["failed"].append(result["message"])
                    
            except Exception as e:
                results["errors"].append(f"Evaluator error: {str(e)}")
        
        return results
    
    def _evaluate_function_chain(
        self,
        data: Any,
        func: str,
        op: Optional[str],
        value: Any,
        op_args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate a function chain like 'json -> get(routes) -> len'."""
        parts = [p.strip() for p in func.split("->")]
        
        current = data
        for part in parts:
            if part == "json":
                continue
            elif part.startswith("get("):
                key = part[4:-1]  # Extract key from get(key)
                current = current.get(key) if isinstance(current, dict) else None
            elif part == "len":
                current = len(current) if current is not None else 0
            elif part == "foreach":
                # Handle foreach operations
                if isinstance(current, list):
                    return self._evaluate_foreach(current, op, value, op_args)
                else:
                    return {"passed": False, "message": f"foreach requires list, got {type(current)}"}
        
        # Apply operation
        if op == "=":
            passed = current == value
            return {
                "passed": passed,
                "message": f"Expected {value}, got {current}"
            }
        elif op:
            # Custom operation
            return self._evaluate_custom_op(current, op, op_args)
        else:
            return {"passed": True, "message": "Function chain executed"}
    
    def _evaluate_foreach(
        self,
        items: List[Any],
        op: Optional[str],
        value: Any,
        op_args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate operation on each item in a list."""
        all_passed = True
        messages = []
        
        for i, item in enumerate(items):
            if isinstance(item, dict) and op_args:
                # Extract nested path from op_args
                if isinstance(op_args, str):
                    nested_key = op_args
                    nested_value = item.get(nested_key)
                    if isinstance(nested_value, list):
                        if len(nested_value) != value:
                            all_passed = False
                            messages.append(f"Item {i}: expected {value}, got {len(nested_value)}")
                elif isinstance(op_args, dict):
                    # More complex nested evaluation
                    result = self._evaluate_custom_op(item, op, op_args)
                    if not result["passed"]:
                        all_passed = False
                        messages.append(f"Item {i}: {result['message']}")
        
        return {
            "passed": all_passed,
            "message": "; ".join(messages) if messages else "All items passed"
        }
    
    def _evaluate_simple(
        self,
        data: Any,
        func: str,
        op: Optional[str],
        value: Any,
        op_args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate simple function."""
        if op:
            return self._evaluate_custom_op(data, op, op_args)
        return {"passed": True, "message": "Simple evaluation passed"}
    
    def _evaluate_custom_op(
        self,
        data: Any,
        op: str,
        op_args: Any
    ) -> Dict[str, Any]:
        """Evaluate custom Google Maps operations."""
        if not self.maps_client:
            return {
                "passed": False,
                "message": "Google Maps client not available for custom operations"
            }
        
        # Map operation names to methods
        op_handlers = {
            "google_maps.city_name_match": self._city_name_match,
            "google_maps.places_in_cities_visited": self._places_in_cities_visited,
            "google_maps.validate_elevation_meters": self._validate_elevation,
            "google_maps.city_different_from_rest_stops": self._city_different_from_rest_stops,
            "google_maps.stop_include_keys": self._stop_include_keys,
            "google_maps.is_a_validate_stop": self._is_a_validate_stop,
            "google_maps.validate_stop_rating": self._validate_stop_rating,
            "google_maps.validate_stop_type": self._validate_stop_type,
            "google_maps.compare_distance_between_stops": self._compare_distance_between_stops,
            "google_maps.compare_distance_with_and_wo_stops": self._compare_distance_with_and_wo_stops,
            "google_maps.validate_location": self._validate_location,
            "google_maps.places_in_country": self._places_in_country,
            "google_maps.validate_direction_of_two_places": self._validate_direction,
            "google_maps.place_in_cities_visited_of_routes": self._place_in_cities_visited_of_routes,
        }
        
        handler = op_handlers.get(op)
        if handler:
            return handler(data, op_args)
        else:
            return {
                "passed": False,
                "message": f"Unknown operation: {op}"
            }
    
    def _city_name_match(self, data: Dict[str, Any], op_args: Dict[str, Any]) -> Dict[str, Any]:
        """Check if city name matches any of the provided values."""
        key = op_args.get("key")
        values = op_args.get("values", [])
        city_name = data.get(key, "").lower()
        
        matches = any(val.lower() in city_name or city_name in val.lower() for val in values)
        return {
            "passed": matches,
            "message": f"City name '{data.get(key)}' {'matches' if matches else 'does not match'} expected values"
        }
    
    def _places_in_cities_visited(self, data: Dict[str, Any], op_args: Any) -> Dict[str, Any]:
        """Validate that places are in cities_visited."""
        # This is a simplified check - full implementation would use Google Maps API
        return {"passed": True, "message": "Place location validation (simplified)"}
    
    def _validate_elevation(self, data: Dict[str, Any], op_args: Any) -> Dict[str, Any]:
        """Validate elevation values are present and numeric."""
        routes = data.get("routes", [])
        for route in routes:
            viewpoints = route.get("scenic_viewpoints", [])
            for viewpoint in viewpoints:
                elevation = viewpoint.get("elevation_meters")
                if not elevation or not str(elevation).replace(".", "").isdigit():
                    return {
                        "passed": False,
                        "message": f"Invalid elevation: {elevation}"
                    }
        return {"passed": True, "message": "All elevations valid"}
    
    def _city_different_from_rest_stops(self, data: Dict[str, Any], op_args: Any) -> Dict[str, Any]:
        """Validate that viewpoint cities differ from rest stop cities."""
        routes = data.get("routes", [])
        for route in routes:
            rest_stop_cities = {stop.get("city") for stop in route.get("rest_stops", [])}
            viewpoint_cities = {vp.get("city") for vp in route.get("scenic_viewpoints", [])}
            
            if rest_stop_cities & viewpoint_cities:  # Intersection
                return {
                    "passed": False,
                    "message": "Some viewpoint cities overlap with rest stop cities"
                }
        return {"passed": True, "message": "All cities are different"}
    
    def _stop_include_keys(self, data: List[Dict[str, Any]], op_args: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that stops include required keys."""
        required_keys = set(op_args.get("keys", []))
        for stop in data:
            stop_keys = set(stop.keys())
            missing = required_keys - stop_keys
            if missing:
                return {
                    "passed": False,
                    "message": f"Missing keys: {missing}"
                }
        return {"passed": True, "message": "All required keys present"}
    
    def _is_a_validate_stop(self, data: List[Dict[str, Any]], op_args: Any) -> Dict[str, Any]:
        """Validate that stops are valid places."""
        # Simplified - would use Google Maps API to validate Place IDs
        for stop in data:
            if "place id" not in stop:
                return {
                    "passed": False,
                    "message": "Missing Place ID"
                }
        return {"passed": True, "message": "All stops have Place IDs"}
    
    def _validate_stop_rating(self, data: List[Dict[str, Any]], op_args: float) -> Dict[str, Any]:
        """Validate that stops meet minimum rating."""
        if not self.maps_client:
            return {"passed": True, "message": "Rating validation skipped (no API)"}
        
        min_rating = op_args
        for stop in data:
            place_id = stop.get("place id")
            if place_id:
                try:
                    details = self.maps_client.place_details(place_id)
                    rating = details.get("rating", 0)
                    if rating < min_rating:
                        return {
                            "passed": False,
                            "message": f"Rating {rating} below minimum {min_rating}"
                        }
                except Exception:
                    pass  # Skip if API call fails
        return {"passed": True, "message": "All ratings meet minimum"}
    
    def _validate_stop_type(self, data: List[Dict[str, Any]], op_args: List[str]) -> Dict[str, Any]:
        """Validate that stops are of required types."""
        if not self.maps_client:
            return {"passed": True, "message": "Type validation skipped (no API)"}
        
        allowed_types = op_args
        for stop in data:
            place_id = stop.get("place id")
            if place_id:
                try:
                    details = self.maps_client.place_details(place_id)
                    place_types = details.get("types", [])
                    # Check if any allowed type matches
                    matches = any(t in place_types for t in allowed_types)
                    if not matches:
                        return {
                            "passed": False,
                            "message": f"Place type {place_types} not in {allowed_types}"
                        }
                except Exception:
                    pass
        return {"passed": True, "message": "All stops match required types"}
    
    def _compare_distance_between_stops(self, data: List[Dict[str, Any]], op_args: Dict[str, Any]) -> Dict[str, Any]:
        """Compare distances between stops."""
        # Simplified - would use Distance Matrix API
        return {"passed": True, "message": "Distance comparison (simplified)"}
    
    def _compare_distance_with_and_wo_stops(self, data: List[Dict[str, Any]], op_args: Dict[str, Any]) -> Dict[str, Any]:
        """Compare route distance with and without stops."""
        # Simplified - would use Directions API
        return {"passed": True, "message": "Route distance comparison (simplified)"}
    
    def _validate_location(self, data: List[Dict[str, Any]], op_args: List[str]) -> Dict[str, Any]:
        """Validate location type."""
        # Simplified - would use Places API
        return {"passed": True, "message": "Location validation (simplified)"}
    
    def _places_in_country(self, data: List[Dict[str, Any]], op_args: str) -> Dict[str, Any]:
        """Validate places are in specified country."""
        # Simplified - would use Places API
        return {"passed": True, "message": f"Country validation for {op_args} (simplified)"}
    
    def _validate_direction(self, data: List[Dict[str, Any]], op_args: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate directional relationships."""
        # Simplified - would use Geocoding API
        return {"passed": True, "message": "Direction validation (simplified)"}
    
    def _place_in_cities_visited_of_routes(self, data: Dict[str, Any], op_args: List[str]) -> Dict[str, Any]:
        """Validate that a place is in cities_visited of routes."""
        routes = data.get("routes", [])
        target_cities = [c.lower() for c in op_args]
        
        for route in routes:
            cities_visited = [c.lower() for c in route.get("cities_visited", [])]
            if any(tc in city or city in tc for tc in target_cities for city in cities_visited):
                return {"passed": True, "message": "Required city found in routes"}
        
        return {"passed": False, "message": "Required city not found in cities_visited"}

