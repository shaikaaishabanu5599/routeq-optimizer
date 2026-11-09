import math
from typing import List, Dict, Any, Tuple, Optional

def validate_route_constraints(route: List[int], expected_stops: List[int]) -> Tuple[bool, List[str]]:
    """
    Backwards-compatible stop check for single subproblem routes.
    """
    violations = []
    if not route or len(route) < 3:
        return False, ["Route is empty or too short."]
    if route[0] != 0 or route[-1] != 0:
        violations.append("Route must start and end at the central depot (node 0).")
    visited_stops = set(route[1:-1])
    expected_set = set(expected_stops)
    if visited_stops != expected_set:
        missing = expected_set - visited_stops
        extra = visited_stops - expected_set
        msg = "Visited stops do not match expected stops."
        if missing:
            msg += f" Missing: {list(missing)}."
        if extra:
            msg += f" Unexpected: {list(extra)}."
        violations.append(msg)
    is_valid = (len(violations) == 0)
    return is_valid, violations


def validate_full_vrp_solution(
    routes: List[Dict[str, Any]],
    stops: List[Dict[str, Any]],
    depot: Dict[str, Any],
    vehicle_capacity: float = 100.0,
    enable_time_windows: bool = False,
    enable_capacity: bool = True,
    depot_return: bool = True
) -> Dict[str, Any]:
    """
    Independent Python constraint verification engine for VRP / CVRP / VRPTW solutions.
    Zero fabrication: Computes exact physical metrics and checks every constraint strictly.
    """
    violations: List[str] = []
    checks = {
        "depot_start": True,
        "depot_return": True,
        "all_stops_visited_once": True,
        "capacity_respected": True,
        "time_windows_respected": True
    }
    
    stop_ids_expected = set(s["id"] for s in stops)
    visited_stops: List[int] = []
    
    # Track stop index lookup
    stop_lookup = {s["id"]: s for s in stops}
    depot_coord = [depot.get("x", 50.0), depot.get("y", 50.0)]
    
    total_distance_km = 0.0
    detailed_timelines: List[Dict[str, Any]] = []

    for r_idx, v_route in enumerate(routes):
        v_id = v_route.get("vehicle_id", r_idx + 1)
        path = v_route.get("route", [])
        
        if not path:
            continue
            
        # 1. Depot Start Check
        if path[0] != 0:
            checks["depot_start"] = False
            violations.append(f"Vehicle {v_id}: Route must start at depot (0). Found start at {path[0]}.")
            
        # 2. Depot Return Check
        if depot_return and path[-1] != 0:
            checks["depot_return"] = False
            violations.append(f"Vehicle {v_id}: Route must end at depot (0). Found end at {path[-1]}.")

        # 3. Step-by-step capacity and time simulation
        current_load = 0.0
        current_time = 0.0
        timeline_steps = []
        
        # Initial depot step
        timeline_steps.append({
            "stop_id": 0,
            "name": "Central Depot",
            "arrival_time": 0.0,
            "wait_time": 0.0,
            "service_time": 0.0,
            "departure_time": 0.0,
            "current_load": 0.0,
            "coords": depot_coord
        })
        
        prev_coord = depot_coord
        for i in range(1, len(path)):
            node_id = path[i]
            
            if node_id == 0:
                # Returning to depot
                curr_coord = depot_coord
                travel_dist = math.hypot(curr_coord[0] - prev_coord[0], curr_coord[1] - prev_coord[1])
                total_distance_km += travel_dist
                current_time += travel_dist # assuming 1 distance unit = 1 time unit (e.g. 60 km/h = 1 km/min)
                timeline_steps.append({
                    "stop_id": 0,
                    "name": "Depot Return",
                    "arrival_time": round(current_time, 2),
                    "wait_time": 0.0,
                    "service_time": 0.0,
                    "departure_time": round(current_time, 2),
                    "current_load": round(current_load, 2),
                    "coords": depot_coord
                })
                prev_coord = curr_coord
                continue
                
            visited_stops.append(node_id)
            node_data = stop_lookup.get(node_id, {})
            curr_coord = [node_data.get("x", 0.0), node_data.get("y", 0.0)]
            demand = float(node_data.get("demand", 0.0))
            travel_dist = math.hypot(curr_coord[0] - prev_coord[0], curr_coord[1] - prev_coord[1])
            total_distance_km += travel_dist
            current_time += travel_dist
            
            # Capacity check
            current_load += demand
            if enable_capacity and current_load > vehicle_capacity + 1e-5:
                checks["capacity_respected"] = False
                violations.append(
                    f"Vehicle {v_id} Capacity Exceeded at Stop {node_id} ('{node_data.get('name', 'Stop')}'): "
                    f"Current load {round(current_load, 1)} exceeds vehicle capacity {vehicle_capacity}."
                )
                
            # Time Window check
            wait_time = 0.0
            tw = node_data.get("time_window", [0.0, 9999.0])
            early, late = float(tw[0]), float(tw[1])
            if enable_time_windows:
                if current_time < early:
                    wait_time = early - current_time
                    current_time = early
                elif current_time > late:
                    checks["time_windows_respected"] = False
                    violations.append(
                        f"Vehicle {v_id} Time Window Breach at Stop {node_id} ('{node_data.get('name', 'Stop')}'): "
                        f"Arrived at {round(current_time, 1)} min, window closes at {late} min (Tardiness: {round(current_time - late, 1)} min)."
                    )
            
            service_time = float(node_data.get("service_time", 5.0))
            departure_time = current_time + service_time
            current_time = departure_time
            
            timeline_steps.append({
                "stop_id": node_id,
                "name": node_data.get("name", f"Stop {node_id}"),
                "arrival_time": round(current_time - service_time, 2),
                "wait_time": round(wait_time, 2),
                "service_time": service_time,
                "departure_time": round(departure_time, 2),
                "current_load": round(current_load, 2),
                "coords": curr_coord
            })
            prev_coord = curr_coord

        detailed_timelines.append({
            "vehicle_id": v_id,
            "final_load": round(current_load, 2),
            "capacity": vehicle_capacity,
            "load_utilization_pct": round((current_load / max(1.0, vehicle_capacity)) * 100, 1),
            "timeline": timeline_steps
        })

    # 4. Visit count check (every stop visited exactly once)
    visited_set = set(visited_stops)
    if len(visited_stops) != len(visited_set):
        checks["all_stops_visited_once"] = False
        duplicates = [x for x in visited_set if visited_stops.count(x) > 1]
        violations.append(f"Duplicate stop visits detected for stops: {duplicates}.")
        
    if visited_set != stop_ids_expected:
        checks["all_stops_visited_once"] = False
        missing = stop_ids_expected - visited_set
        extra = visited_set - stop_ids_expected
        if missing:
            violations.append(f"Unserviced stops: Customer stops {list(missing)} were not visited by any vehicle.")
        if extra:
            violations.append(f"Extraneous stops: Nodes {list(extra)} visited but not in customer manifest.")
            
    is_feasible = (len(violations) == 0)
    
    return {
        "is_feasible": is_feasible,
        "violations_count": len(violations),
        "violations": violations,
        "checks": checks,
        "total_distance_km": round(total_distance_km, 2),
        "vehicle_timelines": detailed_timelines
    }
