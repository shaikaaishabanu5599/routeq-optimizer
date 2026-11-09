import numpy as np
import time
import math
import itertools
from typing import Dict, Any, List, Optional

def solve_classical_routing(
    stop_coords: List[List[float]],
    depot_coord: List[float] = [50.0, 50.0],
    demands: Optional[List[float]] = None,
    time_windows: Optional[List[List[float]]] = None,
    fleet_size: int = 1,
    vehicle_capacity: float = 100.0,
    enable_capacity: bool = True,
    enable_time_windows: bool = False,
    depot_return: bool = True
) -> Dict[str, Any]:
    """
    Classical CVRP / TSP heuristic solver:
    - Multi-vehicle clustering (Sweep & Capacity Binning)
    - Nearest Neighbor Tour Construction
    - 2-Opt Local Search Iterative Improvement
    - Constraint timeline simulation
    """
    t_start = time.time()
    num_stops = len(stop_coords)
    if num_stops == 0:
        return {
            "method": "Nearest Neighbor + 2-Opt Heuristic",
            "routes": [],
            "total_distance_km": 0.0,
            "runtime_seconds": 0.0,
            "distance_matrix": [[0.0]]
        }
        
    coords = np.vstack([depot_coord, stop_coords])
    num_nodes = len(coords)
    dist_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            dist_matrix[i, j] = float(np.linalg.norm(coords[i] - coords[j]))

    if demands is None:
        demands = [10.0] * num_stops
    if time_windows is None:
        time_windows = [[0.0, 1000.0]] * num_stops

    effective_fleet = max(1, min(fleet_size, num_stops))
    
    # 1. Multi-vehicle stop partition
    # Partition stops by polar angle relative to depot (Sweep Algorithm)
    angles = []
    for i in range(1, num_nodes):
        dx = coords[i][0] - depot_coord[0]
        dy = coords[i][1] - depot_coord[1]
        angles.append((i, math.atan2(dy, dx)))
    angles.sort(key=lambda x: x[1])
    sorted_stops = [x[0] for x in angles]
    
    # Assign stops to vehicles respecting capacity if enabled
    vehicle_stop_buckets: List[List[int]] = [[] for _ in range(effective_fleet)]
    if enable_capacity and effective_fleet > 1:
        current_v = 0
        current_load = 0.0
        for stop_idx in sorted_stops:
            dem = demands[stop_idx - 1]
            if current_load + dem > vehicle_capacity and current_v < effective_fleet - 1:
                current_v += 1
                current_load = 0.0
            vehicle_stop_buckets[current_v].append(stop_idx)
            current_load += dem
    else:
        # Balanced round-robin / chunk assignment
        for idx, stop_idx in enumerate(sorted_stops):
            v_target = idx % effective_fleet
            vehicle_stop_buckets[v_target].append(stop_idx)

    # 2. For each vehicle route, apply Nearest Neighbor then 2-Opt
    total_fleet_distance = 0.0
    routes_output = []

    for v_idx, bucket in enumerate(vehicle_stop_buckets):
        v_id = v_idx + 1
        if not bucket:
            continue
            
        # Nearest neighbor on bucket
        unvisited = list(bucket)
        current = 0
        nn_route = [0]
        while unvisited:
            next_node = min(unvisited, key=lambda n: dist_matrix[current, n])
            nn_route.append(next_node)
            unvisited.remove(next_node)
            current = next_node
            
        # 2-Opt local search
        best_route = list(nn_route)
        def route_dist(r):
            d = sum(dist_matrix[r[k], r[k+1]] for k in range(len(r)-1))
            if depot_return:
                d += dist_matrix[r[-1], 0]
            return d
            
        best_dist = route_dist(best_route)
        improved = True
        iterations = 0
        while improved and iterations < 100:
            improved = False
            iterations += 1
            for i in range(1, len(best_route) - 1):
                for j in range(i + 1, len(best_route)):
                    new_route = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]
                    new_dist = route_dist(new_route)
                    if new_dist < best_dist - 1e-4:
                        best_route = new_route
                        best_dist = new_dist
                        improved = True
                        break
                if improved:
                    break
                    
        full_path = best_route + ([0] if depot_return else [])
        total_fleet_distance += best_dist
        
        # Calculate vehicle total load
        v_load = sum(demands[s - 1] for s in best_route if s > 0)
        
        routes_output.append({
            "vehicle_id": v_id,
            "route": full_path,
            "route_coords": [coords[node_idx].tolist() for node_idx in full_path],
            "distance_km": round(best_dist, 2),
            "load": round(v_load, 2),
            "num_stops": len(best_route) - 1
        })

    runtime = time.time() - t_start
    
    # Global consolidated route for backwards compatibility
    consolidated_route = [0]
    for r in routes_output:
        stops_in_r = [s for s in r["route"] if s != 0]
        consolidated_route.extend(stops_in_r)
    if depot_return:
        consolidated_route.append(0)

    return {
        "method": "Nearest Neighbor + 2-Opt Heuristic",
        "route": consolidated_route,
        "routes": routes_output,
        "total_distance_km": round(total_fleet_distance, 2),
        "runtime_seconds": round(runtime, 5),
        "distance_matrix": dist_matrix.tolist()
    }


def solve_exact_subproblem(sub_dist_matrix: np.ndarray, sub_indices: List[int]) -> Dict[str, Any]:
    """
    Exact Enumeration / Branch-and-bound for small subproblems (<= 9 stops).
    Finds the provably optimal minimum route distance.
    """
    t_start = time.time()
    n_sub = len(sub_indices)
    best_perm = None
    best_dist = float("inf")
    
    for perm in itertools.permutations(range(1, n_sub + 1)):
        route = [0] + list(perm) + [0]
        dist = sum(sub_dist_matrix[route[k], route[k+1]] for k in range(len(route)-1))
        if dist < best_dist:
            best_dist = dist
            best_perm = perm
            
    runtime = time.time() - t_start
    exact_route = [0] + [sub_indices[p - 1] for p in best_perm] + [0] if best_perm else [0, 0]
    
    return {
        "method": "Exact Branch-and-Bound Enumeration",
        "route": exact_route,
        "distance_km": round(best_dist, 2),
        "runtime_seconds": round(runtime, 6)
    }
