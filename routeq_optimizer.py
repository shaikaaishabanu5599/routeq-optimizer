# Reusable RouteQ Optimization Engine
import numpy as np
import time
from typing import Dict, List, Any

def optimize_delivery_scenario(stop_coords: List[List[float]], depot_coord: List[float] = [50.0, 50.0]) -> Dict[str, Any]:
    """
    Standalone optimization function for FastAPI / RouteQ backend.
    """
    coords = np.vstack([depot_coord, stop_coords])
    num_nodes = len(coords)
    dist_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            dist_matrix[i, j] = float(np.linalg.norm(coords[i] - coords[j]))

    # Nearest Neighbor + 2-Opt
    unvisited = list(range(1, num_nodes))
    current = 0
    route = [0]
    while unvisited:
        next_node = min(unvisited, key=lambda n: dist_matrix[current, n])
        route.append(next_node)
        unvisited.remove(next_node)
        current = next_node

    best_route = list(route)
    def calc_dist(r):
        return sum(dist_matrix[r[k], r[k+1]] for k in range(len(r)-1)) + dist_matrix[r[-1], r[0]]

    best_dist = calc_dist(best_route)
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best_route) - 1):
            for j in range(i + 1, len(best_route)):
                new_r = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]
                new_d = calc_dist(new_r)
                if new_d < best_dist:
                    best_route = new_r
                    best_dist = new_d
                    improved = True

    return {
        "status": "success",
        "full_route": best_route + [0],
        "total_distance_km": round(best_dist, 2),
        "stops_count": len(stop_coords)
    }
