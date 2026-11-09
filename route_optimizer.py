from __future__ import annotations
import time
from typing import Any


def route_cost(route: list[int], matrix: list[list[float]], return_to_depot: bool = True) -> float:
    if not route:
        return 0.0
    total = sum(float(matrix[a][b]) for a, b in zip(route, route[1:]))
    if return_to_depot and route[-1] != 0:
        total += float(matrix[route[-1]][0])
    return total


def optimize_visit_order(matrix_payload: dict[str, list[list[float]]], return_to_depot: bool = True) -> dict[str, Any]:
    """Nearest-neighbor initialization followed by 2-opt local improvement."""
    matrix = matrix_payload["distance_m"]
    time_matrix = matrix_payload["travel_time_s"]
    n = len(matrix)
    if n < 2 or any(len(row) != n for row in matrix):
        raise ValueError("Distance matrix must be square and include a depot plus at least one stop.")
    if any(value < 0 for row in matrix for value in row):
        raise ValueError("Distance matrix values cannot be negative.")

    started = time.perf_counter()
    unvisited = set(range(1, n))
    route = [0]
    current = 0
    while unvisited:
        next_node = min(unvisited, key=lambda node: matrix[current][node])
        route.append(next_node)
        unvisited.remove(next_node)
        current = next_node

    best = route[:]
    best_cost = route_cost(best, matrix, return_to_depot)
    improved = True
    while improved:
        improved = False
        # Keep depot fixed at position 0; reverse a segment of customer order.
        for i in range(1, len(best) - 1):
            for j in range(i + 1, len(best)):
                candidate = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                cost = route_cost(candidate, matrix, return_to_depot)
                if cost + 1e-9 < best_cost:
                    best, best_cost = candidate, cost
                    improved = True

    full_route = best + ([0] if return_to_depot else [])
    total_time = sum(float(time_matrix[a][b]) for a, b in zip(full_route, full_route[1:]))
    return {
        "route": full_route,
        "total_distance_m": best_cost,
        "total_travel_time_s": total_time,
        "runtime_seconds": round(time.perf_counter() - started, 6),
    }
