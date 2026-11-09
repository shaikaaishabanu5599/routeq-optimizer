from __future__ import annotations
import math
import threading
from typing import Any

try:
    import networkx as nx
except ImportError:  # gives a clear error at runtime rather than an import crash
    nx = None

try:
    import osmnx as ox
except ImportError:
    ox = None


class RoadGraphError(RuntimeError):
    """Raised when the OpenStreetMap road graph cannot be loaded or used."""


class RouteNotFoundError(RuntimeError):
    """Raised when no drivable path can be found between two snapped points."""


def haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance in meters between (lat, lon) coordinates."""
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6_371_000 * 2 * math.asin(math.sqrt(min(1.0, h)))


def astar_graph_path(graph, source, target, weight: str = "length"):
    """
    Explicit A* implementation for a NetworkX MultiDiGraph.
    Uses edge length as g-cost and haversine distance to the target as h-cost.
    The heuristic is admissible when edge length is in meters and road edges
    are not assigned costs below their geographic straight-line length.
    """
    if nx is None:
        raise RoadGraphError("NetworkX is not installed. Install dependencies from requirements.txt.")
    if source not in graph or target not in graph:
        raise RouteNotFoundError("The snapped origin or destination is missing from the road graph.")
    if source == target:
        return [source], 0.0

    import heapq
    counter = 0
    open_heap = []
    g_score = {source: 0.0}
    came_from = {}
    closed = set()

    def heuristic(node):
        data = graph.nodes[node]
        target_data = graph.nodes[target]
        return haversine_m((data["y"], data["x"]), (target_data["y"], target_data["x"]))

    heapq.heappush(open_heap, (heuristic(source), 0.0, counter, source))
    while open_heap:
        _, popped_g, _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        # Ignore stale heap entries.
        if popped_g > g_score.get(current, float("inf")):
            continue
        if current == target:
            path = [target]
            while path[-1] != source:
                path.append(came_from[path[-1]])
            path.reverse()
            return path, g_score[target]

        closed.add(current)
        for neighbor, keyed_edges in graph[current].items():
            # A MultiDiGraph can have multiple directed edges between two nodes.
            edge_data = min(
                keyed_edges.values(),
                key=lambda edge: float(edge.get(weight, edge.get("length", 1.0))),
            )
            edge_cost = float(edge_data.get(weight, edge_data.get("length", 1.0)))
            if not math.isfinite(edge_cost) or edge_cost < 0:
                continue
            tentative_g = g_score[current] + edge_cost
            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                closed.discard(neighbor)
                counter += 1
                heapq.heappush(
                    open_heap,
                    (tentative_g + heuristic(neighbor), tentative_g, counter, neighbor),
                )
    raise RouteNotFoundError("No drivable road path was found between these locations.")


class RoadRouter:
    """Loads a directed OSM road graph on demand and caches it for repeated requests."""

    def __init__(self):
        self._graph = None
        self._graph_center = None
        self._radius_m = None
        self._graph_bounds = None
        self._lock = threading.Lock()

    def _load_graph(self, center: tuple[float, float], radius_m: int):
        if ox is None or nx is None:
            raise RoadGraphError(
                "Road routing needs OSMnx and NetworkX. Install the backend requirements, "
                "then restart the server. OSM data is downloaded on the first request."
            )
        with self._lock:
            if self._graph is not None and self._radius_m == radius_m and self._graph_bounds is not None:
                min_lat, max_lat, min_lon, max_lon = self._graph_bounds
                # Reuse the loaded graph for locations inside its downloaded extent.
                if min_lat <= center[0] <= max_lat and min_lon <= center[1] <= max_lon:
                    return self._graph
            try:
                graph = ox.graph_from_point(
                    center_point=center,
                    dist=radius_m,
                    network_type="drive",
                    simplify=True,
                    retain_all=True,
                )
                graph = ox.add_edge_speeds(graph)
                graph = ox.add_edge_travel_times(graph)
            except Exception as exc:
                raise RoadGraphError(
                    "Could not download/load OpenStreetMap roads. Check internet access, "
                    "the coordinates, and the Overpass service, then retry."
                ) from exc
            self._graph = graph
            self._graph_center = center
            self._radius_m = radius_m
            ys = [float(data["y"]) for _, data in graph.nodes(data=True)]
            xs = [float(data["x"]) for _, data in graph.nodes(data=True)]
            self._graph_bounds = (min(ys), max(ys), min(xs), max(xs)) if ys and xs else None
            return graph

    def route(self, origin: tuple[float, float], destination: tuple[float, float], radius_m: int = 12000):
        center = origin
        graph = self._load_graph(center, radius_m)
        try:
            source = ox.distance.nearest_nodes(graph, X=origin[1], Y=origin[0])
            target = ox.distance.nearest_nodes(graph, X=destination[1], Y=destination[0])
            node_path, _ = astar_graph_path(graph, source, target, weight="length")
        except RouteNotFoundError:
            raise
        except Exception as exc:
            raise RoadGraphError("Failed to snap coordinates to road nodes or calculate A* route.") from exc

        distance_m = 0.0
        time_s = 0.0
        geometry: list[list[float]] = []
        for u, v in zip(node_path, node_path[1:]):
            edge = min(
                graph[u][v].values(),
                key=lambda d: float(d.get("length", 1.0)),
            )
            distance_m += float(edge.get("length", 0.0))
            time_s += float(edge.get("travel_time", 0.0))
            if "geometry" in edge:
                segment = [[float(lat), float(lon)] for lon, lat in edge["geometry"].coords]
            else:
                segment = [
                    [float(graph.nodes[u]["y"]), float(graph.nodes[u]["x"])],
                    [float(graph.nodes[v]["y"]), float(graph.nodes[v]["x"])],
                ]
            if geometry and segment and geometry[-1] == segment[0]:
                geometry.extend(segment[1:])
            else:
                geometry.extend(segment)

        if len(node_path) == 1:
            node = graph.nodes[node_path[0]]
            geometry = [[float(node["y"]), float(node["x"])]]

        return {
            "origin": {"lat": origin[0], "lon": origin[1]},
            "destination": {"lat": destination[0], "lon": destination[1]},
            "distance_m": round(distance_m, 2),
            "distance_km": round(distance_m / 1000, 3),
            "travel_time_s": round(time_s, 2),
            "travel_time_min": round(time_s / 60, 2),
            "node_count": len(node_path),
            "geometry": geometry,
        }

    def distance_matrix(self, coordinates: list[tuple[float, float]], radius_m: int = 12000, objective: str = "distance"):
        n = len(coordinates)
        distance_m = [[0.0] * n for _ in range(n)]
        travel_time_s = [[0.0] * n for _ in range(n)]
        pair_routes: dict[str, dict[str, Any]] = {}
        # Use the same cached graph for all pairs in this request.
        graph = self._load_graph(coordinates[0], radius_m)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # Avoid computing duplicate routes; directed streets mean i->j and j->i
                # are distinct and must each be evaluated.
                result = self.route(coordinates[i], coordinates[j], radius_m=radius_m)
                distance_m[i][j] = result["distance_m"]
                travel_time_s[i][j] = result["travel_time_s"]
                pair_routes[f"{i}-{j}"] = result
        return {"distance_m": distance_m, "travel_time_s": travel_time_s}, pair_routes

