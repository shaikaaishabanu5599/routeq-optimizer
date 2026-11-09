import networkx as nx
import pytest
from astar_router import astar_graph_path, RouteNotFoundError, haversine_m


def graph_fixture():
    g = nx.MultiDiGraph()
    # Coordinates: lat (y), lon (x). A longer direct edge exists; A* should pick via node 2.
    g.add_node(1, y=0.0, x=0.0)
    g.add_node(2, y=0.0, x=0.001)
    g.add_node(3, y=0.001, x=0.001)
    g.add_edge(1, 2, length=100.0)
    g.add_edge(2, 3, length=100.0)
    g.add_edge(1, 3, length=500.0)
    return g


def test_astar_finds_lowest_cost_path():
    path, cost = astar_graph_path(graph_fixture(), 1, 3)
    assert path == [1, 2, 3]
    assert cost == pytest.approx(200.0)


def test_astar_respects_one_way_edges():
    g = graph_fixture()
    with pytest.raises(RouteNotFoundError):
        astar_graph_path(g, 3, 1)


def test_same_node_route():
    path, cost = astar_graph_path(graph_fixture(), 1, 1)
    assert path == [1]
    assert cost == 0


def test_haversine_is_positive():
    assert haversine_m((0, 0), (0, 1)) > 100000
