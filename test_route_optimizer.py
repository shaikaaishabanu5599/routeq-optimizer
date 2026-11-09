from route_optimizer import optimize_visit_order


def test_route_includes_all_stops_and_returns_to_depot():
    matrix = {
        "distance_m": [[0, 10, 20], [10, 0, 5], [20, 5, 0]],
        "travel_time_s": [[0, 10, 20], [10, 0, 5], [20, 5, 0]],
    }
    result = optimize_visit_order(matrix)
    assert result["route"][0] == 0
    assert result["route"][-1] == 0
    assert sorted(result["route"][1:-1]) == [1, 2]
    assert result["total_distance_m"] == 35
