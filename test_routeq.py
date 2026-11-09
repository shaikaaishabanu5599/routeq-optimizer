import pytest
import numpy as np
from fastapi.testclient import TestClient

from main import app
from validation import validate_route_constraints, validate_full_vrp_solution
from classical_solver import solve_classical_routing, solve_exact_subproblem
from qubo_model import build_routing_qubo, get_qubo_matrix_and_ising, validate_candidate_solution
from qaoa_solver import solve_qubo_with_qaoa
from hybrid_solver import solve_hybrid_decomposition

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "qiskit_version" in data
    assert "aer_version" in data

def test_system_diagnostics():
    response = client.get("/api/system-diagnostics")
    assert response.status_code == 200
    data = response.json()
    assert "cpu_percent" in data
    assert "process_memory_mb" in data
    assert data["status"] == "nominal"

def test_presets_endpoint():
    response = client.get("/api/presets")
    assert response.status_code == 200
    presets = response.json()
    assert "micro_grid" in presets
    assert "urban_core" in presets
    assert "industrial_cluster" in presets
    assert len(presets["micro_grid"]["stops"]) == 4

def test_validation_engine_clean_route():
    stops = [
        {"id": 1, "name": "Stop 1", "x": 10.0, "y": 10.0, "demand": 10.0, "time_window": [0.0, 100.0], "service_time": 5.0},
        {"id": 2, "name": "Stop 2", "x": 20.0, "y": 20.0, "demand": 15.0, "time_window": [0.0, 100.0], "service_time": 5.0}
    ]
    depot = {"name": "Depot", "x": 0.0, "y": 0.0}
    routes = [{
        "vehicle_id": 1,
        "route": [0, 1, 2, 0]
    }]
    
    val = validate_full_vrp_solution(routes, stops, depot, vehicle_capacity=50.0)
    assert val["is_feasible"] is True
    assert val["violations_count"] == 0

def test_validation_engine_detects_violations():
    stops = [
        {"id": 1, "name": "Stop 1", "x": 10.0, "y": 10.0, "demand": 30.0, "time_window": [0.0, 10.0], "service_time": 5.0},
        {"id": 2, "name": "Stop 2", "x": 20.0, "y": 20.0, "demand": 30.0, "time_window": [0.0, 10.0], "service_time": 5.0}
    ]
    depot = {"name": "Depot", "x": 0.0, "y": 0.0}
    
    # Capacity violation: 60.0 > 40.0
    routes = [{
        "vehicle_id": 1,
        "route": [0, 1, 2, 0]
    }]
    val = validate_full_vrp_solution(routes, stops, depot, vehicle_capacity=40.0, enable_capacity=True)
    assert val["is_feasible"] is False
    assert any("Capacity Exceeded" in v for v in val["violations"])

def test_qubo_model_and_ising_extraction():
    dist_mat = np.array([
        [0.0, 10.0, 20.0, 30.0],
        [10.0, 0.0, 15.0, 25.0],
        [20.0, 15.0, 0.0, 12.0],
        [30.0, 25.0, 12.0, 0.0]
    ])
    qp, meta = build_routing_qubo(dist_mat)
    assert meta["logical_qubits_required"] == 9  # 3 stops * 3 positions
    assert qp.get_num_vars() == 9
    
    qubo_info = get_qubo_matrix_and_ising(qp)
    assert qubo_info["num_qubits"] == 9
    assert len(qubo_info["qubo_matrix"]) == 9
    assert "ising_hamiltonian" in qubo_info
    assert qubo_info["ising_hamiltonian"]["total_terms"] > 0

def test_classical_solver_and_exact():
    stops = [[10.0, 20.0], [30.0, 40.0], [50.0, 10.0]]
    depot = [0.0, 0.0]
    res = solve_classical_routing(stops, depot)
    assert res["total_distance_km"] > 0.0
    assert len(res["route"]) == 5  # [0, s1, s2, s3, 0]
    
    dist_mat = np.array(res["distance_matrix"])
    exact_res = solve_exact_subproblem(dist_mat, [1, 2, 3])
    assert exact_res["distance_km"] > 0.0
    assert len(exact_res["route"]) == 5

def test_qaoa_aer_simulation_and_qasm():
    # 2-stop test for high-speed unit test execution (4 qubits)
    dist_mat = np.array([
        [0.0, 10.0, 15.0],
        [10.0, 0.0, 8.0],
        [15.0, 8.0, 0.0]
    ])
    qp, _ = build_routing_qubo(dist_mat)
    res = solve_qubo_with_qaoa(qp, reps=1, shots=1024, optimizer_name="COBYLA", backend_provider="aer_simulator", maxiter=5)
    assert res["status"] == "success"
    assert res["num_qubits"] == 4
    assert len(res["bitstring_histogram"]) > 0
    assert "OPENQASM" in res["openqasm"]

def test_api_optimize_endpoint():
    payload = {
        "stops": [
            {"id": 1, "name": "Stop 1", "x": 35.0, "y": 70.0, "demand": 10.0},
            {"id": 2, "name": "Stop 2", "x": 65.0, "y": 80.0, "demand": 15.0},
            {"id": 3, "name": "Stop 3", "x": 75.0, "y": 40.0, "demand": 20.0}
        ],
        "depot": {"name": "Depot", "x": 50.0, "y": 50.0},
        "solver_type": "classical_heuristic",
        "fleet_size": 1,
        "vehicle_capacity": 100.0
    }
    response = client.post("/api/optimize", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["total_distance_km"] > 0.0
    assert data["validation"]["is_feasible"] is True

def test_qubo_inspect_endpoint():
    payload = {
        "stops": [
            {"id": 1, "name": "Stop 1", "x": 10.0, "y": 20.0},
            {"id": 2, "name": "Stop 2", "x": 30.0, "y": 40.0},
            {"id": 3, "name": "Stop 3", "x": 50.0, "y": 60.0}
        ],
        "depot": {"name": "Depot", "x": 0.0, "y": 0.0},
        "subproblem_size": 3,
        "lagrange_penalty": 100.0
    }
    response = client.post("/api/qubo-inspect", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["qubo_details"]["num_qubits"] == 9
    assert len(data["qubo_details"]["qubo_matrix"]) == 9

def test_invalid_inputs_rejected():
    payload = {
        "stops": [],
        "depot": {"name": "Depot", "x": 0.0, "y": 0.0}
    }
    response = client.post("/api/optimize", json=payload)
    assert response.status_code == 400

def test_hybrid_decomposition_solver():
    stops = [
        {"id": 1, "name": "Stop 1", "x": 10.0, "y": 20.0, "demand": 10.0},
        {"id": 2, "name": "Stop 2", "x": 15.0, "y": 25.0, "demand": 10.0},
        {"id": 3, "name": "Stop 3", "x": 20.0, "y": 30.0, "demand": 10.0},
        {"id": 4, "name": "Stop 4", "x": 25.0, "y": 35.0, "demand": 10.0}
    ]
    depot = {"name": "Depot", "x": 0.0, "y": 0.0}
    res = solve_hybrid_decomposition(
        stops=stops,
        depot=depot,
        fleet_size=1,
        vehicle_capacity=100.0,
        qaoa_params={"reps": 1, "shots": 512, "optimizer": "COBYLA", "subproblem_size": 3, "maxiter": 5}
    )
    assert res["status"] == "success"
    assert res["total_distance_km"] > 0.0
    assert len(res["routes"]) > 0

