import os
import time
import math
import psutil
import numpy as np
from typing import List, Dict, Any, Optional, Union
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from classical_solver import solve_classical_routing, solve_exact_subproblem
from qubo_model import build_routing_qubo, get_qubo_matrix_and_ising, validate_candidate_solution
from qaoa_solver import solve_qubo_with_qaoa
from hybrid_solver import solve_hybrid_decomposition
from validation import validate_full_vrp_solution, validate_route_constraints

app = FastAPI(
    title="RouteQ Hybrid Quantum Logistics Platform",
    description="IBM Quantum Hackathon Problem Statement 8 - Production-Grade VRP Optimization Backend",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Active execution counter for diagnostics
ACTIVE_JOBS = 0

# --- Pydantic Data Models ---
class StopItem(BaseModel):
    id: int
    name: str = "Customer Stop"
    x: float
    y: float
    demand: float = 10.0
    time_window: List[float] = [0.0, 300.0]
    service_time: float = 5.0

class QAOAParams(BaseModel):
    reps: int = Field(default=1, ge=1, le=8)
    shots: int = Field(default=2048, ge=256, le=8192)
    optimizer: str = Field(default="COBYLA")
    backend_provider: str = Field(default="aer_simulator")
    noise_enabled: bool = Field(default=False)
    subproblem_size: int = Field(default=3, ge=3, le=5)
    lagrange_penalty: float = Field(default=100.0, ge=1.0)

class OptimizationRequest(BaseModel):
    stops: Union[List[StopItem], List[List[float]]] = Field(
        ...,
        json_schema_extra={"example": [[20.5, 34.2], [55.1, 80.0], [70.3, 15.8], [30.0, 40.0]]}
    )
    depot: Union[Dict[str, Any], List[float]] = Field(default=[50.0, 50.0])
    solver_type: str = Field(default="hybrid_decomposition")
    fleet_size: int = Field(default=1, ge=1, le=5)
    vehicle_capacity: float = Field(default=100.0, ge=10.0)
    enable_time_windows: bool = Field(default=False)
    enable_capacity: bool = Field(default=True)
    depot_return: bool = Field(default=True)
    qaoa_params: Optional[QAOAParams] = None

class QUBOInspectRequest(BaseModel):
    stops: Union[List[StopItem], List[List[float]]]
    depot: Union[Dict[str, Any], List[float]] = [50.0, 50.0]
    subproblem_size: int = 3
    lagrange_penalty: float = 100.0
    depot_return: bool = True


# --- Helper normalizers ---
def normalize_stops(stops_input: Union[List[StopItem], List[List[float]]]) -> List[Dict[str, Any]]:
    normalized = []
    for idx, item in enumerate(stops_input):
        if isinstance(item, StopItem):
            normalized.append(item.model_dump())
        elif isinstance(item, dict):
            normalized.append(item)
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            normalized.append({
                "id": idx + 1,
                "name": f"Stop {idx + 1}",
                "x": float(item[0]),
                "y": float(item[1]),
                "demand": 10.0,
                "time_window": [0.0, 300.0],
                "service_time": 5.0
            })
    return normalized

def normalize_depot(depot_input: Union[Dict[str, Any], List[float]]) -> Dict[str, Any]:
    if isinstance(depot_input, dict):
        return {
            "name": depot_input.get("name", "Central Depot"),
            "x": float(depot_input.get("x", 50.0)),
            "y": float(depot_input.get("y", 50.0))
        }
    elif isinstance(depot_input, (list, tuple)) and len(depot_input) >= 2:
        return {
            "name": "Central Depot",
            "x": float(depot_input[0]),
            "y": float(depot_input[1])
        }
    return {"name": "Central Depot", "x": 50.0, "y": 50.0}


# --- REST Endpoints ---
@app.get("/health")
def health_check():
    import qiskit
    import qiskit_aer
    return {
        "status": "online",
        "engine": "RouteQ Qiskit Aer Backend Active",
        "version": "2.0.0",
        "qiskit_version": qiskit.__version__,
        "aer_version": qiskit_aer.__version__,
        "timestamp": time.time()
    }

@app.get("/api/system-diagnostics")
def system_diagnostics():
    cpu_pct = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory()
    process = psutil.Process(os.getpid())
    process_mem_mb = process.memory_info().rss / (1024 * 1024)
    
    return {
        "status": "nominal",
        "cpu_percent": round(cpu_pct, 1),
        "system_memory_used_pct": round(mem.percent, 1),
        "system_memory_total_gb": round(mem.total / (1024**3), 2),
        "process_memory_mb": round(process_mem_mb, 1),
        "active_qiskit_jobs": ACTIVE_JOBS,
        "python_version": "3.14.3",
        "platform": "Windows 64-bit",
        "qiskit_runtime": "Qiskit Aer 0.17.2 / Qiskit 2.5.2"
    }

@app.get("/api/presets")
def get_scenario_presets():
    return {
        "micro_grid": {
            "id": "micro_grid",
            "name": "Micro-Grid (4 Stops)",
            "description": "Compact urban cluster optimal for direct QAOA simulation (9 logical qubits).",
            "depot": {"name": "Central Hub", "x": 50.0, "y": 50.0},
            "fleet_size": 1,
            "vehicle_capacity": 60.0,
            "stops": [
                {"id": 1, "name": "Boutique Retail", "x": 35.0, "y": 70.0, "demand": 12.0, "time_window": [15.0, 90.0], "service_time": 5.0},
                {"id": 2, "name": "Pharmacy Center", "x": 65.0, "y": 80.0, "demand": 15.0, "time_window": [30.0, 120.0], "service_time": 6.0},
                {"id": 3, "name": "Grocery Superstore", "x": 75.0, "y": 40.0, "demand": 20.0, "time_window": [45.0, 150.0], "service_time": 8.0},
                {"id": 4, "name": "Local Bakery", "x": 25.0, "y": 30.0, "demand": 8.0, "time_window": [20.0, 110.0], "service_time": 4.0}
            ]
        },
        "urban_core": {
            "id": "urban_core",
            "name": "Urban Core (12 Stops)",
            "description": "Dense metropolitan distribution scenario with tight delivery time windows.",
            "depot": {"name": "Downtown Distribution Hub", "x": 50.0, "y": 50.0},
            "fleet_size": 2,
            "vehicle_capacity": 80.0,
            "stops": [
                {"id": 1, "name": "Financial District", "x": 42.0, "y": 62.0, "demand": 10.0, "time_window": [10.0, 60.0], "service_time": 5.0},
                {"id": 2, "name": "Civic Center", "x": 38.0, "y": 75.0, "demand": 14.0, "time_window": [20.0, 90.0], "service_time": 6.0},
                {"id": 3, "name": "North Market", "x": 55.0, "y": 85.0, "demand": 18.0, "time_window": [35.0, 110.0], "service_time": 7.0},
                {"id": 4, "name": "Hospital Complex", "x": 68.0, "y": 78.0, "demand": 12.0, "time_window": [40.0, 130.0], "service_time": 8.0},
                {"id": 5, "name": "University Campus", "x": 80.0, "y": 60.0, "demand": 16.0, "time_window": [50.0, 150.0], "service_time": 5.0},
                {"id": 6, "name": "East Riverfront", "x": 78.0, "y": 42.0, "demand": 11.0, "time_window": [60.0, 160.0], "service_time": 5.0},
                {"id": 7, "name": "Tech Incubator", "x": 65.0, "y": 25.0, "demand": 9.0, "time_window": [70.0, 180.0], "service_time": 4.0},
                {"id": 8, "name": "South Terminal", "x": 50.0, "y": 20.0, "demand": 15.0, "time_window": [80.0, 200.0], "service_time": 6.0},
                {"id": 9, "name": "Arts District", "x": 35.0, "y": 22.0, "demand": 13.0, "time_window": [90.0, 220.0], "service_time": 5.0},
                {"id": 10, "name": "West Gate Mall", "x": 22.0, "y": 38.0, "demand": 17.0, "time_window": [100.0, 240.0], "service_time": 7.0},
                {"id": 11, "name": "Residential Village", "x": 20.0, "y": 55.0, "demand": 14.0, "time_window": [110.0, 260.0], "service_time": 5.0},
                {"id": 12, "name": "Harbor Warehouse", "x": 30.0, "y": 68.0, "demand": 10.0, "time_window": [120.0, 280.0], "service_time": 6.0}
            ]
        },
        "industrial_cluster": {
            "id": "industrial_cluster",
            "name": "Industrial Cluster (25 Stops)",
            "description": "Large multi-fleet CVRP benchmark solved via Hybrid Quantum-Classical Decomposition.",
            "depot": {"name": "Regional Logistics Center", "x": 50.0, "y": 50.0},
            "fleet_size": 3,
            "vehicle_capacity": 120.0,
            "stops": [
                {"id": 1, "name": "Logistics Hub 1", "x": 45.0, "y": 65.0, "demand": 12.0, "time_window": [15.0, 120.0], "service_time": 5.0},
                {"id": 2, "name": "Freight Depot 2", "x": 40.0, "y": 78.0, "demand": 15.0, "time_window": [25.0, 140.0], "service_time": 5.0},
                {"id": 3, "name": "Assembly Plant 3", "x": 52.0, "y": 88.0, "demand": 18.0, "time_window": [35.0, 160.0], "service_time": 6.0},
                {"id": 4, "name": "Chemical Facility 4", "x": 62.0, "y": 82.0, "demand": 10.0, "time_window": [45.0, 180.0], "service_time": 6.0},
                {"id": 5, "name": "Steel Mill 5", "x": 72.0, "y": 75.0, "demand": 22.0, "time_window": [55.0, 200.0], "service_time": 8.0},
                {"id": 6, "name": "Packaging Plant 6", "x": 82.0, "y": 68.0, "demand": 14.0, "time_window": [65.0, 210.0], "service_time": 5.0},
                {"id": 7, "name": "Cold Storage 7", "x": 88.0, "y": 55.0, "demand": 16.0, "time_window": [75.0, 230.0], "service_time": 7.0},
                {"id": 8, "name": "Automotive Annex 8", "x": 84.0, "y": 42.0, "demand": 19.0, "time_window": [85.0, 250.0], "service_time": 6.0},
                {"id": 9, "name": "Dry Goods Hub 9", "x": 75.0, "y": 32.0, "demand": 11.0, "time_window": [95.0, 260.0], "service_time": 5.0},
                {"id": 10, "name": "Container Depot 10", "x": 65.0, "y": 20.0, "demand": 13.0, "time_window": [105.0, 280.0], "service_time": 6.0},
                {"id": 11, "name": "Rail Terminal 11", "x": 52.0, "y": 15.0, "demand": 20.0, "time_window": [115.0, 300.0], "service_time": 8.0},
                {"id": 12, "name": "Bulk Distribution 12", "x": 42.0, "y": 18.0, "demand": 15.0, "time_window": [125.0, 310.0], "service_time": 6.0},
                {"id": 13, "name": "Textile Mill 13", "x": 30.0, "y": 25.0, "demand": 12.0, "time_window": [135.0, 320.0], "service_time": 5.0},
                {"id": 14, "name": "Electronics Plant 14", "x": 22.0, "y": 35.0, "demand": 17.0, "time_window": [145.0, 340.0], "service_time": 6.0},
                {"id": 15, "name": "Paper Mill 15", "x": 18.0, "y": 48.0, "demand": 14.0, "time_window": [155.0, 350.0], "service_time": 5.0},
                {"id": 16, "name": "Plastics Factory 16", "x": 22.0, "y": 62.0, "demand": 16.0, "time_window": [165.0, 360.0], "service_time": 5.0},
                {"id": 17, "name": "Glassworks 17", "x": 28.0, "y": 72.0, "demand": 11.0, "time_window": [175.0, 380.0], "service_time": 6.0},
                {"id": 18, "name": "Pharma Lab 18", "x": 35.0, "y": 82.0, "demand": 9.0, "time_window": [185.0, 390.0], "service_time": 4.0},
                {"id": 19, "name": "Bio-Diesel Unit 19", "x": 48.0, "y": 75.0, "demand": 13.0, "time_window": [195.0, 400.0], "service_time": 5.0},
                {"id": 20, "name": "Hardware Depot 20", "x": 58.0, "y": 70.0, "demand": 15.0, "time_window": [205.0, 410.0], "service_time": 5.0},
                {"id": 21, "name": "Robotics Yard 21", "x": 68.0, "y": 60.0, "demand": 12.0, "time_window": [215.0, 420.0], "service_time": 6.0},
                {"id": 22, "name": "Solar Array Yard 22", "x": 70.0, "y": 45.0, "demand": 14.0, "time_window": [225.0, 430.0], "service_time": 5.0},
                {"id": 23, "name": "Substation 23", "x": 60.0, "y": 35.0, "demand": 10.0, "time_window": [235.0, 440.0], "service_time": 4.0},
                {"id": 24, "name": "Pump Station 24", "x": 45.0, "y": 30.0, "demand": 8.0, "time_window": [245.0, 450.0], "service_time": 4.0},
                {"id": 25, "name": "Security Center 25", "x": 35.0, "y": 45.0, "demand": 7.0, "time_window": [255.0, 460.0], "service_time": 4.0}
            ]
        }
    }

@app.post("/api/qubo-inspect")
def inspect_qubo(request: QUBOInspectRequest):
    try:
        norm_stops = normalize_stops(request.stops)
        norm_depot = normalize_depot(request.depot)
        
        n_sub = min(request.subproblem_size, len(norm_stops))
        sub_stops = norm_stops[:n_sub]
        coords = [[norm_depot["x"], norm_depot["y"]]] + [[s["x"], s["y"]] for s in sub_stops]
        arr = np.array(coords)
        
        dist_mat = np.zeros((len(arr), len(arr)))
        for i in range(len(arr)):
            for j in range(len(arr)):
                dist_mat[i, j] = float(np.linalg.norm(arr[i] - arr[j]))
                
        qp, meta = build_routing_qubo(dist_mat, penalty_weight=request.lagrange_penalty, depot_return=request.depot_return)
        qubo_details = get_qubo_matrix_and_ising(qp, penalty_weight=request.lagrange_penalty)
        
        return {
            "status": "success",
            "metadata": meta,
            "qubo_details": qubo_details
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/optimize")
def run_optimization(request: OptimizationRequest):
    global ACTIVE_JOBS
    ACTIVE_JOBS += 1
    t_start = time.time()
    try:
        norm_stops = normalize_stops(request.stops)
        norm_depot = normalize_depot(request.depot)
        
        if not norm_stops:
            raise HTTPException(status_code=400, detail="Cannot optimize scenario with 0 stops.")
            
        stop_coords = [[s["x"], s["y"]] for s in norm_stops]
        depot_coord = [norm_depot["x"], norm_depot["y"]]
        demands = [float(s.get("demand", 10.0)) for s in norm_stops]
        time_windows = [s.get("time_window", [0.0, 300.0]) for s in norm_stops]
        
        # 1. Always compute Classical Baseline
        classical_baseline = solve_classical_routing(
            stop_coords=stop_coords,
            depot_coord=depot_coord,
            demands=demands,
            time_windows=time_windows,
            fleet_size=request.fleet_size,
            vehicle_capacity=request.vehicle_capacity,
            enable_capacity=request.enable_capacity,
            enable_time_windows=request.enable_time_windows,
            depot_return=request.depot_return
        )
        
        solver_mode = request.solver_type.lower()
        q_params_dict = request.qaoa_params.model_dump() if request.qaoa_params else {
            "reps": 1,
            "shots": 2048,
            "optimizer": "COBYLA",
            "backend_provider": "aer_simulator",
            "noise_enabled": False,
            "subproblem_size": 3,
            "lagrange_penalty": 100.0
        }
        
        # 2. Dispatch according to solver selection
        if solver_mode == "classical_heuristic":
            active_routes = classical_baseline["routes"]
            solver_desc = "Classical Nearest Neighbor + 2-Opt Heuristic"
            total_dist = classical_baseline["total_distance_km"]
            subproblem_comp = None
            
        elif solver_mode == "classical_exact":
            # For exact, if stops <= 8, solve full exact tour; else exact subproblem
            sub_count = min(8, len(norm_stops))
            sub_indices = list(range(1, sub_count + 1))
            sub_arr = np.array([depot_coord] + [stop_coords[i - 1] for i in sub_indices])
            sub_dist = np.zeros((len(sub_arr), len(sub_arr)))
            for i in range(len(sub_arr)):
                for j in range(len(sub_arr)):
                    sub_dist[i, j] = float(np.linalg.norm(sub_arr[i] - sub_arr[j]))
            exact_res = solve_exact_subproblem(sub_dist, sub_indices)
            
            solver_desc = "Classical Exact (Branch-and-Bound Enumeration)"
            active_routes = [{
                "vehicle_id": 1,
                "route": exact_res["route"],
                "route_coords": [depot_coord if s == 0 else stop_coords[s - 1] for s in exact_res["route"]],
                "distance_km": exact_res["distance_km"],
                "load": round(sum(demands[:sub_count]), 2),
                "num_stops": sub_count
            }]
            total_dist = exact_res["distance_km"]
            subproblem_comp = {
                "sub_stops": sub_indices,
                "exact_optimum": exact_res
            }
            
        elif solver_mode in ["qaoa_simulator", "hybrid_decomposition"]:
            # Hybrid Quantum-Classical Decomposition / QAOA execution
            hybrid_result = solve_hybrid_decomposition(
                stops=norm_stops,
                depot=norm_depot,
                fleet_size=request.fleet_size,
                vehicle_capacity=request.vehicle_capacity,
                enable_capacity=request.enable_capacity,
                enable_time_windows=request.enable_time_windows,
                depot_return=request.depot_return,
                qaoa_params=q_params_dict
            )
            solver_desc = hybrid_result["solver_type"]
            active_routes = hybrid_result["routes"]
            total_dist = hybrid_result["total_distance_km"]
            subproblem_comp = hybrid_result["subproblem_comparison"]
            
        else:
            # Default fallback
            active_routes = classical_baseline["routes"]
            solver_desc = "Classical 2-Opt (Fallback)"
            total_dist = classical_baseline["total_distance_km"]
            subproblem_comp = None

        # 3. Independent Constraint Verification on the final chosen plan
        final_validation = validate_full_vrp_solution(
            routes=active_routes,
            stops=norm_stops,
            depot=norm_depot,
            vehicle_capacity=request.vehicle_capacity,
            enable_time_windows=request.enable_time_windows,
            enable_capacity=request.enable_capacity,
            depot_return=request.depot_return
        )

        t_total = time.time() - t_start

        return {
            "status": "success",
            "solver_engine": solver_desc,
            "runtime_seconds": round(t_total, 4),
            "total_distance_km": round(total_dist, 2),
            "vehicles_used": len([r for r in active_routes if r.get("route")]),
            "routes": active_routes,
            "validation": final_validation,
            "classical_baseline": {
                "method": classical_baseline["method"],
                "total_distance_km": classical_baseline["total_distance_km"],
                "runtime_seconds": classical_baseline["runtime_seconds"],
                "routes": classical_baseline["routes"]
            },
            "subproblem_comparison": subproblem_comp,
            "timestamp": time.time()
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        ACTIVE_JOBS = max(0, ACTIVE_JOBS - 1)


# Static mount for frontend production build if directory exists
dist_dir = os.path.join(os.path.dirname(__file__), "frontend", "dist")
if os.path.exists(dist_dir):
    app.mount("/", StaticFiles(directory=dist_dir, html=True), name="static")
