import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo

def build_routing_qubo(
    sub_dist_matrix: np.ndarray,
    penalty_weight: float = 100.0,
    depot_return: bool = True
) -> Tuple[QuadraticProgram, Dict[str, Any]]:
    """
    Builds the Quadratic Program for the subproblem routing graph.
    Decision variables x_{i, p} = 1 if stop i is visited at step p.
    Nodes: 0 = Depot, 1..n_sub = Customer stops.
    Number of variables (qubits) = n_sub * n_sub.
    """
    n_nodes = sub_dist_matrix.shape[0]
    n_sub = n_nodes - 1
    qp = QuadraticProgram("RouteQ_Subproblem_QUBO")
    
    x_vars = {}
    var_list = []
    for i in range(n_sub):
        for p in range(n_sub):
            var_name = f"x_{i}_{p}"
            x_vars[(i, p)] = var_name
            var_list.append(var_name)
            qp.binary_var(var_name)
            
    linear_obj: Dict[str, float] = {}
    quadratic_obj: Dict[Tuple[str, str], float] = {}
    
    # 1. Travel from depot (node 0) to first stop (p = 0)
    for i in range(n_sub):
        var_name = x_vars[(i, 0)]
        linear_obj[var_name] = linear_obj.get(var_name, 0.0) + float(sub_dist_matrix[0, i + 1])
        
    # 2. Inter-stop travel from step p to step p+1
    for p in range(n_sub - 1):
        for i in range(n_sub):
            for j in range(n_sub):
                if i != j:
                    v1 = x_vars[(i, p)]
                    v2 = x_vars[(j, p + 1)]
                    key = (v1, v2)
                    quadratic_obj[key] = quadratic_obj.get(key, 0.0) + float(sub_dist_matrix[i + 1, j + 1])
                    
    # 3. Travel from last stop (p = n_sub - 1) back to depot (node 0) if depot_return
    if depot_return:
        for i in range(n_sub):
            var_name = x_vars[(i, n_sub - 1)]
            linear_obj[var_name] = linear_obj.get(var_name, 0.0) + float(sub_dist_matrix[i + 1, 0])
        
    # Constraint A: Every position p has exactly 1 stop assigned
    for p in range(n_sub):
        qp.linear_constraint(
            linear={x_vars[(i, p)]: 1.0 for i in range(n_sub)},
            sense="==",
            rhs=1.0,
            name=f"pos_constraint_{p}"
        )
        
    # Constraint B: Every stop i is visited at exactly 1 position
    for i in range(n_sub):
        qp.linear_constraint(
            linear={x_vars[(i, p)]: 1.0 for p in range(n_sub)},
            sense="==",
            rhs=1.0,
            name=f"stop_constraint_{i}"
        )
        
    qp.minimize(linear=linear_obj, quadratic=quadratic_obj)
    
    metadata = {
        "num_variables": qp.get_num_vars(),
        "num_constraints": len(qp.linear_constraints),
        "variable_mapping": {f"{k[0]}_{k[1]}": v for k, v in x_vars.items()},
        "var_names": var_list,
        "n_sub": n_sub,
        "logical_qubits_required": n_sub * n_sub,
        "penalty_weight": penalty_weight
    }
    
    return qp, metadata


def get_qubo_matrix_and_ising(qp: QuadraticProgram, penalty_weight: float = 100.0) -> Dict[str, Any]:
    """
    Extracts the explicit QUBO matrix Q and Pauli-Z Ising Hamiltonian terms.
    """
    conv = QuadraticProgramToQubo(penalty=penalty_weight)
    qubo = conv.convert(qp)
    
    var_names = [v.name for v in qubo.variables]
    n_vars = len(var_names)
    var_idx = {name: i for i, name in enumerate(var_names)}
    
    # Build NxN matrix Q such that Cost = x^T Q x + c
    Q_matrix = np.zeros((n_vars, n_vars), dtype=float)
    
    # Linear objective terms go on diagonal of Q
    for name, coeff in qubo.objective.linear.to_dict().items():
        if name in var_idx:
            Q_matrix[var_idx[name], var_idx[name]] += float(coeff)
            
    # Quadratic terms go in off-diagonal
    for (v1, v2), coeff in qubo.objective.quadratic.to_dict().items():
        if v1 in var_idx and v2 in var_idx:
            i, j = var_idx[v1], var_idx[v2]
            Q_matrix[i, j] += float(coeff)
            
    # Ising Hamiltonian conversion: x_i = (I - Z_i) / 2
    ising_op, ising_offset = qubo.to_ising()
    
    pauli_terms = []
    for pauli_str, coeff in zip(ising_op.paulis.to_labels(), ising_op.coeffs):
        real_c = float(np.real(coeff))
        if abs(real_c) > 1e-6:
            pauli_terms.append({
                "pauli": pauli_str,
                "weight": round(real_c, 4)
            })
            
    return {
        "num_qubits": n_vars,
        "variable_names": var_names,
        "qubo_matrix": [[round(val, 2) for val in row] for row in Q_matrix.tolist()],
        "constant_offset": round(float(qubo.objective.constant), 4),
        "penalty_multiplier": penalty_weight,
        "ising_hamiltonian": {
            "num_terms": len(pauli_terms),
            "offset": round(float(ising_offset), 4),
            "sample_terms": pauli_terms[:20],
            "total_terms": len(pauli_terms)
        }
    }


def validate_candidate_solution(var_dict: Dict[str, float], n_sub: int = 3) -> Tuple[bool, int, List[int]]:
    """
    Validates candidate solution variable dictionary for permutation constraints.
    Returns: (is_valid, violations_count, decoded_route)
    """
    matrix = np.zeros((n_sub, n_sub))
    for i in range(n_sub):
        for p in range(n_sub):
            key = f"x_{i}_{p}"
            matrix[i, p] = round(var_dict.get(key, 0.0))
            
    violations = 0
    # Exactly one stop per position
    for p in range(n_sub):
        if np.sum(matrix[:, p]) != 1:
            violations += 1
            
    # Exactly one position per stop
    for i in range(n_sub):
        if np.sum(matrix[i, :]) != 1:
            violations += 1
            
    is_valid = (violations == 0)
    decoded_route = []
    if is_valid:
        step_assignments = [int(np.argmax(matrix[:, p])) for p in range(n_sub)]
        decoded_route = [0] + [idx + 1 for idx in step_assignments] + [0]
        
    return is_valid, violations, decoded_route


def decode_bitstring(
    bitstring: str,
    n_sub: int,
    var_names: List[str],
    dist_matrix: np.ndarray
) -> Dict[str, Any]:
    """
    Decodes a measurement bitstring (e.g. from Aer measurement shots) into a route.
    """
    # Bitstring in Qiskit is little-endian (qubit 0 is rightmost)
    # Convert into var_dict
    var_dict = {}
    reversed_bits = bitstring[::-1]
    for idx, vname in enumerate(var_names):
        if idx < len(reversed_bits):
            var_dict[vname] = float(reversed_bits[idx])
        else:
            var_dict[vname] = 0.0
            
    is_valid, violations, decoded_route = validate_candidate_solution(var_dict, n_sub=n_sub)
    
    distance = 0.0
    if is_valid and len(decoded_route) > 1:
        for k in range(len(decoded_route) - 1):
            distance += dist_matrix[decoded_route[k], decoded_route[k+1]]
            
    return {
        "bitstring": bitstring,
        "is_valid": is_valid,
        "violations": violations,
        "route": decoded_route if is_valid else [],
        "distance_km": round(distance, 2) if is_valid else None
    }
