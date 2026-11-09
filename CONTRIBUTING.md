# Contributing to RouteQ

Thank you for your interest in contributing to **RouteQ – Adaptive Last-Mile Optimisation**! This project was developed by team **nextgen** for the **IBM Qiskit Fall Fest 2026** (Problem Statement `VNQFF-08`: *Quantum-Enhanced Last-Mile Route Optimisation*).

We welcome contributions from quantum algorithm developers, operations research researchers, software engineers, and domain experts in logistics.

---

## 📜 Table of Contents
1. [Code of Conduct](#code-of-conduct)
2. [Zero Fabrication Policy](#zero-fabrication-policy)
3. [Development Setup](#development-setup)
4. [Project Structure](#project-structure)
5. [Testing & Verification](#testing--verification)
6. [Contribution Workflow](#contribution-workflow)
7. [Coding Guidelines](#coding-guidelines)
8. [Reporting Issues & Pull Requests](#reporting-issues--pull-requests)

---

## 🤝 Code of Conduct

All contributors are expected to uphold the [Code of Conduct](CODE_OF_CONDUCT.md). Please maintain a respectful, constructive, and inclusive environment.

---

## ⚖️ Zero Fabrication Policy

RouteQ enforces an uncompromising **Zero Fabrication Policy**:
- **No Mock Quantum Metrics**: Every quantum measurement, expectation value, circuit, and OpenQASM artifact must come from real Qiskit compilation or simulation (`qiskit-aer` or IBM Quantum hardware).
- **No Exaggerated Advantage**: We do not claim quantum supremacy or production-scale hardware advantage on NISQ devices. All comparisons against classical heuristics (Nearest Neighbor, 2-Opt, Exact Branch-and-Bound) must be verified with genuine objective values, wall-clock timing, and physical feasibility checks.
- **Physical Verification**: Route lengths, vehicle capacity limits ($q_i \le Q$), and time window constraints must be evaluated using independent Python verification functions (`validation.py`), not self-reported by solvers.

---

## 🛠️ Development Setup

### Prerequisites
- **Python**: Version 3.10+ (tested up to Python 3.14)
- **Node.js**: Version 18.x or 20.x LTS
- **Git**: Version 2.40+

### Step 1: Clone the Repository
```bash
git clone https://github.com/shaikaaishabanu5599/RouteQ-Quantum-Route-Optimization.git
cd RouteQ-Quantum-Route-Optimization
```

### Step 2: Set Up Backend Environment
```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install backend dependencies
pip install -r requirements.txt
```

### Step 3: Set Up Frontend Environment
```bash
cd frontend
npm install
cp .env.example .env
cd ..
```

---

## 📁 Project Structure

```
RouteQ-Quantum-Route-Optimization/
├── .github/
│   └── workflows/
│       └── ci.yml               # Automated CI pipeline (Pytest + Vite Build)
├── docs/
│   ├── ARCHITECTURE.md          # End-to-end system and solver architecture
│   ├── METHODOLOGY.md           # Mathematical formulation (CVRP -> QUBO -> Ising -> QAOA)
│   ├── QFF2026_PS08_nextgen_Final.pptx # Hackathon presentation slide deck
│   └── assets/                  # Diagrams and dashboard screenshots
├── frontend/
│   ├── src/                     # React 19 + TypeScript + Tailwind UI
│   │   ├── pages/               # 7-page enterprise dashboard modules
│   │   ├── components/          # Reusable UI widgets and SVG visualizer
│   │   ├── types/               # Type definitions
│   │   └── api.ts               # Backend client service
│   ├── dist/                    # Precompiled production build (served by FastAPI)
│   ├── package.json
│   └── vite.config.ts
├── classical_solver.py          # Classical OR heuristics (NN, 2-Opt, Exact B&B)
├── hybrid_solver.py             # Classical-Quantum subproblem clustering orchestrator
├── main.py                      # FastAPI REST application & static file server
├── qaoa_solver.py               # Qiskit QAOA ansatz, circuit synthesis & Aer execution
├── qubo_model.py                # CVRP/TSP to QUBO and Ising Hamiltonian mapping
├── validation.py                # Zero-fabrication independent physical verifier
├── test_routeq.py               # Pytest suite with 12 validation tests
├── routeq_optimizer.py          # Standalone lightweight optimizer module
├── routeq_results.json          # Verified benchmark outputs
├── routeq_benchmark_plot.png    # Convergence and comparative charts
├── requirements.txt             # Python dependencies
├── LICENSE                      # Apache License 2.0
├── CITATION.cff                 # Citation metadata
└── README.md                    # Primary repository overview
```

---

## 🧪 Testing & Verification

Before submitting any code, verify that all tests pass and the frontend compiles cleanly:

### Run Backend Pytest Suite
```bash
python -m pytest test_routeq.py -v
```

### Test Frontend Build
```bash
cd frontend
npm run build
cd ..
```

### Run Local Servers for Manual Verification
```bash
# Terminal 1: Run Backend
python -m uvicorn main:app --port 8000 --reload

# Terminal 2: Run Frontend (Dev mode)
cd frontend
npm run dev
```
Navigate to `http://localhost:5173` to test the UI or `http://127.0.0.1:8000/docs` to test Swagger endpoints.

---

## 🔄 Contribution Workflow

1. **Fork the Repository** on GitHub.
2. **Create a Feature Branch**:
   ```bash
   git checkout -b feature/quantum-enhancement-name
   ```
3. **Make Your Changes**: Adhere to code quality and zero-fabrication standards.
4. **Run Tests**: Ensure all automated checks pass locally.
5. **Commit Your Work**:
   ```bash
   git commit -m "feat(qaoa): add dynamic mixer angle scheduling"
   ```
6. **Push to Your Fork**:
   ```bash
   git push origin feature/quantum-enhancement-name
   ```
7. **Open a Pull Request**: Provide a clear description of changes, motivation, test outputs, and quantum circuit characteristics (qubit count, depth, simulator shots).

---

## 💻 Coding Guidelines

- **Python**: Follow PEP 8 style guide. Use type hints (`from typing import List, Dict, Optional, Tuple`).
- **TypeScript**: Use strict mode (`tsconfig.json`). Explicitly type API request/response models.
- **Quantum Code**:
  - Always parameterize circuit ansatz using standard Qiskit 2.x constructs (`QuantumCircuit`, `Parameter`).
  - Document Hamiltonian terms, penalty coefficients ($\lambda_1, \lambda_2$), and number of qubits ($N = n^2$).
  - Never suppress physical infeasibility; record repair attempts transparently.

---

## 🛡️ License

By contributing to RouteQ, you agree that your contributions will be licensed under the **Apache License 2.0**.
