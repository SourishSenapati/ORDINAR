# CA-SPID: Autonomous Mechanistic Inference Platform

A professional software system for the automated discovery of spatially distributed Gene Regulatory Networks (GRNs). This platform implements the Context-Aware Sparse Identification of Parameterized Interaction Dynamics (CA-SPID) framework, specifically engineered for the Predictioneer challenge.

## Key Features

- **Agentic AI Orchestration**: Multi-agent swarm (DataArchitect, Theorist, Critic) managed by a LangGraph state machine.
- **Mechanistic Discovery**: Utilizes SINDy (Sparse Identification of Nonlinear Dynamics) with discrete interaction constraints.
- **Noise-Robust Physics**: Automated noise-profiling and Gaussian-smoothed differentiation kernels for sparse transcriptomics data.
- **Mathematical Falsification**: Built-in stability analysis using Jacobian eigenvalues to ensure biological plausibility.
- **Digital Twinning**: Capability to perform Counterfactual Simulations (e.g., in silico morphogen removal) with compliant CSV exports.

## Architecture

- **predictioneer_solver.py**: Main API and system orchestrator.
- **caspid_engine.py**: Numerical backend for regression and ODE integration.
- **agents.py**: Domain-specific AI agents (DataOps, Theory, Validation).
- **workflow.py**: The LangGraph directed cyclic graph defining the research loop.

## Installation

```bash
pip install numpy pandas scipy langgraph torch
```

## Usage

```python
from predictioneer_solver import SpatialMechanisticSolver

solver = SpatialMechanisticSolver()
solver.run_pipeline('input_data.csv', 'output_predictions.csv')
```

## Authorship

**Team ORDINAR**
Jadavpur University, Kolkata
Lead Architect: Sourish Senapati
Members: Snehali Sadhukhan, Bidhidipta Dutta, Asmita Bagchi
