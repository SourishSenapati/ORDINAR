"""Main entry point for the CA-SPID Mechanistic Inference Solver."""
import os
import numpy as np
import pandas as pd
from workflow import build_research_workflow
from caspid_engine import CASPIDEngine


class SpatialMechanisticSolver:
    """
    World-Class Autonomous Software for Spatial Biology.
    Orchestrates Agentic AI to discover ODEs and simulate perturbations.
    """

    def __init__(self):
        self.research_engine = build_research_workflow()

    def run_pipeline(self, training_csv, output_prediction_csv):
        """Runs the complete discovery and prediction pipeline."""
        print("\n" + "="*80)
        print("  AUTONOMOUS ARCHITECTURE FOR MECHANISTIC INFERENCE (CA-SPID V1.0)")
        print("="*80)

        # 1. DISCOVERY PHASE
        initial_state = {
            "csv_input": training_csv,
            "processed_data": {},
            "candidate_model": {},
            "audit_report": {},
            "alpha_parameters": [0.1479],  # Starting from best known prior
            "iteration_count": 0,
            "discovery_successful": False
        }

        print("[System] Initializing Agentic Swarm...")
        # Stream the agentic process
        final_state = initial_state
        for event in self.research_engine.stream(initial_state):
            # Aggregating events for simplicity in this script
            for key, value in event.items():
                print(f"[System] Step '{key}' finalized.")
                final_state.update(value)

        model = final_state['candidate_model']
        print("\n[System] Discovery Complete. Identifed GRN Topology:")
        print(f"Gene Interactions (A):\n{model['A']}")
        print(f"Morphogen Coupling (B):\n{model['B']}")
        print(f"Inferred Alpha: {model['alpha']:.4f}")

        # 2. PREDICTION PHASE (Experiment B: Removal of Morphogen M1)
        msg = "\n[System] Initiating Counterfactual Simulation (Experiment B: M1 Removal)..."
        print(msg)
        self.perform_experiment_b(model, training_csv, output_prediction_csv)

    def perform_experiment_b(self, model, ref_csv, out_csv):
        """Simulates biological system behavior under morphogen removal."""
        # Load grid from reference
        ref_df = pd.read_csv(ref_csv)
        unique_coords = np.unique(ref_df[['x', 'y']].values, axis=0)
        t_steps = sorted(ref_df['time'].unique())

        a_mat = model['A']
        b_mat = model['B']
        alpha = model['alpha']

        all_results = []

        for ux, uy in unique_coords:
            # Condition B: Morphogen M1 is OFF, M2 is ON.
            _, m2_prof = CASPIDEngine.get_morphogens(ux, uy)
            m_vec = np.array([0, m2_prof])  # Experiment B condition

            # Initial state s_vec=0 (Steady state prior to perturbation)
            s_vec = np.zeros(4)

            # Use CASPID engine logic for trajectory simulation
            dt_step = t_steps[1] - t_steps[0] if len(t_steps) > 1 else 0.08

            for t_time in t_steps:
                all_results.append(
                    [t_time, ux, uy, s_vec[0], s_vec[1], s_vec[2], s_vec[3]])
                # dS/dt = -alpha*S + AS + B*m
                ds_val = (-alpha * s_vec + a_mat @
                          s_vec + b_mat @ m_vec) * dt_step
                s_vec += ds_val

        # Save to CSV
        out_df = pd.DataFrame(all_results, columns=[
                              'time', 'x', 'y', 'S1', 'S2', 'S3', 'S4'])
        # Sort to match standard format
        out_df = out_df.sort_values(
            by=['time', 'y', 'x']).reset_index(drop=True)
        out_df.to_csv(out_csv, index=False)
        print(f"[System] Virtual Experiment Results exported to: {out_csv}")


if __name__ == "__main__":
    # Ensure dependencies are met
    SOLVER = SpatialMechanisticSolver()

    # Check if data exists
    if os.path.exists('GRN_experiment_M2_removal.csv'):
        SOLVER.run_pipeline('GRN_experiment_M2_removal.csv',
                            'predicted_experiment_b_mechanistic.csv')
    else:
        print("[Error] Training data 'GRN_experiment_M2_removal.csv' not found.")
