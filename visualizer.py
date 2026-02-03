"""
Premium Spatial Visualization Suite for CA-SPID.
Provides high-fidelity renders of gene expression patterns and GRN topologies.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


class SpatialVisualizer:
    """
    Visualization engine for Mechanistic Inference results.
    Generates publication-quality spatial heatmaps and temporal dynamics plots.
    """

    def __init__(self, data_path='predicted_experiment_b.csv'):
        self.data_path = data_path
        if os.path.exists(data_path):
            self.df = pd.read_csv(data_path)
            self.genes = ['S1', 'S2', 'S3', 'S4']
            print(f"[Visualizer] Loaded data from {data_path}")
        else:
            self.df = None
            print(f"[Visualizer] Warning: {data_path} not found.")

    def plot_spatial_snapshots(self, time_points=[0, 1.0, 2.0, 3.0]):
        """
        Renders a grid of spatial snapshots for each gene at specific time points.
        """
        if self.df is None:
            return

        num_times = len(time_points)
        fig, axes = plt.subplots(num_times, 4, figsize=(16, 4 * num_times))
        plt.subplots_adjust(hspace=0.3, wspace=0.3)

        for i, t in enumerate(time_points):
            # Select closest time point available
            actual_t = self.df.iloc[(
                self.df['time'] - t).abs().argsort()[:1]]['time'].values[0]
            t_data = self.df[self.df['time'] == actual_t]

            for j, gene in enumerate(self.genes):
                ax = axes[i, j] if num_times > 1 else axes[j]

                # Pivot for heatmap
                pivot_table = t_data.pivot(index='y', columns='x', values=gene)
                sns.heatmap(pivot_table, ax=ax, cmap='viridis', cbar=True)

                if i == 0:
                    ax.set_title(f"Gene {gene}", fontweight='bold')
                if j == 0:
                    ax.set_ylabel(f"Time {actual_t:.1f}", fontweight='bold')

                ax.set_xticks([])
                ax.set_yticks([])

        plt.suptitle(
            "Spatial Gene Expression Dynamics (Experiment B Simulation)", fontsize=16, y=0.95)
        output_name = 'spatial_mechanistic_render.png'
        plt.savefig(output_name, dpi=300, bbox_inches='tight')
        print(f"[Visualizer] Saved spatial snapshots to {output_name}")

    def plot_topologies(self, a_matrix_path='A_matrix.txt'):
        """
        Generates a heatmap of the identified interaction network.
        """
        if not os.path.exists(a_matrix_path):
            return

        a_mat = np.loadtxt(a_matrix_path)
        plt.figure(figsize=(6, 5))
        sns.heatmap(a_mat, annot=True, cmap='RdBu_r', center=0,
                    xticklabels=self.genes, yticklabels=self.genes)
        plt.title("Discovered GRN Interaction Matrix (A)", fontweight='bold')
        plt.xlabel("Target Gene")
        plt.ylabel("Source Gene")

        output_name = 'grn_topology_matrix.png'
        plt.savefig(output_name, dpi=300, bbox_inches='tight')
        print(f"[Visualizer] Saved topology matrix to {output_name}")


if __name__ == "__main__":
    viz = SpatialVisualizer()
    if viz.df is not None:
        viz.plot_spatial_snapshots()
    viz.plot_topologies()
