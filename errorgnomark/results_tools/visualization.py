import json
import numpy as np
import matplotlib.pyplot as plt
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

class VisualPlot:
    """
    A class for visualizing quantum benchmarking metrics stored in a JSON file.
    It generates heatmaps for various benchmarking results, including single-qubit and 
    two-qubit gate quality metrics such as RB, XEB, and CSB.
    """

    def __init__(self, file_path):
        """
        Initialize the VisualPlot class.
        
        Parameters:
            file_path (str): Path to the JSON file containing benchmarking results.
        """
        self.file_path = file_path
        self.data = self._load_data()

    def _load_data(self):
        """
        Load data from the JSON file.

        Returns:
            dict: Parsed JSON data.
        """
        with open(self.file_path, "r") as f:
            return json.load(f)

    def plot_heatmap_with_spacing(self, grid, title, colorbar_label, cmap='viridis', save_path=None):
        """
        Create a heatmap with grid spacing to visualize benchmarking data.
        
        Parameters:
            grid (ndarray): Heatmap data as a 2D NumPy array.
            title (str): Title of the heatmap.
            colorbar_label (str): Label for the colorbar.
            cmap (str): Colormap for the heatmap (default: 'viridis').
            save_path (str, optional): Path to save the figure as a PNG file. If None, the figure is not saved.
        """
        fig, ax = plt.subplots()
        im = ax.imshow(grid, cmap=cmap, interpolation='nearest')
        plt.colorbar(im, ax=ax, label=colorbar_label)

        # Add white grid lines between cells
        for i in range(grid.shape[0] + 1):
            ax.axhline(i - 0.5, color='white', linewidth=1.5)
            ax.axvline(i - 0.5, color='white', linewidth=1.5)

        # Configure axis labels
        ax.set_xticks(np.arange(grid.shape[1]))
        ax.set_yticks(np.arange(grid.shape[0]))
        ax.set_xticklabels(np.arange(grid.shape[1]))
        ax.set_yticklabels(np.arange(grid.shape[0]))
        ax.set_title(title)
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")

        # Save the figure if save_path is provided
        if save_path:
            plt.savefig(save_path, dpi=300)  # Save with high resolution
            print(f"Figure saved as {save_path}")

        # Show the plot
        plt.show()


    def plot_rbq1(self, grid_size=(5, 5)):
        """
        Generate a heatmap for single-qubit Randomized Benchmarking (RB) error rates.
        
        Parameters:
            grid_size (tuple): Dimensions of the grid (default: (5, 5)).
        """
        rows, cols = grid_size
        grid = np.zeros((rows, cols))
        rb_data = self.data["results"]["res_egmq1_rb"]
        
        for qubit_index, values in rb_data.items():
            qubit_index = int(qubit_index.split('_')[-1])
            row = qubit_index // cols
            col = qubit_index % cols
            grid[row, col] = values["error_rate"]
        
        self.plot_heatmap_with_spacing(grid, "Rbq1 Heatmap", "Error Rate", cmap='viridis')

    def plot_xebq1(self, grid_size=(5, 5)):
        """
        Generate a heatmap for single-qubit Cross-Entropy Benchmarking (XEB) results.
        
        Parameters:
            grid_size (tuple): Dimensions of the grid (default: (5, 5)).
        """
        rows, cols = grid_size
        grid = np.zeros((rows, cols))
        xeb_data = self.data["results"]["res_egmq1_xeb"]["hardware"]
        
        for qubit_index, xeb_value in enumerate(xeb_data):
            row = qubit_index // cols
            col = qubit_index % cols
            grid[row, col] = xeb_value
        
        self.plot_heatmap_with_spacing(grid, "Xebq1 Heatmap", "XEB Value", cmap='viridis')

    def plot_csbq1(self, grid_size=(5, 5)):
        """
        Generate heatmaps for single-qubit CSB metrics, including process infidelities, 
        stochastic infidelities, and angle errors.
        
        Parameters:
            grid_size (tuple): Dimensions of the grid (default: (5, 5)).
        """
        rows, cols = grid_size
        csb_data = self.data["results"]["res_egmq1_csbp2x"]
        process_infidelities = csb_data["process_infidelities"]
        stochastic_infidelities = csb_data["stochastic_infidelities"]
        angle_errors = csb_data["angle_errors"]
        
        process_grid = np.zeros((rows, cols))
        stochastic_grid = np.zeros((rows, cols))
        angle_grid = np.zeros((rows, cols))
        
        for i, (p_val, s_val, a_val) in enumerate(zip(process_infidelities, stochastic_infidelities, angle_errors)):
            row = i // cols
            col = i % cols
            process_grid[row, col] = p_val
            stochastic_grid[row, col] = s_val
            angle_grid[row, col] = a_val
        
        self.plot_heatmap_with_spacing(process_grid, "Csbq1 Process Infidelities Heatmap", "Process Infidelity", cmap='viridis')
        self.plot_heatmap_with_spacing(stochastic_grid, "Csbq1 Stochastic Infidelities Heatmap", "Stochastic Infidelity", cmap='viridis')
        self.plot_heatmap_with_spacing(angle_grid, "Csbq1 Angle Errors Heatmap", "Angle Error", cmap='viridis')

    def plot_rbq2(self):
        """
        Generate a heatmap for two-qubit Randomized Benchmarking (RB) error rates.
        """
        qubit_pairs = self.data["qubit_connectivity"]
        rb_data = self.data["results"]["res_egmq2_rb"]
        
        all_qubits = set()
        for pair in qubit_pairs:
            all_qubits.update(pair)
        qubit_count = max(all_qubits) + 1
        
        grid = np.zeros((qubit_count, qubit_count))
        
        for pair, values in rb_data.items():
            if pair == "average_error_rate":
                continue
            qubit_pair = eval(pair)
            grid[qubit_pair[0], qubit_pair[1]] = values["error_rate"]
            grid[qubit_pair[1], qubit_pair[0]] = values["error_rate"]
        
        self.plot_heatmap_with_spacing(grid, "Rbq2 Heatmap", "RB Error Rate", cmap='viridis')

    def plot_xebq2(self):
        """
        Generate a heatmap for two-qubit Cross-Entropy Benchmarking (XEB) error rates.
        """
        qubit_pairs = self.data["qubit_connectivity"]
        xeb_data = self.data["results"]["res_egmq2_xeb"]["hardware"]
        
        all_qubits = set()
        for pair in qubit_pairs:
            all_qubits.update(pair)
        qubit_count = max(all_qubits) + 1
        
        grid = np.zeros((qubit_count, qubit_count))
        
        for index, pair in enumerate(qubit_pairs):
            grid[pair[0], pair[1]] = xeb_data[index]
            grid[pair[1], pair[0]] = xeb_data[index]
        
        self.plot_heatmap_with_spacing(grid, "Xebq2 Heatmap", "XEB Error Rate", cmap='viridis')


    def plot_ghzqm_fidelity(self):
        """
        Plot GHZ state fidelity for different qubit counts (3, 4, 5, 6, 7, 8).
        """
        # Extract fidelity data from the 'res_egmqm_ghz' key
        ghz_fidelity = self.data['results']['res_egmqm_ghz']['fidelity']
        qubit_indices = self.data['results']['res_egmqm_ghz']['qubit_index']
        
        # Quibit counts we're interested in
        qubit_counts = [3, 4, 5, 6, 7, 8]

        # Interpolate fidelity values for each qubit count
        fidelity_values = []
        for count in qubit_counts:
            # Find the closest matching qubit index in the data
            if count - 3 < len(ghz_fidelity):
                fidelity_values.append(ghz_fidelity[count - 3])
            else:
                fidelity_values.append(np.nan)

        # Plotting
        plt.figure(figsize=(8, 6))
        plt.plot(qubit_counts, fidelity_values, marker='o', linestyle='-', color='b', label='GHZ Fidelity')

        # Adding labels and title
        plt.title('GHZ State Fidelity vs Qubit Count', fontsize=16)
        plt.xlabel('Qubit Count', fontsize=14)
        plt.ylabel('Fidelity', fontsize=14)
        plt.grid(True)
        plt.xticks(qubit_counts)
        plt.yticks(np.linspace(0, 1, 11))

        # Show legend
        plt.legend()

        # Display the plot
        plt.tight_layout()
        plt.show()
        
    def plot_mrbqm(self):
        """
        Generate a heatmap for the MRB (polarization) values.
        This heatmap will plot qubit counts (2, 4, 6, 8) on the x-axis,
        lengths (4, 8, 12) on the y-axis, and polarization values as color intensity.
        """
        # Extract polarization and qubits_for_length from the data
        polarizations = self.data["results"]["res_egmqm_mrb"]["polarizations"]
        qubits_for_length = self.data["results"]["res_egmqm_mrb"]["qubits_for_length"]

        # Get the unique lengths available in the data
        unique_lengths = sorted(set(length for qubit_group in qubits_for_length.values()
                                    for length in qubit_group.keys()))

        # Create a matrix to store the polarization values (rows: lengths, columns: qubit counts)
        grid = np.zeros((len(unique_lengths), len(qubits_for_length)))

        # Iterate over each qubit count and the corresponding lengths
        for i, qubit_count in enumerate(sorted(qubits_for_length.keys())):
            # Get the corresponding qubits for the current qubit_count
            for j, length in enumerate(unique_lengths):
                # Get the polarization value for the given qubit_count and length
                polarization_value = polarizations[i][j]
                grid[j, i] = polarization_value  # Store the value in the matrix

        # Plot the heatmap
        self.plot_heatmap_with_spacing(grid, "MRBQM Heatmap (Polarization)", "Polarization", cmap='Blues')
