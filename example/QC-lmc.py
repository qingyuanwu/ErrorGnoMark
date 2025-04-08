"""
A User Guide for Local QC Measurement & Control System

This script demonstrates how to use the ErrorGnoMark software for quantum chip diagnostics and benchmarking.

If you are using a local quantum computing (QC) measurement and control system, you can access the complete chip information online, including its topology and connectivity.

Notes:
- This example automatically defines a 5x5 chip structure.
- The system selects 9 qubits: [0, 1, 2, 3, 4, 5, 6, 7, 8].
- The chip, qubit index list, and qubit connectivity can be customized by the user.
- To run on an actual chip, users need to register and provide a valid token.
"""

# Import Required Modules
from errorgnomark.errorgnomarker import Errorgnomarker
from errorgnomark.token_manager import define_token, get_token

# Step 1: Define Your Token
# Replace with your actual token
define_token("your token")

# Step 2: Initialize the Errorgnomarker
# Use simulation mode or real hardware mode
egm = Errorgnomarker(chip_name="Baihua", result_get='noisysimulation')  # For simulation mode
# egm = Errorgnomarker(chip_name="Baihua", result_get='hardware')  # For real hardware mode

# Step 3: Run Diagnostics and Benchmarking
results = egm.egm_run(
    rbq1_selected=True,           # Execute Single Qubit RB for Q1
    xebq1_selected=True,          # Execute Single Qubit XEB for Q1
    csbq1_selected=True,          # Execute Single Qubit CSB for Q1
    rbq2_selected=True,           # Execute Two Qubit RB for Q2
    xebq2_selected=True,          # Execute Two Qubit XEB for Q2
    csbq2_selected=True,          # Execute Two Qubit CSB for Q2
    ghzqm_selected=True,          # Execute m-Qubit GHZ Fidelity
    qvqm_selected=True,           # Execute m-Qubit StanQV Fidelity
    mrbqm_selected=True,          # Execute m-Qubit MRB Fidelity
    clopsqm_selected=True,        # Execute m-Qubit Speed CLOPS
    vqeqm_selected=True           # Execute m-Qubit VQE
)

# Step 4: Visualize Results
egm.draw_visual_table()  # Draw the visual table for selected metrics
egm.plot_visual_figure()  # Plot the visual figures for selected metrics

# Results are saved in the `data_egm` folder, along with tables and figures if enabled.
