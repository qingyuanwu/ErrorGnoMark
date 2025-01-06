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

# Step 1: Import Required Modules
from errorgnomark import Errorgnomarker
from errorgnomark.token_manager import define_token, get_token

# ---------------------------------------------
# Simulation Mode
# ---------------------------------------------
egm = Errorgnomarker(chip_name="QXX", result_get='noisysimulation')

# ---------------------------------------------
# Real Quantum Chip Mode
# ---------------------------------------------
# To execute tasks on real quantum hardware, follow these steps:

# 1. Obtain a Token:
#    Get a unique token from the quantum chip provider (e.g., Quafu). Register and retrieve your token.

# 2. Set and Use the Token:
#    Use the token_manager module to define and retrieve your token.

# Example:
# Define your token
define_token("your_unique_token_here")

# Retrieve the token if needed
token = get_token()

# Initialize the ErrorGnoMarker in hardware mode
egm = Errorgnomarker(chip_name="QXX", result_get='hardware')

# ---------------------------------------------
# Step 2: Run Diagnostics and Benchmarking
# ---------------------------------------------
# Run the EGM metrics with selective tasks
results = egm.egm_run(
    egm_level='level_0',           # Set the level of detail for execution (e.g., level_0, level_1, level_2)
    visual_table=True,             # Generate Level02 table (True/False)
    visual_figure=True,            # Generate Level02 figures (True/False)
    q1rb_selected=True,            # Execute Single Qubit RB (True/False)
    q1xeb_selected=True,           # Execute Single Qubit XEB (True/False)
    q1csbp2x_selected=False,       # Execute Single Qubit CSB (True/False)
    q2rb_selected=True,            # Execute Two Qubit RB (True/False)
    qmgate_ghz_selected=True,      # Execute m-Qubit GHZ Fidelity (True/False)
    qmgate_stqv_selected=True,     # Execute m-Qubit StanQV Fidelity (True/False)
    qmgate_mrb_selected=False,     # Execute m-Qubit MRB Fidelity (True/False)
    qmgate_clops_selected=False,   # Execute m-Qubit Speed CLOPS (True/False)
    qmgate_vqe_selected=False      # Execute m-Qubit VQE (True/False)
)

# Results:
# The results will be saved as a JSON file in the `data_egm` folder.
# Optionally, tables and figures will be saved in their respective folders.

"""
Options:
- egm_level='level_2': Retrieves detailed noise information for the chip.
- visual_table=True: Outputs a comprehensive diagnostic report in table format.
- visual_figure=True: Generates and saves visualized diagnostic results.
"""
