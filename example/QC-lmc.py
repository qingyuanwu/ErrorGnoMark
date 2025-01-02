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

from errorgnomark import Errorgnomarker
from errorgnomark import token_manager

# Step 1: Initialize the ErrorGnoMarker object
# ---------------------------------------------

# Default Simulation Mode
# By default, the Errorgnomarker class runs in simulation mode.
egm = Errorgnomarker(chip_name="QXX")

# Real Quantum Chip Mode
# To execute tasks on real quantum hardware, follow these steps:

# 1. Obtain a Token:
#    Get a unique token from the quantum chip provider (e.g., Quafu). Register and retrieve your token.

# 2. Set and Use the Token:
#    Use the token_manager module to define and retrieve your token.
define_token("your_unique_token_here")  # Replace with your actual token
token = get_token()  # Retrieve the token if needed

# 3. Initialize the ErrorGnoMarker object in hardware mode:
egm = Errorgnomarker(chip_name="QXX", result_get='hardware')

# Step 2: Run Diagnostics and Benchmarking
# ----------------------------------------

# This example demonstrates how to run diagnostics and benchmarking for the chip.

# Options:
# - egm_level='level_2': Retrieves detailed noise information for the chip.
# - visual_table=True: Outputs a comprehensive diagnostic report in table format.
# - visual_figure=True: Generates and saves visualized diagnostic results.

# Users can customize the benchmarking schemes as needed. The example below uses default settings.
egm.egm_run(egm_level='level_2', visual_table=True, visual_figure=True)
