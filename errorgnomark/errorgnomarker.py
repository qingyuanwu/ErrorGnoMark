# Standard library imports
import json  # For handling JSON data
import os  # For file and directory management
import sys  # For system-specific parameters and functions
import time  # For timing operations
from datetime import datetime  # For handling date and time

# Third-party library imports
from requests.exceptions import RequestException, ReadTimeout  # For HTTP requests and error handling
from tqdm import tqdm  # For progress bar visualization

# Add the ErrorGnoMark package to the system path
sys.path.append('/Users/ousiachai/Desktop/ErrorGnoMark')
# Local imports
from errorgnomark.cirpulse_generator.qubit_selector import qubit_selection, chip  # For qubit selection and chip setup
from errorgnomark.configuration import (  # For various quality and benchmarking configurations
    QualityQ1Gate,
    QualityQ2Gate,
    QualityQmgate,
    SpeedQmgate,
    ApplicationQmgate
)
from errorgnomark.results_tools.egm_report_tools import EGMReportManager  # For generating reports





class Errorgnomarker(chip):
    """
    Retrieves error and performance metrics for qubits and quantum gates at various levels.
    Supports single-qubit, two-qubit, multi-qubit gates, and application-level tests.
    """

    def __init__(self, chip_name="Baihua", result_get='noisysimulation'):
        """
        Initializes the ErrorGnoMarker with the specified chip configuration.

        Parameters:
            chip_name (str): Name of the chip configuration.
        """
        super().__init__()  # Initialize the base chip class

        self.chip_name = chip_name

        if self.chip_name == "Baihua":
            # Initialize the chip with specific rows and columns
            self.rows = 13
            self.columns = 13
        else:
            # Handle other chip configurations if necessary
            raise ValueError(f"Unsupported chip name: {self.chip_name}")

        self.result_get = result_get
        # Step 1: Define selection options
        self.selection_options = {
            'max_qubits_per_row': 13,
            'min_qubit_index': 0,
            'max_qubit_index': 100
        }

        # Step 2: Initialize qubit selection with constraints
        self.selector = qubit_selection(
            chip=self,
            qubit_index_max=100,
            qubit_number=25,
            option=self.selection_options
        )

        # Perform qubit selection
        self.selection = self.selector.quselected()
        self.qubit_index_list = self.selection["qubit_index_list"]
        self.qubit_connectivity = self.selection["qubit_connectivity"]

        print("=" * 50)
        print("Selected Qubit Indices:", self.qubit_index_list)
        print("Qubit Connectivity:", self.qubit_connectivity)
        print("=" * 50)

        # Initialize configuration objects with the selected qubits and connectivity

        self.config_quality_q1gate = QualityQ1Gate(self.qubit_index_list, result_get=result_get)
        self.config_quality_q2gate = QualityQ2Gate(self.qubit_connectivity, result_get=result_get)
        self.config_quality_qmgate = QualityQmgate(self.qubit_connectivity, self.qubit_index_list, result_get=result_get)
        self.config_speed_qmgate = SpeedQmgate(self.qubit_connectivity, self.qubit_index_list, result_get=result_get)
        self.config_application_qmgate = ApplicationQmgate(self.qubit_connectivity, self.qubit_index_list, result_get=result_get)

    def egm_run(self, egm_level='level_0', visual_table=None, visual_figure=None,
                q1rb_selected=True, q1xeb_selected=True, q1csbp2x_selected=True,
                q2rb_selected=True, qmgate_ghz_selected=True, qmgate_stqv_selected=True,
                qmgate_mrb_selected=True, qmgate_clops_selected=True, qmgate_vqe_selected=True):
        """
        Executes the EGM metrics and saves the results to a JSON file.

        Parameters:
            egm_level (str): Level of detail ('level_0', 'level_1', 'level_2').
            visual_table (bool): If True, generate the level02 table.
            visual_figure (bool): If True, generate the level02 figures.
            q1rb_selected (bool): Whether to execute the Single Qubit RB method.
            q1xeb_selected (bool): Whether to execute the Single Qubit XEB method.
            q1csbp2x_selected (bool): Whether to execute the Single Qubit CSB method.
            q2rb_selected (bool): Whether to execute the Two Qubit RB method.
            qmgate_ghz_selected (bool): Whether to execute the m-Qubit GHZ Fidelity method.
            qmgate_stqv_selected (bool): Whether to execute the m-Qubit StanQV Fidelity method.
            qmgate_mrb_selected (bool): Whether to execute the m-Qubit MRB Fidelity method.
            qmgate_clops_selected (bool): Whether to execute the m-Qubit Speed CLOPS method.
            qmgate_vqe_selected (bool): Whether to execute the m-Qubit VQE method.
        """
        results = {}
        total_tasks = 10  # Total tasks (metrics to be executed)
        progress_bar = tqdm(total=total_tasks, desc="Overall Progress", position=0, leave=True)

        # Start the total execution timer
        total_start_time = time.time()

        try:
            # Single Qubit RB
            if q1rb_selected:
                print("[Running] Single Qubit RB...")
                start_time = time.time()
                res_egmq1_rb = self.config_quality_q1gate.q1rb()
                elapsed_time = time.time() - start_time
                print(f"Single Qubit RB completed in {elapsed_time:.2f} seconds.")
                if res_egmq1_rb is None:
                    raise ValueError("q1rb returned None")
                results['res_egmq1_rb'] = res_egmq1_rb
                progress_bar.update(1)

            # Single Qubit XEB
            if q1xeb_selected:
                print("[Running] Single Qubit XEB...")
                start_time = time.time()
                res_egmq1_xeb = self.config_quality_q1gate.q1xeb()
                elapsed_time = time.time() - start_time
                print(f"Single Qubit XEB completed in {elapsed_time:.2f} seconds.")
                if res_egmq1_xeb is None:
                    raise ValueError("q1xeb returned None")
                results['res_egmq1_xeb'] = res_egmq1_xeb
                progress_bar.update(1)

            # Single Qubit CSB
            if q1csbp2x_selected:
                print("[Running] Single Qubit CSB...")
                start_time = time.time()
                res_egmq1_csbp2x = self.config_quality_q1gate.q1csb_pi_over_2_x()
                elapsed_time = time.time() - start_time
                print(f"Single Qubit CSB completed in {elapsed_time:.2f} seconds.")
                if res_egmq1_csbp2x is None:
                    raise ValueError("q1csb_pi_over_2_x returned None")
                results['res_egmq1_csbp2x'] = res_egmq1_csbp2x
                progress_bar.update(1)

            # Two Qubit RB
            if q2rb_selected:
                print("[Running] Two Qubit RB...")
                start_time = time.time()
                res_egmq2_rb = self.config_quality_q2gate.q2rb()
                elapsed_time = time.time() - start_time
                print(f"Two Qubit RB completed in {elapsed_time:.2f} seconds.")
                if res_egmq2_rb is None:
                    raise ValueError("q2rb returned None")
                results['res_egmq2_rb'] = res_egmq2_rb
                progress_bar.update(1)

            # m-Qubit GHZ Fidelity
            if qmgate_ghz_selected:
                print("[Running] m-Qubit GHZ Fidelity...")
                start_time = time.time()
                res_egmqm_ghz = self.config_quality_qmgate.qmghz_fidelity()
                elapsed_time = time.time() - start_time
                print(f"m-Qubit GHZ Fidelity completed in {elapsed_time:.2f} seconds.")
                if res_egmqm_ghz is None:
                    raise ValueError("qmghz_fidelity returned None")
                results['res_egmqm_ghz'] = res_egmqm_ghz
                progress_bar.update(1)

            # m-Qubit StanQV Fidelity
            if qmgate_stqv_selected:
                print("[Running] m-Qubit StanQV Fidelity...")
                start_time = time.time()
                res_egmqm_stqv = self.config_quality_qmgate.qmstanqv()
                elapsed_time = time.time() - start_time
                print(f"m-Qubit StanQV Fidelity completed in {elapsed_time:.2f} seconds.")
                if res_egmqm_stqv is None:
                    raise ValueError("qmstanqv returned None")
                results['res_egmqm_stqv'] = res_egmqm_stqv
                progress_bar.update(1)

            # m-Qubit MRB Fidelity
            if qmgate_mrb_selected:
                print("[Running] m-Qubit MRB Fidelity...")
                start_time = time.time()
                res_egmqm_mrb = self.config_quality_qmgate.qmmrb()
                elapsed_time = time.time() - start_time
                print(f"m-Qubit MRB Fidelity completed in {elapsed_time:.2f} seconds.")
                if res_egmqm_mrb is None:
                    raise ValueError("qmmrb returned None")
                results['res_egmqm_mrb'] = res_egmqm_mrb
                progress_bar.update(1)

            # m-Qubit Speed CLOPS
            if qmgate_clops_selected:
                print("[Running] m-Qubit Speed CLOPS...")
                start_time = time.time()
                res_egmqm_clops = self.config_speed_qmgate.qmclops()
                elapsed_time = time.time() - start_time
                print(f"m-Qubit Speed CLOPS completed in {elapsed_time:.2f} seconds.")
                if res_egmqm_clops is None:
                    raise ValueError("qmclops returned None")
                results['res_egmqm_clops'] = res_egmqm_clops
                progress_bar.update(1)

            # m-Qubit VQE
            if qmgate_vqe_selected:
                print("[Running] m-Qubit VQE...")
                start_time = time.time()
                res_egmqm_vqe = self.config_application_qmgate.qmVQE()
                elapsed_time = time.time() - start_time
                print(f"m-Qubit VQE completed in {elapsed_time:.2f} seconds.")
                if res_egmqm_vqe is None:
                    raise ValueError("qmVQE returned None")
                results['res_egmqm_vqe'] = res_egmqm_vqe
                progress_bar.update(1)

        except Exception as e:
            print(f"An error occurred during execution: {e}")
        finally:
            progress_bar.close()

        # Compute and print the total execution time
        total_elapsed_time = time.time() - total_start_time
        print(f"Total execution time: {total_elapsed_time:.2f} seconds.")

        # Create 'data_egm' folder if it does not exist
        if not os.path.exists('data_egm'):
            os.makedirs('data_egm')

        # Prepare the filename with chip name and current datetime
        current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.chip_name}_egm_data_{current_datetime}.json"
        filepath = os.path.join('data_egm', filename)  # Save file in 'data_egm' folder

        # Prepare the full results dictionary with title and initial information
        full_results = {
            "title": filename,
            "chip_info": {
                "chip_name": self.chip_name,
                "rows": self.rows,
                "columns": self.columns
            },
            "qubit_index_list": self.qubit_index_list,
            "qubit_connectivity": self.qubit_connectivity,
            "results": results
        }

        # Save the full results dictionary to a JSON file
        try:
            with open(filepath, 'w') as json_file:
                json.dump(full_results, json_file, indent=4)
            print(f"EGM results saved to {filepath}")
        except Exception as e:
            print(f"Failed to save EGM results to JSON file: {e}")

        # Generate the Level02 table or figures if specified
        if visual_table:
            print("[Generating] Level02 Table...")
            report_manager = EGMReportManager(filepath)
            report_manager.egm_level02_table()

        if visual_figure:
            print("[Generating] Level02 Figures...")
            report_manager = EGMReportManager(filepath)
            report_manager.egm_level02_figure()

        # Optionally, you can return the full results dictionary
        return full_results



from errorgnomark.token_manager import define_token, get_token
# Define your token
token = "5vtENo5IEGViJNv:nmgYuZ:ehMobWzUd6qcu7pMeSZW/Rg{dUPyBkO{5DO{BEP4VkO{dUN7JDd5WnJtJDOyp{O1pEOyBjNy1jNy1DOzBkNjpkJ1GXbjxjJvOnMkGnM{mXdiKHRliYbii3ZjpkJzW3d2Kzf"

define_token(token)

# Retrieve the token if needed
token = get_token()


# Example usage:
if __name__ == "__main__":
    egm = Errorgnomarker(chip_name="Baihua",result_get='hardware')
    egm.egm_run(
    visual_table=True, 
    visual_figure=True, 
    q1rb_selected =True, 
    q1xeb_selected=False,          
    q1csbp2x_selected=False,      
    q2rb_selected=False,           
    qmgate_ghz_selected=False,     
    qmgate_stqv_selected=False,    
    qmgate_mrb_selected=False,    
    qmgate_clops_selected=False,  
    qmgate_vqe_selected=False )    
