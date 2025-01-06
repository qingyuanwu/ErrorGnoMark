# Standard library imports
import os  # For operating system-related functionalities
import json  # For working with JSON data
from datetime import datetime  # For handling date and time
import sys

# Third-party imports
import matplotlib.pyplot as plt  # For creating visualizations

# ErrorGnoMark-specific imports
sys.path.append('/Users/ousiachai/Desktop/ErrorGnoMark') 

# ErrorGnoMark-specific imports
from errorgnomark.results_tools.visualization import VisualPlot  # For generating visual plots

class EGMReportManager:
    """
    This class handles the reading of EGM JSON data files and generates 
    text-based tables for various benchmarking metrics. It also saves the 
    tables as text files in designated directories.
    """

    def __init__(self, json_file_path):
        """
        Initialize the EGMReportManager class and load JSON data.

        Args:
            json_file_path (str): Path to the JSON data file.
        """
        self.json_file_path = json_file_path
        self.data = self._read_json()
        self.results = self.data.get("results", {})
        self.qubit_index_list = self.data.get("qubit_index_list", [])
        self.qubit_connectivity = self.data.get("qubit_connectivity", [])
        self.output_dir = "tables"  # Directory for saving tables
        self.figures_dir = "figures"  # Directory for saving figures
        self._create_output_dir()
        self._create_figures_dir()

    def _read_json(self):
        """
        Read the JSON data file.

        Returns:
            dict: The JSON data.
        """
        try:
            with open(self.json_file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            print(f"Successfully read JSON data from {self.json_file_path}")
            return data
        except Exception as e:
            print(f"Failed to read JSON file: {e}")
            raise e

    def _create_output_dir(self):
        """
        Create the directory for saving tables if it doesn't exist.
        """
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created directory '{self.output_dir}' for saving tables.")
        else:
            print(f"Directory '{self.output_dir}' already exists.")

    def _create_figures_dir(self):
        """
        Create the directory for saving figures if it doesn't exist.
        """
        if not os.path.exists(self.figures_dir):
            os.makedirs(self.figures_dir)
            print(f"Created directory '{self.figures_dir}' for saving figures.")

    def format_title(self, metric_name):
        """
        Format the title by converting variable names to a readable format.

        Args:
            metric_name (str): Original metric name.

        Returns:
            str: Formatted title.
        """
        if metric_name.startswith("res_egmq1_"):
            formatted = "EGM_q1_" + metric_name[len("res_egmq1_"):].upper()
        elif metric_name.startswith("res_egmq2_"):
            formatted = "EGM_q2_" + metric_name[len("res_egmq2_"):].upper()
        elif metric_name.startswith("res_egmqm_"):
            formatted = "EGM_MultiQubit_" + metric_name[len("res_egmqm_"):].upper()
        else:
            formatted = metric_name.replace("_", " ").title()
        return formatted

    def _calculate_col_widths(self, column_names, data):
        """
        Calculate column widths based on column names and data content.

        Args:
            column_names (list): List of column names.
            data (list of lists): Table data.

        Returns:
            list: List of column widths.
        """
        col_widths = []
        for i, col in enumerate(column_names):
            max_content_width = max(len(str(row[i])) for row in data) if data else 0
            col_width = max(len(col), max_content_width)
            col_widths.append(col_width)
        return col_widths

    def _build_border(self, left, middle, right, col_widths):
        """
        Build a table border.

        Args:
            left (str): Left border character.
            middle (str): Middle connection character.
            right (str): Right border character.
            col_widths (list): List of column widths.

        Returns:
            str: Constructed border string.
        """
        border = left
        for i, width in enumerate(col_widths):
            border += '─' * width
            if i < len(col_widths) - 1:
                border += middle
            else:
                border += right
        border += '\n'
        return border

    def _build_row(self, row, col_widths):
        """
        Build a row for the table.

        Args:
            row (list): Data for the row.
            col_widths (list): List of column widths.

        Returns:
            str: Constructed row string.
        """
        row_str = '│'
        for i, cell in enumerate(row):
            cell_content = str(cell)
            cell_str = f"{cell_content.center(col_widths[i])}"
            row_str += cell_str + '│'
        row_str += '\n'
        return row_str

    def _draw_table_text(self, title, column_names, data, output_filename):
        """
        Create a text-based table and save it to a file.

        Args:
            title (str): Table title.
            column_names (list): List of column names.
            data (list of lists): Table data.
            output_filename (str): Output file name.

        Returns:
            str: Table as a string.
        """
        col_widths = self._calculate_col_widths(column_names, data)
        top_border = self._build_border('┌', '┬', '┐', col_widths)
        header_border = self._build_border('├', '┼', '┤', col_widths)
        bottom_border = self._build_border('└', '┴', '┘', col_widths)
        header = self._build_row(column_names, col_widths)
        data_lines = ''.join(self._build_row(row, col_widths) for row in data)
        table_width = sum(col_widths) + len(col_widths) + 1
        title_formatted = title.center(table_width) + '\n'
        indent = ' ' * 8

        full_table = f"{indent}{title_formatted}{indent}{top_border}{indent}{header}{indent}{header_border}"
        for line in data_lines.splitlines():
            full_table += f"{indent}{line}\n"
        full_table += f"{indent}{bottom_border}"

        output_path = os.path.join(self.output_dir, output_filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_table)
        print(f"Table '{title}' saved as '{output_path}'.")
        return full_table

    def _format_table(self, column_names, data):
        """
        Format the table data as a string.

        Args:
            column_names (list): The column headers of the table.
            data (list): The data rows of the table.

        Returns:
            str: The formatted table as a string.
        """
        # Calculate column widths
        col_widths = self._calculate_col_widths(column_names, data)

        # Construct the table borders and rows
        top_border = self._build_border('┌', '┬', '┐', col_widths)
        header = self._build_row(column_names, col_widths)
        separator = self._build_border('├', '┼', '┤', col_widths)
        data_rows = ''.join(self._build_row(row, col_widths) for row in data)
        bottom_border = self._build_border('└', '┴', '┘', col_widths)

        # Combine all parts to form the full table
        table = f"{top_border}{header}{separator}{data_rows}{bottom_border}"
        return table


    def draw_res_egmq1_rb(self):
        """
        Draws a table for the res_egmq1_rb metric.
        """
        metric_name = "res_egmq1_rb"
        metric_data = self.results.get(metric_name, {})
        if not metric_data:
            print(f"No data found for {metric_name}.")
            return

        data = []
        for q_idx in self.qubit_index_list:
            error_rate = metric_data.get(f"qubit_{q_idx}", {}).get("error_rate", "N/A")
            formatted_error_rate = f"{error_rate:.4f}" if isinstance(error_rate, float) else error_rate
            data.append([str(q_idx), formatted_error_rate])

        column_names = ["Qubit", "Error Rate"]
        title = self.format_title(metric_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{metric_name}_{timestamp}.txt"

        self._draw_table_text(title, column_names, data, output_filename)

    def draw_res_egmq1_xeb(self):
        """
        Draws a table for the res_egmq1_xeb metric.
        """
        metric_name = "res_egmq1_xeb"
        metric_data = self.results.get(metric_name, {})
        hardware_data = metric_data.get("hardware", [])
        if not hardware_data:
            print(f"No hardware data found for {metric_name}.")
            return

        # Draw according to the order of qubit_index_list
        data = []
        for q_idx, infidelity in zip(self.qubit_index_list, hardware_data):
            formatted_infidelity = f"{infidelity:.4f}" if isinstance(infidelity, float) else infidelity
            data.append([str(q_idx), formatted_infidelity])

        column_names = ["Qubit", "Infid XEB"]
        title = self.format_title(metric_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{metric_name}_{timestamp}.txt"

        self._draw_table_text(title, column_names, data, output_filename)

    def draw_res_egmq1_csbp2x(self):
        """
        Draws a table for the res_egmq1_csbp2x metric.
        """
        metric_name = "res_egmq1_csbp2x"
        metric_data = self.results.get(metric_name, {})
        if not metric_data:
            print(f"No data found for {metric_name}.")
            return

        # Extract process_infidelities, stochastic_infidelities, angle_errors
        process_infidelities = metric_data.get("process_infidelities", [])
        stochastic_infidelities = metric_data.get("stochastic_infidelities", [])
        angle_errors = metric_data.get("angle_errors", [])

        # Draw according to the order of qubit_index_list
        data = []
        for q_idx in self.qubit_index_list:
            proc_inf = process_infidelities[q_idx] if q_idx < len(process_infidelities) else "N/A"
            sto_inf = stochastic_infidelities[q_idx] if q_idx < len(stochastic_infidelities) else "N/A"
            angle_err = angle_errors[q_idx] if q_idx < len(angle_errors) else "N/A"

            # Format values to 4 decimal places
            formatted_proc_inf = f"{proc_inf:.4f}" if isinstance(proc_inf, float) else proc_inf
            formatted_sto_inf = f"{sto_inf:.4f}" if isinstance(sto_inf, float) else sto_inf
            formatted_angle_err = f"{angle_err:.4f}" if isinstance(angle_err, float) else angle_err

            data.append([str(q_idx), formatted_proc_inf, formatted_sto_inf, formatted_angle_err])

        column_names = ["Qubit", "Process Infid", "Stotistic Infid", "Angle Error"]
        title = self.format_title(metric_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{metric_name}_{timestamp}.txt"

        self._draw_table_text(title, column_names, data, output_filename)

    def draw_res_egmq2_rb(self):
        """
        Draws a table for the res_egmq2_rb metric.
        """
        metric_name = "res_egmq2_rb"
        metric_data = self.results.get(metric_name, {})
        if not metric_data:
            print(f"No data found for {metric_name}.")
            return

        data = []
        for pair in self.qubit_connectivity:
            key = f"[{pair[0]}, {pair[1]}]"
            error_rate = metric_data.get(key, {}).get("error_rate", "N/A")
            formatted_error_rate = f"{error_rate:.4f}" if isinstance(error_rate, float) else error_rate
            display_pair = f"({pair[0]},{pair[1]})"
            data.append([display_pair, formatted_error_rate])

        # Add average error rate
        average_error_rate = metric_data.get("average_error_rate", "N/A")
        formatted_average_error_rate = f"{average_error_rate:.4f}" if isinstance(average_error_rate, float) else average_error_rate
        data.append(["Average", formatted_average_error_rate])

        column_names = ["Qubits", "Error Rate"]
        title = self.format_title(metric_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{metric_name}_{timestamp}.txt"

        self._draw_table_text(title, column_names, data, output_filename)

    def draw_res_egmq2_xeb(self):
        """
        Draws a table for the res_egmq2_xeb metric.
        """
        metric_name = "res_egmq2_xeb"
        metric_data = self.results.get(metric_name, {})
        hardware_data = metric_data.get("hardware", [])
        if not hardware_data:
            print(f"No hardware data found for {metric_name}.")
            return

        data = []
        for pair, infidelity in zip(self.qubit_connectivity, hardware_data):
            pair_str = f"({pair[0]},{pair[1]})"
            formatted_infidelity = f"{infidelity:.4f}" if isinstance(infidelity, float) else infidelity
            data.append([pair_str, formatted_infidelity])

        column_names = ["Qubits", "Infid XEB"]
        title = self.format_title(metric_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{metric_name}_{timestamp}.txt"

        self._draw_table_text(title, column_names, data, output_filename)

    def draw_res_egmq2_csb_cz(self):
        """
        Draws a table for the res_egmq2_csb_cz metric.
        """
        metric_name = "res_egmq2_csb_cz"
        metric_data = self.results.get(metric_name, {})
        qubit_pairs_results = metric_data.get("qubit_pairs_results", [])
        if not qubit_pairs_results:
            print(f"No qubit pairs results found for {metric_name}.")
            return

        data = []
        for pair, result in zip(self.qubit_connectivity, qubit_pairs_results):
            process_infidelity = result.get("process_infidelity", "N/A")
            stochastic_infidelity = result.get("stochastic_infidelity", "N/A")
            theta_error = result.get("theta_error", "N/A")
            phi_error = result.get("phi_error", "N/A")

            # Format values to 4 decimal places
            formatted_process_infidelity = f"{process_infidelity:.4f}" if isinstance(process_infidelity, float) else process_infidelity
            formatted_stochastic_infidelity = f"{stochastic_infidelity:.4f}" if isinstance(stochastic_infidelity, float) else stochastic_infidelity
            formatted_theta_error = f"{theta_error:.4f}" if isinstance(theta_error, float) else theta_error
            formatted_phi_error = f"{phi_error:.4f}" if isinstance(phi_error, float) else phi_error

            # Format qubit pair as (x,y)
            pair_str = f"({pair[0]},{pair[1]})"

            data.append([pair_str, formatted_process_infidelity, formatted_stochastic_infidelity, formatted_theta_error, formatted_phi_error])

        column_names = ["Qubits", "Process Infid", "CSB_S", "CSB_T", "CSB_A"]
        title = self.format_title(metric_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{metric_name}_{timestamp}.txt"

        self._draw_table_text(title, column_names, data, output_filename)

    def draw_res_egmqm_ghz(self):
        """
        Draw a table for the metric res_egmqm_ghz.
        """
        metric_name = "res_egmqm_ghz"
        metric_data = self.results.get(metric_name, {})
        if not metric_data:
            print(f"No data found for {metric_name}.")
            return

        fidelity_list = metric_data.get("fidelity", [])
        if len(fidelity_list) != 6:
            print(f"Unexpected number of fidelity entries for {metric_name}. Expected 6, got {len(fidelity_list)}.")
            return

        nqubits = [3, 4, 5, 6, 7, 8]
        data = []
        for nq, fid in zip(nqubits, fidelity_list):
            formatted_fid = f"{fid:.4f}" if isinstance(fid, float) else fid
            data.append([str(nq), formatted_fid])

        column_names = ["NQUBITS", "FIDELITY_GHZ"]
        title = self.format_title(metric_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{metric_name}_{timestamp}.txt"

        self._draw_table_text(title, column_names, data, output_filename)

    def draw_res_egmqm_stqv(self):
        """
        Build a Multi-Qubit Gates Quality - Quantum Volume table.
        Returns the generated table string and the maximum Quantum Volume value.
        """
        metric_name_stqv = "res_egmqm_stqv"
        metric_data_stqv = self.results.get(metric_name_stqv, {})

        if not metric_data_stqv:
            print(f"No data found for {metric_name_stqv}.")
            return "N/A", None

        # Dynamically extract nqubits and quantum_volume data
        nqubits_data = []
        qv_data = []

        for key, value in metric_data_stqv.items():
            nqubits = value.get("total_qubits")
            qv = value.get("quantum_volume")

            # Filter out entries where quantum_volume is None or 0
            if nqubits is not None and qv is not None:
                nqubits_data.append(nqubits)
                qv_data.append(qv)

        if not nqubits_data or not qv_data:
            print(f"Incomplete or invalid data for {metric_name_stqv}.")
            return "N/A", None

        # Ensure data is sorted by nqubits
        sorted_data = sorted(zip(nqubits_data, qv_data), key=lambda x: x[0])
        nqubits_data, qv_data = zip(*sorted_data)

        # Dynamically generate table content
        max_qv = max(qv_data) if qv_data else None
        nqubits_row = ["NQubits"] + [str(nq) for nq in nqubits_data]
        qv_row = ["Quantum Volume"] + [str(int(qv)) for qv in qv_data]  # Ensure integers are displayed

        # Calculate column widths
        col_widths = [max(len(str(nq)), len(str(qv))) for nq, qv in zip(nqubits_row, qv_row)]

        # Construct the table
        top_border = self._build_border('┌', '┬', '┐', col_widths)
        header_row = self._build_row(nqubits_row, col_widths)
        separator = self._build_border('├', '┼', '┤', col_widths)
        qv_data_row = self._build_row(qv_row, col_widths)
        bottom_border = self._build_border('└', '┴', '┘', col_widths)

        table = f"{top_border}{header_row}{separator}{qv_data_row}{bottom_border}"

        return table, max_qv

    def draw_res_egmqm_mrb(self):
        """
        Draw a table for the metric res_egmqm_mrb and return the table string.
        """
        metric_name = "res_egmqm_mrb"
        metric_data = self.results.get(metric_name, {})
        if not metric_data:
            print(f"No data found for {metric_name}.")
            return ""

        qubit_groups = metric_data.get("qubit_groups", {})
        if not qubit_groups:
            print(f"No qubit groups data found for {metric_name}.")
            return ""

        # Define column names
        column_names = ["Qubit Group", "Length 1", "Length 2", "Length 3", "Length 4", "Length 5", "Length 6"]

        # Prepare data rows
        data = []
        for group_name, lengths in qubit_groups.items():
            row = [group_name]
            for i in range(1, 7):
                length_key = f"length_{i}"
                length_value = lengths.get(length_key, "N/A")
                if isinstance(length_value, float):
                    formatted_length = f"{length_value:.4f}"
                else:
                    formatted_length = "N/A"
                row.append(formatted_length)
            data.append(row)

        # Format the title
        title = self.format_title(metric_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{metric_name}_{timestamp}.txt"

        # Draw the table and get the table string
        table_str = self._draw_table_text(title, column_names, data, output_filename)

        return table_str




    def generate_all_tables(self):
        """
        Generate all benchmarking tables.
        """
        self.draw_res_egmq1_rb()
        self.draw_res_egmq1_xeb()
        self.draw_res_egmq1_csbp2x()
        self.draw_res_egmq2_rb()
        self.draw_res_egmq2_xeb()
        self.draw_res_egmq2_csb_cz()
        self.draw_res_egmqm_ghz()
        self.draw_res_egmqm_stqv()
        self.draw_res_egmqm_mrb()


    def egm_level02_table(self):
        """
        Generates a comprehensive Level02 report that includes information on 
        Single-Qubit Gate Quality, Two-Qubit Gate Quality, and Multi-Qubit Gates Quality.
        The report title is based on the JSON data file name.
        """
        # Get the JSON file name
        file_name = os.path.basename(self.json_file_path)

        # Initialize the report string
        report = ""

        # Main title for the report
        main_title = "Errorgnomark Report of 'QXX' Chip"

        # Define column names for different categories
        single_q_columns = ["Qubit", "RB", "XEB", "CSB_P", "CSB_S", "CSB_A"]
        two_q_columns = ["Qubits", "RB", "XEB", "CSB_P", "CSB_S", "CSB_T", "CSB_A"]
        multi_q_columns_ghz = ["NQUBITS", "FIDELITY_GHZ"]

        # Build Single-Qubit Gate Quality data
        single_q_data = []
        for q_idx in self.qubit_index_list:
            rb = self.results.get("res_egmq1_rb", {}).get(f"qubit_{q_idx}", {}).get("error_rate", "N/A")
            rb = f"{rb:.4f}" if isinstance(rb, float) else rb

            xeb = self.results.get("res_egmq1_xeb", {}).get("hardware", [])
            xeb = f"{xeb[q_idx]:.4f}" if isinstance(xeb, list) and q_idx < len(xeb) and isinstance(xeb[q_idx], float) else "N/A"

            csb_p = self.results.get("res_egmq1_csbp2x", {}).get("process_infidelities", [])
            csb_p = f"{csb_p[q_idx]:.4f}" if isinstance(csb_p, list) and q_idx < len(csb_p) and isinstance(csb_p[q_idx], float) else "N/A"

            csb_s = self.results.get("res_egmq1_csbp2x", {}).get("stochastic_infidelities", [])
            csb_s = f"{csb_s[q_idx]:.4f}" if isinstance(csb_s, list) and q_idx < len(csb_s) and isinstance(csb_s[q_idx], float) else "N/A"

            csb_a = self.results.get("res_egmq1_csbp2x", {}).get("angle_errors", [])
            csb_a = f"{csb_a[q_idx]:.4f}" if isinstance(csb_a, list) and q_idx < len(csb_a) and isinstance(csb_a[q_idx], float) else "N/A"

            single_q_data.append([str(q_idx), rb, xeb, csb_p, csb_s, csb_a])

        # Build Two-Qubit Gate Quality data
        two_q_data = []
        for pair in self.qubit_connectivity:
            x, y = pair
            qubits = f"({x},{y})"

            rb = self.results.get("res_egmq2_rb", {}).get(f"[{x}, {y}]", {}).get("error_rate", "N/A")
            rb = f"{rb:.4f}" if isinstance(rb, float) else rb

            xeb_list = self.results.get("res_egmq2_xeb", {}).get("hardware", [])
            index = self.qubit_connectivity.index(pair)
            xeb = f"{xeb_list[index]:.4f}" if isinstance(xeb_list, list) and index < len(xeb_list) and isinstance(xeb_list[index], float) else "N/A"

            csb_p = self.results.get("res_egmq2_csb_cz", {}).get("qubit_pairs_results", [])
            if index < len(csb_p):
                process_infid = csb_p[index].get("process_infidelity", "N/A")
                process_infid = f"{process_infid:.4f}" if isinstance(process_infid, float) else process_infid

                stotistic_infid = csb_p[index].get("stochastic_infidelity", "N/A")
                stotistic_infid = f"{stotistic_infid:.4f}" if isinstance(stotistic_infid, float) else stotistic_infid

                theta_error = csb_p[index].get("theta_error", "N/A")
                phi_error = csb_p[index].get("phi_error", "N/A")
                formatted_csb_t = f"{theta_error:.4f}" if isinstance(theta_error, float) else "N/A"
                formatted_csb_a = f"{phi_error:.4f}" if isinstance(phi_error, float) else "N/A"
            else:
                process_infid = "N/A"
                stotistic_infid = "N/A"
                formatted_csb_t = "N/A"
                formatted_csb_a = "N/A"

            two_q_data.append([qubits, rb, xeb, process_infid, stotistic_infid, formatted_csb_t, formatted_csb_a])

        # Build Multi-Qubit Gates Quality - Fidelity_GHZ data
        multi_q_data_ghz = []
        metric_name_ghz = "res_egmqm_ghz"
        metric_data_ghz = self.results.get(metric_name_ghz, {})
        fidelity_list = metric_data_ghz.get("fidelity", [])
        if len(fidelity_list) == 6:
            nqubits_ghz = [3, 4, 5, 6, 7, 8]
            for nq, fid in zip(nqubits_ghz, fidelity_list):
                formatted_fid = f"{fid:.4f}" if isinstance(fid, float) else fid
                multi_q_data_ghz.append([str(nq), formatted_fid])

        # Generate Quantum Volume and MRB tables
        multi_q_table_stqv, max_qv = self.draw_res_egmqm_stqv()
        multi_q_table_mrb = self.draw_res_egmqm_mrb()

        # Assemble the full report
        report += f"{main_title.center(80)}\n\n"
        report += "Single-Qubit Gate Quality\n\n" + self._format_table(single_q_columns, single_q_data) + "\n\n"
        report += "Two-Qubit Gate Quality\n\n" + self._format_table(two_q_columns, two_q_data) + "\n\n"

        if multi_q_data_ghz:
            report += "Multi-Qubit Gates Quality - Fidelity GHZ\n\n"
            report += self._format_table(multi_q_columns_ghz, multi_q_data_ghz) + "\n\n"

        if multi_q_table_stqv:
            report += "Multi-Qubit Gates Quality - Quantum Volume\n\n" + multi_q_table_stqv + "\n"
            if max_qv is not None:
                report += f"Maximum Quantum Volume: {max_qv}\n"

        if multi_q_table_mrb:
            report += "Multi-Qubit Gates Quality - MRB\n\n" + multi_q_table_mrb + "\n"

        # Save the report to a file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"level02_report_{timestamp}.txt"
        output_path = os.path.join(self.output_dir, output_filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Level02 report saved as '{output_path}'.")

    def egm_level02_figure(self):
        """
        Generate visualizations for metrics and save them as images.
        """
        visualizer = VisualPlot(self.json_file_path)

        # Define output file paths
        rbq1_path = os.path.join(self.figures_dir, "RBQ1_Heatmap.png")
        xebq1_path = os.path.join(self.figures_dir, "XEBQ1_Heatmap.png")
        csbq1_path = os.path.join(self.figures_dir, "CSBQ1_Heatmap.png")
        rbq2_path = os.path.join(self.figures_dir, "RBQ2_Heatmap.png")
        xebq2_path = os.path.join(self.figures_dir, "XEBQ2_Heatmap.png")

        # Generate and save the plots
        visualizer.plot_rbq1(grid_size=(5, 5))
        plt.savefig(rbq1_path)
        plt.close()

        visualizer.plot_xebq1(grid_size=(5, 5))
        plt.savefig(xebq1_path)
        plt.close()

        visualizer.plot_csbq1(grid_size=(5, 5))
        plt.savefig(csbq1_path)
        plt.close()

        visualizer.plot_rbq2()
        plt.savefig(rbq2_path)
        plt.close()

        visualizer.plot_xebq2()
        plt.savefig(xebq2_path)
        plt.close()



# import os
# from errorgnomark.results_tools.egm_report_tools import EGMReportManager  # Import your EGMReportManager class

# # Define the path to the JSON file
# current_dir = os.getcwd()  # Get the current working directory
# file_path = os.path.join(current_dir, "data_egm", "Baihua_egm_data_20250103_153026.json")  # Build the full file path

# # Create an instance of EGMReportManager
# report_manager = EGMReportManager(file_path)

# # Generate the Level02 table report
# report_manager.egm_level02_table()

# # Generate the Level02 figures
# report_manager.egm_level02_figure()

# print("Reports and figures have been successfully generated!")
