# Standard library imports
import os  # For operating system-related functionalities
import json  # For working with JSON data
from datetime import datetime  # For handling date and time

# Third-party imports
import matplotlib.pyplot as plt  # For creating visualizations

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
