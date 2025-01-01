import os
import json
from datetime import datetime
from visualization import VisualPlot
import matplotlib.pyplot as plt

class EGMReportManager:
    """
    负责读取 EGM JSON 数据文件，并为各个指标生成文本格式表格的类。
    """

    def __init__(self, json_file_path):
        """
        初始化 EGMReportManager 类，读取 JSON 数据文件。

        参数:
            json_file_path (str): JSON 数据文件的路径。
        """
        self.json_file_path = json_file_path
        self.data = self._read_json()
        self.results = self.data.get("results", {})
        self.qubit_index_list = self.data.get("qubit_index_list", [])
        self.qubit_connectivity = self.data.get("qubit_connectivity", [])
        self.output_dir = "tables"
        self.figures_dir = "figures"  # 图像保存目录
        self._create_output_dir()
        self._create_figures_dir()

    def _read_json(self):
        """
        读取 JSON 数据文件。

        返回:
            dict: 读取的 JSON 数据。
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
        创建用于保存表格文本文件的目录（如果不存在）。
        """
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created directory '{self.output_dir}' for saving tables.")
        else:
            print(f"Directory '{self.output_dir}' already exists.")

    def format_title(self, metric_name):
        """
        格式化标题，将变量名转为指定格式。
        例如：
            'res_egmq1_rb' -> 'EGM_q1_RB'
            'res_egmq1_csbp2x' -> 'EGM_q1_CSBP2X'
            'res_egmq2_rb' -> 'EGM_q2_RB'
            'res_egmqm_mrb' -> 'EGM_MultiQubit_MRB'
        """
        if metric_name.startswith("res_egmq1_"):
            formatted = "EGM_q1_" + metric_name[len("res_egmq1_"):].upper()
        elif metric_name.startswith("res_egmq2_"):
            formatted = "EGM_q2_" + metric_name[len("res_egmq2_"):].upper()
        elif metric_name.startswith("res_egmqm_"):
            formatted = "EGM_MultiQubit_" + metric_name[len("res_egmqm_"):].upper()
        else:
            # 如果不符合预期格式，按默认方式格式化
            formatted = metric_name.replace("_", " ").title()
        return formatted

    def _calculate_col_widths(self, column_names, data):
        """
        计算每列的宽度，基于列名和数据内容。

        参数:
            column_names (list): 列名列表。
            data (list of lists): 表格数据。

        返回:
            list: 每列的宽度列表。
        """
        col_widths = []
        for i, col in enumerate(column_names):
            max_content_width = max(len(str(row[i])) for row in data) if data else 0
            col_width = max(len(col), max_content_width)  # 无额外内边距
            col_widths.append(col_width)
        return col_widths

    def _build_border(self, left, middle, right, col_widths):
        """
        构建表格的边框。

        参数:
            left (str): 左侧边框字符。
            middle (str): 中间连接字符。
            right (str): 右侧边框字符。
            col_widths (list): 每列的宽度列表。

        返回:
            str: 构建好的边框字符串。
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
        构建表格的一行。

        参数:
            row (list): 行数据。
            col_widths (list): 每列的宽度列表。

        返回:
            str: 构建好的行字符串。
        """
        row_str = '│'
        for i, cell in enumerate(row):
            cell_content = str(cell)
            # 居中对齐
            cell_str = f"{cell_content.center(col_widths[i])}"
            row_str += cell_str + '│'
        row_str += '\n'
        return row_str

    def _draw_table_text(self, title, column_names, data, output_filename):
        """
        使用 Python 字符绘制表格并保存为文本文件，同时返回表格字符串。

        参数:
            title (str): 表格标题。
            column_names (list): 表格的列名列表。
            data (list of lists): 表格的数据，每个子列表代表一行。
            output_filename (str): 输出文本文件名。

        返回:
            str: 构建好的表格字符串。
        """
        # 计算列宽
        col_widths = self._calculate_col_widths(column_names, data)

        # 构建边框
        top_border = self._build_border('┌', '┬', '┐', col_widths)
        header_border = self._build_border('├', '┼', '┤', col_widths)
        bottom_border = self._build_border('└', '┴', '┘', col_widths)

        # 构建表头
        header = self._build_row(column_names, col_widths)

        # 构建数据行
        data_lines = ''
        for row in data:
            data_lines += self._build_row(row, col_widths)

        # 构建标题
        table_width = sum(col_widths) + len(col_widths) + 1  # 总宽度 = 列宽之和 + 竖线数 + 1
        title_formatted = title.center(table_width) + '\n'

        # 定义前导空格（约8个空格）
        indent = ' ' * 8

        # 组合所有部分并添加前导空格
        full_table = f"{indent}{title_formatted}"
        full_table += f"{indent}{top_border}"
        full_table += f"{indent}{header}"
        full_table += f"{indent}{header_border}"
        for line in data_lines.splitlines():
            full_table += f"{indent}{line}\n"
        full_table += f"{indent}{bottom_border}"

        # 保存到文本文件
        output_path = os.path.join(self.output_dir, output_filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_table)
        print(f"Table '{title}' saved as '{output_path}'.")

        return full_table  # 返回表格字符串 

    def draw_res_egmq1_rb(self):
        """
        绘制 res_egmq1_rb 指标的表格。
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
        绘制 res_egmq1_xeb 指标的表格。
        """
        metric_name = "res_egmq1_xeb"
        metric_data = self.results.get(metric_name, {})
        hardware_data = metric_data.get("hardware", [])
        if not hardware_data:
            print(f"No hardware data found for {metric_name}.")
            return

        # 按照 qubit_index_list 顺序绘制
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
        绘制 res_egmq1_csbp2x 指标的表格。
        """
        metric_name = "res_egmq1_csbp2x"
        metric_data = self.results.get(metric_name, {})
        if not metric_data:
            print(f"No data found for {metric_name}.")
            return

        # 提取 process_infidelities, stochastic_infidelities, angle_errors
        process_infidelities = metric_data.get("process_infidelities", [])
        stochastic_infidelities = metric_data.get("stochastic_infidelities", [])
        angle_errors = metric_data.get("angle_errors", [])

        # 按照 qubit_index_list 顺序绘制
        data = []
        for q_idx in self.qubit_index_list:
            if q_idx >= len(process_infidelities):
                proc_inf = "N/A"
            else:
                proc_inf = process_infidelities[q_idx]

            if q_idx >= len(stochastic_infidelities):
                sto_inf = "N/A"
            else:
                sto_inf = stochastic_infidelities[q_idx]

            if q_idx >= len(angle_errors):
                angle_err = "N/A"
            else:
                angle_err = angle_errors[q_idx]

            # 格式化数值为4位有效数字
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
        绘制 res_egmq2_rb 指标的表格。
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

        # 添加平均错误率
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
        绘制 res_egmq2_xeb 指标的表格。
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
        绘制 res_egmq2_csb_cz 指标的表格。
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

            # 格式化数值为4位有效数字
            formatted_process_infidelity = f"{process_infidelity:.4f}" if isinstance(process_infidelity, float) else process_infidelity
            formatted_stochastic_infidelity = f"{stochastic_infidelity:.4f}" if isinstance(stochastic_infidelity, float) else stochastic_infidelity
            formatted_theta_error = f"{theta_error:.4f}" if isinstance(theta_error, float) else theta_error
            formatted_phi_error = f"{phi_error:.4f}" if isinstance(phi_error, float) else phi_error

            # 格式化 qubit pair 为 (x,y)
            pair_str = f"({pair[0]},{pair[1]})"

            # 添加 CSB_T 和 CSB_A
            formatted_csb_t = f"{theta_error:.4f}" if isinstance(theta_error, float) else "N/A"
            formatted_csb_a = f"{phi_error:.4f}" if isinstance(phi_error, float) else "N/A"

            data.append([pair_str, formatted_process_infidelity, formatted_stochastic_infidelity, formatted_csb_t, formatted_csb_a])

        column_names = ["Qubits", "Process Infid", "CSB_S", "CSB_T", "CSB_A"]
        title = self.format_title(metric_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{metric_name}_{timestamp}.txt"

        self._draw_table_text(title, column_names, data, output_filename)

    def draw_res_egmqm_ghz(self):
        """
        绘制 res_egmqm_ghz 指标的表格。
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
        构建 Multi-Qubit Gates Quality - Quantum Volume 表格。
        返回生成的表格字符串和最大 Quantum Volume 值。
        """
        metric_name_stqv = "res_egmqm_stqv"
        metric_data_stqv = self.results.get(metric_name_stqv, {})

        if not metric_data_stqv:
            print(f"No data found for {metric_name_stqv}.")
            return "N/A", None

        # 动态提取 nqubits 和 quantum_volume 数据
        nqubits_data = []
        qv_data = []

        for key, value in metric_data_stqv.items():
            nqubits = value.get("total_qubits")
            qv = value.get("quantum_volume")

            # 过滤掉 quantum_volume 为 None 或 0 的数据
            if nqubits is not None and qv is not None:
                nqubits_data.append(nqubits)
                qv_data.append(qv)

        if not nqubits_data or not qv_data:
            print(f"Incomplete or invalid data for {metric_name_stqv}.")
            return "N/A", None

        # 确保按照 nqubits 顺序排序
        sorted_data = sorted(zip(nqubits_data, qv_data), key=lambda x: x[0])
        nqubits_data, qv_data = zip(*sorted_data)

        # 动态生成表格内容
        max_qv = max(qv_data) if qv_data else None
        nqubits_row = ["NQubits"] + [str(nq) for nq in nqubits_data]
        qv_row = ["Quantum Volume"] + [str(int(qv)) for qv in qv_data]  # 确保整数显示

        # 计算列宽
        col_widths = [max(len(str(nq)), len(str(qv))) for nq, qv in zip(nqubits_row, qv_row)]

        # 构建表格
        top_border = self._build_border('┌', '┬', '┐', col_widths)
        header_row = self._build_row(nqubits_row, col_widths)
        separator = self._build_border('├', '┼', '┤', col_widths)
        qv_data_row = self._build_row(qv_row, col_widths)
        bottom_border = self._build_border('└', '┴', '┘', col_widths)

        table = f"{top_border}{header_row}{separator}{qv_data_row}{bottom_border}"

        return table, max_qv


    def draw_res_egmqm_mrb(self):
        """
        绘制 res_egmqm_mrb 指标的表格，并返回表格字符串。
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

        # 定义列名
        column_names = ["Qubit Group", "Length 1", "Length 2", "Length 3", "Length 4", "Length 5", "Length 6"]

        # 准备数据行
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

        # 格式化标题
        title = self.format_title(metric_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{metric_name}_{timestamp}.txt"

        # 绘制表格并获取表格字符串
        table_str = self._draw_table_text(title, column_names, data, output_filename)

        return table_str

    def generate_all_tables(self):
        """
        生成所有指标的表格。
        """
        self.draw_res_egmq1_rb()
        self.draw_res_egmq1_xeb()
        self.draw_res_egmq1_csbp2x()
        self.draw_res_egmq2_rb()
        self.draw_res_egmq2_xeb()
        self.draw_res_egmq2_csb_cz()
        self.draw_res_egmqm_ghz()      # 保持原有 GHZ 的表格生成
        self.draw_res_egmqm_stqv()     # 新增绘制 res_egmqm_stqv 的表格
        self.draw_res_egmqm_mrb()      # 新增绘制 res_egmqm_mrb 的表格

    def egm_level02_table(self):
                """
                生成整体的 Level02 报表，包含 Single-Qubit Gate Quality, Two-Qubit Gate Quality 和 Multi-Qubit Gates Quality 三大类的信息。
                报表的标题为 JSON 数据文件的文件名。
                """
                # 获取文件名
                file_name = os.path.basename(self.json_file_path)

                # 初始化报表字符串
                report = ""

                # 构建主标题
                main_title = "Errorgnomark Report of 'QXX' Chip"


                # 定义表格的列名
                single_q_columns = ["Qubit", "RB", "XEB", "CSB_P", "CSB_S", "CSB_A"]
                two_q_columns = ["Qubits", "RB", "XEB", "CSB_P", "CSB_S", "CSB_T", "CSB_A"]
                multi_q_columns_ghz = ["NQUBITS", "FIDELITY_GHZ"]

                # 构建 Single-Qubit Gate Quality 数据
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

                # 构建 Two-Qubit Gate Quality 数据
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

                # 计算 Single-Qubit 表格列宽
                single_q_col_widths = self._calculate_col_widths(single_q_columns, single_q_data)
                
                # 构建 Single-Qubit 表格
                single_q_top_border = self._build_border('┌', '┬', '┐', single_q_col_widths)
                single_q_header = self._build_row(single_q_columns, single_q_col_widths)
                single_q_header_border = self._build_border('├', '┼', '┤', single_q_col_widths)
                single_q_data_lines = ''
                for row in single_q_data:
                    single_q_data_lines += self._build_row(row, single_q_col_widths)
                single_q_bottom_border = self._build_border('└', '┴', '┘', single_q_col_widths)
                single_q_table = f"{single_q_top_border}{single_q_header}{single_q_header_border}{single_q_data_lines}{single_q_bottom_border}"

                # 计算 Two-Qubit 表格列宽
                two_q_col_widths = self._calculate_col_widths(two_q_columns, two_q_data)

                # 构建 Two-Qubit 表格
                two_q_top_border = self._build_border('┌', '┬', '┐', two_q_col_widths)
                two_q_header = self._build_row(two_q_columns, two_q_col_widths)
                two_q_header_border = self._build_border('├', '┼', '┤', two_q_col_widths)
                two_q_data_lines = ''
                for row in two_q_data:
                    two_q_data_lines += self._build_row(row, two_q_col_widths)
                two_q_bottom_border = self._build_border('└', '┴', '┘', two_q_col_widths)
                two_q_table = f"{two_q_top_border}{two_q_header}{two_q_header_border}{two_q_data_lines}{two_q_bottom_border}"

                # 构建 Multi-Qubit Gates Quality - Fidelity_GHZ 数据
                multi_q_data_ghz = []
                metric_name_ghz = "res_egmqm_ghz"
                metric_data_ghz = self.results.get(metric_name_ghz, {})
                if not metric_data_ghz:
                    print(f"No data found for {metric_name_ghz}.")
                    # 可以选择跳过，或者添加 N/A 表示缺失
                else:
                    fidelity_list = metric_data_ghz.get("fidelity", [])
                    if len(fidelity_list) != 6:
                        print(f"Unexpected number of fidelity entries for {metric_name_ghz}. Expected 6, got {len(fidelity_list)}.")
                    else:
                        nqubits_ghz = [3, 4, 5, 6, 7, 8]
                        for nq, fid in zip(nqubits_ghz, fidelity_list):
                            formatted_fid = f"{fid:.4f}" if isinstance(fid, float) else fid
                            multi_q_data_ghz.append([str(nq), formatted_fid])

                # 构建 Multi-Qubit Gates Quality - Quantum Volume 数据
                multi_q_table_stqv, max_qv = self.draw_res_egmqm_stqv()

                # 构建 Multi-Qubit Gates Quality - MRB 数据
                multi_q_table_mrb = self.draw_res_egmqm_mrb()

                # Assemble the full report
                report += f"{main_title.center(sum(single_q_col_widths) + len(single_q_col_widths) + 1)}\n\n"
                
                report += "Single-Qubit Gate Quality".center(sum(single_q_col_widths) + len(single_q_col_widths) + 1) + "\n\n"
                report += single_q_table + "\n\n"
                report += "Two-Qubit Gate Quality".center(sum(two_q_col_widths) + len(two_q_col_widths) + 1) + "\n\n"
                report += two_q_table + "\n\n"

                if multi_q_data_ghz or multi_q_table_stqv or multi_q_table_mrb:
                    multi_q_title = "Multi-Qubit Gates Quality"
                    # 计算 Multi-Qubit Gates Quality 部分的总宽度
                    total_multi_q_width = max(
                        sum(self._calculate_col_widths(multi_q_columns_ghz, multi_q_data_ghz)) + len(multi_q_columns_ghz) + 1 if multi_q_data_ghz else 0,
                        len(multi_q_table_stqv.split('\n')[0]) if multi_q_table_stqv else 0,
                        len(multi_q_table_mrb.split('\n')[0]) if multi_q_table_mrb else 0
                    )
                    report += f"{multi_q_title.center(total_multi_q_width)}\n\n"

                    if multi_q_data_ghz:
                        multi_q_col_widths_ghz = self._calculate_col_widths(multi_q_columns_ghz, multi_q_data_ghz)
                        multi_q_top_border_ghz = self._build_border('┌', '┬', '┐', multi_q_col_widths_ghz)
                        multi_q_header_ghz = self._build_row(multi_q_columns_ghz, multi_q_col_widths_ghz)
                        multi_q_header_border_ghz = self._build_border('├', '┼', '┤', multi_q_col_widths_ghz)
                        multi_q_data_lines_ghz = ''
                        for row in multi_q_data_ghz:
                            multi_q_data_lines_ghz += self._build_row(row, multi_q_col_widths_ghz)
                        multi_q_bottom_border_ghz = self._build_border('└', '┴', '┘', multi_q_col_widths_ghz)
                        multi_q_table_ghz = f"{multi_q_top_border_ghz}{multi_q_header_ghz}{multi_q_header_border_ghz}{multi_q_data_lines_ghz}{multi_q_bottom_border_ghz}"
                        report += multi_q_table_ghz + "\n\n"

                    if multi_q_table_stqv:
                        report += multi_q_table_stqv + "\n"

                        if max_qv is not None:
                            report += f"Quantum Volume is  {max_qv}\n"
                        else:
                            report += f"Quantum Volume is  N/A\n"

                    if multi_q_table_mrb:
                        report += multi_q_table_mrb + "\n"

                # 保存报表到文本文件
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"level02_report_{timestamp}.txt"
                output_path = os.path.join(self.output_dir, output_filename)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(report)
                print(f"Level02 report saved as '{output_path}'.")
                
                # 获取 CLOPS 信息
                clops_data = self.results.get("res_egmqm_clops", {})
                clops_value = clops_data.get("CLOPSQM", "N/A")

                # 获取 VQE 信息
                vqe_data = self.results.get("res_egmqm_vqe", {})
                vqe_problem_description = vqe_data.get("Problem Description", "N/A")
                vqe_final_energy = vqe_data.get("Final Energy", "N/A")

                # 打印调试信息
                print("CLOPS Data:", clops_data)
                print("VQE Data:", vqe_data)

                # 构建报告
                report += "\n\n"
                report += f"CLOPS: {clops_value:.4g}\n"
                if clops_value != "N/A":
                    report += "CLOPS is a metric indicating the execution speed of quantum processors, specifically measuring how quickly a processor can run layers of parameterized circuits similar to those used for Quantum Volume."
                else:
                    report += "CLOPS data is unavailable.\n"

                report += "\n\n"
                if vqe_problem_description != "N/A" and vqe_final_energy != "N/A":
                    report += f"VQE Problem: {vqe_problem_description}\n"
                    report += f"Final Energy: {vqe_final_energy:.4f}\n"
                else:
                    report += "VQE information is unavailable.\n"

                # 最终报告打印
                print(report)

                # 保存报告到文件
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"level02_report_{timestamp}.txt"
                output_path = os.path.join(self.output_dir, output_filename)

                # 确认文件路径
                print(f"Saving report to: {output_path}")

                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(report) 

    def _create_figures_dir(self):
        """
        创建用于保存图像的目录（如果不存在）。
        """
        if not os.path.exists(self.figures_dir):
            os.makedirs(self.figures_dir)
            print(f"Created directory '{self.figures_dir}' for saving figures.")
        else:
            print(f"Directory '{self.figures_dir}' already exists.")

    def egm_level02_figure(self):
        """
        生成指标可视化图像并保存。
        """
        # 初始化 VisualPlot 实例
        visualizer = VisualPlot(self.json_file_path)

        # 定义输出文件路径
        rbq1_path = os.path.join(self.figures_dir, "RBQ1_Heatmap.png")
        xebq1_path = os.path.join(self.figures_dir, "XEBQ1_Heatmap.png")
        csbq1_path = os.path.join(self.figures_dir, "CSBQ1_Heatmap.png")
        rbq2_path = os.path.join(self.figures_dir, "RBQ2_Heatmap.png")
        xebq2_path = os.path.join(self.figures_dir, "XEBQ2_Heatmap.png")

        # 绘制并保存图形
        print("Generating RBQ1 Heatmap...")
        visualizer.plot_rbq1(grid_size=(5, 5))
        plt.savefig(rbq1_path)
        plt.close()
        print(f"RBQ1 Heatmap saved at {rbq1_path}")

        print("Generating XEBQ1 Heatmap...")
        visualizer.plot_xebq1(grid_size=(5, 5))
        plt.savefig(xebq1_path)
        plt.close()
        print(f"XEBQ1 Heatmap saved at {xebq1_path}")

        print("Generating CSBQ1 Heatmap...")
        visualizer.plot_csbq1(grid_size=(5, 5))
        plt.savefig(csbq1_path)
        plt.close()
        print(f"CSBQ1 Heatmap saved at {csbq1_path}")

        print("Generating RBQ2 Heatmap...")
        visualizer.plot_rbq2()
        plt.savefig(rbq2_path)
        plt.close()
        print(f"RBQ2 Heatmap saved at {rbq2_path}")

        print("Generating XEBQ2 Heatmap...")
        visualizer.plot_xebq2()
        plt.savefig(xebq2_path)
        plt.close()
        print(f"XEBQ2 Heatmap saved at {xebq2_path}")

# # 示例用法
# if __name__ == "__main__":
#     # 假设 JSON 文件名为 'Baihua_egm_data_20250101_143255.json'
#     json_file = "Baihua_egm_data_20250101_143255.json"
#     # json_file = "Baihua_egm_data_20250101_143255.json"

#     # 创建 EGMReportManager 实例
#     report_manager = EGMReportManager(json_file)

#     # # 生成所有表格（包括 res_egmqm_stqv 和 res_egmqm_mrb）
#     # report_manager.generate_all_tables()

#     # 生成 Level02 综合报表
#     report_manager.egm_level02_table()

#     report_manager.egm_level02_figure()


