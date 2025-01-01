import json
import numpy as np
import matplotlib.pyplot as plt

class VisualPlot:
    def __init__(self, file_path):
        """
        初始化 VisualPlot 类
        :param file_path: JSON 文件路径
        """
        self.file_path = file_path
        self.data = self._load_data()

    def _load_data(self):
        """
        加载 JSON 数据
        """
        with open(self.file_path, "r") as f:
            return json.load(f)

    def plot_heatmap_with_spacing(self, grid, title, colorbar_label, cmap='viridis'):
        """
        绘制带有单元格间隙的热力图
        :param grid: 热力图数据
        :param title: 图标题
        :param colorbar_label: 色条标签
        :param cmap: 色彩映射，默认为 viridis
        """
        fig, ax = plt.subplots()
        im = ax.imshow(grid, cmap=cmap, interpolation='nearest')
        plt.colorbar(im, ax=ax, label=colorbar_label)

        # 创建白色的网格间隙
        for i in range(grid.shape[0] + 1):
            ax.axhline(i - 0.5, color='white', linewidth=1.5)
            ax.axvline(i - 0.5, color='white', linewidth=1.5)

        # 设置整数显示
        ax.set_xticks(np.arange(grid.shape[1]))
        ax.set_yticks(np.arange(grid.shape[0]))
        ax.set_xticklabels(np.arange(grid.shape[1]))
        ax.set_yticklabels(np.arange(grid.shape[0]))
        
        # 设置标题和标签
        ax.set_title(title)
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        plt.show()

    def plot_rbq1(self, grid_size=(5, 5)):
        """
        绘制 RB 错误率热力图
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
        绘制 XEB 错误率热力图
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
        绘制 CSB 相关指标的热力图
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
        绘制两比特门 RB 错误率的热力图
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
        绘制两比特门 XEB 错误率的热力图
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


# # 使用示例：
# file_path = "Baihua_egm_data_20250101_143255.json"  # 替换为实际的 JSON 文件路径
# visualizer = VisualPlot(file_path)
# visualizer.plot_rbq1(grid_size=(5, 5))
# visualizer.plot_xebq1(grid_size=(5, 5))
# visualizer.plot_csbq1(grid_size=(5, 5))
# visualizer.plot_rbq2()
# visualizer.plot_xebq2()
