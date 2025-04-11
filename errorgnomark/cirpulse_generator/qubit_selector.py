import networkx as nx
import matplotlib.pyplot as plt

from Qubit_topology_selection import build_chessboard_graph, visualize_chessboard, node_name_to_index, save_connections
from Chip_topology_matrix_construction import create_chessboard_arrays
from Qubit_selection_scorer import select_connected_nodes


class chip:
    """
    Represents a quantum chip with a 2D grid of qubits.

    This class provides the basic structure of a quantum chip, 
    defined by the number of rows and columns. It serves as 
    the foundation for qubit selection and connectivity in a 
    quantum system.

    Attributes:
        rows (int): Number of rows in the chip grid.
        columns (int): Number of columns in the chip grid.
    """

    def __init__(self, rows=12, columns=13):
        self.rows = rows
        self.columns = columns


def qubits_selection(row=12, col=13, qubits_to_be_used=9,
                     file_path="E:\Repositories\ErrorGnoMark\ScQ-Baihua信息.xlsx",
                     visualize_chessboard_decision=True,
                     visualize_available_board_decision=True):
    # 构建棋盘格图
    chessboard_graph, available_nodes = (
        build_chessboard_graph(chip_row=row, chip_col=col,
                               file_path=file_path))

    if chessboard_graph is not None:
        # ======================可视化棋盘格======================
        # 同时保存txt文件
        if visualize_chessboard_decision:
            visualize_chessboard(chessboard_graph, available_nodes)

        # 打印一些基本信息
        print(f"棋盘格包含 {len(chessboard_graph.nodes)} 个节点")
        print(f"其中 {len(available_nodes)} 个节点可用")
        print(f"棋盘格包含 {len(chessboard_graph.edges)} 条边")
        print("可用节点示例:", available_nodes[:10], "...")  # 只显示前10个可用节点
        print("边列表示例:", list(chessboard_graph.edges)[:5], "...")  # 只显示前5条边
        # 将边列表保存为txt文件
        with open('chessboard_edges.txt', 'w') as f:
            for edge in chessboard_graph.edges:
                f.write(f"{edge[0]}_{edge[1]}\n")
        print("边列表已保存为 chessboard_edges.txt")
        # 将棋盘格保存为txt，可用的节点值为1，不可用为0.
        create_chessboard_arrays(file_path=r"E:\Repositories\ErrorGnoMark\ScQ-Baihua信息.xlsx")

        # ======================根据芯片拓扑结构选择节点======================
        # 根据节点得分，选择节点
        selected_nodes, edge_count = select_connected_nodes(chessboard_graph, available_nodes, qubits_to_be_used)

        # 打印结果
        print(f"\n选中的{qubits_to_be_used}个节点:", selected_nodes)
        print(f"这些节点间的连接数: {edge_count}")
        print(f"平均每个节点的连接数: {edge_count / qubits_to_be_used:.2f}")

        if visualize_available_board_decision:
            # 可视化选中的节点
            pos = nx.get_node_attributes(chessboard_graph, 'pos')
            plt.figure(figsize=(15, 10))

            # 绘制所有节点
            nx.draw_networkx_nodes(chessboard_graph, pos, node_color='lightgray', node_size=100)
            nx.draw_networkx_edges(chessboard_graph, pos, edge_color='lightgray')

            # 高亮显示选中的节点和连接
            subgraph = chessboard_graph.subgraph(selected_nodes)
            nx.draw_networkx_nodes(subgraph, pos, node_color='red', node_size=300)
            nx.draw_networkx_edges(subgraph, pos, edge_color='red', width=2)

            # 绘制标签
            labels = {node: node for node in selected_nodes}
            nx.draw_networkx_labels(chessboard_graph, pos, labels, font_size=8)

            plt.title(f"选中的{qubits_to_be_used}个相邻节点（红色）")
            plt.show()

        # ======================将选中的节点和对应的连接关系转换为best_selection======================
        # 转换为序号并排序
        node_indices = sorted([node_name_to_index(node) for node in selected_nodes])
        # 打印结果
        print(f"节点序号(排序后): {node_indices}")

        # 保存所有连接关系
        all_connections = save_connections(chessboard_graph)
        print(f"已保存{len(all_connections)}条连接关系到connections.txt")
        # 保存选中节点的连接关系
        selected_connections = []
        for edge in chessboard_graph.edges:
            node1, node2 = edge
            if node1 in selected_nodes and node2 in selected_nodes:
                idx1 = node_name_to_index(node1)
                idx2 = node_name_to_index(node2)
                selected_connections.append([idx1, idx2])

        with open('selected_connections.txt', 'w') as f:
            for conn in selected_connections:
                f.write(f"{conn}\n")
        print(f"已保存{len(selected_connections)}条选中节点的连接关系到selected_connections.txt")

        # 保存选中的节点到文件
        with open('selected_nodes.txt', 'w') as f:
            f.write(f"Selected {qubits_to_be_used} nodes:\n")
            f.write("\n".join(selected_nodes))
            f.write(f"\n\nEdge count between selected nodes: {edge_count}")
            f.write(f"\n\n节点序号(排序后): {node_indices}")
        print("\n选中的节点已保存到 selected_nodes.txt")

    return node_indices, selected_connections


class qubit_selection:
    """
    Selects qubits and their connectivity from a quantum chip based on specified constraints.

    Parameters:
        chip (chip): An instance of the chip class, defining the chip layout.
        qubit_index_max (int): Maximum allowable qubit index (default: 50).
        qubit_number (int): Number of qubits to select (default: 9).
        option (dict, optional): Selection options, including:
            - "max_qubits_per_row" (int): Maximum number of qubits per row.
            - "min_qubit_index" (int): Minimum allowable qubit index.
            - "max_qubit_index" (int): Maximum allowable qubit index.

    Methods:
        quselected():
            Returns selected qubit indices and their connectivity as a dictionary.
            Visualizes the selected qubits and connections on the chip grid.

    Features:
        - Adapts qubit selection based on chip layout and constraints.
        - Ensures logical connectivity for selected qubits.
    """

    def __init__(self, chip, qubit_index_max=50, qubit_number=9,
                 file_path="E:\Repositories\ErrorGnoMark\ScQ-Baihua信息.xlsx", option=None):
        self.chip = chip
        self.rows = getattr(chip, 'rows', 12)  # Default to 12 rows (Baihua) if not specified
        self.columns = getattr(chip, 'columns', 13)  # Default to 13 columns (Baihua) if not specified

        # self.qubit_index_max = qubit_index_max
        # self.qubit_number = int(qubit_number)  # Ensure this is an integer

        self.qubit_index_max = self.rows * self.columns - 1  # The number of the chip starts from 0
        self.qubit_number = int(qubit_number)  # Ensure this is an integer
        self.file_path = file_path
        self.option = option if option is not None else {}

    def quselected(self):
        """
        Selects qubits and their connectivity based on the specified constraints.

        Returns:
            dict: 
                - "qubit_index_list" (list): Indices of selected qubits.
                - "qubit_connectivity" (list): Connectivity data as pairs of qubits.
        """
        # Retrieve constraints from options
        # max_qubits_per_row = self.option.get('max_qubits_per_row', self.columns)  # Default to all qubits in a row，未使用
        min_qubit_index = self.option.get('min_qubit_index', 0)
        max_qubit_index = self.option.get('max_qubit_index', self.qubit_index_max)

        # Ensure qubit_number does not exceed the total qubits in the grid
        total_qubits = self.rows * self.columns
        qubit_number = min(self.qubit_number, total_qubits)

        best_selection = {
            "qubit_index_list": [],
            "qubit_connectivity": []
        }

        # Select qubits row by row, left to right
        # TODO qubits序号的选择要考虑芯片的拓扑结构
        # qubit_indices = []
        # for r in range(self.rows):
        #     for c in range(self.columns):
        #         q = r * self.columns + c
        #         if q >= min_qubit_index and q <= max_qubit_index:
        #             qubit_indices.append(q)
        #             if len(qubit_indices) == qubit_number:
        #                 break
        #     if len(qubit_indices) == qubit_number:
        #         break
        #
        # best_selection["qubit_index_list"] = qubit_indices
        #
        # # Create connectivity for horizontal and vertical connections
        # qubit_connectivity = []
        #
        # # Horizontal connectivity (left-right)
        # for r in range(self.rows):
        #     for c in range(self.columns - 1):  # Only connect to the right
        #         q1 = r * self.columns + c
        #         q2 = r * self.columns + (c + 1)
        #         if q1 in qubit_indices and q2 in qubit_indices:
        #             qubit_connectivity.append([q1, q2])
        #
        # # Vertical connectivity (up-down)
        # for r in range(self.rows - 1):  # Only connect down
        #     for c in range(self.columns):
        #         q1 = r * self.columns + c
        #         q2 = (r + 1) * self.columns + c
        #         if q1 in qubit_indices and q2 in qubit_indices:
        #             qubit_connectivity.append([q1, q2])
        #
        # best_selection["qubit_connectivity"] = qubit_connectivity
        #
        # # If no selection was found, return empty lists
        # if not best_selection["qubit_index_list"]:
        #     return best_selection

        qubit_indices, qubit_connectivity = (
            qubits_selection(row=self.rows, col=self.columns,
                             qubits_to_be_used=qubit_number,
                             file_path=self.file_path,
                             visualize_chessboard_decision=True,
                             visualize_available_board_decision=True))

        best_selection["qubit_index_list"] = qubit_indices
        best_selection["qubit_connectivity"] = qubit_connectivity

        # Print chip structure
        # print(f"Chip Structure: {self.rows} rows x {self.columns} columns")
        # print("Selected Qubits and Connectivity:")
        #
        # # Create a grid to visualize selected qubits and connectivity
        # grid = [["." for _ in range(self.columns)] for _ in range(self.rows)]
        #
        # for q in best_selection["qubit_index_list"]:
        #     row = q // self.columns
        #     col = q % self.columns
        #     grid[row][col] = "Q"
        #
        # for conn in best_selection["qubit_connectivity"]:
        #     q1, q2 = conn
        #     r1, c1 = q1 // self.columns, q1 % self.columns
        #     r2, c2 = q2 // self.columns, q2 % self.columns
        #     if r1 == r2:  # Horizontal connection
        #         grid[r1][c1] = "-"
        #     elif c1 == c2:  # Vertical connection
        #         grid[r1][c1] = "|"
        #
        # for row in grid:
        #     print(" ".join(row))

        return best_selection

# Example usage:
# chip_instance = chip(rows=12, columns=13)
# qubit_selector = qubit_selection(chip_instance, qubit_number=10)
# qubit_selector.quselected()
