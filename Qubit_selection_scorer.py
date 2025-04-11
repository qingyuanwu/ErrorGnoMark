import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
# 因为Qubit_topology_selection.py中import了此脚本，所以在运行Qubit_topology_selection时要注释第6行
# from Qubit_topology_selection import build_chessboard_graph

def select_connected_nodes(chessboard_graph, available_nodes, X):
    """
    选择X个相邻且连接关系多的可用节点

    参数:
        chessboard_graph: 棋盘格图
        available_nodes: 可用节点列表
        X: 要选择的节点数量

    返回:
        选中的节点列表和这些节点间的连接数
    """
    # 创建可用节点的子图
    available_graph = chessboard_graph.subgraph(available_nodes).copy()

    # 找出所有连通分量，并按大小排序
    connected_components = sorted(nx.connected_components(available_graph),
                                  key=len, reverse=True)

    # 检查是否有足够大的连通分量
    for component in connected_components:
        if len(component) >= X:
            # 在这个连通分量中寻找包含最多边的X个节点
            subgraph = available_graph.subgraph(component)
            best_nodes = None
            max_edges = -1

            # 由于组合数可能很大，这里采用启发式方法
            # 方法1：从度数最高的节点开始扩展
            degrees = dict(subgraph.degree())
            sorted_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)

            selected = set(sorted_nodes[:1])  # 从度数最高的节点开始

            while len(selected) < X and len(selected) < len(sorted_nodes):
                # 找到与已选节点相连的可用节点
                candidates = set()
                for node in selected:
                    candidates.update(neighbor for neighbor in subgraph.neighbors(node)
                                      if neighbor not in selected)

                if not candidates:
                    break  # 没有更多相连节点

                # 选择度数最高的候选节点
                next_node = max(candidates, key=lambda x: degrees[x])
                selected.add(next_node)

            if len(selected) == X:
                edge_count = subgraph.subgraph(selected).number_of_edges()
                return list(selected), edge_count

            # 方法2：如果方法1失败，尝试随机采样（简化版）
            # 实际应用中可能需要更复杂的算法
            for node in component:
                neighbors = list(nx.neighbors(subgraph, node))
                if len(neighbors) >= X - 1:
                    selected = [node] + neighbors[:X - 1]
                    edge_count = subgraph.subgraph(selected).number_of_edges()
                    return selected, edge_count

    # 如果没有找到足够大的连通分量，返回最大的可能集合
    largest_component = connected_components[0]
    edge_count = available_graph.subgraph(largest_component).number_of_edges()
    return list(largest_component), edge_count


if __name__ == '__main__':
    # 构建棋盘格图
    chessboard_graph, available_nodes = (
        build_chessboard_graph(chip_row=12, chip_col=13,
                               file_path="E:\Repositories\ErrorGnoMark\ScQ-Baihua信息.xlsx"))

    if chessboard_graph is not None:
        # 用户输入要选择的节点数量X
        while True:
            try:
                X = int(input("请输入要选择的节点数量X: "))
                if X <= 0:
                    print("请输入正整数！")
                    continue
                break
            except ValueError:
                print("请输入有效数字！")

        # 选择节点
        selected_nodes, edge_count = select_connected_nodes(chessboard_graph, available_nodes, X)

        # 打印结果
        print(f"\n选中的{X}个节点:", selected_nodes)
        print(f"这些节点间的连接数: {edge_count}")
        print(f"平均每个节点的连接数: {edge_count / X:.2f}")

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

        plt.title(f"选中的{X}个相邻节点（红色）")
        plt.show()

        # 保存选中的节点到文件
        with open('selected_nodes.txt', 'w') as f:
            f.write(f"Selected {X} nodes:\n")
            f.write("\n".join(selected_nodes))
            f.write(f"\n\nEdge count between selected nodes: {edge_count}")
        print("\n选中的节点已保存到 selected_nodes.txt")