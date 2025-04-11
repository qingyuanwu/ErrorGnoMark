import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import rcParams
from Chip_topology_matrix_construction import create_chessboard_arrays
from Qubit_selection_scorer import select_connected_nodes
'''
Baihua量子计算系统棋盘格拓扑结构构建工具
并依据拓扑结构选择合适的量子点

功能描述：
1. 从Baihua量子计算系统的数据视图Excel文件中读取棋盘格拓扑信息
2. 构建包含156个节点(12×13棋盘格)的量子比特连接关系图
3. 可视化量子比特的可用性状态和连接关系

输入数据说明：
- 输入文件：ScQ-Baihua信息.xlsx
- 数据列：
  * 第1列(A列)：标记可用的量子比特节点(QXXYY格式)
  * 第9列(I列)：记录量子比特间的连接关系(QXXYY_QXXYY格式)

输出结果：
1. 可视化图形：
  - 蓝色节点：可用的量子比特
  - 黑色节点：不可用的量子比特
  - 灰色边线：量子比特间的连接关系
2. 控制台输出：拓扑结构的基本统计信息

数据来源：
Baihua量子计算系统数据视图
https://quafu.baqis.ac.cn/#/computerDetail/dataView?system_id=9&type=Baihua

版本要求：
- Python 3.6+
- 依赖库：pandas, networkx, matplotlib

使用说明：
1. 确保Excel文件路径正确(E:\Repositories\ErrorGnoMark\ScQ-Baihua信息.xlsx)
2. 运行脚本将自动生成可视化结果和统计信息
3. 如需修改可视化样式，可调整visualize_chessboard()函数参数

注意事项：
1. 需要系统中安装中文字体(如SimHei)以正确显示中文标签
2. Excel文件应保持约定的数据格式和列位置
'''


def build_chessboard_graph(chip_row, chip_col, file_path=r"E:\Repositories\ErrorGnoMark\ScQ-Baihua信息.xlsx"):
    # 设置中文字体
    rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei']  # 指定常用中文字体
    rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 读取Excel文件
    # file_path = r"E:\Repositories\ErrorGnoMark\ScQ-Baihua信息.xlsx"
    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}")
        return None, None
    except Exception as e:
        print(f"读取Excel文件时出错: {e}")
        return None, None

    # 创建棋盘格图
    G = nx.Graph()

    # 添加所有节点 (QXXYY格式)
    for row in range(1, chip_row + 1):
        for col in range(1, chip_col + 1):
            node_name = f"Q{row:02d}{col:02d}"
            G.add_node(node_name, pos=(col, -row))  # 使用负值使棋盘从上到下编号

    # 处理连接关系 (假设在第9列，索引为8)
    if len(df.columns) < 9:
        print("错误：Excel文件没有足够的列")
        return None, None

    connections = df.iloc[:, 8].dropna()  # 获取第9列并去除空值

    for connection in connections:
        if '_' in connection:
            node1, node2 = connection.split('_')
            if node1 in G.nodes and node2 in G.nodes:
                G.add_edge(node1, node2)
            else:
                print(f"警告：连接关系 {connection} 包含不存在的节点")

    # 获取可用节点列表 (第1列)
    available_nodes = df.iloc[:, 0].dropna().astype(str).tolist()

    return G, available_nodes


def visualize_chessboard(G, available_nodes):
    if G is None:
        return

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei']
    plt.rcParams['axes.unicode_minus'] = False

    # 获取节点位置
    pos = nx.get_node_attributes(G, 'pos')

    # 将节点分为可用和不可用两类
    available = [node for node in G.nodes if node in available_nodes]
    unavailable = [node for node in G.nodes if node not in available_nodes]

    # 绘制图形
    plt.figure(figsize=(15, 10))

    # 绘制可用节点（蓝色）
    nx.draw_networkx_nodes(G, pos, nodelist=available, node_color='lightblue', node_size=300)

    # 绘制不可用节点（黑色）
    nx.draw_networkx_nodes(G, pos, nodelist=unavailable, node_color='black', node_size=300)

    # 绘制所有边
    nx.draw_networkx_edges(G, pos, edge_color='gray')

    # 绘制标签（只显示可用节点的标签）
    labels = {node: node for node in available}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)

    plt.title("棋盘格连接关系图（黑色表示不可用点）")
    plt.tight_layout()  # 自动调整布局
    plt.show()


def node_name_to_index(node_name):
    """将节点名称QXXYY转换为序号(XX*13 + YY)"""
    row = int(node_name[1:3])  # 提取XX部分
    col = int(node_name[3:5])  # 提取YY部分
    return (row - 1) * 13 + col - 1  # Baihua从0开始


def save_connections(chessboard_graph, filename='connections.txt'):
    """保存所有连接关系到文件，格式为[序号1, 序号2]"""
    connections = []
    for edge in chessboard_graph.edges:
        node1, node2 = edge
        idx1 = node_name_to_index(node1)
        idx2 = node_name_to_index(node2)
        connections.append([idx1, idx2])

    # 按照第一个序号排序
    connections.sort(key=lambda x: (x[0], x[1]))

    # 保存到文件
    with open(filename, 'w') as f:
        for conn in connections:
            f.write(f"{conn}\n")

    return connections


if __name__ == '__main__':

    # 构建棋盘格图
    chessboard_graph, available_nodes = (
        build_chessboard_graph(chip_row=12, chip_col=13,
                               file_path="E:\Repositories\ErrorGnoMark\ScQ-Baihua信息.xlsx"))

    if chessboard_graph is not None:
        # ======================可视化棋盘格======================
        # 同时保存txt文件
        if 0:
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

        # 根据节点得分，选择节点
        selected_nodes, edge_count = select_connected_nodes(chessboard_graph, available_nodes, X)

        # 打印结果
        print(f"\n选中的{X}个节点:", selected_nodes)
        print(f"这些节点间的连接数: {edge_count}")
        print(f"平均每个节点的连接数: {edge_count / X:.2f}")

        if 0:
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
            f.write(f"Selected {X} nodes:\n")
            f.write("\n".join(selected_nodes))
            f.write(f"\n\nEdge count between selected nodes: {edge_count}")
            f.write(f"\n\n节点序号(排序后): {node_indices}")
        print("\n选中的节点已保存到 selected_nodes.txt")


