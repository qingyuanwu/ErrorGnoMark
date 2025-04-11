import pandas as pd
import numpy as np


def create_chessboard_arrays(file_path=r"E:\Repositories\ErrorGnoMark\ScQ-Baihua信息.xlsx"):
    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}")
        return None, None
    except Exception as e:
        print(f"读取Excel文件时出错: {e}")
        return None, None

    # 创建12×13的可用节点矩阵（初始化为全0）
    availability_matrix = np.zeros((12, 13), dtype=int)

    # 处理可用节点（第1列）
    available_nodes = df.iloc[:, 0].dropna().astype(str).tolist()

    for node in available_nodes:
        if node.startswith('Q') and len(node) == 5:
            try:
                row = int(node[1:3]) - 1  # 转换为0-based索引
                col = int(node[3:5]) - 1
                if 0 <= row < 12 and 0 <= col < 13:
                    availability_matrix[row, col] = 1
            except ValueError:
                print(f"警告：忽略无效节点名称 {node}")

    # 打印结果
    if availability_matrix is not None:
        print("可用节点矩阵（12×13）：")
        print(availability_matrix)

        # 保存到文件（可选）
        np.savetxt('availability_matrix.txt', availability_matrix, fmt='%d')
        print("\n矩阵已保存到当前目录下的txt文件中")
