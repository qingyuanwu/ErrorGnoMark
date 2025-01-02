# Standard library imports
import random  # For generating random numbers

# Third-party imports
import numpy as np  # For numerical operations
from qiskit import QuantumCircuit, transpile  # For creating and transpiling quantum circuits
import qiskit  # General Qiskit library


class CliffordGateSet:
    """
    Generates random 1-qubit and 2-qubit Clifford gates based on the backend and compile settings.
    """

    def __init__(self, backend, compile=False):
        """
        Initializes the Clifford gate set generator.

        Parameters:
            backend (str): The backend ('quarkstudio', 'pyquafu', etc.).
            compile (bool): Whether to compile the Clifford gates to the backend's basis gates. Default is False.
        """
        self.backend = backend
        self.compile = compile

    def random_1qubit_clifford(self, qubit_indices):
        """
        Generates random 1-qubit Clifford gates applied to specified qubits.

        Parameters:
            qubit_indices (list): List of qubit indices to apply the gate to.

        Returns:
            QuantumCircuit: The generated Clifford gates as a QuantumCircuit.
        """
        sequences = [
            [],  # I
            ['S'],  # S
            ['S', 'S'],  # S^2 = Z
            ['S', 'S', 'S'],  # S^3 = S†
            ['H'],  # H
            ['H', 'S'],  # HS
            ['H', 'S', 'S'],  # HS^2
            ['H', 'S', 'S', 'S'],  # HS^3 = HS†
            ['S', 'H'],  # SH
            ['S', 'H', 'S'],  # SHS
            ['S', 'H', 'S', 'S'],  # SHS^2
            ['S', 'H', 'S', 'S', 'S'],  # SHS^3 = SHS†
            ['S', 'S', 'H'],  # S^2 H
            ['S', 'S', 'H', 'S'],  # S^2 HS
            ['S', 'S', 'H', 'S', 'S'],  # S^2 HS^2
            ['S', 'S', 'H', 'S', 'S', 'S'],  # S^2 HS^3 = S^2 HS†
            ['S', 'S', 'S', 'H'],  # S^3 H = S† H
            ['S', 'S', 'S', 'H', 'S'],  # S† HS
            ['S', 'S', 'S', 'H', 'S', 'S'],  # S† HS^2
            ['S', 'S', 'S', 'H', 'S', 'S', 'S'],  # S† HS^3 = S† HS†
            ['H', 'S', 'H'],  # HSH
            ['H', 'S', 'H', 'S'],  # HSHS
            ['H', 'S', 'H', 'S', 'S'],  # HSHS^2
            ['H', 'S', 'H', 'S', 'S', 'S'],  # HSHS†
        ]

        qc = QuantumCircuit(max(qubit_indices)+1)
        for i in qubit_indices:
            sequence = random.choice(sequences)
            for gate in sequence:
                if gate == 'H':
                    qc.h(i)
                elif gate == 'S':
                    qc.s(i)
                elif gate == 'S†':
                    qc.sdg(i)
        return qc

    def random_2qubit_clifford(self, qubit_pair):
        """
        Generates a random 2-qubit Clifford gate applied to a specified qubit pair.

        Parameters:
            qubit_pair (tuple): A tuple of two qubit indices to apply the 2-qubit gate to (e.g., (0, 1)).

        Returns:
            QuantumCircuit: The generated 2-qubit Clifford gate as a QuantumCircuit.
        """
        if not qubit_pair or len(qubit_pair) != 2:
            raise ValueError("qubit_pair must be a tuple of two qubit indices.")

        # Define 2-qubit Clifford operations (combinations of 1-qubit Clifford gates and CZ gates)
        qc = QuantumCircuit(max(qubit_pair) + 1)

        # Apply random 1-qubit Clifford gates to each qubit in the pair
        qc = qc.compose(self.random_1qubit_clifford(qubit_pair))
        # qc.compose(self.random_1qubit_clifford([qubit_pair[1]]))

        # Apply a CZ gate
        qc.cz(qubit_pair[0], qubit_pair[1])

        # Apply another layer of random 1-qubit Clifford gates to each qubit in the pair
        qc = qc.compose(self.random_1qubit_clifford(qubit_pair))
        # qc.compose(self.random_1qubit_clifford([qubit_pair[1]]), qubits=[qubit_pair[1]], inplace=True)

        return qc

    def random_single_gate_clifford(self, qubit_indices):
        """
        Generates a random single-gate Clifford operation (H, S, S†) applied to specified qubits.

        Parameters:
            qubit_indices (list): List of qubit indices to apply the gate to.

        Returns:
            QuantumCircuit: The generated Clifford gates as a QuantumCircuit.
        """
        qc = QuantumCircuit(max(qubit_indices)+1)
        for i in qubit_indices:
            single_gates = ['H', 'S', 'S†']
            gate = random.choice(single_gates)
            if gate == 'H':
                qc.h(i)
            elif gate == 'S':
                qc.s(i)
            elif gate == 'S†':
                qc.sdg(i)
        return qc

    def random_pauli(self, qubit_indices):
        """
        Generates random 1-qubit Pauli gates (I, X, Y, Z) applied to specified qubits.

        Parameters:
            qubit_indices (list): List of qubit indices to apply the Pauli gates to.

        Returns:
            QuantumCircuit: The generated Pauli gates as a QuantumCircuit.
        """
        qc = QuantumCircuit(max(qubit_indices)+1)
        paulis = ['X', 'Y', 'Z']
        for i in qubit_indices:
            pauli = random.choice(paulis)
            if pauli == 'X':
                qc.x(i)
            elif pauli == 'Y':
                qc.y(i)
            elif pauli == 'Z':
                qc.z(i)
            # 'I' is the identity gate; no need to append anything
        return qc


# errorgnomark/cirpulse_generator/elements.py

import random
from qiskit.circuit.library import CZGate, RXGate, RYGate, RZGate
from qiskit.circuit import Gate

# 定义可能的旋转角度（弧度）
ROTATION_ANGLES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]

# 定义可用的单量子比特旋转轴
SINGLE_QUBIT_GATES = ['rx', 'ry', 'rz']

# 定义两量子比特门（CZGate）实例
CZ_GATE = CZGate()

def get_random_rotation_gate():
    """
    随机选择一个单量子比特旋转门及其旋转角度。

    返回:
        Gate: 随机选择的旋转门实例。
    """
    gate_type = random.choice(SINGLE_QUBIT_GATES)
    angle = random.choice(ROTATION_ANGLES)
    if gate_type == 'rx':
        gate = RXGate(angle)
    elif gate_type == 'ry':
        gate = RYGate(angle)
    elif gate_type == 'rz':
        gate = RZGate(angle)
    else:
        raise ValueError(f"Unsupported gate type: {gate_type}")
    return gate




class csbq1_circuit_generator:
    def __init__(self, rot_axis='x', rot_angle=np.pi / 2, rep=1):
        """
        初始化 CSB Q1 电路生成器。

        参数:
            rot_axis (str): 旋转轴，可以是 'x'、'y' 或 'z'。
            rot_angle (float): 目标旋转门的旋转角度。
            rep (int): 目标旋转门的重复次数。
        """
        if rot_axis not in ['x', 'y', 'z']:
            raise ValueError("rot_axis 必须是 'x'、'y' 或 'z' 之一。")
        self.rot_axis = rot_axis
        self.rot_angle = rot_angle
        self.rep = rep

    def csbq1_circuit(self, lc, ini_mode, qubit_indices=[0]):
        """
        根据指定的电路长度和初始模式生成 CSB Q1 电路。

        参数:
            lc (int): 电路长度（深度）。
            ini_mode (str): 初始状态模式，可以是 'x'、'y' 或 'z'。
            qubit_indices (list): 作用的量子比特索引列表，可以是任意长度的列表。

        返回:
            QuantumCircuit: 生成的量子电路。
        """
        # 确保电路的量子比特数与 qubit_indices 的长度一致
        num_qubits = len(qubit_indices)
        qc = QuantumCircuit(num_qubits, num_qubits)

        # 初始状态准备
        if ini_mode == 'x':
            for q in qubit_indices:
                qc.h(q)
        elif ini_mode == 'y':
            for q in qubit_indices:
                qc.h(q)
                qc.s(q)
        elif ini_mode == 'z':
            for q in qubit_indices:
                qc.x(q)
        else:
            raise ValueError("ini_mode 必须是 'x'、'y' 或 'z' 之一。")
        qc.barrier()
        
        # 重复应用目标旋转门
        for _ in range(lc * self.rep):
            for q in qubit_indices:
                if self.rot_axis == 'x':
                    qc.rx(self.rot_angle, q)
                elif self.rot_axis == 'y':
                    qc.ry(self.rot_angle, q)
                elif self.rot_axis == 'z':
                    qc.rz(self.rot_angle, q)
        qc.barrier()
        
        # 逆操作以返回计算基态
        if ini_mode == 'x':
            for q in qubit_indices:
                qc.h(q)
        elif ini_mode == 'y':
            for q in qubit_indices:
                qc.sdg(q)
                qc.h(q)
        elif ini_mode == 'z':
            for q in qubit_indices:
                qc.x(q)
        qc.barrier()
        
        # 测量
        qc.measure(qubit_indices, qubit_indices)
        return qc

    def x_direction_csbcircuit_pi_over_2(self, lc, ini_mode, qubit_indices=[0]):
        """
        生成一个 x 方向旋转 π/2 的 CSB 电路。

        参数:
            lc (int): 电路长度（深度）。
            ini_mode (str): 初始状态模式，可以是 'x'、'y' 或 'z'。
            qubit_indices (list): 作用的量子比特索引列表。

        返回:
            QuantumCircuit: 生成的量子电路。
        """
        return self.csbq1_circuit(lc, ini_mode, qubit_indices)

    def generate_csbcircuit_for_gate(self, gate_name, lc, ini_mode, qubit_indices=[0]):
        """
        根据输入的量子门名称生成对应的 CSB 电路。

        参数:
            gate_name (str): 量子门的名称，例如 'XGate', 'YGate', 'ZGate', 'IdGate', 'WGate', 'HGate', 'SGate'。
            lc (int): 电路长度（深度）。
            ini_mode (str): 初始状态模式，可以是 'x'、'y' 或 'z'。
            qubit_indices (list): 作用的量子比特索引列表。

        返回:
            QuantumCircuit: 生成的量子电路。
        """
        gate_mapping = {
            'XGate': {'rot_axis': 'x', 'rot_angle': np.pi},
            'YGate': {'rot_axis': 'y', 'rot_angle': np.pi},
            'ZGate': {'rot_axis': 'z', 'rot_angle': np.pi},
            'IdGate': {'rot_axis': 'x', 'rot_angle': 0},
            'WGate': {'rot_axis': 'x', 'rot_angle': np.pi / 4},  # 假设 WGate 对应 RX(pi/4)
            'HGate': {'rot_axis': 'x', 'rot_angle': np.pi / 2},  # 假设 HGate 对应 RX(pi/2)
            'SGate': {'rot_axis': 'z', 'rot_angle': np.pi / 2}
        }

        if gate_name not in gate_mapping:
            raise ValueError(f"不支持的门名称: {gate_name}")

        rot_axis = gate_mapping[gate_name]['rot_axis']
        rot_angle = gate_mapping[gate_name]['rot_angle']

        # 创建一个新的实例，设置对应的旋转轴和旋转角度
        csb_gen = csbq1_circuit_generator(rot_axis=rot_axis, rot_angle=rot_angle, rep=self.rep)
        return csb_gen.csbq1_circuit(lc, ini_mode, qubit_indices)

    # 可选：为每个门定义独立的方法
    def XGate_csbcircuit(self, lc, ini_mode, qubit_indices=[0]):
        return self.generate_csbcircuit_for_gate('XGate', lc, ini_mode, qubit_indices)

    def YGate_csbcircuit(self, lc, ini_mode, qubit_indices=[0]):
        return self.generate_csbcircuit_for_gate('YGate', lc, ini_mode, qubit_indices)

    def ZGate_csbcircuit(self, lc, ini_mode, qubit_indices=[0]):
        return self.generate_csbcircuit_for_gate('ZGate', lc, ini_mode, qubit_indices)

    def IdGate_csbcircuit(self, lc, ini_mode, qubit_indices=[0]):
        return self.generate_csbcircuit_for_gate('IdGate', lc, ini_mode, qubit_indices)

    def WGate_csbcircuit(self, lc, ini_mode, qubit_indices=[0]):
        return self.generate_csbcircuit_for_gate('WGate', lc, ini_mode, qubit_indices)

    def HGate_csbcircuit(self, lc, ini_mode, qubit_indices=[0]):
        return self.generate_csbcircuit_for_gate('HGate', lc, ini_mode, qubit_indices)

    def SGate_csbcircuit(self, lc, ini_mode, qubit_indices=[0]):
        return self.generate_csbcircuit_for_gate('SGate', lc, ini_mode, qubit_indices)


from qiskit.quantum_info import Operator
from qiskit import transpile

class Csbq2_cz_circuit_generator:
    def __init__(self, theta=np.pi):
        self.theta = theta  # 控制-Z 门的相位参数 theta
        self.eigval_list = [1, np.exp(1j * theta), np.exp(1j * theta), 1]  # 本征值列表
        self.eigvec_list = [
            np.array([1, 0, 0, 0]),  # |00> 本征态
            np.array([0, 0, 0, 1]),  # |11> 本征态
            1 / np.sqrt(2) * np.array([0, 1, 1, 0]),  # (|01> + |10>) 本征态
            1 / np.sqrt(2) * np.array([0, 1, -1, 0])   # (|01> - |10>) 本征态
        ]

    def prepare_initial_state(self, qc, mode, qubit_indices=[0, 1]):
        q0, q1 = qubit_indices  # 解包 qubit_indices 为量子比特索引
        if mode == '01':
            # 准备 (|00> + |11>)/sqrt(2)
            qc.h(q0)
            qc.cx(q0, q1)
        elif mode == '02':
            # 准备 (|01> + |10>)/sqrt(2)
            qc.h(q0)
            qc.cx(q0, q1)
            qc.h(q0)
        elif mode == '03':
            # 准备 (|01> - |10>)/sqrt(2)
            qc.h(q0)
            qc.cx(q0, q1)
            qc.h(q0)
            qc.sdg(q1)
        elif mode == '12':
            # 假设 '12' 对应 (|00> + |11>)/sqrt(2) + (|01> + |10>)/sqrt(2)
            qc.h(q0)
            qc.cx(q0, q1)
            qc.h(q1)
        elif mode == '13':
            # 假设 '13' 对应 (|00> + |11>)/sqrt(2) + (|01> - |10>)/sqrt(2)
            qc.h(q0)
            qc.cx(q0, q1)
            qc.h(q1)
            qc.sdg(q1)
        elif mode == '23':
            # 准备 |01>
            qc.x(q1)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def prepare_inverse_initial_state(self, qc, mode, qubit_indices=[0, 1]):
        q0, q1 = qubit_indices  # 解包 qubit_indices 为量子比特索引
        if mode == '01':
            # 逆操作 (|00> + |11>)/sqrt(2)
            qc.cx(q0, q1)
            qc.h(q0)
        elif mode == '02':
            # 逆操作 (|01> + |10>)/sqrt(2)
            qc.h(q0)
            qc.cx(q0, q1)
            qc.h(q0)
        elif mode == '03':
            # 逆操作 (|01> - |10>)/sqrt(2)
            qc.s(q1)
            qc.h(q0)
            qc.cx(q0, q1)
            qc.h(q0)
        elif mode == '12':
            # 逆操作 (假设对应 '12' 模式的准备)
            qc.h(q1)
            qc.cx(q0, q1)
            qc.h(q0)
        elif mode == '13':
            # 逆操作 (假设对应 '13' 模式的准备)
            qc.s(q1)
            qc.h(q1)
            qc.cx(q0, q1)
            qc.h(q0)
        elif mode == '23':
            # 逆操作 |01>
            qc.x(q1)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def csbq2_cz_circuit(self, len_list, mode='01', nrep=1, qubit_indices=[0, 1]):
        """生成并返回二比特 CZ 电路的列表，量子比特索引可以自定义。"""
        circ_list = []

        # 计算量子比特数量（最大索引 + 1）
        max_qubit = max(qubit_indices) + 1

        # 初始状态准备
        qc_ini = QuantumCircuit(max_qubit, max_qubit)  # 创建一个大小为 max_qubit 的量子电路
        self.prepare_initial_state(qc_ini, mode, qubit_indices)
        qc_ini.barrier()  # 添加障碍

        # 定义 CZ gate
        cz_gate = CZGate()

        # 定义逆准备操作
        qc_ini_inverse = QuantumCircuit(max_qubit, max_qubit)
        self.prepare_inverse_initial_state(qc_ini_inverse, mode, qubit_indices)

        for lc in len_list:
            qc_rep = QuantumCircuit(max_qubit, max_qubit)  # 根据最大量子比特索引创建电路
            for k in range(lc * nrep):
                # 正确使用 append 来添加 CZ 门操作
                qc_rep.append(cz_gate, qubit_indices)  # 添加 CZ 门到量子电路

            # 组合初始准备电路和重复电路
            qc = qc_ini.compose(qc_rep)  # 使用 compose 来连接电路

            # 添加逆操作以恢复初始状态
            qc = qc.compose(qc_ini_inverse)  # 添加 qc_ini_inverse

            # 添加测量电路
            qc.measure(qubit_indices, qubit_indices)  # 使用动态的 qubit_indices

            # 添加调试信息，确保 qc 是 QuantumCircuit
            # print(f"Appending circuit of type: {type(qc)}")  # 应该是 <class 'qiskit.circuit.quantumcircuit.QuantumCircuit'>
            circ_list.append(qc)  # 将电路添加到电路列表

        return circ_list


def permute_qubits(num_qubits):
    """
    Generate a random permutation of qubit indices from 0 to num_qubits - 1.
    """
    rng = np.random.default_rng()
    return list(rng.permutation(num_qubits))

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import random_unitary

def apply_random_su4_layer(qc, num_qubits):
    """
    Apply a random SU4 layer to the quantum circuit for a given number of qubits.
    
    This function:
    1. Generates a random permutation of qubits.
    2. Applies a random SU4 unitary to each adjacent pair of qubits based on the permutation.

    Args:
        qc (QuantumCircuit): The quantum circuit to modify.
        num_qubits (int): The number of qubits in the circuit.

    Returns:
        QuantumCircuit: The modified quantum circuit with SU4 layers applied.
    """
    def permute_qubits(num_qubits):
        """
        Generate a random permutation of qubit indices from 0 to num_qubits - 1.
        """
        rng = np.random.default_rng()
        return list(rng.permutation(num_qubits))
    
    # Step 1: Generate a random permutation of qubits
    permuted_qubits = permute_qubits(num_qubits)
    
    # Step 2: Apply random SU4 to each adjacent pair based on the permutation
    for qubit_idx in range(0, num_qubits, 2):
        if qubit_idx < num_qubits - 1:
            # Select the pair of qubits based on the permutation
            q1 = permuted_qubits[qubit_idx]
            q2 = permuted_qubits[qubit_idx + 1]
            
            # Generate a random SU4 unitary matrix (4x4)
            su4_unitary = random_unitary(4).data  # Qiskit's random_unitary returns a Unitery object; .data gives the matrix
            
            # Apply the unitary matrix to the selected qubits
            qc.unitary(su4_unitary, [q1, q2], label='SU4')
    
    return qc


def qv_circuit_layer(qc, num_qubits):
    """
    Add a random SU4 layer to the quantum circuit.
    """
    permute_qubits(num_qubits)
    apply_random_su4_layer(qc, num_qubits)
    return qc


