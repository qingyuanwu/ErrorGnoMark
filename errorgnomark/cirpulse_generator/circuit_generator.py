import sys
# Add your module path if needed
sys.path.append('/Users/ousiachai/Desktop/ErrorGnoMark') 
# Suppress DeprecationWarnings
from copy import deepcopy
import warnings 
warnings.filterwarnings('ignore', category=DeprecationWarning)
import random
import numpy as np
from qiskit import QuantumCircuit
import copy
from errorgnomark.cirpulse_generator.elements import (
    ROTATION_ANGLES,
    SINGLE_QUBIT_GATES,
    CZ_GATE,
    get_random_rotation_gate,
    csbq1_circuit_generator,
    Csbq2_cz_circuit_generator,
    permute_qubits, 
    apply_random_su4_layer, 
    qv_circuit_layer
)

from errorgnomark.cirpulse_generator.elements import CliffordGateSet
import itertools
from contextlib import contextmanager
import sys
import os

@contextmanager
def DisablePrint():
    """
    A context manager that suppresses all print statements within its block.
    """
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = original_stdout
        sys.stderr = original_stderr




class CircuitGenerator:
    def __init__(self, qubit_select, qubit_connectivity, length_max=12, step_size=2):
        """
        Initializes the Circuit Generator.

        Parameters:
            qubit_select (list): List of selected qubits (e.g., [0, 1, 2]).
            qubit_connectivity (list of tuples): Qubit connections (e.g., [(0, 1), (1, 2)]).
            length_max (int): Maximum length of the circuit.
            step_size (int): Step size for the number of gates applied in the circuit.
        """
        self.qubit_select = qubit_select
        self.qubit_connectivity = qubit_connectivity
        self.length_max = length_max
        self.step_size = step_size
        self.clifford_set = CliffordGateSet(backend='quarkstudio', compile=False)

    def rbq1_circuit(self, ncr=30):
        """
        Generate random 1-qubit Clifford gate circuits.

        Parameters:
            ncr (int): Number of circuits generated for each length.

        Returns:
            list: A nested list with circuits generated for each qubit and length.
                  Structure: circuits_rbq1[qubit_index][length][ncr].
        """
        circuits_rbq1 = []
        total_qubits = max(self.qubit_select) + 1  # Determine the total number of qubits

        # Generate circuits for each qubit index
        for qubit_index in self.qubit_select:
            length_list = range(2, self.length_max + 1, self.step_size)  # Generate lengths from 2 to length_max
            length_circuits = []

            # Generate circuits for each length
            for length in length_list:
                length_circuits_ncr = []
                
                # Generate ncr circuits for the current length
                for _ in range(ncr):
                    # Generate a random 1-qubit Clifford gate for the current qubit
                    random_clifford = self.clifford_set.random_1qubit_clifford([qubit_index])
                    circuit = QuantumCircuit(total_qubits, total_qubits)

                    # Apply Clifford gates in steps based on the circuit length
                    num_steps = max(1, length // self.step_size)
                    for _ in range(num_steps):
                        for gate, qargs, cargs in random_clifford.data:
                            # Apply the random 1-qubit Clifford gates to the specified qubit
                            circuit.append(gate, [qubit_index])

                    # Apply the inverse operation
                    circuit_copy = deepcopy(circuit)
                    circuit_inverse = circuit_copy.inverse()
                    circuit = circuit.compose(circuit_inverse)

                    # Measure the qubit
                    circuit.measure(qubit_index, qubit_index)

                    length_circuits_ncr.append(circuit)
                
                length_circuits.append(length_circuits_ncr)
            circuits_rbq1.append(length_circuits)

        return circuits_rbq1

    def rbq2_circuit(self, ncr=30):
        """
        Generate random 2-qubit Clifford gate circuits for random benchmarking.

        Parameters:
            ncr (int): Number of circuits generated for each length.

        Returns:
            list: A nested list of circuits generated for each qubit pair and each circuit length.
        """
        circuits_rbq2 = []

        # Iterate over each qubit pair in the connectivity list
        for qubit_pair in self.qubit_connectivity:
            length_list = range(2, self.length_max + 1, self.step_size)  # Define lengths for circuits
            length_circuits = []  # To store circuits for each length
            num_qubits = max(qubit_pair) + 1  # Number of qubits in the system

            # Generate circuits for each length
            for length in length_list:
                length_circuits_ncr = []  # To store ncr circuits for this length
                for _ in range(ncr):
                    # Generate a random 2-qubit Clifford gate
                    random_clifford = self.clifford_set.random_2qubit_clifford(qubit_pair)
                    circuit = QuantumCircuit(num_qubits, num_qubits)

                    # Apply the Clifford gates based on the circuit length
                    num_steps = max(1, length // self.step_size)  # Determine number of steps based on length
                    for _ in range(num_steps):
                        # Apply each gate from the random Clifford
                        for gate, qargs, cargs in random_clifford.data:
                            circuit.append(gate, qargs)

                    # Apply the inverse of the entire circuit
                    circuit_copy = deepcopy(circuit)
                    circuit_inverse = circuit_copy.inverse()
                    circuit = circuit.compose(circuit_inverse)

                    # Measure the qubits in the pair
                    circuit.measure(qubit_pair[0], qubit_pair[0])
                    circuit.measure(qubit_pair[1], qubit_pair[1])

                    length_circuits_ncr.append(circuit)  # Add this circuit to the list for this length
                length_circuits.append(length_circuits_ncr)  # Add the list of circuits for this length to the main list

            circuits_rbq2.append(length_circuits)  # Add circuits for this qubit pair to the main list

        return circuits_rbq2



    def xebq1_circuit(self, ncr=30):
        """
        Generate random 1-qubit XEB gate circuits.

        Parameters:
            ncr (int): Number of circuits generated for each length.

        Returns:
            list: A nested list with circuits generated for each qubit and length.
        """
        circuits_xebq1 = []
        total_qubits = max(self.qubit_select) + 1  # Determine the total number of qubits

        # Iterate over selected qubits
        for qubit_index in self.qubit_select:
            # Generate lengths from 1 to length_max in steps of step_size
            length_list = range(1, self.length_max + 1, self.step_size) 
            length_circuits = []

            # Iterate over each circuit length
            for length in length_list:
                length_circuits_ncr = []

                # Generate ncr circuits for each length
                for _ in range(ncr):
                    # Initialize the main circuit for the current qubit
                    circuit = QuantumCircuit(total_qubits, total_qubits)

                    # Apply 'length' number of random rotation gates
                    for _ in range(length):
                        gate = get_random_rotation_gate()  # Random gate generator
                        circuit.append(gate, [qubit_index])  # Apply gate to the selected qubit
                    
                    # Add one final random rotation gate at the end
                    final_gate = get_random_rotation_gate()  # Random gate for final layer
                    circuit.append(final_gate, [qubit_index])

                    # Measure the qubit
                    circuit.measure(qubit_index, qubit_index)

                    length_circuits_ncr.append(circuit)  # Store the generated circuit for this length

                length_circuits.append(length_circuits_ncr)  # Store circuits for all lengths

            circuits_xebq1.append(length_circuits)  # Store circuits for all qubits

        return circuits_xebq1

    def xebq2_circuit(self, ncr=30):
        """
        Generate random 2-qubit XEB (cross-entropy benchmarking) gate circuits.

        Parameters:
            ncr (int): Number of circuits generated for each length.

        Returns:
            list: A nested list with circuits generated for each qubit pair and each circuit length.
        """
        circuits_xebq2 = []  # Initialize the list to hold circuits for all qubit pairs

        # Iterate over each qubit pair defined in self.qubit_connectivity
        for qubit_pair in self.qubit_connectivity:
            length_list = range(1, self.length_max + 1, self.step_size)  # Define the lengths of the circuits
            length_circuits = []  # List to hold circuits for each length
            num_qubits = max(qubit_pair) + 1  # The total number of qubits, determined by the highest qubit index

            # Iterate over each circuit length in the range
            for length in length_list:
                length_circuits_ncr = []  # List to hold 'ncr' circuits for this length

                # Generate 'ncr' circuits for the current length
                for _ in range(ncr):
                    # Initialize the main quantum circuit
                    circuit = QuantumCircuit(num_qubits, num_qubits)

                    # Apply initial random rotation gates to each qubit in the pair
                    for qubit in qubit_pair:
                        gate = get_random_rotation_gate()  # Get a random rotation gate
                        circuit.append(gate, [qubit])

                    # Apply 'length' number of CZ gates, with random rotations in between
                    for _ in range(length):
                        # Apply a CZ gate between the qubit pair
                        circuit.append(CZ_GATE, qubit_pair)

                        # Apply random rotation gates to both qubits after the CZ gate
                        for qubit in qubit_pair:
                            gate = get_random_rotation_gate()  # Get a random rotation gate
                            circuit.append(gate, [qubit])

                    # Apply a final random rotation gate to both qubits
                    for qubit in qubit_pair:
                        final_gate = get_random_rotation_gate()  # Get the final random rotation gate
                        circuit.append(final_gate, [qubit])

                    # Measure both qubits
                    circuit.measure(qubit_pair[0], qubit_pair[0])
                    circuit.measure(qubit_pair[1], qubit_pair[1])

                    # Add this circuit to the list for this length
                    length_circuits_ncr.append(circuit)

                # Add the list of 'ncr' circuits for this length to the overall length circuits list
                length_circuits.append(length_circuits_ncr)

            # Add the circuits for this qubit pair to the final list
            circuits_xebq2.append(length_circuits)

        return circuits_xebq2





    def generate_pi_over_2_x_csb_circuits(self, ini_modes=['x', 'z'], rep=1, qubit_indices=[0]):
        """
        Generate π/2-x direction CSB circuits.

        Parameters:
            ini_modes (list): Initial state modes, e.g., ['x', 'z'].
            rep (int): Number of rotations.
            qubit_indices (list): List of qubit indices where the rotation will be applied.

        Returns:
            list: A nested list containing CSB circuits for each qubit.
        """
        circuits_grouped = []

        # Iterate over the selected qubits
        for qubit in self.qubit_select:
            csb_gen = csbq1_circuit_generator(rot_axis='x', rot_angle=np.pi / 2, rep=rep)  # Create generator
            qubit_circuits = []
            
            # Iterate over different initial modes (e.g., ['x', 'z'])
            for ini_mode in ini_modes:
                # Generate circuits for different lengths (from 0 to max length)
                for lc in range(self.length_max + 1):
                    # Generate the circuit for the current length and initial mode
                    qc = csb_gen.csbq1_circuit(lc=lc, ini_mode=ini_mode, qubit_indices=qubit_indices)
                    qubit_circuits.append(qc)
            
            circuits_grouped.append(qubit_circuits)  # Append the circuits for this qubit

        return circuits_grouped



    def generate_csbcircuit_for_gate(self, gate_name, ini_modes=['x', 'z'], rep=1, qubit_indices=[0]):
        """
        Generate CSB circuits for a specific gate.

        Parameters:
            gate_name (str): Name of the gate (e.g., 'XGate', 'YGate', 'ZGate').
            ini_modes (list): Initial state modes, e.g., ['x', 'y', 'z'].
            rep (int): Number of rotations.
            qubit_indices (list): List of qubit indices where the gate will be applied.

        Returns:
            list: A nested list with CSB circuits for each qubit.
        """
        circuits_grouped = []

        # Iterate over the selected qubits
        for qubit in self.qubit_select:
            # Create a new circuit generator for each gate with appropriate rotation axis and angle
            csb_gen = csbq1_circuit_generator(rot_axis='x', rot_angle=np.pi, rep=rep)
            
            # Create a list for the circuits generated for this qubit
            qubit_circuits = []
            
            # Iterate over the initial states (modes)
            for ini_mode in ini_modes:
                # Generate circuits for different lengths of the circuit (from 0 to max length)
                for lc in range(self.length_max + 1):
                    # Generate the CSB circuit for the specific gate and apply it to the provided qubit indices
                    qc = csb_gen.generate_csbcircuit_for_gate(gate_name, lc=lc, ini_mode=ini_mode, qubit_indices=qubit_indices)
                    qubit_circuits.append(qc)
            print ('qubit_circuits',len(qubit_circuits[0]))
            # Append the generated circuits for this qubit to the main list
            circuits_grouped.append(qubit_circuits)

        return circuits_grouped


    def generate_csbcircuit_for_czgate(self):
        """
        Generates CSB circuits for the CPhase-like CZ gate.

        Returns:
            list: A nested list of QuantumCircuit objects.
                - Outer list corresponds to qubit pairs.
                - Inner lists correspond to different modes.
        """
        # Circuit lengths from 0 to max length
        len_list = list(range(self.length_max + 1))
        
        # Number of repetitions for each circuit (customizable, default is 6 repetitions)
        nrep_list = [1 for _ in range(6)]  # Currently, set to 1 repetition
        
        # Define modes for generating circuits
        mode_list = ['01', '02', '03', '12', '13', '23']
        
        # List to store all generated circuits
        circuits = []

        # Iterate over each qubit pair in the qubit connectivity list
        for qubit_pair in self.qubit_connectivity:
            qubit_pair_circuits = []  # Initialize list for circuits of the current qubit pair

            # Create Csbq2_cz_circuit_generator object (using a CZ gate with theta=π)
            cgen = Csbq2_cz_circuit_generator(theta=np.pi)

            # Generate circuits for each mode
            for mode in mode_list:
                # For each mode, generate circuits and add them to the list for the current qubit pair
                qubit_pair_circuits.extend(
                    cgen.csbq2_cz_circuit(len_list, mode=mode, nrep=1, qubit_indices=qubit_pair)
                )
            
            # Add the generated circuits for the current qubit pair to the outer list
            circuits.append(qubit_pair_circuits)

        return circuits


    def ghz_circuits(self, nqghz_list, ncr):
        """
        Generates GHZ circuits for different numbers of qubits specified in nqghz_list,
        each with a given number of circuits (ncr).

        Parameters:
            nqghz_list (list): A list of integers specifying the number of qubits for each GHZ circuit.
            ncr (int): Number of circuits to generate for each value in nqghz_list.

        Returns:
            list: A nested list containing GHZ circuits. The outer list corresponds to different nqghz values,
                and the inner list corresponds to the ncr circuits generated for each nqghz.
        """
        # Initialize the list to hold the circuits
        ghz_circuits = []

        # Loop through each nqghz value in nqghz_list
        for nqghz in nqghz_list:
            # Ensure nqghz is a positive integer
            if nqghz <= 0:
                raise ValueError(f"nqghz must be a positive integer, but got {nqghz}.")

            # Initialize the list to store circuits for the current nqghz value
            circuit_group = []

            # Loop to generate ncr circuits for the current nqghz value
            for _ in range(ncr):
                # Create a new QuantumCircuit with nqghz qubits and nqghz classical bits
                circuit = QuantumCircuit(nqghz, nqghz)

                # Apply Hadamard gate to the first qubit to create the GHZ state
                circuit.h(0)
                circuit.barrier()

                # Apply CZ gates sequentially between consecutive qubits
                for i in range(1, nqghz):
                    circuit.cx(i - 1, i)
                    circuit.barrier()

                # Measure all qubits
                circuit.measure(range(nqghz), range(nqghz))

                # Check that the generated circuit is a valid QuantumCircuit
                if not isinstance(circuit, QuantumCircuit):
                    raise TypeError("Generated circuit is not a valid QuantumCircuit object.")

                # Add the generated circuit to the group for the current nqghz value
                circuit_group.append(circuit)

            # Add the list of circuits for the current nqghz to the main list
            ghz_circuits.append(circuit_group)

        return ghz_circuits




    # qubit_index = [0,1,2,3,4,5,6,7,8]第二组执行会有问题
    # def ghz_circuits(self, nqubit_ghz, qubit_index, ncr):
    #     """
    #     Generates GHZ circuits based on the available qubit indices, with nested lists for multiple circuits.

    #     Parameters:
    #         nqubit_ghz (int): Number of qubits in each GHZ circuit.
    #         qubit_index (list): List of qubit indices to select from.
    #         ncr (int): Number of GHZ circuits to generate for each set of qubits.

    #     Returns:
    #         list: A nested list containing GHZ circuits, where the outer list corresponds to different qubit sets,
    #             and the inner list corresponds to multiple circuits for each qubit set.
    #     """
        
    #     # Validate the number of qubits and indices
    #     if nqubit_ghz > len(qubit_index):
    #         raise ValueError(
    #             f"nqubit_ghz ({nqubit_ghz}) exceeds the total number of available qubits in qubit_index ({len(qubit_index)})."
    #         )

    #     ghz_circuits = []

    #     # Loop to create GHZ circuits for different qubit index selections
    #     for i in range(0, len(qubit_index), nqubit_ghz):
    #         selected_qubits = qubit_index[i:i + nqubit_ghz]  # Select the next nqubit_ghz qubits from the index list

    #         # Ensure we have enough qubits, and avoid selecting duplicate sets
    #         if len(selected_qubits) < nqubit_ghz:
    #             continue  # Skip if not enough qubits for this iteration
            
    #         # Total qubits needed for the circuit: the max qubit index + 1
    #         total_qubits_in_circuit = max(selected_qubits) + 1

    #         # Create a list to hold circuits for the current qubit selection
    #         circuit_group = []

    #         # Create ncr number of circuits for the current qubit set
    #         for _ in range(ncr):
    #             # Create a new QuantumCircuit instance based on the largest qubit index
    #             circuit = QuantumCircuit(total_qubits_in_circuit, total_qubits_in_circuit)

    #             # Apply Hadamard gate to the first qubit
    #             circuit.h(selected_qubits[0])
    #             circuit.barrier()

    #             # Apply CZ gates sequentially for the rest of the qubits
    #             for j in range(1, nqubit_ghz):
    #                 circuit.cz(selected_qubits[j - 1], selected_qubits[j])
    #                 circuit.barrier()

    #             # Measure the qubits
    #             circuit.measure(selected_qubits, selected_qubits)

    #             # Ensure the generated circuit is of type QuantumCircuit
    #             if not isinstance(circuit, QuantumCircuit):
    #                 raise TypeError("Generated circuit is not a valid QuantumCircuit object.")

    #             # Add the circuit to the current group
    #             circuit_group.append(circuit)

    #         # Add the group of circuits to the main list
    #         ghz_circuits.append(circuit_group)

    #     return ghz_circuits



    def stanqvqm_circuit(self, ncr=30,nqubits_max=16):
        """
        Generate Quantum Volume (QV) circuits for each qubit count from 1 to nqubits_max.

        Parameters:
            ncr (int): Number of circuits to generate for each qubit count.
            nqubits_max (int): Maximum number of qubits for the quantum circuits.

        Returns:
            dict: Dictionary containing qubit counts and corresponding circuits.
                Format:
                {
                    "nqubits_<i>": {
                        "total_qubits": int,
                        "circuits": [QuantumCircuit, ...]  # List of circuits
                    }
                }
        """
        # nqubits_max= max(self.qubit_select)
        all_circuits = []

        # Generate circuits for each number of qubits from 1 to nqubits_max
        qvqubit_list = np.arange(2, nqubits_max + 1)
        for nq in  qvqubit_list:
            qc_ncr = []  # List to hold ncr circuits for the current qubit configuration

            # Generate ncr circuits for the current qubit configuration
            for _ in range(ncr):
                qc = QuantumCircuit(nq, nq)  # Initialize quantum circuit with nq qubits

                # Add QV layers to the circuit
                for _ in range(nq):  # Depth of the circuit is equal to the number of qubits
                    qv_circuit_layer(qc, nq)  # Add a random QV layer
                qc.measure_all()

                qc_ncr.append(qc)  # Append the circuit to the list

            # Store the circuits for the current qubit configuration
            all_circuits.append(qc_ncr)  # List of circuits for the current qubit configuration

        return all_circuits


    def mrbqm_circuit(self, density_cz=0.75, ncr=30):
        """
        Generates quantum circuits based on the provided CZ gate density and number of circuits.

        Parameters:
            density_cz (float): Density of CZ gates in the circuit (0 < density_cz <= 1).
            ncr (int): Number of circuits to generate for each length in length_list.

        Returns:
            list of list of list of QuantumCircuit: Generated quantum circuits organized as 
                circuits[num_qubits][length][ncr].
        """
        # Validate the density_cz parameter
        if not (0 < density_cz <= 1):
            raise ValueError("density_cz must be within the range (0, 1].")

        # Generate the list of circuit lengths based on step_size and length_max
        length_list = list(range(self.step_size, self.length_max + 1, self.step_size))

        # Initialize the nested circuits list
        circuits = []  # Structure: circuits[num_qubits][length][ncr]

        # Sort the selected qubits to ensure consistent ordering
        sorted_qubits = sorted(self.qubit_select)

        # Outer loop: Iterate over the number of qubits, starting from 2 and increasing by 2
        for num_qubits in range(2, len(sorted_qubits) + 1, 2):
            current_qubits = sorted_qubits[:num_qubits]
            n_qubits = max(current_qubits) + 1  # Determine the number of qubits for the current circuit

            # Filter qubit connectivity to include only pairs within the current qubits
            current_connectivity = [
                pair for pair in self.qubit_connectivity
                if pair[0] in current_qubits and pair[1] in current_qubits
            ]

            # Initialize the list for the current number of qubits
            circuits_per_qubit = []

            # Middle loop: Iterate over each circuit length
            for length in length_list:
                # Calculate the maximum number of CZ gates for the current length and qubit count
                nmax_czgate = int(density_cz * np.floor((length * num_qubits) / 2))

                # Initialize the list for the current length
                circuits_per_length = []

                # Inner loop: Generate ncr circuits for the current qubit count and length
                for _ in range(ncr):
                    # Initialize the quantum circuit with qubits and classical bits
                    qc = QuantumCircuit(n_qubits, n_qubits)

                    # Step 1: Apply initial random single-qubit Clifford gates
                    clifford_qc = self.clifford_set.random_single_gate_clifford(current_qubits)
                    qc = qc.compose(clifford_qc)
                    qc.barrier()

                    # Initialize lists to store applied Pauli and CZ layers for inversion
                    applied_paulis = []
                    applied_czs = []

                    # Step 2: Generate each layer of the circuit
                    for layer in range(length):
                        # Step 2a: Apply random Pauli gates to all qubits
                        pauli_layer = self.clifford_set.random_pauli(current_qubits)
                        qc = qc.compose(pauli_layer)
                        qc.barrier()
                        applied_paulis.append(pauli_layer)

                        # Step 2b: Decide whether to apply a CZ layer based on density_cz
                        if random.random() <= density_cz:
                            # Step 2b1: Apply a full CZ layer, ensuring all qubits are covered
                            qubits_shuffled = copy.deepcopy(current_qubits)
                            random.shuffle(qubits_shuffled)
                            cz_pairs = [(qubits_shuffled[i], qubits_shuffled[i + 1]) for i in range(0, num_qubits, 2)]

                            # Step 2b2: Apply CZ gates to the selected pairs
                            for pair in cz_pairs:
                                qc.cz(pair[0], pair[1])
                            
                            # Store the CZ pairs for inversion
                            applied_czs.append(cz_pairs)
                        else:
                            # No CZ layer applied in this layer
                            applied_czs.append([])

                    qc.barrier()

                    # Step 3: Apply central random Pauli gates to all qubits
                    central_pauli_layer = self.clifford_set.random_pauli(current_qubits)
                    qc = qc.compose(central_pauli_layer)
                    qc.barrier()

                    # Step 4: Invert the middle layers by re-applying CZ and Pauli gates in reverse order
                    for layer in reversed(range(length)):
                        # Step 4a: Re-apply CZ gates if they were applied in the forward pass
                        cz_pairs = applied_czs[layer]
                        if cz_pairs:
                            for pair in cz_pairs:
                                qc.cz(pair[0], pair[1])
                        qc.barrier()

                        # Step 4b: Re-apply the same Pauli gates
                        pauli_qc = applied_paulis[layer]
                        qc = qc.compose(pauli_qc)
                        qc.barrier()

                    # Step 5: Apply the inverse of the initial Clifford gates in reverse order
                    qc = qc.compose(clifford_qc.inverse())
                    qc.barrier()

                    # Step 6: Measure all qubits
                    qc.measure(current_qubits, current_qubits)

                    # Add the generated circuit to the current length list
                    circuits_per_length.append(qc)

                # Add the current length's circuits to the current qubit count's list
                circuits_per_qubit.append(circuits_per_length)

            # Add the current qubit count's circuits to the main circuits list
            circuits.append(circuits_per_qubit)

        return circuits


    def clopsqm_circuit(self, num_templates=50, num_updates=10, num_qubits=5):
        """
        Generates CLOPSQM circuits by applying random SU4 layers.

        Args:
            num_templates (int): Number of templates (M), usually 100.
            num_updates (int): Number of parameter updates (K), usually 10.
            num_qubits (int, optional): Number of qubits. Defaults to 5.

        Returns:
            list: Nested list of QuantumCircuit objects organized by layer depth, templates, and updates.
                Structure: [
                            [  # Layer 1
                                [qc1_template1_update1, qc1_template1_update2, ...],  # Template 1
                                [qc1_template2_update1, qc1_template2_update2, ...],  # Template 2
                                ...
                            ],
                            [  # Layer 2
                                [qc2_template1_update1, qc2_template1_update2, ...],  # Template 1
                                [qc2_template2_update1, qc2_template2_update2, ...],  # Template 2
                                ...
                            ],
                            ...
                        ]
        """
        # List of circuit lengths based on step size and maximum length
        length_list = list(range(self.step_size, self.length_max + 1, self.step_size))

        with DisablePrint():
            all_circuits = []  # List to store all the circuits for different layers

            # Loop through each layer depth
            for layer_depth in length_list:
                layer_circuits = []  # List to hold circuits for the current layer
                
                # Generate circuits for each template
                for _ in range(num_templates):
                    template_circuits = []  # List to hold circuits for the current template
                    
                    # Generate circuits for each update
                    for _ in range(num_updates):
                        # Initialize a quantum circuit with the specified number of qubits
                        qc = QuantumCircuit(num_qubits)
                        
                        # Apply random SU4 layers for the specified layer depth
                        for _ in range(layer_depth):
                            qv_circuit_layer(qc, num_qubits)  # Add a random SU4 layer to the circuit
                            qc.measure_all()  # Measure all qubits after each layer
                        
                        # Append the generated circuit to the template list
                        template_circuits.append(qc)
                    
                    # Append the template list for the current layer to the layer_circuits list
                    layer_circuits.append(template_circuits)
                
                # Append the layer_circuits list for the current layer to the all_circuits list
                all_circuits.append(layer_circuits)
            
            return all_circuits  # Return the list of generated circuits for all layers





# if __name__ == "__main__":
#     import numpy as np
#     from qiskit.circuit.library import CZGate, RXGate, RYGate, RZGate

#     # 定义 CZ_GATE
#     CZ_GATE = CZGate()

#     # 定义 get_random_rotation_gate 函数
#     def get_random_rotation_gate():
#         gate_class = random.choice([RXGate, RYGate, RZGate])
#         angle = random.uniform(0, 2 * np.pi)
#         return gate_class(angle)

#     # 初始化 CircuitGenerator，假设 backend 为 'quarkstudio'
#     qubit_select = [0, 1, 2, 3]  # 示例量子比特选择
#     qubit_connectivity = [(0, 1), (1, 2), (2, 3)]  # 示例量子比特连接
#     circuit_gen = CircuitGenerator(
#         qubit_select=qubit_select,
#         qubit_connectivity=qubit_connectivity,
#         length_max=10,
#         step_size=2
#     )

#     # 生成 mrbqm 电路，假设 density_cz=0.5, ncr=2
#     generated_mrbqm_circuits = circuit_gen.mrbqm_circuit(density_cz=0.5, ncr=2)

#     # # 打印生成的 mrbqm 电路
#     # print("=== MRBQM Circuits ===")
#     # for idx, qc in enumerate(generated_mrbqm_circuits, start=1):
#     #     print(f"Circuit {idx}:")
#     #     print(qc.draw(output='text'))
#     #     print("\n" + "="*50 + "\n")
# print (' generated_mrbqm_circuits', generated_mrbqm_circuits[1][0][0])


# if __name__ == "__main__":
#     # 定义 qubits 和它们的连接
#     qubit_select = [0, 1, 2, 3]
#     qubit_connectivity = [(0, 1), (1, 2), (2, 3), (0, 3)]

#     # 初始化 CircuitGenerator，假设 backend 为 'default'
#     circuit_gen = CircuitGenerator(
#         qubit_select=qubit_select,
#         qubit_connectivity=qubit_connectivity,
#         length_max=10,
#         step_size=2
#     )

#     # 生成电路，假设 density_cz=0.5，生成每个长度 5 个电路
#     generated_circuits = circuit_gen.ghz_circuits(4,3)

#     # # 打印生成的电路
#     # for idx, qc in enumerate(generated_circuits):
#     #     print(f"电路 {idx+1}:")
#     #     print(qc)

# print (generated_circuits[2])



# # 创建 CircuitGenerator 对象
# generator = CircuitGenerator(
#     qubit_select=[0, 1], 
#     qubit_connectivity=[(0, 1), (1, 2)],  # 量子比特连接对
#     length_max=20
# )

# # 生成 CZ gate 的 CSB 电路
# circuits = generator.generate_csbcircuit_for_gate('XGate')


# print ('csb-circuits', circuits[0][4])





# circuit_gen = CircuitGenerator(
#         qubit_select=[0, 1, 2, 3],
#         qubit_connectivity=[[0, 1],[1,2],[2,3]]
#     )

# # 生成 RBQ1 和 RBQ2 电路
# # rbq1_circuits = circuit_gen.rbq1_circuit()
# rbq2_circuits = circuit_gen.rbq2_circuit()

# # 打印第一个生成的 rbq2 电路
# # print('rbq1_circuits', rbq1_circuits[1][1][1])
# print('rbq2_circuits',rbq2_circuits[1][1][1])


# test_circuit_generator.py



# # 定义您的量子比特选择和连接
# qubit_select = [0, 1, 2, 3]
# qubit_connectivity = [(0, 1),(1, 2)]
# length_max = 4  # 设置较小的长度以便快速测试
# step_size = 1
# ncr = 3  # 每个长度生成一个电路

# # 初始化 CircuitGenerator
# circuit_gen = CircuitGenerator(qubit_select, qubit_connectivity, length_max, step_size)

# # 生成 XEB 1-qubit 电路
# xeb_q1 = circuit_gen.xebq1_circuit(ncr)

# # 生成 XEB 2-qubit 电路
# xeb_q2 = circuit_gen.xebq2_circuit(ncr)


# print ("xeb_q1", xeb_q1[0][1][0])
# print ("xeb_q2", xeb_q2[1][1][0])
# # Now xeb_q1 and xeb_q2 contain the generated circuits


    # def csbq1_circuit(self, ncr):
    #     return circuits_csbq1


    # def csbq2_circuit(self, ncr):
    #     return circuits_csbq2


    # def xebq1_circuit(self, ncr):
    #     return circuits_xebq1

    # def xebq2_circuit(self, ncr):
    #     return circuits_xebq2

    # def xebqm_circuit(self, ncr):
    #     return circuits_xebqm
    
    # def standard_qvqm_circuit(self, ncr):
    #     return circuits_standard_qvqm

    # def ghzqm_circuit(self, ncr):
    #     return circuits_ghzqm
    
    # def clopsqvqm_circuit(self, ncr):
    #     return circuits_clopsqvqm_

    # def mrbqm_circuit(self, ncr):
    #     return circuits_mrbqm
    
    # def xtalkqm_circuit(self, ncr):
    #     return xtalkqm
    



    