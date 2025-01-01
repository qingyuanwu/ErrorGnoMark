import warnings
import time
import re
import random
from requests.exceptions import RequestException, ReadTimeout

import numpy as np
from qiskit import QuantumCircuit, qasm2, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    QuantumError,
    amplitude_damping_error,
    phase_damping_error,
    depolarizing_error
)
from qiskit.circuit.library import (
    IGate, XGate, YGate, ZGate, RXGate, RYGate, RZGate,
    HGate, CXGate, CZGate, SwapGate
)
from qiskit_aer.noise.errors.quantum_error import NoiseError

from errorgnomark.fake_data import generate_fake_data_rbq1, generate_fake_data_rbq2
from quark import Task


# Suppress unnecessary warnings related to multiple results
warnings.filterwarnings(
    "ignore",
    message=r'Result object contained multiple results matching name "circuit-\d+", only first match will be returned.'
)


def build_custom_noise_model():
    """
    Constructs a custom noise model including amplitude damping and phase damping errors.
    This noise model is adjusted to achieve ~10e-3 error for 1-qubit gates and ~10e-2 error for 2-qubit gates.
    The noise model is now applied to arbitrary qubit indices in any quantum circuit.
    """
    noise_model = NoiseModel()
    
    # Define error probabilities (adjusted for target error rates)
    p_amp_1q = 0.1   # Amplitude damping probability for 1-qubit gates
    p_phase_1q = 0.005  # Phase damping probability for 1-qubit gates
    p_identity_1q = 1.0 - p_amp_1q - p_phase_1q  # No-error probability for 1-qubit gates
    
    p_amp_2q = 0.1    # Amplitude damping probability for 2-qubit gates
    p_phase_2q = 0.001  # Phase damping probability for 2-qubit gates
    p_identity_2q = 1.0 - p_amp_2q - p_phase_2q  # No-error probability for 2-qubit gates

    # Validate probabilities
    if p_identity_1q < 0 or p_identity_2q < 0:
        raise ValueError("The sum of p_amp and p_phase should be <= 1 for both 1-qubit and 2-qubit gates.")
    
    # Define single-qubit gates
    single_qubit_gates = ["h", "x", "y", "z", "rx", "ry", "rz"]
    
    # Apply noise to 1-qubit gates (for arbitrary qubit indices)
    for gate in single_qubit_gates:
        identity = QuantumCircuit(1)
        identity.id(0)
        
        # Amplitude damping error circuits and probabilities
        amp_error = amplitude_damping_error(p_amp_1q)
        amp_circs = amp_error.circuits
        amp_probs = amp_error.probabilities
        
        # Phase damping error circuits and probabilities
        phase_error = phase_damping_error(p_phase_1q)
        phase_circs = phase_error.circuits
        phase_probs = phase_error.probabilities
        
        # Build the list of noise operations for QuantumError
        noise_ops = []
        noise_ops.append((identity, p_identity_1q))
        
        # Add amplitude damping error circuits
        for circ, prob in zip(amp_circs, amp_probs):
            noise_ops.append((circ, p_amp_1q * prob))
        
        # Add phase damping error circuits
        for circ, prob in zip(phase_circs, phase_probs):
            noise_ops.append((circ, p_phase_1q * prob))
        
        # Ensure the total probability sums to 1
        total_prob = sum(prob for _, prob in noise_ops)
        if not np.isclose(total_prob, 1.0, atol=1e-6):
            raise ValueError(f"The total noise probability for gate '{gate}' is {total_prob}, which does not equal 1.")
        
        # Create QuantumError object
        error = QuantumError(noise_ops)
        
        # Add QuantumError to the noise model for each qubit
        # Add the error to all qubits dynamically (not just qubit 0)
        noise_model.add_all_qubit_quantum_error(error, gate)
    
    # Define two-qubit gates
    two_qubit_gates = ["cz"]
    
    # Apply noise to 2-qubit gates (for arbitrary qubit indices)
    for gate in two_qubit_gates:
        identity = QuantumCircuit(2)
        identity.id(0)
        identity.id(1)
        
        # Amplitude damping errors on both qubits
        amp_error = amplitude_damping_error(p_amp_2q).tensor(amplitude_damping_error(p_amp_2q))
        amp_circs = amp_error.circuits
        amp_probs = amp_error.probabilities
        
        # Phase damping errors on both qubits
        phase_error = phase_damping_error(p_phase_2q).tensor(phase_damping_error(p_phase_2q))
        phase_circs = phase_error.circuits
        phase_probs = phase_error.probabilities
        
        # Build the list of noise operations for QuantumError
        noise_ops = []
        noise_ops.append((identity, p_identity_2q))
        
        # Add amplitude damping error circuits
        for circ, prob in zip(amp_circs, amp_probs):
            noise_ops.append((circ, p_amp_2q * prob))
        
        # Add phase damping error circuits
        for circ, prob in zip(phase_circs, phase_probs):
            noise_ops.append((circ, p_phase_2q * prob))
        
        # Ensure the total probability sums to 1
        total_prob = sum(prob for _, prob in noise_ops)
        if not np.isclose(total_prob, 1.0, atol=1e-6):
            raise ValueError(f"The total noise probability for gate '{gate}' is {total_prob}, which does not equal 1.")
        
        # Create QuantumError object
        error = QuantumError(noise_ops)
        
        # Add QuantumError to the noise model for the two-qubit gate
        # Add the error to all qubits dynamically (not just qubits 0,1)
        noise_model.add_all_qubit_quantum_error(error, gate)
    
    # Add depolarizing noise to other two-qubit gates (such as "cx" and "swap")
    additional_two_qubit_gates = ["cx", "swap"]
    for gate in additional_two_qubit_gates:
        depol_2q = depolarizing_error(0.005, 2)  # Small depolarizing error
        # Add the depolarizing error to all qubits dynamically
        noise_model.add_all_qubit_quantum_error(depol_2q, gate)
    
    return noise_model



class QuantumJobRunner:
    def __init__(self, circuits):
        """
        Initializes the Quantum Job Runner.

        Parameters:
            circuits (list): A list of QuantumCircuit objects or OpenQASM strings.
                             Each element represents a single circuit execution.
        """
        self.circuits = circuits

    def validate_token(self, token):
        """
        Validates the provided QuarkStudio token.

        Parameters:
            token (str): The QuarkStudio token to validate.

        Returns:
            bool: True if the token is valid, False otherwise.
        """
        token_pattern = re.compile(r"^[\w\-\:\{\/}]+$")
        return bool(token_pattern.match(token))

    def qiskit_to_openqasm(self, circuit):
        """
        Converts a Qiskit QuantumCircuit to OpenQASM format.

        Parameters:
            circuit (QuantumCircuit): The Qiskit quantum circuit to convert.

        Returns:
            str: The circuit in OpenQASM format.
        """
        if not isinstance(circuit, QuantumCircuit):
            raise ValueError("Provided circuit is not a Qiskit QuantumCircuit.")
        return qasm2.dumps(circuit)

    def select_best_chip(self, tmgr):
        """
        Selects the chip with the minimum queue length.

        Parameters:
            tmgr (Task): The Task manager instance for QuarkStudio.

        Returns:
            str: The name of the selected chip.
        """
        status_info = tmgr.status()
        available_chips = {chip: queue for chip, queue in status_info.items() if isinstance(queue, int)}
        if not available_chips:
            raise ValueError("No available chips found.")
        return min(available_chips, key=available_chips.get)

    def quarkstudio_run(
        self,
        compile=True,
        shots=1,
        print_progress=True,
        use_fake_data=None,
        delay_between_tasks=2,
        max_retries=3,
        elapsed_time=False
    ):
        """
        Runs quantum circuits either on real hardware or generates fake data.

        Parameters:
            compile (bool): Whether to transpile the circuit to native gate sets. Default is True.
            shots (int): Number of shots per circuit. Default is 1.
            print_progress (bool): Whether to print progress updates. Default is False.
            use_fake_data (str or None): 
                None: Execute on real hardware.
                'fake_dataq1': 1-qubit fake data.
                'fake_dataq2': 2-qubit fake data.
            delay_between_tasks (int): Seconds to wait after completing a task before submitting the next one. Default is 2 seconds.
            max_retries (int): Maximum number of retries for submitting a task. Default is 3.
            elapsed_time (bool): Whether to return elapsed times along with execution results. Default is False.

        Returns:
            list or tuple:
                - If elapsed_time=False: List of measurement count dictionaries or fake data structure.
                - If elapsed_time=True: Tuple containing the list of measurement counts and the list of elapsed times.
        """

        if use_fake_data not in [None, 'fake_dataq1', 'fake_dataq2']:
            raise ValueError("Invalid use_fake_data value. Must be one of [None, 'fake_dataq1', 'fake_dataq2'].")

        if use_fake_data:
            if use_fake_data == 'fake_dataq1':
                return generate_fake_data_rbq1()
            elif use_fake_data == 'fake_dataq2':
                return generate_fake_data_rbq2()

        token = "5vtENo5IEGViJNv:nmgYuZ:ehMobWzUd6qcu7pMeSZW/Rg{dUPyBkO{5DO{BEP4VkO{dUN7JDd5WnJtJDOyp{O1pEOyBjNy1jNy1DOzBkNjpkJ1GXbjxjJvOnMkGnM{mXdiKHRliYbii3ZjpkJzW3d2Kzf"
        tmgr = Task(token)
        backend = self.select_best_chip(tmgr)

        task_results = []
        elapsed_times = []

        for idx, circuit in enumerate(self.circuits):
            # if print_progress:
            #     print(f"Running circuit {idx+1}/{len(self.circuits)}")

            openqasm_circuit = self.qiskit_to_openqasm(circuit) if isinstance(circuit, QuantumCircuit) else circuit

            task = {
                'chip': backend,
                'name': 'QuantumTask',
                'circuit': openqasm_circuit,
                'compile': compile,
                'correct': False
            }

            attempt = 0
            while attempt < max_retries:
                try:
                    tid = tmgr.run(task, repeat=shots)
                    break
                except RequestException:
                    attempt += 1
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
            else:
                task_results.append({})
                if elapsed_time:
                    elapsed_times.append(0.0)
                continue

            while True:
                try:
                    time.sleep(10)
                    res = tmgr.result(tid)

                    if res and 'status' in res:
                        status = res['status'].lower()
                        if status == 'finished':
                            counts = res.get('count', {})
                            task_results.append(counts)

                            if elapsed_time:
                                task_elapsed_time = res.get('elapsed_time', 0.0)
                                elapsed_times.append(task_elapsed_time)
                            break
                except (ReadTimeout, RequestException):
                    time.sleep(5)

            if delay_between_tasks > 0:
                time.sleep(delay_between_tasks)

        if elapsed_time:
            return task_results, np.mean(elapsed_times)
        else:
            return task_results


    def simulation_ideal_qiskit(
        self,
        compile=True,
        shots=4096,
        print_progress=False,
        noise_model=None,
        elapsed_time=False
    ):
        """
        Runs quantum circuits using Qiskit's Aer simulator and returns measurement results mapped to the original qubits.

        Parameters:
            compile (bool): Whether to transpile circuits to the local gate set. Default is True.
            shots (int): Number of measurement shots per circuit. Default is 1024.
            print_progress (bool): Whether to print progress updates. Default is False.
            noise_model (None or bool): If None, runs ideal simulation. If True, uses a custom noise model with amplitude and phase damping errors.
            elapsed_time (bool): Whether to record the execution time for each circuit.

        Returns:
            If elapsed_time=False:
                list: A list of dictionaries containing measurement results for each circuit, with bitstrings corresponding only to the active qubits.
            If elapsed_time=True:
                tuple: (results, times)
                    - results: A list of dictionaries containing measurement results for each circuit.
                    - times: A list of execution times for each circuit.
        """
        def get_active_qubits_and_cbits(circuit):
            active_qubits, active_cbits = set(), set()
            for instruction, qargs, cargs in circuit.data:
                if instruction.name != 'barrier':  # Ignore barriers
                    active_qubits.update(circuit.qubits.index(qbit) for qbit in qargs)
                if instruction.name == 'measure':
                    active_cbits.update(circuit.clbits.index(cbit) for cbit in cargs)
            total_cbits = len(circuit.clbits)
            return sorted(active_qubits), sorted(active_cbits), total_cbits

        def map_circuit(circuit, active_qubits, active_cbits):
            new_nqubits = len(active_qubits)
            new_ncbits = len(active_cbits)
            new_circuit = QuantumCircuit(new_nqubits, new_ncbits)

            qubit_mapping = {old: new for new, old in enumerate(active_qubits)}
            cbit_mapping = {old: new for new, old in enumerate(active_cbits)}

            for instruction, qargs, cargs in circuit.data:
                if instruction.name == 'measure':
                    new_qargs = [
                        new_circuit.qubits[qubit_mapping[circuit.qubits.index(qbit)]]
                        for qbit in qargs
                    ]
                    new_cargs = [
                        new_circuit.clbits[cbit_mapping[circuit.clbits.index(cbit)]]
                        for cbit in cargs
                    ]
                    new_circuit.append(instruction, new_qargs, new_cargs)
                else:
                    new_qargs = [
                        new_circuit.qubits[qubit_mapping[circuit.qubits.index(qbit)]]
                        for qbit in qargs
                    ]
                    new_circuit.append(instruction, new_qargs, [])
            return new_circuit, qubit_mapping, cbit_mapping

        def remap_counts(remapped_counts, qubit_mapping, cbit_mapping, total_cbits):
            sorted_original_cbits = sorted(cbit_mapping.keys())
            final_counts = {}
            for bitstring, count in remapped_counts.items():
                bitstring = bitstring[::-1]
                extracted_bits = ''.join([
                    bitstring[cbit_mapping[cbit]]
                    for cbit in sorted_original_cbits
                ])
                final_counts[extracted_bits] = final_counts.get(extracted_bits, 0) + count
            return final_counts

        if not self.circuits:
            raise ValueError("No circuits to run.")

        # Build noise model
        if noise_model is True:
            noise_model = build_custom_noise_model()
        elif noise_model is False or noise_model is None:
            noise_model = None
        else:
            raise ValueError("Unsupported noise model type.")

        # Initialize simulator
        simulator = AerSimulator(noise_model=noise_model)

        counts, execution_times = [], []
        total_circuits = len(self.circuits)

        for idx, circuit in enumerate(self.circuits):
            # if print_progress:
            #     print(f"Running circuit {idx+1}/{total_circuits}")

            active_qubits, active_cbits, total_cbits = get_active_qubits_and_cbits(circuit)
            if not active_qubits:
                counts.append({})
                execution_times.append(0.0)
                continue

            mapped_circuit, qubit_mapping, cbit_mapping = map_circuit(
                circuit, active_qubits, active_cbits
            )
            if compile:
                # Set optimization level to 0 and specify basis gates to prevent gate decomposition
                transpiled_circuit = transpile(
                    mapped_circuit,
                    simulator,
                    optimization_level=0,
                    basis_gates=["h", "x", "y", "z", "rx", "ry", "rz", "cz", "cx", "swap"]
                )
            else:
                transpiled_circuit = mapped_circuit

            try:
                start_time = time.time()
                job = simulator.run(transpiled_circuit, shots=shots)
                result = job.result()
                elapsed = time.time() - start_time
                counts_mapped = result.get_counts(transpiled_circuit)
                execution_times.append(elapsed)
            except Exception as e:
                print(f"Error running circuit {idx}: {e}")
                counts.append({})
                execution_times.append(0.0)
                continue

            remapped_counts = remap_counts(
                counts_mapped, qubit_mapping, cbit_mapping, total_cbits
            )
            counts.append(remapped_counts)

        return (counts, np.mean(execution_times)) if elapsed_time else counts
