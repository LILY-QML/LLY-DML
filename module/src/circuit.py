# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Project: LILY-QML
# Version: 2.0.0 LLY-DML
# Author: Leon Kaiser
# Contact: info@lilyqml.de
# Website: www.lilyqml.de
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++

from qiskit_aer import Aer
from qiskit import QuantumCircuit, transpile
import copy

class Circuit:
    def __init__(self, qubits, depth, training_phases, activation_phases, shots):
        """Initializes the Circuit class with qubits, depth, phase matrices, and number of shots."""
        self.qubits = qubits
        self.depth = depth
        self.training_phases = training_phases  # Matrix containing training phases
        self.activation_phases = activation_phases  # Matrix containing activation phases
        self.shots = shots  # Number of shots for simulation
        self.circuit = QuantumCircuit(qubits, qubits)  # Quantum circuit with qubits as both quantum and classical registers
        self.simulation_result = None  # Stores the simulation result
        self.initialize_gates()  # Initialize quantum gates
        self.measure()  # Add measurement to the circuit

    def initialize_gates(self):
        """Initializes the gates using training and activation phase matrices."""
        required_phase_entries = self.qubits  # Number of qubits defines the required rows in the matrices

        # Validate that the phase matrices have the correct dimensions
        if len(self.training_phases) != required_phase_entries or len(self.activation_phases) != required_phase_entries:
            raise ValueError(f"Training and activation phases must each have {required_phase_entries} rows.")

        if any(len(row) != self.depth * 3 for row in self.training_phases) or any(len(row) != self.depth * 3 for row in self.activation_phases):
            raise ValueError(f"Each phase entry must have a length of {self.depth * 3}.")

        # Apply L-gates for each qubit and for each depth level
        for qubit in range(self.qubits):
            for d in range(self.depth):
                self.apply_l_gate(qubit, d)

    def apply_l_gate(self, qubit, depth_index):
        """Applies an L-gate sequence using the training and activation phases."""
        for i in range(3):  # Each L-gate has three phases
            index = depth_index * 3 + i  # Calculate the index within the phase matrix

            # Retrieve the training and activation phase values for the current qubit
            tp_phase = self.training_phases[qubit][index]
            ap_phase = self.activation_phases[qubit][index]

            # Apply the training phase (TP) as a quantum phase gate
            self.circuit.p(tp_phase, qubit)
            # Apply the activation phase (AP) as a quantum phase gate
            self.circuit.p(ap_phase, qubit)

            # Apply Hadamard gates after certain phase pairs
            if i == 1 or i == 2:
                self.circuit.h(qubit)

    def measure(self):
        """Adds measurement operations to all qubits."""
        self.circuit.measure(range(self.qubits), range(self.qubits))

    def run(self):
        """Runs the quantum circuit simulation and returns the result."""
        simulator = Aer.get_backend('aer_simulator')  # Use the Aer simulator backend
        compiled_circuit = transpile(self.circuit, simulator)  # Compile the quantum circuit for the simulator

        # Execute the compiled circuit on the simulator and store the result
        self.simulation_result = simulator.run(compiled_circuit, shots=self.shots).result()

        return self.simulation_result  # Return the simulation result

    def get_counts(self):
        """Returns the measurement counts from the last simulation run."""
        if self.simulation_result is not None:
            return self.simulation_result.get_counts(self.circuit)
        else:
            raise RuntimeError("The circuit has not been executed yet.")

    def __repr__(self):
        """Returns a string representation of the circuit."""
        return self.circuit.draw(output='text').__str__()

    def copy(self):
        """Creates a deep copy of the Circuit instance."""
        return copy.deepcopy(self)

    def __str__(self):
        """Returns a string version of the circuit drawing."""
        return self.circuit.draw(output='text').__str__()

    def to_dict(self):
        """Returns the circuit in dictionary form for storage in JSON format."""
        return {
            "circuit": self.circuit.draw(output='text')
        }
