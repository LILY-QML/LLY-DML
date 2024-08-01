from qiskit_aer import Aer
from qiskit import QuantumCircuit, transpile

class Circuit:
    def __init__(self, qubits, depth, training_phases, activation_phases, shots):
        self.qubits = qubits
        self.depth = depth
        self.training_phases = training_phases  # Training phases matrix
        self.activation_phases = activation_phases  # Activation phases matrix
        self.shots = shots  # Number of measurements
        self.circuit = QuantumCircuit(qubits, qubits)
        self.simulation_result = None  # Store the simulation result
        self.initialize_gates()
        self.measure()  # Set up measurement as part of initialization

    def initialize_gates(self):
        """Initialize the gates using training and activation phases."""
        required_phase_entries = self.depth * 3

        # Validate matrix dimensions
        if len(self.training_phases) != required_phase_entries or len(self.activation_phases) != required_phase_entries:
            raise ValueError(f"Training and activation phases must each have {required_phase_entries} rows.")

        if any(len(row) != self.qubits for row in self.training_phases) or any(len(row) != self.qubits for row in self.activation_phases):
            raise ValueError("Each phase entry must have a length equal to the number of qubits.")

        # Apply L-Gates for each depth level
        for d in range(self.depth):
            for qubit in range(self.qubits):
                self.apply_l_gate(qubit, d)

    def apply_l_gate(self, qubit, depth_index):
        """Apply L-Gate sequence using training and activation phases."""
        for i in range(3):  # Loop over 3 phases for each L-Gate
            tp_phase = self.training_phases[depth_index * 3 + i][qubit]
            ap_phase = self.activation_phases[depth_index * 3 + i][qubit]

            # Apply the training phase (TP)
            self.circuit.p(tp_phase, qubit)
            # Apply the activation phase (AP)
            self.circuit.p(ap_phase, qubit)

            # Apply Hadamard gate between the first and second TP-AP pair, and after the third pair
            if i == 1 or i == 2:
                self.circuit.h(qubit)

    def measure(self):
        """Add measurement to all qubits."""
        self.circuit.measure(range(self.qubits), range(self.qubits))

    def run(self):
        """Run the quantum circuit simulation and store the result."""
        simulator = Aer.get_backend('qasm_simulator')
        compiled_circuit = transpile(self.circuit, simulator)
        self.simulation_result = simulator.run(compiled_circuit, shots=self.shots).result()

    def get_counts(self):
        """Return the counts from the last run simulation."""
        if self.simulation_result is not None:
            return self.simulation_result.get_counts(self.circuit)
        else:
            raise RuntimeError("The circuit has not been run yet.")

    def __repr__(self):
        return self.circuit.draw(output='text').__str__()
