from qiskit_aer import Aer
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.visualization import plot_histogram

class Circuit:
    def __init__(self, qubits, depth):
        self.qubits = qubits
        self.depth = depth
        self.circuit = QuantumCircuit(qubits, qubits)
        self.initialize_gates()

    def initialize_gates(self):
        for _ in range(self.depth):
            for qubit in range(self.qubits):
                self.apply_l_gate(qubit)

    def apply_l_gate(self, qubit):
        # This method applies the L-Gate sequence
        self.circuit.p(0.1, qubit)
        self.circuit.p(0.2, qubit)
        self.circuit.h(qubit)
        self.circuit.p(0.3, qubit)
        self.circuit.p(0.4, qubit)
        self.circuit.h(qubit)
        self.circuit.p(0.5, qubit)
        self.circuit.p(0.6, qubit)

    def measure(self):
        self.circuit.measure(range(self.qubits), range(self.qubits))

    def __repr__(self):
        return self.circuit.draw(output='text')

# Beispielnutzung
circuit = Circuit(5, 10)
circuit.measure()

# Circuit anzeigen
print(circuit)

# Simulieren des Circuits
simulator = Aer.get_backend('qasm_simulator')
compiled_circuit = transpile(circuit.circuit, simulator)
qobj = assemble(compiled_circuit)
result = simulator.run(qobj).result()

# Ergebnisse anzeigen
counts = result.get_counts()
print(counts)
plot_histogram(counts).show()

