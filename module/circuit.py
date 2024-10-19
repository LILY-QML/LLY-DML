# module/circuit.py

from qiskit_aer import Aer
from qiskit import QuantumCircuit, transpile
import copy

class Circuit:
    def __init__(self, qubits, depth, training_phases, activation_phases, shots):
        self.qubits = qubits
        self.depth = depth
        self.training_phases = training_phases  # Trainingsphasenmatrix
        self.activation_phases = activation_phases  # Aktivierungsphasenmatrix
        self.shots = shots  # Anzahl der Messungen
        self.circuit = QuantumCircuit(qubits, qubits)
        self.simulation_result = None  # Speichert das Simulationsergebnis
        self.initialize_gates()
        self.measure()  # Messung als Teil der Initialisierung hinzufügen


    def initialize_gates(self):
        """Initialisiert die Gates unter Verwendung von Trainings- und Aktivierungsphasen."""
        required_phase_entries = self.depth * 3

        # Validierung der Matrixdimensionen
        if len(self.training_phases) != required_phase_entries or len(self.activation_phases) != required_phase_entries:
            raise ValueError(f"Trainings- und Aktivierungsphasen müssen jeweils {required_phase_entries} Zeilen haben.")

        if any(len(row) != self.qubits for row in self.training_phases) or any(len(row) != self.qubits for row in self.activation_phases):
            raise ValueError("Jeder Phaseneintrag muss eine Länge haben, die der Anzahl der Qubits entspricht.")

        # L-Gates für jede Tiefenebene anwenden
        for d in range(self.depth):
            for qubit in range(self.qubits):
                self.apply_l_gate(qubit, d)


    def apply_l_gate(self, qubit, depth_index):
        """Wendet die L-Gate-Sequenz unter Verwendung von Trainings- und Aktivierungsphasen an."""
        for i in range(3):  # Schleife über 3 Phasen für jedes L-Gate
            tp_phase = self.training_phases[depth_index * 3 + i][qubit]
            ap_phase = self.activation_phases[depth_index * 3 + i][qubit]

            # Trainingsphase (TP) anwenden
            self.circuit.p(tp_phase, qubit)
            # Aktivierungsphase (AP) anwenden
            self.circuit.p(ap_phase, qubit)

            # Hadamard-Gate zwischen dem ersten und zweiten TP-AP-Paar und nach dem dritten Paar anwenden
            if i == 1 or i == 2:
                self.circuit.h(qubit)

    def measure(self):
        """Fügt die Messung für alle Qubits hinzu."""
        self.circuit.measure(range(self.qubits), range(self.qubits))

    def run(self):
        """Führt die Quantenschaltkreissimulation aus und gibt das Ergebnis zurück."""
        simulator = Aer.get_backend('aer_simulator')  # Aktualisiert auf AerSimulator
        compiled_circuit = transpile(self.circuit, simulator)

        # Direktes Ausführen des kompilierten Schaltkreises auf dem Simulator
        self.simulation_result = simulator.run(compiled_circuit, shots=self.shots).result()

        return self.simulation_result  # Ergebnis zurückgeben

    def get_counts(self):
        """Gibt die Zählungen aus der letzten ausgeführten Simulation zurück."""
        if self.simulation_result is not None:
            return self.simulation_result.get_counts(self.circuit)
        else:
            raise RuntimeError("Der Schaltkreis wurde noch nicht ausgeführt.")

    def __repr__(self):
        return self.circuit.draw(output='text').__str__()

    def copy(self):
        """Erstellt eine tiefe Kopie der Circuit-Instanz."""
        return copy.deepcopy(self)

    def __str__(self):
        return self.circuit.draw(output='text').__str__()

    def to_dict(self):
        """Gibt den Circuit als Dictionary zurück, um ihn in JSON zu speichern."""
        return {
            "circuit": self.circuit.draw(output='text')  # Korrigiert von self.qc zu self.circuit
        }
