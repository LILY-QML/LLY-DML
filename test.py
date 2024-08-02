import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from qiskit import QuantumCircuit
from module.circuit import Circuit
from module.optimizer import (
    Optimizer,
    OptimizerWithMomentum,
    AdamOptimizer,
    GeneticOptimizer,
    PSOOptimizer,
    BayesianOptimizer,
    SimulatedAnnealingOptimizer,
    QuantumNaturalGradientOptimizer,
)
from module.visual import Visual
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate

# Create output directory
os.makedirs("var", exist_ok=True)

class TestQuantumCircuit(unittest.TestCase):
    def setUp(self):
        # Set up parameters for the tests
        self.qubits = 5
        self.depth = 3
        self.shots = 1024
        self.training_phases = np.random.rand(self.depth * 3, self.qubits).tolist()
        self.activation_phases = np.random.rand(self.depth * 3, self.qubits).tolist()

    def test_circuit_initialization(self):
        # Test if the circuit initializes without error
        circuit = Circuit(
            qubits=self.qubits,
            depth=self.depth,
            training_phases=self.training_phases,
            activation_phases=self.activation_phases,
            shots=self.shots,
        )
        self.assertIsInstance(circuit.circuit, QuantumCircuit)

    def test_circuit_run(self):
        # Test if the circuit runs and produces results
        circuit = Circuit(
            qubits=self.qubits,
            depth=self.depth,
            training_phases=self.training_phases,
            activation_phases=self.activation_phases,
            shots=self.shots,
        )
        result = circuit.run()
        counts = circuit.get_counts()
        self.assertIsNotNone(result)
        self.assertIsInstance(counts, dict)
        self.assertEqual(sum(counts.values()), self.shots)

    def test_circuit_measurement(self):
        # Test if the measurement is added to the circuit
        circuit = Circuit(
            qubits=self.qubits,
            depth=self.depth,
            training_phases=self.training_phases,
            activation_phases=self.activation_phases,
            shots=self.shots,
        )
        circuit.measure()
        # Check if the number of classical bits matches the number of qubits
        self.assertEqual(len(circuit.circuit.clbits), self.qubits)


class TestOptimizers(unittest.TestCase):
    def setUp(self):
        # Set up a circuit for optimizer tests
        self.qubits = 5
        self.depth = 3
        self.shots = 1024
        self.training_phases = np.random.rand(self.depth * 3, self.qubits).tolist()
        self.activation_phases = np.random.rand(self.depth * 3, self.qubits).tolist()
        self.circuit = Circuit(
            qubits=self.qubits,
            depth=self.depth,
            training_phases=self.training_phases,
            activation_phases=self.activation_phases,
            shots=self.shots,
        )
        self.target_state = max(self.circuit.run().get_counts(), key=lambda x: self.circuit.get_counts()[x])
        self.learning_rate = 0.01
        self.max_iterations = 10

    def test_gradient_descent_optimizer(self):
        optimizer = Optimizer(
            circuit=self.circuit,
            target_state=self.target_state,
            learning_rate=self.learning_rate,
            max_iterations=self.max_iterations,
        )
        optimized_phases, losses = optimizer.optimize()
        self.assertIsInstance(optimized_phases, list)
        self.assertIsInstance(losses, list)
        self.assertEqual(len(losses), self.max_iterations)

    def test_momentum_optimizer(self):
        optimizer = OptimizerWithMomentum(
            circuit=self.circuit,
            target_state=self.target_state,
            learning_rate=self.learning_rate,
            max_iterations=self.max_iterations,
        )
        optimized_phases, losses = optimizer.optimize()
        self.assertIsInstance(optimized_phases, list)
        self.assertIsInstance(losses, list)
        self.assertEqual(len(losses), self.max_iterations)

    def test_adam_optimizer(self):
        optimizer = AdamOptimizer(
            circuit=self.circuit,
            target_state=self.target_state,
            learning_rate=self.learning_rate,
            max_iterations=self.max_iterations,
        )
        optimized_phases, losses = optimizer.optimize()
        self.assertIsInstance(optimized_phases, list)
        self.assertIsInstance(losses, list)
        self.assertEqual(len(losses), self.max_iterations)

    def test_genetic_optimizer(self):
        optimizer = GeneticOptimizer(
            circuit=self.circuit,
            target_state=self.target_state,
            learning_rate=self.learning_rate,
            max_iterations=self.max_iterations,
        )
        optimized_phases, losses = optimizer.optimize()
        self.assertIsInstance(optimized_phases, list)
        self.assertIsInstance(losses, list)
        self.assertEqual(len(losses), self.max_iterations)

    def test_pso_optimizer(self):
        optimizer = PSOOptimizer(
            circuit=self.circuit,
            target_state=self.target_state,
            learning_rate=self.learning_rate,
            max_iterations=self.max_iterations,
        )
        optimized_phases, losses = optimizer.optimize()
        self.assertIsInstance(optimized_phases, list)
        self.assertIsInstance(losses, list)
        self.assertEqual(len(losses), self.max_iterations)

    def test_bayesian_optimizer(self):
        bounds = [(0, 2 * np.pi) for _ in range(self.qubits * self.depth)]
        optimizer = BayesianOptimizer(
            circuit=self.circuit,
            target_state=self.target_state,
            learning_rate=self.learning_rate,
            max_iterations=self.max_iterations,
            bounds=bounds,
        )
        optimized_phases, losses = optimizer.optimize()
        self.assertIsInstance(optimized_phases, list)
        self.assertIsInstance(losses, list)
        self.assertEqual(len(losses), self.max_iterations)

    def test_simulated_annealing_optimizer(self):
        optimizer = SimulatedAnnealingOptimizer(
            circuit=self.circuit,
            target_state=self.target_state,
            learning_rate=self.learning_rate,
            max_iterations=self.max_iterations,
        )
        optimized_phases, losses = optimizer.optimize()
        self.assertIsInstance(optimized_phases, list)
        self.assertIsInstance(losses, list)
        self.assertEqual(len(losses), self.max_iterations)

    def test_qng_optimizer(self):
        fisher_information_matrix = np.eye(self.qubits * self.depth)
        optimizer = QuantumNaturalGradientOptimizer(
            circuit=self.circuit,
            target_state=self.target_state,
            learning_rate=self.learning_rate,
            max_iterations=self.max_iterations,
            fisher_information_matrix=fisher_information_matrix,
        )
        optimized_phases, losses = optimizer.optimize()
        self.assertIsInstance(optimized_phases, list)
        self.assertIsInstance(losses, list)
        self.assertEqual(len(losses), self.max_iterations)


class TestVisual(unittest.TestCase):
    def setUp(self):
        # Mock data for testing
        self.results = [
            {
                "Activation Matrix": 1,
                "Target State": "11111",
                "Optimizer": "Gradient Descent",
                "Initial Probability": 0.0012,
                "Final Probability": 0.0234,
                "Final Counts": {"shots": 1024},
                "Optimized Phases": [[0.1, 0.2, 0.3, 0.4, 0.5]],
            },
            {
                "Activation Matrix": 1,
                "Target State": "00000",
                "Optimizer": "Momentum",
                "Initial Probability": 0.005,
                "Final Probability": 0.025,
                "Final Counts": {"shots": 1024},
                "Optimized Phases": [[0.15, 0.25, 0.35, 0.45, 0.55]],
            },
            {
                "Activation Matrix": 1,
                "Target State": "10101",
                "Optimizer": "Adam",
                "Initial Probability": 0.002,
                "Final Probability": 0.022,
                "Final Counts": {"shots": 1024},
                "Optimized Phases": [[0.12, 0.22, 0.32, 0.42, 0.52]],
            },
        ]

        # Generate random initial training phases based on dimensions of activation matrices
        self.activation_matrices = [np.random.rand(9, 5) for _ in range(3)]
        self.initial_training_phases = [np.random.rand(9, 5) for _ in range(3)]

        # Create mock circuits
        self.circuits = [QuantumCircuit(5, 5) for _ in range(3)]

        self.visual = Visual(
            results=self.results,
            target_states=["11111", "00000", "10101"],
            initial_training_phases=self.initial_training_phases,
            activation_matrices=self.activation_matrices,
            circuits=self.circuits,
            num_iterations=100,
            qubits=5,
            depth=3,
        )

    def test_generate_report(self):
        # Generate the report
        self.visual.generate_report("test_report.pdf")
        # Check if the PDF file was created
        self.assertTrue(os.path.exists("test_report.pdf"))
        # Clean up the generated file
        if os.path.exists("test_report.pdf"):
            os.remove("test_report.pdf")

# Run the tests
if __name__ == "__main__":
    unittest.main()
