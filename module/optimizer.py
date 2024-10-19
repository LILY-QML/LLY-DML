import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# module/optimizer.py

import numpy as np
import copy
import logging
import json
from .circuit import Circuit
from skopt import gp_minimize

class Optimizer:
    def __init__(self, circuit, target_state, learning_rate, max_iterations):
        self.circuit = circuit
        self.target_state = target_state
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.initial_distribution = None  # Added to track initial probabilities
        self.initial_probability = 0.0  # Added to track initial target state probability
        self.optimized_phases = None  # Added to store the optimized phases

    def loss_function(self, counts):
        """Calculate the loss as the negative probability of the target state."""
        total_shots = sum(counts.values())
        target_probability = counts.get(self.target_state, 0) / total_shots
        # We want to maximize this probability, so we minimize the negative probability
        loss = -target_probability
        return loss

    def optimize(self):
        # Initialize best phases and loss
        best_phases = np.array(self.circuit.training_phases)
        best_loss = float("inf")
        losses = []  # Track losses over iterations

        # Initial run and distribution
        self.circuit.run()  # Ensure the circuit runs first
        initial_counts = self.circuit.get_counts()
        
        # Calculate the initial distribution
        self.initial_distribution = self.get_distribution(initial_counts)
        
        # Ensure that the initial distribution is not None
        if self.initial_distribution is None:
            print("Warning: Initial distribution is None. Check circuit run and get_counts().")
        
        # Set the initial probability
        self.initial_probability = self.initial_distribution.get(self.target_state, 0.0)

        for iteration in range(self.max_iterations):
            # Evaluate current loss
            current_loss = self.evaluate(best_phases)
            losses.append(current_loss)

            # Update phases to explore new states
            new_phases = self.update_phases(best_phases)
            new_loss = self.evaluate(new_phases)

            # Accept new phases if they improve the loss
            if new_loss < best_loss:
                best_phases = new_phases
                best_loss = new_loss

            # Debugging information
            print(f"Iteration {iteration}, Loss: {best_loss}")

        # Set the optimized training phases
        self.circuit.training_phases = best_phases.tolist()
        self.optimized_phases = best_phases.tolist()  # Store optimized phases

        # Return optimized phases and loss data
        return best_phases.tolist(), losses

    def evaluate(self, training_phases):
        # Update the circuit with the new training phases
        self.circuit.training_phases = training_phases.tolist()
        self.circuit.run()

        # Get the result counts
        counts = self.circuit.get_counts()

        # Calculate the loss
        return self.loss_function(counts)

    def update_phases(self, current_phases):
        # Generate small random changes to each training phase
        new_phases = current_phases + np.random.normal(
            0, self.learning_rate, current_phases.shape
        )
        return new_phases

    def get_distribution(self, counts):
        """Get a sorted probability distribution from counts."""
        total_shots = sum(counts.values())
        if total_shots == 0:
            print("Warning: Total shots is zero. Counts may be incorrect.")
            return {}
        distribution = {state: counts[state] / total_shots for state in counts}
        return dict(sorted(distribution.items(), key=lambda item: item[1], reverse=True))

    def plot_distribution(self, counts, title):
        """Plot a histogram of the state distribution."""
        distribution = self.get_distribution(counts)
        df = pd.DataFrame(distribution.items(), columns=["State", "Probability"])
        df = df.sort_values(by="Probability", ascending=False)

        # Plot the table
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis("tight")
        ax.axis("off")
        ax.table(
            cellText=df.values, colLabels=df.columns, cellLoc="center", loc="center"
        )
        ax.set_title(title, fontsize=16)
        plt.show()

class QuantumNaturalGradientOptimizer(Optimizer):
    def __init__(self, circuit, target_state, learning_rate, max_iterations, fisher_information_matrix):
        super().__init__(circuit, target_state, learning_rate, max_iterations)
        self.fisher_information_matrix = fisher_information_matrix

    def update_phases(self, current_phases):
        gradient = np.random.normal(0, self.learning_rate, current_phases.shape)

        # Debugging dimension output
        fisher_shape = self.fisher_information_matrix.shape
        gradient_shape = gradient.flatten().shape

        # Ensure dimensions match for multiplication
        if fisher_shape[0] != gradient_shape[0] or fisher_shape[1] != gradient_shape[0]:
            raise ValueError(f"Fisher information matrix ({fisher_shape}) and gradient ({gradient_shape}) dimensions do not match.")

        # Reshape gradient for proper dimension matching
        qng_step = np.linalg.inv(self.fisher_information_matrix).dot(gradient.flatten())
        new_phases = current_phases.flatten() - self.learning_rate * qng_step
        return new_phases.reshape(current_phases.shape)

class OptimizerWithMomentum(Optimizer):
    def __init__(self, *args, momentum=0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.momentum = momentum
        self.velocity = np.zeros_like(self.circuit.training_phases)

    def update_phases(self, current_phases):
        gradient = np.random.normal(0, self.learning_rate, current_phases.shape)
        self.velocity = self.momentum * self.velocity + gradient
        new_phases = current_phases + self.velocity
        return new_phases

class AdamOptimizer(Optimizer):
    def __init__(self, *args, beta1=0.9, beta2=0.999, epsilon=1e-8, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = np.zeros_like(self.circuit.training_phases)
        self.v = np.zeros_like(self.circuit.training_phases)
        self.t = 0

    def update_phases(self, current_phases):
        self.t += 1
        gradient = np.random.normal(0, self.learning_rate, current_phases.shape)
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient ** 2)
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        new_phases = current_phases + self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return new_phases

class GeneticOptimizer(Optimizer):
    def __init__(self, circuit, target_state, learning_rate, max_iterations, population_size=20, mutation_rate=0.1, **kwargs):
        super().__init__(circuit, target_state, learning_rate, max_iterations)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        # Ensure that training_phases is a NumPy array
        self.circuit.training_phases = np.array(self.circuit.training_phases)
        self.population = self._initialize_population()

    def _initialize_population(self):
        """
        Initialize the population with random variations of the training phases.
        """
        return [self.circuit.training_phases + np.random.normal(0, self.learning_rate, self.circuit.training_phases.shape)
                for _ in range(self.population_size)]

    def optimize(self):
        """
        Perform optimization using the genetic algorithm.
        """
        losses = []  # Track losses over iterations
        for iteration in range(self.max_iterations):
            # Evaluate all individuals
            population_losses = [self.evaluate(individual) for individual in self.population]
            best_loss_idx = np.argmin(population_losses)
            best_phases = self.population[best_loss_idx]
            best_loss = population_losses[best_loss_idx]
            losses.append(best_loss)

            # Select parents (elitism and random selection)
            sorted_population = [x for _, x in sorted(zip(population_losses, self.population), key=lambda pair: pair[0])]
            parents = sorted_population[:self.population_size // 2]  # Top 50% as parents

            # Create new generation through crossover
            self.population = self._crossover_and_mutate(parents)
            print(f"Iteration {iteration}, Best Loss: {best_loss}")

        self.circuit.training_phases = best_phases.tolist()
        self.optimized_phases = best_phases.tolist()
        return best_phases.tolist(), losses

    def _crossover_and_mutate(self, parents):
        """
        Perform crossover and mutation to produce the next generation.
        """
        next_generation = []
        num_parents = len(parents)
        for _ in range(self.population_size):
            # Select two distinct parents
            parent_indices = np.random.choice(num_parents, 2, replace=False)
            parent1, parent2 = parents[parent_indices[0]], parents[parent_indices[1]]
            child = self._crossover(parent1, parent2)
            child = self._mutate(child)
            next_generation.append(child)
        return next_generation

    def _crossover(self, parent1, parent2):
        """
        Single-point crossover between two parents.
        """
        # Ensure parent arrays are NumPy arrays
        parent1 = np.array(parent1)
        parent2 = np.array(parent2)
        point = np.random.randint(0, parent1.shape[0])
        child = np.concatenate((parent1[:point], parent2[point:]))
        return child

    def _mutate(self, individual):
        """
        Mutate an individual by applying random changes.
        """
        mutation = np.random.normal(0, self.learning_rate, individual.shape)
        mask = np.random.rand(*individual.shape) < self.mutation_rate
        individual[mask] += mutation[mask]
        return individual

class PSOOptimizer(Optimizer):
    def __init__(self, circuit, target_state, learning_rate, max_iterations, num_particles=30, inertia=0.5, cognitive=1.5, social=1.5, **kwargs):
        super().__init__(circuit, target_state, learning_rate, max_iterations)
        self.num_particles = num_particles
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        # Ensure training phases are a NumPy array
        self.circuit.training_phases = np.array(self.circuit.training_phases)
        # Initialize particle positions and velocities
        self.particles = self._initialize_particles()
        self.velocities = np.zeros_like(self.particles)
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_losses = np.full(self.num_particles, float('inf'))
        self.global_best_position = None
        self.global_best_loss = float('inf')

    def _initialize_particles(self):
        """
        Initialize particles with random variations around the training phases.
        """
        # Use np.random.normal with the shape of the NumPy array
        return np.array([self.circuit.training_phases + np.random.normal(0, self.learning_rate, self.circuit.training_phases.shape)
                         for _ in range(self.num_particles)])

    def optimize(self):
        losses = []  # Track global best loss over iterations
        for iteration in range(self.max_iterations):
            # Evaluate particles
            for i in range(self.num_particles):
                loss = self.evaluate(self.particles[i])
                if loss < self.personal_best_losses[i]:
                    self.personal_best_positions[i] = np.copy(self.particles[i])
                    self.personal_best_losses[i] = loss
                if loss < self.global_best_loss:
                    self.global_best_position = np.copy(self.particles[i])
                    self.global_best_loss = loss

            # Update velocities and positions
            for i in range(self.num_particles):
                cognitive_component = self.cognitive * np.random.rand() * (self.personal_best_positions[i] - self.particles[i])
                social_component = self.social * np.random.rand() * (self.global_best_position - self.particles[i])
                self.velocities[i] = self.inertia * self.velocities[i] + cognitive_component + social_component
                self.particles[i] += self.velocities[i]

            # Append the global best loss for this iteration
            losses.append(self.global_best_loss)
            print(f"Iteration {iteration}, Best Loss: {self.global_best_loss}")

        self.circuit.training_phases = self.global_best_position.tolist()
        self.optimized_phases = self.global_best_position.tolist()
        return self.global_best_position.tolist(), losses

class BayesianOptimizer(Optimizer):
    def __init__(self, circuit, target_state, learning_rate, max_iterations, bounds):
        super().__init__(circuit, target_state, learning_rate, max_iterations)
        self.bounds = bounds

    def optimize(self):
        def objective(phases):
            # Reshape to match the circuit's expected phase shape
            phase_shape = np.array(self.circuit.training_phases).shape
            self.circuit.training_phases = np.array(phases).reshape(phase_shape)
            self.circuit.run()
            counts = self.circuit.get_counts()
            return self.loss_function(counts)

        # Ensure the bounds match the flat number of phases
        phase_shape = np.array(self.circuit.training_phases).shape
        flat_phases_size = np.prod(phase_shape)
        self.bounds = [(0, 2 * np.pi)] * flat_phases_size

        result = gp_minimize(
            func=objective,
            dimensions=self.bounds,
            n_calls=self.max_iterations,
            random_state=42
        )

        # Reshape result back into the training phases
        self.circuit.training_phases = np.array(result.x).reshape(phase_shape)
        self.optimized_phases = self.circuit.training_phases.tolist()
        return self.optimized_phases, [result.fun] * self.max_iterations

class SimulatedAnnealingOptimizer(Optimizer):
    def __init__(self, circuit, target_state, learning_rate, max_iterations, initial_temperature=1.0, cooling_rate=0.99, **kwargs):
        super().__init__(circuit, target_state, learning_rate, max_iterations)
        self.temperature = initial_temperature
        self.cooling_rate = cooling_rate

    def optimize(self):
        current_phases = np.copy(self.circuit.training_phases)
        best_phases = np.copy(current_phases)
        best_loss = self.evaluate(current_phases)
        losses = []

        for iteration in range(self.max_iterations):
            new_phases = self.update_phases(current_phases)
            new_loss = self.evaluate(new_phases)

            # Accept new phases with probability based on temperature
            if new_loss < best_loss or np.random.rand() < np.exp((best_loss - new_loss) / self.temperature):
                current_phases = new_phases
                if new_loss < best_loss:
                    best_phases = new_phases
                    best_loss = new_loss

            losses.append(best_loss)
            self.temperature *= self.cooling_rate
            print(f"Iteration {iteration}, Temperature: {self.temperature}, Loss: {best_loss}")

        self.circuit.training_phases = best_phases.tolist()
        self.optimized_phases = best_phases.tolist()
        return best_phases.tolist(), losses



class OptimizerManager:
    def __init__(self, reader, data, train_file, logger):
        self.reader = reader
        self.data = data
        self.train_file = train_file
        self.logger = logger

        # Extrahiere Aktivierungsmatrizen und Trainingsmatrix
        self.activation_labels = list(self.reader.train_data.get('converted_activation_matrices', {}).keys())
        self.training_matrix = self.reader.train_data.get('training_matrix', [])
        self.qubits = self.reader.data.get('qubits', 8)
        self.depth = self.reader.data.get('depth', 8)

        # Überprüfen, ob die Trainingsmatrix geladen wurde
        if not self.training_matrix:
            self.logger.error("Keine Trainingsmatrix in train.json gefunden.")
            raise ValueError("Keine Trainingsmatrix in train.json gefunden.")

        # Konvertiere die Trainingsmatrix in ein NumPy-Array und transponiere sie
        self.training_phases = np.array(self.training_matrix)
        self.logger.debug(f"Geladene Trainingsmatrix hat die Form: {self.training_phases.shape}")

        # Transponiere die Trainingsphasen
        self.training_phases = self.training_phases.T
        self.logger.debug(f"Transponierte Trainingsphasen haben die Form: {self.training_phases.shape}")

        # Lade die Aktivierungsphasen
        self.activation_phases_dict = self.reader.train_data.get('converted_activation_matrices', {})
        for label, activation_phases in self.activation_phases_dict.items():
            activation_array = np.array(activation_phases)
            self.activation_phases_dict[label] = activation_array
            self.logger.debug(f"Aktivierungsphasen für {label} haben die Form: {activation_array.shape}")

        # Überprüfe die Anzahl der Zeilen
        expected_rows = self.depth * 3
        if self.training_phases.shape[0] != expected_rows:
            error_msg = f"Trainingsphasen haben {self.training_phases.shape[0]} Zeilen, erwartet: {expected_rows}."
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        for label, activation_array in self.activation_phases_dict.items():
            if activation_array.shape[0] != expected_rows:
                error_msg = f"Aktivierungsphasen für {label} haben {activation_array.shape[0]} Zeilen, erwartet: {expected_rows}."
                self.logger.error(error_msg)
                raise ValueError(error_msg)

        # Initialisiere die Zielzustände
        self.assign_unique_target_states()
        # Initialisiere die Optimierer
        self.initialize_optimizers()

    def assign_unique_target_states(self):
        """
        Weist jeder Aktivierungsmatrix einen eindeutigen Zielzustand zu.
        """
        self.target_states = {}
        used_states = set()

        for label in self.activation_labels:
            # Erstelle einen Circuit mit der aktuellen Aktivierungsmatrix
            circuit = Circuit(
                qubits=self.qubits,
                depth=self.depth,
                training_phases=copy.deepcopy(self.training_matrix),
                activation_phases=self.reader.train_data['converted_activation_matrices'][label],
                shots=self.reader.data.get('shots', 1024)
            )
            circuit.run()
            counts = circuit.get_counts()

            if not counts:
                self.logger.error(f"Keine Messergebnisse für Aktivierungsmatrix '{label}' erhalten.")
                raise ValueError(f"Keine Messergebnisse für Aktivierungsmatrix '{label}' erhalten.")

            # Ermittle den Zustand mit der höchsten Wahrscheinlichkeit
            sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)
            target_state = sorted_counts[0][0]

            # Stelle sicher, dass der Zielzustand eindeutig ist
            if target_state in used_states:
                for state, _ in sorted_counts[1:]:
                    if state not in used_states:
                        target_state = state
                        break
                else:
                    self.logger.error(f"Keine eindeutigen Zielzustände für '{label}' gefunden.")
                    raise ValueError(f"Keine eindeutigen Zielzustände für '{label}' gefunden.")

            self.target_states[label] = target_state
            used_states.add(target_state)
            self.logger.info(f"Zielzustand für '{label}': {target_state}")

    def initialize_optimizers(self):
        """
        Initialisiert die Optimierer basierend auf den in data.json definierten Optimierern.
        Jeder Optimierer erhält eine Kopie des initialen Trainingsschaltkreises.
        """
        self.optimizer_instances = []
        optimizers_config = self.reader.data.get('optimizers', {})

        if isinstance(optimizers_config, dict):
            optimizer_names = list(optimizers_config.keys())
        elif isinstance(optimizers_config, list):
            optimizer_names = optimizers_config
        else:
            self.logger.error("Unbekanntes Format für Optimizer in data.json.")
            raise ValueError("Unbekanntes Format für Optimizer in data.json.")

        for optimizer_name in optimizer_names:
            optimizer_class = self.get_optimizer_class(optimizer_name)
            if optimizer_class is None:
                self.logger.warning(f"Optimizer '{optimizer_name}' ist unbekannt und wird übersprungen.")
                continue

            for label in self.activation_labels:
                target_state = self.target_states[label]
                # Initialisiere den Circuit mit der initialen Trainingsmatrix und der aktuellen Aktivierungsmatrix
                initial_circuit = Circuit(
                    qubits=self.qubits,
                    depth=self.depth,
                    training_phases=copy.deepcopy(self.training_matrix),
                    activation_phases=self.reader.train_data['converted_activation_matrices'][label],
                    shots=self.reader.data.get('shots', 1024)
                )
                # Initialisiere den Optimierer
                optimizer_instance = optimizer_class(
                    circuit=initial_circuit,
                    target_state=target_state,
                    learning_rate=self.reader.data.get('learning_rate', 0.01),
                    max_iterations=self.reader.data.get('max_iterations', 100),
                    # Weitere Parameter je nach Optimierer hinzufügen
                )
                self.optimizer_instances.append((optimizer_name, label, optimizer_instance))
                self.logger.info(f"Optimizer '{optimizer_name}' für '{label}' initialisiert.")

    def get_optimizer_class(self, name):
        """
        Mappt den Optimierernamen auf die entsprechende Klasse.

        :param name: Name des Optimierers
        :return: Optimierer-Klasse
        """
        mapping = {
            'Basic': Optimizer,
            'Momentum': OptimizerWithMomentum,
            'Adam': AdamOptimizer,
            'Genetic': GeneticOptimizer,
            'PSO': PSOOptimizer,
            'Bayesian': BayesianOptimizer,
            'SimulatedAnnealing': SimulatedAnnealingOptimizer,
            'QuantumNaturalGradient': QuantumNaturalGradientOptimizer
            # Weitere Mapping hinzufügen, falls notwendig
        }
        return mapping.get(name)

    def train_all_optimizers(self):
        """
        Führt das Training aller Optimierer durch. Jeder Optimierer optimiert seine zugewiesene Aktivierungsmatrix.
        Die Trainingsschritte werden geloggt und in train.json gespeichert.
        """
        for optimizer_name, label, optimizer in self.optimizer_instances:
            self.logger.info(f"Starte Training für Optimizer: {optimizer_name}, Aktivierungsmatrix: {label}")
            print(f"Starte Training für Optimizer: {optimizer_name}, Aktivierungsmatrix: {label}")
            try:
                optimized_phases, losses = optimizer.optimize()
                # Logge Optimierungsergebnisse
                self.logger.info(f"Optimierung abgeschlossen für '{label}' mit Optimizer '{optimizer_name}'. Letzter Verlust: {losses[-1]}")
                print(f"Optimierung abgeschlossen für '{label}' mit Optimizer '{optimizer_name}'. Letzter Verlust: {losses[-1]}")

                # Schreibe die Ergebnisse in train.json
                self.write_optimization_to_train_json(
                    optimizer_name=optimizer_name,
                    activation_label=label,
                    optimized_phases=optimized_phases,
                    losses=losses
                )
            except Exception as e:
                self.logger.error(f"Fehler beim Trainieren von Optimizer '{optimizer_name}' für '{label}': {e}")
                print(f"Fehler beim Trainieren von Optimizer '{optimizer_name}' für '{label}': {e}")

        self.logger.info("Alle Optimierer wurden erfolgreich trainiert.")
        print("Alle Optimierer wurden erfolgreich trainiert.")

    def write_optimization_to_train_json(self, optimizer_name, label, optimized_phases, losses):
        """
        Schreibt die Optimierungsschritte in die train.json-Datei.

        :param optimizer_name: Name des Optimierers
        :param label: Label der Aktivierungsmatrix
        :param optimized_phases: Optimierte Trainingsphasen
        :param losses: Liste der Verlustwerte über die Iterationen
        """
        optimization_entry = {
            "optimizer": optimizer_name,
            "activation_matrix": label,
            "optimized_phases": optimized_phases,
            "losses": losses
        }

        # Füge die Optimierungsschritte unter 'optimizer_steps' hinzu
        if 'optimizer_steps' not in self.reader.train_data:
            self.reader.train_data['optimizer_steps'] = []
        self.reader.train_data['optimizer_steps'].append(optimization_entry)

        # Schreibe zurück in train.json
        try:
            with open(self.train_file, 'w') as f:
                json.dump(self.reader.train_data, f, indent=4)
            self.logger.info(f"Optimierungsschritt für '{optimizer_name}' und '{label}' in train.json gespeichert.")
        except Exception as e:
            self.logger.error(f"Fehler beim Schreiben der Optimierungsschritte in train.json: {e}")
            raise
