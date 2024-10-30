# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Project: LILY-QML
# Version: 1.6 LLY-DML
# Author: Leon Kaiser
# Contact: info@lilyqml.de
# Website: www.lilyqml.de
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++

import logging
import json
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from abc import ABC, abstractmethod
import os

# Abstract base class for all Optimizers
class Optimizer(ABC):
    """
    Abstract base class for all optimizers.
    Every subclass must implement the 'optimize' and 'evaluate' methods.
    """
    def __init__(self, circuit, target_state, learning_rate=0.01, max_iterations=100):
        """
        Initializes the Optimizer class.
        
        :param circuit: The circuit object to be optimized.
        :param target_state: The target state the circuit aims to reach.
        :param learning_rate: The learning rate for optimization.
        :param max_iterations: The maximum number of iterations.
        """
        self.circuit = circuit
        self.target_state = target_state
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations

    @abstractmethod
    def optimize(self):
        """
        Executes the optimization process.
        
        :return: Tuple containing optimized phases and a list of optimization steps.
        """
        pass

    @abstractmethod
    def evaluate(self):
        """
        Evaluates the current state of the circuit.
        
        :return: The current loss value.
        """
        pass

# Basic Optimizer class implementation
class BasicOptimizer(Optimizer):
    """
    A simple implementation of a Basic Optimizer.
    """
    def optimize(self):
        logging.info("BasicOptimizer: Starting optimization process")
        optimization_steps = []
        for iteration in range(self.max_iterations):
            loss = self.evaluate()
            optimization_steps.append({"iteration": iteration, "loss": loss})
            logging.debug(f"BasicOptimizer: Iteration {iteration}, Loss: {loss}")
            if loss < 1e-6:
                logging.info("BasicOptimizer: Loss below threshold, optimization complete")
                break
            # Placeholder for actual optimization steps
        optimized_phases = self.circuit.training_phases  # Placeholder example
        logging.info("BasicOptimizer: Optimization finished")
        return optimized_phases, optimization_steps

    def evaluate(self):
        # Example loss calculation
        return np.random.random()

# Momentum Optimizer class implementation
class MomentumOptimizer(Optimizer):
    """
    Implementation of an optimizer with momentum.
    """
    def __init__(self, circuit, target_state, learning_rate=0.01, max_iterations=100, momentum=0.9):
        super().__init__(circuit, target_state, learning_rate, max_iterations)
        self.momentum = momentum
        self.velocity = 0

    def optimize(self):
        logging.info("MomentumOptimizer: Starting optimization process")
        optimization_steps = []
        for iteration in range(self.max_iterations):
            loss = self.evaluate()
            optimization_steps.append({"iteration": iteration, "loss": loss})
            logging.debug(f"MomentumOptimizer: Iteration {iteration}, Loss: {loss}")
            if loss < 1e-6:
                logging.info("MomentumOptimizer: Loss below threshold, optimization complete")
                break
            # Placeholder for optimization steps with momentum
            self.velocity = self.momentum * self.velocity - self.learning_rate * loss
        optimized_phases = self.circuit.training_phases  # Placeholder example
        logging.info("MomentumOptimizer: Optimization finished")
        return optimized_phases, optimization_steps

    def evaluate(self):
        return np.random.random()

# Adam Optimizer class implementation
class AdamOptimizer(Optimizer):
    """
    Implementation of the Adam optimizer.
    """
    def __init__(self, circuit, target_state, learning_rate=0.001, max_iterations=100, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(circuit, target_state, learning_rate, max_iterations)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = 0
        self.v = 0

    def optimize(self):
        logging.info("AdamOptimizer: Starting optimization process")
        optimization_steps = []
        for iteration in range(1, self.max_iterations + 1):
            loss = self.evaluate()
            optimization_steps.append({"iteration": iteration, "loss": loss})
            logging.debug(f"AdamOptimizer: Iteration {iteration}, Loss: {loss}")
            if loss < 1e-6:
                logging.info("AdamOptimizer: Loss below threshold, optimization complete")
                break
            # Adam optimization step
            self.m = self.beta1 * self.m + (1 - self.beta1) * loss
            self.v = self.beta2 * self.v + (1 - self.beta2) * (loss ** 2)
            m_hat = self.m / (1 - self.beta1 ** iteration)
            v_hat = self.v / (1 - self.beta2 ** iteration)
            update = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        optimized_phases = self.circuit.training_phases  # Placeholder example
        logging.info("AdamOptimizer: Optimization finished")
        return optimized_phases, optimization_steps

    def evaluate(self):
        return np.random.random()

# Genetic Optimizer class implementation
class GeneticOptimizer(Optimizer):
    """
    Implementation of a genetic optimizer.
    """
    def __init__(self, circuit, target_state, population_size=50, generations=100):
        super().__init__(circuit, target_state)
        self.population_size = population_size
        self.generations = generations

    def optimize(self):
        logging.info("GeneticOptimizer: Starting optimization process")
        optimization_steps = []
        for generation in range(self.generations):
            loss = self.evaluate()
            optimization_steps.append({"generation": generation, "loss": loss})
            logging.debug(f"GeneticOptimizer: Generation {generation}, Loss: {loss}")
            if loss < 1e-6:
                logging.info("GeneticOptimizer: Loss below threshold, optimization complete")
                break
            # Placeholder for genetic optimization steps
        optimized_phases = self.circuit.training_phases  # Placeholder example
        logging.info("GeneticOptimizer: Optimization finished")
        return optimized_phases, optimization_steps

    def evaluate(self):
        return np.random.random()

# Particle Swarm Optimization (PSO) class implementation
class PSOOptimizer(Optimizer):
    """
    Implementation of a Particle Swarm Optimization (PSO) optimizer.
    """
    def __init__(self, circuit, target_state, swarm_size=30, iterations=100):
        super().__init__(circuit, target_state)
        self.swarm_size = swarm_size
        self.iterations = iterations

    def optimize(self):
        logging.info("PSOOptimizer: Starting optimization process")
        optimization_steps = []
        for iteration in range(self.iterations):
            loss = self.evaluate()
            optimization_steps.append({"iteration": iteration, "loss": loss})
            logging.debug(f"PSOOptimizer: Iteration {iteration}, Loss: {loss}")
            if loss < 1e-6:
                logging.info("PSOOptimizer: Loss below threshold, optimization complete")
                break
            # Placeholder for PSO steps
        optimized_phases = self.circuit.training_phases  # Placeholder example
        logging.info("PSOOptimizer: Optimization finished")
        return optimized_phases, optimization_steps

    def evaluate(self):
        return np.random.random()

# Function to train a single optimizer
def train_optimizer(optimizer_tuple):
    """
    Executes the training of a single optimizer.
    
    :param optimizer_tuple: Tuple containing (optimizer_name, label, optimizer_instance).
    :return: Tuple with (optimizer_name, label, optimized_phases, optimization_steps, error).
    """
    optimizer_name, label, optimizer_instance = optimizer_tuple
    logging.info(f"Training optimizer called for {optimizer_name} - {label}")
    try:
        optimized_phases, optimization_steps = optimizer_instance.optimize()
        logging.info(f"Optimization complete for {optimizer_name} - {label}")
        return (optimizer_name, label, optimized_phases, optimization_steps, None)
    except Exception as e:
        logging.error(f"Error with optimizer {optimizer_name} for {label}: {e}")
        return (optimizer_name, label, None, None, str(e))


# Optimizer Manager class
class OptimizerManager:
    """
    Manages and trains all optimizers.
    """
    def __init__(self, reader, data, train_file, logger):
        """
        Initializes the OptimizerManager.
        
        :param reader: An object containing the training data.
        :param data: Training parameters and configurations.
        :param train_file: Path to the training file (JSON).
        :param logger: Logger object for logging.
        """
        self.reader = reader
        self.data = data
        self.train_file = train_file
        self.logger = logger

        # Initialize optimizer instances
        self.optimizer_instances = []
        activation_matrices = self.reader.train_data.get('converted_activation_matrices', {})
        for label, matrix in activation_matrices.items():
            circuit = Circuit(training_phases=[0.1] * 24)  # Example circuit, adjust as needed
            basic = BasicOptimizer(circuit=circuit, target_state='011000110')
            momentum = MomentumOptimizer(circuit=circuit, target_state='011000110')
            adam = AdamOptimizer(circuit=circuit, target_state='011000110')
            genetic = GeneticOptimizer(circuit=circuit, target_state='011000110')
            pso = PSOOptimizer(circuit=circuit, target_state='011000110')

            self.optimizer_instances.extend([
                ('Basic', label, basic),
                ('Momentum', label, momentum),
                ('Adam', label, adam),
                ('Genetic', label, genetic),
                ('PSO', label, pso)
            ])

    def write_optimization_to_train_json(self, optimizer_name, activation_label, optimization_steps):
        """
        Writes the optimization results to the training file.
        
        :param optimizer_name: Name of the optimizer.
        :param activation_label: Label of the activation matrix.
        :param optimization_steps: List of optimization steps.
        """
        self.reader.train_data.setdefault('optimizer_steps', []).append({
            "optimizer": optimizer_name,
            "activation_matrix": activation_label,
            "optimization_steps": optimization_steps
        })
        try:
            with open(self.train_file, 'w') as f:
                json.dump(self.reader.train_data, f, indent=4)
            self.logger.info(f"Optimization steps for '{optimizer_name}' and '{activation_label}' saved to train.json.")
        except Exception as e:
            self.logger.error(f"Error writing optimization steps to train.json: {e}")

    def train_all_optimizers(self, max_workers=None):
        """
        Trains all optimizers in parallel execution.
        
        :param max_workers: Maximum number of parallel processes.
        """
        self.logger.info("Starting training of all optimizers in parallel execution.")
        print("Starting training of all optimizers in parallel execution.")

        if max_workers is None:
            max_workers = os.cpu_count()

        self.logger.info(f"Number of processes used: {max_workers}")
        print(f"Number of processes used: {max_workers}")

        start_time = datetime.now()
        self.logger.info(f"Training started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}.")
        print(f"Training started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}.")

        results = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(train_optimizer, optimizer_tuple): optimizer_tuple for optimizer_tuple in self.optimizer_instances}
            self.logger.info(f"{len(futures)} optimizer training tasks submitted.")

            for future in as_completed(futures):
                optimizer_tuple = futures[future]
                try:
                    optimizer_name, label, optimized_phases, optimization_steps, error = future.result()
                except Exception as e:
                    self.logger.error(f"Unhandled exception for optimizer {optimizer_tuple[0]} - {optimizer_tuple[1]}: {e}")
                    continue

                if error:
                    self.logger.error(f"Error training optimizer '{optimizer_name}' for '{label}': {error}")
                    print(f"Error training optimizer '{optimizer_name}' for '{label}': {error}")
                    continue

                self.logger.info(f"Optimization complete for '{label}' with optimizer '{optimizer_name}'. Final loss: {optimization_steps[-1]['loss']}")
                print(f"Optimization complete for '{label}' with optimizer '{optimizer_name}'. Final loss: {optimization_steps[-1]['loss']}")

                self.write_optimization_to_train_json(optimizer_name=optimizer_name, activation_label=label, optimization_steps=optimization_steps)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        self.logger.info(f"Training finished at {end_time.strftime('%Y-%m-%d %H:%M:%S')}. Duration: {duration} seconds.")
        print(f"Training finished at {end_time.strftime('%Y-%m-%d %H:%M:%S')}. Duration: {duration} seconds.")

        self.logger.info("++++++++++++++++++++++++++++++ Thread Analysis +++++++++++++")
        self.logger.info(f"Number of processes used: {max_workers}")
        self.logger.info(f"Total computation time: {duration} seconds.")
        self.logger.info("++++++++++++++++++++++++++++++ Thread Analysis +++++++++++++")
        print("++++++++++++++++++++++++++++++ Thread Analysis +++++++++++++")
        print(f"Number of processes used: {max_workers}")
        print(f"Total computation time: {duration} seconds.")
        print("++++++++++++++++++++++++++++++ Thread Analysis +++++++++++++")

        self.logger.info("All optimizers successfully trained.")
        print("All optimizers successfully trained.")
