import json
import os
import pandas as pd
import matplotlib.pyplot as plt

from module.circuit import Circuit
from module.optimizer import (
    Optimizer,
    OptimizerWithMomentum,
    AdamOptimizer,
    GeneticOptimizer,
    PSOOptimizer,
    BayesianOptimizer,
    SimulatedAnnealingOptimizer
)

# Function to read configuration from JSON
def load_config(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

# Load configuration from JSON
json_path = os.path.join('var', 'data.json')
config = load_config(json_path)

# Extract configuration parameters
qubits = config['qubits']
depth = config['depth']
learning_rate = config['learning_rate']
shots = config['shots']
training_phases = config['training_phases']
activation_matrices = config['activation_matrices']
target_states = config['target_states']
max_iterations = config['max_iterations']
optimizers_to_run = config['optimizers']

# List to store the results
all_results = []

# Define optimizer classes
optimizer_classes = {
    "Basic": Optimizer,
    "Momentum": OptimizerWithMomentum,
    "Adam": AdamOptimizer,
    "Genetic": GeneticOptimizer,
    "PSO": PSOOptimizer,
    "Bayesian": BayesianOptimizer,
    "SimulatedAnnealing": SimulatedAnnealingOptimizer
}

# Loop through each activation matrix and target state
for i, (activation_phases, target_state) in enumerate(zip(activation_matrices, target_states)):
    print(f"Optimizing for Activation Matrix {i+1} towards state {target_state}")

    # Create a new circuit for each activation phase
    circuit = Circuit(
        qubits=qubits,
        depth=depth,
        training_phases=training_phases,
        activation_phases=activation_phases,
        shots=shots
    )

    # Run selected optimizers
    for optimizer_name in optimizers_to_run:
        optimizer_class = optimizer_classes.get(optimizer_name)
        
        if optimizer_class is None:
            print(f"Optimizer {optimizer_name} not recognized, skipping...")
            continue

        print(f"Running optimizer: {optimizer_name}")

        # Create an optimizer instance
        if optimizer_name == "Genetic":
            optimizer = optimizer_class(
                circuit, 
                target_state, 
                learning_rate, 
                max_iterations,
                population_size=config.get('population_size', 20),
                mutation_rate=config.get('mutation_rate', 0.1)
            )
        elif optimizer_name == "PSO":
            optimizer = optimizer_class(
                circuit, 
                target_state, 
                learning_rate, 
                max_iterations,
                num_particles=config.get('num_particles', 30),
                inertia=config.get('inertia', 0.5),
                cognitive=config.get('cognitive', 1.5),
                social=config.get('social', 1.5)
            )
        elif optimizer_name == "Bayesian":
            bounds = [(0, 2*np.pi) for _ in range(qubits * depth)]  # Assuming phase bounds
            optimizer = optimizer_class(
                circuit, 
                target_state, 
                learning_rate, 
                max_iterations,
                bounds=bounds
            )
        elif optimizer_name == "SimulatedAnnealing":
            optimizer = optimizer_class(
                circuit, 
                target_state, 
                learning_rate, 
                max_iterations,
                initial_temperature=config.get('initial_temperature', 1.0),
                cooling_rate=config.get('cooling_rate', 0.99)
            )
        else:
            optimizer = optimizer_class(
                circuit, 
                target_state, 
                learning_rate, 
                max_iterations
            )

        # Optimize the circuit
        optimized_phases, losses = optimizer.optimize()

        # Run the circuit again with optimized phases
        circuit.run()
        final_counts = circuit.get_counts()
        final_distribution = optimizer.get_distribution(final_counts)

        # Record the probability of the target state
        if optimizer.initial_distribution is not None:
            initial_probability = optimizer.initial_distribution.get(target_state, 0)
        else:
            initial_probability = 0
            print("Warning: Initial distribution was None.")

        final_probability = final_distribution.get(target_state, 0)

        # Append the results
        all_results.append({
            "Activation Matrix": i + 1,
            "Target State": target_state,
            "Optimizer": optimizer_name,
            "Initial Probability": initial_probability,
            "Final Probability": final_probability
        })

        # Display the optimized training phases
        print("Optimized Training Phases:")
        for phase in optimized_phases:
            print(phase)

        print("\n" + "="*60 + "\n")

# Convert results to DataFrame for easy plotting
results_df = pd.DataFrame(all_results)
print(results_df)

# Plotting the results
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('tight')
ax.axis('off')
ax.table(cellText=results_df.values, colLabels=results_df.columns, cellLoc='center', loc='center')
ax.set_title("Activation Matrix Optimization Results", fontsize=16)
plt.show()
