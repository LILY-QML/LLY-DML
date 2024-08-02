import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from qiskit.visualization import plot_histogram, circuit_drawer

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
from module.visual import Visual

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
activation_matrices = config['activation_matrices']
max_iterations = config['max_iterations']
optimizers_to_run = config['optimizers']

# Function to generate random training phases
def generate_random_training_phases(qubits, depth):
    return np.random.uniform(0, 2 * np.pi, (depth * 3, qubits)).tolist()

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

# Generate and store the initial random training phases matrix
initial_training_phases = generate_random_training_phases(qubits, depth)
print("Initial Training Phases Matrix:")
print(initial_training_phases)

# Loop through each activation matrix
for i, activation_phases in enumerate(activation_matrices):
    print(f"Optimizing for Activation Matrix {i+1}")

    # Reuse the initial training phases for each activation matrix
    training_phases = initial_training_phases

    # Create a new circuit for each activation phase
    circuit = Circuit(
        qubits=qubits,
        depth=depth,
        training_phases=training_phases,
        activation_phases=activation_phases,
        shots=shots
    )

    # Run initial measurement to determine the most likely state
    initial_result = circuit.run()
    initial_counts = initial_result.get_counts()
    initial_distribution = circuit.get_counts()

    # Determine the most likely target state from the initial measurement
    target_state = max(initial_distribution, key=initial_distribution.get)
    print(f"Initial most likely target state for Activation Matrix {i+1}: {target_state}")

    # Store target states to ensure they are unique
    existing_target_states = [result['Target State'] for result in all_results]
    while target_state in existing_target_states:
        # Remove the current target state from distribution and find the next most likely state
        initial_distribution.pop(target_state)
        target_state = max(initial_distribution, key=initial_distribution.get)
        print(f"Adjusted target state to avoid duplicates: {target_state}")

    # Initial visualization
    circuit_image_path = os.path.join("var", f"circuit_initial_{i+1}.png")
    circuit_drawer(circuit.circuit, output='mpl', filename=circuit_image_path)

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
            phase_shape = np.array(training_phases).shape
            flat_phases_size = np.prod(phase_shape)
            bounds = [(0, 2 * np.pi)] * flat_phases_size  # Define bounds correctly
            optimizer = optimizer_class(
                circuit, 
                target_state, 
                learning_rate, 
                max_iterations,
                bounds=bounds  # Pass bounds correctly
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
        final_counts = circuit.run().get_counts()
        final_distribution = optimizer.get_distribution(final_counts)

        # Record the probability of the target state
        initial_probability = initial_distribution.get(target_state, 0)
        final_probability = final_distribution.get(target_state, 0)

        # Append the results
        all_results.append({
            "Activation Matrix": i + 1,
            "Target State": target_state,
            "Optimizer": optimizer_name,
            "Initial Probability": initial_probability,
            "Final Probability": final_probability,
            "Losses": losses,
            "Initial Counts": initial_counts,
            "Final Counts": final_counts,
            "Optimized Phases": optimized_phases
        })

        # Display the optimized training phases
        print("Optimized Training Phases:")
        for phase in optimized_phases:
            print(phase)

        print("\n" + "="*60 + "\n")

# Convert results to DataFrame for analysis
results_df = pd.DataFrame(all_results)

# Display the final results
print(results_df)

# Plotting the results in a table
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('tight')
ax.axis('off')
ax.table(cellText=results_df.values, colLabels=results_df.columns, cellLoc='center', loc='center')
ax.set_title("Optimization Results Summary", fontsize=16)
plt.show()

# Use Visual class to generate detailed visualizations and comparison
visual = Visual(
    results=all_results,
    target_states=[result['Target State'] for result in all_results],
    initial_training_phases=[result['Initial Counts'] for result in all_results],
    optimized_training_phases=[result['Optimized Phases'] for result in all_results],
    activation_matrices=activation_matrices,
    loss_data=[result['Losses'] for result in all_results],
    circuits=[circuit for _ in all_results],
    num_iterations=max_iterations,
    qubits=qubits,
    depth=depth
)

# Generate the report without PDF generation for now
visual.create_report()

# Optionally save the updated configuration back to JSON if any changes were made
# save_config(json_path, config)
