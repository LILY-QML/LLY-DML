# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Project: LILY-QML
# Version: 2.0.0 LLY-DML
# Author: Leon Kaiser
# Contact: info@lilyqml.de
# Website: www.lilyqml.de
# Contributors:
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class MatrixEvaluation:
    
    def __init__(self, results):
        # Load the data from results (which is a list of dictionaries)
        self.data = pd.DataFrame(results)

    def evaluate_matrices(self):
        # For each optimizer, group by activation matrix and get relevant details
        evaluations = {}

        for optimizer, group in self.data.groupby('Optimizer'):
            optimizer_evaluations = []

            for activation_matrix, matrix_group in group.groupby('Activation Matrix'):
                # Extract target state and its probability
                target_state = matrix_group.iloc[0]['Target State']
                target_probability = matrix_group.iloc[0]['Final Probability']

                # Sort the states by their probabilities in descending order
                counts = matrix_group.iloc[0]['Final Counts']
                sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)

                # Get top 4 states (target state and next three)
                top_states = sorted_counts[:4]
                state_names = [state for state, _ in top_states]
                state_probs = [prob / sum(counts.values()) for _, prob in top_states]

                # Determine if the target state is clearly distinguishable
                distinguishable = all(np.abs(state_probs[0] - p) > 0.1 for p in state_probs[1:])

                # Create evaluation summary for the activation matrix
                evaluation_summary = {
                    'Activation Matrix': activation_matrix,
                    'Target State': target_state,
                    'Target Probability': state_probs[0],
                    'Next States': state_names[1:],
                    'Next Probabilities': state_probs[1:],
                    'Clearly Distinguishable': distinguishable
                }

                optimizer_evaluations.append(evaluation_summary)

            evaluations[optimizer] = optimizer_evaluations

        return evaluations

    def plot_evaluation_summary(self, evaluations):
        # Plot a table for each optimizer's evaluation
        for optimizer, optimizer_data in evaluations.items():
            df = pd.DataFrame(optimizer_data)

            # Plotting the table
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.axis('tight')
            ax.axis('off')
            ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
            ax.set_title(f"Evaluation Summary for Optimizer: {optimizer}", fontsize=16)
            plt.tight_layout()
            plt.show()