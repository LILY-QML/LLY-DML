# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Project: LILY-QML
# Version: 1.6 LLY-DML
# Author: Leon Kaiser
# Contact: info@lilyqml.de
# Website: www.lilyqml.de
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import logging

class MatrixEvaluation:
    
    def __init__(self, results, logger=None):
        """
        Initializes the MatrixEvaluation class and loads the data.

        :param results: list of dictionaries containing results.
        :param logger: logging.Logger, optional logger for logging messages.
        """
        self.data = pd.DataFrame(results)
        self.logger = logger

        if self.logger:
            self.logger.info("MatrixEvaluation initialized with results data.")

    def evaluate_matrices(self):
        """
        Evaluates the matrices by grouping data by optimizer and activation matrix.

        :return: Dictionary with evaluations for each optimizer.
        """
        evaluations = {}

        for optimizer, group in self.data.groupby('Optimizer'):
            optimizer_evaluations = []
            if self.logger:
                self.logger.info(f"Evaluating matrices for Optimizer: {optimizer}")

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

                # Log evaluation details
                if self.logger:
                    self.logger.debug(f"Optimizer: {optimizer}, Activation Matrix: {activation_matrix}")
                    self.logger.debug(f"Target State: {target_state}, Target Probability: {state_probs[0]}")
                    self.logger.debug(f"Next States: {state_names[1:]}, Next Probabilities: {state_probs[1:]}")
                    self.logger.debug(f"Clearly Distinguishable: {distinguishable}")

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

        if self.logger:
            self.logger.info("Matrix evaluations completed.")

        return evaluations

    def plot_evaluation_summary(self, evaluations):
        """
        Plots the evaluation summary for each optimizer as a table.

        :param evaluations: Dictionary containing evaluations for each optimizer.
        """
        for optimizer, optimizer_data in evaluations.items():
            df = pd.DataFrame(optimizer_data)
            if self.logger:
                self.logger.info(f"Plotting evaluation summary for Optimizer: {optimizer}")

            # Plotting the table
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.axis('tight')
            ax.axis('off')
            ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
            ax.set_title(f"Evaluation Summary for Optimizer: {optimizer}", fontsize=16)
            plt.tight_layout()
            plt.show()

            if self.logger:
                self.logger.info(f"Plot for Optimizer {optimizer} displayed.")
