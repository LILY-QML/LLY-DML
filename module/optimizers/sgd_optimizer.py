# optimizers/sgd_optimizer.py

from .base_optimizer import BaseOptimizer
import logging
import numpy as np

class SGDOptimizer(BaseOptimizer):
    """
    Implementation des Stochastic Gradient Descent (SGD) Optimierers.
    """
    def __init__(self, data, training_matrix, target_state, learning_rate=0.01, max_iterations=100):
        super().__init__(data, training_matrix, target_state, learning_rate, max_iterations)

    def optimize(self):
        logging.info("SGDOptimizer: Startet den Optimierungsprozess")
        optimization_steps = []
        for iteration in range(1, self.max_iterations + 1):
            loss = self.evaluate()
            optimization_steps.append({"iteration": iteration, "loss": loss})
            logging.debug(f"SGDOptimizer: Iteration {iteration}, Verlust: {loss}")

            if loss == 0:
                logging.info("SGDOptimizer: Verlust unter dem Schwellenwert, Optimierung abgeschlossen")
                break

            # Tatsächliche Gradient-Berechnung
            gradient = self.compute_gradient()

            # SGD-Optimierungsschritt
            update = self.learning_rate * gradient
            self.tuning_parameters -= update  # Update der Tuning-Parameter

            logging.debug(f"SGDOptimizer: Tuning Parameters nach Update: {self.tuning_parameters}")

        logging.info("SGDOptimizer: Optimierungsprozess abgeschlossen")
        return self.tuning_parameters, optimization_steps
