# optimizers/adam_optimizer.py

from .base_optimizer import BaseOptimizer
import logging
import numpy as np

class AdamOptimizer(BaseOptimizer):
    """
    Implementation des Adam-Optimierers.
    """
    def __init__(self, data, training_matrix, target_state, learning_rate=0.001, max_iterations=100, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(data, training_matrix, target_state, learning_rate, max_iterations)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = np.zeros_like(self.tuning_parameters)
        self.v = np.zeros_like(self.tuning_parameters)

    def optimize(self):
        logging.info("AdamOptimizer: Startet den Optimierungsprozess")
        optimization_steps = []
        for iteration in range(1, self.max_iterations + 1):
            loss = self.evaluate()
            optimization_steps.append({"iteration": iteration, "loss": loss})
            logging.debug(f"AdamOptimizer: Iteration {iteration}, Verlust: {loss}")

            if loss == 0:
                logging.info("AdamOptimizer: Verlust unter dem Schwellenwert, Optimierung abgeschlossen")
                break

            # Tatsächliche Gradient-Berechnung
            gradient = self.compute_gradient()

            # Adam-Optimierungsschritt
            self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
            self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient ** 2)
            m_hat = self.m / (1 - self.beta1 ** iteration)
            v_hat = self.v / (1 - self.beta2 ** iteration)
            update = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            self.tuning_parameters -= update  # Update der Tuning-Parameter

            logging.debug(f"AdamOptimizer: Tuning Parameters nach Update: {self.tuning_parameters}")

        logging.info("AdamOptimizer: Optimierungsprozess abgeschlossen")
        return self.tuning_parameters, optimization_steps
