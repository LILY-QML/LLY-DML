# optimizers/nadam_optimizer.py
from .base_optimizer import BaseOptimizer
import logging
import numpy as np

class NadamOptimizer(BaseOptimizer):
    """
    Implementation des Nadam Optimierers.
    """
    def __init__(self, data, training_matrix, target_state, learning_rate=0.002, max_iterations=100, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(data, training_matrix, target_state, learning_rate, max_iterations)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = np.zeros_like(self.tuning_parameters)
        self.v = np.zeros_like(self.tuning_parameters)
    
    def optimize(self):
        logging.info("NadamOptimizer: Startet den Optimierungsprozess")
        optimization_steps = []
        for iteration in range(1, self.max_iterations + 1):
            loss = self.evaluate()
            optimization_steps.append({"iteration": iteration, "loss": loss})
            logging.debug(f"NadamOptimizer: Iteration {iteration}, Verlust: {loss}")

            if loss < 1e-6:
                logging.info("NadamOptimizer: Verlust unter dem Schwellenwert, Optimierung abgeschlossen")
                break

            # Beispielhafte Gradient-Berechnung (hier zufällig, in der Praxis sollte dies die Ableitung des Verlusts sein)
            gradient = np.random.randn(3)  # Placeholder: Zufälliger Gradient

            # Nadam-Optimierungsschritt
            self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
            self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient ** 2)
            m_hat = self.m / (1 - self.beta1 ** iteration)
            v_hat = self.v / (1 - self.beta2 ** iteration)
            update = self.learning_rate * (self.beta1 * m_hat + (1 - self.beta1) * gradient) / (np.sqrt(v_hat) + self.epsilon)
            self.tuning_parameters -= update  # Update der Tuning-Parameter

            logging.debug(f"NadamOptimizer: Tuning Parameters nach Update: {self.tuning_parameters}")

        logging.info("NadamOptimizer: Optimierungsprozess abgeschlossen")
        return self.tuning_parameters, optimization_steps
