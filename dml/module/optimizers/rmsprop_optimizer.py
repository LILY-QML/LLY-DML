# optimizers/rmsprop_optimizer.py
from .base_optimizer import BaseOptimizer
import logging
import numpy as np

class RMSPropOptimizer(BaseOptimizer):
    """
    Implementation des RMSProp Optimierers.
    """
    def __init__(self, data, training_matrix, target_state, learning_rate=0.001, max_iterations=100, beta=0.9, epsilon=1e-8):
        super().__init__(data, training_matrix, target_state, learning_rate, max_iterations)
        self.beta = beta
        self.epsilon = epsilon
        self.Eg = np.zeros_like(self.tuning_parameters)
    
    def optimize(self):
        logging.info("RMSPropOptimizer: Startet den Optimierungsprozess")
        optimization_steps = []
        for iteration in range(1, self.max_iterations + 1):
            loss = self.evaluate()
            optimization_steps.append({"iteration": iteration, "loss": loss})
            logging.debug(f"RMSPropOptimizer: Iteration {iteration}, Verlust: {loss}")

            if loss < 1e-6:
                logging.info("RMSPropOptimizer: Verlust unter dem Schwellenwert, Optimierung abgeschlossen")
                break

            # Beispielhafte Gradient-Berechnung (hier zufällig, in der Praxis sollte dies die Ableitung des Verlusts sein)
            gradient = np.random.randn(3)  # Placeholder: Zufälliger Gradient

            # RMSProp-Optimierungsschritt
            self.Eg = self.beta * self.Eg + (1 - self.beta) * (gradient ** 2)
            update = (self.learning_rate * gradient) / (np.sqrt(self.Eg) + self.epsilon)
            self.tuning_parameters -= update  # Update der Tuning-Parameter

            logging.debug(f"RMSPropOptimizer: Tuning Parameters nach Update: {self.tuning_parameters}")

        logging.info("RMSPropOptimizer: Optimierungsprozess abgeschlossen")
        return self.tuning_parameters, optimization_steps
