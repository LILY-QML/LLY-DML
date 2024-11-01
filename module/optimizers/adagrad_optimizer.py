# optimizers/adagrad_optimizer.py
from .base_optimizer import BaseOptimizer
import logging
import numpy as np

class AdaGradOptimizer(BaseOptimizer):
    """
    Implementation des AdaGrad Optimierers.
    """
    def __init__(self, data, training_matrix, target_state, learning_rate=0.01, max_iterations=100, epsilon=1e-8):
        super().__init__(data, training_matrix, target_state, learning_rate, max_iterations)
        self.epsilon = epsilon
        self.G = np.zeros_like(self.tuning_parameters)
    
    def optimize(self):
        logging.info("AdaGradOptimizer: Startet den Optimierungsprozess")
        optimization_steps = []
        for iteration in range(1, self.max_iterations + 1):
            loss = self.evaluate()
            optimization_steps.append({"iteration": iteration, "loss": loss})
            logging.debug(f"AdaGradOptimizer: Iteration {iteration}, Verlust: {loss}")

            if loss < 1e-6:
                logging.info("AdaGradOptimizer: Verlust unter dem Schwellenwert, Optimierung abgeschlossen")
                break

            # Beispielhafte Gradient-Berechnung (hier zufällig, in der Praxis sollte dies die Ableitung des Verlusts sein)
            gradient = np.random.randn(3)  # Placeholder: Zufälliger Gradient

            # AdaGrad-Optimierungsschritt
            self.G += gradient ** 2
            adjusted_learning_rate = self.learning_rate / (np.sqrt(self.G) + self.epsilon)
            update = adjusted_learning_rate * gradient
            self.tuning_parameters -= update  # Update der Tuning-Parameter

            logging.debug(f"AdaGradOptimizer: Tuning Parameters nach Update: {self.tuning_parameters}")

        logging.info("AdaGradOptimizer: Optimierungsprozess abgeschlossen")
        return self.tuning_parameters, optimization_steps
