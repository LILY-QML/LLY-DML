# optimizers/momentum_optimizer.py
from .base_optimizer import BaseOptimizer
import logging
import numpy as np

class MomentumOptimizer(BaseOptimizer):
    """
    Implementation des Momentum Optimierers.
    """
    def __init__(self, data, training_matrix, target_state, learning_rate=0.01, max_iterations=100, momentum=0.9):
        super().__init__(data, training_matrix, target_state, learning_rate, max_iterations)
        self.momentum = momentum
        self.velocity = np.zeros_like(self.tuning_parameters)
    
    def optimize(self):
        logging.info("MomentumOptimizer: Startet den Optimierungsprozess")
        optimization_steps = []
        for iteration in range(1, self.max_iterations + 1):
            loss = self.evaluate()
            optimization_steps.append({"iteration": iteration, "loss": loss})
            logging.debug(f"MomentumOptimizer: Iteration {iteration}, Verlust: {loss}")

            if loss < 1e-6:
                logging.info("MomentumOptimizer: Verlust unter dem Schwellenwert, Optimierung abgeschlossen")
                break

            # Beispielhafte Gradient-Berechnung (hier zufällig, in der Praxis sollte dies die Ableitung des Verlusts sein)
            gradient = np.random.randn(3)  # Placeholder: Zufälliger Gradient

            # Momentum-Optimierungsschritt
            self.velocity = self.momentum * self.velocity - self.learning_rate * gradient
            self.tuning_parameters += self.velocity  # Update der Tuning-Parameter

            logging.debug(f"MomentumOptimizer: Tuning Parameters nach Update: {self.tuning_parameters}")

        logging.info("MomentumOptimizer: Optimierungsprozess abgeschlossen")
        return self.tuning_parameters, optimization_steps
