# momentum_optimizer.py (patched version)

from src.optimizers.base_optimizer import BaseOptimizer
import logging
import numpy as np

class MomentumOptimizer(BaseOptimizer):
    """
    Implementation des Momentum-Optimierers.
    
    Der Momentum-Optimierer erweitert SGD durch Hinzufügen eines Momentum-Terms,
    der verhindert, dass der Optimierungsprozess in lokalen Minima stecken bleibt und
    die Konvergenz in Richtungen mit konsistenten Gradienten beschleunigt.
    """
    def __init__(self, data, training_matrix, target_state, learning_rate=0.01, max_iterations=100, momentum=0.9):
        super().__init__(data, training_matrix, target_state, learning_rate, max_iterations)
        self.momentum = momentum
        # Initialisiere tuning_parameters mit der flachen Trainingsmatrix
        self.tuning_parameters = np.array(training_matrix)
        # Initialisiere velocity mit den gleichen Dimensionen wie tuning_parameters
        self.velocity = np.zeros_like(self.tuning_parameters)

    def optimize(self):
        """
        Führt den Momentum-Optimierungsprozess durch.
        
        Returns:
            tuple: (optimierte Parameter, Optimierungsschritte)
        """
        logging.info("MomentumOptimizer: Startet den Optimierungsprozess")
        optimization_steps = []
        
        for iteration in range(1, self.max_iterations + 1):
            # Berechne den aktuellen Verlust
            loss = self.evaluate()
            optimization_steps.append({"iteration": iteration, "loss": loss})
            logging.debug(f"MomentumOptimizer: Iteration {iteration}, Verlust: {loss}")
            
            # Frühzeitiger Abbruch, wenn der Verlust bereits minimal ist
            if loss == 0:
                logging.info("MomentumOptimizer: Verlust unter dem Schwellenwert, Optimierung abgeschlossen")
                break
            
            # Berechne den Gradienten
            gradient = self.compute_gradient()
            
            # Momentum-Update
            self.velocity = self.momentum * self.velocity - self.learning_rate * gradient
            self.tuning_parameters += self.velocity
            
            logging.debug(f"MomentumOptimizer: Tuning Parameters nach Update: {self.tuning_parameters}")
        
        logging.info("MomentumOptimizer: Optimierungsprozess abgeschlossen")
        return self.tuning_parameters, optimization_steps