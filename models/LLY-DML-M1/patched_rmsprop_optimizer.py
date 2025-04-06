# rmsprop_optimizer.py (patched version)

from src.optimizers.base_optimizer import BaseOptimizer
import logging
import numpy as np

class RMSpropOptimizer(BaseOptimizer):
    """
    Implementation des RMSprop-Optimierers.
    
    RMSprop ist ein adaptiver Lernraten-Optimierer, der die Lernrate für jeden Parameter
    basierend auf einem gleitenden Durchschnitt der quadrierten Gradienten anpasst.
    """
    def __init__(self, data, training_matrix, target_state, learning_rate=0.001, max_iterations=100, decay_rate=0.9, epsilon=1e-8):
        super().__init__(data, training_matrix, target_state, learning_rate, max_iterations)
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        # Initialisiere tuning_parameters mit der flachen Trainingsmatrix
        self.tuning_parameters = np.array(training_matrix)
        # Initialisiere square_gradient mit den gleichen Dimensionen wie tuning_parameters
        self.square_gradient = np.zeros_like(self.tuning_parameters)

    def optimize(self):
        """
        Führt den RMSprop-Optimierungsprozess durch.
        
        Returns:
            tuple: (optimierte Parameter, Optimierungsschritte)
        """
        logging.info("RMSpropOptimizer: Startet den Optimierungsprozess")
        optimization_steps = []
        
        for iteration in range(1, self.max_iterations + 1):
            # Berechne den aktuellen Verlust
            loss = self.evaluate()
            optimization_steps.append({"iteration": iteration, "loss": loss})
            logging.debug(f"RMSpropOptimizer: Iteration {iteration}, Verlust: {loss}")
            
            # Frühzeitiger Abbruch, wenn der Verlust bereits minimal ist
            if loss == 0:
                logging.info("RMSpropOptimizer: Verlust unter dem Schwellenwert, Optimierung abgeschlossen")
                break
            
            # Berechne den Gradienten
            gradient = self.compute_gradient()
            
            # RMSprop-Update
            self.square_gradient = self.decay_rate * self.square_gradient + (1 - self.decay_rate) * (gradient ** 2)
            adjusted_learning_rate = self.learning_rate / (np.sqrt(self.square_gradient) + self.epsilon)
            self.tuning_parameters -= adjusted_learning_rate * gradient
            
            logging.debug(f"RMSpropOptimizer: Tuning Parameters nach Update: {self.tuning_parameters}")
        
        logging.info("RMSpropOptimizer: Optimierungsprozess abgeschlossen")
        return self.tuning_parameters, optimization_steps