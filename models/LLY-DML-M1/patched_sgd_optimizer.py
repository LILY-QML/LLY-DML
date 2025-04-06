# sgd_optimizer.py (patched version)

from src.optimizers.base_optimizer import BaseOptimizer
import logging
import numpy as np

class SGDOptimizer(BaseOptimizer):
    """
    Implementation des einfachen Stochastic Gradient Descent (SGD) Optimierers.
    
    Diese Klasse implementiert den Basisalgorithmus für stochastische Gradientenverfahren,
    ohne zusätzliche Funktionen wie Momentum, Adaptive Learning Rates oder Regularisierung.
    """
    def __init__(self, data, training_matrix, target_state, learning_rate=0.01, max_iterations=100):
        super().__init__(data, training_matrix, target_state, learning_rate, max_iterations)
        # Initialisiere tuning_parameters mit der flachen Trainingsmatrix
        self.tuning_parameters = np.array(training_matrix)

    def optimize(self):
        """
        Führt den SGD-Optimierungsprozess durch.
        
        Returns:
            tuple: (optimierte Parameter, Optimierungsschritte)
        """
        logging.info("SGDOptimizer: Startet den Optimierungsprozess")
        optimization_steps = []
        
        for iteration in range(1, self.max_iterations + 1):
            # Berechne den aktuellen Verlust
            loss = self.evaluate()
            optimization_steps.append({"iteration": iteration, "loss": loss})
            logging.debug(f"SGDOptimizer: Iteration {iteration}, Verlust: {loss}")
            
            # Frühzeitiger Abbruch, wenn der Verlust bereits minimal ist
            if loss == 0:
                logging.info("SGDOptimizer: Verlust unter dem Schwellenwert, Optimierung abgeschlossen")
                break
            
            # Berechne den Gradienten
            gradient = self.compute_gradient()
            
            # Einfacher SGD-Update-Schritt
            self.tuning_parameters -= self.learning_rate * gradient
            
            logging.debug(f"SGDOptimizer: Tuning Parameters nach Update: {self.tuning_parameters}")
        
        logging.info("SGDOptimizer: Optimierungsprozess abgeschlossen")
        return self.tuning_parameters, optimization_steps