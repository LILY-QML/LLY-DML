# adagrad_optimizer.py (patched version)

from src.optimizers.base_optimizer import BaseOptimizer
import logging
import numpy as np

class AdagradOptimizer(BaseOptimizer):
    """
    Implementation des Adagrad-Optimierers.
    
    Adagrad passt die Lernrate für jeden Parameter individuell an,
    basierend auf der Summe der vergangenen quadrierten Gradienten.
    Dies ist besonders nützlich für dünn besetzte Daten oder Gradienten.
    """
    def __init__(self, data, training_matrix, target_state, learning_rate=0.01, max_iterations=100, epsilon=1e-8):
        super().__init__(data, training_matrix, target_state, learning_rate, max_iterations)
        self.epsilon = epsilon
        # Initialisiere tuning_parameters mit der flachen Trainingsmatrix
        self.tuning_parameters = np.array(training_matrix)
        # Initialisiere gradient_accumulation mit den gleichen Dimensionen wie tuning_parameters
        self.gradient_accumulation = np.zeros_like(self.tuning_parameters)

    def optimize(self):
        """
        Führt den Adagrad-Optimierungsprozess durch.
        
        Returns:
            tuple: (optimierte Parameter, Optimierungsschritte)
        """
        logging.info("AdagradOptimizer: Startet den Optimierungsprozess")
        optimization_steps = []
        
        for iteration in range(1, self.max_iterations + 1):
            # Berechne den aktuellen Verlust
            loss = self.evaluate()
            optimization_steps.append({"iteration": iteration, "loss": loss})
            logging.debug(f"AdagradOptimizer: Iteration {iteration}, Verlust: {loss}")
            
            # Frühzeitiger Abbruch, wenn der Verlust bereits minimal ist
            if loss == 0:
                logging.info("AdagradOptimizer: Verlust unter dem Schwellenwert, Optimierung abgeschlossen")
                break
            
            # Berechne den Gradienten
            gradient = self.compute_gradient()
            
            # Adagrad-Update
            self.gradient_accumulation += gradient ** 2
            adjusted_learning_rate = self.learning_rate / (np.sqrt(self.gradient_accumulation) + self.epsilon)
            self.tuning_parameters -= adjusted_learning_rate * gradient
            
            logging.debug(f"AdagradOptimizer: Tuning Parameters nach Update: {self.tuning_parameters}")
        
        logging.info("AdagradOptimizer: Optimierungsprozess abgeschlossen")
        return self.tuning_parameters, optimization_steps