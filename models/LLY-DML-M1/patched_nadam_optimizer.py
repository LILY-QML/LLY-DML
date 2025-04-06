# nadam_optimizer.py (patched version)

from src.optimizers.base_optimizer import BaseOptimizer
import logging
import numpy as np

class NadamOptimizer(BaseOptimizer):
    """
    Implementation des Nadam-Optimierers (NAG + Adam).
    
    Nadam kombiniert die Vorteile von Nesterov Accelerated Gradient (NAG) und Adam,
    um eine verbesserte Konvergenz zu erreichen, indem sowohl Momentum als auch
    adaptive Lernraten genutzt werden.
    """
    def __init__(self, data, training_matrix, target_state, learning_rate=0.001, max_iterations=100, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(data, training_matrix, target_state, learning_rate, max_iterations)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        # Initialisiere tuning_parameters mit der flachen Trainingsmatrix
        self.tuning_parameters = np.array(training_matrix)
        # Initialisiere m und v mit den gleichen Dimensionen wie tuning_parameters
        self.m = np.zeros_like(self.tuning_parameters)  # Erster Moment (Momentum)
        self.v = np.zeros_like(self.tuning_parameters)  # Zweiter Moment (RMSProp Teil)

    def optimize(self):
        """
        Führt den Nadam-Optimierungsprozess durch.
        
        Returns:
            tuple: (optimierte Parameter, Optimierungsschritte)
        """
        logging.info("NadamOptimizer: Startet den Optimierungsprozess")
        optimization_steps = []
        
        for iteration in range(1, self.max_iterations + 1):
            # Berechne den aktuellen Verlust
            loss = self.evaluate()
            optimization_steps.append({"iteration": iteration, "loss": loss})
            logging.debug(f"NadamOptimizer: Iteration {iteration}, Verlust: {loss}")
            
            # Frühzeitiger Abbruch, wenn der Verlust bereits minimal ist
            if loss == 0:
                logging.info("NadamOptimizer: Verlust unter dem Schwellenwert, Optimierung abgeschlossen")
                break
            
            # Berechne den Gradienten
            gradient = self.compute_gradient()
            
            # Nadam-Update
            self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
            self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient ** 2)
            
            # Bias-korrigierte Momente
            m_hat = self.m / (1 - self.beta1 ** iteration)
            v_hat = self.v / (1 - self.beta2 ** iteration)
            
            # Nesterov-Moment
            m_nesterov = self.beta1 * m_hat + (1 - self.beta1) * gradient / (1 - self.beta1 ** iteration)
            
            # Update
            self.tuning_parameters -= self.learning_rate * m_nesterov / (np.sqrt(v_hat) + self.epsilon)
            
            logging.debug(f"NadamOptimizer: Tuning Parameters nach Update: {self.tuning_parameters}")
        
        logging.info("NadamOptimizer: Optimierungsprozess abgeschlossen")
        return self.tuning_parameters, optimization_steps