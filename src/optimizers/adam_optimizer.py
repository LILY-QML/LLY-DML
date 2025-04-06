# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Project: LILY-QML
# Version: 2.0.0 LLY-DML
# Author: Leon Kaiser
# Contact: info@lilyqml.de
# Website: www.lilyqml.de
# Contributors:
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++

from .base_optimizer import BaseOptimizer
import logging
import numpy as np

class AdamOptimizer(BaseOptimizer):
    """
    Implementation des Adam-Optimizers (Adaptive Moment Estimation).
    
    Adam ist ein Gradientenabstiegsverfahren, das adaptive Lernraten für verschiedene Parameter
    unterstützt. Es kombiniert die Vorteile von AdaGrad und RMSProp durch die Nutzung von sowohl
    dem ersten Moment (Mittelwert) als auch dem zweiten Moment (Varianz) des Gradienten.
    """
    def __init__(self, data, training_matrix, target_state, learning_rate=0.001, max_iterations=100, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Initialisiert den Adam Optimizer mit den notwendigen Parametern.
        
        Args:
            data (dict): Konfigurationsdaten
            training_matrix (list): Die Trainingsmatrix mit [qubits × depth × 3] Parametern
            target_state (str): Der Zielzustand, der erreicht werden soll (ein einzelnes Bit '0' oder '1')
            learning_rate (float): Lernrate für die Optimierung
            max_iterations (int): Maximale Anzahl an Iterationen
            beta1 (float): Exponentieller Abklingfaktor für den Mittelwert des Gradienten
            beta2 (float): Exponentieller Abklingfaktor für die Varianz des Gradienten
            epsilon (float): Kleine Konstante zur Vermeidung von Division durch Null
        """
        super().__init__(data, training_matrix, target_state, learning_rate, max_iterations)
        
        # Lade Parameter aus den Konfigurationsdaten, falls vorhanden
        self.beta1 = data.get("beta1", beta1) if data else beta1
        self.beta2 = data.get("beta2", beta2) if data else beta2
        self.epsilon = data.get("epsilon", epsilon) if data else epsilon
        
        # Initialisiere Momentvektoren für den flachen Parametervektor
        params = np.array(training_matrix).flatten()
        self.m = np.zeros_like(params)
        self.v = np.zeros_like(params)

    def optimize(self):
        """
        Führt den Adam-Optimierungsprozess durch.
        
        Returns:
            tuple: (Optimierte Parameter, Liste der Optimierungsschritte)
        """
        self.logger.info("AdamOptimizer: Startet den Optimierungsprozess")
        optimization_steps = []
        
        params = np.array(self.training_matrix).flatten()  # Aktuelle Parameter
        
        for iteration in range(1, self.max_iterations + 1):
            # Berechne den aktuellen Verlust
            loss = self.evaluate()
            optimization_steps.append({"iteration": iteration, "loss": loss})
            self.logger.debug(f"AdamOptimizer: Iteration {iteration}, Verlust: {loss}")

            if loss <= 0.1:  # Abbruchkriterium: Verlust unter Schwellenwert
                self.logger.info("AdamOptimizer: Verlust unter dem Schwellenwert, Optimierung abgeschlossen")
                break

            # Berechne den Gradienten
            gradient = self.compute_gradient()

            # Adam-Optimierung
            self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
            self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient ** 2)
            
            # Bias-Korrektur
            m_hat = self.m / (1 - self.beta1 ** iteration)
            v_hat = self.v / (1 - self.beta2 ** iteration)
            
            # Parameter-Update
            update = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            params = params - update
            
            # Aktualisiere Trainingsmatrix
            self.training_matrix = params.tolist()
            
            self.logger.debug(f"AdamOptimizer: Parameter nach Update aktualisiert")

        self.logger.info("AdamOptimizer: Optimierungsprozess abgeschlossen")
        return self.training_matrix, optimization_steps