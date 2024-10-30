# base_optimizer.py

import numpy as np

class BaseOptimizer:
    """
    Basisklasse für Optimizer. Definiert die grundlegenden Methoden, die jeder Optimizer implementieren sollte.
    """
    def __init__(self, data, training_matrix, target_state, learning_rate=0.001, max_iterations=100):
        self.data = data
        self.training_matrix = training_matrix
        self.target_state = target_state
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tuning_parameters = np.random.rand(3)  # Drei Tuning-Parameter initialisiert zufällig

    def evaluate(self):
        """
        Bewertet die aktuelle Matrix im Vergleich zum Zielzustand.
        Gibt einen Verlustwert zurück.
        """
        return self.calculate_loss()

    def calculate_loss(self):
        """
        Berechnet den Verlust basierend auf den Tuning-Parametern und dem Zielzustand.
        """
        loss = 0
        for i, param in enumerate(self.tuning_parameters):
            desired_bit = int(self.target_state[i])  # gewünschtes Bit (0 oder 1)
            current_bit = int(round(param)) % 2      # aktuelles Bit aus den Parametern
            if current_bit != desired_bit:
                loss += 1
        return loss

    def compute_gradient(self):
        """
        Berechnet den Gradienten der Verlustfunktion nach den Tuning-Parametern.
        Nutzt eine numerische Approximation.
        """
        gradients = np.zeros_like(self.tuning_parameters)
        for i in range(len(self.tuning_parameters)):
            # Temporäre Parameter, um den Einfluss von Tuning-Parameter i zu bewerten
            temp_params = self.tuning_parameters.copy()
            temp_params[i] += 1e-5  # Kleiner Schritt nach oben
            loss_plus = self.calculate_loss_specific(temp_params)
            temp_params[i] -= 2e-5  # Kleiner Schritt nach unten
            loss_minus = self.calculate_loss_specific(temp_params)
            gradients[i] = (loss_plus - loss_minus) / (2 * 1e-5)  # Numerische Ableitung
        return gradients

    def calculate_loss_specific(self, tuning_parameters):
        """
        Berechnet den Verlust basierend auf den gegebenen Tuning-Parametern und dem Zielzustand.
        Wird für die Gradientenberechnung genutzt.
        """
        loss = 0
        for i, param in enumerate(tuning_parameters):
            desired_bit = int(self.target_state[i])  # gewünschtes Bit (0 oder 1)
            current_bit = int(round(param)) % 2      # aktuelles Bit aus den Parametern
            if current_bit != desired_bit:
                loss += 1
        return loss

    def optimize(self):
        """
        Führt den Optimierungsprozess durch. Muss von den abgeleiteten Klassen implementiert werden.
        """
        raise NotImplementedError("Die Methode optimize muss von der abgeleiteten Klasse implementiert werden.")
