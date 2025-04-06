# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Project: LILY-QML
# Version: 2.0.0 LLY-DML
# Author: Leon Kaiser
# Contact: info@lilyqml.de
# Website: www.lilyqml.de
# Contributors:
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++

import numpy as np
import logging

class BaseOptimizer:
    """
    Basisklasse für Optimizer. Definiert die grundlegenden Methoden, die jeder Optimizer implementieren sollte.
    """
    def __init__(self, data, training_matrix, target_state, learning_rate=0.001, max_iterations=100):
        """
        Initialisiert den Base Optimizer mit den notwendigen Parametern.
        
        Args:
            data (dict): Konfigurationsdaten
            training_matrix (list): Die Trainingsmatrix mit [qubits × depth × 3] Parametern, flach als Liste
            target_state (str): Der Zielzustand, der erreicht werden soll (ein einzelnes Bit '0' oder '1')
            learning_rate (float): Lernrate für die Optimierung
            max_iterations (int): Maximale Anzahl an Iterationen
        """
        self.data = data
        self.training_matrix = training_matrix
        self.target_state = target_state  # Ein einzelnes Bit '0' oder '1'
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        
        # Logger konfigurieren
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Lade spezifische Parameter aus den Konfigurationsdaten, falls vorhanden
        if data and "learning_rate" in data:
            self.learning_rate = data["learning_rate"]
    
    def evaluate(self):
        """
        Bewertet die aktuelle Matrix im Vergleich zum Zielzustand.
        Gibt einen Verlustwert zurück.
        
        Returns:
            float: Der Verlustwert
        """
        return self.calculate_loss()
    
    def calculate_loss(self):
        """
        Berechnet den Verlust basierend auf der Trainingsmatrix und dem Zielzustand.
        
        Diese Implementierung berechnet einen einfachen Verlust basierend auf der Abweichung
        zwischen dem Zielzustand und dem wahrscheinlichsten Zustand, den die aktuelle Matrix erzeugt.
        
        Returns:
            float: Der Verlustwert
        """
        # Verlustfunktion basierend auf dem Zielzustand (ein einzelnes Bit '0' oder '1')
        loss = 0
        matrix_flat = np.array(self.training_matrix)
        
        # Das Zielbit als Integer konvertieren (0 oder 1)
        desired_bit = int(self.target_state)
        
        # Für jeden Parameter in der flachen Matrix
        for i, param in enumerate(matrix_flat):
            # Wir behandeln Parameter als Wahrscheinlichkeiten
            # Kleinere Werte begünstigen 0, größere Werte begünstigen 1
            normalized_param = (np.sin(param) + 1) / 2  # Auf [0, 1] normalisieren
            
            # Abstand zum gewünschten Bit berechnen
            if desired_bit == 0:
                loss += normalized_param  # Wenn 0 gewünscht, sollte der Parameter klein sein
            else:
                loss += (1 - normalized_param)  # Wenn 1 gewünscht, sollte der Parameter groß sein
        
        # Der durchschnittliche Verlust über alle Parameter
        loss = loss / len(matrix_flat) if len(matrix_flat) > 0 else 0
        
        return loss
    
    def calculate_loss_specific(self, params):
        """
        Berechnet den Verlust für spezifische Parameter.
        
        Args:
            params (numpy.ndarray): Die zu bewertenden Parameter
            
        Returns:
            float: Der Verlustwert
        """
        # Ähnliche Implementierung wie calculate_loss, aber mit den übergebenen Parametern
        loss = 0
        desired_bit = int(self.target_state)
        
        for param in params:
            normalized_param = (np.sin(param) + 1) / 2
            
            if desired_bit == 0:
                loss += normalized_param
            else:
                loss += (1 - normalized_param)
        
        # Der durchschnittliche Verlust über alle Parameter
        loss = loss / len(params) if len(params) > 0 else 0
        
        return loss
    
    def compute_gradient(self):
        """
        Berechnet den Gradienten der Verlustfunktion nach den Trainingsparametern.
        Nutzt eine numerische Approximation.
        
        Returns:
            numpy.ndarray: Der Gradient
        """
        params = np.array(self.training_matrix).flatten()
        gradients = np.zeros_like(params)
        
        epsilon = 1e-5  # Kleine Änderung zur Berechnung der numerischen Ableitung
        
        for i in range(len(params)):
            # Temporäre Parameter, um den Einfluss von Parameter i zu bewerten
            params_plus = params.copy()
            params_plus[i] += epsilon
            
            params_minus = params.copy()
            params_minus[i] -= epsilon
            
            # Berechne den Gradienten mittels zentraler Differenz
            loss_plus = self.calculate_loss_specific(params_plus)
            loss_minus = self.calculate_loss_specific(params_minus)
            
            gradients[i] = (loss_plus - loss_minus) / (2 * epsilon)
        
        return gradients
    
    def optimize(self):
        """
        Führt den Optimierungsprozess durch. Muss von den abgeleiteten Klassen implementiert werden.
        
        Raises:
            NotImplementedError: Die Methode muss von der abgeleiteten Klasse implementiert werden.
        """
        raise NotImplementedError("Die Methode optimize muss von der abgeleiteten Klasse implementiert werden.")
