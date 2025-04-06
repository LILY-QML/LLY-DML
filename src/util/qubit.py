# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Project: LILY-QML
# Version: 2.0.0 LLY-DML
# Author: Joan Pujol, Leon Kaiser
# Contact: info@lilyqml.de
# Website: www.lilyqml.de
# Contributors:
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++

class Qubit:
    def __init__(self, qubit_number):
        """
        Initialisiert ein Qubit-Objekt mit einer Qubit-Nummer.

        Args:
            qubit_number (int): Die Nummer des Qubits im Circuit
        """
        self.loss_function = None
        self.target_state = None
        self.state = None
        self.qubit_number = qubit_number
        self.training_matrix = None
        self.actual_distribution = None

    def load_training_matrix(self, training_matrix):
        """
        Lädt die Trainingsmatrix in die Qubit-Instanz.

        Args:
            training_matrix (str): Die Trainingsmatrix als String
        """
        self.training_matrix = training_matrix

    def load_actual_distribution(self, actual_distribution):
        """
        Lädt die aktuelle Verteilung in die Qubit-Instanz.

        Args:
            actual_distribution (str): Die aktuelle Verteilung als String
        """
        self.actual_distribution = actual_distribution

    def load_state(self, state):
        """
        Lädt den Zustand in die Qubit-Instanz.

        Args:
            state (str): Der zu ladende Zustand
        """
        self.state = state

    def read_state(self):
        """
        Liest den Zustand aus der Qubit-Instanz.

        Returns:
            str: Der aktuelle Zustand
        """
        return self.state

    def read_training_matrix(self):
        """
        Liest die Trainingsmatrix aus der Qubit-Instanz.

        Returns:
            str: Die aktuelle Trainingsmatrix
        """
        return self.training_matrix
    
    def read_actual_distribution(self):
        """
        Liest die aktuelle Verteilung aus der Qubit-Instanz.

        Returns:
            str: Die aktuelle Verteilung
        """
        return self.actual_distribution

    def load_function(self, loss_function):
        """
        Lädt die Verlustfunktion in die Qubit-Instanz.

        Args:
            loss_function (float): Der Wert der Verlustfunktion
        """
        self.loss_function = loss_function

    def load_target_state(self, target_state):
        """
        Lädt den Zielzustand in die Qubit-Instanz.

        Args:
            target_state (str): Der zu ladende Zielzustand
        """
        self.target_state = target_state

    def read_function(self):
        """
        Liest die Verlustfunktion aus der Qubit-Instanz.

        Returns:
            float: Der aktuelle Wert der Verlustfunktion
        """
        return self.loss_function

    def read_target_state(self):
        """
        Liest den Zielzustand aus der Qubit-Instanz.

        Returns:
            str: Der aktuelle Zielzustand
        """
        return self.target_state

    def read_qubit_number(self):
        """
        Gibt die Qubit-Nummer zurück.

        Returns:
            int: Die Nummer des Qubits
        """
        return self.qubit_number