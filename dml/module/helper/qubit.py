# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Project: LILY-QML
# Version: 2.0.0 LLY-DML
# Author: Joan Pujol
# Contact: info@lilyqml.de
# Website: www.lilyqml.de
# Contributors:
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
class Qubit:
    def __init__(self, qubit_number):
        self.loss_function = None
        self.target_state = None
        self.qubit_number = qubit_number
        self.training_matrix = None
        self.actual_distribution = None

    def load_training_matrix(self, training_matrix):
        """
        Loads the training matrix to the Qubit instance.
        """
        self.training_matrix = training_matrix

    def load_actual_distribution(self, actual_distribution):
        """
        Loads the actual distribution to the Qubit instance.
        """
        self.actual_distribution = actual_distribution

    def read_training_matrix(self):
        """
        Reads the training matrix from the Qubit instance.
        """
        return self.training_matrix
    
    def read_actual_distribution(self):
        """
        Reads the actual distribution from the Qubit instance.
        """
        return self.actual_distribution

    def load_function(self, loss_function):
        """
        Loads the loss function to the Qubit instance.
        """
        self.loss_function = loss_function

    def load_target_state(self, target_state):
        """
        Loads the target state to the Qubit instance.
        """
        self.target_state = target_state

    def read_function(self):
        """
        Reads the loss function from the Qubit instance.
        """
        return self.loss_function

    def read_target_state(self):
        """
        Reads the target state from the Qubit instance.
        """
        return self.target_state

    def read_qubit_number(self):
        """
        Returns the qubit number.
        """
        return self.qubit_number