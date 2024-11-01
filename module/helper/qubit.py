class Qubit:
    def __init__(self, qubit_number):
        self.loss_function = None
        self.target_state = None
        self.qubit_number = qubit_number


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