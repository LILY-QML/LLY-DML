# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Project: LILY-QML
# Version: 1.6.2 LLY-DML
# Author: Leon Kaiser
# Contact: info@lilyqml.de
# Website: www.lilyqml.de
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++

import numpy as np

class Data:
    def __init__(self, qubits, depth, activation_matrices, labels=None, logger=None):
        """
        Initializes the Data class with qubits, depth, activation_matrices, and optional labels and logger.

        :param qubits: int, Number of qubits, corresponds to the number of rows in the matrices.
        :param depth: int, Depth, corresponds to the number of columns in the matrices.
        :param activation_matrices: list, Contains the three-dimensional activation matrices.
        :param labels: list, Optional list of labels for the activation matrices.
        :param logger: logging.Logger, Optional logger for logging messages.
        :raises ValueError: If no matrices are provided or dimensions do not match.
        :raises TypeError: If the matrices are not NumPy arrays or have incorrect structure.
        """
        if not activation_matrices:
            raise ValueError("At least one activation matrix must be provided.")

        self.qubits = qubits  # Number of qubits, which determines matrix rows
        self.depth = depth  # Depth, corresponding to matrix columns
        self.activation_matrices = []  # List to store activation matrices
        self.labels = labels if labels else []  # Optional labels for the matrices
        self.logger = logger  # Optional logger for debugging purposes

        for idx, matrix in enumerate(activation_matrices):
            # Assign default labels if none are provided
            matrix_label = self.labels[idx] if idx < len(self.labels) else f"matrix{idx+1}"

            if not isinstance(matrix, np.ndarray):
                raise TypeError(f"The {matrix_label} is not a NumPy array.")

            if matrix.ndim != 3:
                raise ValueError(f"The {matrix_label} is not three-dimensional. It has {matrix.ndim} dimension(s).")

            # Check if matrix dimensions match the required qubits and depth
            layers, matrix_qubits, matrix_depth = matrix.shape

            if matrix_qubits != qubits:
                raise ValueError(f"The {matrix_label} has {matrix_qubits} rows, expected: {qubits} (qubits).")

            if matrix_depth != depth:
                raise ValueError(f"The {matrix_label} has {matrix_depth} columns, expected: {depth} (depth).")

            self.activation_matrices.append(matrix)
            self.labels.append(matrix_label)

        # Ensure all matrices have the same dimensions
        first_shape = self.activation_matrices[0].shape
        for matrix in self.activation_matrices:
            if matrix.shape != first_shape:
                raise ValueError("Not all activation matrices have the same dimensions.")

        self.shape = first_shape  # Store the common shape of the matrices

    def get_dimensions(self):
        """
        Returns the dimensions of each activation matrix.

        :return: A list of tuples representing the dimensions of each matrix.
        """
        return [matrix.shape for matrix in self.activation_matrices]

    def get_number_of_matrices(self):
        """
        Returns the number of activation matrices provided.

        :return: Number of matrices.
        """
        return len(self.activation_matrices)

    def summary(self):
        """
        Provides a summary of the activation matrices, including the number and dimensions.

        :return: A formatted string with the summary.
        """
        summary_str = f"Number of Activation Matrices: {self.get_number_of_matrices()}\n"
        for label, shape in zip(self.labels, self.get_dimensions()):
            summary_str += f"{label} Dimensions: {shape}\n"
        summary_str += f"Qubits: {self.qubits}, Depth: {self.depth}\n"
        return summary_str

    def create_training_matrix(self):
        """
        Creates a new 2D matrix with randomized values.
        The dimensions are (qubits, 3 * depth).

        :return: A new 2D NumPy matrix with shape (qubits, 3 * depth).
        :raises ValueError: If the resulting training matrix does not have the expected dimensions.
        """
        training_matrix = np.random.rand(self.qubits, 3 * self.depth)
        if training_matrix.shape != (self.qubits, 3 * self.depth):
            raise ValueError(f"The training matrix has dimensions {training_matrix.shape}, expected: ({self.qubits}, {3 * self.depth}).")
        return training_matrix

    def validate_training_matrix(self, training_matrix):
        """
        Validates that the training matrix has the expected dimensions.

        :param training_matrix: NumPy array, the training matrix to validate.
        :raises ValueError: If the training matrix does not have the expected dimensions.
        """
        expected_shape = (self.qubits, 3 * self.depth)
        if training_matrix.shape != expected_shape:
            raise ValueError(f"The training matrix has dimensions {training_matrix.shape}, expected: {expected_shape}.")

    def convert_activation_matrices_to_2d(self):
        """
        Converts all three-dimensional activation matrices into two-dimensional matrices.
        The layers of the 3D matrix are concatenated, resulting in a matrix of shape (qubits, depth * layers).
        """
        converted_matrices = {}
        for label, matrix in zip(self.labels, self.activation_matrices):
            try:
                # Reshape to (qubits, layers * depth)
                converted_matrix = matrix.transpose(1, 0, 2).reshape(self.qubits, -1)
                converted_matrices[label] = converted_matrix.tolist()
                if self.logger:
                    self.logger.debug(f"{label} converted matrix shape: {converted_matrix.shape}")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error converting {label}: {e}")
                raise
        return converted_matrices

    def get_matrix_names(self):
        """
        Returns the names of all activation matrices.

        :return: A list of labels of the matrices.
        """
        return self.labels.copy()

    def get_matrix_by_name(self, name):
        """
        Returns the three-dimensional activation matrix with the specified name.

        :param name: str, the name of the desired matrix.
        :return: NumPy array of the corresponding three-dimensional matrix.
        :raises KeyError: If the specified name does not exist.
        """
        if name not in self.labels:
            raise KeyError(f"Matrix with the name '{name}' does not exist.")
        index = self.labels.index(name)
        return self.activation_matrices[index]
