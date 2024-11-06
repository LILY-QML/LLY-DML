# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Project: LILY-QML
# Version: 2.0.0 LLY-DML
# Author: Leon Kaiser
# Contact: info@lilyqml.de
# Website: www.lilyqml.de
# Contributors: Claudia Zendejas-Morales (@clausia)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++

import numpy as np
import json
import os


class Data:
    def __init__(self, qubits, depth, activation_matrices=None, working_directory='var'):
        """
        Initializes the Data class for managing quantum circuit activation matrices.

        :param qubits: int, the number of qubits, corresponding to the number of rows in the matrices.
        :param depth: int, the depth of the circuit, corresponding to the number of columns in the matrices.
        :param activation_matrices: list, optional, contains the three-dimensional activation matrices.
        :param working_directory: str, the directory where data files and logs are managed. Defaults to 'var'.

        """
        self.qubits = qubits
        self.depth = depth
        self.activation_matrices = activation_matrices if activation_matrices else []
        self.working_directory = working_directory

    def get_data(self):
        """
        Loads values for qubits, depth, and activation_matrices from data.json.

        :return: None if successful, error code 1028 if data.json is not found,
                 or error code 1029 if train.json is not found.
        """
        try:
            with open(os.path.join(self.working_directory, 'data.json'), 'r') as f:
                data = json.load(f)
                self.qubits = data['qubits']
                self.depth = data['depth']
                self.activation_matrices = [
                    {
                        "name": matrix["name"],
                        "data": np.array(matrix["data"])
                    } for matrix in data['activation_matrices']
                ]
        except FileNotFoundError:
            return "Error Code: 1028 - data.json not found"

        if not os.path.exists(os.path.join(self.working_directory, 'train.json')):
            return "Error Code: 1029 - train.json not found"

    def return_matrices(self):
        """
        Returns the stored activation matrices, ensuring that they have correct properties.

        :return: List of activation matrices if valid, otherwise an error code.
        """
        for matrix in self.activation_matrices:
            if 'name' not in matrix:
                return "Error Code: 1030 - Activation matrix conversion unsuccessful"
            rows, cols, pages = matrix['data'].shape
            if rows != self.qubits or cols != self.depth * 3 or pages != 3:
                return "Error Code: 1030 - Activation matrix conversion unsuccessful"
        return self.activation_matrices

    def convert_matrices(self):
        """
        Converts activation matrices to a specific format by combining values from 3 pages.

        :return: None if successful, error code if initial checks fail or if conversion fails.
        """
        for matrix in self.activation_matrices:
            # Initial check to ensure matrix has the required properties
            if 'name' not in matrix:
                return "Error Code: 1031 - Activation matrix invalid before conversion"
            if matrix['data'].shape[2] != 3:
                return "Error Code: 1031 - Activation matrix invalid before conversion"
            rows, cols, _ = matrix['data'].shape
            if rows != self.qubits or cols != self.depth:
                return "Error Code: 1031 - Activation matrix invalid before conversion"

        # Conversion process
        for i, matrix in enumerate(self.activation_matrices):
            # Create a new matrix with dimensions qubits x (depth * 3)
            converted_matrix = np.zeros((self.qubits, self.depth * 3))
            for q in range(self.qubits):
                for d in range(self.depth):
                    # Flatten the values from the 3 pages into the new matrix
                    converted_matrix[q, d * 3:(d + 1) * 3] = matrix['data'][q, d, :]
            self.activation_matrices[i]['data'] = converted_matrix

        # Validate the final matrices
        for matrix in self.activation_matrices:
            if 'name' not in matrix:
                return "Error Code: 1008 - Activation matrix does not meet required dimensions"
            if not self.check_final_matrix(matrix['data']):
                return "Error Code: 1008 - Activation matrix does not meet required dimensions"

    # def convert_matrices(self):
    #     """
    #     Converts activation matrices to a specific format by combining values from 3 pages.
    #
    #     :return: None if successful, error code if initial checks fail or if conversion fails.
    #     """
    #     for matrix in self.activation_matrices:
    #         # Initial check to ensure matrix has the required properties
    #         if 'name' not in matrix:
    #             return "Error Code: 1031 - Activation matrix invalid before conversion"
    #         if matrix['data'].shape[2] != 3:
    #             return "Error Code: 1031 - Activation matrix invalid before conversion"
    #         rows, cols, _ = matrix['data'].shape
    #         if rows != self.qubits or cols != self.depth:
    #             return "Error Code: 1031 - Activation matrix invalid before conversion"
    #
    #     # Conversion process
    #     for i, matrix in enumerate(self.activation_matrices):
    #         converted_matrix = np.zeros((self.qubits, self.depth, 3))
    #         for q in range(self.qubits):
    #             for d in range(self.depth):
    #                 # Combine values from the 3 pages
    #                 converted_matrix[q, d] = matrix['data'][q, d, :]
    #         self.activation_matrices[i]['data'] = converted_matrix
    #
    #     # Validate the final matrices
    #     for matrix in self.activation_matrices:
    #         if 'name' not in matrix:
    #             return "Error Code: 1008 - Activation matrix does not meet required dimensions"
    #         if not self.check_final_matrix(matrix):
    #             return "Error Code: 1008 - Activation matrix does not meet required dimensions"

    def check_final_matrix(self, matrix):
        """
        Checks if a matrix meets the final required dimensions.

        :param matrix: ndarray, the matrix to check.
        :return: True if valid, False otherwise.
        """
        # Verify that the matrix has the correct dimensions (qubits x depth * 3)
        rows, cols = matrix.shape
        return rows == self.qubits and cols == self.depth * 3

    def create_training_matrix(self):
        """
        Creates a training matrix with randomized values between -2π and 2π.

        :return: None if successful, error code if dimensions are invalid.
        """
        training_matrix = np.random.uniform(-2 * np.pi, 2 * np.pi, (self.qubits, self.depth * 3))

        # Check if the created training matrix meets the required dimensions
        if training_matrix.shape != (self.qubits, self.depth * 3):
            return "Error Code: 1007 - Training matrix does not meet required dimensions"

        # Validate the dimensions using check_final_matrix
        if not self.check_final_matrix(training_matrix):
            return "Error Code: 1007 - Training matrix does not meet required dimensions"

        # Save to train.json
        with open(os.path.join(self.working_directory, 'train.json'), 'w') as f:
            json.dump({"training_matrix": training_matrix.tolist()}, f)

