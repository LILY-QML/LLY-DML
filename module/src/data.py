# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Project: LILY-QML
# Version: 1.6 LLY-DML
# Author: Leon Kaiser
# Contact: info@lilyqml.de
# Website: www.lilyqml.de
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++

import json
import os
import logging
import math
import random

class Data:
    def __init__(self, test_mode=False):
        """
        Initializes a new instance of the Data class.

        :param test_mode: If True, data is read from the 'test/var' directory.
        """
        # Set base directory based on test_mode
        if test_mode:
            self.base_dir = os.path.join('test', 'var')
            print("Test mode activated: Data will be read from 'test/var'.")
        else:
            self.base_dir = 'var'
        
        # Paths to JSON files
        self.data_path = os.path.join(self.base_dir, 'data.json')
        self.train_path = os.path.join(self.base_dir, 'train.json')
        self.config_path = os.path.join(self.base_dir, 'config.json')
        
        # Initialize attributes
        self.qubits = None
        self.depth = None
        self.activation_matrices = {}
        
        # Setup logging
        self.setup_logging()

    def setup_logging(self):
        """
        Sets up the logging configuration based on the log file path in config.json.
        """
        try:
            with open(self.config_path, 'r') as config_file:
                config = json.load(config_file)
                log_file = config.get('log_file', 'var/circuit.log')
        except FileNotFoundError:
            # Default log file if config.json is not found
            log_file = 'var/circuit.log'
            print(f"Warning: {self.config_path} not found. Using default log file at '{log_file}'.")
        except json.JSONDecodeError:
            # Default log file if config.json is invalid
            log_file = 'var/circuit.log'
            print(f"Warning: {self.config_path} contains invalid JSON. Using default log file at '{log_file}'.")

        # Ensure the log directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            filename=log_file,
            filemode='a',
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.INFO
        )
        self.logger = logging.getLogger()

    def get_data(self):
        """
        Loads qubits, depth, and activation_matrices from data.json.
        Also loads training_matrices from train.json.

        :return: True if data loaded successfully, False otherwise.
        """
        # Load data.json
        try:
            with open(self.data_path, 'r') as data_file:
                data = json.load(data_file)
                self.qubits = data.get('qubits')
                self.depth = data.get('depth')
                self.activation_matrices = data.get('activation_matrices', {})
                self.logger.info({
                    "Success Code": 2028,
                    "Message": f"Data successfully loaded from {self.data_path}: qubits={self.qubits}, depth={self.depth}."
                })
        except FileNotFoundError:
            error_message = {
                "Error Code": 1028,
                "Message": "data.json could not be found."
            }
            self.logger.error(error_message)
            return False
        except json.JSONDecodeError:
            error_message = {
                "Error Code": 1021,
                "Message": "data.json contains invalid JSON."
            }
            self.logger.error(error_message)
            return False

        # Load train.json
        try:
            with open(self.train_path, 'r') as train_file:
                train_data = json.load(train_file)
                self.activation_matrices.update(train_data.get('activation_matrices', {}))
                self.logger.info({
                    "Success Code": 2029,
                    "Message": f"Training data successfully loaded from {self.train_path}."
                })
        except FileNotFoundError:
            error_message = {
                "Error Code": 1029,
                "Message": "train.json could not be found."
            }
            self.logger.error(error_message)
            return False
        except json.JSONDecodeError:
            error_message = {
                "Error Code": 1011,
                "Message": "train.json contains invalid JSON."
            }
            self.logger.error(error_message)
            return False

        return True

    def return_matrices(self):
        """
        Returns the stored activation matrices after validating their structure.

        :return: activation_matrices dictionary if valid, None otherwise.
        """
        # Validate activation matrices
        for name, matrix in self.activation_matrices.items():
            if not isinstance(name, str):
                error_message = {
                    "Error Code": 1030,
                    "Message": f"Activation matrix '{name}' does not have a valid name."
                }
                self.logger.error(error_message)
                return None

            if not self._validate_matrix_format(matrix):
                # Check if matrix has 3 pages
                if isinstance(matrix, list) and len(matrix) == 3:
                    error_message = {
                        "Error Code": 1030,
                        "Message": f"Activation matrix '{name}' was not successfully converted."
                    }
                    self.logger.error(error_message)
                    return None
                else:
                    error_message = {
                        "Error Code": 1030,
                        "Message": f"Activation matrix '{name}' does not have the correct format."
                    }
                    self.logger.error(error_message)
                    return None

        # If all matrices are valid
        self.logger.info({
            "Success Code": 2030,
            "Message": "All activation matrices have been successfully validated."
        })
        return self.activation_matrices

    def _validate_matrix_format(self, matrix):
        """
        Validates that the matrix has the correct number of rows and columns.

        :param matrix: The matrix to validate.
        :return: True if valid, False otherwise.
        """
        if not isinstance(matrix, list):
            return False
        if len(matrix) != self.qubits:
            return False
        for row in matrix:
            if not isinstance(row, list):
                return False
            if len(row) != self.depth * 3:
                return False
            if not all(isinstance(elem, (int, float)) for elem in row):
                return False
        return True

    def convert_matrices(self):
        """
        Converts the activation matrices from 3-page format to single-page format.

        :return: True if conversion successful, False otherwise.
        """
        # Pre-test validations
        for name, matrix in self.activation_matrices.items():
            if not isinstance(matrix, list) or len(matrix) != 3:
                error_message = {
                    "Error Code": 1031,
                    "Message": f"Activation matrix '{name}' is invalid before conversion."
                }
                self.logger.error(error_message)
                return False

            # Check each page has correct dimensions
            for page in matrix:
                if not isinstance(page, list) or len(page) != self.depth:
                    error_message = {
                        "Error Code": 1031,
                        "Message": f"Activation matrix '{name}' has invalid dimensions before conversion."
                    }
                    self.logger.error(error_message)
                    return False

        # Conversion process
        converted_matrices = {}
        for name, matrix in self.activation_matrices.items():
            # Initialize empty converted matrix
            converted_matrix = [[] for _ in range(self.qubits)]

            # Merge pages
            for page in matrix:
                for row_idx in range(self.qubits):
                    converted_matrix[row_idx].extend(page[row_idx])

            # Validate converted matrix
            if not self.check_final_matrix(converted_matrix):
                error_message = {
                    "Error Code": 1008,
                    "Message": f"Activation matrix '{name}' is not in the correct dimensions after conversion."
                }
                self.logger.error(error_message)
                return False

            converted_matrices[name] = converted_matrix

        # Update activation_matrices with converted data
        self.activation_matrices = converted_matrices

        # Post-conversion validation
        for name, matrix in self.activation_matrices.items():
            if not self.check_final_matrix(matrix):
                error_message = {
                    "Error Code": 1008,
                    "Message": f"Activation matrix '{name}' failed final matrix validation."
                }
                self.logger.error(error_message)
                return False

        self.logger.info({
            "Success Code": 2031,
            "Message": "All activation matrices have been successfully converted."
        })
        return True

    def check_final_matrix(self, matrix):
        """
        Checks if the matrix has the correct dimensions.

        :param matrix: The matrix to check.
        :return: True if valid, False otherwise.
        """
        if not isinstance(matrix, list):
            return False
        if len(matrix) != self.qubits:
            return False
        for row in matrix:
            if not isinstance(row, list):
                return False
            if len(row) != self.depth * 3:
                return False
        return True

    def create_training_matrix(self):
        """
        Creates a training matrix with random values between -2pi and 2pi.

        :return: training_matrix if successful, None otherwise.
        """
        training_matrix = []
        for _ in range(self.qubits):
            row = [random.uniform(-2 * math.pi, 2 * math.pi) for _ in range(self.depth * 3)]
            training_matrix.append(row)

        # Validate training matrix
        if not self.check_final_matrix(training_matrix):
            error_message = {
                "Error Code": 1007,
                "Message": "Training matrix is not in appropriate dimensions."
            }
            self.logger.error(error_message)
            return None

        self.logger.info({
            "Success Code": 2032,
            "Message": "Training matrix successfully created and validated."
        })
        return training_matrix
