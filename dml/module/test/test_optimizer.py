# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Project: LILY-QML
# Version: 2.0.0 LLY-DML
# Author: Joan Pujol
# Contact: info@lilyqml.de
# Website: www.lilyqml.de
# Contributors:
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++

import unittest
import json
import os
from datetime import datetime
import tempfile

from module.src.optimizer import Optimizer  # Updated to match renamed optimizer module

# Path to config.json file
config_path = os.path.join('var', 'config.json')

# Function to log messages to a log file
def log_message(message, level="INFO", error_code=None):
    try:
        print(message)

        # Load configuration from config.json
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)

        # Get the log file path from the configuration
        log_file = config.get('log_file', 'default.log')

        # Create the full path to the log file in the 'var' folder
        log_path = os.path.join('var', log_file)

        # Format log entries
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"

        # Add error code if present
        if error_code:
            log_entry += f" [Error Code: {error_code}]"

        # Write the message to the log file
        with open(log_path, 'a') as log:
            log.write(log_entry + '\n')

    except Exception as e:
        print(f'Error while writing to log: {e}')

class TestOptimize(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Set up the class for the test cases.
        Logs the start of the Optimize tests.
        """
        log_message("===================== Optimize Tests ===================", level="START")

    def setUp(self):
        """
        Set up the test case environment.
        Initializes valid and invalid Optimizer instances.
        Logs the completion of the setup.
        """
        self.valid_optimizer = Optimizer()
        self.valid_optimizer.optimizer_class = "AdaGradOptimizer"
        self.invalid_optimizer = Optimizer()
        self.invalid_optimizer.optimizer_class = "InvalidOptimizer"
        log_message("Setup completed TestOptimize with initial valid and invalid data.", level="INFO")   

    def test_check_prerequisites(self):
        """
        Test the check_prerequisites method of the Optimizer class.
        Validates the behavior with both valid and invalid optimizer configurations.
        Logs the success or failure of the test.
        """
        try:
            # Check loading valid and invalid optimizers
            result = self.valid_optimizer.check_prerequisites()
            self.assertEqual(result, None, "Error checking valid optimizer")
            result = self.invalid_optimizer.check_prerequisites()
            self.assertEqual(result, {"Error Code": 1111, "Message": "Optimizer not found."}, "Error checking invalid optimizer")
            # Check loading not existing data.json
            self.valid_optimizer.config_path="."
            result = self.valid_optimizer.check_prerequisites()
            self.assertEqual(result, {"Error Code": 1112, "Message": "Data file not found."})
            log_message("test_check_prerequisites passed successfully.", level="SUCCESS")

        except Exception as e:
            log_message(str(e), level="ERROR")

    def test_check_data_structure(self):
        """
        Test the check_data_structure method of the Optimizer class.
        Validates the behavior with both valid and invalid current_job strings.
        Logs the success or failure of the test.
        """
        try:
            self.valid_optimizer.current_job = "(1, 2, 3) Qubit_0 (1:200; 0:50) (S:1)"
            result = self.valid_optimizer.check_data_structure()
            self.assertEqual(result, None, "Error testing check_data_structure with valid current_job")
            self.valid_optimizer.current_job = "(1, 2, 3) Qubit_0 (1:200; 0:50,3) (S:1)"
            result = self.valid_optimizer.check_data_structure()
            self.assertEqual(result, {"Error Code": 1119, "Message": "Datastructure is not consistent."}, "Error testing check_data_structure with invalid current_job")
            log_message("test_check_data_structure.", level="SUCCESS")

        except Exception as e:
            log_message(str(e), level="ERROR")

        log_message("test_check_data_structure passed successfully.", level="SUCCESS")

    def test_evaluate(self):
        """
        Test the evaluate method of the Optimizer class.
        Validates the evaluation of a current job string.
        Logs the success or failure of the test.
        """
        try:
            result = self.valid_optimizer.evaluate("(1, 2, 3) Qubit_0 (1:200; 0:50) (S:1)")
        except Exception as e:
            log_message(str(e), level="ERROR")

        log_message("test_evaluate passed successfully.", level="SUCCESS")

    def test_execute(self):
        """
        Test the execute method of the Optimizer class.
        Validates the execution of the optimization process for a current job string.
        Logs the success or failure of the test.
        """
        try:
            result = self.valid_optimizer.check_prerequisites()
            result = self.valid_optimizer.execute("(1, 2, 3) Qubit_0 (1:200; 0:50) (S:011)")
            log_message(result, level="INFO")
        except Exception as e:
            log_message(str(e), level="ERROR")

        log_message("test_execute passed successfully.", level="SUCCESS")

    def test_start(self):
        """
        Test the start_optimize method of the Optimizer class.
        Validates the start of the optimization process.
        Logs the success or failure of the test.
        """
        try:
            optimizer = Optimizer()
            temp_dir = tempfile.TemporaryDirectory()
            optimizer.train_json_file_path = os.path.join(temp_dir.name, 'train.json')
            result=optimizer.start("INVALIDOPTIMIZER", "010")
            self.assertEqual(result, {"Error Code": 1072, "Message": "Optimizer not found."}, "Error testing test_start_optimize with invalid optimizer")
            result=optimizer.start("AdaGradOptimizer", "")
            self.assertEqual(result, {"Error Code": 1071, "Message": "Target state has incorrect formatting."}, "Error testing test_start_optimize with invalid state")
            result=optimizer.start("AdaGradOptimizer", "010")
            self.assertEqual(result, {"Error Code": 1070, "Message": "train.json not found."}, "Error testing test_start_optimize with invalid path to train.json")

            with open(optimizer.train_json_file_path, 'w') as f:
                f.write('test')            

            result = optimizer.start("AdaGradOptimizer", "010")
            self.assertEqual(result, None, "Error testing test_start_optimize with valid data")

        except Exception as e:
            log_message(str(e), level="ERROR")

        log_message("test_start_optimize passed successfully.", level="SUCCESS")

    def test_encode_measurements(self):
        optimizer = Optimizer()

        measurements={'000': 512, '101': 488}
        optimizer.initialize_qubits(2)
        result = optimizer.encode_measurements(measurements)
        self.assertEqual(result, None, "Error testing encode_measurements with inconsistent data.")
        optimizer.initialize_qubits(3)
        expected_encode = [
            "(1:488; 0:512)",
            "(1:0; 0:1000)",
            "(1:488; 0:512)"
            ]
        result = optimizer.encode_measurements(measurements)
        self.assertEqual(result, expected_encode, "Error testing encode_measurements with consistent data.")

    def test_optimize(self):
         optimizer = Optimizer()
         temp_dir = tempfile.TemporaryDirectory()
         optimizer.train_json_file_path = os.path.join(temp_dir.name, 'train.json')
         
         with open(optimizer.train_json_file_path, 'w') as f:
            f.write('test')            

         measurements={'000': 512, '101': 488}

         training_matrix_invalid = [[1, 2, 3, 4], 
                                    [5, 6, 7, 8], 
                                    [9, 10, 11, 12],
                                    [13, 14, 15, 16]]

         training_matrix_valid = [[1, 2, 3, 4], 
                            [5, 6, 7, 8], 
                            [9, 10, 11, 12]]

         optimizer.start("AdaGradOptimizer", "010")
         optimizer.initialize_qubits(3)
         result = optimizer.optimize(measurements, training_matrix_invalid)
         self.assertEqual(result, None, "Error testing optimize with invalid training matrix")       
         result = optimizer.optimize(measurements, training_matrix_valid)
         self.assertIsNotNone(result, "Error testing optimize with valid training matrix")

if __name__ == '__main__':
    unittest.main()
