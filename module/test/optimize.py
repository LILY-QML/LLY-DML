import unittest
import json
import os
from datetime import datetime

from module.src.optimize import Optimizer  # Assuming the Optimize class is implemented in this path and in test is running as python3 -m module.test.optimize

# Path to config.json file
config_path =  os.path.join('var','config.json')

# Function to log messages to a log file
def log_message(message, level="INFO", error_code=None):
    try:
        print(message)

        # Lade die Konfiguration aus config.json
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)

        # Hole den Pfad der Logdatei aus der Konfiguration
        log_file = config.get('log_file', 'default.log')

        # Erstelle den vollständigen Pfad zur Logdatei im Ordner 'var'
        log_path = os.path.join('var', log_file)

        # Formatierung der Log-Einträge
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"

        # Wenn ein Fehlercode vorhanden ist, füge ihn hinzu
        if error_code:
            log_entry += f" [Error Code: {error_code}]"

        # Schreibe die Nachricht in die Logdatei
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
        self.valid_optimizer = Optimizer(optimizer="AdaGradOptimizer")
        self.invalid_optimizer = Optimizer(optimizer="InvalidOptimizer")
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

if __name__ == '__main__':
    unittest.main()

