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

from module.all import All
from module.src.reader import Reader
import numpy as np

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

class TestAll(unittest.TestCase):

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
        """

        log_message("Setup completed TestAll with initial valid and invalid data.", level="INFO")   

    def test_all(self):
        #self.make_compatible_data_json()
        all = All(config_base_path="var_test")
        reader = Reader(working_directory="var_test")
        reader.move_json_file()
        reader.create_train_file()
        all.optimize(20)

    def make_compatible_data_json(self):
        config_base_path="var_test"

        data_path = os.path.join(config_base_path, 'data_orig.json')
        data_path_compatible = os.path.join(config_base_path, 'data.json')

        with open(data_path, 'r') as f:
            data_json = json.load(f)


        #We need to do the matrix activations with shape (self.data_json['qubits'], self.data_json['depth']*3)
        shape_training_matrix = (data_json['qubits'], data_json['depth']*3)

        for name_matrix, activation_matrix_list in data_json['matrices'].items():
            for i in range(len(activation_matrix_list)):
                activation_matrix_list[i] = (np.random.rand(*shape_training_matrix) * 20 - 10).tolist()

        data_json['matrices_states']={
            "Hund":"10000",
            "Katze":"01000",
            "Vogel":"00100",
            "Maus":"00010",
            "Elefant":"00001"
        }

        data_json['iteration_crosstrain_a'] = 100
        data_json['iteration_crosstrain_b'] = 100


        with open(data_path_compatible, 'w') as f:
            json.dump(data_json, f, indent=4)


if __name__ == '__main__':
    unittest.main()        