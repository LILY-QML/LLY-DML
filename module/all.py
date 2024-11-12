# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Project: LILY-QML
# Version: 2.0.0 LLY-DML
# Author: Claudia Zendejas-Morales (@clausia)
# Contact: info@lilyqml.de
# Website: www.lilyqml.de
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import json
from datetime import datetime
from module.src.reader import Reader
from module.src.data import Data


class All:
    def __init__(self, working_directory='var'):
        """
        Initializes the All class for managing training files and data processes.

        :param working_directory: Path where files such as train.json, config.json, and logs are managed.
        """
        self.working_directory = working_directory
        self.reader = Reader(working_directory)
        self.train_exists = os.path.exists(os.path.join(self.working_directory, 'train.json'))

    def handle_train_file(self):
        """
        Manages the train.json file with options to archive, delete, or exit if it exists,
        and creates a new training file if not.
        """
        # Dynamically check if train.json exists each time this method is called
        train_path = os.path.join(self.working_directory, 'train.json')
        train_exists = os.path.exists(train_path)

        if train_exists:
            creation_date = datetime.fromtimestamp(
                os.path.getctime(train_path)
            ).strftime('%Y-%m-%d')

            # Display options to the user
            print(f"""
            =============================================================================
                                TRAINING - ALL OPTIMIZERS
            =============================================================================

            train.json already exists.
            It was created on {creation_date}.

            The following options are available:

              1. Save it to the archive (recommended)
              2. Delete the current train.json.
              3. Exit
            """)

            user_choice = input("Please enter your choice (1, 2, or 3): ").strip()
            if user_choice == '1':
                # Archive train.json and create a new one
                print("DEBUG: User selected option 1. Attempting to move train.json to the archive.")
                archive_result = self.reader.move_json_file()
                if isinstance(archive_result, dict) and "Error Code" in archive_result:
                    print(f"DEBUG: Error during archiving - {archive_result['Message']}")
                else:
                    print("DEBUG: Archive operation successful, creating a new train.json.")
                    self.reader.create_train_file()
            elif user_choice == '2':
                # Delete existing train.json and create a new one
                print("DEBUG: User selected option 2. Deleting train.json and creating a new one.")
                os.remove(train_path)
                self.reader.create_train_file()
            elif user_choice == '3':
                print("Exiting without changes.")
            else:
                print("Invalid selection. Exiting...")
                return
        else:
            # Create a new train.json file if it doesn't exist
            print("DEBUG: train.json does not exist. Creating a new one.")
            self.reader.create_train_file()

    def data_ready(self, qubits, depth):
        """
        Prepares the necessary data by converting and saving matrices and creating a training matrix.

        :param qubits: The number of qubits for the activation matrix.
        :param depth: The depth of the circuit for the activation matrix.
        """
        # Initialize Data instance and retrieve data
        data_instance = Data(qubits=qubits, depth=depth, working_directory=self.working_directory)
        get_data_result = data_instance.get_data()

        # Handle errors from get_data
        if isinstance(get_data_result, str):
            print(get_data_result)
            return

        # Convert matrices if required
        convert_result = data_instance.convert_matrices()
        if isinstance(convert_result, str):
            print(convert_result)
            return

        # Retrieve activation matrices and save to train.json
        activation_matrices = data_instance.return_matrices()
        if isinstance(activation_matrices, str):
            print(activation_matrices)
            return

        # Save activation matrices to train.json
        with open(os.path.join(self.working_directory, 'train.json'), 'w') as train_file:
            json.dump({"activation_matrices": [matrix['data'].tolist() for matrix in activation_matrices]}, train_file)

        # Create and save training matrix
        training_matrix_result = data_instance.create_training_matrix()
        if isinstance(training_matrix_result, str):
            print(training_matrix_result)

    def Precheck(self):
        """
        Verifies all necessary files and configurations are present before proceeding with data processing and training.
        Retrieves or requests `qubits` and `depth` values if needed.
        """
        # Check if config.json exists
        config_path = os.path.join(self.working_directory, 'config.json')
        if not os.path.exists(config_path):
            print("Error Code: 1099 - config.json not found.")
            return

        # Verify log file existence
        log_check_result = self.reader.checkLog()
        if isinstance(log_check_result, dict) and "Error Code" in log_check_result:
            print("Error Code: 1099 - Logfile not found.")
            return

        # Retrieve qubits and depth from data.json, or prompt if missing
        try:
            with open(os.path.join(self.working_directory, 'data.json'), 'r') as data_file:
                data = json.load(data_file)
                qubits = data.get('qubits')
                depth = data.get('depth')
        except FileNotFoundError:
            print("data.json not found. Please provide values for qubits and depth.")
            qubits = int(input("Enter number of qubits: "))
            depth = int(input("Enter depth: "))
        except json.JSONDecodeError:
            print("data.json is improperly formatted.")
            return

        # Check that both qubits and depth are available
        if qubits is None or depth is None:
            print("Error: qubits and depth values are required.")
            return

        # Proceed with data preparation
        self.data_ready(qubits, depth)

        # Determine state
        self.get_state()

    def get_state(self):
        pass
