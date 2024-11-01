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
from datetime import datetime
import shutil

class Reader:
    def __init__(self, test_mode=False):
        """
        Initializes a new instance of the Reader class.

        :param test_mode: If True, operates within the 'test/var' directory.
        """
        # Set base directory based on test_mode
        if test_mode:
            self.base_dir = os.path.join('test', 'var')
            print("Test mode activated: Operating within 'test/var'.")
        else:
            self.base_dir = 'var'
        
        # Paths to necessary files
        self.config_path = os.path.join(self.base_dir, 'config.json')
        self.log_file = None  # Will be set in setup_logging
        
        # Setup logging
        self.setup_logging()

    def setup_logging(self):
        """
        Sets up the logging configuration based on the log file path in config.json.
        """
        try:
            with open(self.config_path, 'r') as config_file:
                config = json.load(config_file)
                self.log_file = config.get('log_file', os.path.join(self.base_dir, 'reader.log'))
        except FileNotFoundError:
            # Default log file if config.json is not found
            self.log_file = os.path.join(self.base_dir, 'reader.log')
            print(f"Warning: {self.config_path} not found. Using default log file at '{self.log_file}'.")
        except json.JSONDecodeError:
            # Default log file if config.json is invalid
            self.log_file = os.path.join(self.base_dir, 'reader.log')
            print(f"Warning: {self.config_path} contains invalid JSON. Using default log file at '{self.log_file}'.")

        # Ensure the log directory exists
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            filename=self.log_file,
            filemode='a',
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.INFO
        )
        self.logger = logging.getLogger()

    def fileCheck(self):
        """
        Checks for the existence of the 'var' folder and required JSON files.

        :return: Error or Success Code based on the checks.
        """
        var_folder = self.base_dir

        # Check if 'var' folder exists
        if not os.path.isdir(var_folder):
            error_message = {
                "Error Code": 1001,
                "Message": "'var' folder is missing."
            }
            self.logger.error(error_message)
            return error_message

        # Define required files
        required_files = ['data.json', 'config.json']
        train_exists = False

        # Check for required files
        for file in required_files:
            file_path = os.path.join(var_folder, file)
            if not os.path.isfile(file_path):
                if file == 'data.json':
                    error_code = 1002
                    message = "'data.json' file is missing in 'var' folder."
                elif file == 'config.json':
                    error_code = 1003
                    message = "'config.json' file is missing in 'var' folder."
                error_message = {
                    "Error Code": error_code,
                    "Message": message
                }
                self.logger.error(error_message)
                return error_message

        # Check if 'train.json' exists
        train_path = os.path.join(var_folder, 'train.json')
        if os.path.isfile(train_path):
            train_exists = True
            success_message = {
                "Success Code": 2001,
                "Message": "All files are present. 'train.json' exists."
            }
            self.logger.info(success_message)
            return success_message
        else:
            success_message = {
                "Success Code": 2002,
                "Message": "All files are present. 'train.json' is missing."
            }
            self.logger.info(success_message)
            return success_message

    def checkLog(self):
        """
        Manages log files by ensuring that a current log file exists and is up-to-date.
        Creates a new log file if necessary.

        :return: Success or Error Code based on the operations.
        """
        today_str = datetime.now().strftime('%Y-%m-%d')
        expected_log_name = f"{today_str}.log"
        expected_log_path = os.path.join(self.base_dir, expected_log_name)

        # Check if log file is defined in config
        if not self.log_file:
            # If no log file defined, create a new one
            self.createLog()
            return {
                "Success Code": 2003,
                "Message": "New log file created as no log file was defined."
            }

        # Extract log file name
        current_log_name = os.path.basename(self.log_file)
        current_log_path = self.log_file

        if current_log_name == expected_log_name:
            # Check if the log file exists
            if os.path.isfile(current_log_path):
                success_message = {
                    "Success Code": 2004,
                    "Message": "Current log file is up-to-date."
                }
                self.logger.info(success_message)
                return success_message
            else:
                # Create a new log file with today's date
                self.createLog()
                return {
                    "Success Code": 2003,
                    "Message": "New log file created as the current log file was missing."
                }
        else:
            # Create a new log file with today's date
            self.createLog()
            return {
                "Success Code": 2003,
                "Message": "New log file created with today's date."
            }

    def createLog(self):
        """
        Creates a new log file based on the current date and updates config.json accordingly.

        :return: Success or Error Code based on the operations.
        """
        today_str = datetime.now().strftime('%Y-%m-%d')
        new_log_name = f"{today_str}.log"
        new_log_path = os.path.join(self.base_dir, new_log_name)

        # Check if the new log file already exists
        if os.path.isfile(new_log_path):
            # Append a numerical suffix if the file already exists
            suffix = 1
            while True:
                temp_log_name = f"{today_str}-{suffix}.log"
                temp_log_path = os.path.join(self.base_dir, temp_log_name)
                if not os.path.isfile(temp_log_path):
                    new_log_name = temp_log_name
                    new_log_path = temp_log_path
                    break
                suffix += 1

        # Create the new log file
        try:
            with open(new_log_path, 'w') as log_file:
                log_file.write(f"Log created on {today_str}\n")
            self.log_file = new_log_path
            self.logger.info({
                "Success Code": 2005,
                "Message": f"New log file '{new_log_name}' created successfully."
            })
        except Exception as e:
            error_message = {
                "Error Code": 1004,
                "Message": f"Failed to create new log file: {str(e)}"
            }
            self.logger.error(error_message)
            return error_message

        # Update config.json with the new log file name
        config_data = {}
        try:
            if os.path.isfile(self.config_path):
                with open(self.config_path, 'r') as config_file:
                    config_data = json.load(config_file)
            config_data['log_file'] = new_log_name
            with open(self.config_path, 'w') as config_file:
                json.dump(config_data, config_file, indent=4)
            self.logger.info({
                "Success Code": 2006,
                "Message": f"Config updated with new log file '{new_log_name}'."
            })
        except Exception as e:
            error_message = {
                "Error Code": 1005,
                "Message": f"Failed to update config.json: {str(e)}"
            }
            self.logger.error(error_message)
            return error_message

        return {
            "Success Code": 2005,
            "Message": f"New log file '{new_log_name}' created and config.json updated successfully."
        }

    def dataConsistency(self):
        """
        Checks the consistency of data.json by validating its structure and contents.

        :return: Success or Error Code based on the checks.
        """
        data_path = os.path.join(self.base_dir, 'data.json')

        # Check if data.json exists
        if not os.path.isfile(data_path):
            error_message = {
                "Error Code": 1006,
                "Message": "'data.json' is already present."
            }
            self.logger.error(error_message)
            return error_message

        # Load data.json
        try:
            with open(data_path, 'r') as data_file:
                data = json.load(data_file)
        except json.JSONDecodeError:
            error_message = {
                "Error Code": 1007,
                "Message": "'data.json' contains invalid JSON."
            }
            self.logger.error(error_message)
            return error_message

        # Validate basic structure
        required_keys = ['qubits', 'depth', 'optimizers', 'optimizer_arguments', 'matrices']
        for key in required_keys:
            if key not in data:
                error_message = {
                    "Error Code": 1008,
                    "Message": f"Missing key '{key}' in 'data.json'."
                }
                self.logger.error(error_message)
                return error_message

        # Validate 'optimizers' list
        optimizers = data['optimizers']
        if not isinstance(optimizers, list) or not all(isinstance(opt, str) for opt in optimizers):
            error_message = {
                "Error Code": 1009,
                "Message": "'optimizers' must be a list of strings."
            }
            self.logger.error(error_message)
            return error_message

        # Validate 'optimizer_arguments'
        optimizer_args = data['optimizer_arguments']
        if not isinstance(optimizer_args, dict):
            error_message = {
                "Error Code": 1010,
                "Message": "'optimizer_arguments' must be a dictionary."
            }
            self.logger.error(error_message)
            return error_message

        for optimizer in optimizers:
            if optimizer not in optimizer_args:
                error_message = {
                    "Error Code": 1011,
                    "Message": f"Arguments for optimizer '{optimizer}' are missing in 'optimizer_arguments'."
                }
                self.logger.error(error_message)
                return error_message

        # Validate 'matrices'
        matrices = data['matrices']
        if not isinstance(matrices, dict) or not matrices:
            error_message = {
                "Error Code": 1012,
                "Message": "'matrices' must be a non-empty dictionary."
            }
            self.logger.error(error_message)
            return error_message

        for name, matrix in matrices.items():
            if not isinstance(matrix, list) or not matrix:
                error_message = {
                    "Error Code": 1013,
                    "Message": f"Matrix '{name}' must be a non-empty list."
                }
                self.logger.error(error_message)
                return error_message

        # All checks passed
        success_message = {
            "Success Code": 2007,
            "Message": "'data.json' is consistent and valid."
        }
        self.logger.info(success_message)
        return success_message

    def create_train_file(self):
        """
        Creates a new 'train.json' file if it does not already exist.

        :return: Success or Error Code based on the operations.
        """
        train_path = os.path.join(self.base_dir, 'train.json')

        # Check if 'train.json' already exists
        if os.path.isfile(train_path):
            error_message = {
                "Error Code": 1006,
                "Message": "'train.json' is already present."
            }
            self.logger.error(error_message)
            return error_message

        # Create 'train.json' with creation date in the header
        creation_date = datetime.now().strftime('%Y-%m-%d')
        train_data = {
            "creation_date": creation_date,
            "data": {}
        }

        try:
            with open(train_path, 'w') as train_file:
                json.dump(train_data, train_file, indent=4)
            success_message = {
                "Success Code": 2008,
                "Message": "'train.json' created successfully with creation date."
            }
            self.logger.info(success_message)
            return success_message
        except Exception as e:
            error_message = {
                "Error Code": 1014,
                "Message": f"Failed to create 'train.json': {str(e)}"
            }
            self.logger.error(error_message)
            return error_message

    def move_json_file(self):
        """
        Moves an existing 'train.json' file to the 'archive' folder with the creation date as the filename.

        :return: Success or Error Code based on the operations.
        """
        train_path = os.path.join(self.base_dir, 'train.json')

        # Check if 'train.json' exists
        if not os.path.isfile(train_path):
            error_message = {
                "Error Code": 1006,
                "Message": "'train.json' is not present."
            }
            self.logger.error(error_message)
            return error_message

        # Read creation date from 'train.json'
        try:
            with open(train_path, 'r') as train_file:
                train_data = json.load(train_file)
                creation_date = train_data.get('creation_date')
                if not creation_date:
                    error_message = {
                        "Error Code": 1015,
                        "Message": "'train.json' does not contain a creation date."
                    }
                    self.logger.error(error_message)
                    return error_message
        except json.JSONDecodeError:
            error_message = {
                "Error Code": 1016,
                "Message": "'train.json' contains invalid JSON."
            }
            self.logger.error(error_message)
            return error_message

        # Ensure 'archive' directory exists
        archive_dir = os.path.join(self.base_dir, 'archive')
        os.makedirs(archive_dir, exist_ok=True)

        # Determine new filename
        new_log_name = f"{creation_date}.json"
        new_log_path = os.path.join(archive_dir, new_log_name)

        # Check if file with the same name already exists
        if os.path.isfile(new_log_path):
            suffix = 1
            while True:
                temp_log_name = f"{creation_date}-{suffix}.json"
                temp_log_path = os.path.join(archive_dir, temp_log_name)
                if not os.path.isfile(temp_log_path):
                    new_log_name = temp_log_name
                    new_log_path = temp_log_path
                    break
                suffix += 1

        # Move the file
        try:
            shutil.move(train_path, new_log_path)
            success_message = {
                "Success Code": 2009,
                "Message": f"'train.json' moved to archive as '{new_log_name}'."
            }
            self.logger.info(success_message)
            return success_message
        except Exception as e:
            error_message = {
                "Error Code": 1017,
                "Message": f"Failed to move 'train.json' to archive: {str(e)}"
            }
            self.logger.error(error_message)
            return error_message
