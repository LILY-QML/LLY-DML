# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Project: LILY-QML
# Version: 2.0.0 LLY-DML
# Author: Leon Kaiser
# Contact: info@lilyqml.de
# Website: www.lilyqml.de
# Contributors: Claudia Zendejas-Morales (@clausia), Joan Pujol (@supercabb)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++

import json
import os
from datetime import datetime
import logging


class Reader:

    def __init__(self, working_directory='var'):
        """
        Initializes the Reader class, setting up the working directory and configuration file.

        :param working_directory: Path to the directory where files and logs will be managed.
                                  Defaults to 'var'.
        """

        self.logger = None
        self.working_directory = working_directory
        self.checkLog()

    def fileCheck(self):
        """
        Verifies the existence of required files within the specified working directory.

        This method checks if the working directory exists and then verifies the presence
        of essential files: 'train.json', 'config.json', and 'data.json'. If the directory
        or any of these files are missing, it logs an error message and returns a detailed
        error message. If all files are present, it logs and returns a confirmation message.

        Functionality:
        - Folder Check:
          Ensures the working directory (default: 'var') exists. If it does not,
          an error message is logged and returned.

        - File Check:
          Confirms the presence of the required files ('train.json', 'config.json',
          'data.json') within the working directory. If any file is missing,
          a list of missing files is logged and returned.

        - Error Handling:
          If the working directory or any required file is missing, an appropriate error
          message is logged and returned. If all checks pass, an informational log
          confirms the presence of all required files.

        :return: A success message if all files are present, or an error message indicating
                 any missing files.
        """

        if not os.path.exists(self.working_directory):
            error_msg = f"Error: The directory '{self.working_directory}' does not exist."
            self.logger.error(error_msg)
            return error_msg

        required_files = ['train.json', 'config.json', 'data.json']
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(self.working_directory, f))]
        if missing_files:
            error_msg = f"Error: Missing the following files: {', '.join(missing_files)}"
            self.logger.error(error_msg)
            return error_msg

        self.logger.info("All required files are present.")
        return "All required files are present."

    def checkLog(self):
        """
        Manages the log file, ensuring that a current log file exists and is correctly updated
        according to the configuration.

        This method reads the specified configuration file in the working directory to determine
        if a logfile is defined. It then checks if the logfile name matches today’s date. If the
        logfile does not match or is missing, it creates a new log file using the `createLog()`
        method and updates the configuration file accordingly. This process guarantees that each
        day has its own unique logfile, and that the configuration reflects the current log file.

        Functionalities:
        - Read Log from Configuration:
          Opens the configuration file to check if a logfile is specified. If the file does
          not exist, an error is logged, and a new logfile is created.

        - Check Logfile Name:
          Compares the name of the logfile in the configuration file with today’s date. If they
          do not match, this indicates that a new log file should be created.

        - Logfile Existence:
          If the name matches today’s date, it verifies that the logfile exists in the working
          directory. If it does not, a new log file is created.

        - Create New Logfile:
          If the logfile is outdated, missing, or not defined in the configuration file, this
          method creates a new logfile named with the current date (e.g., '2024-04-27.log') and
          updates the configuration file to reflect the new file.

        Error Handling:
        - If the configuration file is missing or cannot be read, a new logfile is created to
          ensure logging continuity, and an error message is logged.
        - If the configuration file contains invalid JSON, an error is logged indicating the
          formatting issue.

        :raises FileNotFoundError: If the configuration file is missing, it creates a new logfile.
        :raises json.JSONDecodeError: If the configuration file has invalid JSON, an error is logged.
        """

        config_path = os.path.join(self.working_directory, 'config.json')

        try:
            with open(config_path, 'r') as config_file:
                config_data = json.load(config_file)
                logfile = config_data.get("logfile")

                today_log = f"{datetime.now().strftime('%Y-%m-%d')}.log"
                if logfile != today_log:
                    self.createLog(today_log)
                    config_data['logfile'] = today_log
                    with open(config_path, 'w') as config_file:
                        json.dump(config_data, config_file, indent=4)
                    self.logger.info(f"Logfile updated to {today_log}")
                else:
                    # Set up logging if today's log file exists
                    log_path = os.path.join(self.working_directory, logfile)
                    logging.basicConfig(
                        filename=log_path,
                        filemode='a',
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        level=logging.INFO
                    )
                    self.logger = logging.getLogger()
                    self.logger.info("Logging initialized with existing log file.")
        except FileNotFoundError:
            self.logger.error("Error: Configuration file not found.")
            self.createLog()
        except json.JSONDecodeError:
            self.logger.error("Error: Configuration file is improperly formatted.")

    def createLog(self, log_name=None):
        """
        Creates a new logfile based on the current date and updates the configuration file
        to reflect the newly created logfile.

        This method generates a new log file with a name formatted as 'YYYY-MM-DD.log',
        where the date corresponds to the current date. If no specific `log_name` is
        provided, the method defaults to using today's date. The new log file is saved
        within the working directory. After creating the file, the method updates
        the configuration file to register the name of the current logfile, ensuring that
        the configuration always points to the latest logfile.

        Functionalities:
        - Logfile Creation:
          A new logfile named in the format 'current-date.log' (e.g., '2024-04-27.log')
          is created in the working directory. If `log_name` is provided, it uses this
          name instead of the default date-based name. A confirmation message is logged.

        - Update Configuration:
          After creating the logfile, the method opens or creates the configuration file in the
          working directory, updating it to include the name of the new logfile under
          the 'logfile' key. This ensures that the configuration file accurately reflects the
          latest logfile.

        Error Handling:
        - If the configuration file exists but is not readable, an error will be logged.
        - If the working directory does not exist or cannot be accessed, the method
          may raise an `OSError`, depending on the system’s permissions.

        :param log_name: Optional; allows specifying a custom name for the logfile.
                         If not provided, the name defaults to today's date.
        :raises OSError: If the working directory or logfile cannot be created due
                         to system permissions.
        """

        log_name = log_name or f"{datetime.now().strftime('%Y-%m-%d')}.log"
        log_path = os.path.join(self.working_directory, log_name)
        open(log_path, 'w').close()  # Create the logfile

        # Set up logging for the new log file
        logging.basicConfig(
            filename=log_path,
            filemode='a',
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.INFO
        )
        self.logger = logging.getLogger()
        self.logger.info(f"New logfile created: {log_path}")

        # Update or create configuration to reflect the new logfile
        config_path = os.path.join(self.working_directory, 'config.json')
        config_data = {}
        if os.path.exists(config_path):
            with open(config_path, 'r') as config_file:
                config_data = json.load(config_file)
        config_data['logfile'] = log_name
        with open(config_path, 'w') as config_file:
            json.dump(config_data, config_file, indent=4)

    def dataConsistency(self):
        """
        Checks the consistency of the data in 'data.json' by validating structural
        and content requirements.

        This method ensures that the 'data.json' file contains the necessary structure
        and data for expected operation. It checks for required keys and verifies that
        each optimizer listed has corresponding arguments defined. Additionally, it
        ensures the presence of matrices data, as these are crucial for certain
        calculations. If any issues are found, appropriate warnings or errors are
        logged, and an error message is returned.

        Functionalities:
        - Basic Structure Check:
          Verifies that 'data.json' contains essential keys such as "qubits", "depth",
          "optimizers", "optimizer_arguments", and "matrices". If any of these are
          missing, an error message is logged and returned.

        - Check Optimizer List:
          Ensures that each optimizer listed in "optimizers" has corresponding entries
          in "optimizer_arguments". If an optimizer is missing arguments, a warning is
          logged for each missing argument but does not stop execution.

        - Matrix Content Check:
          Confirms that data is present in the "matrices" section. The exact structure
          of matrices is not strictly validated, but their presence is essential. If
          "matrices" is missing or empty, an error message is logged and returned.

        Error Handling:
        - If 'data.json' is missing, an error message is logged and returned.
        - If 'data.json' has an invalid JSON format, an error message is logged
          and returned.

        :return: A success message if all consistency checks pass, or an error message
                 indicating any structural or content issues.
        :raises FileNotFoundError: If 'data.json' does not exist.
        :raises json.JSONDecodeError: If 'data.json' is not in valid JSON format.
        """

        data_path = os.path.join(self.working_directory, 'data.json')
        try:
            with open(data_path, 'r') as f:
                data = json.load(f)

            required_keys = ["qubits", "depth", "optimizers", "optimizer_arguments", "matrices"]
            if not all(key in data for key in required_keys):
                error_msg = "Error: Incorrect structure in data.json"
                self.logger.error(error_msg)
                return error_msg

            for optimizer in data["optimizers"]:
                if optimizer not in data["optimizer_arguments"]:
                    self.logger.warning(f"Missing arguments for {optimizer}")

            if not data.get("matrices"):
                error_msg = "Error: Matrices not defined in data.json"
                self.logger.error(error_msg)
                return error_msg

            self.logger.info("data.json passed consistency checks.")
            return "data.json is consistent."

        except FileNotFoundError:
            error_msg = "Error: data.json not found."
            self.logger.error(error_msg)
            return error_msg
        except json.JSONDecodeError:
            error_msg = "Error: data.json is improperly formatted."
            self.logger.error(error_msg)
            return error_msg
        
    def create_train_file(self):
        train_file_path = os.path.join(self.working_directory, 'train.json')

        if os.path.exists(train_file_path):
            self.logger.error("train.json already exists.")
            return {"Error Code": 1199, "Message": "train.json already exists."}

        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        train_data = []
        train_data.append({"creation:": current_datetime})

        with open(train_file_path, 'w') as train_file:
            json.dump(train_data, train_file, indent=4)

    def move_json_file(self):
        
        train_file_path = os.path.join(self.working_directory, 'train.json')
        archive_dir = os.path.join(self.working_directory, 'archive')

        if not os.path.exists(archive_dir):
            os.makedirs(archive_dir)

        if not os.path.exists(train_file_path):
            self.logger.error("Error moving train.json, train.json not found.")
            return {"Error Code": 1188, "Message": "train.json not found."}
        
        try:
            os.rename(train_file_path, os.path.join(archive_dir, f"train_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"))
        except OSError as e:
            self.logger.error(f"Error moving train.json: {e}")
            #return {"Error Code": 1177, "Message": f"Failed to move train.json: {e}"}
