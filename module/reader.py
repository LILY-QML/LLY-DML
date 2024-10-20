# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Project: LILY-QML
# Version: 1.6 LLY-DML
# Author: Leon Kaiser
# Contact: info@lilyqml.de
# Website: www.lilyqml.de
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++

import json
import os
import sys
from datetime import datetime
import logging

class Reader:
    def __init__(
        self,
        data_json_path='var/data.json',
        train_json_path='var/train.json',
        log_db_path='var/log.logdb',
        create_train_json_on_init=True  # New parameter to control train.json creation
    ):
        """
        Initializes the Reader class, loads data from JSON files, and sets up logging.
        
        :param data_json_path: Path to the data JSON file.
        :param train_json_path: Path to the train JSON file.
        :param log_db_path: Path to the log database file.
        :param create_train_json_on_init: Whether to create train.json if it doesn't exist.
        """
        self.data_json_path = data_json_path
        self.train_json_path = train_json_path
        self.log_db_path = log_db_path
        self.data_config = {}
        self.train_data = {}
        self.data_optimizer = []
        self.train_optimizer = []
        
        # Initialize self.data as an empty dictionary
        self.data = {}

        # Variables for JSON properties
        self.qubits = None
        self.depth = None
        self.learning_rate = None
        self.shots = None
        self.max_iterations = None
        self.population_size = None
        self.mutation_rate = None
        self.num_particles = None
        self.inertia = None
        self.cognitive = None
        self.social = None
        self.initial_temperature = None
        self.cooling_rate = None
        self.optimizers = []
        self.activation_matrices = {}
        
        # Setup logging configuration
        self.setup_logging()
        
        self.check_files(create_train_json_on_init)  # Pass the new parameter
        self.load_data_json()
        self.load_train_json()

    def load_data_json(self):
        """
        Loads data from the data JSON file, assigns configuration variables, and logs results.
        """
        try:
            with open(self.data_json_path, 'r') as f:
                self.data_config = json.load(f)
            
            # Assign variables from data_config
            self.qubits = self.data_config.get('qubits')
            self.depth = self.data_config.get('depth')
            self.learning_rate = self.data_config.get('learning_rate')
            self.shots = self.data_config.get('shots')
            self.max_iterations = self.data_config.get('max_iterations')
            self.population_size = self.data_config.get('population_size')
            self.mutation_rate = self.data_config.get('mutation_rate')
            self.num_particles = self.data_config.get('num_particles')
            self.inertia = self.data_config.get('inertia')
            self.cognitive = self.data_config.get('cognitive')
            self.social = self.data_config.get('social')
            self.initial_temperature = self.data_config.get('initial_temperature')
            self.cooling_rate = self.data_config.get('cooling_rate')
            self.optimizers = self.data_config.get('optimizers', [])
            
            # Store activation matrices in a dictionary
            self.activation_matrices = self.data_config.get('activation_matrices', {})

            # Set the loaded data in self.data for other classes to access
            self.data = self.data_config  
            
            # Debug outputs
            self.logger.debug("Loaded optimizers from data.json: %s", self.optimizers)
            self.logger.debug("Loaded activation matrices: %s", self.activation_matrices.keys())
            print("Loaded activation matrices:", self.activation_matrices.keys())
        
        except json.JSONDecodeError as e:
            error_msg = f"Error parsing data.json: {e}"
            print(error_msg)
            self.logger.error(error_msg)
            sys.exit(1)

    def setup_logging(self):
        """
        Sets up logging configuration for the Reader.
        """
        if not os.path.exists('var'):
            os.makedirs('var')
        logging.basicConfig(
            filename=self.log_db_path,
            filemode='a',
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.INFO
        )
        self.logger = logging.getLogger()
        self.logger.debug("Logging started.") 

        # Set the logging level for matplotlib specifically to WARNING
        logging.getLogger('matplotlib').setLevel(logging.WARNING)

    def check_files(self, create_train_json_on_init):
        """
        Checks the existence of necessary files, creates directories, and optionally creates train.json.
        
        :param create_train_json_on_init: Whether to create train.json if it does not exist.
        """
        if not os.path.exists(self.data_json_path):
            error_msg = f"Error: data.json not found at {self.data_json_path}. The program will exit."
            print(error_msg)
            self.logger.error(error_msg)
            sys.exit(1)
        
        if not os.path.exists('var'):
            os.makedirs('var')
            self.logger.debug("Created 'var' directory.")
        
        if not os.path.exists(self.train_json_path) and create_train_json_on_init:
            self.create_train_json()
        
        if not os.path.exists(self.log_db_path):
            open(self.log_db_path, 'a').close()  # Create an empty log file
            self.logger.debug("Created log file.")

    def create_train_json(self):
        """
        Creates a new train.json file with initial values.
        """
        self.train_data = {
            "creation_date": datetime.now().isoformat(),
            "data_json": self.data_json_path,
            "optimizers": {},
            "training_matrix": None,
            "converted_activation_matrices": {},
            "simulation_results": {}
        }
        print("Creating train.json with the following content:")
        print(json.dumps(self.train_data, indent=4))
        self.logger.debug("Creating train.json with the following content:")
        self.logger.debug(json.dumps(self.train_data, indent=4))
        with open(self.train_json_path, 'w') as f:
            json.dump(self.train_data, f, indent=4)
        print(f"train.json created at {self.train_json_path}")
        self.logger.info(f"train.json created at {self.train_json_path}")

    def load_train_json(self):
        """
        Loads data from train.json, or initializes an empty dictionary if train.json does not exist.
        """
        if os.path.exists(self.train_json_path):
            try:
                with open(self.train_json_path, 'r') as f:
                    content = f.read()
                    self.logger.debug("Content of train.json:")
                    self.logger.debug(content)
                    print("Content of train.json:")
                    print(content)
                    if not content.strip():
                        error_msg = "Error: train.json is empty."
                        print(error_msg)
                        self.logger.error(error_msg)
                        self.train_data = {}
                        self.train_optimizer = []
                    else:
                        self.train_data = json.loads(content)
                        self.train_optimizer = list(self.train_data.get('optimizers', {}).keys())
                        self.logger.debug("Already executed optimizers from train.json: %s", self.train_optimizer)
                        print("Already executed optimizers from train.json:", self.train_optimizer)
            except json.JSONDecodeError as e:
                error_msg = f"Error parsing train.json: {e}"
                print(error_msg)
                self.logger.error(error_msg)
                sys.exit(1)
            except Exception as e:
                error_msg = f"An unexpected error occurred while loading train.json: {e}"
                print(f"An error occurred: {e}")
                self.logger.error(error_msg)
                sys.exit(1)
        else:
            self.train_data = {}
            self.train_optimizer = []
            self.logger.info("train.json does not exist.")
            print("train.json does not exist.")

    def set_training_matrix(self, training_matrix):
        """
        Saves the training matrix to train.json.
        
        :param training_matrix: The training matrix to save.
        """
        self.train_data['training_matrix'] = training_matrix
        self.train_data['creation_date'] = datetime.now().isoformat()
        try:
            with open(self.train_json_path, 'w') as f:
                json.dump(self.train_data, f, indent=4)
            success_msg = "Training matrix successfully saved to train.json."
            print(success_msg)
            self.logger.info(success_msg)
        except Exception as e:
            error_msg = f"Error saving the training matrix to train.json: {e}"
            print(error_msg)
            self.logger.error(error_msg)

    def get_matrix_names(self):
        """
        Returns the names of all activation matrices.

        :return: A list of names of the activation matrices.
        """
        return list(self.activation_matrices.keys())
