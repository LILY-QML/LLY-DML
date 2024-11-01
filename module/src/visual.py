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
        create_train_json_on_init=True
    ):
        """
        Initializes the Reader class, loads data from JSON files, sets up logging, and performs consistency checks.
        
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
        self.optimizers = []
        self.activation_matrices = {}
        
        # Additional attributes
        self.qubits = None
        self.depth = None
        
        # Setup logging
        self.setup_logging()

        # Check files and load JSON data
        self.check_files(create_train_json_on_init)
        self.load_data_json()
        self.load_train_json()

        # Perform consistency checks after loading
        self.check_consistency()

    def load_data_json(self):
        """
        Loads data from the data JSON file and assigns values to class attributes.
        """
        try:
            with open(self.data_json_path, 'r') as f:
                self.data_config = json.load(f)
            
            # Load general parameters
            self.qubits = self.data_config.get('qubits')
            self.depth = self.data_config.get('depth')
            self.optimizers = self.data_config.get('optimizers', [])
            self.activation_matrices = self.data_config.get('matrices', {})

            # Assign other variables from data.json (optional based on the actual JSON structure)
            self.optimizer_arguments = self.data_config.get('optimizer_arguments', {})
            self.learning_rate = self.data_config.get('learning_rate')
            self.shots = self.data_config.get('shots')
            self.max_iterations = self.data_config.get('max_iterations')

        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing data.json: {e}")
            sys.exit(1)

    def check_consistency(self):
        """
        Performs consistency checks on the data loaded from data.json:
        - Ensures every optimizer has corresponding arguments.
        - Ensures all matrices have the correct dimensions (qubits x depth).
        """
        # Check if each optimizer has corresponding arguments
        for optimizer in self.optimizers:
            if optimizer not in self.optimizer_arguments:
                error_msg = f"Optimizer '{optimizer}' is listed but no corresponding arguments found."
                self.logger.error(error_msg)
                sys.exit(1)

        # Check matrix dimensions for all matrices
        for matrix_name, matrix in self.activation_matrices.items():
            if len(matrix) != self.qubits:
                error_msg = f"Matrix '{matrix_name}' has {len(matrix)} rows, but qubits is set to {self.qubits}."
                self.logger.error(error_msg)
                sys.exit(1)
            for row in matrix:
                if len(row) != self.depth:
                    error_msg = f"Matrix '{matrix_name}' has a row with {len(row)} columns, but depth is set to {self.depth}."
                    self.logger.error(error_msg)
                    sys.exit(1)
        
        self.logger.info("All consistency checks passed.")

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

    def load_train_json(self):
        """
        Loads data from train.json, or initializes an empty dictionary if train.json does not exist.
        """
        if os.path.exists(self.train_json_path):
            try:
                with open(self.train_json_path, 'r') as f:
                    content = f.read()
                    if not content.strip():
                        self.train_data = {}
                    else:
                        self.train_data = json.loads(content)
            except json.JSONDecodeError as e:
                self.logger.error(f"Error parsing train.json: {e}")
                sys.exit(1)
        else:
            self.train_data = {}
            self.logger.info("train.json does not exist.")

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
        with open(self.train_json_path, 'w') as f:
            json.dump(self.train_data, f, indent=4)
        self.logger.info(f"train.json created at {self.train_json_path}")
