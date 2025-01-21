# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Project: LILY-QML
# Version: 2.0.0 LLY-DML
# Author: Joan Pujol (@supercabb), Claudia Zendejas-Morales (@clausia)
# Contact: info@lilyqml.de
# Website: www.lilyqml.de
# Contributors:
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++

import json
import os
from module.src.optimizer import Optimizer
import numpy as np
from module.src.circuit import Circuit
import logging
from datetime import datetime
from module.src.reader import Reader
from module.src.data import Data


class All:

    def __init__(self, config_base_path='var'):
        self.config_base_path = config_base_path
        self.data_json = None
        self.train_json = None
        self.shape_training_matrix = None
        self.actual_training_matrix = None
        ###VERIFY###
        #This should be a data.json parameter
        self.circuit_shots=1000
        self.logger = logging.getLogger()
                """
        Initializes the All class for managing training files and data processes.

        :param working_directory: Path where files such as train.json, config.json, and logs are managed.
        """
        self.working_directory = working_directory
        self.reader = Reader(working_directory)
        self.train_exists = os.path.exists(os.path.join(self.working_directory, 'train.json'))


    def get_state(self, name_matrix):
        """
        Retrieves the state representation for a given name from a predefined set of states.

        This function uses a hardcoded dictionary to map specific names to their corresponding state representations.
        The state representations are strings of binary digits, where each digit represents a different state.
        """       
        return self.data_json["matrices_states"][name_matrix]

    def optimize(self, iterations):
        """
        Optimize the training process using the specified number of iterations.
        This method reads configuration data from 'data.json' and 'train.json' files,
        initializes the training matrix, and performs training using different optimizers
        specified in the configuration.
        """
        
        data_path = os.path.join(self.config_base_path, 'data.json')
        try:
            with open(data_path, 'r') as f:
                self.data_json = json.load(f)
        except FileNotFoundError:
            error_msg = "Error: data.json not found."
            self.logger.error(error_msg)
            return error_msg          

        train_path = os.path.join(self.config_base_path, 'train.json')
        try:
            with open(train_path, 'r') as f:
                self.train_json = json.load(f)
        except FileNotFoundError:
            error_msg = "Error: train.json not found."
            self.logger.error(error_msg)
            return error_msg                   
        
        self.shape_training_matrix = (self.data_json['qubits'], self.data_json['depth']*3)
        
        for optimizer in self.data_json['optimizers']:
            self.logger.info("Training with optimizer: " + optimizer)
            for name_matrix, activation_matrix_list in self.data_json['matrices'].items():
            ###VERIFY###
            #In the main loop we iterate over the same class matrix activation
                for activation_matrix in activation_matrix_list:
                    self.train_activation_matrix(name_matrix, activation_matrix, optimizer, iterations)
        

    def init_training_matrix(self):
        """
        Initializes the training matrix.
        """

        ###VERIFY###
        #This matrix should be in data,json like initial_matrix
        #return self.data_json['initial_matrix']
        return np.zeros(self.shape_training_matrix).tolist()


    def get_actual_activation_matrix(self, optimizer):
        """
        Retrieves or initializes the actual activation matrix for a given optimizer.
        This method checks if the optimizer exists in the training configuration (train_json).
        If the optimizer does not exist, it initializes a new training matrix and stores it.
        If the optimizer exists but does not have an initial matrix, it initializes and stores it.
        If the optimizer and its initial matrix exist, it retrieves the initial matrix.
        """

        if not "optimizers" in self.train_json:
            self.train_json["optimizers"] = {}

        if not optimizer in self.train_json["optimizers"]:
            self.actual_training_matrix = self.init_training_matrix()
            self.train_json["optimizers"][optimizer] = {"initial_matrix": list(self.actual_training_matrix)}
        elif not "initial_matrix" in self.train_json["optimizers"][optimizer]:
            self.actual_training_matrix = self.init_training_matrix()
            self.train_json["optimizers"][optimizer]["initial_matrix"] = list(self.actual_training_matrix)
        else:
            self.actual_training_matrix = self.train_json["optimizers"][optimizer]["initial_matrix"]
        
        if len(self.actual_training_matrix) != self.shape_training_matrix[0] or any(len(row) != self.shape_training_matrix[1] for row in self.actual_training_matrix):
            error_msg = "Error: training matrix has incorrect dimensions."
            self.logger.error(error_msg)
            return error_msg
        
    def save_training_matrix(self, optimizer, name_matrix, iteration):
        """
        Saves the current training matrix to a JSON file.
        This function updates the training matrix for a given optimizer and matrix name
        at a specific iteration in the `train_json` dictionary and writes the updated
        dictionary to a JSON file.
        """

        if not "optimizers" in self.train_json:
            self.train_json["optimizers"] = {}

        if not optimizer in self.train_json["optimizers"]:
            self.train_json["optimizers"][optimizer] = {}

        if not "activation_matrix" in self.train_json["optimizers"][optimizer]:
            self.train_json["optimizers"][optimizer]["activation_matrix"] = {}

        if not name_matrix in self.train_json["optimizers"][optimizer]["activation_matrix"]:
            self.train_json["optimizers"][optimizer]["activation_matrix"][name_matrix] = {}

        if not "training_matrix" in self.train_json["optimizers"][optimizer]["activation_matrix"][name_matrix]:
            self.train_json["optimizers"][optimizer]["activation_matrix"][name_matrix]["training_matrix"] = {}

        self.train_json["optimizers"][optimizer]["activation_matrix"][name_matrix]["training_matrix"][iteration] = list(self.actual_training_matrix)

        self.logger.info("Saving in train.json training matrix for "+name_matrix+" with "+str(iteration)+" iterations.")
        
        with open(os.path.join(self.config_base_path, 'train.json'), 'w') as f:
            json.dump(self.train_json, f, indent=4)

    def save_final_matrix(self, optimizer, name_matrix):
        """
        Save the final training matrix to the train.json file.

        This method updates the train.json file with the final training matrix for a given optimizer and matrix name.
        It ensures that the necessary keys exist in the JSON structure before saving the matrix.
        """

        if not "optimizers" in self.train_json:
            self.train_json["optimizers"] = {}

        if not optimizer in self.train_json["optimizers"]:
            self.train_json["optimizers"][optimizer] = {}

        if not "activation_matrix" in self.train_json["optimizers"][optimizer]:
            self.train_json["optimizers"][optimizer]["activation_matrix"] = {}

        if not name_matrix in self.train_json["optimizers"][optimizer]["activation_matrix"]:
            self.train_json["optimizers"][optimizer]["activation_matrix"][name_matrix] = {}

        self.train_json["optimizers"][optimizer]["activation_matrix"]["final_ombc"] = list(self.actual_training_matrix)

        self.logger.info("Saving in train.json final matrix for "+name_matrix+".")

        with open(os.path.join(self.config_base_path, 'train.json'), 'w') as f:
            json.dump(self.train_json, f, indent=4)

    def save_target_state(self, optimizer, name_matrix, target_state):
        """
        Save the target state of a given optimizer and activation matrix to the train.json file.
        """

        if not "optimizers" in self.train_json:
            self.train_json["optimizers"] = {}

        if not optimizer in self.train_json["optimizers"]:
            self.train_json["optimizers"][optimizer] = {}

        if not "activation_matrix" in self.train_json["optimizers"][optimizer]:
            self.train_json["optimizers"][optimizer]["activation_matrix"] = {}

        if not name_matrix in self.train_json["optimizers"][optimizer]["activation_matrix"]:
            self.train_json["optimizers"][optimizer]["activation_matrix"][name_matrix] = {}

        self.train_json["optimizers"][optimizer]["activation_matrix"][name_matrix]["target_state"] = target_state

        self.logger.info("Saving in train.json target state for "+name_matrix+".")

        with open(os.path.join(self.config_base_path, 'train.json'), 'w') as f:
            json.dump(self.train_json, f, indent=4)


    def save_optimized_matrix(self, optimizer, name_matrix, optimized_matrix):
        """
        Save the optimized matrix for a given optimizer and matrix name into the train.json file.
        """

        if not "optimizers" in self.train_json:
            self.train_json["optimizers"] = {}

        if not optimizer in self.train_json["optimizers"]:
            self.train_json["optimizers"][optimizer] = {}

        if not "activation_matrix" in self.train_json["optimizers"][optimizer]:
            self.train_json["optimizers"][optimizer]["activation_matrix"] = {}

        if not name_matrix in self.train_json["optimizers"][optimizer]["activation_matrix"]:
            self.train_json["optimizers"][optimizer]["activation_matrix"][name_matrix] = {}

        self.train_json["optimizers"][optimizer]["activation_matrix"][name_matrix]["optimized_training_matrix"] = optimized_matrix.tolist()

        self.logger.info("Saving in train.json optimized matrix for "+name_matrix+".")

        with open(os.path.join(self.config_base_path, 'train.json'), 'w') as f:
            json.dump(self.train_json, f, indent=4)     

    def save_random_numbers(self, optimizer, name_matrix, random_numbers_dict):
        """
        Save random numbers associated with a specific optimizer and activation matrix to a JSON file.
        """

        if not "optimizers" in self.train_json:
            self.train_json["optimizers"] = {}

        if not optimizer in self.train_json["optimizers"]:
            self.train_json["optimizers"][optimizer] = {}

        if not "activation_matrix" in self.train_json["optimizers"][optimizer]:
            self.train_json["optimizers"][optimizer]["activation_matrix"] = {}

        if not name_matrix in self.train_json["optimizers"][optimizer]["activation_matrix"]:
            self.train_json["optimizers"][optimizer]["activation_matrix"][name_matrix] = {}

        if not "random_numbers" in self.train_json["optimizers"][optimizer]["activation_matrix"][name_matrix]:
            self.train_json["optimizers"][optimizer]["activation_matrix"][name_matrix]["random_numbers"] = []

        self.train_json["optimizers"][optimizer]["activation_matrix"][name_matrix]["random_numbers"].append(random_numbers_dict)

        self.logger.info("Saving in train.json random numbers for "+name_matrix+".")

        with open(os.path.join(self.config_base_path, 'train.json'), 'w') as f:
            json.dump(self.train_json, f, indent=4)

    def save_crostrain_summary(self, name_matrix, summary_crosstrain_dict):
        """
        Saves the cross-train summary to the train.json file.
        """

        if not "crosstrain_selection" in self.train_json:
            self.train_json["crosstrain_selection"] = {}

        self.train_json["crosstrain_selection"][name_matrix] = summary_crosstrain_dict

        self.logger.info("Saving in train.json crosstrain summary for "+name_matrix+".")

        with open(os.path.join(self.config_base_path, 'train.json'), 'w') as f:
            json.dump(self.train_json, f, indent=4)             
    
    def train_activation_matrix(self, name_matrix, activation_matrix, optimizer_name, iterations):
        """
        Trains the activation matrix using the specified optimizer and number of iterations.
        Args:
            name_matrix (str): The name of the matrix to be trained.
            activation_matrix (list): The activation matrix to be used for training.
            optimizer_name (str): The name of the optimizer to be used.
            iterations (int): The number of iterations for the training process.
        Returns:
            str or None: Returns an error message if an error occurs during optimization, otherwise None.
        Notes:
            - The method initializes the optimizer and saves the target state.
            - It runs the optimization process for the specified number of iterations.
            - If the optimization is successful, it saves the trained matrix and logs the training process.
            - It also performs cross-training and saves the optimized matrix and cross-training summary if available.
        """
       
        target_state = self.get_state(name_matrix)
        optimizer = Optimizer()

        self.save_target_state(optimizer_name, name_matrix, target_state)

        optimizer.start(optimizer_name, target_state)

        error = self.get_actual_activation_matrix(optimizer_name)

        if error is not None:
            return error
        
        for it in range(iterations):
            circuit = Circuit(self.data_json['qubits'], self.data_json['depth'], self.actual_training_matrix, activation_matrix, self.circuit_shots, "aer_simulator_tensor_network_gpu")
            circuit.run()
            measure = circuit.get_counts()
            ret = optimizer.optimize(measure, self.actual_training_matrix)

            if ret is None:
                error_msg = "Error: optimize method returned error."
                self.logger.error(error_msg)
                return error_msg
            
            self.actual_training_matrix = ret
            self.save_training_matrix(optimizer_name, name_matrix, it)

        self.save_training_matrix(optimizer_name, name_matrix)
        self.logger.info("Activation matrix "+name_matrix+" trained with "+str(self.iterations)+" iterations.")

        ret, crosstrained_matrix_summary = self.crosstraining(self, optimizer_name, optimizer, self.actual_training_matrix, name_matrix)

        if ret is not None:
            self.save_optimized_matrix(optimizer_name, name_matrix, ret)

        if crosstrained_matrix_summary is not None:
            self.save_crostrain_summary(name_matrix, crosstrained_matrix_summary)


        
    def crosstraining(self, optimizer_name, optimizer_class, current_training_matrix, name_actual_matrix):
        """
        Perform cross-training on the given training matrix using the specified optimizer.

        This function performs cross-training by iterating through a specified number of iterations,
        randomly selecting activation matrices, and applying the crosshelper method to optimize the
        training process. The results are logged and saved for further analysis.

        Args:
            optimizer_name (str): The name of the optimizer to be used for cross-training.
            optimizer_class (class): The class of the optimizer to be used for cross-training.
            current_training_matrix (dict): The current training matrix to be optimized.
            name_actual_matrix (str): The name of the actual matrix being trained.

        Returns:
            tuple: A tuple containing the optimized training matrix and a summary of the cross-training process.
                   Returns (None, None) if an error occurs during the cross-training process.
        """

        iteration_crosstrain_a = self.data_json["iteration_crosstrain_a"]
        iteration_crosstrain_b = self.data_json["iteration_crosstrain_b"]            

        matrix_target_states = {}
        matrix_final_ombc = {}
        current_tp = current_training_matrix

        activation_matrix["actual_matrix"] = name_actual_matrix
        crosstrained_matrix_summary = []


        for activation_matrix in self.train_json["optimizers"][optimizer_name]["activation_matrix"].keys():
            matrix_target_states[activation_matrix] = self.train_json["optimizers"][optimizer_name]["activation_matrix"][activation_matrix]["target_state"]
            matrix_final_ombc[activation_matrix] = self.train_json["optimizers"][optimizer_name]["activation_matrix"][activation_matrix]["final_ombc"]


        for it in range(iteration_crosstrain_a):
            ###VERIFY###
            #I'm not sure that we don't need to limit the intern_iter random range..
            intern_iter = np.random.randint()
            index_key_ap_matrix = np.random.randint(0, len(self.train_json["optimizers"][optimizer_name]["activation_matrix"]))

            #We are selecting a random matrix from the list of name_actual_matrix
            index_matrix = np.random.randint(0, len(self.data_json["matrices"]))

            self.logger.info("Crosstraining generated random numbers: "+str(intern_iter)+", "+str(index_key_ap_matrix), ", "+str(index_matrix))

            self.save_random_numbers(optimizer_name, name_actual_matrix, {"intern_iter":intern_iter, "index_key_ap_matrix":index_key_ap_matrix})

            selected_ap_matrix_name =list(self.train_json["optimizers"][optimizer_name]["activation_matrix"].keys())[index_key_ap_matrix]
            
            target_state_ap = matrix_target_states[selected_ap_matrix_name]

            crosstrained_matrix_summary.append({"selected_ap_matrix_name":selected_ap_matrix_name, "intern_iter":intern_iter, "target_state_ap":target_state_ap, "index_matrix":index_matrix})
            
            selected_ap = self.data_json["matrices"][index_matrix]

            ret = self.crosshelper(target_state_ap, selected_ap, selected_ap_matrix_name, current_tp, optimizer_name, optimizer_class, intern_iter, iteration_crosstrain_b, iteration_crosstrain_a)

            if ret is None:
                error_msg = "Error: optimize method returned error in crosshelper."
                self.logger.error(error_msg)
                return None, None

            current_tp = ret


        self.logger.info("Crosstraining for "+name_actual_matrix+" finished.")

        return current_tp, crosstrained_matrix_summary

            
    def crosshelper(self, target_state, selected_ap, selected_ap_matrix_name, current_tp, optimizer_name, optimizer_class, intern_iter, iteration_crosstrain_b, iteration_crosstrain_a):
        """
        Perform cross-training optimization on a given matrix.
        Args:
            target_state (str): The target state for the optimization.
            selected_ap (str): The selected activation matrix.
            selected_ap_matrix_name (str): The name of the activation matrix.
            current_tp (Any): The current training matrix.
            optimizer_name (str): The name of the optimizer to be used.
            optimizer_class (object): The optimizer class instance.
            intern_iter (int): The iter randomly generated.
            iteration_crosstrain_b (int): Parameter about iterations number.
            iteration_crosstrain_a (int): Parameter about iterations number.
        Returns:
            Any: The updated training parameter after optimization, or None if an error occurs.
        """

        if intern_iter > iteration_crosstrain_a:
            b_iterations = iteration_crosstrain_b
        else:
            b_iterations = intern_iter

        self.logger.info("Crosstraining optimizating matrix: "+selected_ap_matrix_name+" with "+str(b_iterations)+" iterations.")

        for it in range(b_iterations):
            circuit = Circuit(self.data_json['qubits'], self.data_json['depth'], current_tp, selected_ap, self.circuit_shots)
            circuit.run()
            measure = circuit.get_counts()

            optimizer_class.start(optimizer_name, target_state)
            ret = optimizer_class.optimize(measure, current_tp)

            if ret is None:
                error_msg = "Error: optimize method returned error in crosshelper."
                self.logger.error(error_msg)
                return None
            
            current_tp = ret

        return current_tp

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
      