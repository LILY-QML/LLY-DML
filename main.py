import os
import sys
import json
from datetime import datetime
import time
from module.reader import Reader  # Import the Reader from module/reader.py
from module.data import Data      # Import the Data from module/data.py
from module.circuit import Circuit  # Import the Circuit from module/circuit.py
import numpy as np
import logging
from module.optimizer import OptimizerManager


# ANSI color codes for console output
class Colors:
    RED = '\033[91m'
    RESET = '\033[0m'

class DML:
    banner = r"""
 ____   __  __   _
|  _ \ |  \/  | | |
| | | || |\/| | | |
| |_| || |  | | | |___
|____/ |_|  |_| |_____|

    """
    info = """
LLY-DML - Part of the LILY Project - Version 1.6 Beta - info@lilyqml.de - lilyqml.de
"""

    def __init__(self):
        self.var_folder = 'var'
        self.data_file = os.path.join(self.var_folder, 'data.json')
        self.log_file = os.path.join(self.var_folder, 'log.logdb')
        self.train_file = os.path.join(self.var_folder, 'train.json')
        self.reports_folder = os.path.join(self.var_folder, 'reports')
        self.ensure_var_directory()

        # Initialize the Reader with create_train_json_on_init=False to prevent automatic creation
        self.reader = Reader(
            data_json_path=self.data_file,
            log_db_path=self.log_file,
            create_train_json_on_init=False  # Prevents automatic creation of train.json
        )
        self.train_json_found = os.path.isfile(self.train_file)

    def ensure_var_directory(self):
        # Create the var folder if it doesn't exist
        if not os.path.isdir(self.var_folder):
            os.makedirs(self.var_folder)
            print(f"Folder '{self.var_folder}' has been created.")
            self.reader.logger.debug(f"Folder '{self.var_folder}' has been created.")

    def clear_console(self):
        """Clears the console."""
        os.system('cls' if os.name == 'nt' else 'clear')

    def start(self):
        """Starts the program."""
        self.check_requirements()
        self.run_menu()

    def check_requirements(self):
        """Checks for the presence of necessary files and folders."""
        print("Checking necessary directories and files...\n")
        errors = False

        # Check if data.json exists (handled by Reader)
        if not os.path.isfile(self.data_file):
            print(f"Error: '{self.data_file}' was not found.")
            self.reader.logger.error(f"Error: '{self.data_file}' was not found.")
            errors = True
        else:
            print(f"File '{self.data_file}' exists.")
            self.reader.logger.debug(f"File '{self.data_file}' exists.")

        # Check if log.logdb exists (handled by Reader)
        if not os.path.isfile(self.log_file):
            print(f"Error: '{self.log_file}' was not found.")
            self.reader.logger.error(f"Error: '{self.log_file}' was not found.")
            errors = True
        else:
            print(f"File '{self.log_file}' exists.")
            self.reader.logger.debug(f"File '{self.log_file}' exists.")

        # Check if train.json exists
        if not self.train_json_found:
            print(f"Warning: '{self.train_file}' was not found.")
            self.reader.logger.warning(f"Warning: '{self.train_file}' was not found.")
        else:
            print(f"File '{self.train_file}' exists.")
            self.reader.logger.debug(f"File '{self.train_file}' exists.")

        if errors:
            print("\nPlease fix the above errors and restart the program.")
            self.reader.logger.error("Program terminated due to missing files.")
            sys.exit(1)
        else:
            if not self.train_json_found:
                print("\ntrain.json not found -- steps are limited - only new training possible.\n")
                self.reader.logger.info("train.json not found. Only new training possible.")
            else:
                print("\nAll necessary files and directories are present.\n")
                self.reader.logger.info("All necessary files and directories are present.")

    def run_menu(self):
        """Displays the main menu and processes user selection."""
        while True:
            self.clear_console()
            print(self.banner)
            print(self.info)
            if not self.train_json_found:
                print(f"{Colors.RED}train.json was not found. Only new training is possible.{Colors.RESET}\n")

            print("\n---\n")
            print("Select an option:")
            print("1. Information - Show more information about the reference files")
            print("2. Training - Choose a training method")
            if self.train_json_found:
                print("3. Report - Report options")
                print("4. Exit - Exit LLY-DML")
            else:
                # Display option 3 in red and mark as unavailable
                print(f"{Colors.RED}3. Report - Report options (not available){Colors.RESET}")
                print("4. Exit - Exit LLY-DML")

            choice = input("Please select an option (1-4): ")

            if choice == '1':
                self.information_menu()
            elif choice == '2':
                self.training_menu()
            elif choice == '3':
                if self.train_json_found:
                    self.report_menu()
                else:
                    print(f"{Colors.RED}Option 3 is currently not available.{Colors.RESET}")
                    self.reader.logger.warning("Attempted to select unavailable option 3.")
                    input("Press Enter to continue.")
            elif choice == '4':
                print("Exiting program. Goodbye!")
                self.reader.logger.info("Program exited by user.")
                sys.exit()
            else:
                print("Invalid input. Please try again.")
                self.reader.logger.warning(f"Invalid input in main menu: {choice}")
                input("Press Enter to continue.")

    def information_menu(self):
        """Displays the information menu."""
        while True:
            self.clear_console()
            print("\n---\n")
            print("INFORMATION")
            print("Choose the following options:")
            print("1. Base Reference - data.json")
            print("2. Final Reference - train.json")
            print("3. Back to Main Menu")
            choice = input("Please select an option (1-3): ")
            if choice == '1':
                self.show_data_json_info()
            elif choice == '2':
                self.show_train_json_info()
            elif choice == '3':
                break
            else:
                print("Invalid input. Please try again.")
                self.reader.logger.warning(f"Invalid input in information menu: {choice}")
                input("Press Enter to continue.")

    def show_data_json_info(self):
        """Displays information from data.json."""
        self.clear_console()
        print("\n---\n")
        if os.path.isfile(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                print("- data.json detected")
                # Example information; adjust as needed
                print(f"  Qubits: {data.get('qubits', 'Not specified')}")
                print(f"  Depth: {data.get('depth', 'Not specified')}")
                optimizers = data.get('optimizers', [])
                if isinstance(optimizers, dict):
                    optimizers_list = list(optimizers.keys())
                elif isinstance(optimizers, list):
                    optimizers_list = optimizers
                else:
                    optimizers_list = []
                print(f"  Optimizers: {', '.join(optimizers_list) if optimizers_list else 'Not specified'}")
            except json.JSONDecodeError:
                print("data.json is not properly formatted.")
                self.reader.logger.error("data.json is not properly formatted.")
            except Exception as e:
                print(f"An error occurred: {e}")
                self.reader.logger.error(f"An error occurred while displaying data.json: {e}")
        else:
            print("- data.json not found.")
            self.reader.logger.warning("Attempted to display data.json, but file was not found.")
        input("\nPress Enter to continue.")

    def show_train_json_info(self):
        """Displays information from train.json."""
        self.clear_console()
        print("\n---\n")
        if self.train_json_found and os.path.isfile(self.train_file):
            try:
                with open(self.train_file, 'r') as f:
                    data = json.load(f)
                print("- train.json detected")
                print(f"  Creation Date: {data.get('creation_date', 'Not specified')}")
                # Display training matrix dimensions
                training_matrix = np.array(data.get('training_matrix', []))
                if training_matrix.size != 0:
                    print(f"  Training Matrix Dimensions: {training_matrix.shape}")
                else:
                    print("  Training Matrix: Not defined")
                # Display converted activation matrices
                converted_activation_matrices = data.get('converted_activation_matrices', {})
                print(f"  Number of converted activation matrices: {len(converted_activation_matrices)}")
                for label, matrix in converted_activation_matrices.items():
                    matrix_np = np.array(matrix)
                    print(f"    {label} converted dimensions: {matrix_np.shape}")
                # Display optimizers
                optimizers = list(data.get('optimizers', {}).keys())
                if optimizers:
                    print(f"  Used Optimizers: {', '.join(optimizers)}")
                else:
                    print("  No optimizers used.")
                # Display simulation results
                simulation_results = data.get('simulation_results', {})
                if simulation_results:
                    print(f"  Simulation Results: {simulation_results}")
                else:
                    print("  No simulation results available.")
            except json.JSONDecodeError:
                print("train.json is not properly formatted.")
                self.reader.logger.error("train.json is not properly formatted.")
            except Exception as e:
                print(f"An error occurred: {e}")
                self.reader.logger.error(f"An error occurred while displaying train.json: {e}")
        else:
            print("- train.json not found.")
            self.reader.logger.warning("Attempted to display train.json, but file was not found.")
        input("\nPress Enter to continue.")

    def training_menu(self):
        """Displays the training menu and processes the selection."""
        while True:
            self.clear_console()
            print("\n---\n")
            print("TRAINING")
            print("Choose one of the following options:")
            print("1. Training of all optimizers")
            print("2. Training of specific optimizers")
            if self.train_json_found:
                print("3. Training of remaining optimizers")
                print("4. Back to Main Menu")
            else:
                # Display option 3 in red and mark as unavailable
                print(f"{Colors.RED}3. Training of remaining optimizers (not available){Colors.RESET}")
                print("4. Back to Main Menu")

            choice = input("Please select an option (1-4): ")
            if choice == '1':
                self.training_all_optimizers()
            elif choice == '2':
                self.training_specific_optimizers()
            elif choice == '3':
                if self.train_json_found:
                    self.training_remaining_optimizers()
                else:
                    print(f"{Colors.RED}Option 3 is currently not available.{Colors.RESET}")
                    self.reader.logger.warning("Attempted to select unavailable option 3.")
                    input("Press Enter to continue.")
            elif choice == '4':
                print("Returning to Main Menu.")
                self.reader.logger.info("Selected to return to Main Menu.")
                input("Press Enter to continue.")
                break
            else:
                print("Invalid input. Please try again.")
                self.reader.logger.warning(f"Invalid input in training menu: {choice}")
                input("Press Enter to continue.")

    def training_all_optimizers(self):
        """Performs the training of all optimizers."""
        self.clear_console()
        print("\nTRAINING OF ALL OPTIMIZERS")
        if self.train_json_found:
            print("An existing train.json was found.")
            print("Do you want to save the current train.json or delete and reset it?")
            print("1. Save")
            print("2. Reset and delete")
            choice = input("Please select an option (1-2): ")
            if choice == '1':
                self.save_train_json()
            elif choice == '2':
                self.reset_train_json()
            else:
                print("Invalid input. Returning to training menu.")
                self.reader.logger.warning(f"Invalid input when saving/deleting train.json: {choice}")
                input("Press Enter to continue.")
                return
        else:
            print("No existing train.json found. A new one will be created.")
            # Manually create train.json
            self.reader.create_train_json()
            self.train_json_found = True
            self.reader.logger.info("train.json was manually created.")
            input("Press Enter to continue.")

        # Proceed with training
        self.execute_training(all_optimizers=True)

    def save_train_json(self):
        """Saves the existing train.json under a new name."""
        self.clear_console()
        print("Enter a name under which the data should be saved:")
        name = input("Name: ").strip()
        if not name:
            print("Name cannot be empty. Returning to training menu.")
            self.reader.logger.warning("Empty name when saving train.json.")
            input("Press Enter to continue.")
            return
        save_path = os.path.join(self.var_folder, f"{name}_train.json")
        if os.path.isfile(self.train_file):
            try:
                os.rename(self.train_file, save_path)
                self.reader.logger.info(f"train.json was saved as {save_path}.")
                print(f"train.json was saved as {save_path}.")
                self.train_json_found = False  # train.json was moved, now it's missing
            except Exception as e:
                self.reader.logger.error(f"Error saving train.json: {e}")
                print(f"Error saving train.json: {e}")
        else:
            print("No train.json found to save.")
            self.reader.logger.warning("Attempted to save train.json, but file was not found.")
        input("Press Enter to continue.")

    def reset_train_json(self):
        """Deletes the existing train.json and resets it."""
        self.clear_console()
        if os.path.isfile(self.train_file):
            try:
                os.remove(self.train_file)
                self.reader.logger.info("train.json was deleted and reset.")
                print("train.json was deleted and reset.")
                self.train_json_found = False  # Update the flag
            except Exception as e:
                self.reader.logger.error(f"Error deleting train.json: {e}")
                print(f"Error deleting train.json: {e}")
        else:
            print("No train.json found to delete.")
            self.reader.logger.warning("Attempted to delete train.json, but file was not found.")
        input("Press Enter to continue.")

    def execute_training(self, all_optimizers=True, specific_optimizers=None):
        """Executes the training procedure using the OptimizerManager."""
        self.clear_console()
        print("\nTRAINING")
        steps = [
            "1. Creating the training matrix",
            "2. Building the environment",
            "3. Checking matrix formats",
            "4. Saving the circuit as an image",
            "5. Training the model with various optimizers"
        ]
        for step in steps:
            self.reader.logger.info(step)
            print(step)
            self.show_loading()
            if step == "1. Creating the training matrix":
                self.create_training_matrix()
            elif step == "2. Building the environment":
                self.build_environment()
            elif step == "3. Checking matrix formats":
                self.check_matrices_format()
            elif step == "4. Saving the circuit as an image":
                self.save_circuit_image()

        if all_optimizers:
            # Initialize the OptimizerManager
            try:
                optimizer_manager = OptimizerManager(
                    reader=self.reader,
                    data=self.data,  # Ensure self.data is correctly initialized
                    train_file=self.train_file,
                    logger=self.reader.logger
                )
                # Start training all optimizers
                optimizer_manager.train_all_optimizers()
            except Exception as e:
                self.reader.logger.error(f"Error initializing OptimizerManager: {e}")
                print(f"Error initializing OptimizerManager: {e}")
                sys.exit(1)
        elif specific_optimizers:
            # If specific optimizers are to be trained
            # Implement similar logic as in OptimizerManager
            pass

        input("Press Enter to continue.")

    def create_training_matrix(self):
        """Creates the training matrix based on qubits and depth from data.json."""
        qubits = self.reader.data.get('qubits', 8)
        depth = self.reader.data.get('depth', 8)

        # Log and display the parameters
        self.reader.logger.info(f"Creating training matrix with qubits: {qubits} and depth: {depth}")
        print(f"Creating training matrix with qubits: {qubits} and depth: {depth}")

        # Load activation matrices
        activation_matrices_json = self.reader.activation_matrices
        activation_matrices = []
        for label in self.reader.get_matrix_names():
            try:
                matrix_list = activation_matrices_json[label]
                matrix_array = np.array(matrix_list)
                self.reader.logger.debug(f"{label} loaded with shape: {matrix_array.shape}")
                activation_matrices.append(matrix_array)
                self.reader.logger.info(f"{label} successfully loaded.")
            except KeyError:
                error_msg = f"Error: {label} is not present in activation_matrices."
                print(error_msg)
                self.reader.logger.error(error_msg)
                sys.exit(1)

        # **Create the training matrix and assign to self.training_matrix**
        self.training_matrix = np.random.rand(qubits, 3 * depth)
        self.reader.logger.debug(f"Created training matrix has shape: {self.training_matrix.shape}")

        # Save training matrix in train.json
        self.reader.set_training_matrix(self.training_matrix.tolist())
        self.reader.logger.info("Training matrix successfully saved in train.json.")

        self.reader.logger.info(f"Training matrix created with shape {self.training_matrix.shape}")
        print(f"Training matrix created with shape {self.training_matrix.shape}")

        # **Create Data instance and assign to self.data**
        self.data = Data(qubits=qubits, depth=depth, activation_matrices=activation_matrices)
        self.reader.logger.debug(f"Data object created with qubits: {qubits}, depth: {depth}")

    def build_environment(self):
        """Builds the environment by creating and executing the circuit."""
        self.clear_console()
        print("\nBUILDING THE ENVIRONMENT")
        qubits = self.reader.qubits
        depth = self.reader.depth

        # Create training phases from the training matrix without transposing
        training_phases = self.training_matrix.tolist()  # Shape: (9, 24)

        # Create activation phases with zeros and correct dimensions
        activation_phases = [[0.0 for _ in range(depth * 3)] for _ in range(qubits)]  # Shape: (9, 24)

        # Initialize and execute the circuit
        try:
            self.circuit = Circuit(
                qubits=qubits,
                depth=depth,
                training_phases=training_phases,
                activation_phases=activation_phases,
                shots=1024  # Number of measurements; can be read from data.json
            )
            self.reader.logger.info("Circuit successfully initialized.")
            print("Circuit successfully initialized.")
        except Exception as e:
            self.reader.logger.error(f"Error initializing the circuit: {e}")
            print(f"Error initializing the circuit: {e}")
            sys.exit(1)


    def check_matrices_format(self):
        """Überprüft, ob die Aktivierungsmatrizen das gleiche Format wie die Trainingsmatrix haben."""
        self.reader.logger.info("Checking the formats of the matrices.")
        print("Checking the formats of the matrices.")

        # Lade die Trainingsmatrix
        training_matrix = np.array(self.reader.train_data.get('training_matrix', []))
        if training_matrix.size == 0:
            error_msg = "Training matrix not found or empty."
            self.reader.logger.error(error_msg)
            print(error_msg)
            sys.exit(1)

        # Shape der Trainingsmatrix
        training_shape = training_matrix.T.shape  # Transponierte Form
        self.reader.logger.debug(f"Shape of the training matrix (after transpose): {training_shape}")

        # Lade die Aktivierungsmatrizen
        converted_activation_matrices = self.reader.train_data.get('converted_activation_matrices', {})
        if not converted_activation_matrices:
            error_msg = "No converted activation matrices found."
            self.reader.logger.error(error_msg)
            print(error_msg)
            sys.exit(1)

        # Überprüfe jede Aktivierungsmatrix
        for label, matrix in converted_activation_matrices.items():
            activation_matrix = np.array(matrix)
            activation_shape = activation_matrix.shape
            self.reader.logger.debug(f"Shape of activation matrix {label}: {activation_shape}")

            if activation_shape != training_shape:
                error_msg = f"Activation matrix {label} has a different shape than the training phases."
                self.reader.logger.error(error_msg)
                print(error_msg)
                sys.exit(1)
            else:
                self.reader.logger.info(f"Activation matrix {label} has the same shape as the training phases.")

        print("All activation matrices have the same format as the training phases.")
        self.reader.logger.info("All activation matrices have the same format as the training phases.")


    def save_circuit_image(self):
        """Speichert den Schaltkreis als Bild im var-Ordner."""
        self.reader.logger.info("Saving the circuit as an image.")
        print("Saving the circuit as an image.")

        # Überprüfe, ob der Circuit vorhanden ist
        if not hasattr(self, 'circuit'):
            error_msg = "Circuit not found. Please ensure the circuit has been created."
            self.reader.logger.error(error_msg)
            print(error_msg)
            return

        try:
            # Pfad zum Speichern des Bildes
            image_path = os.path.join(self.var_folder, 'circuit.png')
            # Speichere das Bild
            self.circuit.circuit.draw(output='mpl', filename=image_path)
            self.reader.logger.info(f"Circuit image saved at {image_path}.")
            print(f"Circuit image saved at {image_path}.")
        except Exception as e:
            error_msg = f"Error saving the circuit image: {e}"
            self.reader.logger.error(error_msg)
            print(error_msg)


    def update_train_json_with_simulation(self, training_matrix, converted_activation_matrices, optimizers, simulation_results):
        """Updates or creates train.json with the new training matrix, converted activation matrices, optimizers, and simulation results."""
        # If train_data already exists, use it; otherwise, create a new dictionary
        if not self.reader.train_data:
            train_data = {
                "creation_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "training_matrix": training_matrix,
                "converted_activation_matrices": {},
                "optimizers": {},
                "simulation_results": {},
                "circuit": ""  # Initialize the field for the circuit
            }
        else:
            train_data = self.reader.train_data

        # Update the training matrix and other fields
        train_data['creation_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        train_data['training_matrix'] = training_matrix
        train_data['converted_activation_matrices'].update({
            label: matrix for label, matrix in converted_activation_matrices.items()
        })
        train_data['optimizers'].update({opt: {"status": "trained"} for opt in optimizers})
        train_data['simulation_results'] = simulation_results  # Update simulation results

        # Circuit is already in train_data (from train_first_optimizer) if applicable

        try:
            with open(self.train_file, 'w') as f:
                json.dump(train_data, f, indent=4)
            self.reader.logger.info(f"train.json has been updated at '{self.train_file}'.")
            self.train_json_found = True  # Update the flag
            self.reader.train_data = train_data  # Update train_data in Reader
        except Exception as e:
            self.reader.logger.error(f"Error updating train.json: {e}")
            sys.exit(1)

    def get_optimizers_from_data_json(self):
        """Reads the optimizers from data.json."""
        try:
            with open(self.data_file, 'r') as f:
                data = json.load(f)
            optimizers = data.get('optimizers', [])
            if isinstance(optimizers, dict):
                optimizers = list(optimizers.keys())
            elif isinstance(optimizers, list):
                optimizers = optimizers
            else:
                optimizers = []
            if not optimizers:
                raise ValueError("No optimizers found in data.json.")
            return optimizers
        except Exception as e:
            self.reader.logger.error(f"Error reading optimizers from data.json: {e}")
            print(f"Error reading optimizers from data.json: {e}")
            sys.exit(1)

    def training_specific_optimizers(self):
        """Performs the training of specific optimizers."""
        self.clear_console()
        print("\nTRAINING OF SPECIFIC OPTIMIZERS")
        available_optimizers = self.get_optimizers_from_data_json()
        print("Available Optimizers:")
        for idx, opt in enumerate(available_optimizers, start=1):
            print(f"{idx}. {opt}")
        choices = input("Select the optimizers you want to train (e.g., 1,3): ")
        selected = choices.split(',')
        selected_optimizers = []
        try:
            for choice in selected:
                idx = int(choice.strip()) - 1
                if 0 <= idx < len(available_optimizers):
                    selected_optimizers.append(available_optimizers[idx])
                else:
                    print(f"Invalid optimizer number: {choice}")
                    self.reader.logger.warning(f"Invalid optimizer number selected: {choice}")
        except ValueError:
            print("Invalid input. Returning to training menu.")
            self.reader.logger.warning(f"Invalid input when selecting optimizers: {choices}")
            input("Press Enter to continue.")
            return
        if not selected_optimizers:
            print("No valid optimizers selected. Returning to training menu.")
            self.reader.logger.warning("No valid optimizers selected.")
            input("Press Enter to continue.")
            return
        # Proceed with training
        self.execute_training(all_optimizers=False, specific_optimizers=selected_optimizers)

    def training_remaining_optimizers(self):
        """Performs the training of the remaining optimizers."""
        self.clear_console()
        print("\nTRAINING OF REMAINING OPTIMIZERS")
        if not self.train_json_found:
            print("No existing train.json found. Please perform a training first.")
            self.reader.logger.warning("Attempted to train remaining optimizers, but train.json is missing.")
            input("Press Enter to continue.")
            return
        try:
            with open(self.train_file, 'r') as f:
                data = json.load(f)
                existing_optimizers = list(data.get('optimizers', {}).keys())
        except json.JSONDecodeError as e:
            error_msg = f"Error parsing train.json: {e}"
            print("train.json is not properly formatted.")
            self.reader.logger.error(error_msg)
            input("Press Enter to continue.")
            return
        except Exception as e:
            error_msg = f"An error occurred while loading train.json: {e}"
            print(f"An error occurred: {e}")
            self.reader.logger.error(error_msg)
            input("Press Enter to continue.")
            return
        all_optimizers = self.get_optimizers_from_data_json()
        remaining_optimizers = [opt for opt in all_optimizers if opt not in existing_optimizers]
        if not remaining_optimizers:
            print("No remaining optimizers to train.")
            self.reader.logger.info("No remaining optimizers to train.")
            input("Press Enter to continue.")
            return
        print("Remaining Optimizers:")
        for idx, opt in enumerate(remaining_optimizers, start=1):
            print(f"{idx}. {opt}")
        choices = input("Select the optimizers you want to train (e.g., 1,2): ")
        selected = choices.split(',')
        selected_optimizers = []
        try:
            for choice in selected:
                idx = int(choice.strip()) - 1
                if 0 <= idx < len(remaining_optimizers):
                    selected_optimizers.append(remaining_optimizers[idx])
                else:
                    print(f"Invalid optimizer number: {choice}")
                    self.reader.logger.warning(f"Invalid optimizer number selected: {choice}")
        except ValueError:
            print("Invalid input. Returning to training menu.")
            self.reader.logger.warning(f"Invalid input when selecting optimizers: {choices}")
            input("Press Enter to continue.")
            return
        if not selected_optimizers:
            print("No valid optimizers selected. Returning to training menu.")
            self.reader.logger.warning("No valid optimizers selected.")
            input("Press Enter to continue.")
            return
        # Proceed with training
        self.execute_training(all_optimizers=False, specific_optimizers=selected_optimizers)

    def report_menu(self):
        """Displays the report menu and processes the selection."""
        while True:
            self.clear_console()
            print("\n---\n")
            print("REPORT")
            print("1. Create a report based on existing data")
            print("2. Back to Main Menu")
            choice = input("Please select an option (1-2): ")
            if choice == '1':
                self.create_report()
            elif choice == '2':
                break
            else:
                print("Invalid input. Please try again.")
                self.reader.logger.warning(f"Invalid input in report menu: {choice}")
                input("Press Enter to continue.")

    def create_report(self):
        """Creates a report based on the existing data."""
        self.clear_console()
        name = input("Please enter a name for the report: ").strip()
        if not name:
            print("Name cannot be empty. Returning to report menu.")
            self.reader.logger.warning("Empty name when creating a report.")
            input("Press Enter to continue.")
            return
        report_path = os.path.join(self.reports_folder, f"{name}_report.txt")
        try:
            # Ensure the reports folder exists if it doesn't already
            if not os.path.isdir(self.reports_folder):
                os.makedirs(self.reports_folder)
                self.reader.logger.info(f"Folder '{self.reports_folder}' has been created.")

            with open(report_path, 'w') as f:
                f.write("LLY-DML Report\n")
                f.write(f"Created on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                # Add specific data here
                if os.path.isfile(self.data_file):
                    f.write("Base Reference - data.json\n")
                    with open(self.data_file, 'r') as df:
                        data = json.load(df)
                        f.write(json.dumps(data, indent=4))
                else:
                    f.write("data.json not found.\n")
                f.write("\n")
                if self.train_json_found and os.path.isfile(self.train_file):
                    f.write("Final Reference - train.json\n")
                    with open(self.train_file, 'r') as tf:
                        train = json.load(tf)
                        f.write(json.dumps(train, indent=4))
                else:
                    f.write("train.json not found.\n")
            self.reader.logger.info(f"Report created at {report_path}.")
            print(f"Report successfully created at {report_path}.")
        except Exception as e:
            self.reader.logger.error(f"Error creating the report: {e}")
            print(f"Error creating the report: {e}")
        input("Press Enter to continue.")

    def show_loading(self, symbols=['|', '/', '-', '\\'], iterations=10, delay=0.1):
        """Simulates a loading screen."""
        for i in range(iterations):
            print(symbols[i % len(symbols)], end='\r')
            time.sleep(delay)
        print(' ', end='\r')

if __name__ == "__main__":
    # Entry point of the program
    dml = DML()
    dml.start()
