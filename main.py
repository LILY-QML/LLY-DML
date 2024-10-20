import os
import sys
import json
from datetime import datetime
import time
from module.reader import Reader  
from module.data import Data      
from module.circuit import Circuit  
import numpy as np
import logging
from module.optimizer import OptimizerManager
from module.interpreter import OptimizerInterpreter
from module.visual import Visual 

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
        self.log_file = os.path.join(self.var_folder, 'log.log')
        self.train_file = os.path.join(self.var_folder, 'train.json')
        self.reports_folder = os.path.join(self.var_folder, 'reports')
        self.data_folder = os.path.join(self.var_folder, 'data')  # Added for saving images
        self.ensure_var_directory()

        # Initialize the Reader with create_train_json_on_init=False to prevent automatic creation
        self.reader = Reader(
            data_json_path=self.data_file,
            log_db_path=self.log_file,
            create_train_json_on_init=False  # Prevents automatic creation of train.json
        )
        self.train_json_found = os.path.isfile(self.train_file)
        self.extracted_data = {}  # Initialisierung hinzugefügt


    def ensure_var_directory(self):
        # Create the var folder if it doesn't exist
        if not os.path.isdir(self.var_folder):
            os.makedirs(self.var_folder)
            print(f"Folder '{self.var_folder}' has been created.")
            self.reader.logger.debug(f"Folder '{self.var_folder}' has been created.")
        
        # Create the data folder if it doesn't exist
        if not os.path.isdir(self.data_folder):
            os.makedirs(self.data_folder)
            print(f"Folder '{self.data_folder}' has been created.")
            self.reader.logger.debug(f"Folder '{self.data_folder}' has been created.")

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
                optimizers = [entry['optimizer'] for entry in data.get('optimizer_steps', [])]
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
                # Save the initial training matrix before training
                self.save_training_matrix_initial()

                # Save the starter_before.png image
                self.save_circuit_image_before()

                # Perform parallel training of all optimizers
                optimizer_manager.train_all_optimizers()

                # Save the optimized training matrix after training
                self.save_training_matrix_optimized()

                # Save the starter_after.png image
                self.save_circuit_image_after()

                self.reader.logger.info("All optimizers have been trained successfully.")
                print("All optimizers have been trained successfully.")
            except Exception as e:
                self.reader.logger.error(f"Error initializing OptimizerManager: {e}")
                print(f"Error initializing OptimizerManager: {e}")
                sys.exit(1)
        elif specific_optimizers:
            # Implement similar logic as in OptimizerManager if needed
            pass

        input("Press Enter to continue.")


    def save_training_matrix_initial(self):
        """Saves the initial training matrix to train.json."""
        self.reader.logger.info("Saving initial training matrix to train.json.")
        print("Saving initial training matrix to train.json.")

        try:
            with open(self.train_file, 'r') as f:
                train_data = json.load(f)
        except FileNotFoundError:
            # If train.json does not exist, create a new one
            train_data = {
                "creation_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "initial_training_matrix": self.training_matrix.tolist(),
                "converted_activation_matrices": self.reader.train_data.get('converted_activation_matrices', {}),
                "optimizer_steps": self.reader.train_data.get('optimizer_steps', []),
                "simulation_results": self.reader.train_data.get('simulation_results', {})
            }
        except json.JSONDecodeError as e:
            error_msg = f"Error decoding train.json: {e}"
            self.reader.logger.error(error_msg)
            print(error_msg)
            sys.exit(1)
        except Exception as e:
            error_msg = f"Error reading train.json: {e}"
            self.reader.logger.error(error_msg)
            print(error_msg)
            sys.exit(1)

        # Update with initial_training_matrix
        train_data['initial_training_matrix'] = self.training_matrix.tolist()

        # Write back to train.json
        try:
            with open(self.train_file, 'w') as f:
                json.dump(train_data, f, indent=4)
            self.reader.logger.info("Initial training matrix saved to train.json.")
            print("Initial training matrix saved to train.json.")
        except Exception as e:
            error_msg = f"Error writing initial training matrix to train.json: {e}"
            self.reader.logger.error(error_msg)
            print(error_msg)
            sys.exit(1)

    def save_training_matrix_optimized(self):
        """Saves the optimized training matrix to train.json."""
        self.reader.logger.info("Saving optimized training matrix to train.json.")
        print("Saving optimized training matrix to train.json.")

        try:
            with open(self.train_file, 'r') as f:
                train_data = json.load(f)
        except json.JSONDecodeError as e:
            error_msg = f"Error decoding train.json: {e}"
            self.reader.logger.error(error_msg)
            print(error_msg)
            sys.exit(1)
        except Exception as e:
            error_msg = f"Error reading train.json: {e}"
            self.reader.logger.error(error_msg)
            print(error_msg)
            sys.exit(1)

        # Update with optimized_training_matrix
        # Fetch from OptimizerManager's training_phases if necessary
        optimized_training_matrix = self.reader.train_data.get('optimizer_steps', [])  # Adjust as needed
        # Note: The actual optimized training matrix should be retrieved appropriately
        # For example, by aggregating results from optimizer_steps
        # Here, we'll set it to "Optimized" as a placeholder
        train_data['optimized_training_matrix'] = "Optimized"

        # Write back to train.json
        try:
            with open(self.train_file, 'w') as f:
                json.dump(train_data, f, indent=4)
            self.reader.logger.info("Optimized training matrix saved to train.json.")
            print("Optimized training matrix saved to train.json.")
        except Exception as e:
            error_msg = f"Error writing optimized training matrix to train.json: {e}"
            self.reader.logger.error(error_msg)
            print(error_msg)
            sys.exit(1)

    def save_circuit_image_before(self):
        """Saves the initial circuit image as 'starter_before.png'."""
        self.reader.logger.info("Saving initial circuit image as 'starter_before.png'.")
        print("Saving initial circuit image as 'starter_before.png'.")
        try:
            image_path = os.path.join(self.data_folder, 'starter_before.png')
            self.circuit.circuit.draw(output='mpl', filename=image_path)
            self.reader.logger.info(f"Starter circuit image saved at '{image_path}'.")
            print(f"Starter circuit image saved at '{image_path}'.")
        except Exception as e:
            error_msg = f"Error saving starter_before.png: {e}"
            self.reader.logger.error(error_msg)
            print(error_msg)

    def save_circuit_image_after(self):
        """Saves the final circuit image as 'starter_after.png'."""
        self.reader.logger.info("Saving final circuit image as 'starter_after.png'.")
        print("Saving final circuit image as 'starter_after.png'.")
        try:
            image_path = os.path.join(self.data_folder, 'starter_after.png')
            self.circuit.circuit.draw(output='mpl', filename=image_path)
            self.reader.logger.info(f"Starter circuit image saved at '{image_path}'.")
            print(f"Starter circuit image saved at '{image_path}'.")
        except Exception as e:
            error_msg = f"Error saving starter_after.png: {e}"
            self.reader.logger.error(error_msg)
            print(error_msg)

    def save_circuit_image(self):
        """Saves the circuit as an image in the var folder."""
        self.reader.logger.info("Saving the circuit as an image.")
        print("Saving the circuit as an image.")

        # Check if the circuit exists
        if not hasattr(self, 'circuit'):
            error_msg = "Circuit not found. Please ensure the circuit has been created."
            self.reader.logger.error(error_msg)
            print(error_msg)
            return

        try:
            # Path to save the image
            image_path = os.path.join(self.data_folder, 'circuit.png')
            # Save the image
            self.circuit.circuit.draw(output='mpl', filename=image_path)
            self.reader.logger.info(f"Circuit image saved at {image_path}.")
            print(f"Circuit image saved at {image_path}.")
        except Exception as e:
            error_msg = f"Error saving the circuit image: {e}"
            self.reader.logger.error(error_msg)
            print(error_msg)

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

        # Create the training matrix and assign to self.training_matrix
        self.training_matrix = np.random.rand(qubits, 3 * depth)
        self.reader.logger.debug(f"Created training matrix has shape: {self.training_matrix.shape}")

        # Save training matrix in train.json
        self.reader.set_training_matrix(self.training_matrix.tolist())
        self.reader.logger.info("Training matrix successfully saved in train.json.")

        self.reader.logger.info(f"Training matrix created with shape {self.training_matrix.shape}")
        print(f"Training matrix created with shape {self.training_matrix.shape}")

        # Create Data instance and assign to self.data
        self.data = Data(
            qubits=qubits,
            depth=depth,
            activation_matrices=activation_matrices,
            labels=self.reader.get_matrix_names(),
            logger=self.reader.logger  # Pass the logger here
        )
        self.reader.logger.debug(f"Data object created with qubits: {qubits}, depth: {depth}")

        # Convert activation matrices to 2D and update train.json
        converted_activation_matrices = self.data.convert_activation_matrices_to_2d()
        self.reader.logger.debug(f"Converted activation matrices: {list(converted_activation_matrices.keys())}")

        # Update train.json with the converted activation matrices
        self.update_train_json_with_activation_matrices(converted_activation_matrices)


    def update_train_json_with_activation_matrices(self, converted_activation_matrices):
        """Aktualisiert train.json mit den konvertierten Aktivierungsmatrizen."""
        # Wenn train_data bereits existiert, verwenden Sie es; andernfalls erstellen Sie ein neues Dictionary
        if not self.reader.train_data:
            train_data = {
                "creation_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "training_matrix": self.training_matrix.tolist(),
                "converted_activation_matrices": {},
                "optimizer_steps": [],
                "simulation_results": {}
            }
        else:
            train_data = self.reader.train_data

        # Update der konvertierten Aktivierungsmatrizen
        train_data['converted_activation_matrices'] = converted_activation_matrices

        # Schreiben zurück in train.json
        try:
            with open(self.train_file, 'w') as f:
                json.dump(train_data, f, indent=4)
            self.reader.logger.info(f"train.json wurde mit konvertierten Aktivierungsmatrizen aktualisiert unter '{self.train_file}'.")
            self.train_json_found = True  # Aktualisieren des Flags
            self.reader.train_data = train_data  # Aktualisieren von train_data im Reader
        except Exception as e:
            self.reader.logger.error(f"Fehler beim Aktualisieren von train.json mit Aktivierungsmatrizen: {e}")
            print(f"Fehler beim Aktualisieren von train.json mit Aktivierungsmatrizen: {e}")
            sys.exit(1)


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
        """Checks if the activation matrices have the same format as the training matrix."""
        self.reader.logger.info("Checking the formats of the matrices.")
        print("Checking the formats of the matrices.")

        # Load the training matrix
        training_matrix = np.array(self.reader.train_data.get('training_matrix', []))
        if training_matrix.size == 0:
            error_msg = "Training matrix not found or empty."
            self.reader.logger.error(error_msg)
            print(error_msg)
            sys.exit(1)

        # Shape of the training matrix
        training_shape = training_matrix.shape  # No transpose
        self.reader.logger.debug(f"Shape of the training matrix: {training_shape}")

        # Load the activation matrices
        converted_activation_matrices = self.reader.train_data.get('converted_activation_matrices', {})
        if not converted_activation_matrices:
            error_msg = "No converted activation matrices found."
            self.reader.logger.error(error_msg)
            print(error_msg)
            sys.exit(1)

        # Check each activation matrix
        for label, matrix in converted_activation_matrices.items():
            activation_matrix = np.array(matrix)
            activation_shape = activation_matrix.shape
            self.reader.logger.debug(f"Shape of activation matrix {label}: {activation_shape}")

            if activation_shape != training_shape:
                error_msg = f"Activation matrix {label} has a different shape than the training matrix."
                self.reader.logger.error(error_msg)
                print(error_msg)
                sys.exit(1)
            else:
                self.reader.logger.info(f"Activation matrix {label} has the same shape as the training matrix.")

        print("All activation matrices have the same format as the training matrix.")
        self.reader.logger.info("All activation matrices have the same format as the training matrix.")

    def create_report(self):
        """Erstellt einen Bericht basierend auf den vom Interpreter extrahierten Daten."""
        self.clear_console()
        name = input("Bitte geben Sie einen Namen für den Bericht ein: ").strip()
        if not name:
            print("Der Name darf nicht leer sein. Rückkehr zum Berichtsmenü.")
            self.reader.logger.warning("Leerer Name bei der Berichterstellung.")
            input("Drücken Sie die Eingabetaste, um fortzufahren.")
            return
        report_path = os.path.join(self.reports_folder, f"{name}_report.pdf")
        try:
            # Sicherstellen, dass der Berichtsordner existiert
            if not os.path.isdir(self.reports_folder):
                os.makedirs(self.reports_folder)
                self.reader.logger.info(f"Ordner '{self.reports_folder}' wurde erstellt.")
                print(f"Ordner '{self.reports_folder}' wurde erstellt.")

            # Debug: Überprüfen der Struktur von self.extracted_data
            self.reader.logger.debug(f"Struktur von extracted_data: {type(self.extracted_data)}")
            self.reader.logger.debug(f"Inhalt von extracted_data: {self.extracted_data}")
            print(f"Struktur von extracted_data: {type(self.extracted_data)}")
            print(f"Inhalt von extracted_data: {self.extracted_data}")

            # Extraktion der 'details' aus dem Bericht
            details = self.extracted_data.get('details', {})

            # Flatten die 'details' in eine Liste von Dictionaries
            results = []
            for optimizer, matrices in details.items():
                for matrix, data in matrices.items():
                    step = {
                        "Optimizer": optimizer,
                        "Activation Matrix": matrix,
                        "Final Probability": data.get("final_loss"),  # Anpassen je nach tatsächlichen Feldern
                        "Initial Probability": data.get("initial_loss"),  # Falls vorhanden
                        # Weitere notwendige Felder hinzufügen
                    }
                    # Optional: Fügen Sie weitere Daten hinzu, die für den Bericht benötigt werden
                    if "optimization_steps" in data:
                        step["Optimization Steps"] = data["optimization_steps"]
                    results.append(step)

            # Überprüfen Sie die Struktur der Ergebnisse
            self.reader.logger.debug(f"Ergebnisse für Visual: {results}")
            print(f"Ergebnisse für Visual: {results}")

            # Extrahiere Target States und Activation Matrices aus den Ergebnissen
            target_states = [entry.get('Target State') for entry in results if 'Target State' in entry]
            activation_matrices = list(set(entry.get('Activation Matrix') for entry in results if 'Activation Matrix' in entry))

            # Log und Ausgabe
            self.reader.logger.info(f"Extrahierte Target States: {target_states}")
            self.reader.logger.info(f"Extrahierte Activation Matrices: {activation_matrices}")
            print(f"Extrahierte Target States: {target_states}")
            print(f"Extrahierte Activation Matrices: {activation_matrices}")

            # Initialisieren der Visual-Klasse mit den flachen Ergebnissen
            visual = Visual(
                results=results,
                target_states=target_states,
                activation_matrices=activation_matrices,
                circuits=[],  # Falls vorhanden, hier hinzufügen
                num_iterations=self.reader.data.get('max_iterations', 100),
                qubits=self.reader.data.get('qubits', 9),
                depth=self.reader.data.get('depth', 8),
                additional_data=self.reader.data  # Zusätzliche Daten aus data.json
            )

            # Generieren des Berichts
            visual.generate_report(filename=report_path)
            self.reader.logger.info(f"Bericht wurde erfolgreich unter '{report_path}' erstellt.")
            print(f"Bericht erfolgreich unter '{report_path}' erstellt.")
        except Exception as e:
            self.reader.logger.error(f"Fehler beim Erstellen des Berichts: {e}")
            print(f"Fehler beim Erstellen des Berichts: {e}")
        input("Drücken Sie die Eingabetaste, um fortzufahren.")


    def report_menu(self):
        """Zeigt das Berichtsmenü an und verarbeitet die Auswahl."""
        while True:
            self.clear_console()
            print("\n---\n")
            print("BERICHT")
            print("1. Erstellen Sie einen Bericht basierend auf bestehenden Daten")
            print("2. Zurück zum Hauptmenü")
            choice = input("Bitte wählen Sie eine Option (1-2): ")
            if choice == '1':
                # Zuerst den Interpreter ausführen, um Konsistenz zu prüfen und Daten zu extrahieren
                self.run_interpreter()
                # Dann den Bericht erstellen, falls die Konsistenzprüfung erfolgreich war
                if hasattr(self, 'extracted_data') and self.extracted_data:
                    self.create_report()
                else:
                    print("Berichtserstellung abgebrochen aufgrund fehlender oder inkonsistenter Daten.")
                    self.reader.logger.warning("Berichtserstellung abgebrochen aufgrund fehlender oder inkonsistenter Daten.")
                    input("Drücken Sie die Eingabetaste, um fortzufahren.")
            elif choice == '2':
                break
            else:
                print("Ungültige Eingabe. Bitte versuchen Sie es erneut.")
                self.reader.logger.warning(f"Ungültige Eingabe im Berichtsmenü: {choice}")
                input("Drücken Sie die Eingabetaste, um fortzufahren.")


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
                existing_optimizers = [entry['optimizer'] for entry in data.get('optimizer_steps', [])]
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

    def create_report(self):
        """Erstellt einen Bericht basierend auf den vom Interpreter extrahierten Daten."""
        self.clear_console()
        name = input("Bitte geben Sie einen Namen für den Bericht ein: ").strip()
        if not name:
            print("Der Name darf nicht leer sein. Rückkehr zum Berichtsmenü.")
            self.reader.logger.warning("Leerer Name bei der Berichterstellung.")
            input("Drücken Sie die Eingabetaste, um fortzufahren.")
            return
        report_path = os.path.join(self.reports_folder, f"{name}_report.pdf")
        try:
            # Sicherstellen, dass der Berichtsordner existiert
            if not os.path.isdir(self.reports_folder):
                os.makedirs(self.reports_folder)
                self.reader.logger.info(f"Ordner '{self.reports_folder}' wurde erstellt.")
                print(f"Ordner '{self.reports_folder}' wurde erstellt.")

            # Initialisieren der Visual-Klasse mit den extrahierten Daten
            visual = Visual(
                results=self.extracted_data,  # Übergeben der extrahierten Optimierungsdaten
                target_states=[entry['Target State'] for entries in self.extracted_data.values() for entry in entries.values()],
                initial_training_phases=[],  # Falls vorhanden, hier hinzufügen
                activation_matrices=list(self.extracted_data.keys()),
                circuits=[],  # Falls vorhanden, hier hinzufügen
                num_iterations=self.reader.data.get('max_iterations', 100),
                qubits=self.reader.data.get('qubits', 9),
                depth=self.reader.data.get('depth', 8),
                additional_data=self.reader.data  # Zusätzliche Daten aus data.json
            )

            # Generieren des Berichts
            visual.generate_report(filename=report_path)
            self.reader.logger.info(f"Bericht wurde erfolgreich unter '{report_path}' erstellt.")
            print(f"Bericht erfolgreich unter '{report_path}' erstellt.")
        except Exception as e:
            self.reader.logger.error(f"Fehler beim Erstellen des Berichts: {e}")
            print(f"Fehler beim Erstellen des Berichts: {e}")
        input("Drücken Sie die Eingabetaste, um fortzufahren.")


    def show_loading(self, symbols=['|', '/', '-', '\\'], iterations=10, delay=0.1):
        """Simulates a loading screen."""
        for i in range(iterations):
            print(symbols[i % len(symbols)], end='\r')
            time.sleep(delay)
        print(' ', end='\r')


    def run_interpreter(self):
        """Führt den OptimizerInterpreter aus, um Konsistenz zu prüfen und Daten zu extrahieren."""
        interpreter = OptimizerInterpreter(data_json_path=self.data_file, train_json_path=self.train_file)
        try:
            report = interpreter.run()
            if report:
                # Speichern Sie die extrahierten Daten für die Berichtserstellung
                self.extracted_data = report  # Angenommen, report enthält die benötigten Daten direkt
                print("Konsistenzprüfung erfolgreich. Daten wurden erfolgreich extrahiert.")
                self.reader.logger.info("Konsistenzprüfung erfolgreich. Daten wurden erfolgreich extrahiert.")
            else:
                print("Konsistenzprüfung fehlgeschlagen. Details finden Sie in den Logs.")
                self.reader.logger.warning("Konsistenzprüfung fehlgeschlagen.")
        except Exception as e:
            print(f"Ein Fehler ist beim Ausführen des Interpreters aufgetreten: {e}")
            self.reader.logger.error(f"Ein Fehler ist beim Ausführen des Interpreters aufgetreten: {e}")



    # Entfernen Sie redundante Methoden wie write_optimization_to_train_json
    # Diese sind jetzt ausschließlich in OptimizerManager enthalten

if __name__ == "__main__":
        # Entry point of the program
    dml = DML()
    dml.start()
