# test.py

import os
import sys
import json
import logging
from datetime import datetime
import time
from module.reader import Reader
from module.data import Data
from module.circuit import Circuit
from module.optimizer import OptimizerManager

class TestSuite:
    def __init__(self, mode='comprehensive'):
        """
        Initializes the TestSuite with the specified mode.

        :param mode: 'comprehensive' or 'quick'
        """
        self.mode = mode
        self.var_folder = 'var'
        self.data_folder = os.path.join(self.var_folder, 'data')
        self.reports_folder = os.path.join(self.var_folder, 'reports')
        self.report_file = os.path.join(self.reports_folder, f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        self.ensure_directories()
        self.setup_logging()
        self.reader = Reader(
            data_json_path=os.path.join(self.var_folder, 'data.json'),
            train_json_path=os.path.join(self.var_folder, 'train.json'),
            log_db_path=os.path.join(self.var_folder, 'log.logdb'),
            create_train_json_on_init=False
        )
        self.load_and_prepare_data()

    def ensure_directories(self):
        """Ensures that necessary directories exist."""
        for directory in [self.var_folder, self.data_folder, self.reports_folder]:
            if not os.path.isdir(directory):
                os.makedirs(directory)
                print(f"Created directory: {directory}")

    def setup_logging(self):
        """Sets up logging for the TestSuite."""
        log_file = os.path.join(self.var_folder, 'test.log')
        logging.basicConfig(
            filename=log_file,
            filemode='a',
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.DEBUG
        )
        self.logger = logging.getLogger('TestSuite')
        self.logger.info(f"Initialized TestSuite in {self.mode} mode.")

    def load_and_prepare_data(self):
        """Loads data.json and train.json, prepares data for testing."""
        self.logger.info("Loading data.json and train.json for testing.")
        try:
            self.reader.load_data_json()
            if os.path.isfile(self.reader.train_json_path):
                self.reader.load_train_json()
            else:
                self.reader.create_train_json()
            self.logger.info("Data loaded successfully.")
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            print(f"Error loading data: {e}")
            sys.exit(1)

        # Adjust parameters based on mode
        if self.mode == 'comprehensive':
            self.logger.info("Setting parameters for comprehensive testing.")
            self.set_parameters(high=True)
        elif self.mode == 'quick':
            self.logger.info("Setting parameters for quick testing.")
            self.set_parameters(high=False)
        else:
            self.logger.error(f"Unknown mode: {self.mode}")
            print(f"Unknown mode: {self.mode}")
            sys.exit(1)

    def set_parameters(self, high=True):
        """Sets optimizer parameters based on the testing mode."""
        if high:
            # Set high parameters for comprehensive testing
            self.learning_rate = 0.05
            self.max_iterations = 200
            self.population_size = 50
            self.mutation_rate = 0.2
            self.num_particles = 50
            self.inertia = 0.7
            self.cognitive = 1.8
            self.social = 1.8
            self.initial_temperature = 5.0
            self.cooling_rate = 0.95
        else:
            # Set lower parameters for quick testing
            self.learning_rate = 0.01
            self.max_iterations = 50
            self.population_size = 20
            self.mutation_rate = 0.1
            self.num_particles = 20
            self.inertia = 0.5
            self.cognitive = 1.5
            self.social = 1.5
            self.initial_temperature = 1.0
            self.cooling_rate = 0.99

    def log_report(self, message):
        """Logs a message to both the report file and the logger."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.report_file, 'a') as f:
            f.write(f"[{timestamp}] {message}\n")
        self.logger.info(message)

    def run_all_tests(self):
        """Runs all tests in the TestSuite."""
        self.log_report("Starting all tests.")
        self.test_reader_loading()
        self.test_data_initialization()
        self.test_circuit_initialization()
        self.test_circuit_execution()
        self.test_optimizer_initialization()
        self.test_optimizer_execution()
        self.test_circuit_image_saving()
        self.test_training_matrix_saving()
        self.log_report("All tests completed.")

    def test_reader_loading(self):
        """Tests the Reader's ability to load data.json and train.json."""
        self.log_report("Testing Reader loading of data.json and train.json.")
        try:
            qubits = self.reader.qubits
            depth = self.reader.depth
            self.log_report(f"Reader loaded qubits: {qubits}, depth: {depth}.")
            print(f"Reader loaded qubits: {qubits}, depth: {depth}.")
            assert qubits is not None and depth is not None, "Qubits or depth not loaded."
            self.log_report("Reader loading test passed.")
        except AssertionError as ae:
            self.log_report(f"Reader loading test failed: {ae}")
            print(f"Reader loading test failed: {ae}")
        except Exception as e:
            self.log_report(f"Reader loading test encountered an error: {e}")
            print(f"Reader loading test encountered an error: {e}")

    def test_data_initialization(self):
        """Tests the Data class initialization and conversion."""
        self.log_report("Testing Data class initialization and activation matrix conversion.")
        try:
            data = Data(
                qubits=self.reader.qubits,
                depth=self.reader.depth,
                activation_matrices=[
                    np.array(self.reader.activation_matrices[label]) for label in self.reader.get_matrix_names()
                ],
                labels=self.reader.get_matrix_names(),
                logger=self.logger
            )
            training_matrix = data.create_training_matrix()
            data.validate_training_matrix(training_matrix)
            converted_matrices = data.convert_activation_matrices_to_2d()
            self.log_report("Data class initialization and conversion test passed.")
            print("Data class initialization and conversion test passed.")
        except AssertionError as ae:
            self.log_report(f"Data class test failed: {ae}")
            print(f"Data class test failed: {ae}")
        except Exception as e:
            self.log_report(f"Data class test encountered an error: {e}")
            print(f"Data class test encountered an error: {e}")

    def test_circuit_initialization(self):
        """Tests the Circuit class initialization."""
        self.log_report("Testing Circuit class initialization.")
        try:
            training_phases = self.reader.train_data.get('training_matrix', [])
            activation_phases = {}
            for label in self.reader.get_matrix_names():
                activation_phases[label] = self.reader.train_data.get('converted_activation_matrices', {}).get(label, [])
            circuit = Circuit(
                qubits=self.reader.qubits,
                depth=self.reader.depth,
                training_phases=training_phases,
                activation_phases=activation_phases[list(activation_phases.keys())[0]],
                shots=self.reader.shots
            )
            self.log_report("Circuit class initialization test passed.")
            print("Circuit class initialization test passed.")
        except AssertionError as ae:
            self.log_report(f"Circuit initialization test failed: {ae}")
            print(f"Circuit initialization test failed: {ae}")
        except Exception as e:
            self.log_report(f"Circuit initialization test encountered an error: {e}")
            print(f"Circuit initialization test encountered an error: {e}")

    def test_circuit_execution(self):
        """Tests running the Circuit and retrieving counts."""
        self.log_report("Testing Circuit execution and count retrieval.")
        try:
            training_phases = self.reader.train_data.get('training_matrix', [])
            activation_phases = {}
            for label in self.reader.get_matrix_names():
                activation_phases[label] = self.reader.train_data.get('converted_activation_matrices', {}).get(label, [])
            circuit = Circuit(
                qubits=self.reader.qubits,
                depth=self.reader.depth,
                training_phases=training_phases,
                activation_phases=activation_phases[list(activation_phases.keys())[0]],
                shots=self.reader.shots
            )
            circuit.run()
            counts = circuit.get_counts()
            self.log_report(f"Circuit executed successfully. Counts: {counts}")
            print(f"Circuit executed successfully. Counts: {counts}")
            assert counts, "No counts retrieved from circuit execution."
            self.log_report("Circuit execution test passed.")
        except AssertionError as ae:
            self.log_report(f"Circuit execution test failed: {ae}")
            print(f"Circuit execution test failed: {ae}")
        except Exception as e:
            self.log_report(f"Circuit execution test encountered an error: {e}")
            print(f"Circuit execution test encountered an error: {e}")

    def test_optimizer_initialization(self):
        """Tests the initialization of OptimizerManager and optimizers."""
        self.log_report("Testing OptimizerManager and optimizer initialization.")
        try:
            data = Data(
                qubits=self.reader.qubits,
                depth=self.reader.depth,
                activation_matrices=[
                    np.array(self.reader.activation_matrices[label]) for label in self.reader.get_matrix_names()
                ],
                labels=self.reader.get_matrix_names(),
                logger=self.logger
            )
            training_matrix = data.create_training_matrix()
            data.validate_training_matrix(training_matrix)
            converted_matrices = data.convert_activation_matrices_to_2d()
            self.reader.set_training_matrix(training_matrix.tolist())
            self.reader.train_data['converted_activation_matrices'] = converted_matrices
            optimizer_manager = OptimizerManager(
                reader=self.reader,
                data=data,
                train_file=self.reader.train_json_path,
                logger=self.logger
            )
            self.log_report("OptimizerManager and optimizers initialized successfully.")
            print("OptimizerManager and optimizers initialized successfully.")
        except Exception as e:
            self.log_report(f"Optimizer initialization test failed: {e}")
            print(f"Optimizer initialization test failed: {e}")

    def test_optimizer_execution(self):
        """Tests the execution of all optimizers."""
        self.log_report("Testing execution of all optimizers.")
        try:
            data = Data(
                qubits=self.reader.qubits,
                depth=self.reader.depth,
                activation_matrices=[
                    np.array(self.reader.activation_matrices[label]) for label in self.reader.get_matrix_names()
                ],
                labels=self.reader.get_matrix_names(),
                logger=self.logger
            )
            training_matrix = data.create_training_matrix()
            data.validate_training_matrix(training_matrix)
            converted_matrices = data.convert_activation_matrices_to_2d()
            self.reader.set_training_matrix(training_matrix.tolist())
            self.reader.train_data['converted_activation_matrices'] = converted_matrices
            optimizer_manager = OptimizerManager(
                reader=self.reader,
                data=data,
                train_file=self.reader.train_json_path,
                logger=self.logger
            )
            optimizer_manager.train_all_optimizers()
            self.log_report("All optimizers executed successfully.")
            print("All optimizers executed successfully.")
        except Exception as e:
            self.log_report(f"Optimizer execution test failed: {e}")
            print(f"Optimizer execution test failed: {e}")

    def test_circuit_image_saving(self):
        """Tests saving circuit images before and after optimization."""
        self.log_report("Testing saving of circuit images.")
        try:
            # Assume that images are saved during optimizer execution
            # Check if starter_before.png and starter_after.png exist
            starter_before = os.path.join(self.data_folder, 'starter_before.png')
            starter_after = os.path.join(self.data_folder, 'starter_after.png')
            if os.path.isfile(starter_before) and os.path.isfile(starter_after):
                self.log_report("Circuit images 'starter_before.png' and 'starter_after.png' exist.")
                print("Circuit images 'starter_before.png' and 'starter_after.png' exist.")
            else:
                self.log_report("Circuit images 'starter_before.png' and/or 'starter_after.png' are missing.")
                print("Circuit images 'starter_before.png' and/or 'starter_after.png' are missing.")
        except Exception as e:
            self.log_report(f"Circuit image saving test encountered an error: {e}")
            print(f"Circuit image saving test encountered an error: {e}")

    def test_training_matrix_saving(self):
        """Tests saving of training matrices before and after optimization."""
        self.log_report("Testing saving of training matrices.")
        try:
            with open(self.reader.train_json_path, 'r') as f:
                train_data = json.load(f)
            initial_matrix = train_data.get('initial_training_matrix', None)
            optimized_matrix = train_data.get('optimized_training_matrix', None)
            if initial_matrix and optimized_matrix:
                self.log_report("Initial and optimized training matrices are saved in train.json.")
                print("Initial and optimized training matrices are saved in train.json.")
            else:
                self.log_report("Initial and/or optimized training matrices are missing in train.json.")
                print("Initial and/or optimized training matrices are missing in train.json.")
        except Exception as e:
            self.log_report(f"Training matrix saving test encountered an error: {e}")
            print(f"Training matrix saving test encountered an error: {e}")

    def run_quick_tests(self):
        """Runs quick tests with reduced parameters."""
        self.log_report("Starting quick tests.")
        self.set_parameters(high=False)
        self.run_all_tests()
        self.log_report("Quick tests completed.")

    def run_comprehensive_tests(self):
        """Runs comprehensive tests with high parameters."""
        self.log_report("Starting comprehensive tests.")
        self.set_parameters(high=True)
        self.run_all_tests()
        self.log_report("Comprehensive tests completed.")

def main():
    """Main function to run the TestSuite."""
    while True:
        print("\n=== Test Suite Menu ===")
        print("1. Run Comprehensive Tests")
        print("2. Run Quick Tests")
        print("3. Exit")
        choice = input("Please select an option (1-3): ")

        if choice == '1':
            test_suite = TestSuite(mode='comprehensive')
            test_suite.run_comprehensive_tests()
            print(f"Comprehensive tests completed. Report saved at {test_suite.report_file}")
        elif choice == '2':
            test_suite = TestSuite(mode='quick')
            test_suite.run_quick_tests()
            print(f"Quick tests completed. Report saved at {test_suite.report_file}")
        elif choice == '3':
            print("Exiting Test Suite.")
            sys.exit()
        else:
            print("Invalid input. Please try again.")

if __name__ == "__main__":
    main()
