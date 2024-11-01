import unittest
import json
import os
from datetime import datetime

from module.src.circuit import Circuit  # Assuming the Circuit class is implemented in this path

# Pfad zur config.json-Datei
config_path = 'var/config.json'

# Funktion zum Schreiben in die Logdatei
def log_message(message, level="INFO", error_code=None):
    try:
        # Lade die Konfiguration aus config.json
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)

        # Hole den Pfad der Logdatei aus der Konfiguration
        log_file = config.get('log_file', 'default.log')

        # Erstelle den vollständigen Pfad zur Logdatei im Ordner 'var'
        log_path = os.path.join('var', log_file)

        # Formatierung der Log-Einträge
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"

        # Wenn ein Fehlercode vorhanden ist, füge ihn hinzu
        if error_code:
            log_entry += f" [Error Code: {error_code}]"

        # Schreibe die Nachricht in die Logdatei
        with open(log_path, 'a') as log:
            log.write(log_entry + '\n')

    except Exception as e:
        print(f'Error while writing to log: {e}')


class TestCircuit(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Trenner am Anfang der Test-Suite
        log_message("===================== Circuit Tests ===================", level="START")

    def setUp(self):
        # Initialisiere Daten vor jedem Test
        self.circuit = Circuit()
        self.valid_data = {
            "qubits": 2,
            "depth": 3,
            "activation_matrices": [[0, 1], [1, 0], [1, 1]],
            "training_matrices": [[0, 1], [1, 1], [0, 0]]
        }
        self.invalid_data = {
            "qubits": 0,  # Ungültige Qubits
            "depth": 0,   # Ungültige Tiefe
            "activation_matrices": [],
            "training_matrices": []
        }
        log_message("Setup completed with initial valid and invalid data.", level="INFO")

    def test_read_data(self):
        log_message("Starting test_read_data.", level="INFO")
        try:
            # Führt die Methode read_data aus
            result = self.circuit.read_data('train.json')
            self.assertEqual(result, 'success', 'Error 4111: No success message found')
            self.assertIsNotNone(self.circuit.qubits, 'Error 4112: Qubit value is 0 or missing')
            self.assertIsNotNone(self.circuit.depth, 'Error 4113: Depth value is 0 or missing')
            self.assertTrue(len(self.circuit.activation_matrices) > 0, 'Error 4114: Activation matrix has incorrect dimensions')
            self.assertTrue(len(self.circuit.training_matrices) > 0, 'Error 4115: Training matrix is inconsistent or missing')

            # Zusätzliche Schritte: Dateien verschieben und Errorcode überprüfen
            self.circuit.move_file('train.json', 'test/var_arv')
            result_code = self.circuit.check_error_code()
            self.assertEqual(result_code, 1020, 'Error 4116: Wrong error code instead of 1060 or 1020')
            log_message("test_read_data passed successfully.", level="SUCCESS")

        except AssertionError as e:
            log_message(str(e), level="ERROR", error_code=4111)

    def test_convert_input_data(self):
        log_message("Starting test_convert_input_data.", level="INFO")
        try:
            input_data = self.circuit.convert_input_data('named')
            self.assertEqual(input_data[0][0], 0, 'Error 1044: Input data format is incorrect')
            self.assertEqual(len(input_data), self.circuit.depth, 'Error 1044: Incorrect number of rows')
            self.assertEqual(len(input_data[0]), self.circuit.qubits, 'Error 1044: Incorrect number of columns')

            # Validiere Matrizen
            self.assertEqual(input_data[0][0], 0, 'First element should be 0')
            self.assertEqual(input_data[1][1], 0, 'Second element should be 0')
            log_message("test_convert_input_data passed successfully.", level="SUCCESS")

        except KeyError as e:
            log_message(str(e), level="ERROR", error_code=1044)

    def test_is_matrix_consistent(self):
        log_message("Starting test_is_matrix_consistent.", level="INFO")
        try:
            consistent_matrix = [[1, 0], [0, 1]]
            result = self.circuit._is_matrix_consistent(consistent_matrix)
            self.assertTrue(result, 'Error 4117: Consistent matrix should return True')

            inconsistent_matrix = [[1], [0, 1]]
            result = self.circuit._is_matrix_consistent(inconsistent_matrix)
            self.assertFalse(result, 'Error 4118: Inconsistent matrix should return False')

            log_message("test_is_matrix_consistent passed successfully.", level="SUCCESS")

        except AssertionError as e:
            log_message(str(e), level="ERROR", error_code=4117)

    def test_is_empty_matrix(self):
        log_message("Starting test_is_empty_matrix.", level="INFO")
        try:
            empty_matrix = [[0, 0], [0, 0]]
            result = self.circuit._is_empty_matrix(empty_matrix)
            self.assertTrue(result, 'Error 4119: Matrix full of zeros should return True')

            non_empty_matrix = [[0, 1], [1, 0]]
            result = self.circuit._is_empty_matrix(non_empty_matrix)
            self.assertFalse(result, 'Error 4120: Matrix with non-zero values should return False')

            log_message("test_is_empty_matrix passed successfully.", level="SUCCESS")

        except AssertionError as e:
            log_message(str(e), level="ERROR", error_code=4119)

    def test_check_input_data(self):
        log_message("Starting test_check_input_data.", level="INFO")
        try:
            valid_input_data = {"qubits": 2, "depth": 3}
            result = self.circuit.check_input_data(valid_input_data)
            self.assertTrue(result, 'Error 4121: Valid input data should return True')

            invalid_input_data = {"qubits": None, "depth": None}
            result = self.circuit.check_input_data(invalid_input_data)
            self.assertFalse(result, 'Error 4122: Invalid input data should return False')

            log_message("test_check_input_data passed successfully.", level="SUCCESS")

        except TypeError as e:
            log_message(str(e), level="ERROR", error_code=4121)

    def test_create_L_gate(self):
        log_message("Starting test_create_L_gate.", level="INFO")
        try:
            valid_entry = {'qubits': 2, 'depth': 3}
            result, code = self.circuit.create_L_gate(valid_entry)
            self.assertEqual(code, 2060, 'Error 4123: Successful L-gate creation should return a valid success code')

            invalid_entry = {'qubits': None, 'depth': None}
            result, code = self.circuit.create_L_gate(invalid_entry)
            self.assertEqual(code, 1050, 'Error 4124: Failed L-gate creation should return a valid error code')

            log_message("test_create_L_gate passed successfully.", level="SUCCESS")

        except AssertionError as e:
            log_message(str(e), level="ERROR", error_code=4123)

    def test_create_initial_circuit(self):
        log_message("Starting test_create_initial_circuit.", level="INFO")
        try:
            result, code = self.circuit.create_initial_circuit('test_matrix')
            self.assertEqual(code, 2066, 'Error 4125: Successful initial circuit creation should return success code 2066')

            result, code = self.circuit.create_initial_circuit(None)
            self.assertEqual(code, 1066, 'Error 4126: Failed circuit creation should return an error code')

            log_message("test_create_initial_circuit passed successfully.", level="SUCCESS")

        except TypeError as e:
            log_message(str(e), level="ERROR", error_code=4125)

    def test_check_circuit(self):
        log_message("Starting test_check_circuit.", level="INFO")
        try:
            self.circuit.qubits = 2
            self.circuit.depth = 3
            result = self.circuit.check_circuit()
            self.assertTrue(result, 'Error 4127: Consistent circuit should return True')

            self.circuit.qubits = 0
            result = self.circuit.check_circuit()
            self.assertFalse(result, 'Error 4128: Inconsistent circuit should return False')

            log_message("test_check_circuit passed successfully.", level="SUCCESS")

        except AssertionError as e:
            log_message(str(e), level="ERROR", error_code=4127)

    def test_measure(self):
        log_message("Starting test_measure.", level="INFO")
        try:
            result, code = self.circuit.measure(100)
            self.assertEqual(code, 2067, 'Error 4129: Successful measurements should return success code 2067')

            result, code = self.circuit.measure(-1)
            self.assertEqual(code, 1075, 'Error 4130: Failed measurements should return error code 1075')

            log_message("test_measure passed successfully.", level="SUCCESS")

        except AssertionError as e:
            log_message(str(e), level="ERROR", error_code=4129)


if __name__ == '__main__':
    unittest.main()
