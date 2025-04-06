# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Project: LILY-QML
# Version: 2.0.0 LLY-DML
# Author: Joan Pujol (@supercabb), Leon Kaiser
# Contact: info@lilyqml.de
# Website: www.lilyqml.de
# Contributors:
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++


from .qubit import Qubit
import importlib
import json
import os
import logging
import re
from datetime import datetime
import numpy as np


class Optimizer:
    def __init__(self, config_path='var'):
        """
        Initialisiert den Optimizer mit einem Konfigurationspfad.
        
        Args:
            config_path (str): Pfad zum Konfigurationsverzeichnis
        """
        self.Qubit_Object = {}
        self.current_job = None
        self.optimizer = None
        self.optimizer_class = None
        self.config_path = config_path
        self.dict_params_current_job = None
        self.target_state = None
        self.train_json_file_path = os.path.join(self.config_path, "train.json")
        self.train_json_data = None
        self.logger = logging.getLogger(__name__)
        self.data_json = None

    def check_prerequisites(self):
        """
        Überprüft den Optimizer und generiert Qubits.
        
        Returns:
            dict oder None: Fehlermeldung bei Problemen oder None bei Erfolg
        """
        Error_loading_optimizer = False

        # Normalisiere den Optimizernamen und bilde den Modulpfad
        optimizer_file_name = self.optimizer.split("Optimizer")[0].lower()+"_optimizer"        
        optimizer_module_name = f"optimizers.{optimizer_file_name}"
        optimizer_class = None
        
        try:
            # Versuche das Optimizer-Modul zu importieren
            optimizer_module = importlib.import_module(optimizer_module_name)
            optimizer_class = getattr(optimizer_module, self.optimizer, None)
        except Exception as e:
            self.logger.error(f"Fehler beim Importieren des Optimizers: {e}")
            Error_loading_optimizer = True

        if optimizer_class is None:
            Error_loading_optimizer = True

        if Error_loading_optimizer:
            return {"Error Code": 1111, "Message": "Optimizer not found."}

        self.optimizer_class = optimizer_class

        # Lade Konfigurationsdaten aus data.json
        path_data_json = os.path.join(self.config_path, 'data.json')
        try:
            with open(path_data_json, 'r') as config_file:
                self.data_json = json.load(config_file)
        except Exception as e:
            self.logger.error(f"Fehler beim Laden der data.json: {e}")
            return {"Error Code": 1112, "Message": "Data file not found."}

        # Initialisiere Qubits basierend auf der Konfiguration
        num_qubits = self.data_json['qubits']
        self.initialize_qubits(num_qubits)
        return None

    def initialize_qubits(self, num_qubits):
        """
        Initialisiert die Qubits.
        
        Args:
            num_qubits (int): Anzahl der zu erstellenden Qubits
        """
        self.Qubit_Object = {}

        for i in range(num_qubits):
            qubit_object = Qubit(qubit_number=i)
            self.Qubit_Object[i] = qubit_object
            self.logger.debug(f"Qubit {i} erstellt")

    def extract_fields_from_job(self, job):
        """
        Extrahiert die Felder aus einem Job-String mithilfe von Regular Expressions.
        
        Args:
            job (str): Der Job-String zur Verarbeitung
            
        Returns:
            dict oder None: Die extrahierten Felder als Dictionary oder None bei Fehler
        """
        fields = None
        
        # Regular Expression zur Extraktion der Parameter aus dem Job-String
        re_pattern = r"\((.*)\) Qubit_(\d+) \(\s*(\d+)\:\s*(\d+);\s*(\d+):\s*(\d+)\) \(\s*S:\s*([0-1]+)\)"

        # Versuche den Job-String zu matchen
        ret = re.match(re_pattern, job)

        if ret is not None:
            fields = {
                "matrix_row_str": ret.group(1),
                "num_qubit": ret.group(2),
                "dist_0_0": ret.group(3),
                "vdist_0_1": ret.group(4),
                "dist_1_0": ret.group(5),
                "dist_1_1": ret.group(6),
                "state": ret.group(7)
            } 

        return fields

    def extract_matrix_from_string(self, matrix_str):
        """
        Extrahiert eine Matrix aus einem String.
        
        Args:
            matrix_str (str): Der String mit den Matrix-Werten
            
        Returns:
            list oder None: Die extrahierte Matrix als Liste oder None bei Fehler
        """
        matrix_elements = None

        # Regular Expression zum Extrahieren der Matrix-Elemente
        re_matrix_str = r"([^,()\s]+),?"
        ret = re.findall(re_matrix_str, matrix_str)

        if ret is not None:
            matrix_elements = []
            for m in ret:
                try:
                    val = float(m)
                except ValueError:
                    self.logger.error(f"Fehler beim Konvertieren des Matrix-Elements zu float: {m}")
                    return None

                matrix_elements.append(val)        

        return matrix_elements

    def check_data_structure(self):
        """
        Überprüft, ob die Datenstruktur korrekt ist.
        
        Returns:
            dict oder None: Fehlermeldung bei Problemen oder None bei Erfolg
        """
        self.dict_params_current_job = self.extract_fields_from_job(self.current_job)

        if self.dict_params_current_job is None:
            error = {"Error Code": 1119, "Message": "Datastructure is not consistent."}
            self.logger.error(error)  
            return error
         
        matrix_elements = self.extract_matrix_from_string(self.dict_params_current_job["matrix_row_str"])

        if matrix_elements is None:
            error = {"Error Code": 1119, "Message": "Datastructure is not consistent."}
            self.logger.error(error)
            return error
        
        self.dict_params_current_job["matrix_elements"] = matrix_elements
        return None

    def evaluate(self, current_job):
        """
        Evaluiert den aktuellen Job durch Überprüfung der Datenstruktur und Laden des Zustands 
        in das entsprechende Qubit.
        
        Args:
            current_job (str): Der zu evaluierende Job-String
        """
        self.current_job = current_job
        ret = self.check_data_structure()
        if ret is None:  # Keine Fehler aufgetreten
            num_qubit = int(self.dict_params_current_job["num_qubit"])
            self.Qubit_Object[num_qubit].load_state(self.dict_params_current_job["state"])

    def execute(self, current_job):
        """
        Führt den Optimierungsprozess für den aktuellen Job aus.
        
        Args:
            current_job (str): Der auszuführende Job-String
            
        Returns:
            str oder None: Der neue Job-String nach der Optimierung oder None bei Fehler
        """
        self.evaluate(current_job)
        
        num_qubit = int(self.dict_params_current_job["num_qubit"])
        # Einzelnes Bit aus dem target_state für dieses Qubit extrahieren
        qubit_target_state = self.target_state[num_qubit]
        
        # Erstelle ein Optimizer-Objekt mit den aktuellen Parametern
        optimizer = self.optimizer_class(
            self.data_json, 
            self.dict_params_current_job["matrix_elements"], 
            qubit_target_state  # Nur das relevante Bit für dieses Qubit
        )
        
        # Führe die Optimierung durch
        optimizer.evaluate()
        loss = optimizer.calculate_loss()
        optimizer.compute_gradient()        
        tuning_parameters, optimization_steps = optimizer.optimize()

        # Überprüfe die Rückgabewerte
        if len(tuning_parameters) != len(self.dict_params_current_job["matrix_elements"]):
            error = {"Error Code": 1113, "Message": "Incorrect returned optimized matrix row!!"}
            self.logger.error(error)
            return None

        # Speichere die Verlustfunktion im Qubit-Objekt
        self.Qubit_Object[num_qubit].load_function(loss)   

        # Erstelle einen neuen Job-String mit den optimierten Parametern
        new_job = "("
        for t in tuning_parameters:
            new_job += str(t) + ","
        new_job = new_job[:-1] + ")"

        new_job += f" Qubit_{self.dict_params_current_job['num_qubit']} " + \
                  f"({self.dict_params_current_job['dist_0_0']}:{self.dict_params_current_job['vdist_0_1']}; " + \
                  f"{self.dict_params_current_job['dist_1_0']}:{self.dict_params_current_job['dist_1_1']}) " + \
                  f"(S:{self.dict_params_current_job['state']})"
        
        self.logger.info(f"Optimierungsergebnis: {new_job}")
        return new_job

    def start(self, optimizer, target_state):
        """
        Startet den Optimierungsprozess mit dem angegebenen Optimizer und Zielzustand.
        
        Args:
            optimizer (str): Name des zu verwendenden Optimizers (z.B. "AdamOptimizer")
            target_state (str): Der Zielzustand für die Optimierung
            
        Returns:
            dict oder None: Fehlermeldung bei Problemen oder None bei Erfolg
        """
        self.optimizer = optimizer
        self.target_state = target_state

        # Überprüfe, ob der Optimizer existiert
        optimizer_file_name = self.optimizer.split("Optimizer")[0].lower() + "_optimizer.py"
        optimizer_file_path = os.path.join("optimizers", optimizer_file_name)

        # Überprüfe Voraussetzungen
        ret = self.check_prerequisites()
        if ret is not None:
            return ret

        # Überprüfe den Target-Zustand
        if len(target_state) != len(self.Qubit_Object):
            error = {"Error Code": 1071, "Message": "Target state has incorrect formatting."}
            self.logger.error(error)
            return error
        
        self.logger.info("Target state und Optimizer erfolgreich validiert und initialisiert.")
        
        # Überprüfe train.json
        if not os.path.exists(self.train_json_file_path):
            error = {"Error Code": 1070, "Message": "train.json not found."}
            self.logger.error(error)
            return error
        
        with open(self.train_json_file_path, 'r') as config_file:
            self.train_json_data = json.load(config_file)
        
        self.logger.info("train.json erfolgreich gefunden und geladen.")
        
        self.logger.info(f"Optimizer {self.optimizer} validiert und geladen.")
        self.logger.info(f"Target State {self.target_state} validiert und geladen.")
        self.logger.info(f"Starte Optimierungsprozess um {datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
        return None

    def optimize(self, measurement, training_matrix):
        """
        Führt die Optimierung basierend auf Messungen und einer Trainingsmatrix durch.
        
        Args:
            measurement (dict): Dictionary mit Messungsergebnissen
            training_matrix (list): Die zu optimierende Trainingsmatrix
            
        Returns:
            list oder None: Die optimierte Trainingsmatrix oder None bei Fehler
        """
        self.logger.info("Starte Optimierungsprozess.")
        self.logger.info("Datensammlung zu Beginn der Optimierung erfolgreich.")

        # Kodiere die Messungen für die Qubits
        qubits_measurement = self.encode_measurements(measurement)
        if qubits_measurement is None:
            error = {"Error Code": 1074, "Message": "Inconsistent data structure after assigning measurement values."}
            self.logger.error(error)
            return None

        # Überprüfe die Dimensionen der Trainingsmatrix
        if len(training_matrix) != len(self.Qubit_Object):
            error = {"Error Code": 1075, "Message": "Inconsistent matrix due to incorrect number of rows."}
            self.logger.error(error)
            return None

        # Lade die Trainingsmatrix und die aktuellen Verteilungen in die Qubit-Objekte
        for index, row in enumerate(training_matrix):
            qubit_matrix_str = "("
            for element in row:
                qubit_matrix_str += str(element) + ","
            qubit_matrix_str = qubit_matrix_str[:-1] + ")"
            
            self.Qubit_Object[index].load_training_matrix(qubit_matrix_str)
            self.Qubit_Object[index].load_actual_distribution(qubits_measurement[index])
            # Lade auch das entsprechende Target-Bit
            self.Qubit_Object[index].load_target_state(self.target_state[index])

        # Führe die Optimierung für jedes Qubit durch
        try:
            new_training_matrix = []
            for num_qubit, qubit in self.Qubit_Object.items():
                # Erstelle den Job-String für das aktuelle Qubit
                current_job = qubit.read_training_matrix() + " " + \
                             f"Qubit_{qubit.read_qubit_number()} " + \
                             f"{qubit.read_actual_distribution()} " + \
                             f"(S:{self.target_state})"
                
                new_job = self.execute(current_job)
                if new_job is None:
                    error = {"Error Code": 1076, "Message": "Optimization error while executing the job."}
                    self.logger.error(error)
                    return None

                # Extrahiere die optimierte Matrix aus dem neuen Job
                extracted_fields = self.extract_fields_from_job(new_job)
                extracted_matrix = self.extract_matrix_from_string(extracted_fields["matrix_row_str"])
                new_training_matrix.append(extracted_matrix)
                self.logger.info(f"Matrix für Qubit {qubit.read_qubit_number()} mit neuen Werten aktualisiert.")
                
        except Exception as e:
            error = {"Error Code": 1077, "Message": f"Optimization error while writing the training matrix: {e}"}
            self.logger.error(error)
            return None

        self.logger.info("Optimierung erfolgreich abgeschlossen und Trainingsmatrix aktualisiert.")
        self.logger.info("Optimierungsprozess beendet.")
        return new_training_matrix

    def encode_measurements(self, measurement):
        """
        Kodiert die Messungsergebnisse für die Optimierung.
        
        Args:
            measurement (dict): Dictionary mit Messungsergebnissen als Bitstings
            
        Returns:
            list oder None: Kodierte Messungsergebnisse oder None bei Fehler
        """
        qubits_measurement = []
        qubits_measurement_count = np.zeros((len(self.Qubit_Object), 2), dtype=int)

        self.logger.info("Starte Kodierung der Messungen.")

        # Überprüfe, ob die Anzahl der Qubits mit den Messungsergebnissen übereinstimmt
        if len(next(iter(measurement))) != len(self.Qubit_Object):
            error = {"Error Code": 1073, "Message": "Inconsistent data due to incorrect number of qubits."}
            self.logger.error(error)
            return None

        # Zähle die 0 und 1 Bits für jedes Qubit
        for key, value in measurement.items():
            for index, c in enumerate(key):
                qubits_measurement_count[index][int(c)] += value

        # Erstelle die kodierten Messungsergebnisse im benötigten Format
        for i in range(len(self.Qubit_Object)):
            qubit_measurement = "("
            qubit_measurement += f"1:{qubits_measurement_count[i][1]}; "
            qubit_measurement += f"0:{qubits_measurement_count[i][0]})"
            
            qubits_measurement.append(qubit_measurement)

        self.logger.info(f"Kodierte Messungen: {qubits_measurement}")
        return qubits_measurement