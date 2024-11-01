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

class Circuit:
    """
    Die Circuit-Klasse dient zur Erstellung, Verwaltung und Validierung von Quanten-Schaltkreisen basierend auf Eingabedaten aus JSON-Dateien.
    Zudem ermöglicht sie die Durchführung von Messungen am Ende des Schaltkreises.
    """

    def __init__(self, data_path='data.json', train_path='train.json', config_path='var/config.json'):
        """
        Initialisiert eine neue Instanz der Circuit-Klasse.

        Parameter:
            data_path (str, optional): Pfad zur data.json-Datei. Standardwert ist 'data.json'.
            train_path (str, optional): Pfad zur train.json-Datei. Standardwert ist 'train.json'.
            config_path (str, optional): Pfad zur config.json-Datei. Standardwert ist 'var/config.json'.
        """
        self.data_path = data_path
        self.train_path = train_path
        self.qubits = None
        self.depth = None
        self.activation_matrices = {}
        self.training_matrices = {}
        self.input_data = []
        self.circuit = {}
        self.measurements = []

        # Initialize logging
        self._initialize_logging(config_path)

    def _initialize_logging(self, config_path):
        """
        Initialisiert das Logging basierend auf der config.json.

        Parameter:
            config_path (str): Pfad zur config.json-Datei.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Konfigurationsdatei {config_path} nicht gefunden.")

        try:
            with open(config_path, 'r') as file:
                config = json.load(file)
            log_file = config.get('log_file', 'default.log')
            log_directory = os.path.dirname(log_file)
            if log_directory and not os.path.exists(log_directory):
                os.makedirs(log_directory)

            logging.basicConfig(
                filename=log_file,
                level=logging.DEBUG,
                format='%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            self.logger = logging.getLogger(__name__)
            self.logger.info("Logging erfolgreich initialisiert.")
        except json.JSONDecodeError:
            raise ValueError(f"Ungültiges JSON in der Konfigurationsdatei {config_path}.")
        except Exception as e:
            raise Exception(f"Fehler beim Initialisieren des Loggings: {str(e)}")

    def read_data(self):
        """
        Liest Daten aus den JSON-Dateien data.json und train.json.
        Extrahiert grundlegende Schaltkreisparameter sowie Aktivierungs- und Trainingsmatrizen.

        Rückgabe:
            dict: Erfolgsmeldung oder Fehlermeldung mit entsprechenden Codes.
        """
        self.logger.info("Start der Methode read_data.")
        # Lesen von data.json
        if not os.path.exists(self.data_path):
            error_msg = f"Datei {self.data_path} nicht gefunden."
            self.logger.error(f"Fehlercode 1060: {error_msg}")
            return {"Fehlercode": 1060, "Fehlermeldung": error_msg}
        try:
            with open(self.data_path, 'r') as file:
                data = json.load(file)
            self.qubits = data.get('qubits')
            self.depth = data.get('depth')
            if self.qubits is None oder self.depth is None:
                error_msg = "Fehlende 'qubits' oder 'depth' in data.json."
                self.logger.error(f"Fehlercode 1060: {error_msg}")
                return {"Fehlercode": 1060, "Fehlermeldung": error_msg}
            success_data = {"Erfolgscode": 2060, "Nachricht": "data.json erfolgreich gelesen."}
            self.logger.info(f"Erfolgscode 2060: {success_data['Nachricht']}")
        except json.JSONDecodeError:
            error_msg = f"Ungültiges JSON in {self.data_path}."
            self.logger.error(f"Fehlercode 1060: {error_msg}")
            return {"Fehlercode": 1060, "Fehlermeldung": error_msg}
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Fehlercode 1060: {error_msg}")
            return {"Fehlercode": 1060, "Fehlermeldung": error_msg}

        # Lesen von train.json
        if not os.path.exists(self.train_path):
            error_msg = f"Datei {self.train_path} nicht gefunden."
            self.logger.error(f"Fehlercode 1061: {error_msg}")
            return {"Fehlercode": 1061, "Fehlermeldung": error_msg}
        try:
            with open(self.train_path, 'r') as file:
                train_data = json.load(file)
            self.activation_matrices = train_data.get('activation_matrices', {})
            self.training_matrices = train_data.get('training_matrices', {})
            success_train = {"Erfolgscode": 2061, "Nachricht": "train.json erfolgreich gelesen."}
            self.logger.info(f"Erfolgscode 2061: {success_train['Nachricht']}")
            return {"Erfolg": [success_data, success_train]}
        except json.JSONDecodeError:
            error_msg = f"Ungültiges JSON in {self.train_path}."
            self.logger.error(f"Fehlercode 1061: {error_msg}")
            return {"Fehlercode": 1061, "Fehlermeldung": error_msg}
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Fehlercode 1061: {error_msg}")
            return {"Fehlercode": 1061, "Fehlermeldung": error_msg}

    def convert_input_data(self, option='empty', matrix_name=None):
        """
        Konvertiert die rohen Eingabedaten aus den JSON-Dateien in eine strukturierte Form.

        Parameter:
            option (str, optional): 'empty' für eine leere Matrix oder 'named' für eine benannte Matrix. Standardwert ist 'empty'.
            matrix_name (str, optional): Name der Aktivierungsmatrix, erforderlich wenn option='named' ist.

        Rückgabe:
            dict: Erfolgsmeldung oder Fehlermeldung mit entsprechenden Codes.
        """
        self.logger.info(f"Start der Methode convert_input_data mit Option='{option}' und matrix_name='{matrix_name}'.")
        if option == 'empty':
            # Erstellen einer leeren Aktivierungsmatrix mit Nullen
            matrix = [[0 for _ in range(self.depth * 3)] for _ in range(self.qubits)]
            if not self._is_empty_matrix(matrix):
                error_msg = "Inkonsistente leere Aktivierungsmatrix."
                self.logger.error(f"Fehlercode 1064: {error_msg}")
                return {"Fehlercode": 1064, "Fehlermeldung": error_msg}
            self.logger.info("Leere Aktivierungsmatrix erfolgreich erstellt.")
        elif option == 'named':
            if not matrix_name:
                error_msg = "matrix_name muss angegeben werden, wenn option='named' ist."
                self.logger.error(f"Fehlercode 1066: {error_msg}")
                return {"Fehlercode": 1066, "Fehlermeldung": error_msg}
            matrix = self.activation_matrices.get(matrix_name)
            if matrix is None:
                error_msg = f"Aktivierungsmatrix '{matrix_name}' nicht gefunden."
                self.logger.error(f"Fehlercode 1063: {error_msg}")
                return {"Fehlercode": 1063, "Fehlermeldung": error_msg}
            if not self._is_matrix_consistent(matrix):
                error_msg = "Inkonsistente benannte Aktivierungsmatrix."
                self.logger.error(f"Fehlercode 1063: {error_msg}")
                return {"Fehlercode": 1063, "Fehlermeldung": error_msg}
            self.logger.info(f"Benannte Aktivierungsmatrix '{matrix_name}' erfolgreich geladen.")
        else:
            error_msg = "Ungültige Option. Verwenden Sie 'empty' oder 'named'."
            self.logger.error(f"Fehlercode 1064: {error_msg}")
            return {"Fehlercode": 1064, "Fehlermeldung": error_msg}

        # Reformatierung der Daten
        try:
            for zeile in range(self.qubits):
                for position in range(self.depth):
                    aktivierungsdreier = matrix[zeile][position*3:(position+1)*3]
                    trainingsdreier = self.training_matrices.get(matrix_name, [])[zeile][position*3:(position+1)*3] if option == 'named' else [0, 0, 0]
                    entry = {
                        "Zeile": zeile,
                        "Position": position,
                        "Aktivierungsdreier": aktivierungsdreier,
                        "Trainingsdreier": trainingsdreier
                    }
                    self.input_data.append(entry)
            self.logger.info("Daten erfolgreich reformatiert.")
            if not self.check_input_data():
                self.input_data = []
                error_msg = "Inkonsistente formatierte Eingabedaten."
                self.logger.error(f"Fehlercode 1061: {error_msg}")
                return {"Fehlercode": 1061, "Fehlermeldung": error_msg}
            success_msg = "Eingabedaten erfolgreich konvertiert."
            self.logger.info(f"Erfolgscode 2064: {success_msg}")
            return {"Erfolgscode": 2064, "Nachricht": success_msg}
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Fehlercode 1063: {error_msg}")
            return {"Fehlercode": 1063, "Fehlermeldung": error_msg}

    def _is_matrix_consistent(self, matrix):
        """
        Überprüft die Konsistenz einer gegebenen Aktivierungsmatrix.

        Parameter:
            matrix (list): Die zu überprüfende Matrix.

        Rückgabe:
            bool: True, wenn die Matrix konsistent ist, sonst False.
        """
        self.logger.debug("Start der Methode _is_matrix_consistent.")
        if not isinstance(matrix, list):
            self.logger.debug("Matrix ist keine Liste.")
            return False
        if len(matrix) != self.qubits:
            self.logger.debug(f"Matrix hat {len(matrix)} Zeilen, erwartet {self.qubits}.")
            return False
        for zeile in matrix:
            if not isinstance(zeile, list):
                self.logger.debug("Eine Zeile in der Matrix ist keine Liste.")
                return False
            if len(zeile) != self.depth * 3:
                self.logger.debug(f"Eine Zeile hat {len(zeile)} Elemente, erwartet {self.depth * 3}.")
                return False
            for element in zeile:
                if not isinstance(element, (int, float)):
                    self.logger.debug(f"Element {element} in der Matrix ist kein numerischer Wert.")
                    return False
        self.logger.debug("Matrix ist konsistent.")
        return True

    def _is_empty_matrix(self, matrix):
        """
        Überprüft, ob die gegebene Matrix ausschließlich aus Nullen besteht.

        Parameter:
            matrix (list): Die zu überprüfende Matrix.

        Rückgabe:
            bool: True, wenn alle Elemente Null sind, sonst False.
        """
        self.logger.debug("Start der Methode _is_empty_matrix.")
        for zeile in matrix:
            for element in zeile:
                if element != 0:
                    self.logger.debug(f"Nicht-Null-Element gefunden: {element}")
                    return False
        self.logger.debug("Matrix besteht ausschließlich aus Nullen.")
        return True

    def check_input_data(self):
        """
        Überprüft die Struktur und Konsistenz der input_data.

        Rückgabe:
            bool: True, wenn input_data konsistent ist, sonst False.
        """
        self.logger.debug("Start der Methode check_input_data.")
        if not isinstance(self.input_data, list):
            self.logger.debug("input_data ist keine Liste.")
            return False
        if len(self.input_data) != self.qubits * self.depth:
            self.logger.debug(f"input_data hat {len(self.input_data)} Einträge, erwartet {self.qubits * self.depth}.")
            return False
        struktur = {"Zeile", "Position", "Aktivierungsdreier", "Trainingsdreier"}
        for eintrag in self.input_data:
            if not isinstance(eintrag, dict):
                self.logger.debug("Ein Eintrag in input_data ist kein Dictionary.")
                return False
            if not struktur.issubset(eintrag.keys()):
                self.logger.debug("Ein Eintrag in input_data fehlt erforderliche Schlüssel.")
                return False
            if not isinstance(eintrag["Zeile"], int) or not isinstance(eintrag["Position"], int):
                self.logger.debug("Zeile oder Position in einem Eintrag ist kein Integer.")
                return False
            if not isinstance(eintrag["Aktivierungsdreier"], list) or not isinstance(eintrag["Trainingsdreier"], list):
                self.logger.debug("Aktivierungsdreier oder Trainingsdreier sind keine Listen.")
                return False
            if len(eintrag["Aktivierungsdreier"]) != 3 or len(eintrag["Trainingsdreier"]) != 3:
                self.logger.debug("Aktivierungsdreier oder Trainingsdreier haben nicht genau drei Elemente.")
                return False
        # Überprüfen der Anzahl der Einträge pro Zeile
        zeilen_counts = {}
        for eintrag in self.input_data:
            zeile = eintrag["Zeile"]
            zeilen_counts[zeile] = zeilen_counts.get(zeile, 0) + 1
        for zeile, count in zeilen_counts.items():
            if count != self.depth:
                self.logger.debug(f"Zeile {zeile} hat {count} Einträge, erwartet {self.depth}.")
                return False
        self.logger.debug("input_data ist konsistent.")
        return True

    def create_L_gate(self, entry):
        """
        Erstellt die Parameter für ein einzelnes L-Gate innerhalb des Quantum-Schaltkreises basierend auf einem gegebenen Eintrag aus input_data.

        Parameter:
            entry (dict): Ein Dictionary aus input_data mit den Schlüsseln:
                - Zeile (int)
                - Position (int)
                - Aktivierungsdreier (list)
                - Trainingsdreier (list)

        Rückgabe:
            dict: Schaltkreisbauplan und Statuscode oder Fehlermeldung mit Fehlercode.
        """
        self.logger.info(f"Erstellung eines L-Gates für Zeile {entry.get('Zeile')}, Position {entry.get('Position')}.")
        # Überprüfung des Eintrags
        erforderliche_schluessel = {"Zeile", "Position", "Aktivierungsdreier", "Trainingsdreier"}
        if not isinstance(entry, dict) or not erforderliche_schluessel.issubset(entry.keys()):
            error_msg = "Ungültiger Eintrag oder fehlende Schlüssel."
            self.logger.error(f"Fehlercode 1060: {error_msg}")
            return {"Fehlercode": 1060, "Fehlermeldung": error_msg}
        if not (isinstance(entry["Aktivierungsdreier"], list) and isinstance(entry["Trainingsdreier"], list)):
            error_msg = "Aktivierungsdreier und Trainingsdreier müssen Listen sein."
            self.logger.error(f"Fehlercode 1060: {error_msg}")
            return {"Fehlercode": 1060, "Fehlermeldung": error_msg}
        if len(entry["Aktivierungsdreier"]) != 3 or len(entry["Trainingsdreier"]) != 3:
            error_msg = "Ungültige Listenlängen in 'Aktivierungsdreier' oder 'Trainingsdreier'."
            self.logger.error(f"Fehlercode 1061: {error_msg}")
            return {"Fehlercode": 1061, "Fehlermeldung": error_msg}

        # Extraktion der Daten
        try:
            aktivierungswerte = entry["Aktivierungsdreier"]
            trainingswerte = entry["Trainingsdreier"]
            self.logger.debug(f"Aktivierungswerte: {aktivierungswerte}, Trainingswerte: {trainingswerte}")
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Fehlercode 1062: {error_msg}")
            return {"Fehlercode": 1062, "Fehlermeldung": error_msg}

        # Zusammenführung der Daten
        try:
            gates = []
            for a, t in zip(aktivierungswerte, trainingswerte):
                gates.append(f"P({a})")
                gates.append(f"P({t})")
                gates.append("H")
            # Entfernen des letzten Hadamard-Gatters, wenn nicht benötigt
            if gates and gates[-1] == "H":
                gates.pop()
            schaltkreisbauplan = {
                "Zeile": entry["Zeile"],
                "End": True,
                "Gates": gates
            }
            self.logger.info(f"Schaltkreisbauplan für L-Gate erstellt: {schaltkreisbauplan}")
            return {"Schaltkreisbauplan": schaltkreisbauplan, "Erfolgscode": 2060}
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Fehlercode 1063: {error_msg}")
            return {"Fehlercode": 1063, "Fehlermeldung": error_msg}

    def create_initial_circuit(self, matrix_name=None):
        """
        Erstellt den initialen Schaltkreis (circuit).

        Parameter:
            matrix_name (str, optional): Name der Aktivierungsmatrix, die verwendet werden soll. Wenn nicht angegeben, wird 'empty' verwendet.

        Rückgabe:
            dict: Erfolgsmeldung oder Fehlermeldung mit entsprechenden Codes.
        """
        self.logger.info("Start der Methode create_initial_circuit.")
        # Schritt 1: Daten lesen
        daten_lesen = self.read_data()
        if "Fehlercode" in daten_lesen:
            self.logger.error(f"Fehler beim Lesen der Daten: {daten_lesen['Fehlercode']} - {daten_lesen['Fehlermeldung']}")
            return daten_lesen

        # Schritt 2: Daten konvertieren
        option = 'named' if matrix_name else 'empty'
        daten_konvertieren = self.convert_input_data(option=option, matrix_name=matrix_name)
        if "Fehlercode" in daten_konvertieren:
            self.logger.error(f"Fehler beim Konvertieren der Eingabedaten: {daten_konvertieren['Fehlercode']} - {daten_konvertieren['Fehlermeldung']}")
            return daten_konvertieren

        # Schritt 3: Circuit initialisieren
        self.circuit = {qubit: [] for qubit in range(self.qubits)}
        self.logger.info("Circuit initialisiert.")

        # Schritt 4: Erstellung der L-Gates
        for entry in self.input_data:
            l_gate = self.create_L_gate(entry)
            if "Fehlercode" in l_gate:
                self.logger.error(f"Fehler beim Erstellen eines L-Gates: {l_gate['Fehlercode']} - {l_gate['Fehlermeldung']}")
                return l_gate
            schaltkreisbauplan = l_gate.get("Schaltkreisbauplan")
            zeile = schaltkreisbauplan["Zeile"]
            self.circuit[zeile].extend(schaltkreisbauplan["Gates"])
            self.logger.debug(f"L-Gate zu Qubit {zeile} hinzugefügt: {schaltkreisbauplan['Gates']}")

        # Schritt 5: Circuit prüfen
        if not self.check_circuit():
            error_msg = "Schaltkreisprüfung fehlgeschlagen."
            self.logger.error(f"Fehlercode 1066: {error_msg}")
            return {"Fehlercode": 1066, "Fehlermeldung": error_msg}

        success_msg = "Initialer Schaltkreis erfolgreich erstellt."
        self.logger.info(f"Erfolgscode 2066: {success_msg}")
        return {"Erfolgscode": 2066, "Nachricht": success_msg}

    def check_circuit(self):
        """
        Überprüft den erstellten Schaltkreis (circuit) auf Konsistenz.

        Rückgabe:
            bool: True, wenn der Schaltkreis konsistent ist, sonst False.
        """
        self.logger.debug("Start der Methode check_circuit.")
        # Überprüfen der Anzahl der Qubits
        if len(self.circuit) != self.qubits:
            self.logger.error(f"Falsche Anzahl von Qubits im Schaltkreis: {len(self.circuit)} statt {self.qubits}.")
            return False

        for qubit, gates in self.circuit.items():
            # Zählen der Phasengatter (P)
            p_gates = [gate for gate in gates if gate.startswith("P(")]
            if len(p_gates) != 6 * self.depth:
                self.logger.error(f"Falsche Anzahl von Phasengattern auf Qubit {qubit}: {len(p_gates)} statt {6 * self.depth}.")
                return False
            # Zählen der Hadamard-Gatter (H)
            h_gates = [gate for gate in gates if gate == "H"]
            if len(h_gates) != 3 * self.depth:
                self.logger.error(f"Falsche Anzahl von Hadamard-Gattern auf Qubit {qubit}: {len(h_gates)} statt {3 * self.depth}.")
                return False

        self.logger.debug("Schaltkreis ist konsistent.")
        return True

    def measure(self, shots):
        """
        Führt Messungen am Schaltkreis durch.

        Parameter:
            shots (int): Gibt an, wie oft der Circuit ausgeführt wird.

        Rückgabe:
            dict: Messergebnisse und Erfolgscode oder Fehlermeldung mit Fehlercode.
        """
        self.logger.info(f"Start der Messungen mit {shots} Shots.")
        try:
            # Simulierte Messung: zufällige Ergebnisse generieren
            # Hier sollte die tatsächliche Quantenberechnung integriert werden
            import random
            self.measurements = [''.join([str(random.randint(0, 1)) for _ in range(self.qubits)]) for _ in range(shots)]
            self.logger.info("Messungen erfolgreich durchgeführt.")
            return {"Messergebnisse": self.measurements, "Erfolgscode": 2066}
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Fehlercode 1069: {error_msg}")
            return {"Fehlercode": 1069, "Fehlermeldung": error_msg}
