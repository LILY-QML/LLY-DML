# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Project: LILY-QML
# Version: 2.0.0 LLY-DML
# Author: Leon Kaiser
# Contact: info@lilyqml.de
# Website: www.lilyqml.de
# Contributors:
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++

import json
import logging
from typing import Dict, List, Any, Optional
import os

class OptimizerInterpreter:
    """
    Eine Klasse zur Interpretation und Überprüfung der Optimierungsergebnisse aus train.json.
    """

    def __init__(self, data_json_path: str, train_json_path: str):
        self.data_json_path = data_json_path
        self.train_json_path = train_json_path
        self.data: Dict[str, Any] = {}
        self.train_data: Dict[str, Any] = {}
        self.optimizer_steps: List[Dict[str, Any]] = []
        self.expected_optimizers: List[str] = []
        self.expected_matrices: List[str] = []
        self.extracted_data: Dict[str, Dict[str, Any]] = {}

        # Konfiguration des Loggings
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('OptimizerInterpreter')

    def load_data(self):
        """
        Lädt die Daten aus data.json und train.json.
        """
        self.logger.info("Lade Daten aus data.json")
        if not os.path.exists(self.data_json_path):
            self.logger.error(f"data.json Datei nicht gefunden: {self.data_json_path}")
            raise FileNotFoundError(f"data.json Datei nicht gefunden: {self.data_json_path}")

        with open(self.data_json_path, 'r') as f:
            self.data = json.load(f)

        self.expected_optimizers = self.data.get('optimizers', [])
        self.logger.info(f"Erwartete Optimizer: {self.expected_optimizers}")

        self.expected_matrices = list(self.data.get('converted_activation_matrices', {}).keys())
        self.logger.info(f"Erwartete Aktivierungsmatrizen: {self.expected_matrices}")

        self.logger.info("Lade Daten aus train.json")
        if not os.path.exists(self.train_json_path):
            self.logger.error(f"train.json Datei nicht gefunden: {self.train_json_path}")
            raise FileNotFoundError(f"train.json Datei nicht gefunden: {self.train_json_path}")

        with open(self.train_json_path, 'r') as f:
            self.train_data = json.load(f)

        self.optimizer_steps = self.train_data.get('optimizer_steps', [])
        self.logger.info(f"Anzahl der Optimizer-Schritte in train.json: {len(self.optimizer_steps)}")

    def check_consistency(self) -> bool:
        """
        Überprüft die Konsistenz der Optimizer-Daten.

        :return: True, wenn die Daten konsistent sind, False sonst.
        """
        self.logger.info("Überprüfe Konsistenz der Optimizer-Daten")
        consistent = True
        
        expected_combinations = {(optimizer, matrix) for optimizer in self.expected_optimizers for matrix in self.expected_matrices}
        actual_combinations = {(step.get('optimizer'), step.get('activation_matrix')) for step in self.optimizer_steps}
        
        missing_combinations = expected_combinations - actual_combinations
        
        # Markiere fehlende oder teilweise ausgeführte Optimierungen
        if missing_combinations:
            consistent = False
            for optimizer, matrix in missing_combinations:
                if (optimizer, matrix) in actual_combinations:
                    self.logger.warning(f"Optimizer '{optimizer}' für Matrix '{matrix}' ist nur teilweise gelaufen.")
                else:
                    self.logger.error(f"Optimizer '{optimizer}' für Matrix '{matrix}' wurde gar nicht ausgeführt.")
        
        # Verbleibende Kombinationen (d.h. erfolgreich gelaufene Optimierungen)
        successful_combinations = actual_combinations - missing_combinations
        if successful_combinations:
            for optimizer, matrix in successful_combinations:
                self.logger.info(f"Optimizer '{optimizer}' für Matrix '{matrix}' ist erfolgreich gelaufen.")
        
        return consistent

    def run(self) -> Optional[Dict[str, Any]]:
        """
        Führt alle Schritte des Interpreters aus: Daten laden, Konsistenz prüfen, Daten extrahieren und Bericht generieren.

        :return: Ein Bericht über die Optimierungsergebnisse, oder None bei Fehlern.
        """
        try:
            self.load_data()
            if not self.check_consistency():
                self.logger.error("Konsistenzprüfung fehlgeschlagen. Bitte beheben Sie die oben aufgeführten Fehler.")
                return None
            extracted = self.extract_data()
            report = self.get_report()
            return report
        except Exception as e:
            self.logger.exception(f"Ein unerwarteter Fehler ist aufgetreten: {e}")
            return None
