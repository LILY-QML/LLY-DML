# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Project: LILY-QML
# Version: 1.6 LLY-DML
# Author: Leon Kaiser
# Contact: info@lilyqml.de
# Website: www.lilyqml.de
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++

from module.src.reader import Reader
import sys

class Start:
    def __init__(self, reader):
        """
        Initialisiert die Start-Klasse und setzt das train_json_exists Flag.
        
        :param reader: Eine Instanz der Reader-Klasse.
        """
        self.reader = reader
        self.train_json_exists = False

    def run(self):
        """
        Hauptausführungsmethode, die Dateiüberprüfungen, Logfile-Überprüfungen und
        Datenkonsistenz-Prüfungen koordiniert.
        """
        # Schritt 1: Dateiüberprüfung
        self.reader.logger.info("Initiating file check...")
        file_check_code = self.reader.fileCheck()

        if file_check_code == 1001:
            # 'var' Ordner fehlt
            error_message = {
                "Error Code": 1001,
                "Message": "'var' folder is missing."
            }
            self.reader.logger.error(error_message)
            print("ERROR: 'var' folder is missing.")
            # Programm nicht terminieren
            input("Press Enter to continue.")
        
        elif file_check_code == 1002:
            # 'data.json' fehlt
            error_message = {
                "Error Code": 1002,
                "Message": "'data.json' file is missing in 'var' folder."
            }
            self.reader.logger.error(error_message)
            print("ERROR: 'data.json' file is missing. Terminating program.")
            sys.exit(file_check_code)

        elif file_check_code == 1003:
            # 'config.json' fehlt
            error_message = {
                "Error Code": 1003,
                "Message": "'config.json' file is missing in 'var' folder."
            }
            self.reader.logger.error(error_message)
            print("ERROR: 'config.json' file is missing. Terminating program.")
            sys.exit(file_check_code)

        elif file_check_code == 2001:
            # Alle Dateien vorhanden, 'train.json' existiert
            self.train_json_exists = True
            success_message = {
                "Success Code": 2001,
                "Message": "All files are present. 'train.json' exists."
            }
            self.reader.logger.info(success_message)
            self.reader.logger.info("'train.json' found. Proceeding with initialization.")

        elif file_check_code == 2002:
            # Alle Dateien vorhanden, 'train.json' fehlt
            self.train_json_exists = False
            success_message = {
                "Success Code": 2002,
                "Message": "All files are present. 'train.json' is missing."
            }
            self.reader.logger.info(success_message)
            self.reader.logger.info("'train.json' missing. Proceeding without training file.")

        else:
            # Unerwarteter Fehlercode
            error_message = {
                "Error Code": 100105,
                "Message": "Unexpected error during file check."
            }
            self.reader.logger.error(error_message)
            print("ERROR: Unexpected error during file check.")
            # Programm nicht terminieren
            input("Press Enter to continue.")

        # Schritt 2: Logfile-Überprüfung
        self.reader.logger.info("Checking for an existing log file...")
        log_check = self.reader.checkLog()
        if not log_check:
            error_message = {
                "Error Code": 100106,
                "Message": "Failed to create or access log file."
            }
            self.reader.logger.error(error_message)
            print("ERROR: Failed to create or access log file.")
            # Programm nicht terminieren
            input("Press Enter to continue.")
        else:
            self.reader.logger.info("Log file is valid and up-to-date.")

        # Schritt 3: Datenkonsistenz-Prüfung
        self.reader.logger.info("Verifying data consistency for 'data.json'...")
        data_consistent = self.reader.dataConsistency()
        if not data_consistent:
            error_message = {
                "Error Code": 100104,
                "Message": "'data.json' file is inconsistent."
            }
            self.reader.logger.error(error_message)
            print("ERROR: 'data.json' file is inconsistent.")
            # Programm nicht terminieren
            input("Press Enter to continue.")
        else:
            success_message = {
                "Success Code": 200103,
                "Message": "'data.json' is consistent. Proceeding with execution."
            }
            self.reader.logger.info(success_message)
            print("SUCCESS: 'data.json' is consistent. Proceeding with execution.")
