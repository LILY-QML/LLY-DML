# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Project: LILY-QML
# Version: 1.6 LLY-DML
# Author: Leon Kaiser
# Contact: info@lilyqml.de
# Website: www.lilyqml.de
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++

import sys
import logging
import json
import os 
from module.src.console import Console
from module.start import Start
from module.src.reader import Reader

# Set up logging
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(
    filename=os.path.join(log_dir, 'lily_qml_dml.log'),
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger()

class DML:
    def __init__(self, reader):
        """
        Initialisiert die DML-Klasse mit einer Reader-Instanz.
        
        :param reader: Eine Instanz der Reader-Klasse.
        """
        self.reader = reader
        self.start_instance = Start(self.reader)
        self.console = None
        logger.info("DML initialized with Start instance.")

    def start(self):
        """
        Führt die Startmethode aus und startet die Konsole.
        """
        logger.info("DML start method initiated.")
        # Starte die Initialisierung durch die Start-Klasse
        self.start_instance.run()
        
        # Extrahiere notwendige Informationen für die Console
        train_exists = self.start_instance.train_json_exists
        train_content = self._get_train_content()
        data_exists = self.start_instance.reader.dataConsistency()
        data_content = self._get_data_content()
        optimizers = self._get_optimizers()
        missing_optimizers = self._get_missing_optimizers()

        # Initialisiere die Console-Klasse mit den extrahierten Daten
        self.console = Console(
            train_exists=train_exists,
            train_content=train_content,
            data_exists=data_exists,
            data_content=data_content,
            optimizers=optimizers,
            missing_optimizers=missing_optimizers
        )
        logger.info("Console instance created.")

        # Starte das Hauptmenü der Console
        while True:
            action = self.console.run_main_menu()
            if action:
                if action.get("action") == "view_information":
                    self.handle_information()
                elif action.get("action") == "train_all_optimizers":
                    self.handle_training_all_optimizers()
                elif action.get("action") == "train_specific_optimizers":
                    self.handle_training_specific_optimizers(action.get("optimizers"))
                elif action.get("action") == "create_report":
                    self.handle_report()
                elif action.get("action") == "exit":
                    self.handle_exit()
                else:
                    logger.warning(f"Unknown action received: {action}")
                    print("Unknown action. Please try again.")

    def handle_information(self):
        """
        Zeigt Informationen über die Referenzdateien an.
        """
        logger.info("Handling information display.")
        self.console.show_data_info()
        self.console.show_train_info()

    def handle_training_all_optimizers(self):
        """
        Führt das Training aller Optimierer durch.
        """
        logger.info("Handling training of all optimizers.")
        # Hier könnte der Trainingsprozess implementiert werden
        print("Training all optimizers...")
        # Beispiel: Simuliere Trainingsdauer
        import time
        time.sleep(2)
        logger.info("All optimizers have been trained successfully.")
        print("All optimizers have been trained successfully.")
        input("Press Enter to continue.")

    def handle_training_specific_optimizers(self, optimizers):
        """
        Führt das Training spezifischer Optimierer durch.
        
        :param optimizers: Liste der zu trainierenden Optimierer.
        """
        if optimizers:
            logger.info(f"Handling training of specific optimizers: {optimizers}")
            print(f"Training specific optimizers: {', '.join(optimizers)}")
            # Hier könnte der spezifische Trainingsprozess implementiert werden
            import time
            time.sleep(2)
            logger.info(f"Optimizers {', '.join(optimizers)} have been trained successfully.")
            print(f"Optimizers {', '.join(optimizers)} have been trained successfully.")
            input("Press Enter to continue.")
        else:
            logger.warning("No optimizers selected for specific training.")
            print("No optimizers selected.")
            input("Press Enter to continue.")

    def handle_report(self):
        """
        Generiert und zeigt Berichte an.
        """
        logger.info("Handling report generation.")
        # Hier könnte die Berichtserstellung implementiert werden
        print("Generating report...")
        # Beispiel: Simuliere Berichtserstellung
        import time
        time.sleep(2)
        logger.info("Report has been generated successfully.")
        print("Report has been generated successfully.")
        input("Press Enter to continue.")

    def handle_exit(self):
        """
        Beendet das Programm.
        """
        logger.info("Handling program exit.")
        print("Exiting the program. Goodbye!")
        sys.exit()

    def _get_train_content(self):
        """
        Holt den Inhalt von train.json.
        
        :return: Inhalt von train.json oder None.
        """
        if self.start_instance.train_json_exists:
            try:
                with open(self.start_instance.reader.train_path, 'r') as f:
                    content = f.read()
                    logger.info("train.json content retrieved successfully.")
                    return content
            except Exception as e:
                logger.error(f"Error reading train.json: {e}")
                return None
        else:
            logger.info("train.json does not exist.")
            return None

    def _get_data_content(self):
        """
        Holt den Inhalt von data.json.
        
        :return: Inhalt von data.json oder None.
        """
        try:
            with open(self.start_instance.reader.data_path, 'r') as f:
                content = f.read()
                logger.info("data.json content retrieved successfully.")
                return content
        except Exception as e:
            logger.error(f"Error reading data.json: {e}")
            return None

    def _get_optimizers(self):
        """
        Holt die Liste der verfügbaren Optimierer.
        
        :return: Liste der Optimierer.
        """
        try:
            with open(self.start_instance.reader.data_path, 'r') as f:
                data = json.load(f)
                optimizers = data.get('optimizers', [])
                logger.info(f"Optimizers retrieved: {optimizers}")
                return optimizers
        except Exception as e:
            logger.error(f"Error retrieving optimizers: {e}")
            return []

    def _get_missing_optimizers(self):
        """
        Holt die Liste der fehlenden Optimierer.
        
        :return: Liste der fehlenden Optimierer.
        """
        # Hier wird angenommen, dass fehlende Optimierer auf irgendeine Weise ermittelt werden
        # Diese Logik muss entsprechend den Anforderungen implementiert werden
        # Beispielhafte Implementierung:
        try:
            with open(self.start_instance.reader.data_path, 'r') as f:
                data = json.load(f)
                available_optimizers = data.get('optimizers', [])
                required_optimizers = [
                    "AdamOptimizer",
                    "SGDOptimizer",
                    "RMSPropOptimizer",
                    "AdaGradOptimizer",
                    "MomentumOptimizer",
                    "NadamOptimizer"
                ]
                missing = [opt for opt in required_optimizers if opt not in available_optimizers]
                logger.info(f"Missing optimizers retrieved: {missing}")
                return missing
        except Exception as e:
            logger.error(f"Error determining missing optimizers: {e}")
            return []


def main():
    # Initialisiere die Reader-Klasse (angenommen, der Reader hat bereits die notwendigen Parameter)
    reader = Reader(test_mode=False)
    
    # Initialisiere die DML-Klasse mit der Reader-Instanz
    dml = DML(reader)
    
    # Starte den DML-Prozess
    dml.start()

if __name__ == "__main__":
    main()