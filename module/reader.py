# module/reader.py

import json
import os
import sys
from datetime import datetime
import logging
class Reader:
    def __init__(
        self,
        data_json_path='var/data.json',
        train_json_path='var/train.json',
        log_db_path='var/log.logdb',
        create_train_json_on_init=True  # Neuer Parameter hinzugefügt
    ):
        self.data_json_path = data_json_path
        self.train_json_path = train_json_path
        self.log_db_path = log_db_path
        self.data_config = {}
        self.train_data = {}
        self.data_optimizer = []
        self.train_optimizer = []
        
        # Hinzufügen von self.data als leeres Dictionary
        self.data = {}

        # Variablen für alle JSON-Properties
        self.qubits = None
        self.depth = None
        self.learning_rate = None
        self.shots = None
        self.max_iterations = None
        self.population_size = None
        self.mutation_rate = None
        self.num_particles = None
        self.inertia = None
        self.cognitive = None
        self.social = None
        self.initial_temperature = None
        self.cooling_rate = None
        self.optimizers = []
        self.activation_matrices = {}
        
        # Setup Logging
        self.setup_logging()
        
        self.check_files(create_train_json_on_init)  # Übergabe des neuen Parameters
        self.load_data_json()
        self.load_train_json()

    def load_data_json(self):
        try:
            with open(self.data_json_path, 'r') as f:
                self.data_config = json.load(f)
            
            # Zuweisen der Variablen aus data_config
            self.qubits = self.data_config.get('qubits')
            self.depth = self.data_config.get('depth')
            self.learning_rate = self.data_config.get('learning_rate')
            self.shots = self.data_config.get('shots')
            self.max_iterations = self.data_config.get('max_iterations')
            self.population_size = self.data_config.get('population_size')
            self.mutation_rate = self.data_config.get('mutation_rate')
            self.num_particles = self.data_config.get('num_particles')
            self.inertia = self.data_config.get('inertia')
            self.cognitive = self.data_config.get('cognitive')
            self.social = self.data_config.get('social')
            self.initial_temperature = self.data_config.get('initial_temperature')
            self.cooling_rate = self.data_config.get('cooling_rate')
            self.optimizers = self.data_config.get('optimizers', [])
            
            # Aktivierungs-Matrizen als Dictionary speichern
            self.activation_matrices = self.data_config.get('activation_matrices', {})

            # Setze die geladenen Daten in self.data, um sie für andere Klassen verfügbar zu machen
            self.data = self.data_config  # Hinzufügen der data-Zuweisung
            
            # Debug-Ausgaben
            self.logger.debug("Geladene Optimierer aus data.json: %s", self.optimizers)
            self.logger.debug("Geladene Aktivierungsmatrizen: %s", self.activation_matrices.keys())
            print("Geladene Aktivierungsmatrizen:", self.activation_matrices.keys())
        
        except json.JSONDecodeError as e:
            error_msg = f"Fehler beim Parsen von data.json: {e}"
            print(error_msg)
            self.logger.error(error_msg)
            sys.exit(1)

    def setup_logging(self):
        # Konfiguration des Loggings
        if not os.path.exists('var'):
            os.makedirs('var')
        logging.basicConfig(
            filename=self.log_db_path,
            filemode='a',
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.DEBUG  # Setzt den Log-Level auf DEBUG
        )
        self.logger = logging.getLogger()
        self.logger.debug("Logging gestartet.")

    def check_files(self, create_train_json_on_init):
        # Prüfe, ob data.json existiert
        if not os.path.exists(self.data_json_path):
            error_msg = f"Fehler: data.json nicht gefunden unter {self.data_json_path}. Das Programm wird beendet."
            print(error_msg)
            self.logger.error(error_msg)
            sys.exit(1)
        
        # Erstelle 'var'-Verzeichnis, falls nicht vorhanden
        if not os.path.exists('var'):
            os.makedirs('var')
            self.logger.debug("Verzeichnis 'var' erstellt.")
        
        # Prüfe und erstelle train.json, falls nicht vorhanden und wenn erlaubt
        if not os.path.exists(self.train_json_path) and create_train_json_on_init:
            self.create_train_json()
        
        # Prüfe und erstelle log.logdb, falls nicht vorhanden
        if not os.path.exists(self.log_db_path):
            open(self.log_db_path, 'a').close()  # Erstelle leere Log-Datei
            self.logger.debug("Logdatei erstellt.")

    def create_train_json(self):
        self.train_data = {
            "creation_date": datetime.now().isoformat(),
            "data_json": self.data_json_path,  # Referenz zur data.json
            "optimizers": {},  # Anfangs leer
            "training_matrix": None,
            "converted_activation_matrices": {},
            "simulation_results": {}
        }
        print("Erstelle train.json mit folgendem Inhalt:")
        print(json.dumps(self.train_data, indent=4))
        self.logger.debug("Erstelle train.json mit folgendem Inhalt:")
        self.logger.debug(json.dumps(self.train_data, indent=4))
        with open(self.train_json_path, 'w') as f:
            json.dump(self.train_data, f, indent=4)
        print(f"train.json erstellt unter {self.train_json_path}")
        self.logger.info(f"train.json erstellt unter {self.train_json_path}")

    def load_train_json(self):
        if os.path.exists(self.train_json_path):
            try:
                with open(self.train_file, 'r') as f:
                    content = f.read()
                    self.logger.debug("Inhalt von train.json:")
                    self.logger.debug(content)
                    print("Inhalt von train.json:")
                    print(content)
                    if not content.strip():
                        error_msg = "Fehler: train.json ist leer."
                        print(error_msg)
                        self.logger.error(error_msg)
                        self.train_data = {}
                        self.train_optimizer = []
                    else:
                        self.train_data = json.loads(content)
                        self.train_optimizer = list(self.train_data.get('optimizers', {}).keys())
                        self.logger.debug("Bereits ausgeführte Optimierer aus train.json: %s", self.train_optimizer)
                        print("Bereits ausgeführte Optimierer aus train.json:", self.train_optimizer)
            except json.JSONDecodeError as e:
                error_msg = f"Fehler beim Parsen von train.json: {e}"
                print(error_msg)
                self.logger.error(error_msg)
                sys.exit(1)
            except Exception as e:
                error_msg = f"Ein unerwarteter Fehler ist beim Laden von train.json aufgetreten: {e}"
                print(f"Ein Fehler ist aufgetreten: {e}")
                self.logger.error(error_msg)
                sys.exit(1)
        else:
            self.train_data = {}
            self.train_optimizer = []
            self.logger.info("train.json existiert nicht.")
            print("train.json existiert nicht.")

    def set_training_matrix(self, training_matrix):
        self.train_data['training_matrix'] = training_matrix
        self.train_data['creation_date'] = datetime.now().isoformat()
        try:
            with open(self.train_json_path, 'w') as f:
                json.dump(self.train_data, f, indent=4)
            success_msg = "Trainingsmatrix erfolgreich in train.json gespeichert."
            print(success_msg)
            self.logger.info(success_msg)
        except Exception as e:
            error_msg = f"Fehler beim Speichern der Trainingsmatrix in train.json: {e}"
            print(error_msg)
            self.logger.error(error_msg)

    def get_matrix_names(self):
        """
        Gibt die Namen aller Aktivierungsmatrizen zurück.

        :return: Eine Liste von Namen der Aktivierungsmatrizen.
        """
        return list(self.activation_matrices.keys())
