# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Project: LILY-QML
# Version: 1.6 LLY-DML
# Author: Leon Kaiser
# Contact: info@lilyqml.de
# Website: www.lilyqml.de
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++

import importlib
import logging
import numpy as np
from helper.extractor import Extractor

class Optimizer:
    def __init__(self, data, training_matrix, optimizing_method, target_state, **optimizer_kwargs):
        """
        Initialisiert die Optimizer-Klasse mit den bereitgestellten Daten, Trainingsmatrix, Optimierungsmethode und Zielzustand.
        Zusätzliche Keyword-Argumente können für spezifische Optimizer verwendet werden.
        """
        self.data = data
        self.training_matrix = training_matrix
        self.optimizing_method = optimizing_method  # Liste der Optimierungsmethoden
        self.target_state = target_state  # Zielzustand als Binärstring
        self.extracted_data = None
        self.flagged = None
        self.optimizer_instance = None
        self.optimizer_kwargs = optimizer_kwargs  # Zusätzliche Argumente für den Optimizer

        # Logging setup
        self.logger = logging.getLogger('Optimizer')
        self.logger.info(f"Optimizer initialized with method: {optimizing_method} and target state: {target_state}")

    def process_data(self):
        """
        Verarbeitet die Daten mit der Extractor-Klasse und speichert das Ergebnis als extracted_data.
        """
        extractor = Extractor(self.data, self.training_matrix)
        self.extracted_data = extractor.assign_matrix_to_positions()
        self.logger.info("Data processed using Extractor. Extracted data assigned.")

    def difference(self):
        """
        Identifiziert den Zustand (Binärstring) mit den meisten Messungen und speichert ihn in flagged.
        """
        max_value = -1
        for binary_state, measurement in self.data.items():
            if measurement > max_value:
                max_value = measurement
                self.flagged = binary_state
        self.logger.info(f"State with the most measurements identified: {self.flagged}")

    def finish(self):
        """
        Verwendet die Extractor's reconstruct-Methode, um die Trainingsmatrix aus extracted_data wiederherzustellen.
        """
        extractor = Extractor(self.data, self.training_matrix)
        extractor.extracted_data = self.extracted_data  # Verwendung der vorhandenen extrahierten Daten
        result = extractor.reconstruct()
        self.logger.info("Training matrix reconstructed from extracted data.")
        return result

    def load_optimizer(self):
        """
        Lädt die entsprechende Optimizer-Klasse basierend auf optimizing_method und instanziiert sie.
        """
        if not self.optimizing_method:
            raise ValueError("Keine Optimierungsmethode angegeben.")

        method = self.optimizing_method[0]  # Annahme: nur eine Methode wird verwendet
        module_name = f"optimizers.{method.lower()}_optimizer"
        class_name = f"{method}Optimizer"

        try:
            optimizer_module = importlib.import_module(module_name)
            optimizer_class = getattr(optimizer_module, class_name)
            self.optimizer_instance = optimizer_class(
                data=self.data,
                training_matrix=self.training_matrix,
                target_state=self.target_state,
                **self.optimizer_kwargs  # Zusätzliche Argumente weitergeben
            )
            self.logger.info(f"Optimizer loaded: {class_name} from {module_name}")
        except (ModuleNotFoundError, AttributeError) as e:
            self.logger.error(f"Error loading optimizer: {class_name} from {module_name}", exc_info=True)
            raise ImportError(f"Die Optimizer-Klasse '{class_name}' konnte nicht aus '{module_name}' importiert werden.") from e

    def run_optimization(self):
        """
        Führt den Optimierungsprozess durch.
        """
        if not self.optimizer_instance:
            raise ValueError("Optimizer-Instanz ist nicht geladen. Rufe zuerst load_optimizer auf.")

        optimized_params, steps = self.optimizer_instance.optimize()
        self.logger.info(f"Optimization completed. Optimized parameters: {optimized_params}")
        self.logger.info(f"Number of optimization steps: {len(steps)}")
        return optimized_params, steps

    def show_optimization_method(self):
        """
        Zeigt die verwendete Optimierungsmethode an.
        """
        return f"Optimierungsmethode verwendet: {self.optimizing_method}"

if __name__ == "__main__":
    # Konfiguration des Loggings
    logging.basicConfig(level=logging.INFO)

    # Erweiterte Daten mit mehr Einträgen (für realistischere Optimierung)
    data = {
        '000000': 514,
        '000001': 300,
        '000010': 250,
        '000011': 510,
        '000100': 400,
        '000101': 350,
        '000110': 275,
        '000111': 600
    }

    # Trainingsmatrix basierend auf der Binärstring-Länge
    binary_length = len(list(data.keys())[0])
    training_matrix = [[i + j for j in range(1, 5)] for i in range(binary_length)]

    # Zielzustand (final), z.B. ein Binärstring, der angibt, welche Bits angestrebt werden
    final = '000111'  # Beispiel: Wir möchten, dass die letzten drei Bits '1' sind

    # Liste der zu testenden Optimierungsmethoden
    optimizer_methods = ['Adam', 'SGD', 'RMSProp', 'AdaGrad', 'Momentum', 'Nadam']

    # Zusätzliche Optimizer-spezifische Argumente (optional)
    optimizer_kwargs = {
        'Adam': {
            'learning_rate': 0.001,
            'max_iterations': 100,
            'beta1': 0.9,
            'beta2': 0.999,
            'epsilon': 1e-8
        },
        'SGD': {
            'learning_rate': 0.01,
            'max_iterations': 100
        },
        'RMSProp': {
            'learning_rate': 0.001,
            'max_iterations': 100,
            'beta': 0.9,
            'epsilon': 1e-8
        },
        'AdaGrad': {
            'learning_rate': 0.01,
            'max_iterations': 100,
            'epsilon': 1e-8
        },
        'Momentum': {
            'learning_rate': 0.01,
            'max_iterations': 100,
            'momentum': 0.9
        },
        'Nadam': {
            'learning_rate': 0.002,
            'max_iterations': 100,
            'beta1': 0.9,
            'beta2': 0.999,
            'epsilon': 1e-8
        }
    }

    # Ergebnisse speichern
    results = {}

    for method in optimizer_methods:
        print(f"\n===== Testen des Optimizers: {method} =====")
        # Zusätzliche Argumente für den aktuellen Optimizer abrufen
        kwargs = optimizer_kwargs.get(method, {})

        # Instanz der Optimizer-Klasse erstellen
        optimizer = Optimizer(
            data=data,
            training_matrix=training_matrix,
            optimizing_method=[method],
            target_state=final,
            **kwargs
        )
        optimizer.process_data()
        optimizer.difference()

        try:
            # Optimizer laden und Optimierungsprozess durchführen
            optimizer.load_optimizer()
            optimized_params, optimization_steps = optimizer.run_optimization()

            # Finale Matrix rekonstruieren (optional, basierend auf optimierten Parametern)
            reconstructed_matrix = optimizer.finish()

            # Ergebnisse speichern
            results[method] = {
                'Optimierte Parameter': optimized_params,
                'Optimierungsschritte': optimization_steps,
                'Rekonstruierte Matrix': reconstructed_matrix,
                'Flagged Zustand': optimizer.flagged
            }

            # Ausgabe der Ergebnisse
            print(f"Rekonstruierte Matrix: {reconstructed_matrix}")
            print(optimizer.show_optimization_method())
            print(f"Flagged Zustand (mit den meisten Messungen): {optimizer.flagged}")
            print(f"Optimierte Tuning-Parameter: {optimized_params}")
        except ImportError as e:
            print(f"Fehler beim Laden des Optimizers {method}: {e}")
        except Exception as e:
            print(f"Ein unerwarteter Fehler ist bei der Verwendung des Optimizers {method} aufgetreten: {e}")
            logging.exception(f"Fehler beim Optimierungsprozess für {method}: {e}")

    # Optional: Zusammenfassung der Ergebnisse
    print("\n===== Zusammenfassung der Optimierungsergebnisse =====")
    for method, result in results.items():
        print(f"\n--- Optimizer: {method} ---")
        print(f"Rekonstruierte Matrix: {result['Rekonstruierte Matrix']}")
        print(f"Flagged Zustand: {result['Flagged Zustand']}")
        print(f"Optimierte Tuning-Parameter: {result['Optimierte Parameter']}")
        print(f"Anzahl der Optimierungsschritte: {len(result['Optimierungsschritte'])}")
        letzter_step = result['Optimierungsschritte'][-1]
        print(f"Endverlust: {letzter_step['loss']}")
