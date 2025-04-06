#!/usr/bin/env python3
"""
@file example.py
@brief Multi-Matrix Optimierungsbeispiel für die LLY-DML Quantum Circuit Implementierung.

@details
Dieses Modul demonstriert einen erweiterten Anwendungsfall der LLY-DML Plattform,
bei dem mehrere Eingabematrizen für die Optimierung von Quantum Circuits verwendet werden.
Das Programm weist jeder Matrix einen eindeutigen Zielzustand zu und optimiert die
Trainingsparameter des Circuits, um die Wahrscheinlichkeit des entsprechenden Zielzustands
zu maximieren.

@author LILY-QML Team
@version 2.0
@date 2025-04-06
@copyright Copyright (c) 2025 LILY-QML
"""

import os
import sys
import json
import logging
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Konfiguriere Pfade für Importe
current_dir = os.path.dirname(os.path.abspath(__file__))
# LLY-DML Stammverzeichnis ist zwei Ebenen höher
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

# Lokale Module
from src.quantum.circuit import Circuit

# Verwende statt der Standard-Optimierer unsere gepatchten Versionen
import os
current_file_dir = os.path.dirname(os.path.abspath(__file__))

# Importiere die gepatchten Optimierer mit benutzerdefinierten Importen
import sys
import importlib.util

# Funktion zum Importieren einer Datei als Modul
def import_module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Importiere alle gepatchten Optimierer
AdamOptimizerModule = import_module_from_file("patched_adam_optimizer", os.path.join(current_file_dir, "patched_adam_optimizer.py"))
SGDOptimizerModule = import_module_from_file("patched_sgd_optimizer", os.path.join(current_file_dir, "patched_sgd_optimizer.py"))
MomentumOptimizerModule = import_module_from_file("patched_momentum_optimizer", os.path.join(current_file_dir, "patched_momentum_optimizer.py"))
RMSpropOptimizerModule = import_module_from_file("patched_rmsprop_optimizer", os.path.join(current_file_dir, "patched_rmsprop_optimizer.py"))
AdagradOptimizerModule = import_module_from_file("patched_adagrad_optimizer", os.path.join(current_file_dir, "patched_adagrad_optimizer.py"))
NadamOptimizerModule = import_module_from_file("patched_nadam_optimizer", os.path.join(current_file_dir, "patched_nadam_optimizer.py"))

# Hole die Optimizer-Klassen aus den Modulen
AdamOptimizer = AdamOptimizerModule.AdamOptimizer
SGDOptimizer = SGDOptimizerModule.SGDOptimizer
MomentumOptimizer = MomentumOptimizerModule.MomentumOptimizer
RMSpropOptimizer = RMSpropOptimizerModule.RMSpropOptimizer
AdagradOptimizer = AdagradOptimizerModule.AdagradOptimizer
NadamOptimizer = NadamOptimizerModule.NadamOptimizer

# Logging-Konfiguration
log_dir = os.path.join(current_dir, "logs")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "multi_matrix_optimization.log")),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("MultiMatrixOptimizer")


def create_input_matrices(num_matrices, qubits, depth):
    """
    Erstellt eine Reihe von Eingabematrizen mit zufälligen Werten.
    
    @param num_matrices Anzahl der zu erstellenden Matrizen
    @param qubits Anzahl der Qubits im Circuit
    @param depth Tiefe des Circuits (Anzahl der L-Gates pro Qubit)
    
    @return Liste von Eingabematrizen
    """
    input_matrices = []
    
    for i in range(num_matrices):
        # Erstelle eine Matrix mit der Dimension [qubits × depth × 3]
        # für die 3 Inputparameter (IP0, IP1, IP2) pro L-Gate
        matrix = np.random.uniform(0, 2*np.pi, (qubits, depth, 3))
        input_matrices.append(matrix)
        logger.info(f"Eingabematrix {i+1} erstellt mit Form {matrix.shape}")
    
    return input_matrices


def determine_most_probable_state(circuit, input_matrix, assigned_states):
    """
    Bestimmt den wahrscheinlichsten Zustand für eine gegebene Eingabematrix.
    Falls dieser Zustand bereits zugewiesen ist, wird der nächstwahrscheinliche gewählt.
    
    @param circuit Das Circuit-Objekt
    @param input_matrix Die Eingabematrix
    @param assigned_states Liste der bereits zugewiesenen Zustände
    
    @return Tupel (wahrscheinlichster noch nicht zugewiesener Zustand, Wahrscheinlichkeit)
    """
    # Erstelle eine Kopie des Circuits, um die Originalparameter nicht zu verändern
    temp_circuit = Circuit(circuit.qubits, circuit.depth)
    
    # Setze die Eingabematrix in die Input-Phasen
    temp_circuit.place_input_parameters(input_matrix)
    
    # Füge Messungen hinzu
    temp_circuit.place_measurement()
    
    # Führe den Circuit aus
    counts = temp_circuit.execute_circuit(shots=1024)
    
    # Berechne die Wahrscheinlichkeiten
    probabilities = temp_circuit.get_state_probabilities(counts)
    
    # Sortiere die Zustände nach absteigender Wahrscheinlichkeit
    sorted_states = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    
    # Wähle den wahrscheinlichsten Zustand, der noch nicht zugewiesen ist
    for state, prob in sorted_states:
        if state not in assigned_states:
            logger.info(f"Zustand '{state}' mit Wahrscheinlichkeit {prob:.4f} zugewiesen")
            return state, prob
    
    # Sollte nie erreicht werden, da es 2^qubits mögliche Zustände gibt
    logger.error("Konnte keinen freien Zustand finden!")
    return None, 0.0


def load_matrices_from_data_json():
    """
    Lädt Matrizen und Zustände aus data.json im models/M1/var Verzeichnis.
    
    @return Tupel (Matrizen, Matrix-Zustandszuordnung, Konfigurationsparameter)
    """
    var_dir = os.path.join(current_dir, "var")
    data_json_path = os.path.join(var_dir, "data.json")
    
    try:
        with open(data_json_path, 'r') as f:
            data = json.load(f)
        
        logger.info(f"Daten erfolgreich aus {data_json_path} geladen")
        
        # Extrahiere Parameter
        qubits = data.get("qubits", 6)
        depth = data.get("depth", 3)
        
        # Extrahiere Matrizen und Zustände
        matrices = {}
        for name, matrix_list in data.get("matrices", {}).items():
            if matrix_list and isinstance(matrix_list, list):
                matrices[name] = np.array(matrix_list[0])  # Nehme die erste Matrix aus der Liste
        
        # Extrahiere Zustandszuordnungen
        state_mapping = data.get("state_mapping", {})
        
        return matrices, state_mapping, {"qubits": qubits, "depth": depth}
    
    except Exception as e:
        logger.error(f"Fehler beim Laden der Daten aus {data_json_path}: {e}")
        return {}, {}, {"qubits": 5, "depth": 3}


def run_multi_matrix_optimization():
    """
    Führt die Multi-Matrix-Optimierung durch.
    
    @details
    Der Prozess umfasst:
    1. Laden der Konfigurationsdaten aus models/M1/var
    2. Erstellen oder Laden von Eingabematrizen
    3. Bestimmen des wahrscheinlichsten Zustands für jede Matrix
    4. Optimieren des Circuits für jeden Zielzustand mit verschiedenen Optimierern
    5. Auswertung und Visualisierung der Ergebnisse
    
    @return Dictionary mit den Optimierungsergebnissen
    """
    logger.info("=" * 80)
    logger.info("   LLY-DML Multi-Matrix Optimierung")
    logger.info("=" * 80)
    
    # Lade Matrizen und Zustände aus data.json
    input_matrices_dict, state_mapping, params = load_matrices_from_data_json()
    
    # Extrahiere Parameter
    qubits = params["qubits"]
    depth = params["depth"]
    
    # Lade config.json für zusätzliche Parameter
    config_path = os.path.join(current_dir, "var", "config.json")
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        training_iterations = config.get("training", {}).get("iterations", 20)
        logger.info(f"Config geladen. Training Iterationen: {training_iterations}")
    except Exception as e:
        logger.warning(f"Konnte config.json nicht laden: {e}. Verwende Standardwerte.")
        training_iterations = 20
    
    # Bereite Dictionaries für Matrizen und ihre Zustände vor
    input_matrices = []
    matrix_names = []
    assigned_states = []
    
    # Verwende die geladenen Matrizen und Zustände, wenn verfügbar
    if input_matrices_dict and state_mapping:
        for name, matrix in input_matrices_dict.items():
            if name in state_mapping:
                input_matrices.append(matrix)
                matrix_names.append(name)
                assigned_states.append(state_mapping[name])
                logger.info(f"Matrix '{name}' mit Zielzustand '{state_mapping[name]}' geladen")
    
    # Wenn keine oder zu wenige Matrizen geladen wurden, erstelle neue
    if len(input_matrices) < 4:
        num_matrices_to_create = 4 - len(input_matrices)
        logger.info(f"Erstelle {num_matrices_to_create} zusätzliche zufällige Matrizen")
        
        # Erstelle einen temporären Circuit zur Zustandsbestimmung
        temp_circuit = Circuit(qubits, depth)
        
        # Erstelle neue Matrizen und weise ihnen Zustände zu
        for i in range(num_matrices_to_create):
            new_matrix = np.random.uniform(0, 2*np.pi, (qubits, depth, 3))
            state, prob = determine_most_probable_state(temp_circuit, new_matrix, assigned_states)
            
            if state:
                input_matrices.append(new_matrix)
                matrix_names.append(f"random_matrix_{i+1}")
                assigned_states.append(state)
    
    logger.info(f"Verwende Circuit mit {qubits} Qubits und Tiefe {depth}")
    logger.info(f"Anzahl der Matrizen: {len(input_matrices)}")
    logger.info(f"Trainings-Iterationen: {training_iterations}")
    
    # Erstelle einen Circuit
    circuit = Circuit(qubits, depth)
    logger.info("Quantum Circuit erstellt")
    
    # Bestimme für jede Matrix den wahrscheinlichsten Zustand
    matrix_state_mapping = {}
    
    for i, matrix in enumerate(input_matrices):
        matrix_state_mapping[i] = {
            "name": matrix_names[i] if i < len(matrix_names) else f"matrix_{i+1}",
            "state": assigned_states[i],
            "initial_probability": 1.0 / (2**qubits)  # Standardwahrscheinlichkeit
        }
    
    logger.info("Zustandszuordnungen:")
    for i, mapping in matrix_state_mapping.items():
        logger.info(f"Matrix {mapping['name']}: Zustand '{mapping['state']}'")
    
    # Definiere die zu verwendenden Optimierer
    optimizers = {
        "Adam": AdamOptimizer,
        "SGD": SGDOptimizer,
        "Momentum": MomentumOptimizer,
        "RMSprop": RMSpropOptimizer,
        "Adagrad": AdagradOptimizer,
        "Nadam": NadamOptimizer
    }
    
    # Optimierungsergebnisse speichern
    optimization_results = {
        "matrices": [],
        "initial_states": {},
        "final_states": {},
        "training_history": {}
    }
    
    # Für jede Matrix und ihren zugewiesenen Zustand eine Optimierung durchführen
    for matrix_idx, mapping in matrix_state_mapping.items():
        target_state = mapping["state"]
        input_matrix = input_matrices[matrix_idx]
        matrix_name = mapping["name"]
        
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Starte Optimierung für Matrix {matrix_name} mit Zielzustand '{target_state}'")
        
        # Speichere die Anfangsbedingungen
        optimization_results["matrices"].append(input_matrix.tolist())
        optimization_results["initial_states"][matrix_idx] = {
            "name": matrix_name,
            "state": target_state
        }
        
        # Optimiere mit jedem Optimierer
        for opt_name, opt_class in optimizers.items():
            logger.info(f"\nOptimierung mit {opt_name}")
            
            # Initialisiere die Trainingsmatrix
            training_matrix = np.random.uniform(0, np.pi/8, (qubits, depth * 3))  # Flache Initialisierung
            
            # Trainingsgeschichte für diese Matrix
            history = []
            
            # Initialisiere Optimizer-Daten
            optimizer_data = {
                "qubits": qubits,
                "depth": depth
            }
            
            # Erstelle den Optimizer - Achtung: BaseOptimizer geht davon aus, dass target_state ein einzelnes Bit ist
            # Wir verwenden das erste Bit des Zielzustands als einfaches Beispiel
            flat_training_matrix = training_matrix.flatten().tolist()
            first_bit = target_state[0]  # Verwende nur das erste Bit des Zielzustands
            
            optimizer = opt_class(
                data=optimizer_data,
                training_matrix=flat_training_matrix,
                target_state=first_bit,  # Für den BaseOptimizer muss das ein einzelnes Bit sein
                learning_rate=0.01 if opt_name == "SGD" or opt_name == "Momentum" or opt_name == "Adagrad" else 0.001,
                max_iterations=training_iterations
            )
            
            # Füge tuning_parameters-Attribut hinzu, falls es nicht existiert
            if not hasattr(optimizer, 'tuning_parameters'):
                optimizer.tuning_parameters = np.array(flat_training_matrix)
            
            # Führe mehrere Trainingsiterationen durch
            try:
                # Optimiere die Parameter
                optimized_params, optimization_steps = optimizer.optimize()
                
                # Stelle sicher, dass die Eingabematrix die richtige Form hat (qubits, depth, 3)
                if input_matrix.shape != (qubits, depth, 3):
                    logger.warning(f"Eingabematrix hat falsche Form {input_matrix.shape}, sollte aber {(qubits, depth, 3)} sein. Korrigiere...")
                    try:
                        # Versuch, eine korrekte Matrix zu erstellen
                        if len(input_matrix.shape) == 2 and input_matrix.shape[0] == qubits:
                            # Umformen, falls es (qubits, depth*3) ist
                            if input_matrix.shape[1] == depth * 3:
                                reshaped_input = input_matrix.reshape(qubits, depth, 3)
                            else:
                                # Sonst einfach neue Matrix erstellen
                                reshaped_input = np.random.uniform(0, 2*np.pi, (qubits, depth, 3))
                        else:
                            reshaped_input = np.random.uniform(0, 2*np.pi, (qubits, depth, 3))
                            
                        input_matrix = reshaped_input
                    except Exception as e:
                        logger.error(f"Fehler beim Umformen der Eingabematrix: {e}")
                        # Erstelle eine neue Matrix, wenn Umformung fehlschlägt
                        input_matrix = np.random.uniform(0, 2*np.pi, (qubits, depth, 3))
                
                # Platziere die Eingabematrix in den Circuit
                circuit.place_input_parameters(input_matrix)
                
                # Stelle sicher, dass die optimierten Parameter die richtige Form haben
                try:
                    # Wenn die optimierten Parameter nicht die richtige Form haben, korrigiere sie
                    if np.array(optimized_params).shape[0] != qubits * depth * 3:
                        logger.warning("Optimierte Parameter haben falsche Größe. Passe an...")
                        # Erstelle eine neue Matrix mit der richtigen Größe
                        corrected_params = np.zeros((qubits * depth * 3,))
                        # Kopiere Werte, soweit vorhanden
                        corrected_params[:min(len(optimized_params), len(corrected_params))] = np.array(optimized_params)[:min(len(optimized_params), len(corrected_params))]
                        optimized_params = corrected_params.tolist()
                    
                    # Reshape zu (qubits, depth, 3)
                    reshaped_optimized_params = np.array(optimized_params).reshape(qubits, depth, 3)
                except Exception as e:
                    logger.error(f"Fehler beim Umformen der optimierten Parameter: {e}")
                    # Im Fehlerfall erstelle eine neue Matrix
                    reshaped_optimized_params = np.random.uniform(0, np.pi/4, (qubits, depth, 3))
                
                # Setze die optimierten Parameter
                circuit.set_train_parameters(reshaped_optimized_params)
                circuit.place_train_matrix()
                
                # Füge Messungen hinzu und führe den Circuit aus
                circuit.place_measurement()
                counts = circuit.execute_circuit(shots=1024)
                
                # Berechne die aktuelle Wahrscheinlichkeit des Zielzustands
                probabilities = circuit.get_state_probabilities(counts)
                final_prob = probabilities.get(target_state, 0.0)
                
                # Erstelle und speichere die Trainingsgeschichte
                for step in optimization_steps:
                    history.append({
                        "iteration": step.get("iteration", 0),
                        "loss": step.get("loss", 0.0)
                    })
                
                # Speichere die Endergebnisse
                if matrix_idx not in optimization_results["final_states"]:
                    optimization_results["final_states"][matrix_idx] = {}
                
                optimization_results["final_states"][matrix_idx][opt_name] = {
                    "state": target_state,
                    "probability": final_prob,
                    "loss": optimization_steps[-1]["loss"] if optimization_steps else 1.0
                }
                
                if matrix_idx not in optimization_results["training_history"]:
                    optimization_results["training_history"][matrix_idx] = {}
                
                optimization_results["training_history"][matrix_idx][opt_name] = history
                
                logger.info(f"Optimierung mit {opt_name} abgeschlossen. Finale Wahrscheinlichkeit für Zustand '{target_state}': {final_prob:.4f}")
                
            except Exception as e:
                logger.error(f"Fehler bei der Optimierung mit {opt_name}: {e}")
                # Hier geben wir mehr Zeit, um den Fehler zu protokollieren und Ressourcen zu bereinigen
                time.sleep(0.5)
                continue
    
    # Visualisiere die Ergebnisse für jede Matrix
    visualize_results(optimization_results)
    
    # Speichere die Ergebnisse
    results_dir = os.path.join(current_dir, "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    results_file = os.path.join(results_dir, f"optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    with open(results_file, 'w') as f:
        json.dump(optimization_results, f, indent=2)
    
    logger.info(f"Ergebnisse gespeichert in {results_file}")
    logger.info("=" * 80)
    logger.info("Multi-Matrix Optimierung abgeschlossen")
    logger.info("=" * 80)
    
    return optimization_results


def visualize_results(results):
    """
    Visualisiert die Optimierungsergebnisse.
    
    @param results Dictionary mit den Optimierungsergebnissen
    """
    results_dir = os.path.join(current_dir, "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Erstelle eine Abbildung für die Trainingsgeschichte pro Optimierer
    for opt_name in results["training_history"][0].keys():  # Nehme erste Matrix, um Optimierer zu identifizieren
        plt.figure(figsize=(14, 8))
        
        for matrix_idx, histories in results["training_history"].items():
            if opt_name in histories:
                history = histories[opt_name]
                iterations = [entry["iteration"] for entry in history]
                losses = [entry["loss"] for entry in history]
                
                matrix_name = results["initial_states"][matrix_idx]["name"]
                target_state = results["initial_states"][matrix_idx]["state"]
                final_state = results["final_states"][matrix_idx][opt_name]
                
                plt.plot(iterations, losses, marker='o', 
                         label=f"{matrix_name} - Zustand '{target_state}' (Final P: {final_state['probability']:.3f})")
        
        plt.title(f"Trainingsfortschritt mit {opt_name} Optimizer")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # Speichere die Abbildung
        plt.savefig(os.path.join(results_dir, f"training_progress_{opt_name}.png"))
        plt.close()
    
    # Erstelle eine Abbildung, die alle Optimierer für jede Matrix vergleicht
    for matrix_idx in results["initial_states"].keys():
        plt.figure(figsize=(12, 6))
        
        matrix_name = results["initial_states"][matrix_idx]["name"]
        target_state = results["initial_states"][matrix_idx]["state"]
        
        # Extrahiere die Ergebnisse für alle Optimierer
        opt_names = []
        final_probs = []
        
        for opt_name, final_state in results["final_states"][matrix_idx].items():
            opt_names.append(opt_name)
            final_probs.append(final_state["probability"])
        
        # Erstelle das Balkendiagramm
        plt.bar(opt_names, final_probs, color='skyblue')
        plt.title(f"Vergleich der Optimierer für {matrix_name} (Zielzustand: {target_state})")
        plt.xlabel("Optimierer")
        plt.ylabel("Finale Wahrscheinlichkeit")
        plt.grid(True, axis='y', alpha=0.3)
        plt.ylim(0, 1.0)
        
        # Füge Werte über den Balken hinzu
        for i, prob in enumerate(final_probs):
            plt.text(i, prob + 0.02, f"{prob:.3f}", ha='center')
        
        plt.tight_layout()
        
        # Speichere die Abbildung
        plt.savefig(os.path.join(results_dir, f"optimizer_comparison_{matrix_name}.png"))
        plt.close()
    
    # Erstelle eine Gesamtübersicht für alle Matrizen und Optimierer
    plt.figure(figsize=(16, 8))
    
    # Vorbereitung der Daten
    matrix_indices = list(results["initial_states"].keys())
    opt_names = list(results["final_states"][matrix_indices[0]].keys()) if matrix_indices else []
    
    # Erstelle ein Gruppiertes Balkendiagramm
    bar_width = 0.8 / (len(opt_names) if opt_names else 1)
    
    for i, opt_name in enumerate(opt_names):
        # Position der Gruppe bestimmen
        x = np.arange(len(matrix_indices))
        
        # Wahrscheinlichkeiten für diesen Optimierer sammeln
        probs = []
        for matrix_idx in matrix_indices:
            if matrix_idx in results["final_states"] and opt_name in results["final_states"][matrix_idx]:
                probs.append(results["final_states"][matrix_idx][opt_name]["probability"])
            else:
                probs.append(0)
        
        # Offset für diesen Optimierer berechnen
        offset = i * bar_width - 0.4 + bar_width/2
        
        # Balken plotten
        plt.bar(x + offset, probs, bar_width, label=opt_name)
    
    # Labels und Legende
    plt.xlabel("Matrix")
    plt.ylabel("Finale Wahrscheinlichkeit des Zielzustands")
    plt.title("Vergleich aller Optimierer für alle Matrizen")
    plt.xticks(np.arange(len(matrix_indices)), 
               [results["initial_states"][idx]["name"] + f" ('{results['initial_states'][idx]['state']}')" 
                for idx in matrix_indices])
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.ylim(0, 1.0)
    plt.tight_layout()
    
    # Speichere die Abbildung
    plt.savefig(os.path.join(results_dir, "overall_comparison.png"))
    plt.close()


if __name__ == "__main__":
    try:
        print("\nStarte Multi-Matrix-Optimierung. Dies kann einige Zeit dauern...\n")
        results = run_multi_matrix_optimization()
        if results:
            print("\nOptimierung erfolgreich abgeschlossen.")
            print(f"Ergebnisse wurden im Verzeichnis {os.path.join(current_dir, 'results')} gespeichert und visualisiert.")
        else:
            print("\nOptimierung fehlgeschlossen. Siehe Logdatei für Details.")
    except Exception as e:
        logger.exception(f"Unerwarteter Fehler: {e}")
        print(f"\nFehler bei der Ausführung: {e}")
        print(f"Details wurden in der Logdatei {os.path.join(log_dir, 'multi_matrix_optimization.log')} gespeichert.")