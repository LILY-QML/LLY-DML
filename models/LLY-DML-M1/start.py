#!/usr/bin/env python3
"""
@file start.py
@brief Startskript für die Ausführung des M1 Modells

@details
Dieses Skript dient als Einstiegspunkt zur Ausführung des M1 Modells.
Es importiert und startet den Multi-Matrix-Optimierungsprozess.

@author LILY-QML Team
@version 1.0
@date 2025-04-06
@copyright Copyright (c) 2025 LILY-QML
"""

import sys
import os
from time import time
import argparse

# Konfiguriere Pfade für Importe
current_dir = os.path.dirname(os.path.abspath(__file__))
# LLY-DML Stammverzeichnis ist zwei Ebenen höher
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

# Starte die Zeit
start_time = time()

def main():
    parser = argparse.ArgumentParser(description="LLY-DML M1 Modell Runner")
    parser.add_argument('--visualize-only', action='store_true', 
                     help='Nur Visualisierung der vorhandenen Ergebnisse durchführen')
    args = parser.parse_args()
    
    # Führe das Hauptprogramm aus
    if args.visualize_only:
        try:
            from example import visualize_results
            import json
            import os
            
            # Finde die neueste Ergebnisdatei
            results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
            result_files = [f for f in os.listdir(results_dir) if f.startswith("optimization_results_") and f.endswith(".json")]
            
            if not result_files:
                print("Keine Ergebnisdateien gefunden. Bitte führen Sie zuerst eine Optimierung durch.")
                return
            
            # Sortiere nach Änderungsdatum, neueste zuerst
            result_files.sort(key=lambda x: os.path.getmtime(os.path.join(results_dir, x)), reverse=True)
            latest_file = os.path.join(results_dir, result_files[0])
            
            # Lade die Ergebnisse
            with open(latest_file, 'r') as f:
                results = json.load(f)
            
            print(f"Visualisiere Ergebnisse aus: {latest_file}")
            visualize_results(results)
            print(f"Visualisierungen wurden im Verzeichnis '{results_dir}' gespeichert.")
            
        except Exception as e:
            print(f"Fehler bei der Visualisierung: {e}")
            return
    else:
        try:
            from example import run_multi_matrix_optimization
            run_multi_matrix_optimization()
        except Exception as e:
            print(f"Fehler beim Ausführen des Modells: {e}")
            return

    # Zeige die Gesamtlaufzeit
    end_time = time()
    print(f"\nGesamtlaufzeit: {end_time - start_time:.2f} Sekunden")

if __name__ == "__main__":
    main()