"""
@file circuit.py
@brief Implementierung einer Quantum Circuit-Klasse für die LLY-DML Plattform.

@details
Dieses Modul definiert die Circuit-Klasse, die als Hauptbestandteil der LLY-DML
Quantumcomputing-Architektur dient. Die Klasse erlaubt das Design und die Manipulation
von Quantenschaltkreisen, speziell mit L-Gate-Strukturen, die für differenzielles
maschinelles Lernen optimiert sind.

@author LILY-QML Team
@version 1.0
@date 2025-04-05
@copyright Copyright (c) 2025 LILY-QML
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.visualization import circuit_drawer
from qiskit_aer import Aer
from qiskit import transpile


class Circuit:
    """
    @class Circuit
    @brief Implementierung eines robusten und erweiterbaren Quantum Circuits.
    
    @details
    Die Circuit-Klasse dient zur Erstellung und Verwaltung eines robusten Quantum Circuits. 
    Sie erlaubt das Design von Gatekombinationen, das Setzen von Trainingsparametern 
    und das Einfügen von Messpunkten. Die Klasse ist so aufgebaut, dass sie 
    skalierbar und erweiterbar bleibt.
    
    Die Hauptfunktionalitäten umfassen:
    - Erstellung von Quantum Circuits mit variabler Anzahl von Qubits
    - Definition von L-Gates als fundamentale Baueinheiten
    - Generierung und Platzierung von Trainingsparametern
    - Visualisierung der Circuit-Struktur
    - Ausführung von Schaltkreisen auf Quantensimulatoren
    - Analyse von Messergebnissen
    
    @note Ein L-Gate besteht aus der Sequenz: 
          TP0 → IP0 → H → TP1 → IP1 → H → TP2 → IP2
          
          Es enthält insgesamt 8 Gates:
          - 3 Trainingsphasen (TP0, TP1, TP2) als Phasengates (P) mit Trainingsparametern
          - 3 Inputphasen (IP0, IP1, IP2) als Phasengates (P) mit festen Inputparametern
          - 2 Hadamard-Gates (H)
    
    @author LILY-QML Team
    @version 1.0
    @date 2025-04-05
    @copyright Copyright (c) 2025 LILY-QML
    """

    def __init__(self, qubits, depth, input_params=None):
        """
        @fn __init__
        @brief Initialisiert ein neues Circuit-Objekt mit der spezifizierten Konfiguration.
        
        @details
        Erstellt ein neues Circuit-Objekt mit der angegebenen Anzahl von Qubits
        und L-Gates pro Qubit (depth). Initialisiert automatisch den Circuit und
        eine zufällige Trainingsmatrix.
        
        Der Konstruktor führt folgende Schritte aus:
        1. Speichern der Basisparameter (qubits, depth)
        2. Initialisierung der internen Zustandsvariablen
        3. Setzen der Inputparameter, wenn angegeben, sonst Standardwerte
        4. Aufruf von build_circuit() zur Erstellung des leeren Schaltkreises
        5. Aufruf von create_train_matrix() zur Generierung zufälliger Trainingsparameter
        
        @param[in] qubits Anzahl der Qubits im Circuit
        @param[in] depth Anzahl der L-Gates pro Qubit
        @param[in] input_params Optional: Array der Inputparameter [IP0, IP1, IP2] 
                   (Standard: [0, 0, 0])
        
        @throws ValueError Wenn ungültige Parameter übergeben werden (z.B. qubits <= 0)
        
        @pre qubits > 0 und depth > 0
        @post Ein vollständig initialisierter Circuit mit qubits * depth L-Gates
        
        @see build_circuit()
        @see create_train_matrix()
        """
        # Validiere die Eingabeparameter
        if qubits <= 0 or depth <= 0:
            raise ValueError("Die Anzahl der Qubits und die Tiefe müssen positive Ganzzahlen sein.")
            
        # Speichere die Basisparameter
        self.qubits = qubits
        self.depth = depth
        
        # Setze Inputparameter (IP0, IP1, IP2)
        if input_params is None:
            self.input_params = np.zeros(3)  # Standard: alle Inputparameter sind 0
        else:
            if len(input_params) != 3:
                raise ValueError("Input-Parameter müssen ein Array der Länge 3 sein: [IP0, IP1, IP2]")
            self.input_params = np.array(input_params)
        
        # Initialisiere interne Zustandsvariablen
        self.circuit = None
        self.train_matrix = None
        
        # Erzeuge den Circuit und die Trainingsmatrix
        self.build_circuit()
        self.create_train_matrix()

    def build_circuit(self):
        """
        @fn build_circuit
        @brief Erstellt einen Quantum Circuit mit der gegebenen Anzahl an Qubits und der festgelegten Depth.
        
        @details
        Diese Methode konstruiert den Grundaufbau des Quantum Circuits entsprechend der 
        im Konstruktor angegebenen Parameter. Jede Qubit-Spalte enthält 'depth' L-Gates, 
        wobei ein L-Gate aus einer definierten Sequenz von Gates besteht.
        
        Der Aufbau eines L-Gates folgt diesem Muster:
        - TP0 → IP0 → H → TP1 → IP1 → H → TP2 → IP2
        
        Pro L-Gate werden folgende Gates in dieser Reihenfolge platziert:
        1. TP0: Trainingsphase 0 (P-Gate) mit Trainingsparameter (initial 0)
        2. IP0: Inputphase 0 (P-Gate) mit Inputparameter (fest)
        3. H: Hadamard-Gate (festes Gate)
        4. TP1: Trainingsphase 1 (P-Gate) mit Trainingsparameter (initial 0)
        5. IP1: Inputphase 1 (P-Gate) mit Inputparameter (fest)
        6. H: Hadamard-Gate (festes Gate)
        7. TP2: Trainingsphase 2 (P-Gate) mit Trainingsparameter (initial 0)
        8. IP2: Inputphase 2 (P-Gate) mit Inputparameter (fest)
        
        @return QuantumCircuit Der erstellte Quantum Circuit mit initialisierten Gates
        
        @note Alle Trainingsparameter-Gates (TP) werden zunächst mit Parameterwert 0 initialisiert.
              Die tatsächlichen Trainingsparameter werden später mit place_train_matrix() gesetzt.
              Die Inputparameter (IP) werden bereits hier mit den im Konstruktor übergebenen
              oder den Standardwerten initialisiert.
        
        @pre self.qubits und self.depth sind gültige positive Ganzzahlen
        @post self.circuit enthält einen vollständig initialisierten Quantum Circuit
        
        @see place_train_matrix()
        """
        # Initialisiere Quantum-Register für die Qubits
        qr = QuantumRegister(self.qubits, 'q')
        self.circuit = QuantumCircuit(qr)
        
        # Für jedes L-Gate in der Tiefe (depth)
        for d in range(self.depth):
            # Für jedes Qubit
            for q in range(self.qubits):
                # L-Gate Sequenz: TP0 → IP0 → H → TP1 → IP1 → H → TP2 → IP2
                
                # Füge Trainingsphase 0 (TP0) hinzu - mit Platzhalter 0, wird später durch Trainingsparameter ersetzt
                self.circuit.p(0, q)
                
                # Füge Inputphase 0 (IP0) hinzu - mit festem Inputparameter
                self.circuit.p(self.input_params[0], q)
                
                # Füge erstes Hadamard-Gate hinzu
                self.circuit.h(q)
                
                # Füge Trainingsphase 1 (TP1) hinzu - mit Platzhalter 0, wird später durch Trainingsparameter ersetzt
                self.circuit.p(0, q)
                
                # Füge Inputphase 1 (IP1) hinzu - mit festem Inputparameter
                self.circuit.p(self.input_params[1], q)
                
                # Füge zweites Hadamard-Gate hinzu
                self.circuit.h(q)
                
                # Füge Trainingsphase 2 (TP2) hinzu - mit Platzhalter 0, wird später durch Trainingsparameter ersetzt
                self.circuit.p(0, q)
                
                # Füge Inputphase 2 (IP2) hinzu - mit festem Inputparameter
                self.circuit.p(self.input_params[2], q)
        
        return self.circuit

    def plot_circuit(self):
        """
        @fn plot_circuit
        @brief Visualisiert den aktuell erstellten Quantum Circuit.
        
        @details
        Diese Methode erzeugt eine grafische Darstellung des aktuellen Quantum Circuits
        mit Hilfe der qiskit Visualisierungswerkzeuge. Sie zeigt die Qubit-Register,
        alle Gates und deren Verbindungen an.
        
        Die Visualisierung enthält:
        - Die Qubit-Linien mit Bezeichnungen
        - Alle angewendeten Gates in Reihenfolge
        - Einen Titel mit Informationen zu Qubits und Depth
        
        @return matplotlib.figure Eine Figure-Instanz, die den Circuit darstellt
                und zur weiteren Bearbeitung oder Anzeige verwendet werden kann
        
        @throws ValueError Wenn der Circuit noch nicht erstellt wurde (self.circuit == None)
        
        @pre self.circuit muss initialisiert sein (nach build_circuit())
        @post Eine matplotlib-Figure wird zurückgegeben, aber nicht automatisch angezeigt
        
        @see build_circuit()
        """
        # Prüfe, ob der Circuit bereits erstellt wurde
        if self.circuit is None:
            raise ValueError("Circuit muss zuerst mit build_circuit() erstellt werden.")
            
        # Zeichne den Circuit mit qiskit's integrierter Funktion
        fig = circuit_drawer(self.circuit, output='mpl')
        
        # Füge Titel und Informationen hinzu
        plt.title(f"Quantum Circuit mit {self.qubits} Qubits und {self.depth} L-Gates")
        
        return fig

    def create_train_matrix(self):
        """
        @fn create_train_matrix
        @brief Generiert eine zufällige Trainingsmatrix für die Trainingsphasen der Quantengates.
        
        @details
        Diese Methode erstellt eine dreidimensionale Matrix mit Zufallswerten,
        die als Trainingsparameter für die TP-Gates (Trainingsphasen) im Quantum Circuit dienen.
        
        Die erzeugte Matrix hat folgende Dimensionen und Verwendungszwecke:
        - Erste Dimension [qubits]: Entspricht jedem Qubit im Circuit
        - Zweite Dimension [depth]: Entspricht jedem L-Gate in der Sequenz
        - Dritte Dimension [3]: Speichert die 3 Trainingsparameter pro L-Gate:
          * Index [q][d][0]: Parameter für TP0 (Trainingsphase 0) - P-Gate
          * Index [q][d][1]: Parameter für TP1 (Trainingsphase 1) - P-Gate
          * Index [q][d][2]: Parameter für TP2 (Trainingsphase 2) - P-Gate
        
        Alle Werte werden zufällig im Bereich [0, 2π] generiert, was dem vollen
        Phasenbereich der Quantengates entspricht.
        
        Wichtig: Diese Matrix enthält NUR die Trainingsparameter, NICHT die Inputparameter.
        Die Inputparameter werden separat im self.input_params Array gespeichert.
        
        @return numpy.ndarray Die erstellte Trainingsmatrix der Dimension [qubits × depth × 3]
        
        @pre self.qubits und self.depth müssen initialisiert sein
        @post self.train_matrix enthält zufällige Werte im Bereich [0, 2π]
        
        @see place_train_matrix()
        """
        # Erzeuge Matrix mit zufälligen Werten zwischen 0 und 2π
        # Dimension: [qubits × depth × 3] für die 3 Trainingsparameter pro L-Gate (TP0, TP1, TP2)
        self.train_matrix = np.random.uniform(0, 2*np.pi, (self.qubits, self.depth, 3))
        
        return self.train_matrix

    def place_train_matrix(self):
        """
        @fn place_train_matrix
        @brief Überträgt die Trainingsdaten aus der Matrix in den Quantum Circuit.
        
        @details
        Diese Methode erstellt einen neuen Circuit und setzt die Werte aus der 
        Trainingsmatrix als Parameter für die Trainingsphase-Gates (TP) im Circuit.
        Der Prozess ersetzt den ursprünglichen Circuit vollständig durch einen neuen
        mit den parametrisierten Gates.
        
        Die mathematische Zuordnung erfolgt für jedes Qubit i und jedes L-Gate j:
        - TP0: self.train_matrix[i][j][0] - Trainingsphase 0 (Trainingsparameter)
        - IP0: self.input_params[0] - Inputphase 0 (fester Inputparameter)
        - H: (kein Parameter) - Hadamard-Gate
        - TP1: self.train_matrix[i][j][1] - Trainingsphase 1 (Trainingsparameter)
        - IP1: self.input_params[1] - Inputphase 1 (fester Inputparameter)
        - H: (kein Parameter) - Hadamard-Gate
        - TP2: self.train_matrix[i][j][2] - Trainingsphase 2 (Trainingsparameter)
        - IP2: self.input_params[2] - Inputphase 2 (fester Inputparameter)
        
        @return QuantumCircuit Der Circuit mit den eingesetzten Trainingsparametern
        
        @throws ValueError Wenn die Trainingsmatrix oder der Circuit nicht erstellt wurden
        
        @pre self.train_matrix und self.circuit müssen initialisiert sein
        @post self.circuit enthält den neuen Circuit mit parametrisierten Gates
        
        @see create_train_matrix()
        @see build_circuit()
        """
        # Prüfe, ob die notwendigen Voraussetzungen erfüllt sind
        if self.train_matrix is None:
            raise ValueError("Train-Matrix muss zuerst mit create_train_matrix() erstellt werden.")
            
        if self.circuit is None:
            raise ValueError("Circuit muss zuerst mit build_circuit() erstellt werden.")
        
        # Erstelle einen neuen Circuit, da wir die Gates mit Parametern ersetzen müssen
        qr = QuantumRegister(self.qubits, 'q')
        new_circuit = QuantumCircuit(qr)
        
        # Für jedes L-Gate in der Tiefe (depth)
        for d in range(self.depth):
            # Für jedes Qubit
            for q in range(self.qubits):
                # L-Gate Sequenz: TP0 → IP0 → H → TP1 → IP1 → H → TP2 → IP2
                
                # Setze Trainingsphase 0 (TP0) - Trainingsparameter aus der Matrix
                new_circuit.p(self.train_matrix[q][d][0], q)
                
                # Setze Inputphase 0 (IP0) - fester Inputparameter
                new_circuit.p(self.input_params[0], q)
                
                # Füge erstes Hadamard-Gate hinzu (keine Parameter)
                new_circuit.h(q)
                
                # Setze Trainingsphase 1 (TP1) - Trainingsparameter aus der Matrix
                new_circuit.p(self.train_matrix[q][d][1], q)
                
                # Setze Inputphase 1 (IP1) - fester Inputparameter
                new_circuit.p(self.input_params[1], q)
                
                # Füge zweites Hadamard-Gate hinzu (keine Parameter)
                new_circuit.h(q)
                
                # Setze Trainingsphase 2 (TP2) - Trainingsparameter aus der Matrix
                new_circuit.p(self.train_matrix[q][d][2], q)
                
                # Setze Inputphase 2 (IP2) - fester Inputparameter
                new_circuit.p(self.input_params[2], q)
        
        # Ersetze den alten Circuit durch den neuen
        self.circuit = new_circuit
        
        return self.circuit

    def place_measurement(self):
        """
        @fn place_measurement
        @brief Fügt am Ende des Circuits für jedes Qubit einen Messpunkt hinzu.
        
        @details
        Diese Methode erweitert den bestehenden Quantum Circuit um klassische Register
        und Messoperationen. Die Messungen werden für jedes Qubit einzeln am Ende
        des Circuits hinzugefügt, um die Quantenzustände zu erfassen.
        
        Der Prozess umfasst:
        1. Erstellung eines klassischen Registers mit gleicher Größe wie das Quantenregister
        2. Erstellung eines neuen Circuits, der Quantum- und klassische Register enthält
        3. Komposition des bestehenden Circuits mit dem neuen Circuit
        4. Hinzufügen von Messoperationen für jedes Qubit
        
        @return qiskit.QuantumCircuit Der Circuit mit hinzugefügten Messpunkten
        
        @throws ValueError Wenn der Circuit noch nicht erstellt wurde
        
        @pre self.circuit muss initialisiert sein
        @post self.circuit enthält klassische Register und Messoperationen
        
        @see build_circuit()
        @see execute_circuit()
        """
        # Prüfe, ob der Circuit bereits erstellt wurde
        if self.circuit is None:
            raise ValueError("Circuit muss zuerst mit build_circuit() erstellt werden.")
            
        # Füge klassische Register für die Messergebnisse hinzu
        cr = ClassicalRegister(self.qubits, 'c')
        circuit_with_measurement = QuantumCircuit(self.circuit.qregs[0], cr)
        
        # Füge die bestehenden Gates hinzu
        circuit_with_measurement.compose(self.circuit, inplace=True)
        
        # Füge Messungen hinzu
        for q in range(self.qubits):
            circuit_with_measurement.measure(q, q)
        
        # Ersetze den alten Circuit durch den neuen
        self.circuit = circuit_with_measurement
        
        return self.circuit
        
    def execute_circuit(self, shots=1024):
        """
        @fn execute_circuit
        @brief Führt den Quantum Circuit auf einem Simulator aus.
        
        @details
        Diese Methode führt den aktuellen Quantum Circuit auf einem Quantensimulator
        aus und gibt die Messergebnisse zurück. Sie stellt sicher, dass Messoperationen
        vorhanden sind und bereitet den Circuit für die Simulation vor.
        
        Der Ausführungsprozess umfasst:
        1. Prüfung, ob der Circuit initialisiert ist und Messungen enthält
        2. Hinzufügen von Messoperationen, falls noch keine vorhanden sind
        3. Transpilierung des Circuits für den Zielsimulator
        4. Ausführung der Simulation mit der angegebenen Anzahl von Shots
        5. Erfassung und Rückgabe der Messergebnisse
        
        @param[in] shots Anzahl der Wiederholungen des Experiments (Standard: 1024)
        
        @return dict Ein Dictionary mit den gemessenen Bitstrings als Keys und der Anzahl als Values
        
        @throws ValueError Wenn der Circuit noch nicht erstellt wurde oder keine Messungen enthält
        
        @pre self.circuit muss initialisiert sein
        @post Die Messergebnisse werden als Dictionary zurückgegeben
        
        @see place_measurement()
        """
        # Prüfe, ob der Circuit bereits erstellt wurde
        if self.circuit is None:
            raise ValueError("Circuit muss zuerst mit build_circuit() erstellt werden.")
            
        # Stelle sicher, dass Messungen vorhanden sind
        if not hasattr(self.circuit, 'cregs') or not self.circuit.cregs:
            self.place_measurement()
            
        # Führe den Circuit auf dem Aer Simulator aus
        simulator = Aer.get_backend('aer_simulator')
        
        # Transpiliere den Circuit für den Simulator
        transpiled_circuit = transpile(self.circuit, simulator)
        
        # Führe den Circuit aus
        job = simulator.run(transpiled_circuit, shots=shots)
        result = job.result()
        counts = result.get_counts(0)
        
        return counts
        
    def set_train_parameters(self, parameters):
        """
        @fn set_train_parameters
        @brief Setzt spezifische Trainingsparameter in der Trainingsmatrix.
        
        @details
        Diese Methode ermöglicht es, bestimmte oder alle Trainingsparameter
        manuell zu setzen, anstatt die zufällig generierten Werte zu verwenden.
        Dies ist besonders nützlich für das Training oder für reproduzierbare Experimente.
        
        Die Methode kann unterschiedliche Parameterformate verarbeiten:
        - Ein einzelner Wert: Wird für alle Trainingsparameter verwendet
        - Eine 3D-Matrix: Wird direkt als neue Trainingsmatrix verwendet
        - Andere Formate: Werden nach Möglichkeit konvertiert oder lösen einen Fehler aus
        
        @param[in] parameters Die zu setzenden Parameter, entweder als Skalar oder als Matrix
        
        @return numpy.ndarray Die aktualisierte Trainingsmatrix
        
        @throws ValueError Wenn die Parameter ein ungültiges Format haben
        
        @pre self.train_matrix muss initialisiert sein
        @post self.train_matrix enthält die neuen Parameter
        
        @see create_train_matrix()
        @see place_train_matrix()
        """
        # Prüfe, ob die Trainingsmatrix initialisiert ist
        if self.train_matrix is None:
            self.create_train_matrix()
            
        # Verarbeite verschiedene Parameterformate
        if np.isscalar(parameters):
            # Wenn ein einzelner Wert übergeben wird, verwende ihn für alle Parameter
            self.train_matrix.fill(parameters)
        elif isinstance(parameters, np.ndarray):
            # Wenn eine Matrix übergeben wird, überprüfe die Dimensionen
            if parameters.shape == self.train_matrix.shape:
                self.train_matrix = parameters
            else:
                raise ValueError(f"Die Parametermatrix muss die Form {self.train_matrix.shape} haben.")
        else:
            # Versuche, die Parameter zu konvertieren
            try:
                parameters = np.array(parameters)
                if parameters.shape == self.train_matrix.shape:
                    self.train_matrix = parameters
                else:
                    raise ValueError(f"Die Parametermatrix muss die Form {self.train_matrix.shape} haben.")
            except:
                raise ValueError("Ungültiges Parameterformat. Erwartet wird ein Skalar oder eine numpy.ndarray.")
                
        return self.train_matrix
        
    def set_input_parameters(self, input_params):
        """
        @fn set_input_parameters
        @brief Setzt die festen Inputparameter für die Inputphasen.
        
        @details
        Diese Methode ermöglicht es, die Inputparameter manuell zu setzen.
        Die Eingabeparameter werden für die drei Inputphasen (IP0, IP1, IP2) in einem
        L-Gate verwendet. Diese Parameter sind fest für alle Qubits und alle L-Gates
        im gesamten Circuit.
        
        Nach dem Setzen der Inputparameter wird der Circuit neu gebaut, um die
        Änderungen zu übernehmen.
        
        @param[in] input_params Array der Inputparameter [IP0, IP1, IP2]
        
        @return numpy.ndarray Das aktualisierte Inputparameter-Array
        
        @throws ValueError Wenn die Parameter ein ungültiges Format haben
        
        @pre self.input_params muss initialisiert sein
        @post self.input_params enthält die neuen Parameter und der Circuit wird neu gebaut
        
        @see build_circuit()
        """
        # Validiere die Eingabeparameter
        if len(input_params) != 3:
            raise ValueError("Input-Parameter müssen ein Array der Länge 3 sein: [IP0, IP1, IP2]")
        
        # Setze die neuen Inputparameter
        self.input_params = np.array(input_params)
        
        # Baue den Circuit neu, um die Änderungen an den Inputparametern zu übernehmen
        self.build_circuit()
        
        # Platziere die Trainingsmatrix im neuen Circuit, falls bereits vorhanden
        if self.train_matrix is not None:
            self.place_train_matrix()
            
        return self.input_params
        
    def place_input_parameters(self, input_data_matrix):
        """
        @fn place_input_parameters
        @brief Platziert benutzerdefinierte Inputdaten in den Quantum Circuit.
        
        @details
        Diese Methode erlaubt es, individuelle Inputparameter für jedes Qubit und jedes
        L-Gate zu setzen, anstatt die globalen Inputparameter zu verwenden. Dies ist nützlich
        für die Verarbeitung von Inputdaten, die von Qubit zu Qubit und von L-Gate zu L-Gate
        variieren sollen.
        
        Die input_data_matrix muss die Dimension [qubits × depth × 3] haben, wobei:
        - Erste Dimension [qubits]: Entspricht jedem Qubit im Circuit
        - Zweite Dimension [depth]: Entspricht jedem L-Gate in der Sequenz
        - Dritte Dimension [3]: Speichert die 3 Inputparameter pro L-Gate:
          * Index [q][d][0]: Parameter für IP0 (Inputphase 0) - P-Gate
          * Index [q][d][1]: Parameter für IP1 (Inputphase 1) - P-Gate
          * Index [q][d][2]: Parameter für IP2 (Inputphase 2) - P-Gate
        
        Diese Methode erstellt einen neuen Circuit mit den angegebenen Inputparametern
        anstelle der globalen self.input_params Werte.
        
        @param[in] input_data_matrix Matrix der Inputparameter der Dimension [qubits × depth × 3]
        
        @return QuantumCircuit Der Circuit mit den eingesetzten Inputparametern
        
        @throws ValueError Wenn die Matrix falsche Dimensionen hat oder der Circuit nicht erstellt wurde
        
        @pre self.circuit muss initialisiert sein
        @post self.circuit enthält den neuen Circuit mit individuellen Inputparametern
        
        @see place_train_matrix()
        @see set_input_parameters()
        """
        # Prüfe, ob der Circuit bereits erstellt wurde
        if self.circuit is None:
            raise ValueError("Circuit muss zuerst mit build_circuit() erstellt werden.")
            
        # Validiere die Eingabematrix
        if not isinstance(input_data_matrix, np.ndarray) or input_data_matrix.shape != (self.qubits, self.depth, 3):
            raise ValueError(f"Die Inputdaten-Matrix muss die Form {(self.qubits, self.depth, 3)} haben.")
        
        # Erstelle einen neuen Circuit, da wir die Gates mit individuellen Parametern ersetzen müssen
        qr = QuantumRegister(self.qubits, 'q')
        new_circuit = QuantumCircuit(qr)
        
        # Trainingsparameter aus der Trainingsmatrix verwenden, falls vorhanden
        if self.train_matrix is None:
            train_matrix = np.zeros((self.qubits, self.depth, 3))
        else:
            train_matrix = self.train_matrix
        
        # Für jedes L-Gate in der Tiefe (depth)
        for d in range(self.depth):
            # Für jedes Qubit
            for q in range(self.qubits):
                # L-Gate Sequenz: TP0 → IP0 → H → TP1 → IP1 → H → TP2 → IP2
                
                # Setze Trainingsphase 0 (TP0) - Trainingsparameter aus der Matrix
                new_circuit.p(train_matrix[q][d][0], q)
                
                # Setze Inputphase 0 (IP0) - individueller Inputparameter aus der Matrix
                new_circuit.p(input_data_matrix[q][d][0], q)
                
                # Füge erstes Hadamard-Gate hinzu (keine Parameter)
                new_circuit.h(q)
                
                # Setze Trainingsphase 1 (TP1) - Trainingsparameter aus der Matrix
                new_circuit.p(train_matrix[q][d][1], q)
                
                # Setze Inputphase 1 (IP1) - individueller Inputparameter aus der Matrix
                new_circuit.p(input_data_matrix[q][d][1], q)
                
                # Füge zweites Hadamard-Gate hinzu (keine Parameter)
                new_circuit.h(q)
                
                # Setze Trainingsphase 2 (TP2) - Trainingsparameter aus der Matrix
                new_circuit.p(train_matrix[q][d][2], q)
                
                # Setze Inputphase 2 (IP2) - individueller Inputparameter aus der Matrix
                new_circuit.p(input_data_matrix[q][d][2], q)
        
        # Ersetze den alten Circuit durch den neuen
        self.circuit = new_circuit
        
        return self.circuit
        
    def get_state_probabilities(self, counts, binary=True):
        """
        @fn get_state_probabilities
        @brief Berechnet die Wahrscheinlichkeiten der gemessenen Quantenzustände.
        
        @details
        Diese Methode konvertiert die Ergebnisse der Circuit-Ausführung (Counts)
        in Wahrscheinlichkeiten. Sie normalisiert die Anzahl der Messungen für
        jeden Zustand durch die Gesamtzahl der Shots.
        
        Optional kann das Ergebnis in binärer oder dezimaler Form zurückgegeben werden:
        - Binär: Die Zustände werden als Bitstrings dargestellt ('00', '01', etc.)
        - Dezimal: Die Zustände werden als Dezimalzahlen dargestellt (0, 1, etc.)
        
        @param[in] counts Dictionary mit den Messergebnissen aus execute_circuit()
        @param[in] binary Flag, ob die Zustände als Binärstrings (True) oder 
                         Dezimalzahlen (False) dargestellt werden sollen (Standard: True)
        
        @return dict Ein Dictionary mit den Zuständen als Keys und ihren Wahrscheinlichkeiten als Values
        
        @pre counts muss ein gültiges Dictionary mit Messergebnissen sein
        @post Ein neues Dictionary mit normalisierten Wahrscheinlichkeiten wird zurückgegeben
        
        @see execute_circuit()
        """
        # Berechne die Gesamtzahl der Shots
        total_shots = sum(counts.values())
        
        # Initialisiere das Ergebnis-Dictionary
        probabilities = {}
        
        # Berechne die Wahrscheinlichkeiten für jeden Zustand
        for state, count in counts.items():
            if binary:
                # Behalte den Zustand als Binärstring
                probabilities[state] = count / total_shots
            else:
                # Konvertiere den Binärstring in eine Dezimalzahl
                decimal_state = int(state, 2)
                probabilities[decimal_state] = count / total_shots
                
        return probabilities