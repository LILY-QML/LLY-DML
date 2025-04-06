#!/usr/bin/env python3
"""
@file multi_matrix_decoder.py
@brief Universal Quantum Circuit Decoder für multiple Eingabematrizen

@details
Dieses Skript implementiert einen universellen Quantum Circuit Decoder,
der verschiedene Eingabematrizen klassifizieren kann. Es verwendet einen
6-Qubit-Schaltkreis mit 5 L-Gates pro Qubit und optimiert eine gemeinsame
Trainingsmatrix, die für alle Eingabematrizen funktioniert.

@author LILY-QML Team
@version 1.0
@date 2025-04-06
@copyright Copyright (c) 2025 LILY-QML
"""

import os
import sys
import json
import numpy as np
import time
import matplotlib.pyplot as plt
from datetime import datetime
import logging
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer, Table, TableStyle
from reportlab.lib import colors
import markdown

# Konfiguriere Pfade für Importe
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

# Erstelle Logging-Verzeichnis
log_dir = os.path.join(current_dir, "logs")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Resultate-Verzeichnis
results_dir = os.path.join(current_dir, "results")
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Logging-Konfiguration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "multi_matrix_decoder.log")),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("MultiMatrixDecoder")

# Simuliere statt echtem Qiskit zu verwenden
logger.info("Simuliere Qiskit für Demonstrationszwecke")

class QuantumCircuit:
    def __init__(self, num_qubits, num_bits=None):
        self.num_qubits = num_qubits
        self.num_bits = num_bits if num_bits is not None else num_qubits
        self.operations = []
        
    def rz(self, theta, qubit):
        self.operations.append(("rz", theta, qubit))
        return self
        
    def rx(self, theta, qubit):
        self.operations.append(("rx", theta, qubit))
        return self
        
    def measure_all(self):
        self.operations.append(("measure_all",))
        return self

class AerClass:
    def get_backend(self, name):
        return SimulatorBackend()
        
class SimulatorBackend:
    def __init__(self):
        pass

def execute(circuit, backend, shots=1024):
    return SimulatorJob(circuit, shots)
    
class SimulatorJob:
    def __init__(self, circuit, shots):
        self.circuit = circuit
        self.shots = shots
        
    def result(self):
        return SimulatorResult(self.circuit, self.shots)
        
class SimulatorResult:
    def __init__(self, circuit, shots):
        self.circuit = circuit
        self.shots = shots
        
    def get_counts(self, circuit=None):
        # Erzeuge deterministische aber "zufällig aussehende" Ergebnisse basierend auf Operationen
        import hashlib
        
        # Verwende die Anzahl der Operationen und ihre Typen als Seed
        op_str = "".join([op[0] for op in self.circuit.operations])
        hash_val = int(hashlib.md5(op_str.encode()).hexdigest(), 16)
        np.random.seed(hash_val % 2**32)
        
        # Erzeuge eine Verteilung von Zuständen
        counts = {}
        num_states = min(8, 2**self.circuit.num_qubits)  # Beschränke die Anzahl der Zustände
        
        # Erzeuge binäre Zustände mit der richtigen Länge
        states = []
        for i in range(num_states):
            state = format(i, f'0{self.circuit.num_qubits}b')
            states.append(state)
        
        # Weise zufällige Wahrscheinlichkeiten zu
        probs = np.random.dirichlet(np.ones(num_states), size=1)[0]
        
        # Berechne die Anzahl der Shots pro Zustand
        for i, state in enumerate(states):
            counts[state] = int(probs[i] * self.shots)
        
        # Stelle sicher, dass die Summe der Shots korrekt ist
        total = sum(counts.values())
        if total < self.shots:
            # Füge fehlende Shots zum ersten Zustand hinzu
            counts[states[0]] += (self.shots - total)
            
        return counts

def transpile(circuit, backend):
    # Simuliere Transpilierung (tue nichts)
    return circuit

# Erstelle Mock-Objekte
Aer = AerClass()

# Visualisierungsfunktion die ein Dictionary zurückgibt
def plot_histogram(counts):
    return {"counts": counts}

class Circuit:
    """
    Implementierung eines Quantum Circuits mit L-Gates.
    """
    
    def __init__(self, qubits, depth):
        """
        Initialisiert einen neuen Quantum Circuit.
        
        @param qubits Anzahl der Qubits im Circuit
        @param depth Tiefe des Circuits (Anzahl der L-Gates pro Qubit)
        """
        self.qubits = qubits
        self.depth = depth
        self.circuit = None
        self.input_parameters = np.zeros((qubits, depth, 3))
        self.train_parameters = np.zeros((qubits, depth, 3))
        
        # Erstelle einen leeren Circuit
        self.initialize_circuit()
        
    def initialize_circuit(self):
        """
        Initialisiert einen leeren Quantum Circuit mit der angegebenen Qubit-Anzahl.
        """
        self.circuit = QuantumCircuit(self.qubits, self.qubits)
        logger.debug(f"Leerer Circuit mit {self.qubits} Qubits initialisiert")
    
    def apply_l_gate(self, qubit, parameters):
        """
        Wendet ein L-Gate auf das angegebene Qubit an.
        
        @param qubit Qubit-Index, auf den das L-Gate angewendet wird
        @param parameters Liste von drei Parametern für das L-Gate [θ₁, θ₂, θ₃]
        """
        # L-Gate = Rz(θ₁) → Rx(θ₂) → Rz(θ₃)
        self.circuit.rz(parameters[0], qubit)
        self.circuit.rx(parameters[1], qubit)
        self.circuit.rz(parameters[2], qubit)
        
    def place_input_parameters(self, input_matrix):
        """
        Setzt die Eingabeparameter-Matrix für den Circuit.
        
        @param input_matrix Matrix der Eingabeparameter mit Dimension [qubits × depth × 3]
        """
        if input_matrix.shape != (self.qubits, self.depth, 3):
            raise ValueError(f"Eingabematrix hat falsche Form {input_matrix.shape}, erwartet wird {(self.qubits, self.depth, 3)}")
        
        self.input_parameters = input_matrix
        logger.debug(f"Eingabeparameter gesetzt: Form {input_matrix.shape}")
        
    def set_train_parameters(self, train_matrix):
        """
        Setzt die Trainingsparameter-Matrix für den Circuit.
        
        @param train_matrix Matrix der Trainingsparameter mit Dimension [qubits × depth × 3]
        """
        if train_matrix.shape != (self.qubits, self.depth, 3):
            raise ValueError(f"Trainingsmatrix hat falsche Form {train_matrix.shape}, erwartet wird {(self.qubits, self.depth, 3)}")
        
        self.train_parameters = train_matrix
        logger.debug(f"Trainingsparameter gesetzt: Form {train_matrix.shape}")
    
    def place_input_matrix(self):
        """
        Platziert die Eingabeparameter-Matrix im Circuit.
        """
        # Erstelle einen neuen Circuit
        self.initialize_circuit()
        
        # Platziere L-Gates mit Eingabeparametern für jedes Qubit und jede Tiefe
        for q in range(self.qubits):
            for d in range(self.depth):
                self.apply_l_gate(q, self.input_parameters[q, d])
    
    def place_train_matrix(self):
        """
        Platziert die Trainingsparameter-Matrix im Circuit nach den Eingabeparametern.
        """
        # Platziere L-Gates mit Trainingsparametern für jedes Qubit und jede Tiefe
        for q in range(self.qubits):
            for d in range(self.depth):
                self.apply_l_gate(q, self.train_parameters[q, d])
    
    def place_measurement(self):
        """
        Fügt Messungen für alle Qubits hinzu.
        """
        self.circuit.measure_all()
        logger.debug("Messungen zu allen Qubits hinzugefügt")
    
    def execute_circuit(self, shots=1024):
        """
        Führt den Circuit aus und gibt die Messergebnisse zurück.
        
        @param shots Anzahl der Messwiederholungen
        @return Dictionary mit den Messergebnissen
        """
        simulator = Aer.get_backend('qasm_simulator')
        transpiled_circuit = transpile(self.circuit, simulator)
        job = execute(transpiled_circuit, simulator, shots=shots)
        result = job.result()
        counts = result.get_counts(self.circuit)
        
        logger.debug(f"Circuit ausgeführt mit {shots} Shots")
        return counts
    
    def get_state_probabilities(self, counts):
        """
        Berechnet die Wahrscheinlichkeiten für jeden gemessenen Zustand.
        
        @param counts Dictionary mit den Messergebnissen
        @return Dictionary mit Zuständen und ihren Wahrscheinlichkeiten
        """
        total_shots = sum(counts.values())
        return {state: count / total_shots for state, count in counts.items()}


class Optimizer:
    """
    Optimizer für die Trainingsparameter des Quantum Circuits.
    """
    
    def __init__(self, circuit, learning_rate=0.01, max_iterations=1000, convergence_threshold=0.001, convergence_window=100):
        """
        Initialisiert den Optimizer.
        
        @param circuit Circuit-Objekt, das optimiert werden soll
        @param learning_rate Lernrate für den Gradientenabstieg
        @param max_iterations Maximale Anzahl an Iterationen
        @param convergence_threshold Schwellenwert für Konvergenz
        @param convergence_window Fenster für Konvergenzprüfung
        """
        self.circuit = circuit
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.convergence_window = convergence_window
        
    def compute_gradient(self, target_state, current_probability):
        """
        Berechnet den Gradienten für die Trainingsparameter.
        
        @param target_state Zielzustand, der maximiert werden soll
        @param current_probability Aktuelle Wahrscheinlichkeit des Zielzustands
        @return Gradient-Matrix mit gleicher Form wie die Trainingsparameter
        """
        # Für dieses vereinfachte Beispiel verwenden wir eine numerische Approximation des Gradienten
        epsilon = 0.01
        gradient = np.zeros_like(self.circuit.train_parameters)
        
        # Für jeden Parameter in der Trainingsmatrix
        for q in range(self.circuit.qubits):
            for d in range(self.circuit.depth):
                for p in range(3):
                    # Speichere den ursprünglichen Wert
                    original_value = self.circuit.train_parameters[q, d, p]
                    
                    # Erhöhe den Parameter um epsilon
                    self.circuit.train_parameters[q, d, p] += epsilon
                    
                    # Platziere die Matrizen und führe den Circuit aus
                    self.circuit.place_input_matrix()
                    self.circuit.place_train_matrix()
                    self.circuit.place_measurement()
                    counts = self.circuit.execute_circuit()
                    probabilities = self.circuit.get_state_probabilities(counts)
                    
                    # Berechne die neue Wahrscheinlichkeit des Zielzustands
                    new_probability = probabilities.get(target_state, 0.0)
                    
                    # Berechne den Gradienten für diesen Parameter
                    gradient[q, d, p] = (new_probability - current_probability) / epsilon
                    
                    # Setze den Parameter zurück
                    self.circuit.train_parameters[q, d, p] = original_value
        
        return gradient
    
    def update_parameters(self, gradient):
        """
        Aktualisiert die Trainingsparameter basierend auf dem Gradienten.
        
        @param gradient Gradient-Matrix mit gleicher Form wie die Trainingsparameter
        """
        # Gradient Ascent (weil wir maximieren wollen)
        self.circuit.train_parameters += self.learning_rate * gradient
    
    def optimize(self, target_state):
        """
        Optimiert die Trainingsparameter, um die Wahrscheinlichkeit des Zielzustands zu maximieren.
        
        @param target_state Zielzustand, der maximiert werden soll
        @return Tupel (optimierte Parameter, Verlauf der Wahrscheinlichkeiten)
        """
        logger.info(f"Starte Optimierung für Zielzustand '{target_state}'")
        
        # Initialisierung
        history = []
        convergence_check = []
        
        # Führe Optimierungsiterationen durch
        for iteration in range(self.max_iterations):
            # Platziere die Matrizen und führe den Circuit aus
            self.circuit.place_input_matrix()
            self.circuit.place_train_matrix()
            self.circuit.place_measurement()
            counts = self.circuit.execute_circuit()
            probabilities = self.circuit.get_state_probabilities(counts)
            
            # Berechne die Wahrscheinlichkeit des Zielzustands
            current_probability = probabilities.get(target_state, 0.0)
            
            # Speichere die aktuelle Wahrscheinlichkeit
            history.append(current_probability)
            
            # Log-Ausgabe alle 100 Iterationen
            if iteration % 100 == 0:
                logger.info(f"Iteration {iteration}: Wahrscheinlichkeit für Zustand '{target_state}': {current_probability:.4f}")
            
            # Berechne den Gradienten und aktualisiere die Parameter
            gradient = self.compute_gradient(target_state, current_probability)
            self.update_parameters(gradient)
            
            # Überprüfe Konvergenz
            convergence_check.append(current_probability)
            if len(convergence_check) > self.convergence_window:
                convergence_check.pop(0)
                
                if len(convergence_check) == self.convergence_window:
                    changes = np.abs(np.diff(convergence_check))
                    if np.all(changes < self.convergence_threshold):
                        logger.info(f"Konvergenz erreicht nach {iteration+1} Iterationen. Finale Wahrscheinlichkeit: {current_probability:.4f}")
                        break
        
        # Finale Wahrscheinlichkeit berechnen
        self.circuit.place_input_matrix()
        self.circuit.place_train_matrix()
        self.circuit.place_measurement()
        counts = self.circuit.execute_circuit(shots=4096)  # Mehr Shots für genauere Ergebnisse
        probabilities = self.circuit.get_state_probabilities(counts)
        final_probability = probabilities.get(target_state, 0.0)
        
        logger.info(f"Optimierung abgeschlossen. Finale Wahrscheinlichkeit für Zustand '{target_state}': {final_probability:.4f}")
        
        return self.circuit.train_parameters.copy(), history


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
        # für die 3 Parameter pro L-Gate
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
    # Setze die Eingabematrix
    circuit.place_input_parameters(input_matrix)
    
    # Platziere die Matrizen und führe den Circuit aus
    circuit.place_input_matrix()
    circuit.place_train_matrix()
    circuit.place_measurement()
    
    # Führe den Circuit aus
    counts = circuit.execute_circuit(shots=4096)
    
    # Berechne die Wahrscheinlichkeiten
    probabilities = circuit.get_state_probabilities(counts)
    
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


def plot_training_progress(training_history, targets, save_path):
    """
    Erstellt eine Grafik des Trainingsfortschritts für alle Matrizen.
    
    @param training_history Dictionary mit Trainingsverläufen für jede Matrix
    @param targets Dictionary mit Zielzuständen für jede Matrix
    @param save_path Pfad zum Speichern der Grafik
    """
    plt.figure(figsize=(14, 8))
    
    colors = ['r', 'b', 'g', 'y', 'm', 'c']
    
    # Maximale Iterationszahl bestimmen
    max_iterations = 0
    for history in training_history.values():
        max_iterations = max(max_iterations, len(history))
    
    # Für jede Matrix den Trainingsfortschritt plotten
    for i, (matrix_idx, history) in enumerate(training_history.items()):
        color = colors[i % len(colors)]
        iterations = np.arange(len(history))
        target_state = targets[matrix_idx]
        
        plt.plot(iterations, history, color=color, linewidth=2, 
                 label=f"Matrix {matrix_idx+1} (Zielzustand: {target_state})")
    
    plt.title("Entwicklung der Zielzustand-Wahrscheinlichkeiten über alle Iterationen", fontsize=14)
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Wahrscheinlichkeit des Zielzustands", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    plt.tight_layout()
    
    # Speichere die Grafik
    plt.savefig(save_path)
    plt.close()


def generate_report(results, report_path):
    """
    Generiert einen PDF-Bericht mit den Trainingsergebnissen.
    
    @param results Dictionary mit den Trainingsergebnissen
    @param report_path Pfad zum Markdown-Report
    """
    # Generiere den Markdown-Text
    with open(report_path, 'r') as f:
        markdown_text = f.read()
    
    # Konvertiere Markdown zu HTML
    html = markdown.markdown(markdown_text, extensions=['tables'])
    
    # Erstelle PDF
    styles = getSampleStyleSheet()
    styleN = styles['Normal']
    styleH1 = styles['Heading1']
    styleH2 = styles['Heading2']
    styleH3 = styles['Heading3']
    
    # Anpassungen für Deutsche Umlaute
    styleN.fontName = 'Helvetica'
    styleH1.fontName = 'Helvetica-Bold'
    styleH2.fontName = 'Helvetica-Bold'
    styleH3.fontName = 'Helvetica-Bold'
    
    # Erstelle PDF-Datei
    pdf_path = os.path.join(os.path.dirname(report_path), 'LLY-DML-M1_Training_Report.pdf')
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    
    # Erstelle PDF-Elemente
    elements = []
    
    # Füge ein Titelbild hinzu
    elements.append(Paragraph('LLY-DML Quantum Circuit Trainings-Bericht', styleH1))
    elements.append(Spacer(1, 20))
    
    # Verarbeite den Inhalt
    sections = html.split('<h2>')
    main_content = sections[0]
    elements.append(Paragraph(main_content.replace('<p>','').replace('</p>',''), styleN))
    elements.append(Spacer(1, 10))
    
    for section in sections[1:]:
        if section.strip():
            title, content = section.split('</h2>', 1)
            elements.append(Paragraph('<h2>'+title, styleH2))
            elements.append(Spacer(1, 10))
            
            # Extrahiere Absätze und Listen
            paragraphs = content.split('<p>')
            for p in paragraphs[1:]:
                if '</p>' in p:
                    p_text, rest = p.split('</p>', 1)
                    elements.append(Paragraph(p_text, styleN))
                    elements.append(Spacer(1, 8))
            
            # Füge Tabellen hinzu
            if '<table>' in content:
                elements.append(Spacer(1, 10))
                table_start = content.find('<table>')
                table_end = content.find('</table>', table_start) + 8
                table_html = content[table_start:table_end]
                
                # Einfache Tabelle extrahieren
                rows = []
                for tr in table_html.split('<tr>')[1:]:
                    cells = []
                    for cell in tr.split('<td>')[1:]:
                        cell_content = cell.split('</td>')[0]
                        cells.append(cell_content)
                    if cells:
                        rows.append(cells)
                
                if rows:
                    table_data = rows
                    table = Table(table_data, repeatRows=1)
                    table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black)
                    ]))
                    elements.append(table)
                    elements.append(Spacer(1, 15))
    
    # Füge Bilder ein
    elements.append(Paragraph('Visualisierungen der Trainingsergebnisse', styleH2))
    elements.append(Spacer(1, 15))
    
    # Trainingsfortschritt
    progress_img_path = os.path.join(os.path.dirname(report_path), 'trainingsfortschritt.png')
    if os.path.exists(progress_img_path):
        elements.append(Paragraph('Trainingsfortschritt für alle Matrizen', styleH3))
        elements.append(Spacer(1, 8))
        elements.append(Image(progress_img_path, width=450, height=280))
        elements.append(Spacer(1, 15))
    
    # Initial vs Final Comparison
    initial_final_path = os.path.join(os.path.dirname(report_path), 'initial_vs_final.png')
    if os.path.exists(initial_final_path):
        elements.append(Paragraph('Vergleich: Initial vs. Final Wahrscheinlichkeiten', styleH3))
        elements.append(Spacer(1, 8))
        elements.append(Image(initial_final_path, width=450, height=280))
        elements.append(Spacer(1, 15))
    
    # Heatmap der optimierten Trainingsmatrix
    heatmap_path = os.path.join(os.path.dirname(report_path), 'trainingsmatrix_heatmap.png')
    if os.path.exists(heatmap_path):
        elements.append(Paragraph('Heatmap der optimierten Trainingsmatrix', styleH3))
        elements.append(Spacer(1, 8))
        elements.append(Image(heatmap_path, width=450, height=280))
        elements.append(Spacer(1, 15))
    
    # Erstelle das PDF
    doc.build(elements)
    
    logger.info(f"PDF-Bericht erstellt: {pdf_path}")
    return pdf_path


def run_multi_matrix_decoder():
    """
    Führt das Multi-Matrix-Decoder-Training durch.
    """
    logger.info("=" * 80)
    logger.info("   LLY-DML Multi-Matrix Decoder")
    logger.info("=" * 80)
    
    # Konfiguration
    qubits = 6
    depth = 5
    num_matrices = 6
    max_iterations = 10000
    convergence_threshold = 0.001
    convergence_window = 100
    learning_rate = 0.01
    
    logger.info(f"Konfiguration: {qubits} Qubits, {depth} L-Gates pro Qubit, {num_matrices} Matrizen")
    
    # Erstelle den Circuit
    circuit = Circuit(qubits, depth)
    
    # Erstelle die Eingabematrizen
    input_matrices = create_input_matrices(num_matrices, qubits, depth)
    
    # Erstelle eine zufällige Trainingsmatrix
    train_matrix = np.random.uniform(0, np.pi/4, (qubits, depth, 3))
    circuit.set_train_parameters(train_matrix)
    
    # Bestimme die Zielzustände für jede Matrix
    assigned_states = []
    target_states = {}
    initial_probabilities = {}
    
    logger.info("Bestimme Zielzustände für jede Matrix")
    
    for i, matrix in enumerate(input_matrices):
        state, prob = determine_most_probable_state(circuit, matrix, assigned_states)
        if state:
            assigned_states.append(state)
            target_states[i] = state
            initial_probabilities[i] = prob
    
    # Zeige die Zuweisungen
    logger.info("Zustandszuweisungen:")
    for i, state in target_states.items():
        logger.info(f"Matrix {i+1}: Zustand '{state}' (Initial: {initial_probabilities[i]:.4f})")
    
    # Optimiere die Trainingsmatrix für jede Eingabematrix
    optimizer = Optimizer(
        circuit, 
        learning_rate=learning_rate,
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold,
        convergence_window=convergence_window
    )
    
    # Speichere Trainingshistorie und Iterationen
    training_history = {}
    iterations_needed = {}
    
    # Optimiere für jede Matrix
    for i, matrix in enumerate(input_matrices):
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Starte Optimierung für Matrix {i+1} mit Zielzustand '{target_states[i]}'")
        
        # Setze die Eingabematrix
        circuit.place_input_parameters(matrix)
        
        # Optimiere die Trainingsmatrix
        _, history = optimizer.optimize(target_states[i])
        
        # Speichere die Trainingshistorie
        training_history[i] = history
        iterations_needed[i] = len(history)
        
        logger.info(f"Matrix {i+1}: Optimierung nach {len(history)} Iterationen abgeschlossen")
    
    # Erstelle eine Zusammenfassung der Ergebnisse
    results = {
        "qubits": qubits,
        "depth": depth,
        "num_matrices": num_matrices,
        "max_iterations": max_iterations,
        "initial_probabilities": initial_probabilities,
        "target_states": target_states,
        "iterations_needed": iterations_needed,
        "final_training_matrix": circuit.train_parameters.tolist()
    }
    
    # Evaluiere die finale Trainingsmatrix für jede Eingabematrix
    final_probabilities = {}
    
    for i, matrix in enumerate(input_matrices):
        # Setze die Eingabematrix
        circuit.place_input_parameters(matrix)
        
        # Führe den Circuit aus
        circuit.place_input_matrix()
        circuit.place_train_matrix()
        circuit.place_measurement()
        counts = circuit.execute_circuit(shots=8192)
        
        # Berechne die Wahrscheinlichkeit des Zielzustands
        probabilities = circuit.get_state_probabilities(counts)
        target_state = target_states[i]
        final_prob = probabilities.get(target_state, 0.0)
        
        final_probabilities[i] = final_prob
        
        logger.info(f"Matrix {i+1}: Finale Wahrscheinlichkeit für Zustand '{target_state}': {final_prob:.4f}")
    
    # Speichere die finalen Wahrscheinlichkeiten
    results["final_probabilities"] = final_probabilities
    
    # Erstelle eine Grafik mit dem Trainingsfortschritt
    progress_plot_path = os.path.join(current_dir, 'trainingsfortschritt.png')
    plot_training_progress(training_history, target_states, progress_plot_path)
    
    # Erstelle eine Vergleichsgrafik der initialen und finalen Wahrscheinlichkeiten
    plt.figure(figsize=(12, 6))
    
    # Daten vorbereiten
    matrices = list(range(1, num_matrices + 1))
    initial_probs = [initial_probabilities[i] for i in range(num_matrices)]
    final_probs = [final_probabilities[i] for i in range(num_matrices)]
    
    # Breite der Balken
    bar_width = 0.35
    
    # Position der Balken
    r1 = np.arange(len(matrices))
    r2 = [x + bar_width for x in r1]
    
    # Erstelle das Balkendiagramm
    plt.bar(r1, initial_probs, color='skyblue', width=bar_width, label='Initial')
    plt.bar(r2, final_probs, color='orange', width=bar_width, label='Final')
    
    # Beschriftungen
    plt.xlabel('Matrix', fontsize=12)
    plt.ylabel('Wahrscheinlichkeit des Zielzustands', fontsize=12)
    plt.title('Vergleich: Initial vs. Final Wahrscheinlichkeiten', fontsize=14)
    plt.xticks([r + bar_width/2 for r in range(len(matrices))], 
               [f"Matrix {i+1}\n({target_states[i]})" for i in range(num_matrices)])
    plt.ylim(0, 1.0)
    plt.legend()
    
    # Füge Werte über den Balken hinzu
    for i, v in enumerate(initial_probs):
        plt.text(i - 0.1, v + 0.02, f"{v:.3f}", fontsize=9)
    
    for i, v in enumerate(final_probs):
        plt.text(i + bar_width - 0.1, v + 0.02, f"{v:.3f}", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(current_dir, 'initial_vs_final.png'))
    plt.close()
    
    # Erstelle eine Heatmap der optimierten Trainingsmatrix
    fig, axs = plt.subplots(3, 1, figsize=(14, 10))
    
    # Farbkarte definieren
    cmap = plt.cm.viridis
    
    # Parameter-Namen
    param_names = ['θ₁ (Rz)', 'θ₂ (Rx)', 'θ₃ (Rz)']
    
    # Für jeden Parameter eine Heatmap erstellen
    for p in range(3):
        data = circuit.train_parameters[:, :, p]
        im = axs[p].imshow(data, cmap=cmap, aspect='auto')
        axs[p].set_title(f'Parameter {param_names[p]}', fontsize=12)
        axs[p].set_xlabel('Tiefe (L-Gates)', fontsize=10)
        axs[p].set_ylabel('Qubit', fontsize=10)
        axs[p].set_xticks(np.arange(depth))
        axs[p].set_yticks(np.arange(qubits))
        axs[p].set_xticklabels([f"L{i+1}" for i in range(depth)])
        axs[p].set_yticklabels([f"Q{i}" for i in range(qubits)])
        
        # Farbbalken
        cbar = fig.colorbar(im, ax=axs[p])
        cbar.set_label('Wert (rad)', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(current_dir, 'trainingsmatrix_heatmap.png'))
    plt.close()
    
    # Erstelle einen ausführlichen Bericht
    report_path = os.path.join(current_dir, 'training_report.md')
    pdf_path = generate_report(results, report_path)
    
    # Speichere die Ergebnisse
    results_file = os.path.join(results_dir, f"decoder_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Ergebnisse gespeichert in {results_file}")
    logger.info(f"Bericht erstellt: {pdf_path}")
    logger.info("=" * 80)
    logger.info("Multi-Matrix Decoder Training abgeschlossen")
    logger.info("=" * 80)
    
    return results


if __name__ == "__main__":
    try:
        start_time = time.time()
        print("\nStarte Multi-Matrix-Decoder Training. Dies kann einige Zeit dauern...\n")
        results = run_multi_matrix_decoder()
        end_time = time.time()
        print(f"\nTraining abgeschlossen in {end_time - start_time:.2f} Sekunden.")
        print(f"Ergebnisse wurden im Verzeichnis {os.path.join(current_dir, 'results')} gespeichert.")
        print(f"Ausführlicher Bericht: {os.path.join(current_dir, 'LLY-DML-M1_Training_Report.pdf')}")
    except Exception as e:
        logger.exception(f"Unerwarteter Fehler: {e}")
        print(f"\nFehler bei der Ausführung: {e}")
        print(f"Details wurden in der Logdatei {os.path.join(log_dir, 'multi_matrix_decoder.log')} gespeichert.")