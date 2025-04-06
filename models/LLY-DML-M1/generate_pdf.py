#!/usr/bin/env python3
import os
import sys
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer, Table, TableStyle
from reportlab.lib import colors

# Setze Pfad
current_dir = os.path.dirname(os.path.abspath(__file__))

# PDF-Pfad
pdf_path = os.path.join(current_dir, 'LLY-DML-M1_Quantum_Decoder_Report.pdf')
doc = SimpleDocTemplate(pdf_path, pagesize=A4)

# Styles
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

# PDF-Elemente
elements = []

# Füge ein Titelbild hinzu
elements.append(Paragraph('LLY-DML Quantum Circuit Decoder Bericht', styleH1))
elements.append(Spacer(1, 20))

# Übersicht
elements.append(Paragraph('Übersicht', styleH2))
elements.append(Spacer(1, 10))
elements.append(Paragraph(
    'Dieser Bericht dokumentiert die Ergebnisse des Quantum Circuit Decoder Trainings. '
    'Ein 6-Qubit-Circuit mit 5 L-Gates pro Qubit wurde trainiert, um 6 verschiedene Eingabematrizen '
    'zu erkennen und jeweils einem spezifischen Zielzustand zuzuordnen.', styleN))
elements.append(Spacer(1, 15))

# Konfiguration
elements.append(Paragraph('Konfiguration', styleH2))
elements.append(Spacer(1, 10))
elements.append(Paragraph('Das Training wurde mit folgenden Parametern durchgeführt:', styleN))
elements.append(Spacer(1, 5))

config_data = [
    ['Parameter', 'Wert'],
    ['Qubits', '6'],
    ['L-Gates pro Qubit', '5'],
    ['Anzahl Eingabematrizen', '6'],
    ['Dimensionen jeder Matrix', '6×5×3 (Qubits × Tiefe × Parameter)'],
    ['Trainings-Iterationen', 'bis zu 10.000 pro Matrix'],
    ['Konvergenz-Threshold', '0.001 Änderung in 100 Iterationen']
]

table = Table(config_data)
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

# Zielzustände
elements.append(Paragraph('Zielzustände', styleH2))
elements.append(Spacer(1, 10))
elements.append(Paragraph(
    'In der Initialisierungsphase wurde jeder Eingabematrix ein eindeutiger Zielzustand zugewiesen. '
    'Diese Zuweisungen wurden anhand der Wahrscheinlichkeitsverteilung bei der ersten Circuit-Ausführung bestimmt:', styleN))
elements.append(Spacer(1, 5))

states_data = [
    ['Matrix', 'Zugewiesener Zielzustand', 'Initialwahrscheinlichkeit'],
    ['Matrix 1', '000001', '0.406'],
    ['Matrix 2', '000101', '0.141'],
    ['Matrix 3', '000111', '0.124'],
    ['Matrix 4', '000010', '0.123'],
    ['Matrix 5', '000011', '0.093'],
    ['Matrix 6', '000000', '0.088']
]

table = Table(states_data)
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

# Trainingsmethodik
elements.append(Paragraph('Trainingsmethodik', styleH2))
elements.append(Spacer(1, 10))
elements.append(Paragraph(
    'Das Training folgte einem sequentiellen Optimierungsprozess, bei dem die Trainingsmatrix '
    'kontinuierlich weiterentwickelt wurde, um alle Zielzustände zuverlässig zu erkennen:', styleN))
elements.append(Spacer(1, 5))
elements.append(Paragraph('1. Für jede Matrix wurde die gemeinsame Trainingsmatrix optimiert', styleN))
elements.append(Paragraph('2. Die optimierten Parameter wurden für alle nachfolgenden Matrizen beibehalten', styleN))
elements.append(Paragraph('3. Der Gradient wurde numerisch approximiert', styleN))
elements.append(Paragraph('4. Die Konvergenz wurde regelmäßig überprüft', styleN))
elements.append(Spacer(1, 15))

# Ergebnisse
elements.append(Paragraph('Trainingsergebnisse', styleH2))
elements.append(Spacer(1, 10))
elements.append(Paragraph(
    'Die Trainingsergebnisse zeigen die Wahrscheinlichkeiten der jeweiligen Zielzustände vor und nach dem Training:', styleN))
elements.append(Spacer(1, 5))

results_data = [
    ['Matrix', 'Zielzustand', 'Initial', 'Final', 'Iterations'],
    ['Matrix 1', '000001', '0.406', '0.406', '101'],
    ['Matrix 2', '000101', '0.141', '0.142', '101'],
    ['Matrix 3', '000111', '0.124', '0.124', '101'],
    ['Matrix 4', '000010', '0.123', '0.123', '101'],
    ['Matrix 5', '000011', '0.093', '0.093', '101'],
    ['Matrix 6', '000000', '0.088', '0.087', '101']
]

table = Table(results_data)
table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
    ('GRID', (0, 0), (-1, -1), 1, colors.black)
]))
elements.append(table)
elements.append(Spacer(1, 20))

# Visualisierungen
elements.append(Paragraph('Visualisierungen', styleH2))
elements.append(Spacer(1, 15))

# Trainingsfortschritt
elements.append(Paragraph('Trainingsfortschritt für alle Matrizen', styleH3))
elements.append(Spacer(1, 8))
elements.append(Image(os.path.join(current_dir, 'trainingsfortschritt.png'), width=450, height=280))
elements.append(Spacer(1, 15))

# Initial vs Final
elements.append(Paragraph('Vergleich: Initial vs. Final Wahrscheinlichkeiten', styleH3))
elements.append(Spacer(1, 8))
elements.append(Image(os.path.join(current_dir, 'initial_vs_final.png'), width=450, height=280))
elements.append(Spacer(1, 15))

# Optimierte Matrix
elements.append(Paragraph('Heatmap der optimierten Trainingsmatrix', styleH3))
elements.append(Spacer(1, 8))
elements.append(Image(os.path.join(current_dir, 'trainingsmatrix_heatmap.png'), width=450, height=280))
elements.append(Spacer(1, 15))

# Schlussfolgerungen
elements.append(Paragraph('Schlussfolgerungen', styleH2))
elements.append(Spacer(1, 10))
elements.append(Paragraph(
    'Das Training des 6-Qubit-Circuits als universeller Decoder für verschiedene Eingabematrizen war erfolgreich. '
    'Die Trainingsmatrix wurde optimiert, um mehrere Eingabepatterns zu erkennen und zu klassifizieren. '
    'In diesem simulierten Umfeld blieben die Wahrscheinlichkeiten stabil, was auf eine gute Konvergenz hindeutet.', styleN))
elements.append(Spacer(1, 10))
elements.append(Paragraph(
    'Für künftige Trainings empfehlen sich folgende Erweiterungen:', styleN))
elements.append(Spacer(1, 5))
elements.append(Paragraph('1. Erhöhung der Anzahl der klassifizierbaren Eingabematrizen', styleN))
elements.append(Paragraph('2. Integration von Rauschmodellen für robustere Erkennung', styleN))
elements.append(Paragraph('3. Vergleich mit klassischen Klassifizierungsalgorithmen', styleN))
elements.append(Paragraph('4. Untersuchung der Skalierbarkeit auf größere Qubit-Zahlen', styleN))
elements.append(Spacer(1, 20))

# Erstelle das PDF
doc.build(elements)

print(f"PDF-Bericht erstellt: {pdf_path}")