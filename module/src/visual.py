# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Project: LILY-QML
# Version: 2.0.0 LLY-DML
# Author: Leon Kaiser
# Contact: info@lilyqml.de
# Website: www.lilyqml.de
# Contributors:
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++

# visual.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak,
    Image, Table, TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from datetime import datetime
import os
import seaborn as sns
from io import BytesIO
from qiskit.visualization import circuit_drawer
from qiskit import QuantumCircuit

# ------------------------------
# Section Classes
# ------------------------------

class TitlePage:
    def __init__(self, title, subtitle, copyright_info,
                 description, date, additional_info):
        self.title = title
        self.subtitle = subtitle
        self.copyright_info = copyright_info
        self.description = description
        self.date = date
        self.additional_info = additional_info

    def build(self, story, styles):
        # Define custom styles
        title_style = ParagraphStyle(
            'title',
            parent=styles['Title'],
            fontSize=36,
            textColor=colors.HexColor("#000080"),
            spaceAfter=24
        )
        subtitle_style = ParagraphStyle(
            'subtitle',
            parent=styles['Heading2'],
            fontSize=16,
            spaceAfter=5
        )
        normal_style = ParagraphStyle(
            'normal',
            parent=styles['Normal'],
            alignment=4,  # Justified
            spaceBefore=10
        )

        # Create Paragraphs
        title_paragraph = Paragraph(self.title, title_style)
        subtitle_paragraph = Paragraph(self.subtitle, subtitle_style)
        copyright_paragraph = Paragraph(
            self.copyright_info, normal_style
        )
        description_paragraph = Paragraph(
            self.description, normal_style
        )
        additional_info_paragraph = Paragraph(
            self.additional_info, normal_style
        )
        date_paragraph = Paragraph(
            f"""<hr/>
            This report was generated on: <b>{self.date}</b>
            <hr/>""",
            normal_style
        )

        # Add elements to story
        story.extend([
            title_paragraph,
            subtitle_paragraph,
            Spacer(1, 20),
            copyright_paragraph,
            Spacer(1, 20),
            description_paragraph,
            Spacer(1, 40),
            additional_info_paragraph,
            Spacer(1, 40),
            date_paragraph,
            PageBreak()
        ])

class TableOfContents:
    def __init__(self, contents):
        self.contents = contents

    def build(self, story, styles):
        toc_title_style = styles['Heading2']
        toc_title = Paragraph("Table of Contents", toc_title_style)

        toc_entries = [toc_title]
        toc_style = ParagraphStyle(
            'toc',
            parent=styles['Normal'],
            spaceBefore=5,
            spaceAfter=5,
            leftIndent=20,
            fontSize=12
        )
        
        for entry in self.contents:
            toc_entry = Paragraph(entry, toc_style)
            toc_entries.append(toc_entry)

        story.extend(toc_entries + [Spacer(1, 40), PageBreak()])

class TitlePageSection:
    def __init__(self, styles):
        self.styles = styles

    def add_title_page(self, story):
        title = "LLY-DML"
        subtitle = "Part of the LILY Project"
        copyright_info = """All rights reserved.<br/>
        Contact: <a href="mailto:info@lilyqml.de">info@lilyqml.de</a><br/>
        Website: <a href="http://lilyqml.de">lilyqml.de</a>"""
        description = """<hr/>
        This is LLY-DML, a model from the LILY Quantum Machine Learning project.<br/>
        The goal is to train datasets using L-gates, quantum machine learning gates.<br/>
        <hr/>"""
        date = datetime.now().strftime("%d.%m.%Y")
        additional_info = f"""<b>Date:</b> {date}<br/>
        <b>Author:</b> LILY Team<br/>
        <b>Version:</b> 1.0<br/>
        <b>Contact:</b> info@lilyqml.de<br/>
        <b>Website:</b> <a href="http://lilyqml.de">lilyqml.de</a><br/>"""

        title_page = TitlePage(
            title, subtitle, copyright_info,
            description, date, additional_info
        )
        title_page.build(story, self.styles)

    def add_table_of_contents(self, story):
        toc = TableOfContents(contents=[
            "<link href='#section1' color='blue'>1. Initialized Data</link>",
            "<link href='#section2' color='blue'>2. Optimization Methods</link>",
            "<link href='#section3' color='blue'>3. Comparison of Methods</link>",
            "<link href='#section4' color='blue'>4. Quantum Circuit</link>",
            "<link href='#section5' color='blue'>5. Final Results</link>",
        ])
        toc.build(story, self.styles)

class OptimizationMethod:
    def __init__(self, title, description, has_parameters=True):
        self.title = title
        self.description = description
        self.has_parameters = has_parameters

    def build(self, story, styles, optimizer_name, additional_data):
        # Titel und Beschreibung des Optimierers hinzufügen
        method_title = Paragraph(f"{self.title}", styles['Heading3'])
        method_description = Paragraph(f"<b>Description:</b> {self.description}", styles['Normal'])

        story.extend([
            method_title, Spacer(1, 10),
            method_description, Spacer(1, 10),
        ])

        if self.has_parameters:
            # Definiere, welche Parameter für jeden Optimierer relevant sind
            optimizer_parameters_map = {
                "Basic Gradient Descent": [],
                "Momentum": ["learning_rate", "shots", "max_iterations"],
                "Adam": ["learning_rate", "shots", "max_iterations"],
                "Genetic Algorithm": ["population_size", "mutation_rate"],
                "Particle Swarm Optimization": ["num_particles", "inertia", "cognitive", "social"],
                "Bayesian Optimization": [],
                "Simulated Annealing": ["initial_temperature", "cooling_rate"],
                "Quantum Natural Gradient": []
            }

            params = optimizer_parameters_map.get(optimizer_name, [])

            if params:
                data = [["Parameter", "Value"]]
                for param in params:
                    # Convert snake_case to Title Case with spaces
                    param_title = param.replace("_", " ").title()
                    value = additional_data.get(param, "N/A")
                    data.append([param_title, value])

                table = Table(data, colWidths=[270, 270])  # 540 / 2 = 270
                table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("FONTSIZE", (0, 0), (-1, -1), 10),
                            ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                        ]
                    )
                )
                story.append(table)
                story.append(Spacer(1, 20))
            else:
                story.append(Paragraph("No parameters defined for this optimizer.", styles['Normal']))
                story.append(Spacer(1, 20))
        else:
            story.append(Paragraph("No parameters defined for this optimizer.", styles['Normal']))
            story.append(Spacer(1, 20))

class InitializedDataSection:
    def __init__(self, data, styles):
        self.data = data  # A dictionary containing all necessary data
        self.styles = styles

    def add_initiated_data(self, story):
        # Add "Initialized Data" heading
        story.append(Paragraph("<a name='section1'/>1. Initialized Data", self.styles['Heading2']))
        story.append(Spacer(1, 20))

        # Subsections: Input Data, Training Data, Quantum Circuit, Additional Parameters
        story.append(Paragraph("1.1 Input Data", self.styles['Heading3']))
        story.append(Spacer(1, 10))
        self.add_input_data_table(story)
        story.append(Spacer(1, 20))

        story.append(Paragraph("1.2 Training Data", self.styles['Heading3']))
        story.append(Spacer(1, 10))
        self.add_training_data_table(story)
        story.append(Spacer(1, 20))

        story.append(Paragraph("1.3 Quantum Circuit", self.styles['Heading3']))
        story.append(Spacer(1, 10))
        self.add_quantum_circuit_table(story)
        story.append(Spacer(1, 20))

        story.append(Paragraph("1.4 Additional Parameters", self.styles['Heading3']))
        story.append(Spacer(1, 10))
        self.add_additional_parameters_table(story)
        story.append(Spacer(1, 20))

        # No PageBreak here to keep all subsections on the same page

    def add_input_data_table(self, story):
        # Input Data Table with columns: Number of Input Matrices, Columns, Rows
        data = [
            ["Number of Input Matrices", "Columns", "Rows"],
            [
                self.data.get("num_input_matrices", 0),
                self.data.get("qubits", 0),
                self.data.get("depth", 0)
            ]
        ]
        table = Table(data, colWidths=[180, 180, 180])  # 540 / 3 = 180
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ]
            )
        )
        story.append(table)

    def add_training_data_table(self, story):
        # Training Data Table with columns: Columns, Rows
        data = [
            ["Columns", "Rows"],
            [
                self.data.get("qubits", 0),
                self.data.get("depth", 0)
            ]
        ]
        table = Table(data, colWidths=[270, 270])  # 540 / 2 = 270
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ]
            )
        )
        story.append(table)

    def add_quantum_circuit_table(self, story):
        # Quantum Circuit Table with columns: Qubits, Depth, Shots, Entanglements
        entanglements = ", ".join(self.data.get("entanglements", [])) if self.data.get("entanglements") else "None"
        data = [
            ["Qubits", "Depth", "Shots", "Entanglements"],
            [
                self.data.get("qubits", 0),
                self.data.get("depth", 0),
                self.data.get("shots", 0),
                entanglements
            ]
        ]
        table = Table(data, colWidths=[135, 135, 135, 135])  # 540 / 4 = 135
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ]
            )
        )
        story.append(table)

    def add_additional_parameters_table(self, story):
        # Additional Parameters Table with relevant parameters
        data = [
            ["Parameter", "Value"],
            ["Learning Rate", self.data.get("learning_rate", 0)],
            ["Shots", self.data.get("shots", 0)],
            ["Max Iterations", self.data.get("max_iterations", 0)],
            ["Population Size", self.data.get("population_size", 0)],
            ["Mutation Rate", self.data.get("mutation_rate", 0)],
            ["Number of Particles", self.data.get("num_particles", 0)],
            ["Inertia", self.data.get("inertia", 0)],
            ["Cognitive", self.data.get("cognitive", 0)],
            ["Social", self.data.get("social", 0)],
            ["Initial Temperature", self.data.get("initial_temperature", 0)],
            ["Cooling Rate", self.data.get("cooling_rate", 0)],
            ["Optimizers", ", ".join(self.data.get("optimizers", []))],
            ["Activation Matrices", ", ".join(map(str, self.data.get("activation_matrices", [])))]
        ]
        table = Table(data, colWidths=[270, 270])  # 540 / 2 = 270
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ]
            )
        )
        story.append(table)

class QuantumCircuitSection:
    def __init__(self, circuits, styles):
        self.circuits = circuits
        self.styles = styles

    def add_quantum_circuit(self, story):
        story.append(Paragraph("<a name='section4'/>4. Quantum Circuit", self.styles['Heading2']))
        story.append(Spacer(1, 20))

        if self.circuits and self.circuits[0] is not None:
            # Use the actual QuantumCircuit object
            circuit_diagram = circuit_drawer(self.circuits[0], output='mpl')
            buf = BytesIO()
            circuit_diagram.savefig(buf, format='png', bbox_inches='tight')
            plt.close()
            buf.seek(0)
            story.append(Image(buf, width=468, height=300))  # Adjusted to match page width
        else:
            story.append(Paragraph("No quantum circuit available.", self.styles['Normal']))
        story.append(Spacer(1, 20))
        story.append(PageBreak())

class OptimizationMethodsSection:
    def __init__(self, results, initial_training_phases, styles, additional_data):
        self.results = results
        self.initial_training_phases = initial_training_phases
        self.styles = styles
        self.additional_data = additional_data

    def add_optimization_methods(self, story):
        # Starten Sie den Abschnitt "2. Optimization Methods"
        story.append(Paragraph("<a name='section2'/>2. Optimization Methods", self.styles['Heading2']))
        story.append(Spacer(1, 20))
        # Kein PageBreak hier, damit der Abschnitt auf derselben Seite beginnt

        optimization_methods = [
            OptimizationMethod(
                title="Basic Gradient Descent (GD)",
                description="A simple optimization algorithm that updates parameters in the opposite direction of the gradient."
            ),
            OptimizationMethod(
                title="Momentum",
                description="An extension of basic gradient descent that accelerates convergence by considering past update directions."
            ),
            OptimizationMethod(
                title="Adam (Adaptive Moment Estimation)",
                description="Combines the benefits of RMSProp and Momentum, adapts learning rates for each parameter, and maintains moving averages of gradients and their squares."
            ),
            OptimizationMethod(
                title="Genetic Algorithm (GA)",
                description="Inspired by natural selection, this algorithm uses operations like mutation, crossover, and selection to evolve solutions over generations."
            ),
            OptimizationMethod(
                title="Particle Swarm Optimization (PSO)",
                description="A population-based optimization algorithm that simulates social behavior, where particles adjust their positions based on their own and neighbors' experiences."
            ),
            OptimizationMethod(
                title="Bayesian Optimization",
                description="Uses a probabilistic model to estimate the objective function and focuses on areas with high probability of finding the minimum."
            ),
            OptimizationMethod(
                title="Simulated Annealing",
                description="Mimics the annealing process in metallurgy, gradually reducing 'temperature' to escape local minima and find a global minimum."
            ),
            OptimizationMethod(
                title="Quantum Natural Gradient (QNG)",
                description="A quantum-aware optimization technique that considers the geometric properties of the parameter space, often leading to better convergence in quantum circuit optimization."
            ),
        ]

        # Erstellen einer Zuordnung von Optimierern zu ihren Ergebnissen
        optimizer_results_map = {}
        for result in self.results:
            optimizer = result['Optimizer']
            if optimizer not in optimizer_results_map:
                optimizer_results_map[optimizer] = []
            optimizer_results_map[optimizer].append(result)

        # Iterieren über die Optimierungsmethoden
        for method in optimization_methods:
            # Extrahiere den Optimiernamen aus dem Titel (vor den Klammern)
            optimizer_name = method.title.split(' (')[0]

            # Rufe die build Methode mit allen erforderlichen Argumenten auf
            method.build(story, self.styles, optimizer_name, self.additional_data)

            if optimizer_name in optimizer_results_map and optimizer_results_map[optimizer_name]:
                optimizer_results = optimizer_results_map[optimizer_name]

                for idx, result in enumerate(optimizer_results):
                    story.append(Paragraph(f"Matrix {result['Activation Matrix']}", self.styles['Heading4']))
                    story.append(Spacer(1, 10))

                    # Heatmap vor dem Training
                    initial_phases = np.array(self.initial_training_phases[idx])
                    self.add_heatmap(story, initial_phases, f"Initial Training Phases for Matrix {result['Activation Matrix']}")

                    # Anfangsmatrix
                    self.add_matrix_table(story, initial_phases, f"Initial Phases Matrix for Matrix {result['Activation Matrix']}")

                    # Optimierte Heatmap
                    optimized_phases = np.array(result['Optimized Phases'])
                    self.add_heatmap(story, optimized_phases, f"Optimized Training Phases for Matrix {result['Activation Matrix']}")

                    # Optimierte Matrix
                    self.add_matrix_table(story, optimized_phases, f"Optimized Phases Matrix for Matrix {result['Activation Matrix']}")

                    # Ergebnistabelle
                    self.add_results_table(story, result, result['Activation Matrix'])

                    story.append(Spacer(1, 20))
            else:
                # Spezielle Behandlung für Optimierer ohne definierte Ergebnisse
                story.append(Paragraph("No results for this optimizer.", self.styles['Normal']))
                story.append(Spacer(1, 20))

            # Seitenumbruch nach jedem Optimierer, außer nach dem letzten
            story.append(PageBreak())

    # Hilfsmethoden zur Erstellung von Heatmaps und Tabellen
    def add_heatmap(self, story, phases, title):
        # Heatmap der Phasen erstellen
        plt.figure(figsize=(8, 6))
        sns.heatmap(phases, cmap='viridis', annot=True, fmt=".2f")
        plt.title(title)
        plt.xlabel('Qubits')
        plt.ylabel('Phases')

        # Speichern als Bild im Speicher
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)

        # Bild zur Geschichte hinzufügen
        story.append(Image(buf, width=468, height=300))  # Angepasst an die Seitenbreite
        story.append(Spacer(1, 10))

    def add_matrix_table(self, story, phases, title):
        # Tabelle aus der Phasenmatrix erstellen
        data = [[f"{value:.2f}" for value in row] for row in phases.tolist()]
        num_cols = len(data[0])
        total_width = 540  # Seitenbreite minus Ränder
        col_width = total_width / num_cols
        col_widths = [col_width] * num_cols
        table = Table(data, colWidths=col_widths)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                ]
            )
        )
        story.append(Paragraph(title, self.styles['Heading5']))
        story.append(table)
        story.append(Spacer(1, 10))

    def add_results_table(self, story, result, matrix_idx):
        # Tabelle mit den Ergebnissen erstellen
        counts = result['Final Counts']
        total_counts = sum(counts.values())
        target_state_prob = counts.get(result['Target State'], 0) / total_counts

        # Bestimmen des nächsten States
        sorted_states = sorted(counts.items(), key=lambda item: item[1], reverse=True)
        next_state = None
        for state, count in sorted_states:
            if state != result['Target State'] and state != 'shots':
                next_state = state
                break
        next_state_prob = counts.get(next_state, 0) / total_counts if next_state else 0

        data = [
            ["Matrix", "Target State", "Probability", "Next State Probability"],
            [f"Matrix {matrix_idx}", result['Target State'], f"{target_state_prob:.4f}", f"{next_state_prob:.4f}"]
        ]

        col_widths = [135, 135, 135, 135]  # 540 / 4 = 135
        table = Table(data, colWidths=col_widths)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ]
            )
        )
        story.append(table)
        story.append(Spacer(1, 20))

class ComparisonOfMethodsSection:
    def __init__(self, results, styles):
        self.results = results
        self.styles = styles

    def add_comparison_section(self, story):
        story.append(Paragraph("<a name='section3'/>3. Comparison of Methods", self.styles['Heading2']))
        story.append(Spacer(1, 20))

        # Get unique activation matrices
        matrices = sorted(set([res['Activation Matrix'] for res in self.results]))
        for matrix_idx in matrices:
            story.append(Paragraph(f"Matrix {matrix_idx}", self.styles['Heading3']))
            story.append(Spacer(1, 10))

            # Get results for the current matrix
            matrix_results = [res for res in self.results if res['Activation Matrix'] == matrix_idx]
            for res in matrix_results:
                improvement = res['Final Probability'] - res['Initial Probability']
                comparison_text = (
                    f"<b>Optimizer:</b> {res['Optimizer']}<br/>"
                    f"<b>Target State:</b> {res['Target State']}<br/>"
                    f"<b>Initial Probability:</b> {res['Initial Probability']:.4f}<br/>"
                    f"<b>Final Probability:</b> {res['Final Probability']:.4f}<br/>"
                    f"<b>Improvement:</b> {improvement:.4f}<br/><br/>"
                )
                story.append(Paragraph(comparison_text, self.styles['Normal']))
                story.append(Spacer(1, 10))

        # Summary table of final probabilities
        story.append(Paragraph(
            "Summary of Final Target State Probabilities for Each Matrix and Optimizer:", self.styles['Heading3']
        ))
        # Get sorted list of optimizers for consistent column ordering
        optimizers = sorted(list(set([res['Optimizer'] for res in self.results])))
        data = [["Matrix"] + optimizers]
        for matrix_idx in matrices:
            row = [f"Matrix {matrix_idx}"]
            for optimizer in optimizers:
                # Find the result for this matrix and optimizer
                matching_res = next((res for res in self.results if res['Activation Matrix'] == matrix_idx and res['Optimizer'] == optimizer), None)
                if matching_res:
                    row.append(f"{matching_res['Final Probability']:.4f}")
                else:
                    row.append("-")
            data.append(row)

        # Define column widths
        num_cols = len(data[0])
        total_width = 540  # Page width minus margins
        col_width = total_width / num_cols
        col_widths = [col_width] * num_cols

        table = Table(data, colWidths=col_widths)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                ]
            )
        )
        story.append(table)
        story.append(Spacer(1, 20))
        story.append(PageBreak())

class FinalResultsSection:
    def __init__(self, results, styles):
        self.results = results
        self.styles = styles

    def add_final_results_section(self, story):
        story.append(Paragraph("<a name='section5'/>5. Final Results", self.styles['Heading2']))
        story.append(Spacer(1, 20))

        final_df = pd.DataFrame(self.results)
        final_df['Improvement'] = final_df['Final Probability'] - final_df['Initial Probability']
        best_optimizer = final_df.loc[final_df['Improvement'].idxmax(), 'Optimizer']

        final_results_content = (
            f"The most effective optimization method was <b>{best_optimizer}</b>, "
            "achieving the highest improvement in target state probability."
        )

        story.append(Paragraph(final_results_content, self.styles['Normal']))
        story.append(Spacer(1, 20))
        story.append(PageBreak())

# ------------------------------
# Main Visual Class
# ------------------------------

class Visual:
    def __init__(
        self,
        results,
        target_states,
        initial_training_phases,
        activation_matrices,
        circuits,
        num_iterations,
        qubits,
        depth,
        additional_data=None,
        optimized_training_phases=None
    ):
        self.results = results
        self.target_states = target_states
        self.initial_training_phases = initial_training_phases
        self.activation_matrices = activation_matrices
        self.circuits = circuits
        self.num_iterations = num_iterations
        self.qubits = qubits
        self.depth = depth
        self.optimized_training_phases = optimized_training_phases
        self.additional_data = additional_data if additional_data else {}
        self.styles = getSampleStyleSheet()

        # Initialize section handlers
        self.title_section = TitlePageSection(self.styles)
        self.initial_data_section = InitializedDataSection(
            self.additional_data, self.styles
        )
        self.circuit_section = QuantumCircuitSection(circuits, self.styles)
        self.optimization_section = OptimizationMethodsSection(
            results, initial_training_phases, self.styles, self.additional_data
        )
        self.comparison_section = ComparisonOfMethodsSection(
            results, self.styles
        )
        self.final_results_section = FinalResultsSection(
            results, self.styles
        )

    def generate_report(self, filename="QuantumCircuitReport.pdf"):
        # Create the document with specified margins
        doc = SimpleDocTemplate(filename, pagesize=letter,
                                rightMargin=36, leftMargin=36,
                                topMargin=36, bottomMargin=36)
        story = []

        # Add sections to the story
        self.title_section.add_title_page(story)
        self.title_section.add_table_of_contents(story)
        self.initial_data_section.add_initiated_data(story)
        self.optimization_section.add_optimization_methods(story)
        self.comparison_section.add_comparison_section(story)
        self.circuit_section.add_quantum_circuit(story)
        self.final_results_section.add_final_results_section(story)

        # Build the PDF
        doc.build(story)
