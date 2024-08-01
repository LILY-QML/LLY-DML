import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    Image,
)
from reportlab.lib.styles import getSampleStyleSheet
from qiskit.visualization import circuit_drawer
import os

class Visual:
    def __init__(
        self,
        results,
        target_states,
        initial_training_phases,
        optimized_training_phases,
        activation_matrices,
        loss_data,
        circuits,
        num_iterations,
        qubits,
        depth,
    ):
        self.results = results
        self.target_states = target_states
        self.initial_training_phases = initial_training_phases
        self.optimized_training_phases = optimized_training_phases
        self.activation_matrices = activation_matrices
        self.loss_data = loss_data  # loss data for each activation matrix
        self.circuits = circuits  # list of circuits for each activation matrix
        self.num_iterations = num_iterations
        self.qubits = qubits
        self.depth = depth

    def generate_pdf_report(self, filename="QuantumCircuitReport.pdf"):
        pdf_generator = PDFGenerator(
            self.results,
            self.target_states,
            self.initial_training_phases,
            self.optimized_training_phases,
            self.activation_matrices,
            self.loss_data,
            self.circuits,
            self.num_iterations,
            self.qubits,
            self.depth,
        )
        pdf_generator.create_pdf(filename)

class PDFGenerator:
    def __init__(
        self,
        results,
        target_states,
        initial_training_phases,
        optimized_training_phases,
        activation_matrices,
        loss_data,
        circuits,
        num_iterations,
        qubits,
        depth,
    ):
        self.results = results
        self.target_states = target_states
        self.initial_training_phases = initial_training_phases
        self.optimized_training_phases = optimized_training_phases
        self.activation_matrices = activation_matrices
        self.loss_data = loss_data
        self.circuits = circuits
        self.num_iterations = num_iterations
        self.qubits = qubits
        self.depth = depth
        self.styles = getSampleStyleSheet()

    def create_pdf(self, filename):
        doc = SimpleDocTemplate(filename, pagesize=letter)
        flowables = []

        # Add the Banner and Header
        self._add_banner_and_header(flowables)

        # Add the Table of Contents
        self._add_table_of_contents(flowables)

        # Add the Introduction
        self._add_introduction(flowables)

        # Add Optimization Results
        self._add_results_table(flowables)

        # Add Quantum Circuits
        self._add_quantum_circuits(flowables)

        # Add Loss Function Graphs
        self._add_loss_function_graphs(flowables)

        # Add Conclusions
        self._add_conclusions(flowables)

        doc.build(flowables)

        # Clean up images
        self._cleanup_images()

    def _add_banner_and_header(self, flowables):
        """Add the banner and header information."""
        # Add Banner
        banner_text = "<font size=18><b>LLY-DML</b></font><br/>Copyright 2024 LILY Project"
        banner = Paragraph(banner_text, self.styles["Title"])
        flowables.append(banner)
        flowables.append(Spacer(1, 12))

        # Add Contact Information
        contact_info = "info@lilyqml.de<br/>Website: www.lilyqml.de"
        contact_paragraph = Paragraph(contact_info, self.styles["Normal"])
        flowables.append(contact_paragraph)
        flowables.append(Spacer(1, 24))

    def _add_table_of_contents(self, flowables):
        """Add the table of contents."""
        toc = Paragraph("Table of Contents", self.styles["Heading1"])
        flowables.append(toc)
        toc_content = [
            "1. Introduction",
            "2. Optimization Results",
            "3. Quantum Circuits",
            "4. Loss Function Graphs",
            "5. Conclusions",
        ]
        for item in toc_content:
            flowables.append(Paragraph(item, self.styles["BodyText"]))
            flowables.append(Spacer(1, 6))
        flowables.append(PageBreak())

    def _add_introduction(self, flowables):
        """Add an introduction section to the report."""
        intro_text = (
            "This report provides an analysis of quantum circuit optimizations performed using "
            "different activation matrices. Each matrix was tailored to maximize the probability "
            "of a specified target state. The results demonstrate the efficacy of the optimization process."
        )
        intro = Paragraph(intro_text, self.styles["BodyText"])
        flowables.append(Paragraph("1. Introduction", self.styles["Heading1"]))
        flowables.append(intro)
        flowables.append(Spacer(1, 12))
        flowables.append(PageBreak())

    def _add_results_table(self, flowables):
        """Add the optimization results table to the report."""
        flowables.append(Paragraph("2. Optimization Results", self.styles["Heading1"]))
        results_df = pd.DataFrame(self.results)

        # Add Experiment Details
        experiment_details = {
            "Number of Iterations": self.num_iterations,
            "Number of Qubits": self.qubits,
            "Circuit Depth": self.depth,
        }
        for key, value in experiment_details.items():
            paragraph = Paragraph(f"{key}: {value}", self.styles["BodyText"])
            flowables.append(paragraph)
            flowables.append(Spacer(1, 6))

        # Add Results Table
        results_table = self._create_results_table(results_df)
        flowables.append(results_table)
        flowables.append(Spacer(1, 12))
        flowables.append(PageBreak())

    def _add_quantum_circuits(self, flowables):
        """Add the quantum circuits section."""
        flowables.append(Paragraph("3. Quantum Circuits", self.styles["Heading1"]))
        for i, circuit in enumerate(self.circuits):
            circuit_image_path = os.path.join("var", f"circuit_{i+1}.png")
            circuit_drawer(
                circuit.circuit,
                output="mpl",
                filename=circuit_image_path,
                style={"backgroundcolor": "#EEEEEE"},
            )
            flowables.append(
                Paragraph(f"Quantum Circuit for Activation Matrix {i + 1}", self.styles["Heading2"])
            )
            flowables.append(Image(circuit_image_path, width=400, height=200))
            flowables.append(Spacer(1, 12))

        flowables.append(PageBreak())

    def _add_loss_function_graphs(self, flowables):
        """Add loss function graphs."""
        flowables.append(Paragraph("4. Loss Function Graphs", self.styles["Heading1"]))
        for i, loss_data in enumerate(self.loss_data):
            loss_graph_path = os.path.join("var", f"loss_graph_{i + 1}.png")
            self._plot_loss_function(
                loss_data,
                f"Loss Function for Activation Matrix {i + 1}",
                loss_graph_path,
            )
            flowables.append(
                Paragraph(f"Loss Function for Activation Matrix {i + 1}", self.styles["Heading2"])
            )
            flowables.append(Image(loss_graph_path, width=400, height=200))
            flowables.append(Spacer(1, 12))

        flowables.append(PageBreak())

    def _add_conclusions(self, flowables):
        """Add conclusions section."""
        flowables.append(Paragraph("5. Conclusions", self.styles["Heading1"]))
        overall_conclusion_text = (
            "The optimization process showed significant improvements in the probabilities of target states. "
            "The optimized training phases resulted in higher probabilities for the desired states, indicating "
            "the effectiveness of the chosen optimization strategy."
        )
        overall_conclusion = Paragraph(overall_conclusion_text, self.styles["BodyText"])
        flowables.append(overall_conclusion)
        flowables.append(Spacer(1, 12))

    def _create_results_table(self, results_df):
        """Helper method to create a table for the results."""
        data = [results_df.columns.tolist()] + results_df.round(4).values.tolist()
        table = Table(data)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                ]
            )
        )
        return table

    def _plot_loss_function(self, loss_data, title, filename):
        """Helper method to plot the loss function over iterations."""
        plt.figure(figsize=(10, 6))
        plt.plot(loss_data, marker="o", linestyle="-")
        plt.title(title)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.savefig(filename)
        plt.close()

    def _cleanup_images(self):
        """Helper method to remove generated image files after PDF creation."""
        for filename in os.listdir("var"):
            if filename.endswith(".png"):
                os.remove(os.path.join("var", filename))

class CircuitPlotter:
    def __init__(self, output_directory):
        self.output_directory = output_directory

    def plot_circuit(self, circuit, filename):
        circuit_path = os.path.join(self.output_directory, filename)
        circuit_drawer(circuit.circuit, output="mpl", filename=circuit_path, style={"backgroundcolor": "#EEEEEE"})
        return circuit_path

class LossPlotter:
    def __init__(self, output_directory):
        self.output_directory = output_directory

    def plot_loss_function(self, loss_data, title, filename):
        plt.figure(figsize=(10, 6))
        plt.plot(loss_data, marker="o", linestyle="-")
        plt.title(title)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.grid(True)
        loss_path = os.path.join(self.output_directory, filename)
        plt.savefig(loss_path)
        plt.close()
        return loss_path

class Utils:
    @staticmethod
    def create_results_table(results_df):
        data = [results_df.columns.tolist()] + results_df.round(4).values.tolist()
        table = Table(data)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                ]
            )
        )
        return table

    @staticmethod
    def cleanup_images(output_directory):
        for filename in os.listdir(output_directory):
            if filename.endswith(".png"):
                os.remove(os.path.join(output_directory, filename))
