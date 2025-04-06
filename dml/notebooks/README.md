# LLY-DML Tutorials

Dieses Verzeichnis enthält interaktive Jupyter Notebooks, die als Einführung und Anleitung zur Verwendung des LLY-DML Frameworks dienen.

## Verfügbare Tutorials

### 1. LLY_DML_Tutorial.ipynb
Ein umfassendes Tutorial, das den Umgang mit der Circuit-Klasse und dem Adam-Optimierer demonstriert. Es führt durch den Prozess der:
- Erstellung eines Quantum Circuits mit L-Gates
- Parametrisierung des Circuits
- Optimierung der Circuit-Parameter für verschiedene Eingabematrizen
- Visualisierung und Analyse der Ergebnisse

## Schnellstart

Um die Tutorials zu verwenden, stellen Sie sicher, dass Sie folgende Abhängigkeiten installiert haben:
```bash
pip install jupyter numpy matplotlib seaborn qiskit
```

Starten Sie dann den Jupyter Server im Projektverzeichnis:
```bash
jupyter notebook
```

Navigieren Sie im Browser zu diesem Verzeichnis und öffnen Sie das gewünschte Notebook.

## Hinweise

- Die Tutorials sind so konzipiert, dass alle Zellen nacheinander ausgeführt werden sollten.
- Ändern Sie die Parameter (wie Anzahl der Qubits, Tiefe, Iterationen) nach Belieben, um mit verschiedenen Konfigurationen zu experimentieren.
- Die Optimierung kann je nach Hardwareleistung einige Zeit in Anspruch nehmen.