# LLY-DML: Differentiable Machine Learning

## Projekt-Überblick
LLY-DML ist eine Kernkomponente des LILY-Projekts, die auf die Entwicklung und Optimierung von Quantenschaltkreisen mit Techniken des differenzierbaren maschinellen Lernens fokussiert ist. Das Projekt ermöglicht Forschern und Entwicklern, mit quantenbasierten Modellen in einer benutzerfreundlichen Umgebung zu experimentieren.

### Kernfunktionalitäten
- **Quantenschaltkreis-Optimierung**: Erstellen und Optimieren von Quantenalgorithmen durch differenzierbaren Gradientenabstieg
- **Multiple Optimizer**: Verschiedene Optimierungsalgorithmen (Adam, SGD, RMSProp, etc.) für unterschiedliche Trainingsszenarien
- **Crosstraining**: Training von mehreren Aktivierungsmatrizen mit Zufallsauswahl
- **Berichtgenerierung**: Automatische Erstellung von PDF-Berichten mit Trainings- und Testergebnissen

### Status-Übersicht
- **Implementiert**: Basisarchitektur, Quantenschaltkreise, Optimierungs-Frameworks, Datenverwaltung, Basisreporting
- **In Entwicklung**: Erweiterte Visualisierungen, Test-Segment, Überprüfung und Verbesserung des Extractor-Moduls
- **Geplant**: Umfangreiche Berichterstellung, Crosstraining-Verbesserungen, Benutzeroberfläche-Erweiterungen

## Architektur
LLY-DML folgt einer modularen Architektur, die Quantenberechnungen, differenzierbare Optimierung und Datenverwaltung integriert.

### Klassendiagramm
```
+-------------------+     +-----------------+     +----------------+
|       DML         |     |      Start      |     |     Reader     |
+-------------------+     +-----------------+     +----------------+
| - console         |     | - reader        |     | - working_dir  |
+-------------------+     | - train_json_ex |     +----------------+
| + start()         |     +-----------------+     | + fileCheck()  |
| + train_all_opt() |     | + run()         |     | + checkLog()   |
| + train_spec_opt()|     | + setupDynamic()|     | + dataConsis() |
| + start_testing() |     +-----------------+     | + validateLG() |
| + create_reports()|           |                 | + moveJson()   |
+-------------------+           |                 +----------------+
        |                       |                         |
        |                       |                         |
        v                       v                         v
+-------------------+     +-----------------+     +----------------+
|      Console      |     |       All       |     |    Circuit     |
+-------------------+     +-----------------+     +----------------+
| + run()           |     | - reader        |     | - qubits       |
+-------------------+     | - working_dir   |     | - depth        |
                          +-----------------+     | - train_phases |
                          | + optimize()    |     | - activ_phases |
                          | + crosstraining()|    | - circuit      |
                          | + crosshelper() |     +----------------+
                          | + data_ready()  |     | + initialize() |
                          | + Precheck()    |     | + apply_l_gate()|
                          +-----------------+     | + measure()    |
                                |                 | + run()        |
                                |                 +----------------+
                                v                         |
+-------------------+     +-----------------+             |
|   BaseOptimizer   |     |    Visual      |<------------+
+-------------------+     +-----------------+
| - data            |     | + generate_pdf()|
| - training_matrix |     | + overview()    |
| - target_state    |     | + training()    |
| - learning_rate   |     | + testing()     |
+-------------------+     +-----------------+
| + evaluate()      |
| + calculate_loss()|
| + compute_grad()  |
| + optimize()      |
+-------------------+
        ^
        |
+-------+-------+-------+-------+-------+
|               |       |       |       |
v               v       v       v       v
+-------+   +-------+   +-----+   +------+   +------+
| Adam  |   |  SGD  |   | RMS |   | Ada  |   | Nadam|
+-------+   +-------+   +-----+   +------+   +------+
```

## Workflows

### Trainingsprozess
1. **Initialisierung**: Start-Klasse überprüft erforderliche Dateien und Konfigurationen
2. **Datenvorbereitungen**: Reader lädt Daten aus `data.json` 
3. **Optimierung**: All-Klasse koordiniert das Training mit verschiedenen Optimierern
4. **Crosstraining**: Zufällige Auswahl von Aktivierungsmatrizen für robusteres Training
5. **Reporting**: Generierung von PDF-Berichten mit Trainingsmetriken

### Testprozess
1. **Testdatengenerierung**: Erstellen von Testdaten für die Evaluierung
2. **Schaltkreisevaluierung**: Ausführen der optimierten Schaltkreise mit Testdaten
3. **Messung**: Erfassung und Analyse der Quantenmessungen
4. **Berichterstellung**: Generierung eines Test-PDF-Berichts

## Dateistruktur und Datenformat
- **`var/config.json`**: Konfigurationseinstellungen für Logs und Programm
- **`var/data.json`**: Enthält Trainingsdaten, Matrixdefinitionen und Optimierungskonfigurationen
- **`var/visual/`**: Enthält Visualisierungskomponenten (Logo, Beschreibungen, etc.)

### Datenformat-Beispiel
`data.json` enthält:
- Quantenschaltkreis-Parameter (Qubits, Tiefe)
- Aktivierungsmatrizen für verschiedene Zustände (Hund, Katze, etc.)
- Optimizer-Konfigurationen mit spezifischen Parametern
- Zustandsdefinitionen für das Training

## Kommandos
- Run all tests: `python dml/test.py`
- Run specific test: `pytest dml/module/test/test_data.py`
- Run specific test function: `pytest dml/module/test/test_all.py::test_Precheck_config_missing`
- Start the application: `python dml/main.py`

## Code-Stil
- **Imports**: Standard library first, then third-party (qiskit, numpy, etc.), then local modules
- **Naming**: Classes = PascalCase, methods/functions = snake_case, constants = UPPER_SNAKE_CASE
- **Documentation**: Classes and methods use docstrings in triple quotes """
- **Error Handling**: Use try/except with specific exceptions and numbered error codes (e.g., 1001)
- **Comments**: Include project header with author/contact info at the top of each file

## Geplante Erweiterungen
1. **Overview-Segment**: Verbesserte Visualisierung der Trainingsumgebung
2. **Test-Segment**: Automatische Generierung von Testergebnisabschnitten
3. **Crosstraining-Optimierung**: Verbesserte Algorithmen für robusteres Crosstraining
4. **Visual-Erweiterungen**: Mehr Diagramme und Visualisierungen für besseres Debugging und Analyse