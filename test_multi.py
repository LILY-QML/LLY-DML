from module.src.circuit import Circuit
from module.src.data import Data


# Beispielverwendung
if __name__ == "__main__":
    # Normaler Modus
    print("=== Normaler Modus ===")
    circuit = Circuit()
    circuit.read_data()

    # Option 1: Leere Run generieren
    print("\nGeneriere leere Run:")
    circuit.convert_input_data(option='empty')
    print(json.dumps(circuit.input_data, indent=4))

    # Option 2: Aktivierungsmatrix nach Name verarbeiten
    # Angenommen, es gibt eine Aktivierungsmatrix namens "matrix1" in train.json
    print("\nVerarbeite Aktivierungsmatrix 'matrix1':")
    circuit.convert_input_data(option='named', matrix_name='matrix1')
    print(json.dumps(circuit.input_data, indent=4))

    # Erstelle den initialen Schaltkreis
    print("\nErstelle den initialen Schaltkreis:")
    circuit.create_initial_circuit()
    print(json.dumps(circuit.circuit, indent=4))

    # Validieren des Schaltkreises
    print("\nValidiere den Schaltkreis:")
    is_valid = circuit.check_circuit()
    print(f"Schaltkreis validiert: {is_valid}")

    # Führe Messungen durch
    if is_valid:
        print("\nFühre Messungen durch:")
        shots = 5  # Beispiel: 5 shots
        measurements = circuit.measure(shots)
        print(json.dumps(measurements, indent=4))

    # Testmodus
    print("\n=== Testmodus ===")
    test_circuit = Circuit(test_mode=True)
    test_circuit.read_data()

    # Option 1: Leere Run generieren
    print("\nGeneriere leere Run im Testmodus:")
    test_circuit.convert_input_data(option='empty')
    print(json.dumps(test_circuit.input_data, indent=4))

    # Option 2: Aktivierungsmatrix nach Name verarbeiten
    print("\nVerarbeite Aktivierungsmatrix 'matrix1' im Testmodus:")
    test_circuit.convert_input_data(option='named', matrix_name='matrix1')
    print(json.dumps(test_circuit.input_data, indent=4))

    # Erstelle den initialen Schaltkreis im Testmodus
    print("\nErstelle den initialen Schaltkreis im Testmodus:")
    test_circuit.create_initial_circuit()
    print(json.dumps(test_circuit.circuit, indent=4))

    # Validieren des Schaltkreises im Testmodus
    print("\nValidiere den Schaltkreis im Testmodus:")
    test_is_valid = test_circuit.check_circuit()
    print(f"Schaltkreis validiert im Testmodus: {test_is_valid}")

    # Führe Messungen im Testmodus durch
    if test_is_valid:
        print("\nFühre Messungen im Testmodus durch:")
        test_shots = 3  # Beispiel: 3 shots
        test_measurements = test_circuit.measure(test_shots)
        print(json.dumps(test_measurements, indent=4))
