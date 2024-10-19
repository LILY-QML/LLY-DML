import numpy as np

class Data:
    def __init__(self, qubits, depth, activation_matrices):
        """
        Initialisiert die Data-Klasse mit qubits, depth und activation_matrices.

        :param qubits: int, Anzahl der Qubits, entspricht der Anzahl der Zeilen in den Matrizen.
        :param depth: int, Tiefe, entspricht der Anzahl der Spalten in den Matrizen.
        :param activation_matrices: list, enthält die dreidimensionalen Aktivierungsmatrizen.
        :raises ValueError: Wenn keine Matrizen übergeben werden oder Dimensionen nicht übereinstimmen.
        :raises TypeError: Wenn die Matrizen nicht vom Typ NumPy-Array sind oder die Struktur nicht korrekt ist.
        """
        if not activation_matrices:
            raise ValueError("Es muss mindestens eine Aktivierungsmatrix übergeben werden.")

        self.qubits = qubits
        self.depth = depth
        self.activation_matrices = []
        self.labels = []

        for idx, matrix in enumerate(activation_matrices, start=1):
            matrix_label = f"matrix{idx}"

            if not isinstance(matrix, np.ndarray):
                raise TypeError(f"Die {matrix_label} ist kein NumPy-Array.")

            if matrix.ndim != 3:
                raise ValueError(f"Die {matrix_label} ist nicht dreidimensional. Sie hat {matrix.ndim} Dimension(en).")

            # Überprüfen der Dimensionen: (layers, qubits, depth)
            layers, matrix_qubits, matrix_depth = matrix.shape

            if matrix_qubits != qubits:
                raise ValueError(
                    f"Die {matrix_label} hat {matrix_qubits} Zeilen, erwartet: {qubits} (qubits)."
                )

            if matrix_depth != depth:
                raise ValueError(
                    f"Die {matrix_label} hat {matrix_depth} Spalten, erwartet: {depth} (depth)."
                )

            self.activation_matrices.append(matrix)
            self.labels.append(matrix_label)

        # Sicherstellen, dass alle Matrizen die gleiche Form haben (Layers, Qubits, Depth)
        first_shape = self.activation_matrices[0].shape
        for matrix in self.activation_matrices:
            if matrix.shape != first_shape:
                raise ValueError("Nicht alle Aktivierungsmatrizen haben dieselben Dimensionen.")

        self.shape = first_shape  # Alle Matrizen haben die gleiche Form

    def get_dimensions(self):
        """
        Gibt die Dimensionen jeder Aktivierungsmatrix zurück.

        :return: Eine Liste von Tupeln, die die Dimensionen jeder Matrix darstellen.
        """
        return [matrix.shape for matrix in self.activation_matrices]

    def get_number_of_matrices(self):
        """
        Gibt die Anzahl der übergebenen Aktivierungsmatrizen zurück.

        :return: Anzahl der Matrizen.
        """
        return len(self.activation_matrices)

    def summary(self):
        """
        Gibt eine Zusammenfassung der Aktivierungsmatrizen aus, einschließlich der Anzahl und Dimensionen.

        :return: Eine formatierte Zeichenkette mit der Zusammenfassung.
        """
        summary_str = f"Anzahl der Activation Matrizen: {self.get_number_of_matrices()}\n"
        for label, shape in zip(self.labels, self.get_dimensions()):
            summary_str += f"{label} Dimensionen: {shape}\n"
        summary_str += f"Qubits: {self.qubits}, Depth: {self.depth}\n"
        return summary_str

    def create_training_matrix(self):
        """
        Erstellt eine neue 2D-Matrix mit randomisierten Werten.
        Die Dimensionen sind (qubits, 3 * depth).

        :return: Eine neue 2D-NumPy-Matrix mit der Form (qubits, 3 * depth).
        :raises ValueError: Wenn die resultierende Trainingsmatrix nicht die erwarteten Dimensionen hat.
        """
        training_matrix = np.random.rand(self.qubits, 3 * self.depth)
        if training_matrix.shape != (self.qubits, 3 * self.depth):
            raise ValueError(
                f"Die Trainingsmatrix hat die Dimensionen {training_matrix.shape}, erwartet: ({self.qubits}, {3 * self.depth})."
            )
        return training_matrix

    def validate_training_matrix(self, training_matrix):
        """
        Validiert, ob die Trainingsmatrix die erwarteten Dimensionen hat.

        :param training_matrix: NumPy-Array, die zu validierende Trainingsmatrix.
        :raises ValueError: Wenn die Trainingsmatrix nicht die erwarteten Dimensionen hat.
        """
        expected_shape = (self.qubits, 3 * self.depth)
        if training_matrix.shape != expected_shape:
            raise ValueError(
                f"Die Trainingsmatrix hat die Dimensionen {training_matrix.shape}, erwartet: {expected_shape}."
            )

    def convert_activation_matrices_to_2d(self):
        """
        Konvertiert alle dreidimensionalen Aktivierungsmatrizen in zweidimensionale Matrizen.
        Dabei werden die Schichten der 3D-Matrix hintereinander gesetzt, sodass die resultierende
        Matrix die Form (depth * 3, qubits) hat.
        """
        converted_matrices = {}
        for label, matrix in zip(self.labels, self.activation_matrices):
            # Keine Transposition notwendig
            layers, qubits, depth = matrix.shape  # Erwartet (3, 9, 8)
            expected_rows = self.depth * 3  # 8 * 3 = 24
            actual_rows = layers * depth  # 3 * 8 = 24

            if actual_rows != expected_rows:
                raise ValueError(
                    f"The {label} has {actual_rows} rows after reshaping, expected: {expected_rows} (depth * 3)."
                )

            try:
                # Reshape zu (layers * depth, qubits) => (24, 9)
                converted_matrix = matrix.reshape(layers * self.depth, self.qubits).tolist()
                converted_matrices[label] = converted_matrix
            except ValueError as e:
                raise ValueError(f"Error reshaping {label}: {e}")
        return converted_matrices



    def get_matrix_names(self):
        """
        Gibt die Namen aller Aktivierungsmatrizen zurück.

        :return: Eine Liste von Labels der Matrizen.
        """
        return self.labels.copy()

    def get_matrix_by_name(self, name):
        """
        Gibt die dreidimensionale Aktivierungsmatrix mit dem angegebenen Namen zurück.

        :param name: str, der Name der gewünschten Matrix.
        :return: NumPy-Array der entsprechenden dreidimensionalen Matrix.
        :raises KeyError: Wenn der angegebene Name nicht existiert.
        """
        if name not in self.labels:
            raise KeyError(f"Matrix mit dem Namen '{name}' existiert nicht.")
        index = self.labels.index(name)
        return self.activation_matrices[index]
