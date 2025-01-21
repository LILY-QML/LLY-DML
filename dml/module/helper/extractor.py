# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Project: LILY-QML
# Version: 2.0.0 LLY-DML
# Author: Leon Kaiser
# Contact: info@lilyqml.de
# Website: www.lilyqml.de
# Contributors:
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++

class Extractor:
    def __init__(self, data, matrix):
        """
        Initializes the Extractor class with a dictionary where keys are binary strings and values are integers,
        and a 2D matrix that corresponds to the positions of the binary strings.
        """
        self.data = data
        self.matrix = matrix
        self.extracted_data = None

    def analyze_positions(self):
        """
        Analyzes the positions of 0s and 1s in the binary strings and returns a count for each position.
        """
        position_counts = [{'0': 0, '1': 0} for _ in range(len(list(self.data.keys())[0]))]

        # Processing each binary string
        for binary_str in self.data.keys():
            for i, char in enumerate(binary_str):
                position_counts[i][char] += 1
        
        return position_counts

    def assign_matrix_to_positions(self):
        """
        Assigns each row of the matrix to the corresponding position in the binary strings.
        Additionally, includes the counts of 0s and 1s at each position.
        """
        result_arrays = []
        position_counts = self.analyze_positions()

        for i in range(len(self.matrix)):
            matrix_row = self.matrix[i]
            zero_count = position_counts[i]['0']
            one_count = position_counts[i]['1']
            result_arrays.append([matrix_row, zero_count, one_count])

        self.extracted_data = result_arrays
        return self.extracted_data

    def reconstruct(self):
        """
        Takes the rows from extracted_data and converts them back into the original matrix.
        """
        if not self.extracted_data:
            raise ValueError("No extracted data to reconstruct the matrix from.")

        reconstructed_matrix = [row[0] for row in self.extracted_data]  # Extract the original matrix rows
        return reconstructed_matrix