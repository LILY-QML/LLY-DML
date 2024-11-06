# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Project: LILY-QML
# Version: 2.0.0 LLY-DML
# Author: Claudia Zendejas-Morales (@clausia)
# Contact: info@lilyqml.de
# Website: www.lilyqml.de
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import json
import numpy as np
import pytest
import tempfile
from unittest.mock import patch
from module.src.data import Data


@pytest.fixture
def temp_data():
    # Creates a temporary directory and initializes a Data instance for testing
    temp_dir = tempfile.TemporaryDirectory()
    data_instance = Data(qubits=3, depth=4, working_directory=temp_dir.name)
    yield data_instance
    temp_dir.cleanup()


# Test for get_data
def test_get_data_missing_files(temp_data):
    result = temp_data.get_data()
    assert result == "Error Code: 1028 - data.json not found"

    data_content = {"qubits": 3, "depth": 4, "activation_matrices": []}
    with open(os.path.join(temp_data.working_directory, 'data.json'), 'w') as f:
        json.dump(data_content, f)

    result = temp_data.get_data()
    assert result == "Error Code: 1029 - train.json not found"


def test_get_data_success(temp_data):
    data_content = {"qubits": 3, "depth": 4, "activation_matrices": []}
    with open(os.path.join(temp_data.working_directory, 'data.json'), 'w') as f:
        json.dump(data_content, f)
    with open(os.path.join(temp_data.working_directory, 'train.json'), 'w') as f:
        json.dump({}, f)

    result = temp_data.get_data()
    assert result is None
    assert temp_data.qubits == 3
    assert temp_data.depth == 4
    assert temp_data.activation_matrices == []


# Test for return_matrices
def test_return_matrices_valid(temp_data):
    temp_data.activation_matrices = [
        {
            "name": "Matrix_1",
            "data": np.zeros((3, 12, 3))  # 3 rows, 12 columns, 3 pages
        }
    ]
    result = temp_data.return_matrices()
    assert result == temp_data.activation_matrices


def test_return_matrices_invalid(temp_data):
    temp_data.activation_matrices = [{"data": np.zeros((3, 12, 3))}]
    result = temp_data.return_matrices()
    assert result == "Error Code: 1030 - Activation matrix conversion unsuccessful"


# Test for convert_matrices
def test_convert_matrices_success(temp_data):
    temp_data.activation_matrices = [
        {
            "name": "Matrix_1",
            "data": np.random.rand(3, 4, 3)  # 3 rows, 4 columns, 3 pages before conversion
        }
    ]
    result = temp_data.convert_matrices()
    assert result is None
    for matrix in temp_data.activation_matrices:
        assert temp_data.check_final_matrix(matrix['data'])


def test_convert_matrices_invalid_before_conversion(temp_data):
    temp_data.activation_matrices = [np.random.rand(3, 4, 3)]
    result = temp_data.convert_matrices()
    assert result == "Error Code: 1031 - Activation matrix invalid before conversion"


# Test for check_final_matrix
def test_check_final_matrix_valid(temp_data):
    # Update to use a valid 2D matrix after conversion
    valid_matrix = np.zeros((3, 12))  # 3 rows, 12 columns after conversion
    assert temp_data.check_final_matrix(valid_matrix) is True


def test_check_final_matrix_invalid(temp_data):
    # Update to use a 2D matrix with incorrect dimensions
    invalid_matrix = np.zeros((3, 10))  # Incorrect number of columns after conversion
    assert temp_data.check_final_matrix(invalid_matrix) is False


# Test for create_training_matrix
def test_create_training_matrix_success(temp_data):
    result = temp_data.create_training_matrix()
    assert result is None

    train_path = os.path.join(temp_data.working_directory, 'train.json')
    assert os.path.exists(train_path)

    with open(train_path, 'r') as f:
        data = json.load(f)
    assert "training_matrix" in data
    assert len(data["training_matrix"]) == temp_data.qubits
    assert len(data["training_matrix"][0]) == temp_data.depth * 3


def test_create_training_matrix_invalid(temp_data):
    # Simulate that creating training_matrix produces a matrix with incorrect dimensions
    with patch.object(Data, 'check_final_matrix', return_value=False):
        result = temp_data.create_training_matrix()
        assert result == "Error Code: 1007 - Training matrix does not meet required dimensions"
