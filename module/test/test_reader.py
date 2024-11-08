# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Project: LILY-QML
# Version: 2.0.0 LLY-DML
# Author: Claudia Zendejas-Morales (@clausia)
# Contact: info@lilyqml.de
# Website: www.lilyqml.de
# Contributors: Joan Pujol (@supercabb)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import json
import tempfile
from datetime import datetime
import pytest
from module.src.reader import Reader


@pytest.fixture
def temp_reader():
    # Creates a temporary directory and initializes a Reader object with it
    temp_dir = tempfile.TemporaryDirectory()

    # Define a minimal config.json content for the tests
    config_data = {
        "logfile": "2024-01-01.log"  # Placeholder logfile name
    }

    # Write config.json in the temporary directory
    config_path = os.path.join(temp_dir.name, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config_data, f)

    # Initialize Reader with the temporary directory
    reader = Reader(working_directory=temp_dir.name)
    yield reader
    temp_dir.cleanup()  # Cleans up the temporary directory after each test


# Test for fileCheck
def test_file_check_missing_files(temp_reader):
    # Runs fileCheck without required files to check the error message
    result = temp_reader.fileCheck()
    assert "Error: Missing the following files:" in result


def test_file_check_with_files(temp_reader):
    # Creates the required files in the temporary directory
    required_files = ['train.json', 'config.json', 'data.json']
    for filename in required_files:
        open(os.path.join(temp_reader.working_directory, filename), 'w').close()

    result = temp_reader.fileCheck()
    assert result == "All required files are present."


# Test for checkLog
def test_check_log_creates_logfile(temp_reader):
    # Creates a config.json without specifying a logfile
    config_path = os.path.join(temp_reader.working_directory, 'config.json')
    with open(config_path, 'w') as f:
        json.dump({}, f)

    temp_reader.checkLog()

    # Checks that a log file with today's date has been created
    today_log = f"{datetime.now().strftime('%Y-%m-%d')}.log"
    log_path = os.path.join(temp_reader.working_directory, today_log)
    assert os.path.exists(log_path)

    # Verifies that config.json is updated with the new log file name
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    assert config_data.get('logfile') == today_log


# Test for createLog
def test_create_log(temp_reader):
    # Calls createLog to generate a new log file
    temp_reader.createLog()

    # Checks that the log file with today's date is created
    today_log = f"{datetime.now().strftime('%Y-%m-%d')}.log"
    log_path = os.path.join(temp_reader.working_directory, today_log)
    assert os.path.exists(log_path)

    # Verifies that config.json is updated with the name of the new log file
    config_path = os.path.join(temp_reader.working_directory, 'config.json')
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    assert config_data.get('logfile') == today_log


# Test for dataConsistency
def test_data_consistency_valid(temp_reader):
    # Creates a valid data.json file with the required structure and data
    data = {
        "qubits": 5,
        "depth": 10,
        "optimizers": ["AdamOptimizer"],
        "optimizer_arguments": {
            "AdamOptimizer": {
                "learning_rate": 0.001,
                "beta_1": 0.9,
                "beta_2": 0.999
            }
        },
        "matrices": {
            "Hund": [[1, 2], [3, 4]]
        }
    }
    data_path = os.path.join(temp_reader.working_directory, 'data.json')
    with open(data_path, 'w') as f:
        json.dump(data, f)

    result = temp_reader.dataConsistency()
    assert result == "data.json is consistent."


def test_data_consistency_invalid_structure(temp_reader):
    # Creates a data.json with an invalid structure (missing required keys)
    data = {
        "qubits": 5,
        "depth": 10
        # Missing "optimizers", "optimizer_arguments", "matrices"
    }
    data_path = os.path.join(temp_reader.working_directory, 'data.json')
    with open(data_path, 'w') as f:
        json.dump(data, f)

    result = temp_reader.dataConsistency()
    assert "Error: Incorrect structure in data.json" in result

def test_create_train_file(temp_reader):
    result = temp_reader.create_train_file()
    assert result == None

    result = temp_reader.create_train_file()
    assert result == {"Error Code": 1199, "Message": "train.json already exists."}

def test_move_json_file(temp_reader):

    if os.path.exists(os.path.join(temp_reader.working_directory, 'train.json')):
        os.remove(os.path.join(temp_reader.working_directory, 'train.json'))

    result = temp_reader.move_json_file()
    assert result == {"Error Code": 1188, "Message": "train.json not found."}

    with open(os.path.join(temp_reader.working_directory, 'train.json'), 'w') as f:
        f.write('test')

    result = temp_reader.move_json_file()
    assert result == None

