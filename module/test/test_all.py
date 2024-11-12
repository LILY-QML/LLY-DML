# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Project: LILY-QML
# Version: 2.0.0 LLY-DML
# Author: Claudia Zendejas-Morales (@clausia)
# Contact: info@lilyqml.de
# Website: www.lilyqml.de
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import tempfile
import pytest
import json
import numpy as np
from datetime import datetime
from module.all import All


@pytest.fixture
def setup_all():
    # Set up a temporary directory and create necessary files for the All class
    temp_dir = tempfile.TemporaryDirectory()

    # Define a minimal config.json content for the tests
    config_data = {"logfile": f"{datetime.now().strftime('%Y-%m-%d')}.log"}
    config_path = os.path.join(temp_dir.name, 'config.json')

    with open(config_path, 'w') as f:
        json.dump(config_data, f)

    all_instance = All(working_directory=temp_dir.name)
    yield all_instance
    temp_dir.cleanup()


def test_handle_train_file_exists_option1(setup_all, monkeypatch):
    # Paths for the train.json file and archive directory
    train_path = os.path.join(setup_all.working_directory, 'train.json')
    archive_dir = os.path.join(setup_all.working_directory, 'archive')

    # Create train.json with dummy data and ensure 'archive' directory exists
    with open(train_path, 'w') as train_file:
        json.dump({"dummy": "data"}, train_file)
    os.makedirs(archive_dir, exist_ok=True)

    # Print debug output to confirm initial conditions
    print(f"Initial state: train.json exists at original path: {os.path.exists(train_path)}")
    print(f"Initial state: archive directory exists: {os.path.exists(archive_dir)}")

    # Simulate user input for option 1 (to archive the file)
    monkeypatch.setattr('builtins.input', lambda _: "1")
    setup_all.handle_train_file()

    # Define the expected archive path by listing the archive directory contents
    archive_files = os.listdir(archive_dir)
    archive_path = os.path.join(archive_dir, archive_files[0]) if archive_files else None

    # Debugging output to verify where train.json is located after archiving
    print(f"After handle_train_file: train.json exists at original path: {os.path.exists(train_path)}")
    print(f"After handle_train_file: train.json exists at archive path: {os.path.exists(archive_path) if archive_path else False}")

    # Assertions to verify that train.json was archived
    assert archive_path is not None and os.path.exists(archive_path), "train.json should exist in the archive directory."

    # Verify that a new train.json file has been created in the original location
    assert os.path.exists(train_path), "A new train.json should be created after archiving."


def test_handle_train_file_exists_option2(setup_all, monkeypatch):
    # Simulate that train.json exists and capture user input for option 2 (delete)
    train_path = os.path.join(setup_all.working_directory, 'train.json')
    with open(train_path, 'w') as train_file:
        json.dump({"dummy": "data"}, train_file)

    # Simulate user input for "2" (delete option)
    monkeypatch.setattr('builtins.input', lambda _: "2")
    setup_all.handle_train_file()

    # Verify that train.json still exists since it was deleted and then recreated
    assert os.path.exists(train_path), "New train.json should be created after deleting the old one."

    # (Optional) If you want to verify that the file was recreated:
    # Check that the content of the new file is different (i.e., not the dummy data)
    with open(train_path, 'r') as new_train_file:
        content = json.load(new_train_file)
    assert content != {"dummy": "data"}, "The new train.json should have different content from the old one."


def test_handle_train_file_not_exists(setup_all):
    # Ensure train.json does not exist before running the method
    train_path = os.path.join(setup_all.working_directory, 'train.json')
    if os.path.exists(train_path):
        os.remove(train_path)

    # Run handle_train_file, which should create a new train.json if it doesn't exist
    setup_all.handle_train_file()

    # Assert that train.json is created
    assert os.path.exists(train_path), "train.json should be created if it does not exist."

    # Check that the created train.json is a valid JSON and has expected content
    try:
        with open(train_path, 'r') as train_file:
            content = json.load(train_file)  # Attempt to load JSON
            assert isinstance(content, list), "train.json content should be a list."
            assert len(content) == 1, "train.json should have one entry."
            assert "creation" in content[0], "train.json should contain 'creation' key."
            assert isinstance(content[0]["creation"], str), "'creation' value should be a string representing a timestamp."
    except json.JSONDecodeError:
        assert False, "train.json should contain valid JSON."


def test_data_ready_success(setup_all):
    # Simulate valid data.json content with a properly formatted activation matrix as lists
    data_content = {
        "qubits": 3,
        "depth": 4,
        "activation_matrices": [
            {
                "name": "Matrix_1",
                "data": np.zeros((3, 4, 3)).tolist()
            }
        ]
    }

    # Write data.json with correct activation matrix format
    data_path = os.path.join(setup_all.working_directory, 'data.json')
    with open(data_path, 'w') as data_file:
        json.dump(data_content, data_file)

    # Create an empty valid train.json file to pass the get_data() check
    train_content = [{"creation": "2024-11-11 12:00:00"}]
    train_path = os.path.join(setup_all.working_directory, 'train.json')
    with open(train_path, 'w') as train_file:
        json.dump(train_content, train_file)

    # Run data_ready to check matrix conversion and training matrix creation
    setup_all.data_ready(qubits=3, depth=4)

    # Verify that train.json now contains the activation and training matrices
    with open(train_path, 'r') as train_file:
        train_data = json.load(train_file)

    # Assertions for activation_matrices
    assert "activation_matrices" in train_data, "train.json should contain 'activation_matrices'."
    assert len(train_data["activation_matrices"]) == 1, "There should be one activation matrix."
    assert len(train_data["activation_matrices"][0]) == 3, "Activation matrix should have 3 rows for 3 qubits."
    assert len(train_data["activation_matrices"][0][0]) == 12, "Activation matrix should have 12 columns (depth * 3)."

    # Assertions for training_matrix
    assert "training_matrix" in train_data, "train.json should contain 'training_matrix'."
    assert len(train_data["training_matrix"]) == 3, "Training matrix should have 3 rows for 3 qubits."
    assert len(train_data["training_matrix"][0]) == 12, "Training matrix should have 12 columns for depth * 3."


def test_Precheck_config_missing(setup_all, capsys):
    # Remove config.json to simulate it missing
    config_path = os.path.join(setup_all.working_directory, 'config.json')
    if os.path.exists(config_path):
        os.remove(config_path)

    # Run Precheck
    setup_all.Precheck()

    # Capture output and check for error code
    captured = capsys.readouterr()
    assert "Error Code: 1099 - config.json not found." in captured.out


def test_Precheck_log_missing(setup_all, monkeypatch, capsys):
    # Ensure config.json exists
    config_path = os.path.join(setup_all.working_directory, 'config.json')
    with open(config_path, 'w') as config_file:
        json.dump({}, config_file)

    # Simulate missing log file by having checkLog return an error
    monkeypatch.setattr(setup_all.reader, 'checkLog', lambda: {"Error Code": 1099, "Message": "Logfile not found"})

    # Run Precheck
    setup_all.Precheck()

    # Capture output and check for error code
    captured = capsys.readouterr()
    assert "Error Code: 1099 - Logfile not found." in captured.out


def test_Precheck_data_json_missing(setup_all, monkeypatch, capsys):
    # Ensure config.json and log exist
    config_path = os.path.join(setup_all.working_directory, 'config.json')
    with open(config_path, 'w') as config_file:
        json.dump({}, config_file)
    monkeypatch.setattr(setup_all.reader, 'checkLog', lambda: None)

    # Remove data.json to simulate it missing
    data_path = os.path.join(setup_all.working_directory, 'data.json')
    if os.path.exists(data_path):
        os.remove(data_path)

    # Simulate user input for qubits and depth
    monkeypatch.setattr('builtins.input', lambda prompt: '4')

    # Run Precheck
    setup_all.Precheck()

    # Capture output and check for prompt about missing data.json
    captured = capsys.readouterr()
    assert "data.json not found. Please provide values for qubits and depth." in captured.out


def test_Precheck_data_json_malformed(setup_all, capsys):
    # Ensure config.json and log exist
    config_path = os.path.join(setup_all.working_directory, 'config.json')
    with open(config_path, 'w') as config_file:
        json.dump({}, config_file)

    # Ensure log check passes
    setup_all.reader.checkLog = lambda: None

    # Create a malformed data.json
    data_path = os.path.join(setup_all.working_directory, 'data.json')
    with open(data_path, 'w') as data_file:
        data_file.write("{malformed: true,}")  # Invalid JSON format

    # Run Precheck
    setup_all.Precheck()

    # Capture output and check for JSON decode error message
    captured = capsys.readouterr()
    assert "data.json is improperly formatted." in captured.out


def test_Precheck_missing_qubits_depth(setup_all, capsys):
    # Ensure config.json and log exist
    config_path = os.path.join(setup_all.working_directory, 'config.json')
    with open(config_path, 'w') as config_file:
        json.dump({}, config_file)

    # Ensure log check passes
    setup_all.reader.checkLog = lambda: None

    # Create data.json without qubits and depth
    data_path = os.path.join(setup_all.working_directory, 'data.json')
    with open(data_path, 'w') as data_file:
        json.dump({"activation_matrices": []}, data_file)  # Missing 'qubits' and 'depth'

    # Run Precheck
    setup_all.Precheck()

    # Capture output and check for error message about missing qubits/depth
    captured = capsys.readouterr()
    assert "Error: qubits and depth values are required." in captured.out

