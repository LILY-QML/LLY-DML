import pytest

# Run tests for test_data.py, test_optimizer.py, and test_reader.py
if __name__ == "__main__":
    pytest.main([
        "module/test/test_data.py",
        "module/test/test_optimizer.py",
        "module/test/test_reader.py"
    ])
list