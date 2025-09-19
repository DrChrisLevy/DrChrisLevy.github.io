import os
import shutil

import pytest


@pytest.fixture(scope="session")
def test_db_path():
    """Create a temporary database file for testing."""
    # Create a temporary directory
    temp_dir = "data/test_data/"
    os.makedirs(temp_dir, exist_ok=True)
    db_file = os.path.join(temp_dir, "test_teaching.db")

    # Set the environment variable for the database path
    os.environ["DB_PATH"] = str(db_file)

    yield str(db_file)

    # Clean up the temporary directory after tests
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="function")
def test_db(test_db_path):
    """Provide a fresh database for each test function."""
    # Import here to ensure DB_PATH environment variable is set before import
    from db.db import DataBase

    # Reset/initialize the database for each test
    # If the file exists, delete it
    if os.path.exists(test_db_path):
        os.unlink(test_db_path)

    # The tables will be recreated when importing the db module

    yield DataBase

    # Optional: clean up after the test
    # For faster tests, you might want to skip this and just reset at the beginning
