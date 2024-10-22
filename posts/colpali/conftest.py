import os
import sys

from utils import read_from_queue

# Add the root project directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def pytest_sessionfinish(session, exitstatus):
    while True:
        message = read_from_queue()
        if not message:
            break
    print("\nTest Suite Finished.")
    print(f"Modal Queue Empty. Exit status: {exitstatus}")
