from courses.modal.section5.lesson16 import ModalSandbox


class TestModalSandboxIntegration:
    """Integration test suite for ModalSandbox class using real Modal sandboxes"""

    def test_sandbox_initialization_and_basic_execution(self):
        """Test creating a new sandbox and executing basic code"""
        sandbox = None
        try:
            sandbox = ModalSandbox()

            # Verify sandbox was created and has an ID
            assert sandbox.sandbox_id is not None
            assert len(sandbox.sandbox_id) > 0

            # Test basic code execution
            result = sandbox.run_code("print('Hello, World!')")

            # Verify result structure
            assert "stdout" in result
            assert "stderr" in result
            assert "Hello, World!" in result["stdout"]
            assert result["stderr"] == ""

        finally:
            if sandbox:
                sandbox.terminate()

    def test_sandbox_reconnection_by_id(self):
        """Test reconnecting to an existing sandbox by ID"""
        sandbox1 = None
        sandbox2 = None
        try:
            # Create first sandbox and set a variable
            sandbox1 = ModalSandbox()
            sandbox_id = sandbox1.sandbox_id

            result1 = sandbox1.run_code("x = 42")
            assert result1["stderr"] == ""

            # Reconnect to the same sandbox using its ID
            sandbox2 = ModalSandbox(sandbox_id=sandbox_id)

            # Verify we're connected to the same sandbox
            assert sandbox2.sandbox_id == sandbox_id

            # Verify the variable is still available (state persistence)
            result2 = sandbox2.run_code("print(x)")
            assert "42" in result2["stdout"]
            assert result2["stderr"] == ""

        finally:
            # Only terminate once since both refer to the same sandbox
            if sandbox2:
                sandbox2.terminate()
            elif sandbox1:
                sandbox1.terminate()

    def test_code_execution_with_error(self):
        """Test code execution that produces an error"""
        sandbox = None
        try:
            sandbox = ModalSandbox()

            # Execute code that will cause an error
            result = sandbox.run_code("print(undefined_variable)")

            # Verify error was captured
            assert result["stdout"] == ""
            assert "NameError" in result["stderr"]
            assert "undefined_variable" in result["stderr"]

        finally:
            if sandbox:
                sandbox.terminate()

    def test_multiple_code_executions_with_state(self):
        """Test multiple code executions that build on each other"""
        sandbox = None
        try:
            sandbox = ModalSandbox()

            # Define a variable
            result1 = sandbox.run_code("counter = 0")
            assert result1["stderr"] == ""

            # Increment the variable
            result2 = sandbox.run_code("counter += 5")
            assert result2["stderr"] == ""

            # Check the variable value
            result3 = sandbox.run_code("print(f'Counter is: {counter}')")
            assert "Counter is: 5" in result3["stdout"]
            assert result3["stderr"] == ""

            # Define a function and use it
            result4 = sandbox.run_code("""
def multiply_counter(factor):
    global counter
    counter *= factor
    return counter

result = multiply_counter(3)
print(f'Result: {result}')
""")
            assert "Result: 15" in result4["stdout"]
            assert result4["stderr"] == ""

        finally:
            if sandbox:
                sandbox.terminate()

    def test_pandas_functionality(self):
        """Test that pandas is available and working in the sandbox"""
        sandbox = None
        try:
            sandbox = ModalSandbox()

            # Test pandas operations
            result = sandbox.run_code("""
import pandas as pd

# Create a simple DataFrame
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'city': ['New York', 'London', 'Tokyo']
})

print("DataFrame created successfully:")
print(df.to_string())
print(f"Shape: {df.shape}")
""")

            assert result["stderr"] == ""
            assert "DataFrame created successfully:" in result["stdout"]
            assert "Alice" in result["stdout"]
            assert "Shape: (3, 3)" in result["stdout"]

        finally:
            if sandbox:
                sandbox.terminate()

    def test_init_script_execution(self):
        """Test sandbox initialization with an init script"""
        sandbox = None
        try:
            init_script = """
import json
INITIALIZED = True
init_data = {'status': 'ready', 'version': '1.0'}
print('Sandbox initialized with custom script')
"""

            sandbox = ModalSandbox(init_script=init_script)

            # Verify init script ran and variables are available
            result = sandbox.run_code("""
print(f'Initialized: {INITIALIZED}')
print(f'Init data: {init_data}')
""")

            assert result["stderr"] == ""
            assert "Initialized: True" in result["stdout"]
            assert "status" in result["stdout"]
            assert "ready" in result["stdout"]

        finally:
            if sandbox:
                sandbox.terminate()

    def test_long_running_computation(self):
        """Test a computation that takes some time to complete"""
        sandbox = None
        try:
            sandbox = ModalSandbox()

            # Execute a computation that takes a moment
            result = sandbox.run_code("""
import time

print("Starting computation...")
total = 0
for i in range(1000):
    total += i * i
    if i % 200 == 0:
        print(f"Progress: {i}/1000")

print(f"Final result: {total}")
""")

            assert result["stderr"] == ""
            assert "Starting computation..." in result["stdout"]
            assert "Final result:" in result["stdout"]
            assert "Progress:" in result["stdout"]

        finally:
            if sandbox:
                sandbox.terminate()

    def test_json_data_processing(self):
        """Test processing JSON data in the sandbox"""
        sandbox = None
        try:
            sandbox = ModalSandbox()

            result = sandbox.run_code("""
import json

# Create some test data
data = {
    'users': [
        {'id': 1, 'name': 'Alice', 'score': 85},
        {'id': 2, 'name': 'Bob', 'score': 92},
        {'id': 3, 'name': 'Charlie', 'score': 78}
    ]
}

# Process the data
high_scorers = [user for user in data['users'] if user['score'] > 80]
average_score = sum(user['score'] for user in data['users']) / len(data['users'])

print(f"High scorers: {len(high_scorers)}")
print(f"Average score: {average_score:.1f}")

for user in high_scorers:
    print(f"- {user['name']}: {user['score']}")
""")

            assert result["stderr"] == ""
            assert "High scorers: 2" in result["stdout"]
            assert "Average score: 85.0" in result["stdout"]
            assert "Alice: 85" in result["stdout"]
            assert "Bob: 92" in result["stdout"]

        finally:
            if sandbox:
                sandbox.terminate()

    def test_custom_timeout(self):
        """Test creating sandbox with custom timeout"""
        sandbox = None
        try:
            # Create sandbox with 30-minute timeout
            sandbox = ModalSandbox(timeout=30 * 60)

            # Verify it works normally
            result = sandbox.run_code("print('Custom timeout sandbox working')")
            assert "Custom timeout sandbox working" in result["stdout"]
            assert result["stderr"] == ""

        finally:
            if sandbox:
                sandbox.terminate()

    def test_exception_handling_in_code(self):
        """Test that exceptions in user code are properly captured"""
        sandbox = None
        try:
            sandbox = ModalSandbox()

            result = sandbox.run_code("""
try:
    result = 10 / 0
except ZeroDivisionError as e:
    print(f"Caught exception: {e}")
    print("Division by zero handled gracefully")

# This should still work after the exception
print("Continuing after exception...")
""")

            assert result["stderr"] == ""
            assert "Caught exception:" in result["stdout"]
            assert "Division by zero handled gracefully" in result["stdout"]
            assert "Continuing after exception..." in result["stdout"]

        finally:
            if sandbox:
                sandbox.terminate()

    def test_import_and_use_tabulate(self):
        """Test that tabulate package is available and working"""
        sandbox = None
        try:
            sandbox = ModalSandbox()

            result = sandbox.run_code("""
from tabulate import tabulate

# Create table data
data = [
    ['Alice', 25, 'Engineer'],
    ['Bob', 30, 'Designer'],
    ['Charlie', 35, 'Manager']
]

headers = ['Name', 'Age', 'Role']

# Create formatted table
table = tabulate(data, headers=headers, tablefmt='grid')
print("Formatted table:")
print(table)
""")

            assert result["stderr"] == ""
            assert "Formatted table:" in result["stdout"]
            assert "Alice" in result["stdout"]
            assert "Engineer" in result["stdout"]
            # Grid format should include borders
            assert "+" in result["stdout"] or "|" in result["stdout"]

        finally:
            if sandbox:
                sandbox.terminate()

    def test_reconnect_to_non_existent_sandbox(self):
        """Test behavior when trying to reconnect to non-existent sandbox"""
        sandbox = None
        try:
            # Try to connect to a sandbox with a fake ID
            fake_id = "fake-sandbox-id-12345"

            # This should create a new sandbox instead of failing
            sandbox = ModalSandbox(sandbox_id=fake_id)

            # Verify we got a new sandbox (different ID)
            assert sandbox.sandbox_id != fake_id

            # Verify it works
            result = sandbox.run_code("print('New sandbox created')")
            assert "New sandbox created" in result["stdout"]

        finally:
            if sandbox:
                sandbox.terminate()
