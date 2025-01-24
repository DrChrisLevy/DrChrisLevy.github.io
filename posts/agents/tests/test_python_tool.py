import unittest

# Assuming PythonSession and ExecutionResult are in python_session.py
from python_tool import PythonSession


class TestPythonSession(unittest.TestCase):
    def setUp(self):
        """Create a fresh PythonSession for each test."""
        self.session = PythonSession()

    def tearDown(self):
        """Reset the session after each test to avoid cross-test pollution."""
        self.session.reset()

    # === Basic Usage Tests ===
    def test_simple_execution(self):
        """Test basic code execution and state capture."""
        result = self.session.run_code("x = 42")
        self.assertTrue(result.is_success(), "Execution should succeed with no errors.")
        self.assertIn("x", self.session.get_state(), "Variable x should appear in session state.")
        self.assertEqual(self.session.get_state()["x"], 42, "x should be set to 42.")

    def test_print_capture(self):
        """Test stdout capture."""
        result = self.session.run_code('print("hello")')
        self.assertTrue(result.is_success())
        self.assertEqual(result.stdout.strip(), "hello", "Should capture printed output.")

    def test_return_value(self):
        """Test capturing the final expression as return_value."""
        result = self.session.run_code("42")
        self.assertTrue(result.is_success())
        self.assertEqual(result.return_value, 42, "Return value should be 42.")

    # === State Management Tests ===
    def test_state_persistence(self):
        """Test variable persistence between runs."""
        self.session.run_code("x = 10")
        result = self.session.run_code("x + 5")
        self.assertTrue(result.is_success())
        self.assertEqual(result.return_value, 15, "Should see x as 10 and produce 15.")

    def test_state_reset(self):
        """Test state reset clears previously defined variables."""
        self.session.run_code("x = 10")
        self.session.reset()
        result = self.session.run_code("print(x)")
        self.assertFalse(result.is_success(), "Accessing x after reset should fail.")
        self.assertIsNotNone(result.exception)
        self.assertIn("NameError", result.traceback)

    def test_update_state(self):
        """Test directly updating the session state with a dict."""
        self.session.update_state({"x": 100, "y": 200})
        result = self.session.run_code("x + y")
        self.assertTrue(result.is_success())
        self.assertEqual(result.return_value, 300, "x+y=300 after updating session state.")

    # === Advanced Usage Tests ===
    def test_multiline_code(self):
        """Test multiline code execution and capturing both prints and return."""
        code = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

result = factorial(5)
print(f"Factorial: {result}")
result"""
        result = self.session.run_code(code)
        self.assertTrue(result.is_success())
        self.assertEqual(result.return_value, 120)
        self.assertIn("Factorial: 120", result.stdout)

    def test_class_definition(self):
        """Test class definition and usage with a subsequent statement."""
        code1 = """
class Person:
    def __init__(self, name):
        self.name = name
    
    def greet(self):
        return f"Hello, {self.name}!"
        """
        self.session.run_code(code1)

        result = self.session.run_code('person = Person("Alice"); person.greet()')
        self.assertTrue(result.is_success())
        self.assertEqual(result.return_value, "Hello, Alice!")

    def test_generator_function(self):
        """Test generator function and capturing the next() call result."""
        code = """
def gen():
    yield 1
    yield 2
    yield 3

g = gen()
next(g)"""
        result = self.session.run_code(code)
        self.assertTrue(result.is_success())
        self.assertEqual(result.return_value, 1, "First next(g) should yield 1.")

    # === Error Handling Tests ===
    def test_syntax_error(self):
        """Test syntax error handling (compile-time error)."""
        result = self.session.run_code('if True print("wrong")')
        self.assertFalse(result.is_success())
        self.assertIn("SyntaxError", result.traceback)

    def test_runtime_error(self):
        """Test runtime error handling (division by zero)."""
        result = self.session.run_code("1/0")
        self.assertFalse(result.is_success())
        self.assertIn("ZeroDivisionError", result.traceback)
        self.assertIn("division by zero", result.exception)

    def test_name_error(self):
        """Test undefined variable handling."""
        result = self.session.run_code("undefined_var")
        self.assertFalse(result.is_success())
        assert result.exception == "name 'undefined_var' is not defined"

    # === Edge Cases ===
    def test_empty_code(self):
        """Test empty code string."""
        result = self.session.run_code("")
        self.assertTrue(result.is_success(), "Empty code should not fail.")
        self.assertEqual(result.stdout, "", "No output expected.")
        self.assertIsNone(result.return_value, "No return value expected.")

    def test_only_whitespace(self):
        """Test whitespace-only code."""
        result = self.session.run_code("    \n    \t    \n")
        self.assertTrue(result.is_success())
        self.assertEqual(result.stdout, "", "No output for whitespace-only code.")
        self.assertIsNone(result.return_value, "No return value expected.")

    def test_infinite_loop_prevention(self):
        """
        Test handling of potential infinite loops.

        NOTE: Using signal.alarm may not work on Windows by default.
              This test is potentially system-dependent.
        """
        code = """
import signal

def handler(signum, frame):
    raise TimeoutError("Code execution timed out")

signal.signal(signal.SIGALRM, handler)
signal.alarm(1)  # Set timeout for 1 second

i = 0
while True:
    i += 1"""
        result = self.session.run_code(code)
        self.assertFalse(result.is_success(), "Infinite loop should be cut off by TimeoutError.")
        self.assertIn("TimeoutError", str(result.traceback))
        assert result.exception == "Code execution timed out"

    def test_large_output(self):
        """Test handling of large output (1 million characters)."""
        code = 'print("x" * 1000000)'  # 1M characters
        result = self.session.run_code(code)
        self.assertTrue(result.is_success())
        # +1 for the newline appended by print
        self.assertEqual(len(result.stdout), 1000001)

    def test_recursive_function(self):
        """Test deep recursion handling that should raise RecursionError."""
        code = """
def recurse(n):
    return recurse(n + 1)

recurse(0)
        """
        result = self.session.run_code(code)
        self.assertFalse(result.is_success(), "Should hit recursion limit.")
        assert result.exception == "maximum recursion depth exceeded"

    def test_complex_return_value(self):
        """Test handling of a more complex returned object."""
        code = """
class ComplexObj:
    def __init__(self):
        self.data = {"a": [1, 2, {"b": 3}]}
    def __repr__(self):
        return "ComplexObj"

ComplexObj()"""
        result = self.session.run_code(code)
        self.assertTrue(result.is_success())
        self.assertIsNotNone(result.return_value, "Should return a ComplexObj instance.")
        # It's up to you if you want to do further inspection.
        # By default, return_value.__repr__ == "ComplexObj"

    def test_semicolon_separation(self):
        """Test handling semicolon-separated statements."""
        self.session.run_code("x = 10")
        result = self.session.run_code('print("hello"); x')
        self.assertTrue(result.is_success())
        self.assertEqual(result.stdout.strip(), "hello")
        self.assertEqual(result.return_value, 10)

    def test_stderr_capture(self):
        """Test capturing messages sent to stderr."""
        code = """
import sys
sys.stderr.write("error message\\n")
        """
        result = self.session.run_code(code)
        self.assertTrue(result.is_success())
        self.assertEqual(result.stderr.strip(), "error message")

    def test_multiple_exceptions(self):
        """Test handling of multiple exceptions in a try/except block."""
        code = """
try:
    1/0
except ZeroDivisionError:
    undefined_var
        """
        result = self.session.run_code(code)
        self.assertFalse(result.is_success(), "Should fail after second exception.")
        assert "name 'undefined_var' is not defined" == result.exception


if __name__ == "__main__":
    unittest.main()
