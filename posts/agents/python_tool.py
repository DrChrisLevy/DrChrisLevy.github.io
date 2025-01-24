import ast
import io
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from typing import Any, Optional

import modal

image = modal.Image.debian_slim().pip_install("fastapi[standard]")
app = modal.App("python-tool", image=image)


@dataclass
class ExecutionResult:
    """Container for execution results"""

    stdout: str
    stderr: str
    traceback: Optional[str] = None
    exception: Optional[str] = None
    return_value: Optional[Any] = None

    def is_success(self) -> bool:
        """Check if the execution was successful"""
        return self.exception is None


class PythonSession:
    """
    Maintains an interactive Python session across multiple calls.
    Captures all output and maintains state between executions.
    """

    def __init__(self):
        # Single namespace for both globals and locals for persistence
        self._namespace = {}

    def run_code(self, code: str) -> ExecutionResult:
        """
        Executes code in the current session context and returns all outputs.

        Args:
            code: String containing Python code to execute

        Returns:
            ExecutionResult containing captured stdout, stderr, exception details,
            traceback, and optionally the last expression's return value.
        """
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        return_value = None
        try:
            # Parse the AST
            tree = ast.parse(code, mode="exec")

            # If the last node is an expression, pop it out for separate eval
            last_expr = None
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                last_expr = tree.body.pop()

            # Compile the modified tree (minus the last expression)
            mod = compile(tree, filename="<string>", mode="exec")

            # Execute everything with captured output
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                exec(mod, self._namespace, self._namespace)

                # If we had an expression, compile and eval it separately
                if last_expr is not None:
                    expr_code = compile(ast.Expression(last_expr.value), "<string>", mode="eval")
                    return_value = eval(expr_code, self._namespace, self._namespace)

            return ExecutionResult(stdout=stdout_buffer.getvalue(), stderr=stderr_buffer.getvalue(), return_value=return_value)

        except Exception as e:
            # Any exception here (syntax or runtime) is captured
            return ExecutionResult(
                stdout=stdout_buffer.getvalue(), stderr=stderr_buffer.getvalue(), exception=str(e), traceback=traceback.format_exc()
            )

    def reset(self):
        """Reset the session state completely"""
        self._namespace.clear()

    def get_state(self) -> dict:
        """Get a copy of the current session state."""
        return dict(self._namespace)

    def update_state(self, new_vars: dict):
        """Update the session with a dict of new variables."""
        self._namespace.update(new_vars)


@app.function()
@modal.web_endpoint(docs=True)
def run_python_code(code: str):
    session = PythonSession()
    result = session.run_code(code)
    return result
