import json

import modal

# Create image with IPython installed
image = modal.Image.debian_slim().pip_install("ipython")


# Create the driver program that will run in the sandbox
def create_driver_program():
    return """
import json
import sys
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.io import capture_output

# Create a persistent IPython shell instance
shell = InteractiveShell()

# Keep reading commands from stdin
while True:
    try:
        # Read a line of JSON from stdin
        command = json.loads(input())
        code = command.get('code')
        
        if code is None:
            print(json.dumps({"error": "No code provided"}))
            continue
            
        # Execute the code and capture output
        with capture_output() as captured:
            result = shell.run_cell(code)
            
        # Format the response
        response = {
            "stdout": captured.stdout,
            "stderr": captured.stderr,
            "success": result.success,
            "result": repr(result.result) if result.success else None,
            "error": str(result.error_in_exec) if not result.success else None
        }
        
        # Send the response
        print(json.dumps(response), flush=True)
        
    except Exception as e:
        print(json.dumps({"error": str(e)}), flush=True)
"""


def create_sandbox():
    """Creates and returns a Modal sandbox running an IPython shell."""
    app = modal.App.lookup("ipython-sandbox", create_if_missing=True)

    # Create the sandbox with the driver program
    with modal.enable_output():
        sandbox = modal.Sandbox.create("python", "-c", create_driver_program(), image=image, app=app)

    return sandbox


def run_code(sandbox, code: str) -> dict:
    """
    Runs code in the persistent IPython shell.

    Args:
        sandbox: Modal sandbox instance
        code: Python code to execute

    Returns:
        dict with execution results
    """
    # Send the code to the sandbox
    sandbox.stdin.write(json.dumps({"code": code}))
    sandbox.stdin.write("\n")
    sandbox.stdin.drain()

    # Get the response
    response = next(iter(sandbox.stdout))
    return json.loads(response)


if __name__ == "__main__":
    # Create the sandbox
    sandbox = create_sandbox()

    try:
        # Example usage
        print("Testing sequential execution with persistent state:")
        test_codes = ["x = 42", "y = x * 2", "print(f'x = {x}, y = {y}')", "result = x + y\nresult"]

        for code in test_codes:
            print("\nExecuting:", code)
            result = run_code(sandbox, code)
            print("Success:", result["success"])
            if result["stdout"]:
                print("Stdout:", result["stdout"].rstrip())
            if result["result"]:
                print("Return value:", result["result"])
            if result["error"]:
                print("Error:", result["error"])

    finally:
        # Clean up
        sandbox.terminate()
