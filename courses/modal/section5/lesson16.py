"""
Note: this could change in the future as modal improves.
    But at the time of writing this, the stdout channel does not work well
    with from_id at the moment. That is why we make use of the file system.

    This module implements a file-based code execution driver in a Modal sandbox.
    This module was specifically designed to support detached execution. This means
    that you can pass around the Sandbox's object ID and control the same process
    from a different process later.
    It reads commands from '/modal/io/stdin.txt'; each JSON command must include
    a "code" field and a user-supplied "command_id". The execution output (stdout and stderr)
    is written to '/modal/io/<command_id>.txt'.

    Based off this GIST from Peyton (Modal Developer)
    https://gist.github.com/pawalt/7cd4dc56de29e9cddba4d97decaab1ad
"""

import json
import os
import time
from typing import Any, Dict, Optional
from uuid import uuid4

import modal

DRIVER_PROGRAM = """
import json
import os
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from typing import Any, Generator

IO_DATA_DIR = '/modal/io'
os.makedirs(IO_DATA_DIR, exist_ok=True)
STDIN_FILE = os.path.join(IO_DATA_DIR, 'stdin.txt')

with open(STDIN_FILE, 'w') as f:
    f.write('')


def tail_f(filename: str) -> Generator[str, None, None]:
    # Continuously yields new lines from the file.
    with open(filename, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.1)
                continue
            yield line


globals: dict[str, Any] = {}
for line in tail_f(STDIN_FILE):
    line = line.strip()
    print(f'Received line: {line} len: {len(line)}')
    if not line:
        continue

    command = json.loads(line)
    if (code := command.get('code')) is None:
        print(json.dumps({'error': 'No code to execute'}))
        continue

    if (command_id := command.get('command_id')) is None:
        print(json.dumps({'error': 'No command_id'}))
        continue

    stdout_io, stderr_io = StringIO(), StringIO()
    with redirect_stdout(stdout_io), redirect_stderr(stderr_io):
        try:
            exec(code, globals)
        except Exception as e:
            print(f'{type(e).__name__}: {e}', file=sys.stderr)

    with open(os.path.join(IO_DATA_DIR, f'{command_id}.txt'), 'w') as f:
        f.write(
            json.dumps(
                {
                    'stdout': stdout_io.getvalue(),
                    'stderr': stderr_io.getvalue(),
                }
            )
        )
"""


class ModalSandbox:
    IMAGE = modal.Image.debian_slim().pip_install("pandas", "tabulate")
    IO_DATA_DIR = "/modal/io"
    STDIN_FILE = os.path.join(IO_DATA_DIR, "stdin.txt")

    def __init__(
        self,
        sandbox_id: Optional[str] = None,
        timeout: int = 10 * 60,
        init_script: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        # check if running Sandbox already exists
        if sandbox_id is not None:
            existing_sb = self._get_running_sandbox_from_id(sandbox_id)
            if existing_sb is not None:
                self.sandbox = existing_sb
                return

        app = modal.App.lookup("python-sandbox", create_if_missing=True)
        self.sandbox = modal.Sandbox.create(
            "python",
            "-c",
            DRIVER_PROGRAM,
            image=self.IMAGE,
            app=app,
            timeout=timeout,
            **kwargs,
        )
        if init_script:
            self.run_code(init_script)

    @classmethod
    def _get_running_sandbox_from_id(cls, sb_id: str) -> Optional[modal.Sandbox]:
        # Returns None if the sandbox is not running or if the sb_id is not found
        # or some error occurs
        try:
            sb = modal.Sandbox.from_id(sb_id)
        except Exception:
            return None
        # check if the sandbox is running
        if sb.poll() is None:
            return sb

        return None

    @property
    def sandbox_id(self) -> str:
        return self.sandbox.object_id

    def terminate(self) -> None:
        self.sandbox.terminate()

    def run_code(self, code: str) -> Dict[str, str]:
        command_id = uuid4().hex

        # 1. Write code into a STDIN file on the sandbox.
        with self.sandbox.open(self.STDIN_FILE, "a") as f:
            f.write(json.dumps({"code": code, "command_id": command_id}))
            f.write("\n")

        # 2. The sandbox polls this STDIN file for changes,
        # executes the added code, then saves the output to a file.
        out_file = os.path.join(self.IO_DATA_DIR, f"{command_id}.txt")

        # 3. We poll the Sandbox to check if it has created the output file,
        # and if so, return the output from the file.
        while True:
            try:
                with self.sandbox.open(out_file, "r") as f:
                    result = json.load(f)
                    return result
            except FileNotFoundError:
                time.sleep(0.1)
