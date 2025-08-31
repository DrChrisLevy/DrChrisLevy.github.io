# Intro to Modal Sandboxes

Here are the docs on Modal Sandboxes:

- [Sandboxes](https://modal.com/docs/guide/sandbox)
- [Running Commands in Sandboxes](https://modal.com/docs/guide/sandbox-spawn)
- [Networking and security](https://modal.com/docs/guide/sandbox-networking)
- [Filesystem Access](https://modal.com/docs/guide/sandbox-files)
- [Sanpshots](https://modal.com/docs/guide/sandbox-snapshots)
- [Memory Snapshots](https://modal.com/docs/guide/sandbox-memory-snapshots)

Here are some examples of using Modal Sandboxes:

- [Build a stateful, sandboxed code interpreter](https://modal.com/docs/examples/simple_code_interpreter#build-a-stateful-sandboxed-code-interpreter)
- [Run arbitrary code in a sandboxed environment](https://modal.com/docs/examples/safe_code_execution#run-arbitrary-code-in-a-sandboxed-environment)
- [Build a coding agent with Modal Sandboxes and LangGraph](https://modal.com/docs/examples/agent#build-a-coding-agent-with-modal-sandboxes-and-langgraph)
- [Run a Jupyter notebook in a Modal Sandbox](https://modal.com/docs/examples/jupyter_sandbox#run-a-jupyter-notebook-in-a-modal-sandbox)


**Question**:

*But what's the difference between using a Sandbox vs say a regular Modal function/container.*

**Answer**:

*The main difference is in a Function's access to other Modal APIs vs a Sandbox's locked down nature. A Modal Function can call other Functions, create Sandboxes, etc. A Sandbox cannot - it's just an isolated environment in which to execute code. So it prevents the case of an LLM going rouge and calling other Modal functions for example.
The goal is to limit the blast radius of malicious code, whether that's written by an LLM or a malicious user.*