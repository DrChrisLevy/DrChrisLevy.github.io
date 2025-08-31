# Adding a FrontEnd to our Simple Agent with Coding Abilities

In this lesson I take our simple agent with coding abilities and add a frontend to it
using FastHTML and MonsterUI. I did this simply by vibe coding and providing some context
to Claude Sonnet in Cursor. I did not review much of the code, but it did take some iterations.
The point is not the code. The main takeaway is that you 
can build little interactive web applications to visualize LLM traces and outputs.

- [fasthtml](https://www.fastht.ml/)
- [monsterui](https://monsterui.answer.ai/)


You can put this code in a file such as `agent_app.py` and run it with `python agent_app.py`.
You would need to install `python-fasthtml` and `monsterui` into your environment.

```python
# ruff: noqa: F403, F405
"""
This code was written by Claude Sonnet in Cursor.
It's good for visualizing the agent's traces and progress.
"""

import ast
import json
import uuid
from concurrent import futures
from threading import Thread
from typing import Any, Callable, Dict

from fasthtml.common import *
from monsterui.all import *
from openai import OpenAI

from coding_sandbox import ModalSandbox

hdrs = (
    Theme.neutral.headers(highlightjs=True),
    Script(src="https://unpkg.com/htmx-ext-sse@2.2.1/sse.js"),
)
app, rt = fast_app(
    hdrs=hdrs,
    live=True,
)

# Agent code copied from simple_agent_from_scratch.py
SYSTEM_PROMPT = """
# Role and Objective

You are a helpful AI Agent that can call tools to assist in solving problems for the user.

# Agent Instructions

- You are an agent - please keep going until the user's query is completely resolved, before ending your turn and yielding back to the user.
Only terminate your turn when you are sure that the problem is solved.
- If you are not sure about something pertaining to the user's request, use your tools to gather the relevant information. Do not guess or make up an answer.
- You must plan extensively before each action and/or function call, and reflect on the outcomes of the previous actions and/or function calls. Do not do this entire process by making actions/function calls only, as this can impair your ability to solve the problem and think insightfully.
- Always begin by thinking about what needs to be done before using any tools.
- Observe the output of tools to inform your next steps.
- Repeat this agentic loop of thinking, tool use, and observing the output until you've solved the problem completely or if you need to ask the user for more information.
- If you encounter any errors fix them along the way and continue the loop.

# Python Environment Instructions

- You have access to a python sandbox environment that you can use to write code and call functions to assist with your task from the user.
- The python sandbox environment is safe, isolated, and supports running arbitrary Python code.
- State is maintained between code snippets. Variables and definitions persist across executions.
- In addition to the Python Standard Library, pandas is already installed in the environment.
- If modules are not found, install them using: os.system('pip install <package_name>')
- Remember to use print() on any code that you need to see the output of.
- String Formatting Requirements:
   - All print statements must use double backslashes for escape characters
   - Example: `print("\\\\nHello")` instead of `print("\\nHello")`
   - This applies to all string literals containing `\\n`, `\\r`, `\\t` etc.
   - This is required to prevent string termination errors in the sandbox
- One handy trick is to convert data frames to markdown using the df.to_markdown() method.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_code",
            "description": """Execute the given Python code in
                a sandboxed environment.
                print() any code that you need to see the output of.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python code to execute.",
                    },
                },
                "required": [
                    "code",
                ],
            },
        },
    }
]

SAND_BOX = ModalSandbox()


def execute_code(code: str):
    return SAND_BOX.run_code(code)


TOOLS_LOOKUP = {
    "execute_code": execute_code,
}

client = OpenAI()

# In-memory storage for agent sessions
agent_sessions = {}


def call_tool(tool: Callable, tool_args: Dict) -> Any:
    return tool(**tool_args)


def run_step(messages, tools, tools_lookup, model="gpt-4.1"):
    messages = messages.copy()
    response = client.chat.completions.create(
        model=model, messages=messages, tools=tools
    )
    response_message = response.choices[0].message.model_dump()
    response_message.pop("function_call", None)
    tool_calls = response_message.get("tool_calls", [])
    messages.append(response_message)

    if not tool_calls:
        response_message.pop("tool_calls", None)
        return messages

    tools_args_list = [json.loads(t["function"]["arguments"]) for t in tool_calls]
    tools_callables = [tools_lookup[t["function"]["name"]] for t in tool_calls]
    tasks = [(tools_callables[i], tools_args_list[i]) for i in range(len(tool_calls))]
    with futures.ThreadPoolExecutor(max_workers=10) as executor:
        tool_results = list(executor.map(lambda p: call_tool(p[0], p[1]), tasks))
    for tool_call, tool_result in zip(tool_calls, tool_results):
        messages.append(
            {
                "tool_call_id": tool_call["id"],
                "role": "tool",
                "content": str(tool_result),
                "name": tool_call["function"]["name"],
            }
        )
    return messages


def llm_with_tools_traced(session_id, messages, tools, tools_lookup, max_steps=50):
    """Modified version that updates session state after each step"""
    for step in range(max_steps):
        messages = run_step(messages, tools, tools_lookup)

        # Update session state after each step
        agent_sessions[session_id] = {
            "messages": messages,
            "completed": False,
            "rendered_count": agent_sessions[session_id].get("rendered_count", 0),
        }

        # Stop if assistant replied without tool calls
        if (
            messages[-1]["role"] == "assistant"
            and messages[-1].get("content")
            and not messages[-1].get("tool_calls")
        ):
            break

    # Mark as completed
    agent_sessions[session_id]["completed"] = True
    return messages


def format_message(msg):
    """Format a message for display"""
    role = msg.get("role", "unknown")
    content = msg.get("content", "")

    if role == "system":
        return Div(
            H4("🔧 System", cls="text-sm font-semibold text-slate-600"),
            render_md(content[:200] + "..." if len(content) > 200 else content),
            cls="p-4 bg-slate-50 rounded-lg mb-4",
        )
    elif role == "user":
        return Div(
            H4("👤 User", cls="text-sm font-semibold text-blue-600"),
            render_md(content),
            cls="p-4 bg-blue-50 rounded-lg mb-4",
        )
    elif role == "assistant":
        tool_calls = msg.get("tool_calls", [])
        parts = [H4("🤖 Assistant", cls="text-sm font-semibold text-green-600")]

        if content:
            parts.append(render_md(content))

        if tool_calls:
            parts.append(H5("Tool Calls:", cls="text-sm font-semibold mt-2"))
            for tool_call in tool_calls:
                # Since we only have execute_code, simplify
                code = json.loads(tool_call["function"]["arguments"])["code"]
                parts.append(
                    Div(
                        P("🔧 execute_code"),
                        render_md(f"```python\n{code}\n```"),
                        cls="ml-4 p-2 bg-gray-100 rounded",
                    )
                )

        return Div(*parts, cls="p-4 bg-green-50 rounded-lg mb-4")

    elif role == "tool":
        # Simplified - we know it's always execute_code results
        try:
            result_dict = ast.literal_eval(content)
            stdout = result_dict.get("stdout", "")
            stderr = result_dict.get("stderr", "")

            parts = [H4("⚙️ Tool Result", cls="text-sm font-semibold text-purple-600")]

            if stdout:
                parts.extend(
                    [
                        H5("Output:", cls="text-sm font-semibold mt-2 text-green-600"),
                        render_md(f"```\n{stdout}\n```"),
                    ]
                )

            if stderr:
                parts.extend(
                    [
                        H5("Errors:", cls="text-sm font-semibold mt-2 text-red-600"),
                        render_md(f"```\n{stderr}\n```"),
                    ]
                )

            return Div(*parts, cls="p-4 bg-purple-50 rounded-lg mb-4")
        except:
            # Simple fallback
            return Div(
                H4("⚙️ Tool Result", cls="text-sm font-semibold text-purple-600"),
                render_md(f"```\n{content}\n```"),
                cls="p-4 bg-purple-50 rounded-lg mb-4",
            )

    # We only handle known roles in this demo
    return Div()


@rt("/")
def get():
    return Container(
        H1("🤖 AI Agent Traces", cls="text-3xl font-bold mb-6"),
        P(
            "Enter a question and watch the agent work step by step!",
            cls="text-gray-600 mb-6",
        ),
        Form(
            Div(
                Textarea(
                    name="question",
                    placeholder="Ask the agent anything...",
                    rows=4,
                    required=True,
                    cls="w-full",
                ),
                cls="mb-4",
            ),
            Button("Run Agent", type="submit", cls=ButtonT.primary),
            hx_post="/run_agent",
            hx_target="#traces",
            hx_swap="innerHTML",
        ),
        Div(id="traces", cls="mt-8"),
    )


@rt("/run_agent")
def post(question: str):
    session_id = str(uuid.uuid4())

    # Initialize session
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    agent_sessions[session_id] = {
        "messages": messages,
        "completed": False,
        "rendered_count": 0,
    }

    # Start agent in background thread
    def run_agent():
        llm_with_tools_traced(session_id, messages, TOOLS, TOOLS_LOOKUP)

    Thread(target=run_agent, daemon=True).start()

    # Return initial messages and start polling for new ones
    initial_traces = [format_message(msg) for msg in messages]

    # Mark initial messages as already rendered
    agent_sessions[session_id]["rendered_count"] = len(messages)

    return Div(
        H2("🔄 Agent Working...", cls="text-xl font-semibold mb-4"),
        Div(*initial_traces, id="trace-content"),
        Div(
            id="status-indicator",
            hx_get=f"/traces/{session_id}",
            hx_trigger="every 1s",
            hx_target="#trace-content",
            hx_swap="beforeend",
        ),
    )


@rt("/traces/{session_id}")
def get_traces(session_id: str):
    session = agent_sessions.get(session_id)
    if not session:
        return ""

    messages = session["messages"]
    rendered_count = session["rendered_count"]
    completed = session["completed"]

    # Only format NEW messages that haven't been rendered yet
    new_messages = messages[rendered_count:]
    new_trace_divs = [format_message(msg) for msg in new_messages]

    # Update rendered count
    agent_sessions[session_id]["rendered_count"] = len(messages)

    if completed:
        # Add completion message and stop polling
        new_trace_divs.append(
            Div(
                "✅ Agent completed!",
                cls="p-4 bg-green-100 text-green-800 rounded-lg mt-4 font-semibold",
            )
        )
        # Remove the polling element
        new_trace_divs.append(
            Script("document.getElementById('status-indicator').remove();")
        )
        return Div(*new_trace_divs)
    else:
        # Just return new messages, polling continues
        return Div(*new_trace_divs)


serve()
```