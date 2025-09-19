# Building a Simple Agent with Coding Abilities

In this lesson we will build a simple LLM agent that operates in a loop
with function calling. For simplicity we will use the OpenAI API
and function/tool calling. One of the functions we will give the LLM
is the coding sandbox function we built in the previous lessons.

The word "agent" is a loaded term in the LLM world.
I wrote a couple blog posts on this topic in the past
which may be useful to provide some context:

- [my blog](https://drchrislevy.github.io/blog)
    - **Agents - Part 1** has a good overview of the topic as well as external links
- [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)

- You will need an OpenAI API key to run this code. Head over to OpenAI to create an account
and create an API key [here](https://platform.openai.com/api-keys). If you are using another
LLM provider that is OpenAI compatible then that might work as well. For example,
you could use [litellm](https://docs.litellm.ai/) which allows you to use many LLMs
through an OpenAI compatible API.

- Put the `OPENAI_API_KEY` in your environment within the `.env` file.
- Here is the [GPT-4.1 Prompting Guide](https://cookbook.openai.com/examples/gpt4-1_prompting_guide)
which informs some of logic in my system prompt down below.
- Create a new python file called `simple_agent_from_scratch.py` and add the following code to it.
- make sure you still have the `coding_sandbox.py` file in the same directory as this file from the previous lessons
as we will be importing it in the file, `simple_agent_from_scratch.py`.

```python
import json
from concurrent import futures
from typing import Any, Callable, Dict

from openai import OpenAI

from coding_sandbox import ModalSandbox

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


def call_tool(tool: Callable, tool_args: Dict) -> Any:
    return tool(**tool_args)


def run_step(messages, tools=None, tools_lookup=None, model="gpt-4.1", **kwargs):
    messages = messages.copy()
    response = client.chat.completions.create(
        model=model, messages=messages, tools=tools, **kwargs
    )
    response_message = response.choices[0].message.model_dump()
    response_message.pop("function_call", None)  # deprecated field in OpenAI API
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


def llm_with_tools(
    messages, tools=None, tools_lookup=None, model="gpt-4.1", max_steps=10, **kwargs
):
    done_calling_tools = False
    for _ in range(max_steps):
        messages = run_step(messages, tools, tools_lookup, model=model, **kwargs)
        done_calling_tools = (
            messages[-1]["role"] == "assistant"
            and messages[-1].get("content")
            and not messages[-1].get("tool_calls")
        )
        if done_calling_tools:
            break
    return messages


if __name__ == "__main__":
    # Example
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": "What is the capital of the United States?",
        },
    ]
    messages = llm_with_tools(messages, tools=TOOLS, tools_lookup=TOOLS_LOOKUP)
    print(messages)

    # Example
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": "What is the square root of 5. Get the exact answer to 10 decimal places.",
        },
    ]
    messages = llm_with_tools(messages, tools=TOOLS, tools_lookup=TOOLS_LOOKUP)
    print(messages)

    # Example
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": """
            I want you to perform a comprehensive stock market analysis for Apple (AAPL) and Tesla (TSLA). 
            Please do the following:
            
            1. Download the last 2 months of stock data for both companies
            2. Calculate key technical indicators (moving averages, RSI, MACD)
            3. Perform a correlation analysis between the two stocks
            4. Use machine learning to predict next week's price direction
            5. Provide investment insights and recommendations
            
            Make sure to install any required libraries and handle any errors you encounter.
            """,
        },
    ]

    messages = llm_with_tools(
        messages,
        tools=TOOLS,
        tools_lookup=TOOLS_LOOKUP,
        max_steps=30,  # Allow more steps for complex analysis
    )
    print(messages)

```


This is just a small minimal proof of concept. 
There is so much you could do with this if your are interested in expanding on this.

- Build out a chatbot that can use this agent to answer questions
- Save conversations and messages to some sort of storage such as Modal volumes or Modal Dictionaries (key-value store)
- Build a simple frontend to view the agent traces for debugging and evaluation 
- Add more tools to the agent to make it more powerful
- Try plugging in the coding tool to another Agent Library such as OpenAI Agents SDK