import json
from concurrent import futures
from typing import Any, Callable, Dict

from litellm import completion


def call_tool(tool: Callable, tool_args: Dict) -> Any:
    return tool(**tool_args)


def run_step(messages, tools=None, tools_lookup=None, model="gpt-4o-mini", **kwargs):
    response = completion(model=model, messages=messages, tools=tools, **kwargs)
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


def llm_with_tools(messages, tools=None, tools_lookup=None, model="gpt-4o-mini", max_steps=10, **kwargs):
    done_calling_tools = False
    for counter in range(max_steps):
        print(f"ITERATION {counter=}")
        messages = run_step(messages, tools, tools_lookup, model="gpt-4o-mini", **kwargs)
        done_calling_tools = messages[-1]["role"] == "assistant" and messages[-1].get("content") and not messages[-1].get("tool_calls")
        if done_calling_tools:
            break
    return messages
