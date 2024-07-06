from dotenv import load_dotenv
from openai import OpenAI

import numpy as np
import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import Any, Callable, Dict, List
import anthropic
import time
from functools import wraps

load_dotenv()


def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time

        if isinstance(result, dict):
            result["execution_time"] = execution_time
        else:
            print(f"Warning: {func.__name__} did not return a dictionary. Execution time: {execution_time:.4f} seconds")

        return result

    return wrapper


class BaseLLM:
    LLM_MAX_OUTER_TOOL_CALLS = 10
    LLM_MAX_INNER_TOOL_CALLS = 10

    def __init__(self, **kwargs):
        self.client = self._get_client(**kwargs)

    def _get_client(self, **kwargs) -> Any:
        raise NotImplementedError()

    def call(
        self,
        messages: List[Dict],
        model: str,
        **kwargs: Any,
    ) -> Dict:
        raise NotImplementedError()

    def _parse_tool_resp(self, resp: Dict) -> tuple:
        raise NotImplementedError()

    def _handle_final_tool_message(self, response_message: Any, new_messages: List) -> tuple:
        raise NotImplementedError()

    def _setup_tool_callables(self, messages, new_messages, response_message, tool_calls, functions_look_up) -> tuple:
        raise NotImplementedError()

    def _tool_loop(self, tool_calls, tool_results, tool_calls_details, messages, new_messages) -> tuple:
        raise NotImplementedError()

    @measure_time
    def generate_with_function_calling(
        self,
        messages: List[Dict],
        tools: List[Dict],
        functions_look_up: Dict[str, Callable[..., dict]],
        model: str,
        **kwargs: Any,
    ) -> Dict:
        messages = deepcopy(messages)
        calls = 1
        new_messages = []  # type: ignore
        tool_calls_details: Dict[str, dict] = dict()
        error_message = {
            "role": "assistant",
            "content": "Sorry, the question is too complex.",
        }
        too_complex_result = {
            "message": error_message,
            "new_messages": [error_message],
            "model": None,
            "tool_calls_details": tool_calls_details,
            "token_usage": None,
        }

        def call_tool(tool: Callable, tool_args: Dict) -> Any:
            return tool(**tool_args)

        while calls <= self.LLM_MAX_OUTER_TOOL_CALLS:
            resp = self.call(messages, model, tools=tools, **kwargs)
            if "error" in resp:
                return resp
            response_message, tool_calls = self._parse_tool_resp(resp)
            if len(tool_calls) > self.LLM_MAX_INNER_TOOL_CALLS:
                return too_complex_result
            if not tool_calls:
                final_message, new_messages = self._handle_final_tool_message(response_message, new_messages)
                return {
                    "message": {"content": final_message, "role": "assistant"},
                    "new_messages": new_messages,
                    "model": resp["model"],
                    "tool_calls_details": tool_calls_details,
                    "token_usage": resp["token_usage"],
                }
            messages, new_messages, tools_args_list, tools_callables = self._setup_tool_callables(
                messages, new_messages, response_message, tool_calls, functions_look_up
            )
            tasks = [(tools_callables[i], tools_args_list[i]) for i in range(len(tool_calls))]
            with ThreadPoolExecutor(max_workers=self.LLM_MAX_INNER_TOOL_CALLS) as executor:
                tool_results = list(executor.map(lambda p: call_tool(p[0], p[1]), tasks))
            tool_calls_details, messages, new_messages = self._tool_loop(tool_calls, tool_results, tool_calls_details, messages, new_messages)
            calls += 1
        return too_complex_result


class OpenAiLMM(BaseLLM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_client(self, **kwargs) -> OpenAI:
        return OpenAI(**kwargs)

    def call(
        self,
        messages: List[Dict],
        model: str = "gpt-3.5-turbo-0125",
        **kwargs: Any,
    ) -> Dict:
        try:
            resp = self.client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore
                stream=False,
                **kwargs,  # type: ignore
            )
        except Exception as e:
            return {
                "error": {
                    "code": e.code,  # type: ignore
                    "status_code": e.status_code,  # type: ignore
                    "type": e.type,  # type: ignore
                    "message": e.body["message"],  # type: ignore
                }
            }
        if kwargs.get("n", 1) > 1:
            raise ValueError("We only support n=1 for choices.")
        response_message = resp.choices[0].message  # type: ignore
        response_message = response_message.model_dump()
        # OpenAPI deprecated the field function_call but still returns it so remove it here
        response_message.pop("function_call", None)
        if response_message["tool_calls"] is None:
            response_message.pop("tool_calls")
        model = resp.model  # type: ignore
        token_usage = resp.usage.model_dump()  # type: ignore
        return {"message": response_message, "model": model, "token_usage": token_usage}

    def _parse_tool_resp(self, resp: Dict) -> tuple:
        response_message = resp["message"]
        tool_calls = response_message.get("tool_calls", [])
        return response_message, tool_calls

    def _handle_final_tool_message(self, response_message: Any, new_messages: List) -> tuple:
        final_message = response_message["content"]
        new_messages.append(response_message)
        return final_message, new_messages

    def _tool_loop(self, tool_calls, tool_results, tool_calls_details, messages, new_messages):
        for tool_call, tool_result in zip(tool_calls, tool_results):
            function_data = tool_result["data"]
            tool_info = {
                "tool_call_id": tool_call["id"],
                "role": "tool",
                "name": tool_call["function"]["name"],
                "content": str(function_data),
            }
            tool_calls_details[tool_call["id"]] = {
                "tool_result": tool_result,
                "id": tool_call["id"],
                "input": json.loads(tool_call["function"]["arguments"]),
                "name": tool_call["function"]["name"],
                "type": "tool_use",
            }
            messages.append(tool_info)
            new_messages.append(tool_info)
        return tool_calls_details, messages, new_messages

    def _setup_tool_callables(self, messages, new_messages, response_message, tool_calls, functions_look_up):
        messages.append(response_message)
        new_messages.append(response_message)

        tools_args_list = [json.loads(t["function"]["arguments"]) for t in tool_calls]
        tools_callables = [functions_look_up[t["function"]["name"]] for t in tool_calls]
        return messages, new_messages, tools_args_list, tools_callables

    def get_embeddings(self, texts, model="text-embedding-3-small"):
        def task(text):
            text = text.replace("\n", " ")
            return self.client.embeddings.create(input=[text], model=model).data[0].embedding

        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(task, texts))
        return np.array(results)


class AnthropicLLM(BaseLLM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_client(self, **kwargs) -> anthropic.Anthropic:
        return anthropic.Anthropic(**kwargs)

    def call(
        self,
        messages: List[Dict],
        model: str = "claude-3-5-sonnet-20240620",
        max_tokens=4000,
        **kwargs: Any,
    ) -> Dict:
        # put system prompt in OpenAI format
        if messages[0]["role"] == "system":
            system = messages[0]["content"]
            messages = messages[1:]
        else:
            system = []
        try:
            resp = self.client.messages.create(model=model, system=system, messages=messages, stream=False, max_tokens=max_tokens, **kwargs)  # type: ignore
            resp = resp.model_dump()  # type: ignore
        except Exception as e:
            return {
                "error": {
                    "code": e.body["error"]["type"],  # type: ignore
                    "status_code": e.status_code,  # type: ignore
                    "type": e.body["error"]["type"],  # type: ignore
                    "message": e.body["error"]["message"],  # type: ignore
                }
            }
        return {
            "message": {"content": resp["content"], "role": "assistant"},  # type: ignore
            "model": model,
            "token_usage": {
                "completion_tokens": resp["usage"]["output_tokens"],  # type: ignore
                "prompt_tokens": resp["usage"]["input_tokens"],  # type: ignore
                "total_tokens": resp["usage"]["output_tokens"] + resp["usage"]["input_tokens"],  # type: ignore
            },
        }

    def _parse_tool_resp(self, resp: Dict) -> tuple:
        response_messages = resp["message"]["content"]
        tool_calls = [m for m in response_messages if m["type"] == "tool_use"]
        return response_messages, tool_calls

    def _handle_final_tool_message(self, response_message: Any, new_messages: List) -> tuple:
        new_messages.append({"role": "assistant", "content": response_message})
        final_message = [m for m in response_message if m["type"] == "text"][0]["text"]
        return final_message, new_messages

    def _setup_tool_callables(self, messages, new_messages, response_message, tool_calls, functions_look_up):
        messages.append({"role": "assistant", "content": response_message})
        new_messages.append({"role": "assistant", "content": response_message})

        tools_args_list = [t["input"] for t in tool_calls]
        tools_callables = [functions_look_up[t["name"]] for t in tool_calls]
        return messages, new_messages, tools_args_list, tools_callables

    def _tool_loop(self, tool_calls, tool_results, tool_calls_details, messages, new_messages):
        user_response = {"role": "user", "content": []}
        for tool_call, tool_result in zip(tool_calls, tool_results):
            function_data = tool_result["data"]
            tool_info = {
                "tool_use_id": tool_call["id"],
                "type": "tool_result",
                "content": str(function_data),
            }
            user_response["content"].append(tool_info)  # type: ignore
            tool_calls_details[tool_call["id"]] = {"tool_result": tool_result, **tool_call}
        messages.append(user_response)
        new_messages.append(user_response)
        return tool_calls_details, messages, new_messages
