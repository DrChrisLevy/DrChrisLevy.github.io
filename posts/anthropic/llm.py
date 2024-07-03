from dotenv import load_dotenv
from openai import OpenAI

import numpy as np
import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, TypedDict
import anthropic

load_dotenv()


class LLModelConfig(TypedDict):
    model: str
    base_url: Optional[str]
    api_key: Optional[str]


class LLModel:
    GPT4_0125 = LLModelConfig(model="gpt-4-0125-preview", base_url=None, api_key=None)
    GPT3_0125 = LLModelConfig(model="gpt-3.5-turbo-0125", base_url=None, api_key=None)
    SONNET_35 = LLModelConfig(model="claude-3-5-sonnet-20240620", base_url=None, api_key=None)


class OpenAICompatibleChatCompletion:
    clients: Dict = dict()
    LLM_MAX_OUTER_TOOL_CALLS = 10
    LLM_MAX_INNER_TOOL_CALLS = 10

    @classmethod
    def _load_client(cls, base_url: Optional[str] = None, api_key: Optional[str] = None) -> OpenAI:
        client_key = (base_url, api_key)
        if cls.clients.get(client_key) is None:
            cls.clients[client_key] = OpenAI(base_url=base_url, api_key=api_key)
        return cls.clients[client_key]

    @classmethod
    def call(
        cls,
        messages: List[Dict],
        model: str = LLModel.GPT3_0125["model"],
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict:
        client = cls._load_client(base_url, api_key)
        try:
            resp = client.chat.completions.create(
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
        response_message = resp.choices[0].message  # type: ignore
        response_message = response_message.model_dump()
        # OpenAPI deprecated the field function_call but still returns it so remove it here
        response_message.pop("function_call", None)
        if response_message["tool_calls"] is None:
            response_message.pop("tool_calls")
        model = resp.model  # type: ignore
        token_usage = resp.usage.model_dump()  # type: ignore
        return {"message": response_message, "model": model, "token_usage": token_usage}

    @classmethod
    def generate_with_function_calling(
        cls,
        messages: List[Dict],
        tools: List[Dict],
        functions_look_up: Dict[str, Callable[..., dict]],
        model: str = LLModel.GPT3_0125["model"],
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict:
        messages = deepcopy(messages)
        calls = 1
        new_messages = []
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

        while calls <= cls.LLM_MAX_OUTER_TOOL_CALLS:
            resp = cls.call(messages, model, base_url, api_key, tools=tools, **kwargs)
            if "error" in resp:
                return resp
            response_message = resp["message"]
            tool_calls = response_message.get("tool_calls", [])
            if len(tool_calls) > cls.LLM_MAX_INNER_TOOL_CALLS:
                return too_complex_result
            if not tool_calls:
                new_messages.append(response_message)
                return {
                    "message": response_message,
                    "new_messages": new_messages,
                    "model": resp["model"],
                    "tool_calls_details": tool_calls_details,
                    "token_usage": resp["token_usage"],
                }
            messages.append(response_message)
            new_messages.append(response_message)

            tools_args_list = [json.loads(t["function"]["arguments"]) for t in tool_calls]
            tools_callables = [functions_look_up[t["function"]["name"]] for t in tool_calls]
            tasks = [(tools_callables[i], tools_args_list[i]) for i in range(len(tool_calls))]

            with ThreadPoolExecutor(max_workers=cls.LLM_MAX_INNER_TOOL_CALLS) as executor:
                tool_results = list(executor.map(lambda p: call_tool(p[0], p[1]), tasks))

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
            calls += 1

        return too_complex_result

    def get_embeddings(self, texts, model="text-embedding-3-small"):
        def task(text):
            text = text.replace("\n", " ")
            return self.clients[(None, None)].embeddings.create(input=[text], model=model).data[0].embedding

        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(task, texts))
        return np.array(results)


class AnthropicLLM:
    client = None
    LLM_MAX_OUTER_TOOL_CALLS = 10
    LLM_MAX_INNER_TOOL_CALLS = 10

    @classmethod
    def _load_client(cls) -> anthropic.Anthropic:
        if cls.client is None:
            return anthropic.Anthropic()

    @classmethod
    def call(
        cls,
        messages: List[Dict],
        model: str = LLModel.SONNET_35["model"],
        max_tokens=4000,
        **kwargs: Any,
    ) -> Dict:
        client = cls._load_client()
        # put system prompt in OpenAI format
        if messages[0]["role"] == "system":
            system = messages[0]["content"]
            messages = messages[1:]
        else:
            system = []
        try:
            resp = client.messages.create(model=model, system=system, messages=messages, stream=False, max_tokens=max_tokens, **kwargs)  # type: ignore
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

    @classmethod
    def generate_with_function_calling(
        cls,
        messages: List[Dict],
        tools: List[Dict],
        functions_look_up: Dict[str, Callable[..., dict]],
        model: str = "claude-3-5-sonnet-20240620",
        **kwargs: Any,
    ) -> Dict:
        messages = deepcopy(messages)
        calls = 1
        new_messages = []
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

        def call_tool(tool, tool_args):
            return tool(**tool_args)

        while calls <= cls.LLM_MAX_OUTER_TOOL_CALLS:
            resp = cls.call(messages, model, tools=tools, **kwargs)
            if "error" in resp:
                return resp
            response_messages = resp["message"]["content"]
            tool_calls = [m for m in response_messages if m["type"] == "tool_use"]

            if len(tool_calls) > cls.LLM_MAX_INNER_TOOL_CALLS:
                return too_complex_result
            if not tool_calls:
                new_messages.append({"role": "assistant", "content": response_messages})
                final_message = [m for m in response_messages if m["type"] == "text"][0]["text"]
                return {
                    "message": {"content": final_message, "role": "assistant"},
                    "new_messages": new_messages,
                    "model": resp["model"],
                    "tool_calls_details": tool_calls_details,
                    "token_usage": resp["token_usage"],
                }
            messages.append({"role": "assistant", "content": response_messages})
            new_messages.append({"role": "assistant", "content": response_messages})

            tools_args_list = [t["input"] for t in tool_calls]
            tools_callables = [functions_look_up[t["name"]] for t in tool_calls]
            tasks = [(tools_callables[i], tools_args_list[i]) for i in range(len(tool_calls))]

            with ThreadPoolExecutor(max_workers=cls.LLM_MAX_INNER_TOOL_CALLS) as executor:
                tool_results = list(executor.map(lambda p: call_tool(p[0], p[1]), tasks))

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
            calls += 1

        return too_complex_result
