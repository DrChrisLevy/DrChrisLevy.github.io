from dotenv import load_dotenv
from openai import OpenAI

import numpy as np

import json
from concurrent import futures
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, TypedDict, Union

load_dotenv()


class LLModelConfig(TypedDict):
    model: str
    base_url: Optional[str]
    api_key: Optional[str]


class LLModel:
    GPT4_0125 = LLModelConfig(model="gpt-4-0125-preview", base_url=None, api_key=None)
    GPT3_0125 = LLModelConfig(model="gpt-3.5-turbo-0125", base_url=None, api_key=None)


class OpenAICompatibleChatCompletion:
    clients: Dict = dict()
    LLM_MAX_OUTER_TOOL_CALLS = 10
    LLM_MAX_INNER_TOOL_CALLS = 10

    @classmethod
    def _load_client(cls, base_url: Optional[str] = None, api_key: Optional[str] = None) -> OpenAI:
        """
        By default, will connect directly to OpenAI API. However, there is the option to connect
        to endpoints which provide compatibility for the OpenAI API standard. For example, to
        connect to together.ai.
        """
        client_key = (base_url, api_key)
        if cls.clients.get(client_key) is None:
            cls.clients[client_key] = OpenAI(base_url=base_url, api_key=api_key)
        return cls.clients[client_key]

    @classmethod
    def call(
        cls,
        messages: List[Dict],
        model: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict:
        """
        For a LLM call without tool calls the end result should look something like:
        {
            'message': {'content': 'How can I help you today', 'role': 'assistant'},
            'model': 'gpt-3.5-turbo-0125'
        }

        For a LLM call with tool calls the end result should look something like:
        {'message': {'content': None,
          'role': 'assistant',
          'tool_calls': [{'function': {'arguments': '{"arg1": "value1"}', 'name': 'func_name'},
                                       'id': 'id1', 'type': 'function'}]}
         'model': 'gpt-3.5-turbo-0125'
        }

        If the LLM provider raises an Exception then we return a dict with the field 'error'.
        For example:
        {'error': {
                'code': 'context_length_exceeded',
                'status_code': 400,
                 'type': 'invalid_request_error',
                 'message': "This model's maximum context length is 16385 tokens."
                 }
        }
        """
        client = cls._load_client(base_url, api_key)
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=False,
                **kwargs,
            )
        except Exception as e:
            # handle particulars for each LLM model/provider.
            # OpenAI Particulars:
            return {
                "error": {
                    "code": e.code,
                    "status_code": e.status_code,
                    "type": e.type,
                    "message": e.body["message"],
                }
            }
        # handle particulars for each LLM model/provider.
        # OpenAI Particulars:
        response_message = resp.choices[0].message
        response_message = response_message.model_dump()
        # OpenAPI deprecated the field function_call but still returns it so remove it here
        response_message.pop("function_call", None)
        if response_message["tool_calls"] is None:
            response_message.pop("tool_calls")
        model = resp.model

        # return consistent format. We can add more keys/values in the future as we need them.
        # For example, token usage and so on.
        return {"message": response_message, "model": model}

    @classmethod
    def create_chat_completions_async(cls, task_args_list: List[Dict], concurrency: int = 10) -> List[Dict]:
        """
        Make a series of calls to chat.completions.create endpoint in parallel and collect back
        the results. Returns a list of dicts with the same format as the output of call()
        :param task_args_list: A list of dictionaries where each dictionary contains the keyword
            arguments required for call method.
        :param concurrency: the max number of workers
        """

        def create_chat_task(
            task_args: Dict,
        ) -> Union[Dict]:
            return cls().call(**task_args)

        with futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            results = list(executor.map(create_chat_task, task_args_list))
        return results

    @classmethod
    def generate_with_function_calling(
        cls,
        messages: List[Dict],
        tools: List[Dict],
        functions_look_up: Dict[str, Callable[..., dict]],
        model: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict:
        """Nested and parallel function calling.
        Parallel means that one single call can return multiple tool calls which can
        be called independently and in parallel. Nested means that we also attempt
        followup calls with additional tools calls if the LLM proposes that.

        {
                'message': {
                    'role': 'assistant',
                    'content': "Final message from the assistant.",
                },
                'new_messages': List of OpenAI compatible chat history dicts with the
                                messages and tool calls. It contains every **new** message
                                generated up to and including the final assistant message.
                'model': LLModel.GPT4_0125['model'],
                'tool_calls_details': A dictionary keyed by tool_id with details about each
                    tool call.
        }

        This code executes the tools/functions with the arguments
        and gets back the data and returns to the LLM. Each tool/function must return
        a dictionary with this minimal format:
        {
            'data': This is the data passed back to the LLM
        }
        The 'data' field will be used for the LLM to interpret the answer of the resulting
        tool call. Other keys/values can be passed in this dictionary and those details
        will be stored in the output dictionary 'tool_calls_details'.
        """
        messages = deepcopy(messages)
        calls = 1
        new_messages = []
        tool_calls_details: Dict[str, dict] = dict()
        error_message = {
            "role": "assistant",
            "content": "Sorry, the question is too complex.",  # TODO: work on copy
        }
        too_complex_result = {
            "message": error_message,
            "new_messages": [error_message],
            "show_your_work": [],
            "model": None,
            "tool_calls_details": tool_calls_details,
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
                }
            messages.append(response_message)
            new_messages.append(response_message)

            tools_args_list = [json.loads(t["function"]["arguments"]) for t in tool_calls]
            tools_callables = [functions_look_up[t["function"]["name"]] for t in tool_calls]
            tasks = [(tools_callables[i], tools_args_list[i]) for i in range(len(tool_calls))]

            with futures.ThreadPoolExecutor(max_workers=cls.LLM_MAX_INNER_TOOL_CALLS) as executor:
                tool_results = list(executor.map(lambda p: call_tool(p[0], p[1]), tasks))

            for tool_call, tool_result in zip(tool_calls, tool_results):
                function_data = tool_result["data"]
                tool_info = {
                    "tool_call_id": tool_call["id"],
                    "role": "tool",
                    "name": tool_call["function"]["name"],
                    "content": str(function_data),
                }
                tool_calls_details[tool_call["id"]] = {"tool_result": tool_result, **tool_call}
                messages.append(tool_info)
                new_messages.append(tool_info)
            calls += 1

        return too_complex_result

    def get_embeddings(self, texts, model="text-embedding-3-small"):
        def task(text):
            text = text.replace("\n", " ")
            return self.clients[(None, None)].embeddings.create(input=[text], model=model).data[0].embedding

        with futures.ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(task, texts))
        return np.array(results)
