from python_sandbox import execute_python_code
from web_tools import visit_web_page, web_search

execute_python_code_tool = {
    "type": "function",
    "function": {
        "name": "execute_python_code",
        "description": "Run and execute the python code and return the results.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The python code to execute.",
                },
            },
            "required": ["code"],
        },
    },
}

web_search_tool = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for the query and return the results.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The query to search for."},
            },
            "required": ["query"],
        },
    },
}

visit_web_page_tool = {
    "type": "function",
    "function": {
        "name": "visit_web_page",
        "description": "Visit the web page and return the results.",
        "parameters": {"type": "object", "properties": {"url": {"type": "string", "description": "The URL to visit."}}, "required": ["url"]},
    },
}


TOOLS = [execute_python_code_tool, web_search_tool, visit_web_page_tool]

TOOL_LKP = {"web_search": web_search, "execute_python_code": execute_python_code, "visit_web_page": visit_web_page}
