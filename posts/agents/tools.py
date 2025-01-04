import re

import requests
from duckduckgo_search import DDGS
from markdownify import markdownify
from python_sandbox import execute_python_code
from requests.exceptions import RequestException

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


def web_search(query: str) -> str:
    with DDGS() as ddgs:
        return ddgs.text(query, max_results=5)


def visit_web_page(url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        # Convert to markdown with custom options
        markdown_content = markdownify(
            response.text,
            heading_style="ATX",  # Use # style headings
        ).strip()

        # Clean up excessive whitespace
        markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)

        return markdown_content

    except RequestException as e:
        return f"Error fetching the webpage: {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"


TOOLS = [execute_python_code_tool, web_search_tool, visit_web_page_tool]

TOOL_LKP = {"web_search": web_search, "execute_python_code": execute_python_code, "visit_web_page": visit_web_page}
