import re

import requests
from duckduckgo_search import DDGS
from markdownify import markdownify
from requests.exceptions import RequestException


def web_search(query: str) -> str:
    with DDGS() as ddgs:
        return ddgs.text(query, max_results=5)


def visit_web_page(url, max_chars: int = 20000, timeout: int = 10):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()

        # Convert to markdown with custom options
        markdown_content = markdownify(
            response.text,
            heading_style="ATX",  # Use # style headings
        ).strip()

        # Clean up excessive whitespace
        markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)

        # Truncate content if it exceeds max_chars
        if len(markdown_content) > max_chars:
            markdown_content = markdown_content[:max_chars] + "\n\n... (content truncated)"

        return markdown_content

    except requests.Timeout:
        return f"Error: Request timed out after {timeout} seconds"
    except RequestException as e:
        return f"Error fetching the webpage: {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"
