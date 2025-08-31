import argparse
import json
import re
import sys
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


def scrape_modal_docs(url):
    """
    Scrape Modal documentation and convert to markdown

    Args:
        url: The URL of the Modal documentation page

    Returns:
        Markdown string of the documentation content
    """
    # Fetch the page
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}", file=sys.stderr)
        return ""

    # Parse the HTML
    soup = BeautifulSoup(response.text, "html.parser")

    # Modal's docs appear to be a Next.js app, try several strategies to find the content
    main_content = find_main_content(soup)

    # Extract the content as markdown
    if main_content:
        markdown = process_element(main_content, url)
        # Clean up the markdown
        markdown = post_process_markdown(markdown)
        return markdown
    else:
        return "Could not extract content from the page."


def find_main_content(soup):
    """Find the main content element in Modal docs"""
    # Strategy 1: Look for article or main elements
    main_content = soup.find("article") or soup.find("main")
    if main_content:
        return main_content

    # Strategy 2: Look for divs with content-related class names
    content_classes = ["content", "docs-content", "markdown", "main-content"]
    for cls in content_classes:
        content = soup.find("div", class_=re.compile(cls, re.I))
        if content:
            return content

    # Strategy 3: Look for a div that contains the main heading
    headings = soup.find_all(
        ["h1", "h2", "h3"],
        string=re.compile("^Introduction$|^Getting Started$|^Overview$", re.I),
    )
    for heading in headings:
        # Find parent container with multiple headers
        parent = heading.parent
        while parent and parent.name != "body":
            if parent.name == "div" and len(parent.find_all(["h1", "h2", "h3"])) > 2:
                return parent
            parent = parent.parent

    # Strategy 4: Look for the largest div with the most text and highest heading density
    divs = soup.find_all("div")
    if divs:
        # Calculate content score based on text length and heading count
        div_scores = []
        for div in divs:
            text_length = len(div.get_text())
            heading_count = len(div.find_all(["h1", "h2", "h3", "h4"]))
            if (
                text_length > 200 and heading_count > 0
            ):  # Only consider reasonably sized content
                div_scores.append((div, text_length * heading_count))

        if div_scores:
            div_scores.sort(key=lambda x: x[1], reverse=True)
            return div_scores[0][0]

    # If we couldn't find anything else, just return the body
    return soup.body


def process_element(element, base_url, depth=0, context=None):
    """
    Process an HTML element and its children recursively to extract markdown content.

    Args:
        element: BeautifulSoup element to process
        base_url: Base URL for resolving relative links
        depth: Current nesting depth (for debugging)
        context: Context information for nested elements

    Returns:
        Markdown string
    """
    # Initialize context if not provided
    if context is None:
        context = {"in_block": False, "list_type": None}

    # Skip common navigation or sidebar elements
    if element.name and element.has_attr("class"):
        # Skip elements that are likely navigation, sidebars, etc.
        skip_classes = ["nav", "sidebar", "menu", "navigation", "toc", "footer"]
        element_classes = element.get("class", [])
        if any(
            cls
            for cls in element_classes
            if any(skip in cls.lower() for skip in skip_classes)
        ):
            return ""

    # Process the element based on its type
    if element.name is None:  # Text node
        text = element.string
        if text and text.strip():
            return text.strip() + " "
        return ""

    # Handle different HTML elements
    if element.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
        level = int(element.name[1])
        text = element.get_text().strip()
        return f"\n\n{'#' * level} {text}\n\n"

    elif element.name == "p":
        content = process_children(element, base_url, depth, context)
        if not content.strip():
            return ""
        return f"\n\n{content}\n\n"

    elif element.name == "a":
        href = element.get("href", "")
        text = element.get_text().strip()
        if href and text:
            # Create full URL if it's a relative link
            if not href.startswith(("http://", "https://")):
                href = urljoin(base_url, href)
            return f"[{text}]({href})"
        return process_children(element, base_url, depth, context)

    elif element.name in ["ul", "ol"]:
        old_list_type = context.get("list_type")
        context["list_type"] = element.name
        result = f"\n\n{process_list(element, base_url, depth, context)}\n\n"
        context["list_type"] = old_list_type
        return result

    elif element.name == "li":
        prefix = "- "
        if context.get("list_type") == "ol":
            prefix = "1. "  # Markdown will auto-number
        content = process_children(element, base_url, depth, context)
        return f"{prefix}{content}\n"

    elif element.name == "pre":
        # Handle code blocks
        code_elem = element.find("code")
        language = ""

        if code_elem and code_elem.has_attr("class"):
            for cls in code_elem.get("class", []):
                if cls.startswith("language-"):
                    language = cls.replace("language-", "")
                    break

        # Use the code element's text if available, otherwise use the pre element's text
        if code_elem:
            code = code_elem.get_text()
        else:
            code = element.get_text()

        # Clean up the code by removing common leading whitespace
        code = textwrap_dedent(code)

        return f"\n\n```{language}\n{code}\n```\n\n"

    elif element.name == "code":
        # Handle inline code
        text = element.get_text()
        if not text.strip():
            return ""
        return f"`{text}`"

    elif element.name in ["strong", "b"]:
        text = process_children(element, base_url, depth, context)
        if not text.strip():
            return ""
        return f"**{text}**"

    elif element.name in ["em", "i"]:
        text = process_children(element, base_url, depth, context)
        if not text.strip():
            return ""
        return f"*{text}*"

    elif element.name == "img":
        alt = element.get("alt", "")
        src = element.get("src", "")
        if src:
            if not src.startswith(("http://", "https://")):
                src = urljoin(base_url, src)
            return f"\n\n![{alt}]({src})\n\n"
        return ""

    elif element.name == "table":
        return f"\n\n{process_table(element, base_url, depth, context)}\n\n"

    elif element.name == "br":
        return "\n"

    elif element.name == "hr":
        return "\n\n---\n\n"

    # For divs, spans and other containers, just process the children
    return process_children(element, base_url, depth, context)


def process_children(element, base_url, depth, context=None):
    """Process all child elements and combine their markdown content."""
    if context is None:
        context = {"in_block": False, "list_type": None}

    result = []
    for child in element.children:
        child_content = process_element(child, base_url, depth + 1, context)
        if child_content:
            result.append(child_content)

    return "".join(result)


def process_list(list_element, base_url, depth, context=None):
    """Process an HTML list element (ul/ol) into markdown list."""
    if context is None:
        context = {"in_block": False, "list_type": list_element.name}
    else:
        context["list_type"] = list_element.name

    result = []
    is_ordered = list_element.name == "ol"

    for i, item in enumerate(list_element.find_all("li", recursive=False)):
        # Create a new context for each list item to handle nested lists
        item_context = context.copy()

        prefix = f"{i + 1}. " if is_ordered else "- "
        content = process_children(item, base_url, depth + 1, item_context).strip()

        # Handle nested lists - if content already has list items, indent them properly
        content_lines = content.split("\n")
        formatted_content = content_lines[0] if content_lines else ""

        if len(content_lines) > 1:
            # The first line goes with the prefix, the rest are indented if they're part of a nested list
            for line in content_lines[1:]:
                if line.strip().startswith(("- ", "1. ", "* ")):
                    # This is a nested list item, indent it
                    formatted_content += "\n    " + line
                else:
                    # This is a continuation of the current list item
                    formatted_content += "\n" + line

        result.append(f"{prefix}{formatted_content}")

    return "\n".join(result)


def process_table(table_element, base_url, depth, context=None):
    """Process an HTML table into a markdown table."""
    if context is None:
        context = {"in_block": False, "list_type": None}

    result = []

    # Process header
    headers = []
    header_row = table_element.find("thead")
    if header_row:
        for th in header_row.find_all("th"):
            header_text = process_children(th, base_url, depth + 1, context).strip()
            # Replace any newlines in header with spaces
            header_text = re.sub(r"\s+", " ", header_text)
            headers.append(header_text)
    else:
        # Try to get headers from the first row
        first_row = table_element.find("tr")
        if first_row:
            for cell in first_row.find_all(["th", "td"]):
                header_text = process_children(
                    cell, base_url, depth + 1, context
                ).strip()
                # Replace any newlines in header with spaces
                header_text = re.sub(r"\s+", " ", header_text)
                headers.append(header_text)

    if headers:
        result.append("| " + " | ".join(headers) + " |")
        result.append("| " + " | ".join(["---"] * len(headers)) + " |")

    # Process table body
    body_rows = table_element.find("tbody")
    if body_rows:
        rows = body_rows.find_all("tr")
    else:
        rows = table_element.find_all("tr")
        if headers and rows:  # Skip the header row if we already processed it
            rows = rows[1:]

    for row in rows:
        cells = []
        for cell in row.find_all("td"):
            cell_text = process_children(cell, base_url, depth + 1, context).strip()
            # Replace any newlines with spaces to keep the table format intact
            cell_text = re.sub(r"\s+", " ", cell_text)
            cells.append(cell_text)

        if cells:
            result.append("| " + " | ".join(cells) + " |")

    return "\n".join(result)


def textwrap_dedent(text):
    """Remove common leading whitespace from a block of text."""
    if not text:
        return text

    # Split into lines
    lines = text.splitlines()

    # Find minimum common whitespace
    def line_indent(line):
        return len(line) - len(line.lstrip())

    non_empty_lines = [line for line in lines if line.strip()]
    if not non_empty_lines:
        return text

    min_indent = min(line_indent(line) for line in non_empty_lines)

    # Remove the common whitespace from each line
    if min_indent > 0:
        dedented_lines = []
        for line in lines:
            if line.strip():  # Only remove indent from non-empty lines
                dedented_lines.append(line[min_indent:])
            else:
                dedented_lines.append("")

        return "\n".join(dedented_lines)

    return text


def post_process_markdown(markdown):
    """
    Apply post-processing to clean up the markdown output.

    Args:
        markdown: The raw markdown text

    Returns:
        Cleaned markdown text
    """
    # Fix excessive whitespace
    markdown = re.sub(r"\n{3,}", "\n\n", markdown)

    # Fix list items spacing
    markdown = re.sub(r"(\n\s*-\s+[^\n]+)(\n\s*-\s+)", r"\1\n\2", markdown)

    # Ensure proper spacing for headers
    markdown = re.sub(r"(#+)([^ #])", r"\1 \2", markdown)

    # Fix code blocks spacing
    markdown = re.sub(r"```(\w*)\s+", r"```\1\n", markdown)
    markdown = re.sub(r"\s+```", r"\n```", markdown)

    # Fix link references
    markdown = re.sub(r"\]\s+\(", r"](", markdown)

    # Fix bullet lists appearing in the same line
    markdown = re.sub(r"(\w+)\s*-\s+", r"\1\n\n- ", markdown)

    # Clean up multiple spaces
    markdown = re.sub(r"[ \t]+", " ", markdown)

    # Fix multiple line breaks
    markdown = re.sub(r"\n\s*\n\s*\n+", "\n\n", markdown)

    return markdown.strip()


def extract_title(soup):
    """Extract the page title from the HTML."""
    title_tag = soup.find("title")
    if title_tag:
        title = title_tag.get_text().strip()
        # Remove any site name suffix (e.g., "| Modal Docs")
        title = re.sub(r"\s*\|.*$", "", title)
        return title
    return None


def is_path(url_path):
    """Check if a URL path is a file or directory path."""
    return "." in url_path.split("/")[-1] or not url_path


def crawl_links(base_url, markdown, max_depth=1, visited=None, current_depth=0):
    """
    Crawl all links in markdown that are part of the same site

    Args:
        base_url: Base URL of the site
        markdown: Markdown content with links to extract
        max_depth: Maximum recursion depth for crawling
        visited: Set of already visited URLs
        current_depth: Current recursion depth

    Returns:
        Dictionary mapping URLs to their markdown content
    """
    if visited is None:
        visited = set()

    if current_depth > max_depth:
        return {}

    result = {base_url: markdown}
    visited.add(base_url)

    # Extract all links from the markdown
    base_domain = urlparse(base_url).netloc
    link_pattern = r"\[.*?\]\((https?://[^)]+)\)"
    links = re.findall(link_pattern, markdown)

    # Filter links to only include those from the same site
    site_links = [link for link in links if urlparse(link).netloc == base_domain]

    # Process each link
    for link in site_links:
        if link in visited:
            continue

        # Skip links to files or directories
        url_path = urlparse(link).path
        if is_path(url_path):
            continue

        # Crawl the link
        print(f"Crawling: {link}", file=sys.stderr)
        link_markdown = scrape_modal_docs(link)
        if link_markdown:
            result[link] = link_markdown
            visited.add(link)

            # Recursively crawl links in the new markdown if not at max depth
            if current_depth < max_depth:
                result.update(
                    crawl_links(
                        link, link_markdown, max_depth, visited, current_depth + 1
                    )
                )

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Scrape Modal documentation and convert to markdown"
    )
    parser.add_argument("url", help="URL of the Modal documentation page")
    parser.add_argument(
        "-o", "--output", help="Output file (if not specified, prints to stdout)"
    )
    parser.add_argument("--crawl", action="store_true", help="Crawl linked pages")
    parser.add_argument(
        "--depth", type=int, default=1, help="Maximum crawl depth (default: 1)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON with URLs as keys and markdown as values",
    )

    args = parser.parse_args()

    try:
        # Scrape the initial URL
        markdown = scrape_modal_docs(args.url)

        # Crawl links if requested
        if args.crawl:
            print(f"Crawling links with max depth {args.depth}...", file=sys.stderr)
            results = crawl_links(args.url, markdown, args.depth)
        else:
            results = {args.url: markdown}

        # Process output
        if args.json:
            # Output as JSON
            output = json.dumps(results, indent=2)
        else:
            # Output just the markdown for the initial URL
            output = results[args.url]

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(output)
            print(f"Output saved to {args.output}")
        else:
            print(output)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
