import json

from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax

YELLOW_HEX = "#d4b702"
GREEN_HEX = "#00ff00"
INPUT_COLOR = "#4A9EED"  # Cool blue
OUTPUT_COLOR = "#22C55E"  # Warm green
console = Console()


def console_print_user_request(messages, model):
    first_user_message = next(m for m in messages if m["role"] == "user")["content"]
    console.print(
        Panel(
            f"\n[bold]{first_user_message}\n",
            title="[bold]User Request",
            subtitle=model,
            border_style=YELLOW_HEX,
            subtitle_align="left",
        )
    )


def console_print_step(iteration):
    console.print("\n")
    console.print(Rule(f"[bold]Step {iteration}", characters="━", style=YELLOW_HEX))
    console.print("\n")


def console_print_tool_call_inputs(assistant_content, tool_calls):
    tools_args_list = [json.loads(t["function"]["arguments"]) for t in tool_calls]
    tool_names = [t["function"]["name"] for t in tool_calls]
    tool_call_ids = [t["id"] for t in tool_calls]

    # Create panels list to store individual panels
    panels = []
    panels.append(
        Panel(
            f"[bold]Assistant Message:[/bold]\n {assistant_content}",
            title="[bold]Assistant Content",
            border_style=INPUT_COLOR,
            title_align="center",
        )
    )

    # Convert list comprehension to for loop
    for tool_name, tool_args, tool_call_id in zip(tool_names, tools_args_list, tool_call_ids):
        if tool_name == "execute_python_code":
            content = Syntax(
                tool_args["code"],
                lexer="python",
                theme="monokai",
                word_wrap=True,
                line_numbers=True,
            )
        else:
            content = f"\n[bold]{tool_args}\n"

        panel = Panel(
            content,
            title="[bold]Tool Call",
            subtitle=f"{tool_name} - {tool_call_id}",
            border_style=INPUT_COLOR,
            subtitle_align="left",
            title_align="left" if tool_name == "do_code" else "center",
        )
        panels.append(panel)

    # Create group from collected panels
    tool_panels = Group(*panels)

    # Wrap the group in an outer panel
    console.print(
        Panel(
            tool_panels,
            title="[bold]Parallel Tool Calls Inputs",
            border_style=INPUT_COLOR,
        )
    )


def console_print_tool_call_outputs(tool_calls, tool_results):
    panels = []

    for tool_call, tool_result in zip(tool_calls, tool_results):
        tool_name = tool_call["function"]["name"]
        tool_id = tool_call["id"]

        # Format the content based on tool type
        if tool_name == "web_search":
            content = format_search_results(tool_result)
        elif tool_name == "execute_python_code":
            content = format_code_result(tool_result)
        else:
            content = f"\n[bold]{tool_result[:300]}...<truncated>\n"

        # Create panel for this tool output
        panel = Panel(
            content,
            title="[bold]Tool Call Output",
            subtitle=f"{tool_name} - {tool_id}",
            border_style=OUTPUT_COLOR,
            subtitle_align="left",
        )
        panels.append(panel)

    # Wrap all panels in a group and outer panel
    tool_panels = Group(*panels)
    console.print(
        Panel(
            tool_panels,
            title="[bold]Parallel Tool Calls Outputs",
            border_style=OUTPUT_COLOR,
        )
    )


def console_print_llm_output(llm_output):
    console.print(
        Panel(
            Markdown(llm_output),
            title="[bold]Final Answer",
            subtitle="final-answer",
            border_style=GREEN_HEX,
            subtitle_align="left",
        )
    )


def format_search_results(results):
    formatted = []
    for i, result in enumerate(results, 1):
        title = result["title"][:100] + "..." if len(result["title"]) > 100 else result["title"]
        body = result["body"][:150] + "..." if len(result["body"]) > 150 else result["body"]

        formatted.append(
            f"[bold]Result {i}:[/bold]\n" f"[bold]Title:[/bold] {title}\n" f"[bold]URL:[/bold] {result['href']}\n" f"[dim]{body}[/dim]\n"
        )

    return "\n".join(formatted)


def format_code_result(result):
    output = []

    # Add stdout if present (with syntax highlighting)
    stdout = result["stdout"].strip()
    if stdout:
        output.append(f"[bold blue]stdout:[/bold blue]\n" f"{stdout}")
    stderr = result["stderr"].strip()
    if stderr:
        output.append(f"[bold red]stderr:[/bold red]\n" f"[red]{stderr}[/red]")

    # Add the final result with syntax highlighting
    is_success = result["success"]
    color = "green" if is_success else "red"
    output.append(f"[bold {color}]is_success:[/bold {color}] {is_success}")

    if result["error"]:
        output.append(f"[bold red]Error:[/bold red]\n" f"[red]{result['error']}[/red]")

    return "\n\n".join(output)