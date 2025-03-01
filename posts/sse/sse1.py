# ruff: noqa: F403, F405
from asyncio import sleep

from fasthtml.common import *
from monsterui.all import *

app, rt = fast_app(
    hdrs=(
        Theme.blue.headers(highlightjs=True),  # monsterui styling
        Script(src="https://unpkg.com/htmx-ext-sse@2.2.1/sse.js"),  # Include this to use the SSE extension
    ),
    live=True,
)


@rt("/")
def index():
    return Container(
        H3("Intro SSE Example"),
        Div(
            P("The contents of this <div> will be updated in real time with each SSE message received."),
            hx_ext="sse",  # To connect to an SSE server, use the hx_ext="sse" attribute to install the extension on that HTML element
            sse_swap="EventName",  # This default event name is "message" if we don't specify it otherwise
            sse_connect="/sse-stream",  # This is the URL of the SSE endpoint we create and connect to
            hx_swap="beforeend show:bottom",  # Determines how the content will be inserted into that target element. Here, each new message is added at the end of the div and the page automatically scrolls to show the new message
            hx_target=None,  # None is the default. By not specifying a target for the swap, it defaults to the element that triggered the request i.e. id="sse-content"
            id="sse-content",
        ),
    )


async def message_generator():
    for i in range(10):
        yield sse_message(Div(P(f"message number {i}")), event="EventName")  # must match the sse_swap attribute
        await sleep(0.5)

    # This function converts an HTML element into the specific format required for Server-Sent Events (SSE) streaming.
    # The first argument is an FT component (FastHTML element) that you want to send via SSE.
    # The second argument is the name of the SSE event (defaults to "message" if not specified)
    yield sse_message(Div(P("DONE")), event="EventName")  # event="EventName" must match the sse_swap attribute above


@rt("/sse-stream")
async def sse_stream():
    return EventStream(message_generator())


serve(port=5010)
