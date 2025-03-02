# ruff: noqa: F403, F405
from asyncio import sleep

from fasthtml.common import *
from monsterui.all import *

app, rt = fast_app(
    hdrs=(
        Theme.blue.headers(highlightjs=True),
        Script(src="https://unpkg.com/htmx-ext-sse@2.2.1/sse.js"),
    ),
    live=True,
)


@rt("/")
def index():
    return Container(
        H3("Multiple events in different elements (from the same source)."),
        Div(
            DivFullySpaced(
                Div(P("event1"), sse_swap="event1", id="event1-content"),
                Div(P("event2"), sse_swap="event2", id="event2-content"),
            ),
            hx_ext="sse",
            sse_connect="/sse-stream",
            hx_swap="beforeend show:bottom",
            sse_close="close",
        ),
    )


async def message_generator():
    for i in range(10):
        event_name = "event1" if i % 2 == 0 else "event2"
        yield sse_message(Div(P(f"message number {i} from {event_name}")), event=event_name)
        await sleep(0.5)

    yield sse_message(Div(P("DONE event1")), event="event1")
    yield sse_message(Div(P("DONE event2")), event="event2")
    yield sse_message(Div(), event="close")


@rt("/sse-stream")
async def sse_stream():
    return EventStream(message_generator())


serve(port=5010)
