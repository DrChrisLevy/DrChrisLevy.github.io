# ruff: noqa: F403, F405
from asyncio import sleep

from fasthtml.common import *
from monsterui.all import *

app, rt = fast_app(
    hdrs=(
        Theme.blue.headers(),
        Script(src="https://unpkg.com/htmx-ext-sse@2.2.1/sse.js"),
    ),
    live=True,
)


@rt("/")
def index():
    return Container(
        H3("Multiple Events in the Same Element"),
        Div(
            P("The contents of this <div> will be updated in real time with each SSE message received from both event1 and event2."),
            hx_ext="sse",
            sse_swap="event1,event2",  # Multiple events can be listened to
            sse_connect="/sse-stream",
            hx_swap="beforeend show:bottom",
            hx_target="#sse-content",
            id="sse-content",
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
