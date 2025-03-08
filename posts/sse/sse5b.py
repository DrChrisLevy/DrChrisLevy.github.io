# ruff: noqa: F403, F405
from asyncio import sleep

from fasthtml.common import *
from monsterui.all import *

app, rt = fast_app(
    hdrs=(Theme.blue.headers(), Script(src="https://unpkg.com/htmx-ext-sse@2.2.1/sse.js")),
    live=True,
)


@rt("/")
def index():
    return Container(
        Div(
            H3("This Div Opens Up the SSE Connection"),
            Div(
                P("This Child Div Counts the Messages. Total Message Count: "),
                hx_get="/count-messages",
                hx_swap="innerHTML",
                hx_trigger="sse:EventName",  # The SSE connection with event name "EventName" will trigger a request hx_get="/count-messages",
            ),
            Div(P("This Child Div Receives SSE Messages"), sse_swap="EventName", hx_swap="beforeend"),
            hx_ext="sse",
            sse_connect="/sse-stream",
            sse_close="close",
        ),
    )


async def message_generator():
    global count
    count = 0
    for i in range(10):
        yield sse_message(Div(P(f"message number {i}")), event="EventName")
        await sleep(0.5)

    yield sse_message(Div(P("DONE")), event="EventName")
    yield sse_message(Div(), event="close")


@rt("/sse-stream")
async def sse_stream():
    return EventStream(message_generator())


@rt("/count-messages")
def count_messages():
    global count
    count += 1
    return Div(P(f"This Child Div Counts the Messages. Total Message Count: {count}"))


serve(port=5010)
