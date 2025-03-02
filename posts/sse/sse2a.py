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
        H3("How NOT to Start SSE with Button Click"),
        Form(
            Button(
                "Start SSE",
                hx_ext="sse",
                sse_swap="EventName",
                sse_connect="/sse-stream",
                hx_swap="beforeend show:bottom",
                hx_target="#sse-content",  # This is the target element that will receive the SSE messages
            ),
        ),
        Div(
            P("The contents of this <div> will be updated in real time with each SSE message received."),
            id="sse-content",
        ),
    )


async def message_generator():
    for i in range(10):
        yield sse_message(Div(P(f"message number {i}")), event="EventName")
        await sleep(0.5)

    yield sse_message(Div(P("DONE")), event="EventName")


@rt("/sse-stream")
async def sse_stream():
    return EventStream(message_generator())


serve(port=5010)
