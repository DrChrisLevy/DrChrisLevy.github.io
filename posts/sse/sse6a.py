# ruff: noqa: F403, F405
import random
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


@rt
def index():
    return Titled(
        "SSE Random Number Generator",
        P("Generate pairs of random numbers, as the list grows scroll downwards."),
        Div(hx_ext="sse", sse_connect="/number-stream", hx_swap="beforeend show:bottom", sse_swap="message"),
    )


async def number_generator():
    data = Article(random.randint(1, 100))
    yield sse_message(data)
    await sleep(0.01)


@rt("/number-stream")
async def get():
    return EventStream(number_generator())


serve(port=5010)
