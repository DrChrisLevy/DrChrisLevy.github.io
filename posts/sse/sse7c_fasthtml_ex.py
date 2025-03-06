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

final_message = ""


@app.route("/")
def get():
    return Container(
        H1("Streamed Chat"),
        Form(
            Input(
                type="text",
                name="msg",
                id="msg-input",
                placeholder="Type a message",
            ),
            Button("Send"),
            hx_post="/send-message",
            hx_target="#chat-response",
            hx_swap="innerHTML",
        ),
        Div(
            id="chat-response",
        ),
    )


@app.post("/send-message")
def send_message(msg: str):
    print(f"SEND MESSAGE: {msg}")
    assistant_msg = Div(id="chat-content", hx_ext="sse", sse_connect="/get-message", sse_swap="EventName", sse_close="close", hx_swap="beforeend")

    return assistant_msg


async def message_generator():
    global final_message
    r = ["# This ", "is ", "a ", "- test\n- test2"]
    for chunk in r:
        final_message += chunk
        yield sse_message(chunk, event="EventName")
        await sleep(0.5)
    yield sse_message(Div(), event="close")


@app.get("/get-message")
async def get_message():
    return EventStream(message_generator())


serve(port=5010)
