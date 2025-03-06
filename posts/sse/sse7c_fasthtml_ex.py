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


def ChatInput():
    return Input(
        type="text",
        name="msg",
        id="msg-input",
        placeholder="Type a message",
        hx_swap_oob="true",  # TODO: WHAT DOES IT DO?
    )


@app.route("/")
def get():
    page = Body(
        H1("Streamed Chat"),
        Form(
            (ChatInput(), Button("Send")),
            hx_post="/send-message",
            hx_target="#chat-response",
            hx_swap="innerHTML",
        ),
        Div(
            id="chat-response",
        ),
    )
    return Title("Chatbot Demo"), page


@app.post("/send-message")
def send_message(msg: str):
    print(f"SEND MESSAGE: {msg}")
    assistant_msg = Div(
        Div(
            Div(id="chat-content", hx_ext="sse", sse_connect="/get-message", sse_swap="EventName", sse_close="close", hx_swap="beforeend"),
            id="chat-message",
        )
    )
    return ChatInput(), assistant_msg


async def message_generator():
    r = ["This ", "is ", "a ", "test"]
    for chunk in r:
        yield sse_message(chunk, event="EventName")
        await sleep(0.1)
    yield sse_message(Div(), event="close")


@app.get("/get-message")
async def get_message():
    return EventStream(message_generator())


serve(port=5010)
