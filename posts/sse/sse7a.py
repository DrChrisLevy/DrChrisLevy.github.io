# ruff: noqa: F403, F405
from asyncio import sleep
from urllib.parse import quote, unquote

from fasthtml.common import *
from monsterui.all import *

app, rt = fast_app(
    hdrs=(
        Theme.blue.headers(),
        Script(src="https://unpkg.com/htmx-ext-sse@2.2.1/sse.js"),
    ),
    live=True,
)


@app.route("/")
def get():
    return Container(
        H1("Streamed Chat"),
        Form(
            TextArea(
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
    msg = quote(msg)
    assistant_msg = Div(
        id="chat-content",
        hx_ext="sse",
        sse_connect="/get-message?msg=" + msg,
        sse_swap="EventName",
        sse_close="close",
        hx_swap="beforeend show:bottom",
    )

    return assistant_msg


async def message_generator(msg: str):
    import os

    from dotenv import load_dotenv
    from google import genai

    final_message = ""
    load_dotenv()
    msg = unquote(msg)
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    for chunk in client.models.generate_content_stream(model="gemini-2.0-flash-001", contents=msg):
        chunk = chunk.text
        final_message += chunk
        yield sse_message(chunk, event="EventName")
        await sleep(0.025)
    yield sse_message(Div(), event="close")


@app.get("/get-message")
async def get_message(msg: str):
    return EventStream(message_generator(msg))


serve(port=5010)
