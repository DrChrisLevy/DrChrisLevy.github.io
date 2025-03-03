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


@rt("/")
def index():
    return Container(
        H2("Simple Streaming Chat Demo"),
        # Input form
        Form(
            TextArea(placeholder="Type your message here...", id="message-input"),
            Button(
                "Send",
                type="submit",
                cls=ButtonT.primary,
            ),
            hx_post="/send-message",
            hx_target="#chat-container",
            hx_swap="innerHTML",
        ),
        # Chat messages container
        Div(id="chat-container"),
    )


@rt("/send-message", methods=["POST"])
async def send_message(req):
    # Get the message from the form
    form_data = await req.form()
    message = form_data.get("message-input", "").strip()
    encoded_message = quote(message)
    # Create bot response container
    bot_container = Div(
        id="current-response",
        hx_ext="sse",
        sse_connect="/chat-stream?message=" + encoded_message,
        sse_swap="token",
        hx_swap="innerHTML",
        sse_close="close",
    )

    return bot_container


async def stream_response(message: str):
    response = f"""Thank you for your message! This is a simple demonstration of streaming text using Server-Sent Events.
    Here is your original message:\n\n{message}"""
    message = ""
    for chunk in response:
        message += chunk
        yield sse_message(render_md(message), event="token")
        await sleep(0.01)

    yield sse_message(Div(), event="close")


@rt("/chat-stream")
async def chat_stream(message: str):
    decoded_message = unquote(message)
    return EventStream(stream_response(decoded_message))


serve(port=5010)
