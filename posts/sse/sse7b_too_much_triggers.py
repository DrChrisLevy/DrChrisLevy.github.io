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

final_message = ""


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
        Div(H3("Place Holder")),
        Div(id="chat-container"),
    )


@rt("/send-message", methods=["POST"])
async def send_message(req):
    # Get the message from the form
    form_data = await req.form()
    message = form_data.get("message-input", "").strip()
    encoded_message = quote(message)

    return Div(id="chat-container"), Div(
        P("The contents of this <div> will be updated in real time with each SSE message received."),
        Div(id="trigger-test", hx_get="/trigger-test", hx_trigger="sse:EventName", hx_swap="innerHTML"),
        hx_ext="sse",
        sse_swap="EventName",
        sse_connect=f"/sse-stream?message={encoded_message}",
        sse_close="close",
        hx_swap="beforeend show:bottom",
        # hx_target="#sse-content",
        # id="sse-content",
    )


@rt("/trigger-test")
async def trigger_test():
    global final_message
    return render_md(final_message)


async def message_generator(message: str):
    global final_message
    message = unquote(message)
    for chunk in message:
        final_message += chunk
        yield sse_message(Div(), event="EventName")
        await sleep(0.01)
    # for i in range(10):
    #     final_message += f"message number {i}\n"
    #     yield sse_message(Div(P(f"message number {i}")), event="EventName")
    #     await sleep(0.5)

    yield sse_message(Div(), event="close")


@rt("/sse-stream")
async def sse_stream(message: str):
    return EventStream(message_generator(message))


serve(port=5010)
