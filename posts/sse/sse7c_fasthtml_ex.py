# ruff: noqa: F403, F405
import asyncio

from fasthtml.common import *
from monsterui.all import *

app, rt = fast_app(
    hdrs=(
        Theme.blue.headers(highlightjs=True),
        Script(src="https://unpkg.com/htmx-ext-sse@2.2.1/sse.js"),
    ),
    live=True,
)

# Set up a chat model client and list of messages (https://claudette.answer.ai/)

messages = []


# Send messages to the chat model and yield the responses
async def message_generator():
    print("message_generator", messages)
    r = ["This ", "is ", "a ", "test"]
    for chunk in r:
        messages[-1]["content"] += chunk
        yield sse_message(chunk)
        await asyncio.sleep(0.5)
    yield sse_message(Div(), event="close")


# Chat message component (renders a chat bubble)
# Now with a unique ID for the content and the message
def ChatMessage(msg_idx, **kwargs):
    msg = messages[msg_idx]
    bubble_class = "chat-bubble-primary" if msg["role"] == "user" else "chat-bubble-secondary"
    chat_class = "chat-end" if msg["role"] == "user" else "chat-start"
    return Div(
        Div(msg["role"], cls="chat-header"),
        Div(
            msg["content"],
            id=f"chat-content-{msg_idx}",  # Target if updating the content
            cls=f"chat-bubble {bubble_class}",
            **kwargs,
        ),
        id=f"chat-message-{msg_idx}",  # Target if replacing the whole message
        cls=f"chat {chat_class}",
    )


# The input field for the user message. Also used to clear the
# input field after sending a message via an OOB swap
def ChatInput():
    return Input(
        type="text",
        name="msg",
        id="msg-input",
        placeholder="Type a message",
        cls="input input-bordered w-full",
        hx_swap_oob="true",
    )


# The main screen
@app.route("/")
def get():
    page = Body(
        H1("Chatbot SSE (server-sent events) Demo"),
        Div(
            *[ChatMessage(msg) for msg in messages],
            id="chatlist",
            cls="chat-box h-[73vh] overflow-y-auto",
        ),
        Form(
            Group(ChatInput(), Button("Send", cls="btn btn-primary")),
            hx_post="/send-message",
            hx_target="#chatlist",
            hx_swap="beforeend",
            cls="flex space-x-2 mt-2",
        ),
        cls="p-4 max-w-lg mx-auto",
    )
    return Title("Chatbot Demo"), page


@app.get("/get-message")
async def get_message():
    return EventStream(message_generator())


@app.post("/send-message")
def send_message(msg: str):
    messages.append({"role": "user", "content": msg})
    user_msg = Div(ChatMessage(len(messages) - 1))
    messages.append({"role": "assistant", "content": ""})
    # The returned assistant message uses the SSE extension, connect to the /get-message endpoint and get all messages until the close event
    assistant_msg = Div(
        ChatMessage(len(messages) - 1, hx_ext="sse", sse_connect="/get-message", sse_swap="message", sse_close="close", hx_swap="beforeend")
    )
    return user_msg, assistant_msg, ChatInput()


serve(port=5010)
