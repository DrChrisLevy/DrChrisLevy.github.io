from asyncio import sleep

import marko
import modal
from fasthtml.common import *

from utils import log_to_queue, read_from_queue

picocss = "https://cdn.jsdelivr.net/npm/@picocss/pico@latest/css/pico.indigo.min.css"
picolink = (Link(rel="stylesheet", href=picocss), Style(":root { --pico-font-size: 100%; }"))
md = Script(src="https://cdn.jsdelivr.net/npm/marked/marked.min.js")
app, rt = fast_app(hdrs=(md, MarkdownJS(), *picolink, Script(src="https://unpkg.com/htmx-ext-sse@2.2.1/sse.js")))

chunks = []


@rt("/")
def get():
    return Titled(
        "Ask Questions about any PDF Document",
        Grid(
            Div(
                Form(hx_post="/process", hx_target="#images_used", hx_swap="innerHTML")(
                    Group(
                        Label("PDF URL:", Input(name="pdf_url", type="url", required=True, value="https://arxiv.org/pdf/2407.01449")),
                        Label(
                            "Number of Pages for Context",
                            Select(name="top_k", required=True)(
                                Option("1", value="1", selected=True),
                                Option("2", value="2"),
                                Option("3", value="3"),
                                Option("4", value="4"),
                                Option("5", value="5"),
                            ),
                        ),
                    ),
                    Label(
                        "Additional Instructions (Optional)",
                        Textarea(name="additional_instructions", required=False, type="text")(
                            "Use markdown headers and bullet points in your responses."
                        ),
                    ),
                    Label(
                        "Question",
                        Input(
                            name="question",
                            required=True,
                            type="text",
                            value="How does the latency between ColPali and standard retrieval methods compare?",
                        ),
                    ),
                    Button("Submit", type="submit"),
                )
            ),
            Div(
                id="terminal_stream",
                style="background-color: rgba(26, 26, 26, 0.8); color: #00ff00; font-family: 'Courier New', monospace; height: 362px; overflow-y: auto; padding: 15px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); backdrop-filter: blur(5px);",
                hx_ext="sse",
                sse_connect="/poll-queue",
                hx_swap="beforeend",
                sse_swap="message",
            ),
        ),
        Div(H1("Answer:")),
        Div(
            Div(
                id="stream-container",
            ),
            hx_ext="sse",
            sse_connect="/call-openai",
            # sse_close="completed",
            sse_swap="message",
            hx_swap="show:#stream-container:bottom",
        ),
        Div(H1("Pages Used for Context:"), id="images_used"),
    )


@rt("/process")
def post(pdf_url: str, question: str, top_k: int, additional_instructions: str):
    global chunks
    chunks = []
    answer_question_with_image_context = modal.Function.lookup("multi-modal-rag", "answer_question_with_image_context")
    log_to_queue("Starting Multi Modal RAG")
    res = answer_question_with_image_context.remote_gen(
        pdf_url=pdf_url,
        query=question,
        top_k=top_k,
        use_cache=True,
        max_new_tokens=8000,
        additional_instructions=additional_instructions,
    )
    for r in res:
        if isinstance(r, str):
            chunks.append(r)
        else:
            images_data = r
    log_to_queue("Done Calling OpenAI")
    image_elements = Grid(
        *(
            Div(
                Img(
                    src=im,
                    style="width: 100%; height: auto;",
                )
            )
            for im in images_data
        )
    )
    final_html = H1("Pages Used for Context:"), Div(image_elements)
    chunks = []
    return final_html


shutdown_event = signal_shutdown()


async def terminal_log_generator():
    print("Opening SSE for Terminal Logs")
    while not shutdown_event.is_set():
        message = read_from_queue()
        if message:
            yield sse_message(Div(message, style="white-space: pre-wrap;"))
        await sleep(0.1)


@rt("/poll-queue")
async def get():
    return EventStream(terminal_log_generator())


async def stream_chunks():
    print("Opening SSE for OpenAI")
    global chunks
    while not shutdown_event.is_set():
        if not chunks:
            # yield sse_message(" ", event="completed")
            await sleep(0.05)
            continue
        chunks_stream = "".join([c for c in chunks if isinstance(c, str)])
        yield sse_message(Div(NotStr(marko.convert("".join(chunks_stream))), id="stream-container"))
        await sleep(0.01)


@rt("/call-openai")
async def get():
    return EventStream(stream_chunks())


serve()
