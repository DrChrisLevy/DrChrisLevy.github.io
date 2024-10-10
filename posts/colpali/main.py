from asyncio import sleep

import modal
from fasthtml.common import *
from utils import log_to_queue, read_from_queue

picocss = "https://cdn.jsdelivr.net/npm/@picocss/pico@latest/css/pico.indigo.min.css"
picolink = (Link(rel="stylesheet", href=picocss), Style(":root { --pico-font-size: 100%; }"))
app, rt = fast_app(hdrs=(MarkdownJS(), *picolink, Script(src="https://unpkg.com/htmx-ext-sse@2.2.1/sse.js")))


@rt("/")
def get():
    return Titled(
        "Ask Questions about any PDF Document",
        Grid(
            Div(
                Form(hx_post="/process", hx_target="#results", hx_swap="innerHTML")(
                    Group(
                        Label("PDF URL:", Input(name="pdf_url", type="url", required=True, value="https://arxiv.org/pdf/2410.02525")),
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
                        Input(name="question", required=True, type="text", value="What is this paper about? Give me a detailed summary"),
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
        Div(H1("Answer:"), id="results"),
    )


@rt("/process")
def post(pdf_url: str, question: str, top_k: int, additional_instructions: str):
    answer_questions_with_image_context = modal.Function.lookup("multi-modal-rag", "answer_questions_with_image_context")
    log_to_queue("Starting Multi Modal RAG")
    res, all_images_data = answer_questions_with_image_context.remote(
        pdf_url=pdf_url,
        queries=[question],
        top_k=top_k,
        use_cache=True,
        max_new_tokens=8000,
        additional_instructions=additional_instructions,
        model="gpt-4o-mini",  # "gpt-4o-mini" or ="Qwen/Qwen2-VL-7B-Instruct"
    )
    image_elements = Grid(
        *(
            Div(
                Img(
                    src=im,
                    style="width: 100%; height: auto;",
                )
            )
            for im in all_images_data[0]
        )
    )
    #
    return Div(H1("Answer:"), *(Div(r, cls="marked") for r in res), H1("Pages Used for Context"), image_elements)


shutdown_event = signal_shutdown()


async def time_generator():
    while not shutdown_event.is_set():
        message = read_from_queue()
        if message:
            yield sse_message(Div(message, style="white-space: pre-wrap;"))
        await sleep(0.1)


@rt("/poll-queue")
async def get():
    "Send time to all connected clients every second"
    return EventStream(time_generator())


serve()
