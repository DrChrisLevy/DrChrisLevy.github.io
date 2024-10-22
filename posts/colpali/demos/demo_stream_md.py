from asyncio import sleep

import marko
from fasthtml.common import *

md = Script(src="https://cdn.jsdelivr.net/npm/marked/marked.min.js")

app, rt = fast_app(
    hdrs=(
        md,
        MarkdownJS(),
        Script(src="https://unpkg.com/htmx-ext-sse@2.2.2/sse.js"),
    )
)

# fmt: off
chunks = ['\n', '###', ' J', 'okes', ' to', ' Bright', 'en', ' Your', ' Day', '\n\n', '-', ' **', 'Why', " don't", ' skeleton', 's', ' fight', ' each', ' other', '?', '**', '  \n', ' ', ' They', " don't", ' have', ' the', ' guts', '!\n\n', '-', ' **', 'What', ' do', ' you', ' call', ' cheese', ' that', " isn't", ' yours', '?', '**', '  \n', ' ', ' Nach', 'o', ' cheese', '!\n\n', '-', ' **', 'Why', ' did', ' the', ' scare', 'crow', ' win', ' an', ' award', '?', '**', '  \n', ' ', ' Because', ' he', ' was', ' outstanding', ' in', ' his', ' field', '!\n\n', '-', ' **', 'What', ' did', ' the', ' zero', ' say', ' to', ' the', ' eight', '?', '**', '  \n', ' ', ' Nice', ' belt', '!']
# fmt: on


@rt("/")
def get():
    return Titled(
        Div(H1("Markdown Works for Static Content")),
        Div("".join(chunks), cls="marked"),
        Div(H1("Markdown Does not work for Streaming Content")),
        Div(
            Div(
                id="stream-container",
            ),
            hx_ext="sse",
            sse_connect="/call-openai",
            sse_close="completed",
            sse_swap="message",
            hx_swap="show:#stream-container:bottom",
        ),
    )


async def call_openai():
    print("Opening Stream to OpenAI")
    # demo stream with fake data
    ls = []
    for chunk in chunks:
        ls.append(chunk)
        yield sse_message(Div(NotStr(marko.convert("".join(ls))), id="stream-container"))
        await sleep(0.1)
    yield sse_message(" ", event="completed")


@rt("/call-openai")
async def get():
    return EventStream(call_openai())


serve()
