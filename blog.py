# ruff: noqa: F403, F405
"""
Took many code snippets from https://github.com/Isaac-Flath/web/blob/main/app/blog.py.
See there for more complet nb-dev if that's your thing.
"""
import yaml
from execnb.nbio import read_nb
from fastcore.ansi import ansi2html
from fasthtml.common import *
from monsterui.all import *

from utils import layout


def extract_directives(cell):
    """Extract Quarto directives (starting with # |) from a notebook cell.
    Returns a dict of directive names and their values."""
    directives = {}
    if cell.source:
        lines = cell.source.split("\n")
        for line in lines:
            if line.startswith("# |"):
                # Remove '# |' and split into directive and value
                parts = line[3:].strip().split(":")
                if len(parts) >= 2:
                    key = parts[0].strip() + ":"  # Add back the colon to match your format
                    value = ":".join(parts[1:]).strip()
                    directives[key] = [value]  # Store value in a list to match nbdev format
                else:
                    # Handle boolean directives without values
                    # directives[parts[0].strip() + ':'] = ['true']
                    # TODO
                    pass
    return directives


ar = APIRouter(prefix="/blog", body_wrap=layout)


def get_meta(nb):
    return yaml.safe_load(nb.cells[0].source.split("---")[1])


def render_code_output(cell, directives, wrapper=Footer):
    if not cell.outputs:
        return ""
    # TODO: {"{'warning:': ['false']}", "{'output:': ['false']}", "{'echo:': ['false']}"}
    #   HANDLE other cases/directives
    if "include:" in directives and directives["include:"][0] == "false":
        return ""
    if "output:" in directives and directives["output:"][0] == "false":
        return ""

    def render_output(out):
        otype = out["output_type"]
        if otype == "stream":
            txt = ansi2html("".join(out["text"]))
            xtra = "" if out["name"] == "stdout" else "class='stderr'"
            is_err = "<span class" in txt
            return Safe(f"<pre {xtra}><code class='{'nohighlight hljs' if is_err else ''}'>{txt}</code></pre>")
        elif otype in ("display_data", "execute_result"):
            data = out["data"]
            _g = lambda t: "".join(data[t]) if t in data else None
            if d := _g("text/html"):
                return Safe(apply_classes(d))
            if d := _g("application/javascript"):
                return Safe(f"<script>{d}</script>")
            if d := _g("text/markdown"):
                return render_md(d)
            if d := _g("text/latex"):
                return Safe(f'<div class="math">${d}$</div>')
            if d := _g("image/jpeg"):
                return Safe(f'<img src="data:image/jpeg;base64,{d}"/>')
            if d := _g("image/png"):
                return Safe(f'<img src="data:image/png;base64,{d}"/>')
            if d := _g("text/plain"):
                return Safe(f"<pre><code>{d}</code></pre>")
            if d := _g("image/svg+xml"):
                return Safe(d)
        return ""

    res = Div(*map(render_output, cell.outputs))
    if res:
        return wrapper(res)


def render_code_input(cell, directives, lang="python"):
    code = f"""```{lang}\n{cell.source}\n```\n"""
    if "include:" in directives and directives["include:"][0] == "false":
        return ""
    if "echo:" in directives and directives["echo:"][0] == "false":
        return ""
    if "code-fold:" in directives and directives["code-fold:"][0] == "true":
        return Details(Summary("See Code"), render_md(code))
    return render_md(code)


def remove_directives(cell):
    "Remove #| directives from start of cell"
    lines = cell.source.split("\n")
    while lines and lines[0].startswith("# |"):
        lines.pop(0)
    cell.source = "\n".join(lines)


def render_nb(nb):
    "Render a notebook as a list of html elements"
    res = []
    meta = get_meta(nb)
    res.append(Div(H1(meta["title"]), Subtitle(meta.get("subtitle", "")), cls="my-9"))
    for cell in nb.cells[1:]:
        if cell["cell_type"] == "code":
            directives = extract_directives(cell)
            remove_directives(cell)
            _output = render_code_output(cell, directives)
            res.append(render_code_input(cell, directives))
            res.append(Card(_output) if _output else "")
        elif cell["cell_type"] == "markdown":
            res.append(render_md(cell.source))
    return res


@ar
def index():
    fpaths = [
        "posts/dspy/dspy.ipynb",
        "posts/bits_and_bytes/bits_bytes.ipynb",
        "posts/modal_fun/modal_blog.ipynb",
        "posts/gemini/gemini2.ipynb",
        "posts/intro_fine_tune/intro_fine_tune.ipynb",
        "posts/intro_modal/intro_modal.ipynb",
        "posts/vllms/vllm.ipynb",
        "posts/agents/agents.ipynb",
        "posts/colpali/colpali_blog.ipynb",
        "posts/fine_tune_jarvis/fine_tune_jarvis.ipynb",
        "posts/china_ai/qwen_deepseek.ipynb",
        "posts/basic_transformer_notes/transformers.ipynb",
        "posts/llm_inference_class/llm_inference.ipynb",
        "posts/modern_bert/modern_bert.ipynb",
        "posts/anthropic/anthropic.ipynb",
        "posts/open_hermes_pro/open_hermes.ipynb",
        "posts/llm_lunch_talk/llm_talk_slides.ipynb",
    ]
    metas = []
    for fpath in fpaths:
        folder = fpath.split("/")[1]
        _meta = get_meta(read_nb(fpath))
        _meta["fpath"] = fpath
        _meta["folder"] = folder
        _meta["image"] = f'../posts/static_blog_imgs/{_meta.get("image", "")}'
        metas.append(_meta)
    metas.sort(key=lambda x: x["date"], reverse=True)
    return Div(
        Div(H1("My Blog", cls="mb-2"), Subtitle("Some Random Things I'm Interested In", cls=TextT.gray + TextT.lg), cls="text-center py-8"),
        Div(Grid(*map(blog_card, metas), cols=1), cls="max-w-4xl mx-auto px-4"),
    )


@ar
def blog_post(fpath: str):
    return render_nb(read_nb(fpath))


def blog_card(meta):
    def Tags(cats):
        return DivLAligned(map(Label, cats))

    return Card(
        DivLAligned(
            A(Img(src=meta.get("image", ""), style="width:200px"), href=blog_post.to(fpath=meta["fpath"])),
            Div(cls="space-y-3 w-full")(
                H4(meta["title"]),
                P(meta.get("description", "")),
                DivFullySpaced(map(Small, [meta["author"], meta["date"]]), cls=TextT.meta),
                DivFullySpaced(Tags(meta.get("tags", [])), A("Read", cls=("uk-btn", ButtonT.primary, "h-6"), href=blog_post.to(fpath=meta["fpath"]))),
            ),
        ),
        cls=CardT.hover,
    )
