# ruff: noqa: F403, F405
from fasthtml.common import *
from monsterui.all import *

app, rt = fast_app(
    hdrs=(Theme.blue.headers(highlightjs=True),),
    live=True,
)


@rt("/")
def index():
    return Container(
        H3("Move the Mouse Over Me", hx_get="/get-content", hx_target="#target-div", hx_swap="beforeend", hx_trigger="mouseover"),
        Div(id="target-div"),
    )


@rt("/get-content")
def get_content():
    return P("MORE CONTENT")


serve(port=5010)
