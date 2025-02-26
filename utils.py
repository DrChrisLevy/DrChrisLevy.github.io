# ruff: noqa: F403, F405
from fasthtml.common import *
from monsterui.all import *


def NavigationBar():
    nav_items = [A("Blog", href="/blog")]
    return NavBar(
        *nav_items,
        brand=A("Home", href="/"),
    )


def layout(content, req):
    return Div(
        NavigationBar(),
        Container(content, cls=ContainerT.lg),
    )
