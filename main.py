from fasthtml.common import *
from fasthtml.oauth import *
from monsterui.all import *

from blog import ar as ar_blog
from utils import layout

app, rt = fast_app(
    hdrs=Theme.blue.headers(highlightjs=True),
    routes=(Mount("/blog/static_blog_imgs", StaticFiles(directory="posts/static_blog_imgs")),),
    body_wrap=layout,
    live=True,
)


@rt
def index():
    return DivCentered(PicSumImg(cls="rounded-full w-96 h-96 object-cover mb-6"), H1("Chris Levy"))


@rt
def index():
    def _section(*c):
        return Section(Article(*c, cls="prose max-w-5xl mx-auto space-y-5 pt-16"), cls=("uk-padding-remove-vertical",))

    return Div(
        Section(
            DivCentered(
                Img(src="posts/static_blog_imgs/pic_me.png", cls="rounded-full w-64 h-64 object-cover"),
                H1("Chris Levy", cls=TextT.center),
                cls="space-y-4 mt-12",
            ),
            cls="uk-padding-remove-vertical",
        ),
        _section(
            render_md(
                """**Hello!** I'm Chris Levy, an AI Engineer. I build practical AI systems using Python and modern ML technologies, and I'm always excited to learn new approaches.

I focus on building AI applications, with my main skill being backend development and Python. With a PhD in applied mathematics (2015) and years of industry experience across the ML stack, I enjoy tackling complex problems and turning them into practical solutions.

Outside of day to day work, I enjoy spending time with my family and three kids, working out, swimming, cycling, playing guitar, and coding and writing."""
            )
        ),
    )


ar_blog.to_app(app)


serve()
