from functools import partial

from fasthtml.common import *
from fasthtml.oauth import *
from monsterui.all import *

# %% ../nbs/main.ipynb
# from social_media import ar as ar_social
from blog import ar as ar_blog
from utils import *

# %% ../nbs/main.ipynb
app, rt = fast_app(
    hdrs=Theme.blue.headers(highlightjs=True),
    routes=(Mount("/blog/static_blog_imgs", StaticFiles(directory="posts/static_blog_imgs")),),
    body_wrap=layout,
    live=True,
)


# %% ../nbs/main.ipynb
@rt
def index():
    return DivCentered(PicSumImg(cls="rounded-full w-96 h-96 object-cover mb-6"), H1("Chris Levy"))


# %% ../nbs/main.ipynb
@rt
def index():
    _href = partial(A, cls=AT.primary, target="_blank")

    def _project_slider_card(title, subtitle, href=None, cls=""):
        return Card(A(H3(title), href=href, cls=AT.primary if href else "", target="_blank", disabled=href is None), Subtitle(subtitle), cls=cls)

    def _section(*c):
        return Section(Article(*c, cls="prose max-w-5xl mx-auto space-y-5 pt-16"), cls=("uk-padding-remove-vertical",))

    return Div(
        # Header section with profile
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


# %% ../nbs/main.ipynb


# %% ../nbs/main.ipynb
# ar_social.to_app(app)
# ar_todo.to_app(app)
ar_blog.to_app(app)

# %% ../nbs/main.ipynb
redir_path = "/redirect"
skip = (
    "/login",
    "/blog",
    "/blog/",
    r"/blog/.*",
    "/",
    "social_media/share_thread",
    "social_media/share_thread*",
    redir_path,
    r"/.*\.(png|jpg|ico|css|js)",
)


class Auth(OAuth):
    def get_auth(self, info, ident, session, state):
        email = info.email or ""
        session["user_name"] = email
        if info.email_verified and email.split("@")[-1] == "answer.ai":
            return RedirectResponse("/", status_code=303)


client = GoogleAppClient(os.environ.get("GOOGLE_CLIENT_ID"), os.environ.get("GOOGLE_SECRET"))
oauth = Auth(app, client, skip=skip, redir_path=redir_path)


# %% ../nbs/main.ipynb
serve()
