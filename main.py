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
    return DivCentered(PicSumImg(cls="rounded-full w-96 h-96 object-cover mb-6"), H1("Isaac Flath"))


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
        Section(DivCentered(Img(src="pic_me.jpeg"), H1("Chris Levy", cls=TextT.center), cls="space-y-4 mt-12"), cls="uk-padding-remove-vertical"),
        _section(
            render_md(
                """**Hello!** I'm Chris Levy. I work in ML/AI and backend Python development.
## About Me

I spent a good amount of time in school where I completed a PhD in applied math back in 2015. After graduating
I shifted away from academia and started working in industry. I mostly do backend python development these days,
and build ML/AI applications/services. I work across the entire stack from research, to training and evaluating models,
to deploying models, and getting in the weeds of the infrastructure and devops pipelines.

Outside of AI/ML stuff, I enjoy spending time with my family and three kids, working out, swimming, cycling, and playing guitar."""
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


@rt
def login(req):
    return Container(
        Card(
            H3("Login"),
            Subtitle("This page is just for Isaac's use!"),
            P(
                "If you're interested in what's here, reach out and I'm happy to give you a demo.  It's an app I built that help me with my own productivity."
            ),
            A("Log in", href=oauth.login_link(req), cls="uk-button" + ButtonT.primary),
        )
    )


@rt
def logout(sess):
    del sess["user_name"]
    return oauth.logout(sess)


# %% ../nbs/main.ipynb
serve(port=5005)
