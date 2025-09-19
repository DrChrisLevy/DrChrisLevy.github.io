# ruff: noqa: F403, F405
import os
from asyncio import sleep
from datetime import datetime

import bcrypt
from dotenv import load_dotenv
from fasthtml.common import *
from litellm import completion
from monsterui.all import *

from blog import ar as ar_blog
from db.db import Course, DataBase, Lesson, User

load_dotenv()


def before(req, sess):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if "user_id" not in sess:
        return RedirectResponse("/login", status_code=303)
    else:
        existing_user = DataBase.get_user_by_id(sess["user_id"])
        if not existing_user:
            # TODO: This should never happen
            # User doesn't exist, create new one
            sess.pop("user_id")
            return RedirectResponse("/login", status_code=303)
        else:
            # Update last active time for existing user
            DataBase.update_user(
                user_id=sess["user_id"],
                last_active=now,
            )

    # for only allowing admin access to the app
    # if not existing_user.email == "admin":
    #     return RedirectResponse("/login", status_code=303)
    req.scope["user_id"] = sess["user_id"]


beforeware = Beforeware(
    before,
    skip=[
        r"/favicon\.ico",
        r"/static/.*",
        r".*\.css",
        r".*\.js",
        "/login",
        "/signup",
        r"/blog/static_blog_imgs/.*",
        r"/blog/?$",
        r"/blog/blog_post.*",
        "/",
    ],
)

hdrs = (
    Theme.neutral.headers(highlightjs=True),
    Script(src="https://unpkg.com/htmx-ext-sse@2.2.1/sse.js"),
)
app, rt = fast_app(
    before=beforeware,
    hdrs=hdrs,
    routes=(
        Mount(
            "/static", StaticFiles(directory="static")
        ),  # Teaching platform static files
        Mount(
            "/blog/static_blog_imgs", StaticFiles(directory="posts/static_blog_imgs")
        ),  # Blog images
    ),
    secret_key=os.getenv(
        "FAST_APP_SECRET"
    ),  # cryptographically sign the cookie used by the session
    max_age=365 * 24 * 3600,  # Session cookie expiry time --> TODO
    live=True,
)

CHAT_MESSAGE_STYLING = "w-[85%] my-2 border border-gray-200 rounded-xl shadow-sm"
COURSE_CARD_STYLING = "card bg-base-100 shadow-xs uk-width-small"


def _Section(*c, **kwargs):
    return Section(*c, cls="space-y-3 my-10", **kwargs)


def is_admin(user: User):
    # TODO: we should use a more secure method to check admin status
    #   For example, we could store is_admin in the database, and check that
    #   And default it to False.
    return user is not None and user.email == "admin"


def require_admin(handler):
    """Decorator to check admin status from database"""

    def wrapped(req):
        user = DataBase.get_user_by_id(req.scope["user_id"])
        if not user or not is_admin(user):
            return RedirectResponse("/", status_code=303)
        return handler(req)

    return wrapped


def NavigationBar(user: User | None = None, show_auth_controls: bool = True):
    """Navigation bar with optional authentication controls"""
    nav_items = [A("Blog", href="/blog"), A("Courses", href="/courses")]

    # Only show auth controls in authenticated sections
    if show_auth_controls and user:
        if is_admin(user):
            nav_items.append(A("Admin", href=admin_dashboard))
        nav_items.append(A("Logout", href=logout))

    return NavBar(
        *nav_items,
        brand=A("Chris Levy", href="/"),
    )


@rt("/admin")
@require_admin
def admin_dashboard(req):
    user = DataBase.get_user_by_id(req.scope["user_id"])
    # TODO: We should not fetch all users here, but rather use a paginated approach
    all_users = [u for u in DataBase.fetch_all_users()]
    user_table = Table(
        Thead(
            Tr(
                Th("ID"),
                Th("Email"),
                Th("User Name"),
                Th("Created At"),
                Th("Last Active"),
            )
        ),
        Tbody(
            *[
                Tr(
                    Td(u.id),
                    Td(u.email),
                    Td(u.user_name),
                    Td(u.created_at),
                    Td(u.last_active),
                )
                for u in all_users
            ]
        ),
        Tfoot(Tr(Td("Total Users"), Td(len(all_users)))),
    )
    return (
        NavigationBar(user),
        Title("Admin Dashboard"),
        Container(
            H1("Admin Dashboard"),
            Card(
                DivVStacked(
                    H3("User Management"),
                    P(f"Total Users: {len(all_users)}"),
                    user_table,
                ),
            ),
            cls=ContainerT.xl,
        ),
    )


def lesson_completion_toggle(lesson_id: int, user_id: int):
    is_completed = DataBase.get_lesson_completion_status(lesson_id, user_id)
    text = "Mark As Complete"
    return Form(
        LabelSwitch(
            text,
            checked=is_completed,
            name="switch_value",
            hx_post="/toggle_lesson_completion",
            hx_target=f"#lesson-card-completion-toggle-{lesson_id}",
            hx_vals=f'{{"lesson_id": "{lesson_id}", "user_id": "{user_id}"}}',
            hx_swap="outerHTML",
            input_cls="scale-125",
            lbl_cls=(TextT.muted, TextT.medium),
            cls="flex items-center space-x-3 px-2 py-1",
        ),
        id=f"lesson-card-completion-toggle-{lesson_id}",
    )


@rt("/toggle_lesson_completion", methods=["POST"])
def toggle_lesson_completion(lesson_id: int, req, switch_value: bool = None):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    user_id = req.scope["user_id"]

    course_id = DataBase.get_lesson_by_id(lesson_id).course_id

    lesson_completion = DataBase.get_lesson_completion_by_user_id_and_lesson_id(
        user_id, lesson_id
    )
    if lesson_completion:
        DataBase.update_lesson_completion(
            lesson_completion_id=lesson_completion.id,
            user_id=user_id,
            lesson_id=lesson_id,
            course_id=course_id,
            completed_at=now if switch_value else None,
            updated_at=now,
        )
    else:
        DataBase.insert_lesson_completion(
            user_id=user_id,
            lesson_id=lesson_id,
            course_id=course_id,
            completed_at=now if switch_value else None,
            created_at=now,
            updated_at=now,
        )
    return lesson_completion_toggle(lesson_id, user_id)


def lesson_card(lesson: Lesson, user_id: int):
    return Card(
        DivHStacked(
            DivFullySpaced(
                DivLAligned(
                    Div(cls="space-y-3 uk-width-expand")(
                        A(
                            H4(lesson.title),
                            href=view_lesson.to(lesson_id=lesson.id),
                            cls=AT.primary,
                        ),
                        P(lesson.description, cls=TextPresets.muted_sm),
                        DivLAligned(
                            UkIcon("clock"), P(lesson.duration, cls=(TextT.muted,))
                        ),
                    ),
                ),
                lesson_completion_toggle(lesson.id, user_id),
            ),
        ),
        cls=(CardT.hover, COURSE_CARD_STYLING),
    )


def calculate_course_duration(course_id: int) -> str:
    """
    Calculate the total duration of all lessons in a course.

    Args:
        course_id: The ID of the course

    Returns:
        Formatted string of total duration (e.g. "1:04:32")
    """
    # Get all non-deleted lessons for this course
    course_lessons = DataBase.get_lessons_by_course_id(course_id)
    num_lessons = 0
    total_seconds = 0

    for lesson in course_lessons:
        num_lessons += 1
        duration = lesson.duration
        # Split by colon to get hours, minutes, seconds
        parts = duration.split(":")

        if len(parts) == 2:  # Format: "4:50" (minutes:seconds)
            minutes, seconds = map(int, parts)
            total_seconds += minutes * 60 + seconds
        elif len(parts) == 3:  # Format: "1:04:32" (hours:minutes:seconds)
            hours, minutes, seconds = map(int, parts)
            total_seconds += hours * 3600 + minutes * 60 + seconds

    # Convert total seconds back to string format
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours > 0:
        return f"{hours}:{minutes:02d}:{seconds:02d}", num_lessons
    else:
        return f"{minutes}:{seconds:02d}", num_lessons


def progress_bar(progress: int):
    return (
        DivFullySpaced(P("Progress", cls=(TextT.muted)), f"{progress}%"),
        Progress(
            value=progress,  # Progress percentage
            max=100,  # Maximum value (usually 100)
            cls="h-2 rounded-full",
        ),
    )


def course_card(course: Course, user_id: int):
    duration, num_lessons = calculate_course_duration(course.id)
    # Count enrolled users for this course
    enrolled_count = DataBase.get_course_enrollment_count(course.id)

    is_enrolled = DataBase.is_enrolled(user_id, course.id)

    completed_lessons = DataBase.get_completed_lessons_for_user_id_and_course_id(
        user_id, course.id
    )

    progress = round(completed_lessons / num_lessons * 100 if num_lessons > 0 else 0, 0)

    return Card(
        Img(src=course.thumbnail, cls="h-48 w-full object-cover"),
        DivFullySpaced(
            H3(course.title, cls=(TextT.xl, TextT.bold, TextT.primary)),
            DivRAligned(
                UkIcon("check-circle", cls=(TextT.success), height=25, width=25),
                P("Enrolled", cls=(TextT.success, TextT.bold, TextT.lg)),
            )
            if is_enrolled
            else "",
        ),
        P(course.description, cls=(TextT.muted, "mb-4")),
        (
            progress_bar(0),
            A(
                Button(
                    "Enroll",
                    cls=(ButtonT.primary, "mt-3 w-full"),
                ),
                href=enroll.to(course_id=course.id),
            ),
        )
        if not is_enrolled
        else (
            progress_bar(progress),
            A(
                Button(
                    "Continue Learning",
                    cls=(ButtonT.primary, "mt-3 w-full"),
                ),
                href=course_lessons.to(course_id=course.id),
            ),
        ),
        DivFullySpaced(
            DivLAligned(
                UkIcon("clock"),
                P(f"Duration: {duration}", cls=(TextT.gray, TextT.medium)),
            ),
            DivLAligned(
                UkIcon("file"),
                P(f"{num_lessons} lessons", cls=(TextT.gray, TextT.medium)),
            ),
            DivLAligned(
                UkIcon("users"),
                P(f"{enrolled_count} users", cls=(TextT.gray, TextT.medium)),
            ),
        ),
        cls=(CardT.hover, COURSE_CARD_STYLING),
    )


def chat_bubble(message: dict):
    assistant_message = DivRAligned(
        Card(
            UkIcon("bot-message-square"),
            render_md(message["content"]),
            cls=CHAT_MESSAGE_STYLING,
        )
    )
    user_message = DivLAligned(
        Card(UkIcon("user"), render_md(message["content"]), cls=CHAT_MESSAGE_STYLING)
    )
    if message["role"] == "assistant":
        return assistant_message
    else:
        return user_message


def fetch_conversation(conversation_id: int, user_id: int):
    conversation_messages = DataBase.conversation_messages(conversation_id, user_id)
    return [{"role": m.role, "content": m.content} for m in conversation_messages]


def chat(conversation_id: int, lesson_id: int, user_id: int):
    if conversation_id:
        card_messages = [
            chat_bubble(m) for m in fetch_conversation(conversation_id, user_id)
        ]
    else:
        card_messages = []
    return Div(
        # this div is for targeting the chat messages
        Div(*card_messages, id="chat-messages"),
        # this div is for targeting the chat form with the buttons for sending messages and clearing the conversation
        Div(
            (
                Form(
                    TextArea(
                        Placeholder="Ask a question about this lesson...",
                        name="message",
                        cls=CHAT_MESSAGE_STYLING,
                        hx_post="/start-stream",
                        hx_target="#stream-assistant",
                        hx_vals=f'{{"lesson_id": "{lesson_id}", "conversation_id": "{conversation_id}"}}',
                        hx_trigger="keydown[metaKey&&key=='Enter'], keydown[ctrlKey&&key=='Enter']",
                    ),
                    DivFullySpaced(
                        Div(
                            Button(
                                "Send",
                                cls=ButtonT.primary,
                                hx_post="/start-stream",
                                hx_target="#stream-assistant",
                                hx_vals=f'{{"lesson_id": "{lesson_id}", "conversation_id": "{conversation_id}"}}',
                            ),
                        ),
                        Button(
                            "Clear Conversation",
                            cls=ButtonT.destructive,
                            hx_post="/clear_conversation",
                            hx_target="#chat",
                            hx_swap="outerHTML",
                            hx_vals=f'{{"conversation_id": "{conversation_id}"}}',
                        )
                        if card_messages
                        else "",
                    ),
                ),
            ),
            id="chat-form",
        ),
        # this div is for targeting the stream of assistant messages
        Div(
            id="stream-assistant",
        ),
        id="chat",
    )


@rt(methods=["POST"])
def clear_conversation(conversation_id: int, req):
    user_id = req.scope["user_id"]
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conversation_messages = DataBase.conversation_messages(conversation_id, user_id)
    for m in conversation_messages:
        DataBase.update_message(m.id, deleted_at=now)
    conversation = DataBase.get_conversation_by_id_and_user_id(conversation_id, user_id)
    lesson_id = conversation.lesson_id
    DataBase.update_conversation(
        conversation_id=conversation.id, deleted_at=now, updated_at=now
    )
    return chat(None, lesson_id, user_id)


@rt("/start-stream", methods=["POST"])
def start_stream(message: str, lesson_id: int, conversation_id: int, req):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    user_id = req.scope["user_id"]
    if not conversation_id:
        new_conversation = DataBase.insert_conversation(
            user_id=user_id,
            lesson_id=lesson_id,
            created_at=now,
            updated_at=now,
            deleted_at=None,
        )
        conversation_id = new_conversation.id
    DataBase.insert_message(
        user_id=user_id,
        conversation_id=conversation_id,
        content=message,
        role="user",
        created_at=now,
    )

    card_messages = [
        chat_bubble(m) for m in fetch_conversation(conversation_id, user_id)
    ]

    assistant_msg = Div(
        # We need an inner DIV here to keep the SSE connection alive
        # We first update the messages conversation with the new user message
        # Then we immediately clear out the chat form
        # Then add a nice little loading indicator to get immediate feedback
        # which then disappears when the stream starts
        Div(*card_messages, id="chat-messages", hx_swap_oob="true"),
        Div(id="chat-form", hx_swap_oob="true"),
        Div(Loading(cls=LoadingT.ring), id="stream-content"),
        hx_ext="sse",
        sse_connect=f"/get-message?conversation_id={conversation_id}&lesson_id={lesson_id}",
        sse_swap="EventName",
        sse_close="close",
        hx_swap="innerHTML",
        hx_target="#stream-content",
        id="stream-assistant",
    )
    return assistant_msg


async def message_generator(conversation_id: int, lesson_id: int, req):
    user_id = req.scope["user_id"]
    lesson = DataBase.get_lesson_by_id(lesson_id)
    system_prompt = lesson.system_prompt

    message_history = fetch_conversation(conversation_id, user_id)
    message_history.insert(
        0,
        {
            "role": "system",
            "content": system_prompt,
        },
    )
    resp = completion(
        model="openai/gpt-4.1",
        messages=message_history,
        stream=True,
        # TODO: add thinking_budget
    )

    final_message = ""

    for chunk in resp:
        chunk = chunk.choices[0].delta.content
        if chunk is not None:
            final_message += chunk
            yield sse_message(
                chat_bubble({"role": "assistant", "content": final_message}),
                event="EventName",
            )
        await sleep(0.025)

    # Add assistant response
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    DataBase.insert_message(
        user_id=user_id,
        conversation_id=conversation_id,
        content=final_message,
        role="assistant",
        created_at=now,
    )
    DataBase.update_conversation(
        conversation_id=conversation_id,
        updated_at=now,
    )
    yield sse_message(
        Div(chat(conversation_id, lesson_id, user_id), id="chat", hx_swap_oob="true"),
        event="EventName",
    )
    yield sse_message(Div(), event="close")


@rt("/get-message")
async def get_message(conversation_id: int, lesson_id: int, req):
    return EventStream(message_generator(conversation_id, lesson_id, req))


def video_iframe(url: str):
    return f"""<div style="max-width: 2000px; margin: 0 auto;">
    <div class="video-container" style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden;">
        <iframe src="{url}?autoplay=0&amp;showinfo=0&amp;rel=0&amp;modestbranding=1&amp;playsinline=1" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;" allowfullscreen></iframe>
    </div>
</div>"""


@rt(methods=["GET"])
def view_lesson(lesson_id: int, req):
    user_id = req.scope["user_id"]
    user = DataBase.get_user_by_id(user_id)
    lesson = DataBase.get_lesson_by_id(lesson_id)
    is_enrolled = DataBase.is_enrolled(user_id, lesson.course_id)
    if not is_enrolled:
        return RedirectResponse("/", status_code=303)

    next_lesson = DataBase.get_next_lesson(lesson.course_id, lesson.sort_order)
    previous_lesson = DataBase.get_previous_lesson(lesson.course_id, lesson.sort_order)
    title = f"# {lesson.title}"
    conv = DataBase.get_conversation_by_lesson_id_and_user_id(lesson.id, user_id)
    conversation_id = conv.id if conv else None
    return NavigationBar(user), Container(
        DivFullySpaced(
            A(
                DivLAligned(UkIcon("arrow-left"), H5("Back to Lessons")),
                href=course_lessons.to(course_id=lesson.course_id),
                cls=(AT.primary, TextT.bold),
            ),
            DivRAligned(
                lesson_completion_toggle(lesson.id, user_id),
            ),
        ),
        DivFullySpaced(
            A(
                DivLAligned(UkIcon("arrow-left"), H5("Previous Lesson")),
                href=view_lesson.to(lesson_id=previous_lesson.id),
                cls=(AT.primary, TextT.bold),
            )
            if previous_lesson
            else Div(),
            A(
                DivRAligned(H5("Next Lesson"), UkIcon("arrow-right")),
                href=view_lesson.to(lesson_id=next_lesson.id),
                cls=(AT.primary, TextT.bold),
            )
            if next_lesson
            else Div(),
            cls="my-10",
        ),
        DivCentered(render_md(title)),
        _Section(render_md(video_iframe(lesson.video_url))),
        _Section(render_md(lesson.content)),
        _Section(
            Card(
                H1("AI Assistant"),
                chat(conversation_id, lesson_id, user_id),
                cls="backdrop-blur-md bg-white/70 rounded-2xl border border-white/30 shadow-lg p-6  w-full",
            )
        ),
        cls=ContainerT.expand,
    )


def hash_password(password: str) -> str:
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode(), salt)
    return hashed.decode()


def verify_password(password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed_password.encode())


def display_login_page(error: str = None):
    return (
        NavigationBar(user=None, show_auth_controls=False),  # Public login page
        Container(
            DivCentered(
                DivVStacked(
                    H3("Login", cls=TextT.primary),
                    Form(
                        Input(
                            type="text",
                            name="email",
                            placeholder="Email address*",
                            required=True,
                        ),
                        Input(
                            type="password",
                            name="password",
                            placeholder="Password*",
                            required=True,
                        ),
                        Button("Continue", type="submit", cls=ButtonT.primary),
                        method="post",
                        action="/login",
                    ),
                    P(error, cls=TextT.error) if error else P(""),
                    DivHStacked(
                        P("Don't have an account?"),
                        P(A("Sign up", href="/signup", cls=AT.primary)),
                    ),
                ),
                cls="h-screen",  # Make container full height
            ),
        ),
    )


def display_signup_page(error: str = None):
    return (
        NavigationBar(user=None, show_auth_controls=False),  # Public signup page
        Container(
            DivCentered(
                DivVStacked(
                    H3("Register", cls=TextT.primary),
                    Form(
                        Input(
                            type="text",
                            name="email",
                            placeholder="Email address*",
                            required=True,
                        ),
                        Input(
                            type="password",
                            name="password",
                            placeholder="Password*",
                            required=True,
                        ),
                        Input(
                            type="password",
                            name="password_confirm",
                            placeholder="Confirm password*",
                            required=True,
                        ),
                        Button("Register", type="submit", cls=ButtonT.primary),
                        method="post",
                        action="/signup",
                    ),
                    P(error, cls=TextT.error) if error else P(""),
                    DivHStacked(
                        P("Already have an account?"),
                        A("Login", href="/login", cls=AT.primary),
                    ),
                ),
                cls="h-screen",  # Make container full height
            ),
        ),
    )


@rt("/login", methods=["GET", "POST"])
def login_page(req, sess, email: str = None, password: str = None):
    if req.method == "POST":
        user = DataBase.get_user_by_email(email)
        if not user:
            return display_login_page(error="There is no user with that email address.")
        if not verify_password(password, user.password):
            return display_login_page(
                error="The password is not correct for that email address."
            )
        sess["user_id"] = user.id
        req.scope["user_id"] = user.id
        return RedirectResponse("/courses", status_code=303)
    return display_login_page()


@rt("/logout", methods=["GET"])
def logout(req, sess):
    sess.pop("user_id", None)
    req.scope.pop("user_id", None)
    return RedirectResponse("/login", status_code=303)


@rt("/signup", methods=["GET", "POST"])
def signup_page(
    req,
    sess,
    email: str = None,
    password: str = None,
    password_confirm: str = None,
):
    if req.method == "POST":
        user = DataBase.get_user_by_email(email)
        if user:
            return display_signup_page(
                error="There is already a user with that email address."
            )
        if password != password_confirm:
            return display_signup_page(error="The passwords do not match.")
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        user = DataBase.insert_user(
            email=email,
            user_name=None,  # TODO: Add user name
            password=hash_password(password),
            created_at=now,
            last_active=now,
        )
        sess["user_id"] = user.id
        req.scope["user_id"] = user.id
        return RedirectResponse("/courses", status_code=303)
    return display_signup_page()


@rt("/course_lessons/{course_id}")
def course_lessons(course_id: int, sess, req):
    if not sess.get("user_id"):
        return login_page()
    user_id = sess["user_id"]
    user = DataBase.get_user_by_id(user_id)

    # TODO: This should be a single query - lessons sorted by section sorted by sort order
    sects = DataBase.get_sections_by_course_id(course_id)
    sects_list = []
    for s in sects:
        lessons_list = DataBase.get_lessons_by_section_id(s.id)
        sects_list.append(
            Container(
                _Section(
                    H2(s.title),
                    render_md(s.description),
                    *[lesson_card(l, user_id) for l in lessons_list],
                )
            )
        )

    return (
        NavigationBar(user),
        Title("Course"),
        DivCentered(H1("Lessons")),
        *sects_list,
    )


@rt("")
def index(req, sess):
    """Public homepage - Chris Levy Platform"""
    user_id = sess.get("user_id")
    user = DataBase.get_user_by_id(user_id) if user_id else None

    def _section(*c):
        return Section(
            Article(*c, cls="prose max-w-5xl mx-auto space-y-5 pt-16"),
            cls=("uk-padding-remove-vertical",),
        )

    return (
        NavigationBar(user, show_auth_controls=False),  # Public homepage - no logout
        Title("Chris Levy"),
        Div(
            Section(
                DivCentered(
                    Img(
                        src="/blog/static_blog_imgs/pic_me.jpg",
                        cls="rounded-full w-64 h-64 object-cover",
                    ),
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
        ),
    )


@rt("/courses")
def courses_page(req):
    """Courses page - requires authentication"""
    user_id = req.scope["user_id"]
    user = DataBase.get_user_by_id(user_id)

    enrolled_courses, not_enrolled_courses = DataBase.users_courses(user_id)

    return (
        NavigationBar(user),
        Title("Courses"),
        DivCentered(H1("Courses")),
        _Section(
            Container(
                DivCentered(H2("Continue Your Learning", cls="my-10")),
                Grid(
                    *[course_card(c, user_id) for c in enrolled_courses],
                    cols_max=2,
                ),
            )
        )
        if enrolled_courses
        else Div(),
        _Section(
            Container(
                DivCentered(H2("Start a New Course", cls="my-10")),
                Grid(
                    *[course_card(c, user_id) for c in not_enrolled_courses],
                    cols_max=2,
                ),
            )
        )
        if not_enrolled_courses
        else Div(),
    )


@rt("/enroll/{course_id}", methods=["GET"])
def enroll(course_id: int, req):
    user_id = req.scope["user_id"]
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Check if already enrolled
    is_enrolled = DataBase.is_enrolled(user_id, course_id)

    # If not enrolled, create enrollment record
    if not is_enrolled:
        DataBase.insert_enrollment(
            user_id=user_id,
            course_id=course_id,
            enrolled_at=now,
            completed_at=None,
            deleted_at=None,
        )

    # Redirect to course lessons
    return RedirectResponse(course_lessons.to(course_id=course_id), status_code=303)


# Add blog routes to the app
ar_blog.to_app(app)

serve()
