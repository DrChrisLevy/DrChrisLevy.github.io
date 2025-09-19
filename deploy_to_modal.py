import modal

app = modal.App("drchrislevy")
vol = modal.Volume.from_name("teaching-volume", create_if_missing=True)


@app.function(
    volumes={"/root/data": vol},
    image=modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_pyproject("pyproject.toml")
    .add_local_dir("static", remote_path="/root/static")
    .add_local_dir("posts", remote_path="/root/posts", ignore=["__pycache__"])  # Blog posts and images
    .add_local_file("main.py", remote_path="/root/main.py")
    .add_local_file("blog.py", remote_path="/root/blog.py")  # Blog functionality
    .add_local_file("db/db.py", remote_path="/root/db/db.py")
    .add_local_file(
        "scripts/course_lessons_db.py", remote_path="/root/scripts/course_lessons_db.py"
    )
    .add_local_file(
        "scripts/modal_docs_scraper.py",
        remote_path="/root/scripts/modal_docs_scraper.py",
    )
    # Section 1 lessons
    .add_local_file(
        "courses/modal/section1/lesson1.md",
        remote_path="/root/courses/modal/section1/lesson1.md",
    )
    .add_local_file(
        "courses/modal/section1/lesson2.md",
        remote_path="/root/courses/modal/section1/lesson2.md",
    )
    .add_local_file(
        "courses/modal/section1/lesson3.md",
        remote_path="/root/courses/modal/section1/lesson3.md",
    )
    .add_local_file(
        "courses/modal/section1/lesson4.md",
        remote_path="/root/courses/modal/section1/lesson4.md",
    )
    .add_local_file(
        "courses/modal/section1/lesson5.md",
        remote_path="/root/courses/modal/section1/lesson5.md",
    )
    .add_local_file(
        "courses/modal/section1/lesson6.md",
        remote_path="/root/courses/modal/section1/lesson6.md",
    )
    # Section 2 lessons
    .add_local_file(
        "courses/modal/section2/lesson7.md",
        remote_path="/root/courses/modal/section2/lesson7.md",
    )
    .add_local_file(
        "courses/modal/section2/lesson8.md",
        remote_path="/root/courses/modal/section2/lesson8.md",
    )
    # Section 3 lessons
    .add_local_file(
        "courses/modal/section3/lesson9.md",
        remote_path="/root/courses/modal/section3/lesson9.md",
    )
    .add_local_file(
        "courses/modal/section3/lesson12.md",
        remote_path="/root/courses/modal/section3/lesson12.md",
    )
    .add_local_file(
        "courses/modal/section3/lesson13.md",
        remote_path="/root/courses/modal/section3/lesson13.md",
    )
    # Section 4 lessons
    .add_local_file(
        "courses/modal/section4/lesson14.md",
        remote_path="/root/courses/modal/section4/lesson14.md",
    )
    # section 5 lessons
    .add_local_file(
        "courses/modal/section5/lesson15.md",
        remote_path="/root/courses/modal/section5/lesson15.md",
    )
    .add_local_file(
        "courses/modal/section5/lesson16.md",
        remote_path="/root/courses/modal/section5/lesson16.md",
    )
    .add_local_file(
        "courses/modal/section5/lesson17.md",
        remote_path="/root/courses/modal/section5/lesson17.md",
    )
    .add_local_file(
        "courses/modal/section5/lesson18.md",
        remote_path="/root/courses/modal/section5/lesson18.md",
    )
    .add_local_file(
        "courses/modal/section5/lesson19.md",
        remote_path="/root/courses/modal/section5/lesson19.md",
    )
    .add_local_file(
        "courses/modal/section5/lesson20.md",
        remote_path="/root/courses/modal/section5/lesson20.md",
    )
    # section 6 lessons
    .add_local_file(
        "courses/modal/section6/lesson21.md",
        remote_path="/root/courses/modal/section6/lesson21.md",
    )
    # section 7 lessons
    .add_local_file(
        "courses/modal/section7/lesson22.md",
        remote_path="/root/courses/modal/section7/lesson22.md",
    )
    .add_local_file(
        "courses/modal/section7/lesson22_part2.md",
        remote_path="/root/courses/modal/section7/lesson22_part2.md",
    )
    .add_local_file(
        "courses/modal/section7/lesson22_part3.md",
        remote_path="/root/courses/modal/section7/lesson22_part3.md",
    )
    # section 8 lessons
    .add_local_file(
        "courses/modal/section8/lesson23.md",
        remote_path="/root/courses/modal/section8/lesson23.md",
    )
    # section 9 lessons
    .add_local_file(
        "courses/modal/section9/lesson24.md",
        remote_path="/root/courses/modal/section9/lesson24.md",
    ),
    secrets=[modal.Secret.from_dotenv()],
    scaledown_window=10 * 60,
)
@modal.concurrent(max_inputs=100)  # TODO
@modal.asgi_app()
def serve():
    from main import app

    return app
