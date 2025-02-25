import modal

app = modal.App("drchrislevy")


@app.function(
    image=modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_pyproject("pyproject.toml")
    .add_local_dir("posts", remote_path="/root/posts")
    .add_local_file("utils.py", remote_path="/root/utils.py")
    .add_local_file("main.py", remote_path="/root/main.py")
    .add_local_file("blog.py", remote_path="/root/blog.py"),
    allow_concurrent_inputs=100,  # TODO
    secrets=[modal.Secret.from_dotenv()],
)
@modal.asgi_app()
def serve():
    from main import app

    return app
