"""
This is a demo of how to stream from Modal Function.
"""
import modal

app = modal.App("demo-stream")

image = modal.Image.debian_slim(python_version="3.11").pip_install("python-dotenv", "openai")


@app.function(image=image, container_idle_timeout=60 * 3, secrets=[modal.Secret.from_dotenv()])
def demo_stream():
    print("Opening Stream to OpenAI")
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Tell ons short joke"}],
        stream=True,
    )
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            yield content
    yield 10


@app.local_entrypoint()
def __main__():
    for chunk in demo_stream.remote_gen():
        if isinstance(chunk, str):
            print(chunk)
        else:
            print(f"The final answer is {chunk}")
