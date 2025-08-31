# ruff: noqa: F403, F405

from datetime import datetime

import modal

from db.db import DataBase
from scripts.modal_docs_scraper import scrape_modal_docs


def create_system_prompt(
    lesson_content: str, lesson_transcript: str, modal_urls: str
) -> str:
    modal_docs = [scrape_modal_docs(url) for url in modal_urls]
    referenced_resources = ""
    for url, doc in zip(modal_urls, modal_docs):
        referenced_resources += f"{url}\n{doc}\n\n" + "--" * 50 + "\n\n"
    prompt = """
< Role and Objective >

You are an AI teaching assistant.  
Your job is to help students understand and apply the material
from this lesson. Here is the context for this lesson
to aid in your responses.


< Lesson Markdown Content >

{lesson_content}

< Lesson Audio Transcript >

{lesson_transcript}

< Referenced Resources >

{referenced_resources}

< Instructions >

- chat with the user and answer their questions
- use the lesson context above to answer the user's questions"""
    return prompt.format(
        lesson_content=lesson_content,
        lesson_transcript=lesson_transcript,
        referenced_resources=referenced_resources,
    )


transcribe_audio = modal.Function.from_name("transcribe-audio", "transcribe_audio")


def read_markdown_content(file_path):
    try:
        with open(file_path, "r") as f:
            return f.read()
    except FileNotFoundError:
        print(f"Warning: File {file_path} not found")
        return ""


def main():
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    COURSES = [
        {
            "title": "AI Engineering with Modal",
            "description": "A hands-on guide to building and deploying scalable AI systems with Modal.",
            "thumbnail": "static/images/modal_course_thumbnail.webp",
            "lesson_content": [
                {
                    "section_title": "Introduction to Modal",
                    "section_description": """In this section we will setup our Modal account and environment.
                    We will also run our first few functions locally and remotely on Modal's infrastructure.""",
                    "lessons": [
                        {
                            "lesson_title": "Introduction to Modal",
                            "lesson_description": "In this lesson we we will setup our Modal account and environment.",
                            "duration": "04:38",
                            "content": read_markdown_content(
                                "courses/modal/section1/lesson1.md"
                            ),
                            "video_url": "https://www.youtube.com/embed/-FL50zIyfUI",
                            "system_prompt": create_system_prompt(
                                lesson_content=read_markdown_content(
                                    "courses/modal/section1/lesson1.md"
                                ),
                                lesson_transcript=transcribe_audio.remote(
                                    "/lesson1.m4a"
                                )["transcript"],
                                modal_urls=["https://modal.com/docs/guide"],
                            ),
                        },
                        {
                            "lesson_title": "Run Your First Modal Function",
                            "lesson_description": "In this lesson we will run our first Modal function.",
                            "duration": "11:23",
                            "content": read_markdown_content(
                                "courses/modal/section1/lesson2.md"
                            ),
                            "video_url": "https://www.youtube.com/embed/gNo1T2y-7LE",
                            "system_prompt": create_system_prompt(
                                lesson_content=read_markdown_content(
                                    "courses/modal/section1/lesson2.md"
                                ),
                                lesson_transcript=transcribe_audio.remote(
                                    "/lesson2.m4a"
                                )["transcript"],
                                modal_urls=[
                                    "https://modal.com/docs/reference/cli/container#modal-container-list",
                                    "https://modal.com/docs/reference/cli/shell#modal-shell",
                                    "https://modal.com/docs/examples/hello_world",
                                ],
                            ),
                        },
                        {
                            "lesson_title": "Configuring Auto Scaling and Input Concurrency",
                            "lesson_description": "In this lesson we will see how you can configure the auto scaling and input concurrency behavior of your Modal functions.",
                            "duration": "10:24",
                            "content": read_markdown_content(
                                "courses/modal/section1/lesson3.md"
                            ),
                            "video_url": "https://www.youtube.com/embed/h2_An8DahXI",
                            "system_prompt": create_system_prompt(
                                lesson_content=read_markdown_content(
                                    "courses/modal/section1/lesson3.md"
                                ),
                                lesson_transcript=transcribe_audio.remote(
                                    "/lesson3.m4a"
                                )["transcript"],
                                modal_urls=[
                                    "https://modal.com/docs/guide/scale",
                                    "https://modal.com/docs/guide/concurrent-inputs",
                                ],
                            ),
                        },
                        {
                            "lesson_title": "Defining Images",
                            "lesson_description": "In this lesson we see how to define and build custom container images.",
                            "duration": "11:48",
                            "content": read_markdown_content(
                                "courses/modal/section1/lesson4.md"
                            ),
                            "video_url": "https://www.youtube.com/embed/kwUnKUUUiVk",
                            "system_prompt": create_system_prompt(
                                lesson_content=read_markdown_content(
                                    "courses/modal/section1/lesson4.md"
                                ),
                                lesson_transcript=transcribe_audio.remote(
                                    "/lesson4.m4a"
                                )["transcript"],
                                modal_urls=["https://modal.com/docs/guide/images"],
                            ),
                        },
                        {
                            "lesson_title": "GPU Acceleration and Setting CPU/Memory Resources",
                            "lesson_description": "In this lesson we see how to utilize GPUs as well as set resource limits on CPUs and Memory.",
                            "duration": "8:45",
                            "content": read_markdown_content(
                                "courses/modal/section1/lesson5.md"
                            ),
                            "video_url": "https://www.youtube.com/embed/Zqm8ZA1Fmuk",
                            "system_prompt": create_system_prompt(
                                lesson_content=read_markdown_content(
                                    "courses/modal/section1/lesson5.md"
                                ),
                                lesson_transcript=transcribe_audio.remote(
                                    "/lesson5.m4a"
                                )["transcript"],
                                modal_urls=["https://modal.com/docs/guide/resources"],
                            ),
                        },
                        {
                            "lesson_title": "Intro to Volumes",
                            "lesson_description": "In this lesson we see how persist data across function calls and containers by using volume objects.",
                            "duration": "9:51",
                            "content": read_markdown_content(
                                "courses/modal/section1/lesson6.md"
                            ),
                            "video_url": "https://www.youtube.com/embed/gAAbJ5TxgXM",
                            "system_prompt": create_system_prompt(
                                lesson_content=read_markdown_content(
                                    "courses/modal/section1/lesson6.md"
                                ),
                                lesson_transcript=transcribe_audio.remote(
                                    "/lesson6.m4a"
                                )["transcript"],
                                modal_urls=["https://modal.com/docs/guide/volumes"],
                            ),
                        },
                    ],
                },
                {
                    "section_title": "Transcribing Audio Files with ASR model: nvidia/parakeet-tdt-0.6b-v2",
                    "section_description": "In this section we will build a simple ASR pipeline using Modal.",
                    "lessons": [
                        {
                            "lesson_title": "Automatic Speech Recognition: Part 1",
                            "lesson_description": "In this lesson we will build a simple ASR pipeline using Modal.",
                            "duration": "20:52",
                            "content": read_markdown_content(
                                "courses/modal/section2/lesson7.md"
                            ),
                            "video_url": "https://www.youtube.com/embed/FnjTR0oBME8",
                            "system_prompt": create_system_prompt(
                                lesson_content=read_markdown_content(
                                    "courses/modal/section2/lesson7.md"
                                ),
                                lesson_transcript=transcribe_audio.remote(
                                    "/lesson7.m4a"
                                )["transcript"],
                                modal_urls=[],
                            ),
                        },
                        {
                            "lesson_title": "Automatic Speech Recognition: Part 2",
                            "lesson_description": "In this lesson we add a couple more features such as chunking and a storage volume to store the results.",
                            "duration": "23:31",
                            "content": read_markdown_content(
                                "courses/modal/section2/lesson8.md"
                            ),
                            "video_url": "https://www.youtube.com/embed/_LHXPzsI0Bs",
                            "system_prompt": create_system_prompt(
                                lesson_content=read_markdown_content(
                                    "courses/modal/section2/lesson8.md"
                                ),
                                lesson_transcript=transcribe_audio.remote(
                                    "/lesson8.m4a"
                                )["transcript"],
                                modal_urls=[
                                    "https://modal.com/docs/guide/scale#starmap"
                                ],
                            ),
                        },
                    ],
                },
                {
                    "section_title": "Fine-Tuning Transformer Encoder Models for Classification",
                    "section_description": "In this section we will fine-tune a transformer encoder model for classification tasks.",
                    "lessons": [
                        {
                            "lesson_title": "Fine-Tuning ModernBERT for Classification: Part 1",
                            "lesson_description": "In this lesson we talk a little about the ModernBERT encoder model and walk through some of the training code at a high level.",
                            "duration": "19:30",
                            "content": read_markdown_content(
                                "courses/modal/section3/lesson9.md"
                            ),
                            "video_url": "https://www.youtube.com/embed/sr-HFdxQDOM",
                            "system_prompt": create_system_prompt(
                                lesson_content=read_markdown_content(
                                    "courses/modal/section3/lesson9.md"
                                ),
                                lesson_transcript=transcribe_audio.remote(
                                    "/lesson9.m4a"
                                )["transcript"],
                                modal_urls=[
                                    "https://modal.com/docs/reference/modal.App#cls",
                                    "https://modal.com/docs/guide/lifecycle-functions",
                                    "https://modal.com/docs/reference/cli/shell#modal-shell",
                                    "https://modal.com/docs/reference/cli/launch#modal-launch-jupyter",
                                    "https://modal.com/docs/guide/developing-debugging#developing-and-debugging",
                                ],
                            ),
                        },
                        # lesson 10
                        {
                            "lesson_title": "Fine-Tuning ModernBERT for Classification: Part 2",
                            "lesson_description": "In this lesson we review different ways we can debug the training code on Modal containers interactively.",
                            "duration": "10:57",
                            "content": read_markdown_content(
                                "courses/modal/section3/lesson9.md"
                            ),
                            "video_url": "https://www.youtube.com/embed/DirBO5YFqck",
                            "system_prompt": create_system_prompt(
                                lesson_content=read_markdown_content(
                                    "courses/modal/section3/lesson9.md"
                                ),
                                lesson_transcript=transcribe_audio.remote(
                                    "/lesson10.m4a"
                                ),
                                modal_urls=[
                                    "https://modal.com/docs/reference/modal.App#cls",
                                    "https://modal.com/docs/guide/lifecycle-functions",
                                    "https://modal.com/docs/reference/cli/shell#modal-shell",
                                    "https://modal.com/docs/reference/cli/launch#modal-launch-jupyter",
                                    "https://modal.com/docs/guide/developing-debugging#developing-and-debugging",
                                ],
                            ),
                        },
                        # lesson 11
                        {
                            "lesson_title": "Fine-Tuning ModernBERT for Classification: Part 3",
                            "lesson_description": "In this lesson we run the training code as well as look at the results in the Modal UI dashboard as well as the wandb dashboard.",
                            "duration": "10:07",
                            "content": read_markdown_content(
                                "courses/modal/section3/lesson9.md"
                            ),
                            "video_url": "https://www.youtube.com/embed/UTjbU5sLDw0",
                            "system_prompt": create_system_prompt(
                                lesson_content=read_markdown_content(
                                    "courses/modal/section3/lesson9.md"
                                ),
                                lesson_transcript=transcribe_audio.remote(
                                    "/lesson11.m4a"
                                )["transcript"],
                                modal_urls=[
                                    "https://modal.com/docs/reference/modal.App#cls",
                                    "https://modal.com/docs/guide/lifecycle-functions",
                                    "https://modal.com/docs/reference/cli/shell#modal-shell",
                                    "https://modal.com/docs/reference/cli/launch#modal-launch-jupyter",
                                    "https://modal.com/docs/guide/developing-debugging#developing-and-debugging",
                                ],
                            ),
                        },
                        # lesson 12
                        {
                            "lesson_title": "Deploying the Trained Encoder Model to a Modal Web Endpoint",
                            "lesson_description": "In this lesson we learn how to deploy our trained Encoder model to a web endpoint.",
                            "duration": "23:33",
                            "content": read_markdown_content(
                                "courses/modal/section3/lesson12.md"
                            ),
                            "video_url": "https://www.youtube.com/embed/IZabkAdRRqc",
                            "system_prompt": create_system_prompt(
                                lesson_content=read_markdown_content(
                                    "courses/modal/section3/lesson12.md"
                                ),
                                lesson_transcript=transcribe_audio.remote(
                                    "/lesson12.m4a"
                                )["transcript"],
                                modal_urls=[
                                    "https://modal.com/docs/examples/basic_web"
                                ],
                            ),
                        },
                        {
                            "lesson_title": "Fine-Tuning ModernBERT on a Different Classification Dataset",
                            "lesson_description": "In this lesson we quickly show some results of fine-tuning ModernBERT on a different classification dataset, MAGE, to predict if text is generated by a human or an LLM.",
                            "duration": "4:47",
                            "content": read_markdown_content(
                                "courses/modal/section3/lesson13.md"
                            ),
                            "video_url": "https://www.youtube.com/embed/oJSHFyID16c",
                            "system_prompt": create_system_prompt(
                                lesson_content=read_markdown_content(
                                    "courses/modal/section3/lesson13.md"
                                ),
                                lesson_transcript=transcribe_audio.remote(
                                    "/lesson13.m4a"
                                )["transcript"],
                                modal_urls=[
                                    "https://modal.com/docs/examples/basic_web"
                                ],
                            ),
                        },
                    ],
                },
                {
                    "section_title": "Deploying an OpenAI Compatible Server on Modal for LLM Inference with vLLM",
                    "section_description": "In this section we will deploy some OpenAI Compatible endpoints on Modal for LLM Inference with vLLM",
                    "lessons": [
                        {
                            "lesson_title": "Deploying vLLM on Modal",
                            "lesson_description": "In this lesson we show how to deploy an endpoint on Modal for LLM inference with vLLM.",
                            "duration": "34:34",
                            "content": read_markdown_content(
                                "courses/modal/section4/lesson14.md"
                            ),
                            "video_url": "https://www.youtube.com/embed/myClJvra0Rk",
                            "system_prompt": create_system_prompt(
                                lesson_content=read_markdown_content(
                                    "courses/modal/section4/lesson14.md"
                                ),
                                lesson_transcript=transcribe_audio.remote(
                                    "/lesson14.m4a"
                                )["transcript"],
                                modal_urls=[
                                    "https://modal.com/docs/examples/vllm_inference",
                                    "https://modal.com/docs/reference/modal.web_server#modalweb_server",
                                    "https://modal.com/docs/examples/basic_web",
                                ],
                            ),
                        },
                    ],
                },
                {
                    "section_title": "Building a Coding Agent with Modal Sandboxes",
                    "section_description": "In this section we will learn about Modal Sandboxes and how to build a coding agent with them.",
                    "lessons": [
                        {
                            "lesson_title": "Intro to Modal Sandboxes",
                            "lesson_description": "In this lesson we introduce Modal Sandboxes.",
                            "duration": "5:55",
                            "content": read_markdown_content(
                                "courses/modal/section5/lesson15.md"
                            ),
                            "video_url": "https://www.youtube.com/embed/OfNTnw7n9y8",
                            "system_prompt": create_system_prompt(
                                lesson_content=read_markdown_content(
                                    "courses/modal/section5/lesson15.md"
                                ),
                                lesson_transcript=transcribe_audio.remote(
                                    "/lesson15.m4a"
                                )["transcript"],
                                modal_urls=[
                                    "https://modal.com/docs/guide/sandbox",
                                    "https://modal.com/docs/guide/sandbox-spawn",
                                    "https://modal.com/docs/guide/sandbox-networking",
                                    "https://modal.com/docs/guide/sandbox-files",
                                    "https://modal.com/docs/guide/sandbox-snapshots",
                                    "https://modal.com/docs/guide/sandbox-memory-snapshots",
                                    "https://modal.com/docs/examples/simple_code_interpreter#build-a-stateful-sandboxed-code-interpreter",
                                    "https://modal.com/docs/examples/safe_code_execution#run-arbitrary-code-in-a-sandboxed-environment",
                                    "https://modal.com/docs/examples/agent#build-a-coding-agent-with-modal-sandboxes-and-langgraph",
                                    "https://modal.com/docs/examples/jupyter_sandbox#run-a-jupyter-notebook-in-a-modal-sandbox",
                                ],
                            ),
                        },
                        {
                            "lesson_title": "Building a Python Coding Sandbox",
                            "lesson_description": "In this lesson we look at our main class for interacting with the Python coding sandbox.",
                            "duration": "9:30",
                            "content": read_markdown_content(
                                "courses/modal/section5/lesson16.md"
                            ),
                            "video_url": "https://www.youtube.com/embed/akdNfVvQdwI",
                            "system_prompt": create_system_prompt(
                                lesson_content=read_markdown_content(
                                    "courses/modal/section5/lesson16.md"
                                ),
                                lesson_transcript=transcribe_audio.remote(
                                    "/lesson16.m4a"
                                )["transcript"],
                                modal_urls=[
                                    "https://modal.com/docs/guide/sandbox",
                                    "https://modal.com/docs/examples/simple_code_interpreter#build-a-stateful-sandboxed-code-interpreter",
                                    "https://modal.com/docs/examples/safe_code_execution#run-arbitrary-code-in-a-sandboxed-environment",
                                ],
                            ),
                        },
                        {
                            "lesson_title": "Digging More into the Coding Sandbox",
                            "lesson_description": "In this lesson we dig a little deeper into the code behind the coding sandbox.",
                            "duration": "9:23",
                            "content": read_markdown_content(
                                "courses/modal/section5/lesson17.md"
                            ),
                            "video_url": "https://www.youtube.com/embed/1zjXpG3Ya9s",
                            "system_prompt": create_system_prompt(
                                lesson_content=read_markdown_content(
                                    "courses/modal/section5/lesson17.md"
                                ),
                                lesson_transcript=transcribe_audio.remote(
                                    "/lesson17.m4a"
                                )["transcript"],
                                modal_urls=[
                                    "https://modal.com/docs/guide/sandbox",
                                    "https://modal.com/docs/examples/simple_code_interpreter#build-a-stateful-sandboxed-code-interpreter",
                                    "https://modal.com/docs/examples/safe_code_execution#run-arbitrary-code-in-a-sandboxed-environment",
                                ],
                            ),
                        },
                        {
                            "lesson_title": "Customizing the Coding Sandbox",
                            "lesson_description": "In this lesson we see how to customize the coding sandbox by defining any custom image and dependencies we want to use.",
                            "duration": "7:42",
                            "content": read_markdown_content(
                                "courses/modal/section5/lesson18.md"
                            ),
                            "video_url": "https://www.youtube.com/embed/CfdTdWg7BhQ",
                            "system_prompt": create_system_prompt(
                                lesson_content=read_markdown_content(
                                    "courses/modal/section5/lesson18.md"
                                ),
                                lesson_transcript=transcribe_audio.remote(
                                    "/lesson18.m4a"
                                )["transcript"],
                                modal_urls=[
                                    "https://modal.com/docs/guide/sandbox",
                                    "https://modal.com/docs/examples/simple_code_interpreter#build-a-stateful-sandboxed-code-interpreter",
                                    "https://modal.com/docs/examples/safe_code_execution#run-arbitrary-code-in-a-sandboxed-environment",
                                ],
                            ),
                        },
                        {
                            "lesson_title": "Building a Simple Agent with Coding Abilities",
                            "lesson_description": "In this lesson we will build a simple LLM agent that operates in a loop with function calling. For simplicity we will use the OpenAI API and function/tool calling. One of the functions we will give the LLM is the coding sandbox function we built in the previous lessons.",
                            "duration": "24:42",
                            "content": read_markdown_content(
                                "courses/modal/section5/lesson19.md"
                            ),
                            "video_url": "https://www.youtube.com/embed/FWKH9xTeeWE",
                            "system_prompt": create_system_prompt(
                                lesson_content=read_markdown_content(
                                    "courses/modal/section5/lesson19.md"
                                ),
                                lesson_transcript=transcribe_audio.remote(
                                    "/lesson19.m4a"
                                )["transcript"],
                                modal_urls=[
                                    "https://modal.com/docs/guide/sandbox",
                                    "https://modal.com/docs/examples/simple_code_interpreter#build-a-stateful-sandboxed-code-interpreter",
                                    "https://modal.com/docs/examples/safe_code_execution#run-arbitrary-code-in-a-sandboxed-environment",
                                ],
                            ),
                        },
                        {
                            "lesson_title": "Adding a Front End Web App for our Simple Agent with Coding Abilities",
                            "lesson_description": "In this lesson I show off the LLM Agent traces within a little web app I vibe coded in Cursor.",
                            "duration": "15:08",
                            "content": read_markdown_content(
                                "courses/modal/section5/lesson20.md"
                            ),
                            "video_url": "https://www.youtube.com/embed/FjRGW1ZuzDQ",
                            "system_prompt": create_system_prompt(
                                lesson_content=read_markdown_content(
                                    "courses/modal/section5/lesson20.md"
                                ),
                                lesson_transcript=transcribe_audio.remote(
                                    "/lesson20.m4a"
                                )["transcript"],
                                modal_urls=[
                                    "https://modal.com/docs/guide/sandbox",
                                    "https://modal.com/docs/examples/simple_code_interpreter#build-a-stateful-sandboxed-code-interpreter",
                                    "https://modal.com/docs/examples/safe_code_execution#run-arbitrary-code-in-a-sandboxed-environment",
                                ],
                            ),
                        },
                    ],
                },
                {
                    "section_title": "Processing Images and Image Embeddings With Siglip2",
                    "section_description": "In this section we process thousands of image urls and create image embeddings. We will then serve the image embeddings in a small FAISS index in an endpoint for similarity search.",
                    "lessons": [
                        {
                            "lesson_title": "Image Embeddings with Siglip2 - Part 1",
                            "lesson_description": "In this lesson we go over the idea of image embeddings and how to create them with Siglip2.",
                            "duration": "14:54",
                            "content": read_markdown_content(
                                "courses/modal/section6/lesson21.md"
                            ),
                            "video_url": "https://www.youtube.com/embed/OJrszG1rnPI",
                            "system_prompt": create_system_prompt(
                                lesson_content=read_markdown_content(
                                    "courses/modal/section6/lesson21.md"
                                ),
                                lesson_transcript=transcribe_audio.remote(
                                    "/lesson21_part1.m4a"
                                )["transcript"],
                                modal_urls=[
                                    "https://modal.com/docs/guide/webhooks",
                                    "https://modal.com/docs/guide/gpu",
                                    "https://modal.com/docs/guide/images",
                                    "https://modal.com/docs/guide/volumes",
                                    "https://modal.com/docs/guide/scale",
                                ],
                            ),
                        },
                        {
                            "lesson_title": "Image Embeddings with Siglip2 - Part 2",
                            "lesson_description": "In this lesson we build out the image processing scripts on Modal to scale to thousands of images.",
                            "duration": "29:30",
                            "content": read_markdown_content(
                                "courses/modal/section6/lesson21.md"
                            ),
                            "video_url": "https://www.youtube.com/embed/OBC_wXt0THM",
                            "system_prompt": create_system_prompt(
                                lesson_content=read_markdown_content(
                                    "courses/modal/section6/lesson21.md"
                                ),
                                lesson_transcript=transcribe_audio.remote(
                                    "/lesson21_part2.m4a"
                                )["transcript"],
                                modal_urls=[
                                    "https://modal.com/docs/guide/webhooks",
                                    "https://modal.com/docs/guide/gpu",
                                    "https://modal.com/docs/guide/images",
                                    "https://modal.com/docs/guide/volumes",
                                    "https://modal.com/docs/guide/scale",
                                ],
                            ),
                        },
                        {
                            "lesson_title": "Image Embeddings with Siglip2 - Part 3",
                            "lesson_description": "In this lesson we build an endpoint for image similarity search by indexing our embeddings in FAISS.",
                            "duration": "17:08",
                            "content": read_markdown_content(
                                "courses/modal/section6/lesson21.md"
                            ),
                            "video_url": "https://www.youtube.com/embed/WjTvTu9sMjI",
                            "system_prompt": create_system_prompt(
                                lesson_content=read_markdown_content(
                                    "courses/modal/section6/lesson21.md"
                                ),
                                lesson_transcript=transcribe_audio.remote(
                                    "/lesson21_part3.m4a"
                                )["transcript"],
                                modal_urls=[
                                    "https://modal.com/docs/guide/webhooks",
                                    "https://modal.com/docs/guide/gpu",
                                    "https://modal.com/docs/guide/images",
                                    "https://modal.com/docs/guide/volumes",
                                    "https://modal.com/docs/guide/scale",
                                ],
                            ),
                        },
                    ],
                },
                {
                    "section_title": "Image Generation",
                    "section_description": "Deploy endpoints for image generation.",
                    "lessons": [
                        {
                            "lesson_title": "Image Generation with FLUX.1 Krea",
                            "lesson_description": "In this lesson we build an endpoint for image generation with FLUX.1 Krea.",
                            "duration": "14:08",
                            "content": read_markdown_content(
                                "courses/modal/section7/lesson22.md"
                            ),
                            "video_url": "https://www.youtube.com/embed/WJwyvVYSg68",
                            "system_prompt": create_system_prompt(
                                lesson_content=read_markdown_content(
                                    "courses/modal/section7/lesson22.md"
                                ),
                                lesson_transcript=transcribe_audio.remote(
                                    "/lesson22.m4a"
                                )["transcript"],
                                modal_urls=[
                                    "https://modal.com/docs/guide/webhooks",
                                    "https://modal.com/docs/guide/gpu",
                                    "https://modal.com/docs/guide/images",
                                    "https://modal.com/docs/guide/volumes",
                                    "https://modal.com/docs/guide/scale",
                                ],
                            ),
                        },
                        {
                            "lesson_title": "Image Generation with Qwen/Qwen-Image",
                            "lesson_description": "In this lesson we build an endpoint for image generation with Qwen/Qwen-Image.",
                            "duration": "6:07",
                            "content": read_markdown_content(
                                "courses/modal/section7/lesson22_part2.md"
                            ),
                            "video_url": "https://www.youtube.com/embed/4ebUc26g1jM",
                            "system_prompt": create_system_prompt(
                                lesson_content=read_markdown_content(
                                    "courses/modal/section7/lesson22_part2.md"
                                ),
                                lesson_transcript=transcribe_audio.remote(
                                    "/lesson22_part2.m4a"
                                )["transcript"],
                                modal_urls=[
                                    "https://modal.com/docs/guide/webhooks",
                                    "https://modal.com/docs/guide/gpu",
                                    "https://modal.com/docs/guide/images",
                                    "https://modal.com/docs/guide/volumes",
                                    "https://modal.com/docs/guide/scale",
                                ],
                            ),
                        },
                        {
                            "lesson_title": "Image Editing with Qwen/Qwen-Image-Edit",
                            "lesson_description": "In this lesson we build an endpoint for image editing with Qwen/Qwen-Image-Edit.",
                            "duration": "15:09",
                            "content": read_markdown_content(
                                "courses/modal/section7/lesson22_part3.md"
                            ),
                            "video_url": "https://www.youtube.com/embed/02NuraPr21g",
                            "system_prompt": create_system_prompt(
                                lesson_content=read_markdown_content(
                                    "courses/modal/section7/lesson22_part3.md"
                                ),
                                lesson_transcript=transcribe_audio.remote(
                                    "/lesson22_part3.m4a"
                                )["transcript"],
                                modal_urls=[
                                    "https://modal.com/docs/guide/webhooks",
                                    "https://modal.com/docs/guide/gpu",
                                    "https://modal.com/docs/guide/images",
                                    "https://modal.com/docs/guide/volumes",
                                    "https://modal.com/docs/guide/scale",
                                ],
                            ),
                        },
                    ],
                },
                {
                    "section_title": "Setting up Flash Attention and a Container for Fine-Tuning LLMS with Axolotl",
                    "section_description": "In this section we will briefly show how to setup an environment for a container with Flash Attention, and fine-tuning LLMs with Axolotl.",
                    "lessons": [
                        {
                            "lesson_title": "Setting up Flash Attention and a Container for Fine-Tuning LLMS with Axolotl",
                            "lesson_description": "In this lesson we will briefly show how to setup an environment for a container with Flash Attention, and fine-tuning LLMs with Axolotl.",
                            "duration": "11:54",
                            "content": read_markdown_content(
                                "courses/modal/section8/lesson23.md"
                            ),
                            "video_url": "https://www.youtube.com/embed/0CpP2h8xgDo",
                            "system_prompt": create_system_prompt(
                                lesson_content=read_markdown_content(
                                    "courses/modal/section8/lesson23.md"
                                ),
                                lesson_transcript=transcribe_audio.remote(
                                    "/lesson23.m4a"
                                )["transcript"],
                                modal_urls=[
                                    "https://modal.com/docs/guide/gpu",
                                    "https://modal.com/docs/guide/cuda",
                                ],
                            ),
                        },
                    ],
                },
                {
                    "section_title": "Deploying a Remote MCP Server on Modal",
                    "section_description": "In this section we show how to deploy a remote MCP server on Modal.",
                    "lessons": [
                        {
                            "lesson_title": "Deploying a Remote MCP Server on Modal",
                            "lesson_description": "In this lesson we show how to deploy a remote MCP server on Modal.",
                            "duration": "30:52",
                            "content": read_markdown_content(
                                "courses/modal/section9/lesson24.md"
                            ),
                            "video_url": "https://www.youtube.com/embed/MCFES06LMUs",
                            "system_prompt": create_system_prompt(
                                lesson_content=read_markdown_content(
                                    "courses/modal/section9/lesson24.md"
                                ),
                                lesson_transcript=transcribe_audio.remote(
                                    "/lesson24.m4a"
                                )["transcript"],
                                modal_urls=[
                                    "https://modal.com/docs/guide/webhooks",
                                    "https://modal.com/docs/guide/sandbox#sandboxes",
                                ],
                            ),
                        },
                    ],
                },
            ],
        },
    ]
    for course in COURSES:
        c = DataBase.insert_course(
            title=course["title"],
            description=course["description"],
            thumbnail=course["thumbnail"],
            created_at=now_str,
            updated_at=now_str,
        )
        lesson_count = 0
        for i, sec in enumerate(course["lesson_content"], start=1):
            s = DataBase.insert_section(
                course_id=c.id,
                title=sec["section_title"],
                description=sec["section_description"],
                sort_order=i,
                created_at=now_str,
                updated_at=now_str,
            )
            for l in sec["lessons"]:
                l = DataBase.insert_lesson(
                    course_id=c.id,
                    section_id=s.id,
                    title=l["lesson_title"],
                    description=l["lesson_description"],
                    video_url=l["video_url"],
                    # thumbnail='',
                    duration=l["duration"],
                    system_prompt=l["system_prompt"],
                    created_at=now_str,
                    updated_at=now_str,
                    content=l["content"],
                    sort_order=lesson_count,
                )
                lesson_count += 1
