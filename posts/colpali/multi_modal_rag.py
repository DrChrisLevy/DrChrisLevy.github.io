import base64
import os
import time
from io import BytesIO
from typing import List

import modal
from utils import generate_unique_folder_name, log_to_queue

app = modal.App("multi-modal-rag")

vol = modal.Volume.from_name("pdf-retriever-volume", create_if_missing=True)


image = modal.Image.debian_slim(python_version="3.10").pip_install("Pillow", "python-dotenv", "openai")


def pil_image_to_data_url(pil_image):
    # Convert PIL Image to bytes
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")

    # Encode to base64
    img_str = base64.b64encode(buffered.getvalue()).decode()

    # Format as data URL
    return f"data:image/png;base64,{img_str}"


def load_images(pdf_url: str, img_idxs: List[int], max_retries=5, retry_delay=1):
    from PIL import Image

    try:
        vol.reload()
    except RuntimeError as e:
        log_to_queue(f"Error reloading volume: {str(e)}. Attempting to continue...")
    cache_dir = generate_unique_folder_name(pdf_url)
    data_type = "pdf_images"
    cache_path = os.path.join(f"/data/{data_type}", f"{cache_dir}")

    images = []
    for idx in img_idxs:
        vol.reload()
        img_path = os.path.join(cache_path, f"{idx}.png")
        for attempt in range(max_retries):
            try:
                with Image.open(img_path) as image:
                    image_copy = image.copy()
                images.append(image_copy)
                break
            except FileNotFoundError:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    log_to_queue(f"Image file not found after {max_retries} attempts: {img_path}. Refresh Page and Try Again.")
                    raise FileNotFoundError(f"Image file not found after {max_retries} attempts: {img_path}")

    return images


@app.function(volumes={"/data": vol}, image=image, container_idle_timeout=60 * 3, secrets=[modal.Secret.from_dotenv()])
def answer_questions_with_image_context(
    pdf_url, queries, top_k=1, use_cache=True, max_new_tokens=400, additional_instructions="", model="gpt-4o-mini"
):
    log_to_queue("Entered Multi Modal RAG Container")
    vol.reload()
    pdf_retriever = modal.Function.lookup("pdf-retriever", "PDFRetriever.top_pages")
    # vision_language_model = modal.Function.lookup("vision-language-model", "VisionLanguageModel.forward")
    idxs_top_k = pdf_retriever.remote(pdf_url, queries, use_cache=use_cache, top_k=top_k)

    messages_list = []
    all_images_data = []
    for i, idxs in enumerate(idxs_top_k):
        query = queries[i]

        images = load_images(pdf_url, idxs)
        images_data = [pil_image_to_data_url(img) for img in images]
        all_images_data.append(images_data)

        if model == "Qwen/Qwen2-VL-7B-Instruct":
            content = [{"type": "image", "image": img_data} for img_data in images_data]
        elif model == "gpt-4o-mini":
            content = [{"type": "image_url", "image_url": {"url": img_data}} for img_data in images_data]
        else:
            raise ValueError(f"Model {model} not supported")
        content.append({"type": "text", "text": f"{additional_instructions}\n\n{query}"})
        messages_list.append([{"role": "user", "content": content}])

    log_to_queue("PDF Retrieval Complete. Sending Context and Question to Vision Language Model")
    if model == "Qwen/Qwen2-VL-7B-Instruct":
        raise (Exception("Qwen/Qwen2-VL-7B-Instruct not supported yet"))
        # res = vision_language_model.remote(messages_list, max_new_tokens=max_new_tokens, show_stream=True), all_images_data
    elif model == "gpt-4o-mini":
        resp_gen = openai_vision_language_model(messages_list, max_new_tokens)
        for chunk in resp_gen:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                yield content
        yield all_images_data
    else:
        raise ValueError(f"Model {model} not supported")


def openai_vision_language_model(messages_list, max_new_tokens):
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()

    client = OpenAI()
    messages = messages_list[0]  # TODO: Assuming messages_list is length 1
    log_to_queue("Calling OpenAI . . .")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=max_new_tokens,
        stream=True,
    )
    return response
