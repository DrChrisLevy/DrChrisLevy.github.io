import base64
import os
from io import BytesIO
from typing import List

import modal
from utils import generate_unique_folder_name

app = modal.App("multi-modal-rag")

vol = modal.Volume.from_name("pdf-retriever-volume", create_if_missing=True)


image = modal.Image.debian_slim(python_version="3.10").pip_install("Pillow", "python-dotenv")


def pil_image_to_data_url(pil_image):
    # Convert PIL Image to bytes
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")

    # Encode to base64
    img_str = base64.b64encode(buffered.getvalue()).decode()

    # Format as data URL
    return f"data:image/png;base64,{img_str}"


def load_images(pdf_url: str, img_idxs: List[int]):
    from PIL import Image

    cache_dir = generate_unique_folder_name(pdf_url)
    data_type = "pdf_images"
    cache_path = os.path.join(f"/data/{data_type}", f"{cache_dir}")

    images = []
    for idx in img_idxs:
        img_path = os.path.join(cache_path, f"{idx}.png")
        images.append(Image.open(img_path))
    return images


@app.function(volumes={"/data": vol}, image=image)
def answer_questions_with_image_context(pdf_url, queries, top_k=1, use_cache=True, max_new_tokens=1024, show_stream=False):
    vol.reload()
    pdf_retriever = modal.Function.lookup("pdf-retriever", "PDFRetriever.top_pages")
    vision_language_model = modal.Function.lookup("vision-language-model", "VisionLanguageModel.forward")
    print(f'{queries=}')
    print(f'{pdf_url=}')
    print(f'{top_k=}')
    idxs_top_k = pdf_retriever.remote(pdf_url, queries, use_cache=use_cache, top_k=top_k)
    print(f'{idxs_top_k=}')

    messages_list = []
    for i, idxs in enumerate(idxs_top_k):
        print(f'Preparing Iteration {i} for {idxs=}')
        query = queries[i]
        images = load_images(pdf_url, idxs)
        content = [{"type": "image", "image": pil_image_to_data_url(img)} for img in images]
        content.append({"type": "text", "text": f"{query}"})
        messages_list.append([{"role": "user", "content": content}])
        print(f'Content of length {len(content)} added to messages list')
    print(f'Sending {len(messages_list)} messages to the vision language model')
    return vision_language_model.remote(messages_list, max_new_tokens=max_new_tokens, show_stream=show_stream)
