import base64
from io import BytesIO

import requests
from PIL import Image

# API endpoint base URL - replace with your actual Modal endpoint


def text_and_image_to_text(image_url: str, prompt: str) -> str:
    """
    Send an image URL and text prompt to get a text response
    """
    url = "https://drchrislevy--deepseek-janus-pro-model-text-and-image-to-text.modal.run"
    response = requests.post(url, json={"image_url": image_url, "content": prompt})
    response.raise_for_status()
    return response.json()


def text_to_image(prompt: str) -> list:
    """
    Send a text prompt to generate images
    Returns a list of PIL Images
    """
    url = "https://drchrislevy--deepseek-janus-pro-model-text-to-image.modal.run"
    response = requests.post(url, json={"content": prompt})
    response.raise_for_status()
    image_base64_list = response.json()
    # Convert base64 encoded images to PIL Images
    images = []
    for base64_img in image_base64_list:
        # Remove the data URL prefix if present
        if base64_img.startswith("data:image/jpeg;base64,"):
            base64_img = base64_img.split(",")[1]

        # Decode base64 string to bytes
        img_bytes = base64.b64decode(base64_img)

        # Convert bytes to PIL Image
        img = Image.open(BytesIO(img_bytes))
        images.append(img)

    return images
