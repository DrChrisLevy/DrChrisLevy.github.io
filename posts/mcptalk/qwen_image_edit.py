import io
import os
import time

import modal

# Modal Volume URL configuration
MODAL_WORKSPACE = "drchrislevy"  # replace with your modal workspace
MODAL_ENVIRONMENT = "main"  # replace with your modal environment
VOLUME_NAME = "qwen_edited_images"

# Image with required dependencies
image = (
    modal.Image.debian_slim()
    .apt_install(["git"])
    .pip_install(
        [
            "torch",
            "torchvision",
            "git+https://github.com/huggingface/diffusers",
            "transformers",
            "accelerate",
            "pillow",
            "sentencepiece",
            "python-dotenv",
            "requests",
        ]
    )
)

app = modal.App("qwen-image-editor", image=image)

hf_hub_cache = modal.Volume.from_name("hf_hub_cache", create_if_missing=True)
images_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.cls(
    image=image,
    gpu="H100",
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
    ],
    timeout=60 * 5,
    volumes={
        "/root/.cache/huggingface/hub/": hf_hub_cache,
        "/root/edited_images": images_volume,
    },
    scaledown_window=10 * 60,
    max_containers=2,
    # enable_memory_snapshot=True, # in alpha # https://modal.com/blog/gpu-mem-snapshots
    # experimental_options={"enable_gpu_snapshot": True}
)
@modal.concurrent(max_inputs=1)
class QwenImageEditor:
    @modal.enter()  # snap=True to try gpu memory snapshot
    def setup(self):
        """Load Qwen-Image-Edit model once per container"""
        import torch
        from diffusers import QwenImageEditPipeline

        print("Loading Qwen/Qwen-Image-Edit model...")
        self.pipe = QwenImageEditPipeline.from_pretrained("Qwen/Qwen-Image-Edit")
        self.pipe.to(torch.bfloat16)
        self.pipe.to("cuda")
        self.pipe.set_progress_bar_config(disable=None)
        print("Model loaded successfully!")
        self.images_path = "/root/edited_images"

    def _download_image_from_url(self, image_url: str):
        """Download image from URL and convert to PIL Image"""
        import requests
        from PIL import Image

        response = requests.get(image_url)
        response.raise_for_status()

        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        return image

    def edit_image(
        self,
        image_url: str,
        prompt: str,
        negative_prompt: str = " ",
        true_cfg_scale: float = 4.0,
        seed: int = 0,
        randomize_seed: bool = False,
        num_inference_steps: int = 50,
    ):
        import random
        import uuid
        from datetime import datetime

        import numpy as np
        import torch

        input_image = self._download_image_from_url(image_url)

        MAX_SEED = np.iinfo(np.int32).max
        if randomize_seed:
            seed = random.randint(0, MAX_SEED)

        print(f"Editing image {image_url} with prompt: {prompt}")

        # Edit image using Qwen-Image-Edit exactly like the original
        inputs = {
            "image": input_image,
            "prompt": prompt,
            "generator": torch.manual_seed(seed),
            "true_cfg_scale": true_cfg_scale,
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
        }

        with torch.inference_mode():
            output = self.pipe(**inputs)
            edited_image = output.images[0]

        # Create unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"qwen_edited_{timestamp}_{unique_id}.png"
        file_path = os.path.join(self.images_path, filename)

        edited_image.save(file_path, format="PNG")
        print(f"Edited image saved successfully to volume: {file_path}")

        # Generate Modal Volume URL
        # Format: https://modal.com/api/volumes/{workspace}/{env}/{volume_name}/files/content?path={filename}
        image_url_output = f"https://modal.com/api/volumes/{MODAL_WORKSPACE}/{MODAL_ENVIRONMENT}/{VOLUME_NAME}/files/content?path={filename}"

        return {
            "original_image_url": image_url,
            "image_url": image_url_output,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "true_cfg_scale": true_cfg_scale,
            "seed": seed,
            "num_inference_steps": num_inference_steps,
        }

    @modal.fastapi_endpoint(
        method="POST",
        docs=True,
    )
    def edit_image_endpoint(
        self,
        image_url: str,
        prompt: str,
        negative_prompt: str = " ",
        true_cfg_scale: float = 4.0,
        seed: int = 0,
        randomize_seed: bool = False,
        num_inference_steps: int = 50,
    ):
        """Public FastAPI endpoint for image editing"""
        return self.edit_image(
            image_url=image_url,
            prompt=prompt,
            negative_prompt=negative_prompt,
            true_cfg_scale=true_cfg_scale,
            seed=seed,
            randomize_seed=randomize_seed,
            num_inference_steps=num_inference_steps,
        )

