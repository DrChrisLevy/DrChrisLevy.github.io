import io
import os
import time
from typing import List
import modal
from pydantic import BaseModel

# S3 configuration
S3_BUCKET = "dev-dashhudson-static"
S3_PREFIX = "research/chris/images"


class ImageEditRequest(BaseModel):
    image_urls: List[str]
    prompt: str
    negative_prompt: str = " "
    true_cfg_scale: float = 4.0
    seed: int = 0
    randomize_seed: bool = False
    num_inference_steps: int = 40
    guidance_scale: float = 1.0
    num_images_per_prompt: int = 1

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
            "peft",
            "boto3",
        ]
    )
)

app = modal.App("qwen-image-editor", image=image)

hf_hub_cache = modal.Volume.from_name("hf_hub_cache", create_if_missing=True)


@app.cls(
    image=image,
    gpu="H100",
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("s3-bucket-dev"),
    ],
    timeout=60 * 5,
    volumes={
        "/root/.cache/huggingface/hub/": hf_hub_cache,
    },
    scaledown_window=30 * 60,
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
        from diffusers import QwenImageEditPlusPipeline

        print("Loading Qwen/Qwen-Image-Edit-2509 model...")
        self.pipe = QwenImageEditPlusPipeline.from_pretrained("Qwen/Qwen-Image-Edit-2509", torch_dtype=torch.bfloat16)
        self.pipe.to("cuda")
        self.pipe.set_progress_bar_config(disable=None)
        print("Model loaded successfully!")
        
        # Initialize S3 client
        import boto3
        self.s3_client = boto3.client("s3")

    def _download_image_from_url(self, image_url: str):
        """Download image from URL and convert to PIL Image"""
        import requests
        from PIL import Image

        response = requests.get(image_url)
        response.raise_for_status()

        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        return image
    
    def _upload_image_to_s3(self, image, filename: str) -> str:
        """Upload PIL Image to S3 and return the S3 URL"""
        import boto3
        from botocore.exceptions import ClientError
        
        # Convert PIL Image to bytes
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        # S3 key (path)
        s3_key = f"{S3_PREFIX}/{filename}"
        
        try:
            # Upload to S3
            self.s3_client.upload_fileobj(
                img_buffer,
                S3_BUCKET,
                s3_key,
                ExtraArgs={'ContentType': 'image/png'}
            )
            
            # Return S3 URL
            s3_url = f"https://{S3_BUCKET}.s3.amazonaws.com/{s3_key}"
            print(f"Image uploaded successfully to S3: {s3_url}")
            return s3_url
            
        except ClientError as e:
            print(f"Error uploading to S3: {e}")
            raise

    def edit_image(
        self,
        image_urls: List[str],
        prompt: str,
        negative_prompt: str = " ",
        true_cfg_scale: float = 4.0,
        seed: int = 0,
        randomize_seed: bool = False,
        num_inference_steps: int = 40,
        guidance_scale: float = 1.0,
        num_images_per_prompt: int = 1,
    ):
        import random
        import uuid
        from datetime import datetime

        import numpy as np
        import torch

        # Download all input images
        input_images = [self._download_image_from_url(url) for url in image_urls]

        MAX_SEED = np.iinfo(np.int32).max
        if randomize_seed:
            seed = random.randint(0, MAX_SEED)

        print(f"Editing {len(image_urls)} image(s) with prompt: {prompt}")

        # Edit image using Qwen-Image-Edit-Plus with new parameters
        inputs = {
            "image": input_images,
            "prompt": prompt,
            "generator": torch.manual_seed(seed),
            "true_cfg_scale": true_cfg_scale,
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "num_images_per_prompt": num_images_per_prompt,
        }

        with torch.inference_mode():
            output = self.pipe(**inputs)
            edited_image = output.images[0]

        # Create unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"qwen_edited_{timestamp}_{unique_id}.png"

        # Upload to S3
        image_url_output = self._upload_image_to_s3(edited_image, filename)

        return {
            "original_image_urls": image_urls,
            "image_url": image_url_output,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "true_cfg_scale": true_cfg_scale,
            "seed": seed,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "num_images_per_prompt": num_images_per_prompt,
        }

    @modal.fastapi_endpoint(
        method="POST",
        docs=True,
    )
    def edit_image_endpoint(self, request: ImageEditRequest):
        """Public FastAPI endpoint for image editing"""
        return self.edit_image(
            image_urls=request.image_urls,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            true_cfg_scale=request.true_cfg_scale,
            seed=request.seed,
            randomize_seed=request.randomize_seed,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            num_images_per_prompt=request.num_images_per_prompt,
        )

