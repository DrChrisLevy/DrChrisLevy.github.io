import io
import math
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
    num_inference_steps: int = 8
    height: int = None
    width: int = None
    num_images_per_prompt: int = 1  # Always 1 for single output


# Image with required dependencies
image = (
    modal.Image.debian_slim()
    .apt_install(["git"])
    .pip_install(
        [
            "torch",
            "torchvision",
            "git+https://github.com/huggingface/diffusers.git",
            "transformers",
            "accelerate",
            "safetensors",
            "sentencepiece",
            "dashscope",
            "kernels",
            "peft",
            "torchao==0.11.0",
            "pillow",
            "python-dotenv",
            "requests",
            "boto3",
        ]
    )
    .add_local_dir("posts/mcptalk/qwenimage", "/root/qwenimage")
)

app = modal.App("qwen-image-editor-fast-v2", image=image)

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
    max_containers=5,
    # enable_memory_snapshot=True, # in alpha # https://modal.com/blog/gpu-mem-snapshots
    # experimental_options={"enable_gpu_snapshot": True}
)
@modal.concurrent(max_inputs=1)
class QwenImageEditor:
    @modal.enter()  # snap=True to try gpu memory snapshot
    def setup(self):
        """Load Qwen-Image-Edit model with Lightning acceleration once per container"""
        import torch
        from diffusers import FlowMatchEulerDiscreteScheduler
        from qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
        from qwenimage.qwen_fa3_processor import QwenDoubleStreamAttnProcessorFA3
        from qwenimage.transformer_qwenimage import QwenImageTransformer2DModel

        print("Loading Qwen/Qwen-Image-Edit model with Lightning acceleration...")
        dtype = torch.bfloat16
        device = "cuda"
        scheduler_config = {
            "base_image_seq_len": 256,
            "base_shift": math.log(3),
            "invert_sigmas": False,
            "max_image_seq_len": 8192,
            "max_shift": math.log(3),
            "num_train_timesteps": 1000,
            "shift": 1.0,
            "shift_terminal": None,
            "stochastic_sampling": False,
            "time_shift_type": "exponential",
            "use_beta_sigmas": False,
            "use_dynamic_shifting": True,
            "use_exponential_sigmas": False,
            "use_karras_sigmas": False,
        }
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
        self.pipe = QwenImageEditPlusPipeline.from_pretrained("Qwen/Qwen-Image-Edit-2509", scheduler=scheduler, torch_dtype=dtype).to(device)
        self.pipe.load_lora_weights(
            "lightx2v/Qwen-Image-Lightning",
            weight_name="Qwen-Image-Lightning-8steps-V2.0-bf16.safetensors",
        )
        self.pipe.fuse_lora()

        # Apply optimizations
        self.pipe.transformer.__class__ = QwenImageTransformer2DModel
        self.pipe.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorFA3())

        self.pipe.set_progress_bar_config(disable=None)
        print("Model loaded successfully with Lightning acceleration!")

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

    def _download_images_from_urls(self, image_urls: list):
        """Download multiple images from URLs and convert to PIL Images"""
        pil_images = []
        for image_url in image_urls:
            try:
                pil_image = self._download_image_from_url(image_url)
                pil_images.append(pil_image)
            except Exception as e:
                print(f"Error downloading image from {image_url}: {e}")
                continue
        return pil_images

    def _upload_image_to_s3(self, image, filename: str) -> str:
        """Upload PIL Image to S3 and return the S3 URL"""
        from botocore.exceptions import ClientError

        # Convert PIL Image to bytes
        img_buffer = io.BytesIO()
        image.save(img_buffer, format="PNG")
        img_buffer.seek(0)

        # S3 key (path)
        s3_key = f"{S3_PREFIX}/{filename}"

        try:
            # Upload to S3
            self.s3_client.upload_fileobj(img_buffer, S3_BUCKET, s3_key, ExtraArgs={"ContentType": "image/png"})

            # Return S3 URL
            s3_url = f"https://{S3_BUCKET}.s3.amazonaws.com/{s3_key}"
            print(f"Image uploaded successfully to S3: {s3_url}")
            return s3_url

        except ClientError as e:
            print(f"Error uploading to S3: {e}")
            raise

    def edit_image(
        self,
        image_urls: list,
        prompt: str,
        negative_prompt: str = " ",
        true_cfg_scale: float = 4.0,
        seed: int = 0,
        randomize_seed: bool = False,
        num_inference_steps: int = 8,
        height: int = None,
        width: int = None,
    ):
        import random
        import uuid
        from datetime import datetime

        import numpy as np
        import torch

        # Download multiple images
        input_images = self._download_images_from_urls(image_urls) if image_urls else []

        MAX_SEED = np.iinfo(np.int32).max
        if randomize_seed:
            seed = random.randint(0, MAX_SEED)

        print(f"Editing {len(input_images)} images with prompt: {prompt}")

        # Edit image using Qwen-Image-Edit-Plus
        inputs = {
            "image": input_images if len(input_images) > 0 else None,
            "prompt": prompt,
            "generator": torch.manual_seed(seed),
            "true_cfg_scale": true_cfg_scale,
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
            "height": height,
            "width": width,
            "num_images_per_prompt": 1,  # Always generate only 1 image
        }

        with torch.inference_mode():
            output = self.pipe(**inputs)
            edited_images = output.images

        # Upload the single generated image to S3
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"qwen_edited_{timestamp}_{unique_id}.png"

        # Take the first (and only) generated image
        edited_image = edited_images[0]
        image_url_output = self._upload_image_to_s3(edited_image, filename)

        return {
            "original_image_urls": image_urls,
            "image_url": image_url_output,  # Single image URL instead of list
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "true_cfg_scale": true_cfg_scale,
            "seed": seed,
            "num_inference_steps": num_inference_steps,
            "height": height,
            "width": width,
            "num_images_per_prompt": 1,  # Always 1
        }

    @modal.fastapi_endpoint(
        method="POST",
        docs=True,
    )
    def edit_image_endpoint(self, request: ImageEditRequest):
        """Public FastAPI endpoint for image editing with single image output"""
        return self.edit_image(
            image_urls=request.image_urls,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            true_cfg_scale=request.true_cfg_scale,
            seed=request.seed,
            randomize_seed=request.randomize_seed,
            num_inference_steps=request.num_inference_steps,
            height=request.height,
            width=request.width,
        )
