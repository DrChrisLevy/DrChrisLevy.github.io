import os

import modal

# Modal Volume URL configuration
MODAL_WORKSPACE = "drchrislevy"  # replace with your modal workspace
MODAL_ENVIRONMENT = "main"  # replace with your modal environment
VOLUME_NAME = "qwen_generated_images"  # replace with your modal volume name

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
        ]
    )
)

app = modal.App("qwen-image-generator", image=image)

hf_hub_cache = modal.Volume.from_name("hf_hub_cache", create_if_missing=True)
images_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.cls(
    image=image,
    gpu="H100",
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
    ],
    timeout=60 * 10,
    volumes={
        "/root/.cache/huggingface/hub/": hf_hub_cache,
        "/root/generated_images": images_volume,
    },
    scaledown_window=60 * 60,
    max_containers=2,
)
@modal.concurrent(max_inputs=1)
class QwenImageGenerator:
    @modal.enter()
    def setup(self):
        """Load Qwen-Image model once per container"""
        import torch
        from diffusers import DiffusionPipeline

        print("Loading Qwen/Qwen-Image model...")

        # Set device and dtype (CUDA is available)
        self.torch_dtype = torch.bfloat16
        self.device = "cuda"

        # Load the pipeline
        self.pipe = DiffusionPipeline.from_pretrained(
            "Qwen/Qwen-Image",
            torch_dtype=self.torch_dtype,
            cache_dir="/root/.cache/huggingface/hub",
        )
        self.pipe = self.pipe.to(self.device)

        print("Model loaded successfully!")

        # Set up volume path for storing images
        self.images_path = "/root/generated_images"

        # Define aspect ratios
        self.aspect_ratios = {
            "1:1": (1328, 1328),
            "16:9": (1664, 928),
            "9:16": (928, 1664),
            "4:3": (1472, 1140),
            "3:4": (1140, 1472),
        }

    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        aspect_ratio: str = "16:9",
        true_cfg_scale: float = 3.5,
        seed: int = 42,
        randomize_seed=False,
        num_inference_steps: int = 50,
    ):
        """Generate image from text prompt and save to Modal Volume"""
        import random
        import uuid
        from datetime import datetime

        import numpy as np
        import torch

        MAX_SEED = np.iinfo(np.int32).max
        if randomize_seed:
            seed = random.randint(0, MAX_SEED)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        print(f"Generating image for prompt: {prompt}")

        # Get dimensions from aspect ratio
        if aspect_ratio in self.aspect_ratios:
            width, height = self.aspect_ratios[aspect_ratio]
        else:
            width, height = self.aspect_ratios["16:9"]  # default

        # Generate image
        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            true_cfg_scale=true_cfg_scale,
            generator=generator,
        ).images[0]

        # Create unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"qwen_generated_{timestamp}_{unique_id}.png"
        file_path = os.path.join(self.images_path, filename)

        # Save image to Modal Volume
        try:
            image.save(file_path, format="PNG")
            print(f"Image saved successfully to volume: {file_path}")

            # Generate Modal Volume URL
            # Format: https://modal.com/api/volumes/{workspace}/{env}/{volume_name}/files/content?path={filename}
            image_url = f"https://modal.com/api/volumes/{MODAL_WORKSPACE}/{MODAL_ENVIRONMENT}/{VOLUME_NAME}/files/content?path={filename}"

            return {
                "image_url": image_url,
                "filename": filename,
                "file_path": file_path,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "aspect_ratio": aspect_ratio,
                "dimensions": {"width": width, "height": height},
                "volume_path": f"/root/generated_images/{filename}",
            }

        except Exception as e:
            print(f"Error saving to volume: {e}")
            raise

    @modal.fastapi_endpoint(
        method="POST",
        docs=True,
    )
    def generate_image_endpoint(
        self,
        prompt: str,
        negative_prompt: str = "",
        aspect_ratio: str = "16:9",
        true_cfg_scale: float = 4.0,
        seed: int = 42,
        randomize_seed: bool = False,
        num_inference_steps: int = 50,
    ):
        """Public FastAPI endpoint for image generation"""
        return self.generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            aspect_ratio=aspect_ratio,
            true_cfg_scale=true_cfg_scale,
            seed=seed,
            randomize_seed=randomize_seed,
            num_inference_steps=num_inference_steps,
        )
