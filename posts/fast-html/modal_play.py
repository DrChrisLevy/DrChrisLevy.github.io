import modal
from modal import Image, build, enter
import os
from dotenv import load_dotenv

load_dotenv()
app = modal.App("black-forest-labs")

image = Image.debian_slim(python_version="3.11").run_commands(
    "apt-get update && apt-get install -y git",
    "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124",
    "pip install transformers",
    "pip install accelerate",
    "pip install sentencepiece",
    "pip install git+https://github.com/huggingface/diffusers.git",
    "pip install awscli",
    "pip install python-dotenv",
    f'huggingface-cli login --token {os.environ["HUGGING_FACE_ACCESS_TOKEN"]}',
)


@app.cls(image=image, secrets=[modal.Secret.from_dotenv()], gpu="A100", cpu=4, timeout=600, container_idle_timeout=300)
class Model:
    @build()
    @enter()
    def setup(self):
        import torch
        from diffusers import FluxPipeline
        from transformers.utils import move_cache

        # black-forest-labs/FLUX.1-schnell"
        self.pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to("cuda")
        move_cache()

    @modal.web_endpoint(method="POST", docs=True)
    def f(self, data: dict):
        import torch
        import random

        prompts = data["prompts"]
        fnames = data["fnames"]
        num_inference_steps = data.get("num_inference_steps", 4)  # use more for DEV flux
        seed = data.get("seed", random.randint(1, 2**63 - 1))
        guidance_scale = data.get("guidance_scale", 3.5)
        S3_BUCKET = os.environ["S3_BUCKET"]
        S3_PREFIX = os.environ["S3_PREFIX"]

        for prompt, fname in zip(prompts, fnames):
            image = self.pipe(
                prompt,
                output_type="pil",
                num_inference_steps=num_inference_steps,
                generator=torch.Generator("cpu").manual_seed(seed),
                guidance_scale=guidance_scale,
            ).images[0]
            image.save(f"{fname}_{seed}.png")
            os.system(f"aws s3 cp {fname}_guidance_scale_{guidance_scale}_{seed}.png s3://{S3_BUCKET}{S3_PREFIX}")
