import modal
from modal import Image, build, enter
import os
from dotenv import load_dotenv
from itertools import islice

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
    def __init__(self, model_id="black-forest-labs/FLUX.1-schnell"):  # black-forest-labs/FLUX.1-dev
        self.model_id = model_id

    @build()  # add another step to the image build
    def download_model_to_folder(self):
        from huggingface_hub import snapshot_download

        snapshot_download(self.model_id)

    @enter()
    def setup(self):
        import torch
        from diffusers import FluxPipeline

        self.pipe = FluxPipeline.from_pretrained(self.model_id, torch_dtype=torch.bfloat16).to("cuda")
        # self.pipe.enable_model_cpu_offload() # less memory usage, but more latency

    @modal.method()
    def f(self, prompts, fnames, num_inference_steps=4, seed=None):
        import torch
        import random

        if not seed:
            seed = random.randint(1, 2**63 - 1)
        S3_BUCKET = os.environ["S3_BUCKET"]
        S3_PREFIX = os.environ["S3_PREFIX"]
        if self.model_id == "black-forest-labs/FLUX.1-dev":
            num_inference_steps = max(num_inference_steps, 25)

        for prompt, fname in zip(prompts, fnames):
            image = self.pipe(
                prompt,
                output_type="pil",
                num_inference_steps=num_inference_steps,
                generator=torch.Generator("cpu").manual_seed(seed) if seed else None,
            ).images[0]
            image.save(f"{fname}.png")
            os.system(f"aws s3 cp {fname}.png s3://{S3_BUCKET}{S3_PREFIX}")


@app.local_entrypoint()
def main():
    # Function to batch your list of tuples
    def batch(iterable, batch_size=5):
        it = iter(iterable)
        while True:
            batch_tuple = list(islice(it, batch_size))
            if not batch_tuple:
                break
            yield batch_tuple

    image_prompts = [("""A tasty cake""", "tasty_cake"), ("""An astronaut riding a horse""", "astro_horse")]
    all_prompts = []
    batch_size = 5
    for b in batch(image_prompts, batch_size):
        first_args = [x[0] for x in b]
        second_args = [x[1] for x in b]
        all_prompts.append((first_args, second_args))

    list(Model(model_id="black-forest-labs/FLUX.1-schnell").f.starmap(all_prompts))
