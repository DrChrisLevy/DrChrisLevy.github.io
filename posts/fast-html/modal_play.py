import modal
from modal import Image, build, enter
import os

app = modal.App("example-get-started")

image = Image.debian_slim(python_version="3.11").run_commands(
    "apt-get update && apt-get install -y git",
    "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124",
    "pip install transformers",
    "pip install accelerate",
    "pip install sentencepiece",
    "pip install git+https://github.com/huggingface/diffusers.git",
    "pip install awscli",
)


@app.cls(image=image, secrets=[modal.Secret.from_dotenv()], gpu="A100", cpu=4, timeout=600)
class Model:
    model_id = "black-forest-labs/FLUX.1-schnell"  # you can also use `black-forest-labs/FLUX.1-dev`

    @build()  # add another step to the image build
    def download_model_to_folder(self):
        from huggingface_hub import snapshot_download

        # os.makedirs(MODEL_DIR, exist_ok=True)
        snapshot_download(Model.model_id)

    @enter()
    def setup(self):
        import torch
        from diffusers import FluxPipeline

        self.pipe = FluxPipeline.from_pretrained(Model.model_id, torch_dtype=torch.bfloat16).to("cuda")
        # self.pipe.enable_model_cpu_offload() # less memory usage, but slightly slower

    @modal.method()
    def f(self, prompts, fnames, seed=None):
        import torch

        S3_BUCKET = os.environ["S3_BUCKET"]
        S3_PREFIX = os.environ["S3_PREFIX"]
        for prompt, fname in zip(prompts, fnames):
            image = self.pipe(
                prompt,
                output_type="pil",
                num_inference_steps=4,  # use a larger number if you are using [dev]
                generator=torch.Generator("cpu").manual_seed(seed) if seed else None,
            ).images[0]
            image.save(f"{fname}.png")
            os.system(f"aws s3 cp {fname}.png s3://{S3_BUCKET}{S3_PREFIX}")


@app.local_entrypoint()
def main():
    image_prompts = [
        (
            "A steampunk-inspired flying ship with brass gears and billowing sails, soaring through a sunset sky filled with colorful hot air balloons",
            1,
        ),
        ("An underwater cityscape with bioluminescent buildings, inhabited by merpeople riding seahorses through coral-lined streets", 2),
        (
            "A futuristic Tokyo street scene at night, with holographic advertisements, flying cars, and robots mingling with humans under neon lights",
            3,
        ),
        ("A whimsical treehouse library in an ancient redwood forest, with spiral staircases, floating books, and owl librarians", 4),
        ("A cyberpunk-style human-robot hybrid DJ performing at a high-tech nightclub with a crowd of diverse aliens and humans", 5),
        (
            "An Art Nouveau-inspired portrait of Mother Nature, with flowing hair made of leaves and flowers, surrounded by a frame of intertwining vines and woodland creatures",
            6,
        ),
        (
            "A surreal desert landscape with melting clocks draped over cacti, inspired by Salvador Dali, with a fiery orange sky and mysterious shadows",
            7,
        ),
        ("A hyper-realistic close-up of a dewdrop on a vibrant blue morpho butterfly wing, reflecting a miniature rainforest scene", 8),
        (
            "An epic battle scene between medieval knights and fire-breathing dragons on a craggy mountaintop, with magic spells illuminating the dark storm clouds",
            9,
        ),
        ("A peaceful Zen garden on an alien planet, with levitating rocks, bioluminescent plants, and two moons rising in the pastel sky", 10),
    ]
    prompts = [p[0] for p in image_prompts]
    fnames = [p[1] for p in image_prompts]
    Model().f.remote(prompts, fnames)
