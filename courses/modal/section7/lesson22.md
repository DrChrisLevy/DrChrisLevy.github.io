# Image Generation with FLUX.1 Krea

- Here are some links about this Model which came out at the end of July 2025
    - [blog post 1](https://www.krea.ai/blog/flux-krea-open-source-release)
    - [blog post 2](https://bfl.ai/announcements/flux-1-krea-dev)
    - [model card](https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev)

Create a python file called `image_gen.py` and add the following code below.


```python
import os

import modal

# Modal Volume URL configuration
MODAL_WORKSPACE = "drchrislevy"  # replace with your modal workspace
MODAL_ENVIRONMENT = "main"  # replace with your modal environment
VOLUME_NAME = "flux_generated_images"  # replace with your modal volume name

# Image with required dependencies
image = modal.Image.debian_slim().pip_install(
    [
        "torch",
        "torchvision",
        "diffusers",
        "transformers",
        "accelerate",
        "pillow",
        "sentencepiece",
        "python-dotenv",
    ]
)

app = modal.App("flux-image-generator", image=image)

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
    scaledown_window=10 * 60,
    max_containers=2,
)
@modal.concurrent(max_inputs=1)
class FluxImageGenerator:
    @modal.enter()
    def setup(self):
        """Load FLUX model once per container"""
        import torch
        from diffusers import FluxPipeline

        print("Loading FLUX.1-Krea-dev model...")
        # https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev
        self.pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Krea-dev",
            torch_dtype=torch.bfloat16,
            cache_dir="/root/.cache/huggingface/hub",
        )

        self.pipe.enable_model_cpu_offload()

        print("Model loaded successfully!")

        # Set up volume path for storing images
        self.images_path = "/root/generated_images"

    def generate_image(
        self,
        prompt: str,
        height: int = 720,
        width: int = 1280,
        guidance_scale: float = 4.5,
        seed: int = 42,
        randomize_seed=False,
        num_inference_steps: int = 28,
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
        generator = torch.Generator().manual_seed(seed)
        print(f"Generating image for prompt: {prompt}")

        # Generate image
        image = self.pipe(
            prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            generator=generator,
            num_inference_steps=num_inference_steps,
        ).images[0]

        # Create unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"flux_generated_{timestamp}_{unique_id}.png"
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
        height: int = 720,
        width: int = 1280,
        guidance_scale: float = 4.5,
        seed: int = 42,
        randomize_seed: bool = False,
        num_inference_steps: int = 28,
    ):
        """Public FastAPI endpoint for image generation"""
        return self.generate_image(
            prompt=prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            seed=seed,
            randomize_seed=randomize_seed,
            num_inference_steps=num_inference_steps,
        )


def test_image_generation():
    """
    Test function for the FLUX image generator endpoint.

    Prerequisites:
    1. First deploy the modal app: uv run modal deploy image_gen.py
    2. Replace the endpoint URL below with your actual deployment URL
    """

    import requests

    # Replace this with your actual deployment endpoint URL
    # You can find it after running: uv run modal deploy image_gen.py
    endpoint_url = "https://drchrislevy--flux-image-generator-fluximagegenerator-gen-a9b738.modal.run"

    # Photorealistic test prompts
    test_prompts = [
        "Candid street portrait of a woman waiting at a crosswalk in light rain, clear umbrella, city lights reflecting on wet asphalt, 35mm lens, f/2, ISO 800, 1/125s, natural tungsten street lighting, shallow depth of field",
        "Dad serving pancakes to three kids in a cozy kitchen at 7am, steam rising, messy table with syrup and fruit, 50mm lens, f/2.8, ISO 400, 1/200s, window side-light, real crumbs and spills",
        "Runner tying shoelaces on a frosty boardwalk at sunrise by the ocean, visible breath, worn sneakers, 24mm lens, f/4, ISO 200, 1/800s, golden hour backlight, authentic athletic wear",
        "Close-up of a mechanic’s hands replacing a brake pad in a small garage, grease under fingernails, scattered tools, 85mm lens, f/2.2, ISO 640, 1/250s, overhead fluorescent lighting",
        "Two friends playing chess in a park on an overcast afternoon, pigeons nearby, fallen leaves on the bench, 35mm lens, f/2.8, ISO 320, 1/250s, soft diffused light",
        "Small-town barbershop interior, mid-fade haircut in progress, mirrors with fingerprints, hair on the floor, 28mm lens, f/4, ISO 800, 1/160s, mixed window and neon sign lighting",
        "Teenager practicing guitar in a cluttered bedroom, poster-covered walls, open laptop, natural window light, 35mm lens, f/1.8, ISO 500, 1/200s, authentic cable mess",
        "Grocery store checkout moment, cashier scanning produce, reusable bags, real pricing stickers, 50mm lens, f/3.2, ISO 1000, 1/160s, overhead supermarket lighting",
        "Rainy bus stop scene at night, commuters in hoodies, bus approaching with headlights flaring, puddle splashes, 35mm lens, f/2, ISO 1600, 1/125s, cinematic wet reflections",
        "Home office desk at 10pm, developer debugging code, secondary monitor glow, half-finished tea, sticky notes, 35mm lens, f/2.2, ISO 800, 1/60s, practical lamp light only",
        "Saturday farmers’ market, vendor weighing tomatoes on a scale, canvas tents, uneven pavement, 24mm lens, f/5.6, ISO 200, 1/400s, bright but slightly cloudy daylight",
        "Family minivan in a driveway, teen learning to check oil, instruction manual open, real smudges on hands, 35mm lens, f/2.8, ISO 400, 1/320s, midday shade",
        "Elderly couple walking a small dog on a suburban sidewalk at dusk, porch lights turning on, 85mm lens, f/2, ISO 1600, 1/200s, gentle backlight with natural flare",
        "University library table, student highlighting a textbook, laptop open with notes, coffee ring stains, 50mm lens, f/2.8, ISO 640, 1/125s, overhead warm lighting",
        "Street food truck lunch rush, cook flipping tortillas on a griddle, steam and smoke, people waiting with phones, 28mm lens, f/4, ISO 400, 1/500s, harsh noon light with shadows",
        "Snowy driveway at dawn, person scraping ice off a windshield, visible snow grains, breath fog, 35mm lens, f/2.8, ISO 800, 1/320s, cold blue ambient light",
        "Apartment laundry room, person folding warm clothes on a metal table, detergent bottle and lone sock, 35mm lens, f/2.5, ISO 1000, 1/160s, fluorescent lighting",
        "Suburban backyard barbecue, dad flipping burgers, kids chasing a soccer ball, paper plates on a picnic table, 24mm lens, f/4, ISO 200, 1/800s, late afternoon sun",
        "Morning commuter train interior, soft crowd, reflections in the window, person reading a paperback, 35mm lens, f/2, ISO 1600, 1/100s, mixed daylight and overhead lighting",
        "Corner hardware store aisle, customer comparing paint swatches, dust motes in the air, uneven shelves, 50mm lens, f/2.8, ISO 800, 1/200s, practical store lights",
    ]

    print("Testing FLUX Image Generator...")
    print(f"Endpoint: {endpoint_url}")
    print("-" * 50)

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nTest {i}: {prompt[:50]}...")

        try:
            # Make POST request to the endpoint with query parameters
            response = requests.post(
                endpoint_url,
                params={
                    "prompt": prompt,
                    "height": 720,
                    "width": 1280,
                    "guidance_scale": 4.5,
                    "seed": 42,
                    "randomize_seed": True,
                    "num_inference_steps": 28,
                },
                timeout=120,  # 2 minute timeout for image generation
            )

            if response.status_code == 200:
                result = response.json()
                print(f"✅ Success! Image URL: {result['image_url']}")
                print(f"   Filename: {result['filename']}")
            else:
                print(f"❌ Error {response.status_code}: {response.text}")

        except requests.exceptions.Timeout:
            print("❌ Request timed out")
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")

    print("\n" + "=" * 50)
    print("Image generation testing completed!")
```