# Image Generation with Image Qwen

- Here are some links about this Model which came out August, 2025
    - [blog post](https://qwenlm.github.io/blog/qwen-image/)
    - [model card](https://huggingface.co/Qwen/Qwen-Image)

Create a python file called `image_gen_qwen.py` and add the following code below.
Deploy it with `uv run modal deploy image_gen_qwen.py`.

```python
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
    scaledown_window=10 * 60,
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


def test_image_generation():
    """
    Test function for the Qwen Image generator endpoint.

    Prerequisites:
    1. First deploy the modal app: uv run modal deploy image_gen.py
    2. Replace the endpoint URL below with your actual deployment URL
    """

    import requests

    # Replace this with your actual deployment endpoint URL
    # You can find it after running: uv run modal deploy image_gen.py
    endpoint_url = "https://drchrislevy--qwen-image-generator-qwenimagegenerator-gen-5fbcf5.modal.run/"

    # Test prompts for the new Qwen model
    test_prompts = [
        "A man in a business suit standing in front of a large screen, shot from behind-the-scenes perspective of a game show set. The suit is split vertically down the middle: left half bright green, right half bright orange. The screen behind him is also split vertically: left half pink, right half blue. Realistic photography style, professional lighting, behind-the-scenes candid shot, game show studio environment, detailed fabric textures on the suit, crisp color separation",
        """Bookstore window display. A sign displays “New Arrivals This Week”. Below, a shelf tag with the text “Best-Selling Novels Here”. To the side, a colorful poster advertises “Author Meet And Greet on Saturday” with a central portrait of the author. There are four books on the bookshelf, namely “The light between worlds” “When stars are scattered” “The slient patient” “The night circus”""",
        """slide featuring artistic, decorative shapes framing neatly arranged textual information styled as an elegant infographic. At the very center, the title “Habits for Emotional Wellbeing” appears clearly, surrounded by a symmetrical floral pattern. On the left upper section, “Practice Mindfulness” appears next to a minimalist lotus flower icon, with the short sentence, “Be present, observe without judging, accept without resisting”. Next, moving downward, “Cultivate Gratitude” is written near an open hand illustration, along with the line, “Appreciate simple joys and acknowledge positivity daily”. Further down, towards bottom-left, “Stay Connected” accompanied by a minimalistic chat bubble icon reads “Build and maintain meaningful relationships to sustain emotional energy”. At bottom right corner, “Prioritize Sleep” is depicted next to a crescent moon illustration, accompanied by the text “Quality sleep benefits both body and mind”. Moving upward along the right side, “Regular Physical Activity” is near a jogging runner icon, stating: “Exercise boosts mood and relieves anxiety”. Finally, at the top right side, appears “Continuous Learning” paired with a book icon, stating “Engage in new skill and knowledge for growth”. The slide layout beautifully balances clarity and artistry, guiding the viewers naturally along each text segment.""",
        """A man in a suit is standing in front of the window, looking at the bright moon outside the window. The man is holding a yellowed paper with handwritten words on it: “A lantern moon climbs through the silver night, Unfurling quiet dreams across the sky, Each star a whispered promise wrapped in light, That dawn will bloom, though darkness wanders by.” There is a cute cat on the windowsill.""",
        """A movie poster. The first row shows the title in bold serif: “The Last Lightkeeper”. The second row is the tagline: “When the sea forgets, only one remembers.” The third row reads “Starring: Aria Noven”. The fourth row reads “Directed by: Lysander Vale”. The central image shows a solitary lighthouse perched on a jagged cliff, battered by towering waves under a stormy, violet-hued sky. A lone figure stands at the edge of the lighthouse balcony, holding a glowing lantern that casts a beam into the chaos. In the light, ghostly silhouettes of ships and lost souls shimmer in the mist. The color palette is moody and cinematic—deep blues, foggy whites, and spectral light effects. The background is layered with clouds, crashing water, and scattered stars piercing through the storm. At the bottom, the text reads “In Theatres November 2025” in subtle metallic lettering, with the glint of salt-worn steel. The style combines atmospheric realism with supernatural drama, rendered in high-resolution digital painting, with cinematic lighting and emotional depth.""",
    ]

    print("Testing Qwen Image Generator...")
    print(f"Endpoint: {endpoint_url}")
    print("-" * 50)

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nTest {i}: {prompt[:100]}...")

        try:
            # Make POST request to the endpoint with query parameters
            response = requests.post(
                endpoint_url,
                params={
                    "prompt": prompt,
                    "negative_prompt": "blurry, low quality, distorted",
                    "aspect_ratio": "16:9",
                    "true_cfg_scale": 2.0,
                    "randomize_seed": True,
                    "num_inference_steps": 50,
                },
                timeout=120,  # 2 minute timeout for image generation
            )

            if response.status_code == 200:
                result = response.json()
                print(f"✅ Success! Image URL: {result['image_url']}")
                print(f"   Filename: {result['filename']}")
                print(
                    f"   Dimensions: {result['dimensions']['width']}x{result['dimensions']['height']}"
                )
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