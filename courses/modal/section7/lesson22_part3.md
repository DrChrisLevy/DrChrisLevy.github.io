
# Running Qwen Image Edit on Modal

## Qwen-Image-Edit

- The editing version of Qwen Image just came out on August 19, 2025, `Qwen-Image-Edit`.
- [blog post](https://qwenlm.github.io/blog/qwen-image-edit/)
- [X announcement](https://x.com/Alibaba_Qwen/status/1957500569029079083)
- [Hugging Face Model Card](https://huggingface.co/Qwen/Qwen-Image-Edit)
- [Github Repo](https://github.com/QwenLM/Qwen-Image)


# The Code

## Qwen Image Edit inference

- put the code in a file, `qwen_image_edit.py` and deploy the endpoint with `modal deploy qwen_image_edit.py`

```python
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


```

## Qwen Image Edit Inference (FAST!)

- code is nearly identical to above except it uses [lightx2v/Qwen-Image-Lightning](https://huggingface.co/lightx2v/Qwen-Image-Lightning)
- The inference code snippet I got from [here](https://huggingface.co/spaces/multimodalart/Qwen-Image-Edit-Fast/blob/main/app.py)
- uses less inference steps, 8 instead of 50
- put the code in a file, `qwen_image_edit_fast.py` and deploy the endpoint with `modal deploy qwen_image_edit_fast.py`


```python
import io
import math
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
            "peft",
        ]
    )
)

app = modal.App("qwen-image-editor-fast", image=image)

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
        """Load Qwen-Image-Edit model with Lightning acceleration once per container"""
        import torch
        from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPipeline

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
        self.pipe = QwenImageEditPipeline.from_pretrained(
            "Qwen/Qwen-Image-Edit", scheduler=scheduler, torch_dtype=dtype
        ).to(device)
        self.pipe.load_lora_weights(
            "lightx2v/Qwen-Image-Lightning",
            weight_name="Qwen-Image-Lightning-8steps-V1.1.safetensors",
        )
        self.pipe.fuse_lora()
        self.pipe.set_progress_bar_config(disable=None)
        print("Model loaded successfully with Lightning acceleration!")
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
        num_inference_steps: int = 8,
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
        num_inference_steps: int = 8,
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


# Testing the Endpoints - Examples

## Testing Script 1:

```python
def test_image_editing():
    """
    Test function for the Qwen Image Editor endpoint.

    Prerequisites:
    1. First deploy the modal app: uv run modal deploy qwen_image_edit.py
    2. Replace the endpoint URL below with your actual deployment URL
    3. Prepare test image urls
    """

    import requests

    # Replace this with your actual deployment endpoint URL
    # You can find it after running: uv run modal deploy qwen_image_edit.py
    endpoint_url = "https://drchrislevy--qwen-image-editor-qwenimageeditor-edit-imag-10fda9.modal.run/"
    seed = 0
    randomize_seed = False
    true_cfg_scale = 4.0
    negative_prompt = " "
    num_inference_steps = 40
    test_cases = [
        {
            "image_url": "https://img.alicdn.com/imgextra/i3/O1CN01XfJ71c1qokTchToKf_!!6000000005543-2-tps-1248-832.png?x-oss-process=image/resize,m_mfit,w_320,h_320",
            "prompt": "Add a small wooden sign in the foreground in front of the penguins with the text 'Welcome to Penguin Beach'",
        },
        {
            "image_url": "https://img.alicdn.com/imgextra/i3/O1CN01m3Jkqd1UoMb4edofx_!!6000000002564-2-tps-832-1248.png?x-oss-process=image/resize,m_mfit,w_320,h_320",
            "prompt": "change the girl's hair color to blond",
        },
        {
            "image_url": "https://img.alicdn.com/imgextra/i2/O1CN01Ytal7H1LVHYOdVgWX_!!6000000001304-2-tps-1024-1024.png?x-oss-process=image/resize,m_mfit,w_320,h_320",
            "prompt": "remove the fork",
        },
        {
            "image_url": "https://img.alicdn.com/imgextra/i2/O1CN01Ytal7H1LVHYOdVgWX_!!6000000001304-2-tps-1024-1024.png?x-oss-process=image/resize,m_mfit,w_320,h_320",
            "prompt": "change the price of the menu item 'kineses' to $20",
        },
    ]

    print("Testing Qwen Image Editor...")
    print(f"Endpoint: {endpoint_url}")
    print("-" * 70)

    for i, test_case in enumerate(test_cases, 1):
        start_time = time.time()
        print(f"\nTest {i}: {test_case['prompt'][:50]}...")
        print(f"Image URL: {test_case['image_url']}")
        print(f"Prompt: {test_case['prompt']}")

        response = requests.post(
            endpoint_url,
            params={
                "image_url": test_case["image_url"],
                "prompt": test_case["prompt"],
                "negative_prompt": negative_prompt,
                "true_cfg_scale": true_cfg_scale,
                "seed": seed,
                "randomize_seed": randomize_seed,
                "num_inference_steps": num_inference_steps,
            },
            timeout=180,  # 3 minute timeout for image editing
        )
        result = response.json()
        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")
        print(f"✅ Success! Edited image URL: {result['image_url']}")
```

## Testing Script 2:

```{python}
import subprocess
import time

import requests
from PIL import Image

endpoint_url = "https://drchrislevy--qwen-image-editor-qwenimageeditor-edit-imag-10fda9.modal.run/"
endpoint_url_fast = "https://drchrislevy--qwen-image-editor-fast-qwenimageeditor-edit-516e26.modal.run/"


def download_generated_image(result):
    time.sleep(5)  # wait for volume sync
    filename = result["image_url"].split("path=")[-1]
    download_command = f"modal volume get qwen_edited_images {filename} ../static_blog_imgs/{filename} --force"
    subprocess.run(download_command, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    p = f"../static_blog_imgs/{filename}"
    return Image.open(p)


def generate_image(endpoint, image_url, prompt, num_inference_steps=None, negative_prompt=" ", true_cfg_scale=4.0, seed=0, randomize_seed=False):
    if num_inference_steps is None:
        if endpoint == endpoint_url:
            num_inference_steps = 50
        elif endpoint == endpoint_url_fast:
            num_inference_steps = 8
        else:
            raise ValueError(f"Invalid endpoint: {endpoint}")

    response = requests.post(
        endpoint,
        params={
            "image_url": image_url,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "true_cfg_scale": true_cfg_scale,
            "seed": seed,
            "randomize_seed": randomize_seed,
            "num_inference_steps": num_inference_steps,
        },
        timeout=180,
    )
    res = response.json()
    # image = Image.open(requests.get(image_url, stream=True).raw)
    # display(image)
    download_generated_image(res)
    # display(download_generated_image(res))
    return res
```

```{python}
image_url = (
    "https://img.alicdn.com/imgextra/i3/O1CN01XfJ71c1qokTchToKf_!!6000000005543-2-tps-1248-832.png?x-oss-process=image/resize,m_mfit,w_320,h_320"
)
res = generate_image(endpoint_url_fast, image_url, "transform this image into Ghibli style")
```


```{python}
image_url = (
    "https://img.alicdn.com/imgextra/i3/O1CN01m3Jkqd1UoMb4edofx_!!6000000002564-2-tps-832-1248.png?x-oss-process=image/resize,m_mfit,w_320,h_320"
)
res = generate_image(endpoint_url, image_url, "change the color of the purse to the same color of the jacket", num_inference_steps=10)
```


```{python}
image_url = (
    "https://img.alicdn.com/imgextra/i3/O1CN01m3Jkqd1UoMb4edofx_!!6000000002564-2-tps-832-1248.png?x-oss-process=image/resize,m_mfit,w_320,h_320"
)
res = generate_image(endpoint_url_fast, image_url, "change the color of the purse to the same color of the jacket", num_inference_steps=8)
```


```{python}
image_url = (
    "https://img.alicdn.com/imgextra/i3/O1CN01m3Jkqd1UoMb4edofx_!!6000000002564-2-tps-832-1248.png?x-oss-process=image/resize,m_mfit,w_320,h_320"
)
res = generate_image(endpoint_url_fast, image_url, "change the color of the hair to blonde", num_inference_steps=8)
```


```{python}
image_url = (
    "https://img.alicdn.com/imgextra/i3/O1CN01m3Jkqd1UoMb4edofx_!!6000000002564-2-tps-832-1248.png?x-oss-process=image/resize,m_mfit,w_320,h_320"
)
res = generate_image(endpoint_url, image_url, "change the color of the hair to blonde", num_inference_steps=20)
```


```{python}
image_url = (
    "https://img.alicdn.com/imgextra/i4/O1CN01hZNlck1mYwLJKmEaI_!!6000000004967-2-tps-1024-1024.png?x-oss-process=image/resize,m_mfit,w_320,h_320"
)
res = generate_image(endpoint_url_fast, image_url, "face to the right")
```


```{python}
image_url = (
    "https://img.alicdn.com/imgextra/i4/O1CN01hZNlck1mYwLJKmEaI_!!6000000004967-2-tps-1024-1024.png?x-oss-process=image/resize,m_mfit,w_320,h_320"
)
res = generate_image(endpoint_url_fast, image_url, "change the color of the door to yellow")
```


```{python}
image_url = (
    "https://img.alicdn.com/imgextra/i4/O1CN01hZNlck1mYwLJKmEaI_!!6000000004967-2-tps-1024-1024.png?x-oss-process=image/resize,m_mfit,w_320,h_320"
)
res = generate_image(endpoint_url_fast, image_url, "turn the dog around")
```


```{python}
image_url = (
    "https://img.alicdn.com/imgextra/i4/O1CN01hZNlck1mYwLJKmEaI_!!6000000004967-2-tps-1024-1024.png?x-oss-process=image/resize,m_mfit,w_320,h_320"
)
res = generate_image(endpoint_url_fast, image_url, "change the dog to sitting")
```


```{python}
image_url = (
    "https://img.alicdn.com/imgextra/i4/O1CN01GzeRec1ji9SfxtvRm_!!6000000004581-2-tps-1184-896.png?x-oss-process=image/resize,m_mfit,w_320,h_320"
)
prompt = "Restore old photograph, remove scratches, reduce noise, enhance details, high resolution, realistic, natural skin tones, clear facial features, no distortion, vintage photo restoration."
res = generate_image(endpoint_url_fast, image_url, prompt)
```


```{python}
image_url = "https://travelbynatasha.com/wp-content/uploads/2023/09/dsc_2272.jpg"
prompt = "Add a sunset to this image"
res = generate_image(endpoint_url_fast, image_url, prompt)
```


```{python}
image_url = "https://travelbynatasha.com/wp-content/uploads/2023/09/dsc_2272.jpg"
prompt = "Add a fishing boat to the image"
res = generate_image(endpoint_url_fast, image_url, prompt)
```

```{python}
image_url = "https://travelbynatasha.com/wp-content/uploads/2023/09/dsc_2272.jpg"
prompt = "Add a pod of killer whales swimming out of the cove"
res = generate_image(endpoint_url_fast, image_url, prompt, num_inference_steps=20)
```
