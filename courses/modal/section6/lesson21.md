# Image Embedding Processing

- In this lesson we start with a series of image urls
- We will process the image urls in two phases
    - stage 1: download the images on CPU containers and save the pre-processed images to a volume
    - stage 2: load in the pre-processed images from the volume and pass them to a GPU to run model inference to 
    create embeddings 
- The embedding model we will use is `google/siglip2-so400m-patch16-naflex` which
you can find on Hugging Face [here](https://huggingface.co/google/siglip2-so400m-patch16-naflex).
- To proceed with this lesson you need a file with a list of image urls. This can come from anywhere, for example [Open Images Dataset](https://github.com/cvdfoundation/open-images-dataset). 


In this lesson we will be creating three python files and one text file.

First create a python file called `image_pre_processing.py` and add the following code:

```python
import asyncio
import hashlib
import io
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List

import aiohttp
import modal

app = modal.App("image-pre-processing")

# Configuration
USE_FLOAT16 = True  # Set to False to use float32

download_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "aiohttp", "pillow", "requests", "transformers", "torch", "torchvision"
)

image_volume = modal.Volume.from_name(
    "image-processing-storage", create_if_missing=True
)
hf_hub_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

VOLUME_MOUNT_PATH = "/vol/images"
HF_CACHE_PATH = "/root/.cache/huggingface/hub/"

with download_image.imports():
    import torch
    from PIL import Image as PILImage
    from transformers import AutoProcessor


@app.function(
    image=download_image,
    volumes={VOLUME_MOUNT_PATH: image_volume, HF_CACHE_PATH: hf_hub_cache},
    timeout=60 * 30,
)
async def download_and_store_images(
    image_urls: List[str],
    batch_name: str,
    max_concurrent: int = 50,
    timeout_per_image: int = 30,
) -> Dict[str, Any]:
    """
    Stage 1: Download images asynchronously and store them on Modal Volume

    Args:
        image_urls: List of URLs to download
        batch_name: Name for this batch (for organizing storage)
        max_concurrent: Max concurrent downloads
        timeout_per_image: Timeout per image download

    Returns:
        Dict with download statistics and batch info
    """

    # Initialize processor
    ckpt = "google/siglip2-so400m-patch16-naflex"
    processor = AutoProcessor.from_pretrained(ckpt)

    # Create batch directory
    batch_dir = Path(VOLUME_MOUNT_PATH) / batch_name
    batch_dir.mkdir(parents=True, exist_ok=True)

    # Semaphore to limit concurrent downloads
    semaphore = asyncio.Semaphore(max_concurrent)

    async def download_single_image(
        session: aiohttp.ClientSession, url: str
    ) -> Dict[str, Any]:
        """Download a single image with error handling - returns image data for batch storage"""
        async with semaphore:
            try:
                # Create deterministic filename based on URL hash
                url_hash = hashlib.md5(url.encode("utf-8")).hexdigest()

                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=timeout_per_image)
                ) as response:
                    if response.status == 200:
                        content = await response.read()

                        # Convert to PIL Image

                        pil_image = PILImage.open(io.BytesIO(content))
                        if pil_image.mode != "RGB":
                            pil_image = pil_image.convert("RGB")

                        # Return image data for batch storage (not saving individually)
                        return {
                            "status": "success",
                            "url": url,
                            "url_hash": url_hash,
                            "pil_image": pil_image,
                            "download_time": time.time(),
                            "image_size": pil_image.size,
                        }
                    else:
                        print(f"HTTP {response.status} for {url[:50]}...")
                        return {
                            "status": "failed",
                            "url": url,
                            "error": f"HTTP {response.status}",
                        }

            except Exception as e:
                print(f"Download error for {url[:50]}...: {e}")
                return {"status": "failed", "url": url, "error": str(e)}

    # Start all downloads concurrently
    start_time = time.time()

    async with aiohttp.ClientSession() as session:
        tasks = [download_single_image(session, url) for url in image_urls]
        results = await asyncio.gather(*tasks)

    download_time = time.time() - start_time

    # Process results
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]

    # Process all successful images with the processor
    processing_start = time.time()
    pil_images = [r["pil_image"] for r in successful]

    if pil_images:
        # Process all images at once
        processed_inputs = processor(images=pil_images, return_tensors="pt")

        # Convert to float16 if enabled
        if USE_FLOAT16:
            for key in processed_inputs:
                if processed_inputs[key].dtype == torch.float32:
                    processed_inputs[key] = processed_inputs[key].half()
    else:
        processed_inputs = None

    processing_time = time.time() - processing_start

    # Prepare batch data - collect all processed inputs and metadata
    batch_data = {
        "batch_name": batch_name,
        "processed_inputs": processed_inputs,  # Contains the processed tensor inputs
        "image_metadata": [],  # Metadata for each image
        "metadata": {
            "total_urls": len(image_urls),
            "successful_downloads": len(successful),
            "failed_downloads": len(failed),
            "download_time": download_time,
            "processing_time": processing_time,
            "download_rate": len(successful) / download_time
            if download_time > 0
            else 0,
            "failed_urls": [r["url"] for r in failed],
            "created_at": time.time(),
        },
    }

    # Add metadata for each successful image
    for result in successful:
        image_entry = {
            "url_hash": result["url_hash"],
            "original_url": result["url"],
            "download_time": result["download_time"],
            "image_size": result["image_size"],
        }
        batch_data["image_metadata"].append(image_entry)

    # Save entire batch as SINGLE file
    batch_file_path = batch_dir / "batch_processed.pkl"
    with open(batch_file_path, "wb") as f:
        pickle.dump(batch_data, f)

    return batch_data["metadata"]


@app.local_entrypoint()
def main():
    # Load image URLs
    with open("image_urls.txt", "r") as f:
        image_urls = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(image_urls)} image URLs")

    # Split URLs into batches
    batch_size = 512
    url_batches = [
        image_urls[i : i + batch_size] for i in range(0, len(image_urls), batch_size)
    ]

    print(f"Split into {len(url_batches)} batches of up to {batch_size} URLs each")

    # Download and store images using Modal map for parallel processing
    base_timestamp = int(time.time())
    batch_names = [f"batch_{base_timestamp}_{i:03d}" for i in range(len(url_batches))]

    print(f"\n=== Downloading {len(url_batches)} batches in parallel ===")

    # Use Modal starmap to process all download batches in parallel
    download_inputs = list(zip(url_batches, batch_names))
    for d in download_and_store_images.starmap(download_inputs):
        pass

```

The second file we will create is `image_embeddings.py` and add the following code:

```python
import pickle
import time
from pathlib import Path
from typing import Any, Dict

import modal

app = modal.App("image-embeddings")

gpu_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch", "transformers", "accelerate", "pillow", "torchvision"
)

with gpu_image.imports():
    import torch
    from transformers import AutoModel, AutoProcessor

image_volume = modal.Volume.from_name(
    "image-processing-storage", create_if_missing=True
)
hf_hub_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

VOLUME_MOUNT_PATH = "/vol/images"
HF_CACHE_PATH = "/root/.cache/huggingface/hub/"

INPUT_CONCURRENCY = 1
MAX_CONTAINERS = 5


@app.cls(
    image=gpu_image,
    volumes={VOLUME_MOUNT_PATH: image_volume, HF_CACHE_PATH: hf_hub_cache},
    gpu="L40S",
    timeout=60 * 60,
    max_containers=MAX_CONTAINERS,
)
@modal.concurrent(max_inputs=INPUT_CONCURRENCY)
class ImageProcessor:
    @modal.enter()
    def setup(self):
        """Load model once"""
        print("Loading SigLIP model...")
        ckpt = "google/siglip2-so400m-patch16-naflex"
        self.model = AutoModel.from_pretrained(ckpt, device_map="auto").eval()
        self.processor = AutoProcessor.from_pretrained(ckpt)
        print("Model loaded")

    @modal.method()
    def process_batch_chunk(self, batch_files: list) -> Dict[str, Any]:
        """Process a chunk of batch files"""
        print(f"Starting GPU processing for {len(batch_files)} batch files")

        total_processed = 0
        total_start_time = time.time()

        # Loop over each batch file
        for batch_file_path in batch_files:
            # Construct full path by combining volume mount path with volume path
            full_path = Path(VOLUME_MOUNT_PATH) / batch_file_path.lstrip("/")

            # Time entire batch processing
            batch_total_start = time.time()

            # Time file reading
            read_start = time.time()

            with open(full_path, "rb") as f:
                batch_data = pickle.load(f)

            processed_inputs = batch_data["processed_inputs"]
            image_metadata = batch_data["image_metadata"]
            read_time = time.time() - read_start

            # Check if we have processed inputs
            if processed_inputs is None:
                print(f"No processed inputs found in {batch_file_path}")
                continue

            # Transfer to GPU
            inputs = processed_inputs.to(self.model.device)

            # Time forward pass
            forward_start = time.time()
            with torch.no_grad():
                embeddings = self.model.get_image_features(**inputs)
            # Wait for the kernel to complete before stopping the timer
            torch.cuda.synchronize()
            batch_inference_time = time.time() - forward_start

            # Move embeddings to CPU and save them
            embeddings_cpu = embeddings.cpu()

            # Create embeddings data structure
            embeddings_data = {
                "embeddings": embeddings_cpu,
                "image_metadata": image_metadata,
                "batch_name": batch_data["batch_name"],
                "embedding_shape": embeddings_cpu.shape,
                "created_at": time.time(),
                "model_checkpoint": "google/siglip2-so400m-patch16-naflex",
            }

            # Save embeddings to the same directory as the batch file
            batch_dir = full_path.parent
            embeddings_file_path = batch_dir / "batch_embeddings.pkl"

            save_start = time.time()
            with open(embeddings_file_path, "wb") as f:
                pickle.dump(embeddings_data, f)
            save_time = time.time() - save_start

            total_processed += inputs["pixel_values"].shape[0]

            batch_total_time = time.time() - batch_total_start

            # Single print statement per batch with all timing info
            print(
                f"Batch {batch_file_path}: {len(image_metadata)} imgs | "
                f"read={read_time:.3f}s, "
                f"inference={batch_inference_time:.3f}s, "
                f"save={save_time:.3f}s, total={batch_total_time:.3f}s | "
                f"embeddings shape: {embeddings_cpu.shape}"
            )

        total_time = time.time() - total_start_time
        throughput = total_processed / total_time

        print(
            f"Completed! {total_processed} images in {total_time:.2f}s ({throughput:.2f} img/sec)"
        )

        return {
            "total_processed": total_processed,
            "total_time": total_time,
            "throughput": throughput,
        }


@app.local_entrypoint()
def main():
    # Find all batch files using volume's listdir method
    all_files = image_volume.listdir("/", recursive=True)
    batch_files = [f.path for f in all_files if f.path.endswith("batch_processed.pkl")]
    print(f"Found {len(batch_files)} batch files")

    # Split batch files into chunks for concurrent processing
    chunk_size = max(1, len(batch_files) // (MAX_CONTAINERS * INPUT_CONCURRENCY))
    chunks = [
        batch_files[i : i + chunk_size] for i in range(0, len(batch_files), chunk_size)
    ]
    print(f"Split into {len(chunks)} chunks")

    processor = ImageProcessor()
    # Use starmap to process chunks concurrently
    chunk_args = [(chunk,) for chunk in chunks]
    results = list(processor.process_batch_chunk.starmap(chunk_args))

    # Combine results
    total_processed = sum(r["total_processed"] for r in results)
    total_time = max(r["total_time"] for r in results)
    throughput = total_processed / total_time

    print(
        f"Combined Result: {total_processed} images in {total_time:.2f}s ({throughput:.2f} img/sec)"
    )

```

And the final file we will create is `image_similarity_demo.py` and add the following code:

```python
import io
import pickle
from pathlib import Path
from typing import Any, Dict

import modal

app = modal.App("image-similarity-demo")

# Image with required dependencies
demo_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch",
    "transformers",
    "accelerate",
    "pillow",
    "faiss-cpu",
    "numpy",
    "fastapi",
    "aiohttp",
    "requests",
)

# Use the same volumes as the existing apps
image_volume = modal.Volume.from_name(
    "image-processing-storage", create_if_missing=True
)
hf_hub_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

VOLUME_MOUNT_PATH = "/vol/images"
HF_CACHE_PATH = "/root/.cache/huggingface/hub/"

with demo_image.imports():
    import aiohttp
    import faiss
    import numpy as np
    import torch
    from PIL import Image as PILImage
    from transformers import AutoModel, AutoProcessor


@app.cls(
    image=demo_image,
    volumes={VOLUME_MOUNT_PATH: image_volume, HF_CACHE_PATH: hf_hub_cache},
    timeout=60 * 30,
    scaledown_window=60 * 10,
)
class ImageSimilaritySearch:
    @modal.enter()
    def setup(self):
        """Load embeddings and build FAISS index on container startup"""
        print("Setting up image similarity search...")

        # Load the same model for processing input images
        print("Loading SigLIP model...")
        ckpt = "google/siglip2-so400m-patch16-naflex"
        self.model = AutoModel.from_pretrained(ckpt).eval()
        self.processor = AutoProcessor.from_pretrained(ckpt)
        print("Model loaded")

        # Load all embeddings from volume
        print("Loading embeddings from volume...")
        self.embeddings_list = []
        self.metadata_list = []

        # Find all embedding files
        all_files = image_volume.listdir("/", recursive=True)
        embedding_files = [
            f.path for f in all_files if f.path.endswith("batch_embeddings.pkl")
        ]
        print(f"Found {len(embedding_files)} embedding files")

        for embedding_file_path in embedding_files:
            full_path = Path(VOLUME_MOUNT_PATH) / embedding_file_path.lstrip("/")

            try:
                with open(full_path, "rb") as f:
                    embeddings_data = pickle.load(f)

                embeddings = embeddings_data["embeddings"]  # torch tensor
                image_metadata = embeddings_data["image_metadata"]

                # Convert to numpy for FAISS
                embeddings_np = embeddings.numpy()

                # Store embeddings and metadata
                self.embeddings_list.append(embeddings_np)
                self.metadata_list.extend(image_metadata)

                print(
                    f"Loaded {embeddings_np.shape[0]} embeddings from {embedding_file_path}"
                )

            except Exception as e:
                print(f"Error loading {embedding_file_path}: {e}")

        if not self.embeddings_list:
            print("No embeddings found!")
            self.index = None
            self.all_embeddings = None
            return

        # Concatenate all embeddings
        self.all_embeddings = np.vstack(self.embeddings_list)
        print(f"Total embeddings: {self.all_embeddings.shape}")

        # Build FAISS index
        print("Building FAISS index...")
        dimension = self.all_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.all_embeddings)
        self.index.add(self.all_embeddings.astype(np.float32))

        print(f"FAISS index built with {self.index.ntotal} vectors")
        print("Setup complete!")

    async def process_url_and_search(
        self, image_url: str, k: int = 5
    ) -> Dict[str, Any]:
        """Download image from URL, get embedding, and find similar images"""
        if self.index is None:
            return {"error": "No embeddings loaded"}

        try:
            # Download image
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    image_url, timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status != 200:
                        return {
                            "error": f"Failed to download image: HTTP {response.status}"
                        }

                    content = await response.read()

            # Convert to PIL Image
            pil_image = PILImage.open(io.BytesIO(content))
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")

            # Process with the same processor
            inputs = self.processor(images=[pil_image], return_tensors="pt")

            # Get embedding
            with torch.no_grad():
                embedding = self.model.get_image_features(**inputs)

            # Convert to numpy and normalize
            query_embedding = embedding.cpu().numpy().astype(np.float32)
            faiss.normalize_L2(query_embedding)

            # Search for similar images
            scores, indices = self.index.search(query_embedding, k)

            # Prepare results
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.metadata_list):
                    metadata = self.metadata_list[idx]
                    results.append(
                        {
                            "rank": i + 1,
                            "similarity_score": float(score),
                            "image_url": metadata["original_url"],
                            "image_size": metadata["image_size"],
                        }
                    )

            return {
                "query_url": image_url,
                "query_image_size": pil_image.size,
                "similar_images": results,
                "total_database_size": len(self.metadata_list),
            }

        except Exception as e:
            return {"error": f"Error processing image: {str(e)}"}

    @modal.fastapi_endpoint(method="GET")
    async def search_similar_images(self, image_url: str, k: int = 5):
        """Web endpoint to search for similar images"""
        if not image_url:
            return {"error": "image_url parameter is required"}

        if k < 1 or k > 50:
            return {"error": "k must be between 1 and 50"}

        result = await self.process_url_and_search(image_url, k)
        return result

    @modal.fastapi_endpoint(method="GET")
    async def health(self):
        """Health check endpoint"""
        return {
            "status": "healthy",
            "database_size": len(self.metadata_list)
            if hasattr(self, "metadata_list")
            else 0,
            "index_ready": self.index is not None if hasattr(self, "index") else False,
        }


def main():
    """Test the similarity search locally via web endpoint"""
    import requests

    test_url = "https://c6.staticflickr.com/5/4066/4332278877_e90c3d4598_o.jpg"
    # CHANGE THIS TO YOUR ENDPOINT URL
    endpoint_url = "https://drchrislevy--image-similarity-demo-imagesimilaritysearch-e57712.modal.run"

    print(f"Testing similarity search with: {test_url}")

    response = requests.get(endpoint_url, params={"image_url": test_url, "k": 10})

    if response.status_code == 200:
        result = response.json()
        print("\nResults:")
        print(f"Query URL: {result.get('query_url', 'N/A')}")
        print(f"Database size: {result.get('total_database_size', 'N/A')}")

        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print("\nSimilar images:")
            for img in result.get("similar_images", []):
                print(
                    f"  Rank {img['rank']}: {img['similarity_score']:.3f} - {img['image_url']}"
                )
    else:
        print(f"HTTP Error: {response.status_code}")
        print(response.text)
```

The 4th file is a list of image urls separated by new lines which can go in
a file called `image_urls.txt`. Example format is:

```
https://farm1.staticflickr.com/5615/15335861457_ec2be7a54e_o.jpg
https://farm5.staticflickr.com/5582/18233009494_029b52ca79_o.jpg
https://c5.staticflickr.com/9/8549/8704105458_a035405f0f_o.jpg
https://farm7.staticflickr.com/7416/9940862744_4ef1834d86_o.jpg
```






