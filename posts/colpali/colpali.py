import base64
import hashlib
import os
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import List, cast

import modal
from dotenv import load_dotenv
from modal import build, enter

load_dotenv()
app = modal.App("colpali")

cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"
image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install("git")
    .pip_install(
        "ninja",  # required to build flash-attn
        "packaging",  # required to build flash-attn
        "wheel",  # required to build flash-attn
    )
    .run_commands(
        "pip install torch --index-url https://download.pytorch.org/whl/cu124",
        "pip install git+https://github.com/huggingface/transformers",
        "pip install accelerate",
        "pip install git+https://github.com/illuin-tech/colpali.git",
        "pip install python-dotenv",
        f'huggingface-cli login --token {os.environ["HUGGING_FACE_ACCESS_TOKEN"]}',
        "pip install flash-attn --no-build-isolation",
    )
    .apt_install("poppler-utils")
    .pip_install("pdf2image", "PyPDF2", "Pillow", "requests")
)

vol = modal.Volume.from_name("colpali-volume", create_if_missing=True)


@app.cls(image=image, secrets=[modal.Secret.from_dotenv()], volumes={"/data": vol}, gpu="a10g", cpu=4, timeout=600, container_idle_timeout=300)
class ColPaliModel:
    @build()
    @enter()
    def setup(self):
        import torch
        from colpali_engine.models import ColPali
        from colpali_engine.models.paligemma.colpali.processing_colpali import ColPaliProcessor
        from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor

        # Define adapter name
        base_model_name = "vidore/colpaligemma-3b-pt-448-base"
        adapter_name = "vidore/colpali-v1.2"

        # Load model
        self.model = ColPali.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        ).eval()
        self.model.load_adapter(adapter_name)
        self.processor = cast(ColPaliProcessor, ColPaliProcessor.from_pretrained("google/paligemma-3b-mix-448"))

        if not isinstance(self.processor, BaseVisualRetrieverProcessor):
            raise ValueError("Processor should be a BaseVisualRetrieverProcessor")

    @modal.method()
    def top_pages(self, pdf_url: str, queries: list[str], top_k=2):
        import numpy as np
        import torch
        from colpali_engine.utils.torch_utils import ListDataset
        from torch.utils.data import DataLoader
        from tqdm import tqdm

        images = self.pdf_to_images(pdf_url)
        batch_size = 8
        # Run inference - docs
        dataloader = DataLoader(
            dataset=ListDataset[str](images),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda x: self.processor.process_images(x),
        )
        ds: List[torch.Tensor] = []
        for batch_doc in tqdm(dataloader):
            with torch.no_grad():
                batch_doc = {k: v.to(self.model.device) for k, v in batch_doc.items()}
                embeddings_doc = self.model(**batch_doc)
            ds.extend(list(torch.unbind(embeddings_doc.to("cpu"))))

        # Run inference - queries
        dataloader = DataLoader(
            dataset=ListDataset[str](queries),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda x: self.processor.process_queries(x),
        )
        qs: List[torch.Tensor] = []
        for batch_query in dataloader:
            with torch.no_grad():
                batch_query = {k: v.to(self.model.device) for k, v in batch_query.items()}
                embeddings_query = self.model(**batch_query)
            qs.extend(list(torch.unbind(embeddings_query.to("cpu"))))

        # Run scoring
        scores = self.processor.score(qs, ds).cpu().numpy()
        # The top k indices for each query
        idxs_top_k = np.argsort(scores, axis=-1)[:, -top_k:][:, ::-1].tolist()
        return idxs_top_k

    def pdf_to_images(self, pdf_url):
        import requests
        from pdf2image import convert_from_bytes, pdfinfo_from_bytes

        start_time = time.time()

        response = requests.get(pdf_url)
        if response.status_code != 200:
            raise Exception(f"Failed to download PDF from {pdf_url}")

        pdf_bytes = BytesIO(response.content)
        pdf_info = pdfinfo_from_bytes(pdf_bytes.getvalue())
        num_pages = pdf_info["Pages"]

        def process_page(page_num):
            page_images = convert_from_bytes(pdf_bytes.getvalue(), first_page=page_num, last_page=page_num)
            return page_images[0]

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(process_page, page_num) for page_num in range(1, num_pages + 1)]
            images = [future.result() for future in futures]

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"PDF processing time: {execution_time:.2f} seconds")

        start_time = time.time()
        self.cache_pdf_images(pdf_url, images)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"PDF caching time: {execution_time:.2f} seconds")
        return images

    def pil_image_to_data_url(self, pil_image):
        # Convert PIL Image to bytes
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")

        # Encode to base64
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # Format as data URL
        return f"data:image/png;base64,{img_str}"

    def generate_unique_folder_name(self, pdf_url: str) -> str:
        # Create a hash of the URL
        url_hash = hashlib.md5(pdf_url.encode()).hexdigest()
        # Get the last part of the URL as the filename
        original_filename = os.path.basename(pdf_url)
        # Remove the file extension if present
        base_name = os.path.splitext(original_filename)[0]
        # Combine the base name and hash
        return f"{base_name}_{url_hash[:8]}"

    def cache_pdf_images(self, pdf_url: str, images: list):
        vol.reload()

        # Create a directory for caching if it doesn't exist
        cache_dir = f"/data/pdf_images/{self.generate_unique_folder_name(pdf_url)}"
        # Check if the directory already exists
        if os.path.exists(cache_dir):
            print(f"Cache directory already exists for {pdf_url}. Skipping image caching.")
            return cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        # Save each image with a name corresponding to its page index
        for index, image in enumerate(images):
            image_path = os.path.join(cache_dir, f"{index}.png")
            image.save(image_path, "PNG")
        vol.commit()