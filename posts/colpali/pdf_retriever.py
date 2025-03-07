import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import List, cast

import modal
from dotenv import load_dotenv
from modal import build, enter

from utils import generate_unique_folder_name, log_to_queue

load_dotenv()
app = modal.App("pdf-retriever")

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

vol = modal.Volume.from_name("pdf-retriever-volume", create_if_missing=True)


@app.cls(image=image, secrets=[modal.Secret.from_dotenv()], volumes={"/data": vol}, gpu="a10g", cpu=4, timeout=600, container_idle_timeout=60 * 2)
class PDFRetriever:
    @build()
    @enter()
    def setup(self):
        log_to_queue("Starting PDF Retriever Container and Loading Model . . .")
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
        log_to_queue("Finished Creation of PDF Retriever Container")

    @modal.method()
    def forward(self, inputs):
        import torch
        from colpali_engine.utils.torch_utils import ListDataset
        from torch.utils.data import DataLoader

        if type(inputs[0]) == str:
            process_fn = self.processor.process_queries
        else:
            process_fn = self.processor.process_images
        batch_size = 8
        dataloader = DataLoader(
            dataset=ListDataset[str](inputs),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda x: process_fn(x),
        )
        ds: List[torch.Tensor] = []
        total_batches = len(dataloader)
        for batch_idx, batch_doc in enumerate(dataloader):
            with torch.no_grad():
                batch_doc = {k: v.to(self.model.device) for k, v in batch_doc.items()}
                embeddings_doc = self.model(**batch_doc)
            ds.extend(list(torch.unbind(embeddings_doc.to("cpu"))))

            # Log the number of pages left
            pages_left = (total_batches - batch_idx - 1) * batch_size
            log_to_queue(f"Processed batch {batch_idx + 1}/{total_batches}. Approximately {pages_left} pages left.")

        return ds

    @modal.method()
    def top_pages(self, pdf_url: str, queries: list[str], top_k=1, use_cache=True):
        import numpy as np

        # Run inference - PDF pages
        ds = self.load_embeddings(pdf_url) if use_cache else None
        if ds is None:
            log_to_queue(f"New PDF Source Detected: {pdf_url}")
            log_to_queue("Processing will take longer this first time but subsequent calls will use cached results.")
            log_to_queue("Generating Images for PDF . . .")
            images = self.pdf_to_images(pdf_url)
            log_to_queue("Images Generated for PDF")
            log_to_queue("Generating Embeddings for PDF . . .")
            ds = self.forward.local(images)
            self.cache_embeddings(pdf_url, ds)
            log_to_queue("Embeddings Generated for PDF and cached")

        log_to_queue("Generating Query Embeddings and Ranking Pages")
        # Run inference - queries
        qs = self.forward.local(queries)

        # Run scoring
        scores = self.processor.score(qs, ds).cpu().numpy()
        res = np.argsort(scores, axis=-1)[:, -top_k:][:, ::-1].tolist()
        # The top k indices for each query
        log_to_queue(f"Finished Ranking Pages and Found Top Pages: {res}")
        return res

    def pdf_to_images(self, pdf_url):
        import requests
        from pdf2image import convert_from_bytes, pdfinfo_from_bytes

        log_to_queue(f"Downloading PDF from {pdf_url}")
        response = requests.get(pdf_url)
        if response.status_code != 200:
            raise Exception(f"Failed to download PDF from {pdf_url}")

        pdf_bytes = BytesIO(response.content)
        pdf_info = pdfinfo_from_bytes(pdf_bytes.getvalue())
        num_pages = pdf_info["Pages"]
        log_to_queue(f"PDF downloaded successfully. Total pages: {num_pages}")

        def process_page(page_num):
            page_images = convert_from_bytes(pdf_bytes.getvalue(), first_page=page_num, last_page=page_num)
            log_to_queue(f"Processed page {page_num}/{num_pages}")
            return page_images[0]

        log_to_queue("Starting page processing...")
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(process_page, page_num) for page_num in range(1, num_pages + 1)]
            images = []
            for future in futures:
                images.append(future.result())
                completed = len(images)
                log_to_queue(f"Progress: {completed}/{num_pages} pages processed")

        log_to_queue("All pages processed. Caching PDF images...")
        self.cache_pdf_images(pdf_url, images)
        log_to_queue("PDF images cached successfully")
        return images

    def cache_pdf_images(self, pdf_url: str, images: list):
        self.cache_data(pdf_url, images, "pdf_images")

    def cache_embeddings(self, pdf_url: str, embeddings):
        self.cache_data(pdf_url, embeddings, "embeddings")

    def cache_data(self, pdf_url: str, data, data_type: str):
        vol.reload()
        cache_dir = generate_unique_folder_name(pdf_url)
        cache_path = os.path.join(f"/data/{data_type}", f"{cache_dir}")

        if os.path.exists(cache_path):
            return

        os.makedirs(cache_path, exist_ok=True)

        save_method = self._save_images if data_type == "pdf_images" else self._save_embeddings
        save_method(data, cache_path)
        vol.commit()

    def _save_images(self, images, cache_dir):
        def save_image(args):
            index, image = args
            image_path = os.path.join(cache_dir, f"{index}.png")
            image.save(image_path, "PNG")

        with ThreadPoolExecutor(max_workers=8) as executor:
            executor.map(save_image, enumerate(images))

    def _save_embeddings(self, embeddings, cache_path):
        with open(os.path.join(cache_path, "embeddings.pkl"), "wb") as f:
            pickle.dump(embeddings, f)

    def load_embeddings(self, pdf_url: str):
        vol.reload()
        cache_dir = generate_unique_folder_name(pdf_url)
        cache_path = os.path.join("/data/embeddings", f"{cache_dir}")

        if os.path.exists(cache_path):
            with open(os.path.join(cache_path, "embeddings.pkl"), "rb") as f:
                return pickle.load(f)
        return None
