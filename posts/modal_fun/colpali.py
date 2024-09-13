import modal
from modal import build, enter
import os
from dotenv import load_dotenv

load_dotenv()
app = modal.App("colpali")

cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"
image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install("git", "poppler-utils")
    .pip_install(
        "ninja",  # required to build flash-attn
        "packaging",  # required to build flash-attn
        "wheel",  # required to build flash-attn
    )
    .run_commands(
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124",
        "pip install git+https://github.com/huggingface/transformers",
        "pip install accelerate",
        "pip install git+https://github.com/illuin-tech/colpali.git",  # TODO: maybe pin version once pipy updated released inference bug fix
        "pip install requests pdf2image PyPDF2",
        "pip install python-dotenv",
        f'huggingface-cli login --token {os.environ["HUGGING_FACE_ACCESS_TOKEN"]}',
    )
    .run_commands("pip install flash-attn --no-build-isolation")
)


@app.cls(image=image, secrets=[modal.Secret.from_dotenv()], gpu="a10g", cpu=4, timeout=600, container_idle_timeout=300)
class Model:
    @build()
    @enter()
    def setup(self):
        from typing import cast

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

    def pdf_to_images(self, pdf_url):
        # Function to download and convert PDF url to images
        import requests
        from io import BytesIO
        from pdf2image import convert_from_bytes

        # Step 1: Download the PDF from the provided URL
        response = requests.get(pdf_url)
        if response.status_code != 200:
            raise Exception(f"Failed to download PDF from {pdf_url}")

        # Step 2: Convert the PDF into images (in-memory)
        pdf_bytes = BytesIO(response.content)
        images = convert_from_bytes(pdf_bytes.read())

        # Step 3: Return the list of PIL images
        return images

    @modal.method()
    def f(self, pdf_url: str, queries: list[str]):
        from typing import List

        import torch
        from torch.utils.data import DataLoader
        from tqdm import tqdm

        from colpali_engine.utils.torch_utils import ListDataset

        images = self.pdf_to_images(pdf_url)

        # Run inference - docs
        dataloader = DataLoader(
            dataset=ListDataset[str](images),
            batch_size=4,
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
            batch_size=4,
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
        idx_top_1 = scores.argmax(axis=-1)
        return idx_top_1.tolist()


@app.local_entrypoint()
def main():
    model = Model()
    model.f.remote(
        "https://arxiv.org/pdf/1706.03762",
        ["Who are the authors of the paper?", "What is the formula for self attention?", "What Optimizer was used for training?"],
    )
