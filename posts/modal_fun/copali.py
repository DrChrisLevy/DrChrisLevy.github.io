import modal
from modal import build, enter
import os
from dotenv import load_dotenv

load_dotenv()
app = modal.App("copali")

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
        "pip install colpali-engine",
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
        import torch
        from typing import cast

        from colpali_engine.models import ColPali, ColPaliProcessor
        from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor

        model_name = "vidore/colpali-v1.2"
        self.model = ColPali.from_pretrained("vidore/colpaligemma-3b-pt-448-base", torch_dtype=torch.bfloat16, device_map="cuda").eval()
        self.model.load_adapter(model_name)
        self.processor = cast(ColPaliProcessor, ColPaliProcessor.from_pretrained("google/paligemma-3b-mix-448"))  # TODO: took out of readme.
        if not isinstance(self.processor, BaseVisualRetrieverProcessor):
            raise ValueError("Processor should be a BaseVisualRetrieverProcessor")

    @modal.method()
    def f(self):
        import os

        os.system("pip install datasets")
        from typing import cast

        import torch
        from datasets import Dataset, load_dataset
        from torch.utils.data import DataLoader
        from tqdm import tqdm

        images = cast(Dataset, load_dataset("vidore/docvqa_test_subsampled", split="test"))["image"]
        queries = [
            "From which university does James V. Fiorca come ?",
            "Who is the japanese prime minister?",
            "What is the Log-in No. ?",
            "Which is the railways company?",
            "What is the % Promoted Volume in EDLP stores?",
        ]

        # run inference - docs
        dataloader = DataLoader(
            images,
            batch_size=4,
            shuffle=False,
            collate_fn=lambda x: self.processor.process_images(x),
        )
        ds = []
        for batch_doc in tqdm(dataloader):
            with torch.no_grad():
                batch_doc = {k: v.to(self.model.device) for k, v in batch_doc.items()}
                embeddings_doc = self.model(**batch_doc)
            ds.extend(list(torch.unbind(embeddings_doc.to("cpu"))))

        # run inference - queries
        dataloader = DataLoader(
            queries,
            batch_size=4,
            shuffle=False,
            collate_fn=lambda x: self.processor.process_queries(x),
        )

        qs = []
        for batch_query in dataloader:
            with torch.no_grad():
                batch_query = {k: v.to(self.model.device) for k, v in batch_query.items()}
                embeddings_query = self.model(**batch_query)
            qs.extend(list(torch.unbind(embeddings_query.to("cpu"))))

        # run evaluation
        scores = self.processor.score(qs, ds)
        print(scores.argmax(axis=1))
        return scores


@app.local_entrypoint()
def main():
    model = Model()
    model.f.remote()
