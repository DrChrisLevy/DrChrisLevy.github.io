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
        from typing import cast

        import torch

        from colpali_engine.models import ColPali, ColPaliProcessor

        self.model = cast(
            ColPali,
            ColPali.from_pretrained(
                "vidore/colpali-v1.2",
                torch_dtype=torch.bfloat16,
                device_map="cuda:0",  # or "mps" if on Apple Silicon
            ),
        )

        self.processor = cast(ColPaliProcessor, ColPaliProcessor.from_pretrained("google/paligemma-3b-mix-448"))

    @modal.method()
    def f(self):
        from PIL import Image
        import torch

        os.system("pip install datasets")
        # Your inputs
        images = [
            Image.new("RGB", (32, 32), color="white"),
            Image.new("RGB", (16, 16), color="black"),
        ]
        queries = [
            "Is attention really all you need?",
            "Are Benjamin, Antoine, Merve, and Jo best friends?",
        ]

        # Process the inputs
        batch_images = self.processor.process_images(images).to(self.model.device)
        batch_queries = self.processor.process_queries(queries).to(self.model.device)

        # Forward pass
        with torch.no_grad():
            image_embeddings = self.model(**batch_images)
            query_embeddings = self.model(**batch_queries)

        scores = self.processor.score_multi_vector(query_embeddings, image_embeddings)
        print(batch_images)
        print(batch_queries)
        print(image_embeddings)
        print(query_embeddings)
        print(scores)
        return scores


@app.local_entrypoint()
def main():
    model = Model()
    model.f.remote()
