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
    .apt_install("git")
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
        "pip install python-dotenv",
        f'huggingface-cli login --token {os.environ["HUGGING_FACE_ACCESS_TOKEN"]}',
    )
    .run_commands("pip install flash-attn --no-build-isolation")
)


@app.cls(image=image, secrets=[modal.Secret.from_dotenv()], gpu="a10g", cpu=4, timeout=600, container_idle_timeout=300)
class Model:

    def pdf_to_images(self, pdf_url):
        pass
    
    @build()
    @enter()
    def setup(self):
        from colpali_engine.models.paligemma_colbert_architecture import ColPali
        import torch
        from transformers import AutoProcessor

        # Load model
        model_name = "vidore/colpali-v1.2"
        model = ColPali.from_pretrained("vidore/colpaligemma-3b-pt-448-base", torch_dtype=torch.bfloat16, device_map="cuda").eval()
        model.load_adapter(model_name)
        self.model = model.eval()
        self.processor = AutoProcessor.from_pretrained(model_name)


    @modal.method()
    def f(self):
        os.system('pip install datasets')
        import torch
        from torch.utils.data import DataLoader
        from tqdm import tqdm

        from PIL import Image

        from colpali_engine.utils.colpali_processing_utils import process_images, process_queries
        from colpali_engine.utils.image_from_page_utils import load_from_dataset

        """Example script to run inference with ColPali"""
        # select images -> load_from_pdf(<pdf_path>),  load_from_image_urls(["<url_1>"]), load_from_dataset(<path>)
        images = load_from_dataset("vidore/docvqa_test_subsampled")
        queries = ["From which university does James V. Fiorca come ?", "Who is the japanese prime minister?"]

        # run inference - docs
        dataloader = DataLoader(
            images,
            batch_size=4,
            shuffle=False,
            collate_fn=lambda x: process_images(self.processor, x),
        )
        ds = []
        for batch_doc in tqdm(dataloader):
            with torch.no_grad():
                batch_doc = {k: v.to(self.model.device) for k, v in batch_doc.items()}
                embeddings_doc = self.model(**batch_doc)
            ds.extend(list(torch.unbind(embeddings_doc.to("cpu"))))

        modal.interact()
        import IPython
        IPython.embed()

        # run inference - queries
        dataloader = DataLoader(
            queries,
            batch_size=4,
            shuffle=False,
            collate_fn=lambda x: process_queries(self.processor, x, Image.new("RGB", (448, 448), (255, 255, 255))),
        )

        qs = []
        for batch_query in dataloader:
            with torch.no_grad():
                batch_query = {k: v.to(self.model.device) for k, v in batch_query.items()}
                embeddings_query = self.model(**batch_query)
            qs.extend(list(torch.unbind(embeddings_query.to("cpu"))))

        print(qs)
        print(ds)


@app.local_entrypoint()
def main():
    model = Model()
    model.f.remote()
