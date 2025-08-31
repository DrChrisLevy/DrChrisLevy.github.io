# Setting up Flash Attention and a Container for Fine-Tuning LLMS with Axolotl

Here is a brief snippet of code to show how to setup an environment for a container with Flash Attention, and fine-tuning LLMs with Axolotl.

Create a python file called `flash_attention_image.py` and add the following code below.

Setting up Flash Attention can be tricky.
It is recommended to use an official docker image from Nvidia.
Make sure the `flash_attn_release` link matches the version of torch abd python you are 
using within the container.
Read more about that on Modal documentation [here](https://modal.com/docs/guide/cuda).

The best way to install flash-attn is to use the prebuilt wheels 
from [https://github.com/Dao-AILab/flash-attention/releases](https://github.com/Dao-AILab/flash-attention/releases). 
Be sure to pick the wheel that matches the Python version and torch version.
There is also an example in the Modal examples [here](https://github.com/modal-labs/modal-examples/blob/main/02_building_containers/install_flash_attn.py).

```python
import modal

app = modal.App("example-install-flash-attn")

flash_attn_release = "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.2/flash_attn-2.8.2+cu12torch2.7cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"


cuda_version = "12.8.1"  # should be no greater than host CUDA version
flavor = "devel"  # includes full CUDA toolkit
operating_sys = "ubuntu24.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"
python_version = "3.12"
torch_version = "2.7.0"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python=python_version)
    .pip_install(
        f"torch=={torch_version}",
        flash_attn_release,
    )
    # https://docs.axolotl.ai/docs/installation.html#sec-pypi
    .run_commands("pip3 install -U packaging==23.2 setuptools==75.8.0 wheel ninja")
    .run_commands("pip3 install --no-build-isolation axolotl[flash-attn,deepspeed]")
)


@app.function(gpu="L40S", image=image)
def run_flash_attn():
    import torch
    from flash_attn import flash_attn_func

    # Print CUDA version
    print(f"CUDA version: {torch.version.cuda}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name()}")

    batch_size, seqlen, nheads, headdim, nheads_k = 2, 4, 3, 16, 3

    q = torch.randn(batch_size, seqlen, nheads, headdim, dtype=torch.float16).to("cuda")
    k = torch.randn(batch_size, seqlen, nheads_k, headdim, dtype=torch.float16).to(
        "cuda"
    )
    v = torch.randn(batch_size, seqlen, nheads_k, headdim, dtype=torch.float16).to(
        "cuda"
    )

    out = flash_attn_func(q, k, v)
    assert out.shape == (batch_size, seqlen, nheads, headdim)
    print("Flash Attention test passed successfully!")
```

Test that the container is working and that Flash Attention is installed correctly by running the following command:

```
uv run modal run flash_attention_image.py
```

For playing around with the Axolotl examples, you can run the following commands interactively.

```
uv run modal shell flash_attention_image.py
```

In the container:

```
axolotl fetch examples
```

```
axolotl train examples/llama-3/lora-1b.yml
```


Learn more about Axolotl [here](https://docs.axolotl.ai/).