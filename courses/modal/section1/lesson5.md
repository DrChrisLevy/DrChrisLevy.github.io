# GPUs and Other Resources


## GPU Acceleration

- Go and read the docs [here](https://modal.com/docs/guide/gpu)
to see how easy it is to use GPUs with Modal!

- For more complex use cases, check out these [docs](https://modal.com/docs/guide/cuda)

- pricing page [here](https://modal.com/pricing)

- create a file `demo_gpu.py` and copy this code into it:

```python
import time

import modal

app = modal.App()
image = modal.Image.debian_slim().pip_install("torch")


@app.function(image=image, gpu="A10G")
def run():
    import torch

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    # Set matrix size
    size = 10000

    # Create tensors on the appropriate device
    tensor1 = torch.randn(size, size, device=device)
    tensor2 = torch.randn(size, size, device=device)

    # Warm up
    _ = torch.matmul(tensor1, tensor2)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Time the matrix multiplication
    start_time = time.time()
    result = torch.matmul(tensor1, tensor2)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_time = time.time() - start_time

    print(
        f"Matrix multiplication result shape: {result.shape} completed in {elapsed_time:.4f} seconds"
    )
```

- Run the function with `modal run demo_gpu.py`

- Shell into a container, `modal shell demo_gpu.py`, and test out some commands to see the GPU info:
- `nvidia-smi` 
- `nvidia-smi -L` 
- `nvidia-smi -l`
- Try out some different GPUs such as (`'A100'`, `'L40S'`, `'L40'`, `'A10G'`, `'H100'`, etc.)


## Reserving CPU and memory

- You can also reserve memory/CPU for your function by using the `memory` and `cpu` arguments.
- Read the docs [here](https://modal.com/docs/guide/resources).

## Test it Out

- play around with the code in `demo_gpu.py` and see how it performs on different GPUs using the `gpu` argument.
- remove the GPU entirely and see how it performs on the CPU.
- try different numbers of CPU cores using the `cpu` argument.