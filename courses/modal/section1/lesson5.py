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
