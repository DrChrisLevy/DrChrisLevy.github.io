# ruff: noqa: E501
import modal

# TODO: Just using Default Settings for now.
#   For example, FlashInfer is not available. Falling back to the PyTorch-native implementation of top-p & top-k sampling. For the best performance, please install FlashInfer.
vllm_image = modal.Image.debian_slim(python_version="3.12").pip_install("vllm")

app = modal.App("vllm-openai-compatible")


MINUTES = 60  # seconds
VLLM_PORT = 8000
MAX_INPUTS = 100  # how many requests can one replica can handle - tune carefully!
STARTUP_TIMEOUT = 60 * MINUTES
TIMEOUT = 10 * MINUTES
hf_hub_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")

FUNC_ARGS = dict(
    image=vllm_image,
    gpu="A100-80GB",
    max_containers=1,
    scaledown_window=15 * MINUTES,
    timeout=TIMEOUT,
    volumes={
        "/root/.cache/huggingface/hub/": hf_hub_cache,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    secrets=[hf_secret],
)


@app.function(**FUNC_ARGS)
@modal.concurrent(max_inputs=MAX_INPUTS)
@modal.web_server(port=VLLM_PORT, startup_timeout=STARTUP_TIMEOUT)
def qwen2_5_7b_instruct():
    print("Starting VLLM server")
    import subprocess

    cmd = [
        "vllm",
        "serve",
        "--uvicorn-log-level=info",
        "--enable-prefix-caching",
        "Qwen/Qwen2.5-7B-Instruct",
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
    ]
    subprocess.Popen(" ".join(cmd), shell=True)


@app.function(**FUNC_ARGS)
@modal.concurrent(max_inputs=MAX_INPUTS)
@modal.web_server(port=VLLM_PORT, startup_timeout=STARTUP_TIMEOUT)
def gemma_3_12b_it():
    print("Starting VLLM server")
    import subprocess

    cmd = [
        "vllm",
        "serve",
        "--uvicorn-log-level=info",
        "--enable-prefix-caching",
        "--dtype bfloat16",
        "google/gemma-3-12b-it",
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
    ]
    subprocess.Popen(" ".join(cmd), shell=True)


h200_args = FUNC_ARGS.copy()
h200_args["gpu"] = "H200"


@app.function(**h200_args)
@modal.concurrent(max_inputs=MAX_INPUTS)
@modal.web_server(port=VLLM_PORT, startup_timeout=STARTUP_TIMEOUT)
def qwen2_5_32b_instruct():
    print("Starting VLLM server")
    import subprocess

    cmd = [
        "vllm",
        "serve",
        "--uvicorn-log-level=info",
        "--enable-prefix-caching",
        "--max_num_batched_tokens 16384",
        "--dtype bfloat16",
        "Qwen/Qwen2.5-32B-Instruct",
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
    ]
    subprocess.Popen(" ".join(cmd), shell=True)


# ---------- Qwen 2.5‑72B‑Instruct ----------
@app.function(**h200_args)
@modal.concurrent(max_inputs=MAX_INPUTS)
@modal.web_server(port=VLLM_PORT, startup_timeout=STARTUP_TIMEOUT)
def qwen2_5_72b_instruct():
    import subprocess

    cmd = [
        "vllm",
        "serve",
        "--uvicorn-log-level=info",
        "--enable-prefix-caching",
        "--quantization",
        "fp8",
        "--gpu-memory-utilization",
        "0.85",
        "--max_num_batched_tokens",
        "8192",  # better for short requests
        # Uncomment next line ONLY if you hit first‑prompt OOMs
        # "--cpu-offload-gb", "8",
        "Qwen/Qwen2.5-72B-Instruct",
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
    ]
    subprocess.Popen(" ".join(cmd), shell=True)
