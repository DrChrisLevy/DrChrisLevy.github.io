import modal

hf_hub_cache = modal.Volume.from_name("hf_hub_cache", create_if_missing=True)
app = modal.App(name="transcribe-audio")


image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("torch")
    .run_commands('pip install -U nemo_toolkit["asr"]')
    .apt_install("ffmpeg")
    .pip_install("cuda-python>=12.3")
    .add_local_file("audio.wav", "/audio.wav")
)


@app.function(
    image=image,
    volumes={
        "/root/.cache/huggingface/hub/": hf_hub_cache,
    },
    gpu="A10G",
)
def transcribe_audio(file_path: str) -> dict:
    import nemo.collections.asr as nemo_asr

    asr_model = nemo_asr.models.ASRModel.from_pretrained(
        model_name="nvidia/parakeet-tdt-0.6b-v2"
    )
    output = asr_model.transcribe([file_path], timestamps=True)
    # by default, timestamps are enabled for char, word and segment level
    word_timestamps = output[0].timestamp["word"]
    # segment level timestamps
    segment_timestamps = output[0].timestamp["segment"]

    result = {
        "transcript": output[0].text,
        "word_timestamps": word_timestamps,
        "segment_timestamps": segment_timestamps,
    }
    return result


@app.local_entrypoint()
def main():
    file_paths = [
        "/audio.wav",
    ]
    results = [r for r in transcribe_audio.map(file_paths)]
    for r, file_path in zip(results, file_paths):
        print(f"Transcript for {file_path}")
        print(r["transcript"][:100])
        print(r["word_timestamps"][:10])
        print(r["segment_timestamps"][:10])
