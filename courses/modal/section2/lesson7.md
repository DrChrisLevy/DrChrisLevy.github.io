# Transcribing Audio Files with ASR model: `nvidia/parakeet-tdt-0.6b-v2` (Part 1)

- There are many popular models for transcribing audio files with automatic speech recognition (ASR)
- One of the most popular models is OpenAI's [Whisper](https://github.com/openai/whisper)
- You can find the Whisper models on [Hugging Face](https://huggingface.co/openai?search_models=whisper)
- You can also find other ASR models on [Hugging Face](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition&sort=trending)
- In this lesson we will use a more recent model from Nvidia, `nvidia/parakeet-tdt-0.6b-v2` which can be found [here](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)
- It's a  600-million-parameter model designed for high-quality English transcription. It also provides
accurate word-level timestamp predictions and automatic punctuation and capitalization.


When it comes to running inference on models from Hugging Face, it's usually as simple as
copying and pasting the code from the corresponding model card. However, this
assumes you have access to a GPU and proper environment setup.

This is the beauty of Modal. We can define our own custom container images with the dependencies we need to run inference,
and run it on GPUs. Then we can scale inference to many containers without worrying about the underlying infrastructure.

We will be using the code from this [model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)
as our base code for the inference.

To get started you first need a local `.wav` file with the audio you want to transcribe.
You can find any file off the web. For example you can try using the sample audio file from the model card:

```
wget https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav -O audio.wav
```

Or you can create a simple python script `download_audio.py` to download an audio file:

```python
import wget

url = "https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav"
wget.download(url, 'audio.wav')
```

Then run the script with `uv run --with wget download_audio.py` for example.

Here is the final code we will build up in this lesson from this video.
Put the code in a file called `asr_v1.py` and run it with `modal run asr_v1.py`.

The purpose of this lesson is to get the most simplest example working with
this ASR model. In the next lesson we will add a couple more features
such as chunking, to handle longer audio files.

```python
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
```