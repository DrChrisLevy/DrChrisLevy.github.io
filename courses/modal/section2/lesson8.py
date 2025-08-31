import json
import os
import shutil
import subprocess
import tempfile
from typing import Dict, List, Tuple

import modal

hf_hub_cache = modal.Volume.from_name("hf_hub_cache", create_if_missing=True)
audio_transcripts = modal.Volume.from_name("audio_transcripts", create_if_missing=True)
app = modal.App(name="transcribe-audio")


image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("torch")
    .run_commands('pip install -U nemo_toolkit["asr"]')
    .apt_install("ffmpeg")
    .pip_install("cuda-python>=12.3")
    .add_local_file("audio.wav", "/audio.wav")
    .add_local_file("courses/modal/section7/lesson22_part3.m4a", "/lesson22_part3.m4a")
)

# ---------------------------------------------------------------------------
# ffmpeg helpers
# ---------------------------------------------------------------------------


def _run(cmd: List[str]) -> str:
    proc = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if proc.returncode:
        raise RuntimeError("Command failed: {}\n{}".format(" ".join(cmd), proc.stderr))
    return proc.stdout


def _probe_audio(path: str) -> Tuple[int, int, int]:
    out = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,channels,bits_per_raw_sample",
            "-of",
            "json",
            path,
        ]
    )
    s = json.loads(out)["streams"][0]
    sr = int(s.get("sample_rate", 16000))
    ch = int(s.get("channels", 1))
    bits = int(s.get("bits_per_raw_sample") or 16)
    return sr, ch, bits


def _convert_to_wav(src: str, dst: str, target_sr: int = 16000) -> None:
    sr, ch, bits = _probe_audio(src)
    if sr == target_sr and ch == 1 and bits == 16 and src.lower().endswith(".wav"):
        shutil.copy(src, dst)
        return
    print(f"[FFMPEG] Converting → 16 kHz mono WAV: {os.path.basename(src)}")
    _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            src,
            "-ac",
            "1",
            "-ar",
            str(target_sr),
            "-sample_fmt",
            "s16",
            dst,
        ]
    )


def _duration(path: str) -> float:
    return float(
        _run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                path,
            ]
        ).strip()
    )


def _slice_audio(
    src: str, chunk: float, overlap: float, out_dir: str
) -> List[Tuple[str, float]]:
    dur = _duration(src)
    offs = 0.0
    idx = 0
    out = []
    while offs < dur:
        path = os.path.join(out_dir, f"chunk_{idx:04d}.wav")
        _run(
            [
                "ffmpeg",
                "-y",
                "-i",
                src,
                "-ss",
                f"{offs}",
                "-t",
                f"{chunk + overlap}",
                "-c",
                "copy",
                path,
            ]
        )
        out.append((path, offs))
        offs += chunk - overlap
        idx += 1
    return out


# ---------------------------------------------------------------------------
# Simple concat merge  of chunks
# ---------------------------------------------------------------------------


def _concat_results(chunks: List[Tuple[Dict, float]]) -> Dict:
    out = {"transcript": "", "word_timestamps": [], "segment_timestamps": []}
    for idx, (res, offset) in enumerate(chunks):
        # text
        if idx:
            out["transcript"] += " "
        out["transcript"] += res["transcript"].lstrip() if idx else res["transcript"]
        # timestamps (shift by chunk start)
        out["word_timestamps"].extend(
            {"word": w["word"], "start": w["start"] + offset, "end": w["end"] + offset}
            for w in res["word_timestamps"]
        )
        out["segment_timestamps"].extend(
            {
                "segment": s["segment"],
                "start": s["start"] + offset,
                "end": s["end"] + offset,
            }
            for s in res["segment_timestamps"]
        )
    return out


# ---------------------------------------------------------------------------
# Public entry point (Modal‑decorated)
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    volumes={
        "/root/.cache/huggingface/hub/": hf_hub_cache,
        "/audio_transcripts": audio_transcripts,
    },
    gpu="A10G",
)
def transcribe_audio(
    input_path: str,
    chunk_len_sec: float = 600.0,
    overlap_sec: float = 0.5,
    keep_chunks: bool = False,
    force_recompute: bool = False,
) -> Dict:
    """Convert *input_path* to standard WAV, chunk with overlap, run ASR, merge.
    Returns dict without character‑offsets (only seconds)."""
    file_name = os.path.basename(input_path).split(".")[0]
    if os.path.exists(f"/audio_transcripts/{file_name}.json") and not force_recompute:
        print("Skipping transcription because file already exists")
        return json.load(open(f"/audio_transcripts/{file_name}.json"))
    import nemo.collections.asr as nemo_asr

    asr_model = nemo_asr.models.ASRModel.from_pretrained(
        model_name="nvidia/parakeet-tdt-0.6b-v2"
    )

    get_timestamp = lambda s: f"{int(s // 60)}:{int(s % 60):02}"

    tmp = tempfile.mkdtemp(prefix=f"asr_tmp_{file_name}_", dir="/audio_transcripts")
    try:
        wav = os.path.join(tmp, "input.wav")
        _convert_to_wav(input_path, wav)
        chunks = _slice_audio(wav, chunk_len_sec, overlap_sec, tmp)

        results: List[Tuple[Dict, float]] = []
        for p, offs in chunks:
            print(f"[ASR] {os.path.basename(p)} @ {offs:.2f}s")
            out = asr_model.transcribe([p], timestamps=True)[0]
            # strip offsets right away b/c I don't need them
            words = [
                {
                    "word": d["word"],
                    "start": d["start"],
                    "end": d["end"],
                }
                for d in out.timestamp["word"]
            ]
            segs = [
                {
                    "segment": d["segment"],
                    "start": d["start"],
                    "end": d["end"],
                }
                for d in out.timestamp["segment"]
            ]
            results.append(
                (
                    {
                        "transcript": out.text,
                        "word_timestamps": words,
                        "segment_timestamps": segs,
                    },
                    offs,
                )
            )

        combined_results = _concat_results(results)
        for word in combined_results["word_timestamps"]:
            word["timestamp_start"] = get_timestamp(word["start"])
            word["timestamp_end"] = get_timestamp(word["end"])
        for segment in combined_results["segment_timestamps"]:
            segment["timestamp_start"] = get_timestamp(segment["start"])
            segment["timestamp_end"] = get_timestamp(segment["end"])

        # write the combined_results to JSON
        with open(f"/audio_transcripts/{file_name}.json", "w") as f:
            json.dump(combined_results, f, indent=4)
        return combined_results

    finally:
        if not keep_chunks:
            shutil.rmtree(tmp, ignore_errors=True)


@app.local_entrypoint()
def main():
    chunk_len_sec = 600.0
    overlap_sec = 0.5
    keep_chunks = False
    force_recompute = False
    # list out the audio files here to transcribe.
    # They must be on the container image.
    file_paths = [
        "/lesson22_part3.m4a",
    ]
    results = [
        r
        for r in transcribe_audio.starmap(
            [
                (file_path, chunk_len_sec, overlap_sec, keep_chunks, force_recompute)
                for file_path in file_paths
            ]
        )
    ]
    for r, file_path in zip(results, file_paths):
        print(f"Transcript for {file_path}")
        print(r["transcript"][:100])
        print(r["word_timestamps"][:10])
        print(r["segment_timestamps"][:10])
