import io
import json
import os

import modal
from dotenv import load_dotenv

load_dotenv()

S3_BUCKET = os.environ["S3_BUCKET"]
S3_PREFIX = os.environ["S3_PREFIX"]
stub = modal.Stub("video-transcription")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .run_commands("apt-get update", "apt update && apt install ffmpeg -y")
    .pip_install(
        "openai-whisper",
        "pytube",
        "boto3",
        "python-dotenv",
    )
)


def upload_file(filename, s3_filename, bucket_name):
    import boto3

    client = boto3.client("s3")
    headers = {"ACL": "public-read"}
    headers["CacheControl"] = "max-age %d" % (3600 * 24 * 365)
    client.upload_file(filename, bucket_name, s3_filename, ExtraArgs=headers)
    return f"https://{bucket_name}.s3.amazonaws.com/{s3_filename}"


def check_file(filename, bucket_name):
    import boto3
    from botocore.errorfactory import ClientError

    client = boto3.client("s3")
    file_exists = True
    try:
        client.head_object(Bucket=bucket_name, Key=filename)
    except ClientError:
        file_exists = False
    return file_exists


def upload_fileobj(file_object, s3_filename, bucket_name):
    import boto3

    client = boto3.client("s3")
    headers = {"ACL": "public-read"}
    headers["CacheControl"] = "max-age %d" % (3600 * 24 * 365)
    client.upload_fileobj(file_object, bucket_name, s3_filename, ExtraArgs=headers)
    return f"https://{bucket_name}.s3.amazonaws.com/{s3_filename}"


def dict_to_s3(record, s3_filename, bucket_name):
    in_mem_file = io.BytesIO()
    in_mem_file.write(json.dumps(record, sort_keys=True, indent=4).encode())
    in_mem_file.seek(0)
    upload_fileobj(in_mem_file, s3_filename, bucket_name)
    return f"https://{bucket_name}.s3.amazonaws.com/{s3_filename}"


@stub.function(
    image=image,
    secrets=[modal.Secret.from_dotenv()],
    cpu=2,
    memory=1024 * 3,
    gpu="A10G",
    timeout=600,
)
def process_video(url):
    import re

    import whisper
    from pytube import YouTube

    video_id = re.search(r"v=([a-zA-Z0-9_-]{11})", url).group(1)
    file_name_audio = f"{video_id}.mp4"
    s3_file_name_audio = os.path.join(S3_PREFIX, "audio_files", file_name_audio)
    s3_file_name_transcript = os.path.join(S3_PREFIX, "transcripts", f"{video_id}.json")
    if check_file(s3_file_name_transcript, S3_BUCKET):
        print(f"Already processed {s3_file_name_audio}. Skipping.")
        return

    yt = YouTube(url)
    audio_stream = yt.streams.get_audio_only()
    audio_stream.download(filename=file_name_audio)
    upload_file(file_name_audio, s3_file_name_audio, S3_BUCKET)

    # transcribe video
    model = whisper.load_model("small", device="cuda")
    audio = whisper.load_audio(file_name_audio)
    # audio = whisper.pad_or_trim(audio)  # useful for debugging
    result = model.transcribe(audio, fp16=True)
    # for debugging in ipython shell
    # modal.interact()
    # import IPython
    # IPython.embed()
    return dict_to_s3(result, s3_file_name_transcript, S3_BUCKET)


@stub.local_entrypoint()
def main():
    import random

    from pytube import Playlist

    all_urls = set()
    for playlist_url in [
        "https://www.youtube.com/playlist?list=PL8xK8kBHHUX4NW8GqUsyFhBF_xCnzIdPe",
        "https://www.youtube.com/playlist?list=PL8xK8kBHHUX7VsJPqv6OYp71Qj24zcTIr",
        "https://www.youtube.com/playlist?list=PL8xK8kBHHUX5X-jGZlltoZOpv5sKXeGVV",
    ]:
        p = Playlist(playlist_url)
        for url in p.video_urls:
            all_urls.add(url)

    urls = random.sample(list(all_urls), 20)
    for msg in process_video.map(urls):
        print(msg)
