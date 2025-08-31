import modal


def test_audio_transcription():
    transcribe_audio = modal.Function.from_name("transcribe-audio", "transcribe_audio")
    res = transcribe_audio.remote("/lesson1.m4a", force_recompute=True)
    assert list(res.keys()) == ["transcript", "word_timestamps", "segment_timestamps"]
    assert (
        res["transcript"][:94]
        == "Hey, in this video, we're going to look at the website for Modal, and we're also going to show"
    )
    assert (
        res["transcript"][-100:]
        == "with it. And then in the next video, we'll start looking at running our first example. Okay, see ya."
    )
    assert res["word_timestamps"][:4] == [
        {
            "word": "Hey,",
            "start": 0.88,
            "end": 1.2,
            "timestamp_start": "0:00",
            "timestamp_end": "0:01",
        },
        {
            "word": "in",
            "start": 1.36,
            "end": 1.52,
            "timestamp_start": "0:01",
            "timestamp_end": "0:01",
        },
        {
            "word": "this",
            "start": 1.52,
            "end": 1.6,
            "timestamp_start": "0:01",
            "timestamp_end": "0:01",
        },
        {
            "word": "video,",
            "start": 1.6,
            "end": 1.92,
            "timestamp_start": "0:01",
            "timestamp_end": "0:01",
        },
    ]
    assert res["segment_timestamps"][:4] == [
        {
            "segment": "Hey, in this video, we're going to look at the website for Modal, and we're also going to show how to install it and get you up and running.",
            "start": 0.88,
            "end": 8.72,
            "timestamp_start": "0:00",
            "timestamp_end": "0:08",
        },
        {
            "segment": "That's the purpose of this lesson.",
            "start": 9.040000000000001,
            "end": 10.8,
            "timestamp_start": "0:09",
            "timestamp_end": "0:10",
        },
        {
            "segment": "Modal is a really cool infrastructure platform that's serverless, meaning you can scale out your code and all your application logic onto like hundreds of containers without worrying about DevOps and infrastructure and all that stuff.",
            "start": 11.36,
            "end": 25.04,
            "timestamp_start": "0:11",
            "timestamp_end": "0:25",
        },
        {
            "segment": "You just add, you know, little decorators to your functions right in your Python code.",
            "start": 25.28,
            "end": 29.76,
            "timestamp_start": "0:25",
            "timestamp_end": "0:29",
        },
    ]
