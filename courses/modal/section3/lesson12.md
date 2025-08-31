# Deploying the Trained Encoder Model to a Modal Web Endpoint

In the previous lessons we fine-tuned some transformer encoder models
on an emotion classification task. Here we will look at some inference code
and how we can deploy it to a Modal web endpoint.

Here is a list of texts we can demo the model with:

```python
texts = [
    "I just got a promotion at work!",
    "She broke up with me last night.",
    "The way he smiled at me melted my heart.",
    "I can't believe they lied to me again.",
    "I'm terrified of what might happen tomorrow.",
    "Wow, I didn't see that coming at all!",
    "This is the happiest day of my life.",
    "He left without saying goodbye.",
    "Every moment with you feels magical.",
    "I can't stop shaking—what if they don't make it?",
    "That was the most beautiful proposal ever.",
    "Why does everything always go wrong for me?",
    "I never expected to win the lottery!",
    "Being near you makes everything better.",
    "I feel like I'm drowning in anxiety.",
    "We had such a good time at the beach today!",
    "They said they never wanted to see me again.",
    "You mean the world to me.",
    "I hate that I trusted you.",
    "I heard a noise and thought someone was in the house.",
    "My heart is racing—I can't believe it worked!",
    "She passed away last night. I'm devastated.",
    "He brought me flowers for no reason.",
    "They ignored everything I said. Again.",
    "I don't know how to deal with this fear.",
    "You make me feel so alive.",
    "Why would they do something so cruel?",
    "I'm worried I'll mess everything up.",
    "That was the best surprise party ever!",
    "Everything reminds me of them. It's unbearable.",
    "We danced under the stars all night.",
    "I'm so mad I could scream.",
    "They haven't called, and I'm scared something's wrong.",
    "He told me he loved me for the first time.",
    "I've never felt so alone in my life.",
    "The puppy ran to me wagging its tail.",
    "I'm sick of being treated like this.",
    "Just thinking about it makes me panic.",
    "I still remember our first kiss.",
    "They betrayed me, and I'll never forget it.",
    "This roller coaster is insane—I wasn't ready!",
    "You are the light of my life.",
    "Why does everything good come to an end?",
    "I could barely breathe when I heard the news.",
    "Seeing them again filled my heart with joy.",
    "That scream in the dark scared me so much.",
    "He wrote me the sweetest note.",
    "I'm shaking with rage right now.",
    "I'm so lucky to have a friend like you.",
    "I never expected them to remember my birthday.",
]
```


Here is the code you can put in a file: `encoder_inference.py`.
Make sure to change the `check_point` variable to the path to the checkpoint you want to use.

```python
import time
from typing import List

import modal

# Modal Image
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch",
    "transformers",
    "accelerate",
)

app = modal.App("hf-inference-encoder-classifier")

trainer_vol = modal.Volume.from_name("trainer-vol", create_if_missing=True)


@app.cls(
    image=image,
    volumes={"/data": trainer_vol},
    timeout=60 * 5,
    scaledown_window=30,
    cpu=8,
    memory=3000,
)
class ModelInference:
    @modal.enter()
    def setup(self):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        ct = time.time()
        check_point = "/data/demo4321-dair-ai/emotion-None-answerdotai/ModernBERT-base-batch_size=128-learning_rate=5e-05-num_train_epochs=2/checkpoint-250/"
        self.tokenizer_max_length = 512
        self.model = AutoModelForSequenceClassification.from_pretrained(
            check_point,
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(check_point)
        self.device = self.model.device
        print("Running on device: ", self.device)
        print(f"Time taken to load model: {time.time() - ct} seconds")

    @modal.fastapi_endpoint(
        method="POST",
        docs=True,  # adds interactive documentation in the browser
    )
    def predict(self, texts: List[str]):
        ct = time.time()
        import torch

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.tokenizer_max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            output = self.model(**inputs)
            probs = torch.softmax(output.logits, dim=-1)
            probs = probs.float().to("cpu")

        # Convert tensor to list of dictionaries with emotion labels
        results = []
        for prob_row in probs:
            emotion_scores = {}
            for label_id, prob_value in enumerate(prob_row):
                emotion_label = self.model.config.id2label[label_id]
                emotion_scores[emotion_label] = round(float(prob_value), 2)
            results.append(emotion_scores)

        print(f"Time taken to predict: {time.time() - ct} seconds")
        return results
```

The new thing we are adding here is the `@modal.fastapi_endpoint` decorator.
This is an easy way to turn your Modal functions into a FastAPI web endpoint.
You can read more about it in the [documentation](https://modal.com/docs/examples/basic_web).

```
modal serve encoder_inference.py
```

By running the endpoint with `modal serve`, you create a temporary endpoint that will disappear if you interrupt your terminal. These temporary endpoints are great for debugging — when you save a change to any of your dependent files, the endpoint will redeploy. The path for the endpoint will be printed in the terminal. It may look something like this:

```
https://drchrislevy--hf-inference-encoder-classifier-modelin-73da09-dev.modal.run/
```

And you can visit this URL with `/docs` appended to the end to see the interactive documentation.

```
https://drchrislevy--hf-inference-encoder-classifier-modelin-73da09-dev.modal.run/docs
```

You can call the endpoint with a POST request:

```
curl -X 'POST' \
  'https://drchrislevy--hf-inference-encoder-classifier-modelin-73da09-dev.modal.run/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '[
  "Hey I am happy", "I am feeling sad"
]'
```

Or you can call it with some simple Python code:

```python
import requests
texts = ["Hey I am happy", "I am feeling sad"]
res = requests.post(
    "https://drchrislevy--hf-inference-encoder-classifier-modelin-73da09-dev.modal.run/",
    json=texts,
)
print(res.json())
``` 

If you want to deploy the endpoint as a permanent endpoint, you can do so with:

```
modal deploy encoder_inference.py
```

Remember that you can debug code within `ipython` on a Modal container with `uv run modal shell encoder_inference.py`.