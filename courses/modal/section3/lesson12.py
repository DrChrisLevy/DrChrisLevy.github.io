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
