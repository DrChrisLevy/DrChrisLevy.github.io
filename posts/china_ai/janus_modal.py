import modal
from modal import Image, enter

app = modal.App("deepseek-janus-pro")

image = (
    Image.debian_slim(python_version="3.11")
    .run_commands(
        "apt-get update && apt-get install -y git",
        "git clone https://github.com/deepseek-ai/Janus.git",
        "cd Janus && pip install -e .",
    )
    .env({"HF_HUB_CACHE": "/cache"})
)
cache_vol = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)


@app.cls(image=image, volumes={"/cache": cache_vol}, gpu="A100", cpu=4, timeout=600, container_idle_timeout=300)
class Model:
    @enter()
    def setup(self):
        import torch
        from janus.models import MultiModalityCausalLM, VLChatProcessor
        from transformers import AutoModelForCausalLM

        # specify the path to the model
        model_path = "deepseek-ai/Janus-Pro-7B"
        self.vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
        self.tokenizer = self.vl_chat_processor.tokenizer

        self.vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        self.vl_gpt = self.vl_gpt.to(torch.bfloat16).cuda().eval()

    def url_to_base64(self, url: str) -> str:
        import base64
        from io import BytesIO

        import requests
        from PIL import Image

        # Download the image from URL
        response = requests.get(url)
        # Convert to PIL Image first to ensure it's a valid JPEG
        img = Image.open(BytesIO(response.content))
        # Convert to RGB mode if it's not
        if img.mode != "RGB":
            img = img.convert("RGB")
        # Save as JPEG to BytesIO
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        # Convert to base64
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        # Return in the format expected by Janus
        return f"data:image/jpeg;base64,{img_base64}"

    @modal.web_endpoint(method="POST", docs=True)
    def f(self, data: dict):
        from janus.utils.io import load_pil_images

        question = data["question"]
        image_url = data["image_url"]
        base64_image = self.url_to_base64(image_url)
        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image_placeholder>\n{question}",
                "images": [base64_image],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        # load images and prepare for inputs
        pil_images = load_pil_images(conversation)
        prepare_inputs = self.vl_chat_processor(conversations=conversation, images=pil_images, force_batchify=True).to(self.vl_gpt.device)

        # # run image encoder to get the image embeddings
        inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        # # run the model to get the response
        outputs = self.vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=512 * 8,
            do_sample=False,
            use_cache=True,
        )

        answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        return answer
