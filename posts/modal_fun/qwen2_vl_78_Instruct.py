import os

import modal
from dotenv import load_dotenv
from modal import build, enter

load_dotenv()
app = modal.App("qwen2_vl_78_Instruct")

cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"
image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install("git")
    .pip_install(
        "ninja",  # required to build flash-attn
        "packaging",  # required to build flash-attn
        "wheel",  # required to build flash-attn
    )
    .run_commands(
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124",
        "pip install git+https://github.com/huggingface/transformers",
        "pip install accelerate",
        "pip install qwen-vl-utils",
        "pip install python-dotenv",
        f'huggingface-cli login --token {os.environ["HUGGING_FACE_ACCESS_TOKEN"]}',
    )
    .run_commands("pip install flash-attn --no-build-isolation")
)


@app.cls(image=image, secrets=[modal.Secret.from_dotenv()], gpu="a100", cpu=4, timeout=600, container_idle_timeout=300)
class Model:
    @build()
    @enter()
    def setup(self):
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, TextStreamer

        # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            vision_config={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
        # default processor
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

        # The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
        # min_pixels = 256*28*28
        # max_pixels = 1280*28*28
        # processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

        self.streamer = TextStreamer(self.processor, skip_prompt=True, skip_special_tokens=True)

    @modal.method()
    def f(self, messages_list, max_new_tokens=256, show_stream=False):
        from qwen_vl_utils import process_vision_info

        def messages_inference(messages):
            # Preparation for inference
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            # Inference: Generation of the output
            if show_stream:
                print("\n\n-----------------------------------------------------------\n\n")
                generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, streamer=self.streamer)
            else:
                generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            return output_text

        return [messages_inference(messages) for messages in messages_list]


@app.local_entrypoint()
def main():
    s3_bucket = os.environ["S3_BUCKET"]  # where my images are hosted
    s3_prefix = os.environ["S3_PREFIX"]  # where my images are hosted
    model = Model()

    messages_list = [
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"https://{s3_bucket}.s3.amazonaws.com{s3_prefix}tropical_island_paradise_guidance_scale_3.5_num_inference_steps_50_seed_5013302010029533033_model_black-forest-labs_FLUX.1-dev.png",
                    },
                    {"type": "text", "text": "Give a short description of this image."},
                ],
            }
        ],
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"https://{s3_bucket}.s3.amazonaws.com{s3_prefix}sci_fi_forest_ship_guidance_scale_3.5_num_inference_steps_50_seed_6510529542810937186_model_black-forest-labs_FLUX.1-dev.png",
                    },
                    {"type": "text", "text": "Give a long detailed description of this image."},
                ],
            }
        ],
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"https://{s3_bucket}.s3.amazonaws.com{s3_prefix}tropical_island_paradise_guidance_scale_3.5_num_inference_steps_50_seed_5013302010029533033_model_black-forest-labs_FLUX.1-dev.png",
                    },
                    {
                        "type": "image",
                        "image": f"https://{s3_bucket}.s3.amazonaws.com{s3_prefix}sci_fi_forest_ship_guidance_scale_3.5_num_inference_steps_50_seed_6510529542810937186_model_black-forest-labs_FLUX.1-dev.png",
                    },
                    {"type": "text", "text": "For each image explain whether you think it is real or AI generated. Explain your reasoning."},
                ],
            }
        ],
    ]

    print("Testing Single Inference with show_stream=True")
    print(model.f.remote(messages_list[1:2], max_new_tokens=1000, show_stream=True))

    print("Testing Multiple Container with show_stream=False")
    for res in model.f.starmap([(messages_list[:1], 1000, False), (messages_list[1:2], 1000, False), (messages_list[2:3], 1000, False)]):
        print(res)
