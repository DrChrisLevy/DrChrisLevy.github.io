from fasthtml.common import *

app, rt = fast_app()


@rt("/")
def get():
    return Titled(
        "Image Generator",
        Form(
            Input(id="prompt", name="prompt", placeholder="Enter your prompt"),
            Button("Generate Image"),
            hx_post="/generate",
            target_id="result",
            hx_swap="beforeend",
        ),
        Div(id="result"),
    )


@rt("/generate")
def post(prompt: str):
    import time
    ct = time.time()
    fname = generate_image(prompt)
    print(f"Time taken for generation: {time.time() - ct}")

    ct = time.time()
    res = Div(P(f"You requested an image for the prompt: {prompt}"), id="result"), Img(
        src=f"images/{fname}.png"
    )
    print(f"Time taken for response: {time.time() - ct}")
    return res


def generate_image(prompt: str):
    import requests
    import base64
    import os

    os.makedirs("images", exist_ok=True)
    # Your API endpoint URL
    API_URL = "https://drchrislevy--black-forest-labs-flux-model-f.modal.run/"  # Replace with your actual Modal app URL

    # use uuid for fname
    from uuid import uuid4
    fname = str(uuid4())
    # Sample data
    data = {
        "prompts": [prompt],
        "fnames": [fname],
        "num_inference_steps": 4,
        "guidance_scale": 5,
    }

    # Make the API request
    response = requests.post(API_URL, json=data)

    if response.status_code == 200:
        results = response.json()

        for result in results:
            img_data = base64.b64decode(result["image"])

            with open(os.path.join("images", f'{fname}.png'), "wb") as f:
                f.write(img_data)
            print(f"Saved: {fname}.png")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

    print("All images have been downloaded to the 'images/' folder.")
    return fname


serve()
