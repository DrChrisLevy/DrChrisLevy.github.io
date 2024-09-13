def main():
    import requests
    import base64
    import os

    os.makedirs("images", exist_ok=True)
    # Your API endpoint URL
    API_URL = "https://drchrislevy--black-forest-labs-flux-model-f-dev.modal.run"  # Replace with your actual Modal app URL

    # Sample data
    data = {
        "prompts": [
            "A pristine tropical island paradise with crystal-clear turquoise waters lapping at white sandy shores. Palm trees sway gently in the breeze along the coastline. In the foreground, the words 'Welcome to Modal' are elegantly written in the smooth wet sand, with small seashells decorating the letters. The sun is setting in the background, painting the sky with vibrant hues of orange, pink, and purple. A few scattered clouds reflect the warm sunset colors.",
        ],
        "fnames": [
            "modal_island"
        ],
        "num_inference_steps": 4,
        "guidance_scale": 5,
    }

    # Make the API request
    response = requests.post(API_URL, json=data)

    if response.status_code == 200:
        results = response.json()

        for result in results:
            filename = result["filename"]
            img_data = base64.b64decode(result["image"])

            with open(os.path.join("images", filename), "wb") as f:
                f.write(img_data)
            print(f"Saved: {filename}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

    print("All images have been downloaded to the 'images/' folder.")
