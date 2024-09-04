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
            "A serene mountain landscape at sunset",
            "A futuristic cityscape with flying cars",
            "An underwater scene with colorful coral reefs",
            "A steampunk-inspired clockwork dragon",
            "A bioluminescent forest at midnight",
            "An ancient library filled with floating books",
            "A surreal Salvador Dali-inspired melting cityscape",
            "A cyberpunk street market in neon-lit rain",
            "A whimsical tea party on a giant mushroom",
            "An intergalactic spaceport with alien travelers",
        ],
        "fnames": [
            "mountain_sunset",
            "future_city",
            "underwater_coral",
            "steampunk_dragon",
            "bioluminescent_forest",
            "floating_library",
            "melting_cityscape",
            "cyberpunk_market",
            "mushroom_teaparty",
            "alien_spaceport",
        ],
        "num_inference_steps": 4,
        "guidance_scale": 7,
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
