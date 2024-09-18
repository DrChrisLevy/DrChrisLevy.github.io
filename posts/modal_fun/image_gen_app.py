from fastcore.parallel import threaded
from fasthtml.common import *
import uuid
import os
import requests


# gens database for storing generated image details
tables = database("data/gens.db").t
gens = tables.gens
if gens not in tables:
    gens.create(prompt=str, id=int, folder=str, pk="id")
Generation = gens.dataclass()


# Flexbox CSS (http://flexboxgrid.com/)
gridlink = Link(rel="stylesheet", href="https://cdnjs.cloudflare.com/ajax/libs/flexboxgrid/6.3.1/flexboxgrid.min.css", type="text/css")

# Our FastHTML app
app = FastHTML(hdrs=(picolink, gridlink))


# Main page
@app.get("/")
def home():
    inp = Input(id="new-prompt", name="prompt", placeholder="Enter a prompt")
    add = Form(Group(inp, Button("Generate")), hx_post="/", target_id="gen-list", hx_swap="afterbegin")
    gen_containers = [generation_preview(g) for g in gens(limit=10)]  # Start with last 10
    gen_list = Div(*reversed(gen_containers), id="gen-list", cls="row")  # flexbox container: class = row
    return Title("Image Generation Demo"), Main(H1("Image Generation"), add, gen_list, cls="container")


# Show the image (if available) and prompt for a generation
def generation_preview(g):
    grid_cls = "box col-xs-12 col-sm-6 col-md-4 col-lg-3"
    image_path = f"{g.folder}/{g.id}.png"
    if os.path.exists(image_path):
        return Div(
            Card(
                Img(src=image_path, alt="Card image", cls="card-img-top"),
                Div(P(B("Prompt: "), g.prompt, cls="card-text"), cls="card-body"),
            ),
            id=f"gen-{g.id}",
            cls=grid_cls,
        )
    return Div(
        f"Generating gen {g.id} with prompt {g.prompt}",
        id=f"gen-{g.id}",
        hx_get=f"/gens/{g.id}",
        hx_trigger="every 2s",
        hx_swap="outerHTML",
        cls=grid_cls,
    )


# A pending preview keeps polling this route until we return the image preview
@app.get("/gens/{id}")
def preview(id: int):
    return generation_preview(gens.get(id))


# For images, CSS, etc.
@app.get("/{fname:path}.{ext:static}")
def static(fname: str, ext: str):
    return FileResponse(f"{fname}.{ext}")


# Generation route
@app.post("/")
def post(prompt: str):
    folder = f"data/gens/{str(uuid.uuid4())}"
    os.makedirs(folder, exist_ok=True)
    g = gens.insert(Generation(prompt=prompt, folder=folder))
    generate_and_save(g.prompt, g.id, g.folder)
    clear_input = Input(id="new-prompt", name="prompt", placeholder="Enter a prompt", hx_swap_oob="true")
    return generation_preview(g), clear_input


# Generate an image and save it to the folder (in a separate thread)
@threaded
def generate_and_save(prompt, id, folder):
    import base64
    import os

    # Your API endpoint URL
    API_URL = "https://drchrislevy--black-forest-labs-flux-model-f.modal.run/"  # Replace with your actual Modal app URL

    # use uuid for fname
    # Sample data
    fname = f"{folder}/{id}"
    data = {
        "prompts": [prompt],
        "fnames": [fname],
        "num_inference_steps": 4,
        "guidance_scale": 3.5,
    }

    # Make the API request
    response = requests.post(API_URL, json=data)

    if response.status_code == 200:
        results = response.json()

        for result in results:
            img_data = base64.b64decode(result["image"])

            with open(os.path.join("", f"{fname}.png"), "wb") as f:
                f.write(img_data)
            print(f"Saved: {fname}.png")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

    print("All images have been downloaded to the 'gens/' folder.")
    return True


serve()
