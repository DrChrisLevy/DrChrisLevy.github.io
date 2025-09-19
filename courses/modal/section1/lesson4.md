# Defining Images

After you watch the video, head over to the Modal documentation
for [Defining Images](https://modal.com/docs/guide/images) and give it a read through.

Some notes worth highlighting here are:

- Add Python packages with `pip_install`
- Add local files with `add_local_dir` and `add_local_file`
- Can import libraries that are not installed locally within the Modal Function body
- Remember that if you return objects from a Modal Function you will need the corresponding library installed locally
to handle the returned objects.
- Run shell commands with `.run_commands`
- Image caching and rebuilds
- Run `image.add_local_*` commands last in your image build to avoid rebuilding images with every local file change. Modal will then add these files to containers on startup instead, saving build time.

Here is a simple example to illustrate some of these concepts.

- create a temporary directory called `assets`: `mkdir assets`
- put a file in it: `echo "Hello World" > assets/junk.txt`
- create a python file called `modal_image_demo.py` and copy this code into it

```python
# modal_image_demo.py

import modal

app = modal.App("image-demo")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("wget", "git")
    .pip_install("pandas==2.2.3", "pyfiglet==1.0.2", "tabulate==0.9.0")
    .run_commands(
        "echo 'Fetching sample CSV …'",
        "mkdir -p /assets",
        "wget -q -O /assets/iris.csv https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
    )
    .add_local_dir("assets", remote_path="/assets")
    .add_local_file(__file__, remote_path="/demo.py")
)


@app.function(image=image)
def get_summary() -> dict:
    """
    Returns a dict so it can deserialize even if pandas isn't on your local machine.
    (If we returned the DataFrame itself, your local machine would also need pandas.)
    """
    import pandas as pd

    df = pd.read_csv("/assets/iris.csv")
    return df.describe().to_markdown()


@app.function(image=image)
def banner(msg: str) -> str:
    """
    pyfiglet lives only in the container; importing it here means we don't
    need it installed on your local machine.
    """
    import pyfiglet

    return str(pyfiglet.figlet_format(msg))


@app.local_entrypoint()
def main():
    print("=== dataset summary ===")
    print(get_summary.remote())

    print("\n=== pretty banner ===")
    print(banner.remote("Modal FTW"))

```


We also use the command modal shell again:

```
uv run modal shell modal_image_demo.py::get_summary
uv run modal shell modal_image_demo.py::banner
```

Run one of these to shell into the container and see what's going on.
Run these commands within the container:

- this is the file that was copied into the container (with contents from this `modal_image_demo.py` file)
```
cat /demo.py 
```

```
ls /assets
```

Run all the logic with:

```
modal run modal_image_demo.py
```

- Play around with the code and see how it works.
- Experiment with commenting out lines and see how it affects the image build.
- Try adding a new library to the image and see how it affects the image build.
- Try creating a new function with completely different logic