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
