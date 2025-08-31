import pathlib

import modal

app = modal.App("demo-volume")

volume = modal.Volume.from_name("demo-volume", create_if_missing=True)

p = pathlib.Path("/data/demo.txt")


@app.function(volumes={"/data/": volume})
def f(msg):
    p.write_text(msg)
    volume.commit()  # Persist changes


@app.function(volumes={"/data/": volume})
def g(reload: bool = False):
    if reload:
        volume.reload()  # Fetch latest changes
    if p.exists():
        print(p.read_text())
    else:
        print(f"{p=} does not exist! Execute the function f() first.")


@app.local_entrypoint()
def main():
    f.remote("HELLO CHRIS")
    g.remote(reload=False)  # prints ---> HELLO
    f.remote("BYE")  # container for `f` starts, commits file to read BYE
    g.remote(reload=False)  # reuses container for `g`, no reload, prints ---> HELLO
    for _ in range(10):
        # reuses container for `g`, **no** reload, prints ---> HELLO
        g.remote(reload=False)
    # reuses container for `g`, with **reload** to get recent commits, prints ---> BYE
    g.remote(reload=True)
