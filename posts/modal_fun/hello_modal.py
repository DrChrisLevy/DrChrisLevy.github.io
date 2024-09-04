import modal

app = modal.App("hello-modal")


@app.function()
def f(i):
    print(f"hello modal! {i} + {i} is {i+i}\n")
    return i


@app.local_entrypoint()
def main():
    print("This is running locally")
    print(f.local(1))

    print("This is running remotely on Modal")
    print(f.remote(2))

    print("This is running in parallel and remotely on Modal")
    total = 0
    for ret in f.map(range(2500)):
        total += ret
    print(total)
