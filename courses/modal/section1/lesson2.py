import random
import time

import modal

app = modal.App(name="my-first-modal-app")


@app.function()
def compute(x: int):
    # Simulate some work
    time.sleep(2)
    result = x + random.randint(0, 10)
    return result


@app.local_entrypoint()
def main():
    # Run the function locally on your own machine
    res = compute.local(10)
    print("Got Back Result:", res)

    # Run the function remotely on Modal's infrastructure
    res = compute.remote(10)
    print("Got Back Result:", res)

    # Run the function remotely on Modal's infrastructure
    # in parallel across multiple containers
    results = [res for res in compute.map(range(50))]
    print(len(results))
    print(f"Total Sum: {sum(results)}")
