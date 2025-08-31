import random
import time

import modal

app = modal.App(name="my-first-modal-app")


# Play around with these container scaling settings
@app.function(max_containers=None, min_containers=None, scaledown_window=None)
# @modal.concurrent(max_inputs=120, target_inputs=100) # Play around with these settings
def compute(x: int):
    # Simulate some work
    time.sleep(2)
    result = x + random.randint(0, 10)
    return result


@app.local_entrypoint()
def main():
    # Run the function remotely on Modal's infrastructure
    # in parallel across multiple containers
    results = [res for res in compute.map(range(50))]
    print(len(results))
    print(f"Total Sum: {sum(results)}")
