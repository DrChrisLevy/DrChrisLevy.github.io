# Scaling and Input Concurrency

- In many cases, you won’t need to worry about scaling, as Modal's auto scaler will automatically manage it for you. However, there are situations where you might want to customize the scaling preferences. This documentation provides guidance on how to configure the modal settings effectively.
- Read the documentation on [scaling](https://modal.com/docs/guide/scale)
- Read the documentation on [input concurrency](https://modal.com/docs/guide/concurrent-inputs)

Experiment with the scaling settings and input concurrency settings to see how they affect the performance of this simple function.
By default, all the settings are commented out which means the Modal auto scaler will manage the scaling for you.
You can adjust the `sleep` time as well as the number of inputs to see how they affect the performance.
As well, remember you can checkout your running app in the Modal dashboard.

```python
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
```

Place this code in a python file, for example `lesson3.py`, and run it with `modal run lesson3.py`.
