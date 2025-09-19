# Run Your First Modal Function

Create your own simple Python function. 
I'll use the following for illustration:

```python
import random
import time


def compute(x: int):
    # Simulate some work
    time.sleep(2)
    result = x + random.randint(0, 10)
    return result
```

The amazing thing about Modal is that we can add
several lines of code and instantly run this in the cloud
on Modal's infrastructure.

Here is the complete code:

```python
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
```

Put the code in a file and name it `my_first_modal_app.py`.
Then execute the following command:

```bash
modal run my_first_modal_app.py
```

- Run the code a few times and inspect the results in the terminal.
- Try updating the function observing the results in the terminal.
- Explore the Modal dashboard to see the logs and other container metrics.
- You can list the running containers with `modal container list`, see docs [here](https://modal.com/docs/reference/cli/container#modal-container-list)
- You get into a running container with `modal shell <container_id>`, see docs [here](https://modal.com/docs/reference/cli/shell#modal-shell)
- Read the "Hello World" example in the Modal docs [here](https://modal.com/docs/examples/hello_world)
