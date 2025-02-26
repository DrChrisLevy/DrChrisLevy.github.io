import time

import modal

app = modal.App("concurrent-requests-container")


@app.function(allow_concurrent_inputs=100, concurrency_limit=1)
def blocking_function():
    print("running IO bound task")
    time.sleep(5)
    return 42


@app.function(allow_concurrent_inputs=100, concurrency_limit=1)
async def async_function(i):
    x = await blocking_function.remote.aio()
    return x * i


@app.local_entrypoint()
def blocking_main():
    total = 0
    for ret in async_function.map(range(80)):
        total += ret
    print(total)
