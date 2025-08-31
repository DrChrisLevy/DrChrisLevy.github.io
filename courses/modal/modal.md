# AI Engineering with Modal: Build and Deploy Real-World Applications

## Thoughts

- the modal docs are already so good, I don't want to re-write or even summarize them
- We can point to them for more information. But it does not make sense to try to cover everything there
just in videos. 
- More importantly, would be to build projects and applications using Modal

## Introduction / Overview

- [explain what modal is](https://modal.com/docs/guide)
- set up a environment with `uv` and how to create an account
- run the first hello world like example

### Defining Images

- [defining images](https://modal.com/docs/guide/images)

### Simple examples

- go through some simple examples
- show the modal dashboard

### Scaling Out

- Let's do this before going into GPU
- [scaling out](https://modal.com/docs/guide/scale)
- [input concurrency](https://modal.com/docs/guide/concurrent-inputs)


### Apps, Functions, and entrypoints

- [see here and more](https://modal.com/docs/guide/apps)


### Dicts and Queues
- [dicts and queues](https://modal.com/docs/guide/dicts-and-queues)


### Batch Processing
- https://modal.com/docs/guide/batch-processing

### Secrets and Environment Variables

- [secrets](https://modal.com/docs/guide/secrets)

### Web Endpoints

- [web endpoints](https://modal.com/docs/guide/webhooks)

### Data Sharing and Storage

- [see here](https://modal.com/docs/guide/local-data)
- [volumes](https://modal.com/docs/guide/volumes)

### Sandboxes

- [sandboxes](https://modal.com/docs/guide/sandbox)

## Project Ideas

- Building Encoder Classifiers - Fine Tuning Modern Bert
    - serving with vLLM? - [here](https://x.com/vanstriendaniel/status/1915423144321470501)
- Embedding Texts - We could serve in FAISS or something or some other cool local vector DB
    - https://modal.com/docs/examples/amazon_embeddings
- Whisper Transcription - Audio - YouTube Videos
- Image Generation
- Serving LLMS with vLLM
    - [fast mode](https://x.com/modal_labs/status/1918352425414516981)

- Fine-Tuning LLMs with Axolotl or Unsloth
- Building and Serving Web Applications with FastHTML and MonsterUI
    - a lot of the above things could be done this way
    - building evaluation look at your data apps
- ColPALI and late-interaction models 
- 