# Multimodal Retrieval with ColPali, Modal, and FastHTML

This was a project to force myself to build my first FastHTML app.
It used Modal for the backend and FastHTML for the frontend.

# Setup

- you need an account with Modal, OpenAI and Hugging Face.
- create a `.env` file with the following variables:

```
HUGGING_FACE_ACCESS_TOKEN=YOUR_TOKEN
GITHUB_ACCESS_TOKEN=YOUR_TOKEN
OPENAI_API_KEY=sk-YOUR_KEY
```

- Create a virtual environment and install the dependencies:

```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

- Deploy the backend Modal apps:

```
modal deploy multi_modal_rag.py
modal deploy pdf_retriever.py
```

# Testing Modal Backend

This will run the backend tests directly on Modal.
If these pass it means you have successfully deployed the backend Modal apps.

```
pytest -n 10 tests
```

# Running the FastHTML App

```
python main.py
```

# Lint

```
ruff check . --select F,I --fix; ruff format .