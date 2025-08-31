# Introduction to Modal

## Setting up a Project Folder and Virtual Environment

It's always good practice to create a virtual environment when starting a new project.
You can create a virtual environment with various tools.
Once the environment is created, you can set up Modal within that environment.

In this course, I will personally be using `uv`.
Check it out on [GitHub](https://github.com/astral-sh/uv) and read the docs [here](https://docs.astral.sh/uv/getting-started/installation/). You can choose any other tool you want.
After installing uv, you can set up the project for this course like this:

```
uv init modal-course
``` 

```
cd modal-course
```

## Installing Modal

Begin by reading the introduction and getting started guide on the Modal website.
You can find it [here](https://modal.com/docs/guide).

Here are the main steps you need to follow to get started:

1. Create an account at [modal.com](https://modal.com/)
2. Run `pip install modal` to install the Modal Python package
3. Run `modal setup` to authenticate (if this doesn't work, try `python -m modal setup`)

You don't need to add any payment information to get started,
and Modal provides $30 of compute credits every month.

Install Modal (note that I am using uv to install the package):

```
uv add modal
uv run modal setup
```

## Optional: Add Modal Tokens to Environment Variables

The above command puts the `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`
in the file `~/.modal.toml`. 

```
cat ~/.modal.toml
```

You can optionally add the `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`
to environment variables in a `.env` file, if you prefer.
This may be necessary if you're setting up Modal within a Docker container
or some other environment.

```
MODAL_TOKEN_ID=<your-modal-token-id>
MODAL_TOKEN_SECRET=<your-modal-token-secret>
```

## Walkthrough of the Modal Website

- [Modal](https://modal.com/)
- [Pricing](https://modal.com/pricing)
- [Blog](https://modal.com/blog)
- [Docs](https://modal.com/docs)
- [Dashboard](https://modal.com/apps/drchrislevy/main)

Spend some time looking through the website, docs, and dashboard.