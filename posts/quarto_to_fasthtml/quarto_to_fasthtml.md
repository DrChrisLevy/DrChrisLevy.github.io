---
title: Migrating My Blog from Quarto to fastHTML and MonsterUI
author: Chris Levy
date: '2025-02-25'
date-modified: '2025-02-25'
image: quarto2fasthtml.png
toc: true
description: In this blog post, I walk through the process of migrating my blog from Quarto to fastHTML.
tags:
  - fasthtml
  - quarto
  - modal
  - MonsterUI
---


# Switching From Quarto to fastHTML and MonsterUI

## Introduction

[Quarto](https://quarto.org/) is an amazing resource for creating a blog. 
It's a static site generator that can output to many different formats.
I've been using Quarto to create this blog for a while now. It was the best way for me to get started with blogging
back in February 2024. Seventeen blog posts later, and one year later, it's time for a change.

There is absolutely nothing wrong with Quarto. It has been a fantastic tool for creating and maintaining a static blog. However, as my interests and projects have evolved, I've found the need to expand into more dynamic content and web applications. My background is primarily in Python and backend development, and I don't have much experience with front-end development or web applications as of writing this. This is where [fastHTML](https://docs.fastht.ml/) and [MonsterUI](https://monsterui.answer.ai/) come into play. These libraries are built for making web applications with Python, providing the flexibility and functionality required to build interactive and dynamic web applications, which aligns better with my current interests and future goals. 

My immediate goal was to achieve a bare minimum conversion from Quarto to fastHTML. I don't need a lot of bells and whistles yet; I just want to switch over my main content and have it all working. Even if it means I have less functionality then I had with Quarto. However, I have plans to add more functionality in the near future. I know that in the long run this will give me the most flexibility and control.

## The Motivation/Starting Code

Isaac Flath and Jeremy Howard work together on fastHTML and MonsterUI, as well as others from the team over at [Answer.AI](https://answer.ai).
You can read about the MonsterUI release blog post [here](https://www.answer.ai/posts/2025-01-15-monsterui.html). 

Isaac Flath had a blog created with Quarto [here](https://isaac-flath.github.io/website/). He then pointed me in the direction of his blog which he switched to fastHTML and MonsterUI [here](https://isaac.up.railway.app/). And here is the [code](https://github.com/Isaac-Flath/web/blob/main/app/blog.py) for the converted blog. I simply copy/pasted this and used it as a starting point. The main thing I was looking for were some functions for converting and rendering the notebook cells. All my previous blog posts were in the form of Jupyter notebooks, so I needed to figure out how to convert those to fastHTML and MonsterUI.

The team over at Answer.AI are experts with Jupyter notebooks, since they have developed libraries such as 
[nbdev](https://nbdev.fast.ai/) and [execnb](https://github.com/fastai/execnb). They already know how to convert notebook
cells to various formats and render them. The main approach for creating the blog is to:

- store the post source files as Jupyter notebooks in a folder called `posts`
- loop over the notebooks meta data to create the page which renders the list of blog posts
- for each blog post, extract the cells and render the input/output based on the cell type

I recommend looking at the code in [Isaac's blog](https://github.com/Isaac-Flath/web/blob/main/app/blog.py) for the details.
I don't use nbdev (yet) so I just used a bare minimum of functions to get the job done. You could refer to
my [code](https://github.com/DrChrisLevy/DrChrisLevy.github.io/blob/main/blog.py) for the details as well.

There was one issue with rendering certain types of notebook cells. I saw these issues in Isaac's older blog posts
as well as mine. The issue was that some of the Jupyter cell outputs would have odd symbols such as unicode hex characters.
Another issue was that some "complex" data types did not look as nice compared to the original notebook cells.
It took me a little while to determine the issue but here was the fix. In the function `render_output`
change the lines 

```python
if d := _g('text/plain'):
  return escape(d)
```

to

```python
if d := _g("text/plain"):
    return Safe(f"<pre><code>{d}</code></pre>")
```

Other things I had to change along the way in my previous posts were:

- place all the static images into a designated folder called `static_blog_imgs`. I went through all the previous posts and made code changes such as `![](imgs/fasthtml_demo2.png)` to `![](static_blog_imgs/fasthtml_demo2.png)`

- fix all the youtube video links i.e. `{{< video https://www.youtube.com/watch?v=YoXkFCA0qC8 >}}` changed to `<iframe src=\"https://www.youtube.com/embed/YoXkFCA0qC8\" width=\"960\" height=\"540\" allowfullscreen uk-responsive></iframe>\`

- Remove all the Quarto Specific code and files i.e. delete files `blog.qmd`, `index.qmd`, etc.

- My quarto blog was hosted on GitHub pages. GitHub pages is for hosting static websites. For now I'm going to host the converted blog on [Modal](https://modal.com/). To keep that all working, I added a redirect route on my branch `gh-pages` to redirect to the new blog. This means that anyone who visits [https://drchrislevy.github.io/](https://drchrislevy.github.io/) will be redirected to the new blog. This was as simple as asking a LLM what to do and the result was to create two files `index.html` and `blog.html` in the root directory of the repo on the `gh-pages` branch. See [here](https://github.com/DrChrisLevy/DrChrisLevy.github.io/tree/gh-pages) for the details.

For all the juicy details see the [PR](https://github.com/DrChrisLevy/DrChrisLevy.github.io/pull/21) which made all the changes.

## Deploying to Modal

Down the road I want to get a custom domain for my blog.
Modal offers $30/month of compute credits per month for free, and I only pay for when the container is running (when people are visiting the blog).
I set a 3 minute timeout on the container for now and the containers spin up in 1-2 seconds anyway. 
Down the road I can improve this. Modal does support custom domains
but not on the free tier right now.

Here is the code I use to deploy the blog to Modal.

```python
import modal

app = modal.App("drchrislevy")


@app.function(
    image=modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_pyproject("pyproject.toml")
    .add_local_dir("posts", remote_path="/root/posts")
    .add_local_file("utils.py", remote_path="/root/utils.py")
    .add_local_file("main.py", remote_path="/root/main.py")
    .add_local_file("blog.py", remote_path="/root/blog.py"),
    allow_concurrent_inputs=100,  # TODO
    container_idle_timeout=3 * 60,
    secrets=[modal.Secret.from_dotenv()],
)
@modal.asgi_app()
def serve():
    from main import app

    return app
```

```bash
uv run modal deploy deploy_to_modal.py
```




