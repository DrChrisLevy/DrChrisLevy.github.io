# Personal Website

This is the code for my [personal website](https://drchrislevy--drchrislevy-serve.modal.run/), created with [fastHTML](https://docs.fastht.ml/), and
[monsterui](https://monsterui.answer.ai/). For the first years it used [quarto](https://quarto.org/). Then I moved to fastHTML and monsterui
because I want to start building some dynamic content over time.

The original version of the site was hosted on github pages ---> [https://drchrislevy.github.io/](https://drchrislevy.github.io/)
Now this link is a redirect to the newer version, which is currently hosted on modal ---> [https://drchrislevy--drchrislevy-serve.modal.run/](https://drchrislevy--drchrislevy-serve.modal.run/).

I would like to get my own domain name eventually.

## Adding A New Blog Post

- Create a new notebook in the `posts` folder (see other examples for how to do this).
- Add the notebook to the `posts` folder.
- Store images in the `posts/static_blog_imgs` folder.
- Add the path of the notebook to the `fpaths` list in `blog.py`.


## Deploying

- `uv run modal deploy deploy_to_modal.py`

## Running locally

- `uv run main.py`

## Redirect GH Pages

- `gh-pages` branch only contains `index.html` and `blog.html`. They contain hardcoded urls with redirects.

## TOOD:

- more content and more to navbar
- more custom styling
- ability to write/add posts as notebooks or markdown or fastHTML
- clean up functions, and improve them, in blog.py for rendering
- blog more!