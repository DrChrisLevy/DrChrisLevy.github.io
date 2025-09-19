# Personal Website

This is the code for my [personal website](https://drchrislevy--drchrislevy-serve.modal.run/), created with [fastHTML](https://docs.fastht.ml/), and
[monsterui](https://monsterui.answer.ai/).

The original version of the site was hosted on github pages ---> [https://drchrislevy.github.io/](https://drchrislevy.github.io/)
Now this link is a redirect to the newer version, which is currently hosted on modal ---> [https://drchrislevy--drchrislevy-serve.modal.run/](https://drchrislevy--drchrislevy-serve.modal.run/).

I would like to get my own domain name eventually.

## Adding A New Blog Post

- Create a new notebook in the `posts` folder (see other examples for how to do this).
- Add the notebook to the `posts` folder.
- Store images in the `posts/static_blog_imgs` folder.
- Add the path of the notebook to the `fpaths` list in `blog.py`.

## Running locally

- `uv run main.py`

## Redirect GH Pages

- `gh-pages` branch only contains `index.html` and `blog.html`. They contain hardcoded urls with redirects.

## Unit Tests

```bash
uv run pytest tests
```


## Transcription of the Course Videos

Manually edit the file to point to the new audio files and run the script.
```
uv run modal run scripts/transcribe_audio.py
```


## Adding New Content Courses

- Add the new content to the `courses` directory.
- need an `.md` file for each lesson.
- `uv run modal run scripts/transcribe_audio.py` # edit and run to generate the transcript.
- add a new entry to `scripts/course_lessons_db.py`

## Deploy to Modal

```bash
uv run modal deploy deploy_to_modal.py
```

- If deleted the DB then you need to
    - create a user with user_name `admin`
    - run the `scripts/courses_lessons_db.py` script to create the courses and lessons. Run this in ipython within a modal container.