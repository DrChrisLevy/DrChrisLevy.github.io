# server.py
import modal

image = (
    modal.Image.debian_slim()
    .pip_install("fastapi[standard]", "fastmcp>=2.3.2")
    .add_local_file("posts/mcptalk/python_sandbox.py", remote_path="/root/python_sandbox.py")
)

app = modal.App("fastmcp-modal-demo", image=image)


@app.function(scaledown_window=60 * 60)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def mcp_asgi():
    # Everything below runs inside the Modal container
    from fastmcp import Context, FastMCP
    from python_sandbox import ModalSandbox
    from starlette.responses import JSONResponse

    sb = ModalSandbox()

    mcp = FastMCP(
        name="HelpfulAssistant",
        instructions="""This server provides some useful tools for the user.""",
    )

    @mcp.tool
    def execute_python_code(code: str) -> dict:
        """Run arbitrary python code in a sandboxed environment. It is stateful and persistent between calls.
        Install packages with: os.system("pip install <package_name>")
        """
        return sb.run_code(code)

    @mcp.tool
    def edit_image(
        image_url: str,
        prompt: str,
        negative_prompt: str = " ",
        true_cfg_scale: float = 4.0,
        seed: int = 0,
        randomize_seed: bool = False,
        num_inference_steps: int = 8,
    ) -> dict:
        """Edit an existing image using AI based on a text prompt.
        Provide the URL of the image to edit and a description of how you want to modify it.
        Returns the edited image URL and metadata."""
        import requests

        endpoint_url = "https://drchrislevy--qwen-image-editor-fast-qwenimageeditor-edit-516e26.modal.run/"
        response = requests.post(
            endpoint_url,
            params={
                "image_url": image_url,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "true_cfg_scale": true_cfg_scale,
                "seed": seed,
                "randomize_seed": randomize_seed,
                "num_inference_steps": num_inference_steps,
            },
            timeout=120,
        )
        return response.json()

    @mcp.resource("library://{media_id}/document")
    def get_media(media_id: int) -> dict:
        """Retrieves a media by ID."""
        if media_id == "1":
            return {"id": 1,"url": "https://dev-dashhudson-static.s3.amazonaws.com/research/chris/images/jordan_and_kobe.jpg"}
        elif media_id == "2":
            return {"id": 2, "url": "https://dev-dashhudson-static.s3.amazonaws.com/research/chris/images/star_history.png"}
        else:
            return {"id": media_id, "url": "https://dev-dashhudson-static.s3.amazonaws.com/research/chris/images/jordan_and_kobe.jpg"}

    @mcp.resource("data://media")
    def get_media_data() -> list[dict]:
        """Retrieve the all the media data in the library"""
        return [{"id": 1,"url": "https://dev-dashhudson-static.s3.amazonaws.com/research/chris/images/jordan_and_kobe.jpg"},
               {"id": 2, "url": "https://dev-dashhudson-static.s3.amazonaws.com/research/chris/images/star_history.png"}]

    # Optional plain HTTP health check (easy to curl)
    @mcp.custom_route("/health", methods=["GET"])
    async def health(_req):
        return JSONResponse({"status": "ok"})

    @mcp.prompt
    def ask_about_topic(topic: str) -> str:
        """Generates a user message asking for an explanation of a topic."""
        return f"Can you please explain the concept of '{topic}'?"

    @mcp.prompt
    def create_shortcut_ticket(epic: str, title: str, description: str) -> str:
        """Generates a shortcut ticket"""
        return f"Can you please create a shortcut ticket for the epic '{epic}', title '{title}', and description '{description}'?"

    @mcp.tool
    async def analyze_sentiment(text: str, ctx: Context) -> dict:
        """Analyze the sentiment of text using the client's LLM."""
        prompt = f"""Analyze the sentiment of the following text as positive, negative, or neutral. 
        Just output a single word - 'positive', 'negative', or 'neutral'.
        
        Text to analyze: {text}"""

        # Request LLM analysis
        response = await ctx.sample(prompt)

        # Process the LLM's response
        sentiment = response.text.strip().lower()

        # Map to standard sentiment values
        if "positive" in sentiment:
            sentiment = "positive"
        elif "negative" in sentiment:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        return {"text": text, "sentiment": sentiment}

    # Expose the MCP server as an ASGI app.
    # Default transport is Streamable HTTP; we mount at /mcp
    return mcp.http_app(path="/mcp")
