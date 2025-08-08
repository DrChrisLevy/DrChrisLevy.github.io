# server.py
import modal

image = (
    modal.Image.debian_slim()
    .pip_install("fastapi[standard]", "fastmcp>=2.3.2")
    .add_local_file("posts/mcp/python_sandbox.py", remote_path="/root/python_sandbox.py")
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
    def text2image(
        prompt: str,
        negative_prompt: str = "blurry, low quality, distorted",
        aspect_ratio: str = "16:9",
        true_cfg_scale: float = 4.0,
        randomize_seed: bool = True,
        num_inference_steps: int = 50,
        seed: int = 42,
    ) -> dict:
        """Given the text prompt and other parameters,
        uses an AI text2image model to generate an image and return the image url.
        Always return at least the image url so the user can see the image."""
        import requests

        endpoint_url = "https://drchrislevy--qwen-image-generator-qwenimagegenerator-gen-5fbcf5.modal.run/"
        response = requests.post(
            endpoint_url,
            params={
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "aspect_ratio": aspect_ratio,
                "true_cfg_scale": true_cfg_scale,
                "randomize_seed": randomize_seed,
                "num_inference_steps": num_inference_steps,
                "seed": seed,
            },
            timeout=120,
        )
        return response.json()

    @mcp.resource("users://{user_id}/profile")
    def get_user_profile(user_id: int) -> dict:
        """Retrieves a user's profile by ID."""
        # The {user_id} in the URI is extracted and passed to this function
        return {"id": user_id, "name": f"User {user_id}", "status": "active"}

    @mcp.resource("data://config")
    def get_config() -> dict:
        """Provides application configuration as JSON."""
        return {
            "theme": "dark",
            "version": "1.2.0",
            "features": ["tools", "resources"],
        }

    # Optional plain HTTP health check (easy to curl)
    @mcp.custom_route("/health", methods=["GET"])
    async def health(_req):
        return JSONResponse({"status": "ok"})

    @mcp.prompt
    def ask_about_topic(topic: str) -> str:
        """Generates a user message asking for an explanation of a topic."""
        return f"Can you please explain the concept of '{topic}'?"

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
