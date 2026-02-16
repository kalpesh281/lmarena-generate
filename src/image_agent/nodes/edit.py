"""Edit node: modify an existing image using OpenAI."""

from __future__ import annotations

from openai import BadRequestError

from image_agent.providers.openai_image import edit_openai_image
from image_agent.providers.image_utils import image_to_base64
from image_agent.state import ImageAgentState


def edit_node(state: ImageAgentState) -> dict:
    """Edit an existing image using OpenAI's image edit API."""
    # For edits, use original_prompt (the user's edit instruction).
    # Enrich with context about what the previous image contained so
    # pronouns like "he", "it" resolve correctly.
    prompt = state["original_prompt"]
    last_prompt = state.get("last_prompt")
    if last_prompt:
        prompt = f"Original image: {last_prompt}\nEdit instruction: {prompt}"
    source_path = state.get("source_image_path")

    if not source_path:
        return {"error": "No source image provided for editing."}

    params = state.get("generation_params", {})
    try:
        image_bytes = edit_openai_image(
            prompt,
            source_path,
            size=params.get("size", "1024x1024"),
        )
    except BadRequestError as exc:
        return {"error": f"OpenAI rejected the edit request: {exc.message}"}

    return {
        "generation_metadata": {
            "provider": "openai",
            "model": "gpt-image-1",
            "mode": "edit",
            "prompt_used": prompt,
            "source_image": source_path,
            "image_b64": image_to_base64(image_bytes),
        },
    }
