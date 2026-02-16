"""Edit node: modify an existing image using OpenAI."""

from __future__ import annotations

from image_agent.providers.openai_image import edit_openai_image
from image_agent.providers.image_utils import image_to_base64
from image_agent.state import ImageAgentState


def edit_node(state: ImageAgentState) -> dict:
    """Edit an existing image using OpenAI's image edit API."""
    prompt = state.get("enhanced_prompt") or state["original_prompt"]
    source_path = state.get("source_image_path")

    if not source_path:
        return {"error": "No source image provided for editing."}

    params = state.get("generation_params", {})
    image_bytes = edit_openai_image(
        prompt,
        source_path,
        size=params.get("size", "1024x1024"),
    )

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
