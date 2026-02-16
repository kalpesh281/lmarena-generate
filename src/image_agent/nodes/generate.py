"""Generation nodes: call OpenAI or Flux to produce an image."""

from __future__ import annotations

from image_agent.providers.openai_image import generate_openai_image
from image_agent.providers.flux_image import generate_flux_image
from image_agent.providers.image_utils import image_to_base64
from image_agent.state import ImageAgentState


def openai_generate_node(state: ImageAgentState) -> dict:
    """Generate an image using OpenAI gpt-image-1."""
    prompt = state.get("enhanced_prompt") or state["original_prompt"]
    params = state.get("generation_params", {})

    image_bytes = generate_openai_image(
        prompt,
        size=params.get("size", "1024x1024"),
        quality=params.get("quality", "high"),
        n=params.get("n", 1),
    )

    return {
        "generation_metadata": {
            "provider": "openai",
            "model": "gpt-image-1",
            "prompt_used": prompt,
            "params": params,
            "image_b64": image_to_base64(image_bytes),
        },
    }


def flux_generate_node(state: ImageAgentState) -> dict:
    """Generate an image using Flux via Hugging Face Inference API."""
    prompt = state.get("enhanced_prompt") or state["original_prompt"]
    params = state.get("generation_params", {})

    # Parse size string into width/height
    size = params.get("size", "1024x1024")
    width, height = (int(d) for d in size.split("x"))

    image_bytes = generate_flux_image(
        prompt,
        width=width,
        height=height,
    )

    return {
        "generation_metadata": {
            "provider": "flux",
            "model": "flux-1.1-pro",
            "prompt_used": prompt,
            "params": params,
            "image_b64": image_to_base64(image_bytes),
        },
    }
