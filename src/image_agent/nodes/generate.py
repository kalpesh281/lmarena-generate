"""Generation nodes: call OpenAI or Flux to produce an image."""

from __future__ import annotations

from openai import BadRequestError

from image_agent.providers.openai_image import generate_openai_image
from image_agent.providers.flux_image import generate_flux_image
from image_agent.providers.gemini_image import generate_gemini_image
from image_agent.providers.image_utils import image_to_base64
from image_agent.state import ImageAgentState


def openai_generate_node(state: ImageAgentState) -> dict:
    """Generate an image using OpenAI gpt-image-1."""
    prompt = state.get("enhanced_prompt") or state["original_prompt"]
    params = state.get("generation_params", {})

    try:
        image_bytes = generate_openai_image(
            prompt,
            size=params.get("size", "1024x1024"),
            quality=params.get("quality", "high"),
            n=params.get("n", 1),
        )
    except BadRequestError as exc:
        return {"error": f"OpenAI rejected the request: {exc.message}"}

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

    try:
        image_bytes = generate_flux_image(
            prompt,
            width=width,
            height=height,
        )
    except Exception as exc:
        return {"error": f"Flux generation failed: {exc}"}

    return {
        "generation_metadata": {
            "provider": "flux",
            "model": "flux-1.1-pro",
            "prompt_used": prompt,
            "params": params,
            "image_b64": image_to_base64(image_bytes),
        },
    }


def gemini_generate_node(state: ImageAgentState) -> dict:
    """Generate an image using Gemini 2.5 Flash."""
    from image_agent.config import get_settings

    prompt = state.get("enhanced_prompt") or state["original_prompt"]
    params = state.get("generation_params", {})

    # Map size to aspect ratio (Gemini uses ratio strings like "1:1", "16:9")
    size = params.get("size", "1024x1024")
    w, h = (int(d) for d in size.split("x"))
    if w == h:
        aspect_ratio = "1:1"
    elif w > h:
        aspect_ratio = "16:9"
    else:
        aspect_ratio = "9:16"

    try:
        image_bytes = generate_gemini_image(prompt, aspect_ratio=aspect_ratio)
    except Exception as exc:
        return {"error": f"Gemini generation failed: {exc}"}

    return {
        "generation_metadata": {
            "provider": "gemini",
            "model": get_settings().gemini_image_model,
            "prompt_used": prompt,
            "params": params,
            "image_b64": image_to_base64(image_bytes),
        },
    }
