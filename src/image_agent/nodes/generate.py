"""Generation nodes: call OpenAI, Flux, or Gemini to produce an image."""

from __future__ import annotations

import logging

from openai import BadRequestError

from image_agent.providers.openai_image import generate_openai_image, generate_openai_image_with_refs
from image_agent.providers.flux_image import generate_flux_image
from image_agent.providers.gemini_image import generate_gemini_image
from image_agent.providers.image_utils import image_to_base64
from image_agent.state import ImageAgentState
from image_agent.utils.logger import log_pipeline_step

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-provider size mapping helpers
# ---------------------------------------------------------------------------

def map_size_openai(width: int, height: int) -> str:
    """Map ideal size to closest OpenAI-supported size."""
    if width > height:
        return "1536x1024"
    elif height > width:
        return "1024x1536"
    return "1024x1024"


def map_size_gemini(width: int, height: int) -> str:
    """Map ideal size to closest Gemini aspect ratio string."""
    if width > height:
        return "16:9"
    elif height > width:
        return "3:4"
    return "1:1"


def map_size_flux(width: int, height: int) -> tuple[int, int]:
    """Map ideal size to closest Flux-compatible size (multiples of 64)."""
    w = round(width / 64) * 64
    h = round(height / 64) * 64
    return w, h


def _parse_size(size_str: str) -> tuple[int, int]:
    """Parse a 'WxH' size string into (width, height)."""
    w, h = (int(d) for d in size_str.split("x"))
    return w, h


def openai_generate_node(state: ImageAgentState) -> dict:
    """Generate an image using OpenAI gpt-image-1."""
    prompt = state.get("enhanced_prompt") or state["original_prompt"]
    params = state.get("generation_params", {})
    ref_images = state.get("reference_images") or []

    # Map ideal size to OpenAI-supported size
    w, h = _parse_size(params.get("size", "1024x1024"))
    openai_size = map_size_openai(w, h)

    try:
        if ref_images:
            logger.info("Using OpenAI image edit with %d reference images", len(ref_images))
            image_bytes = generate_openai_image_with_refs(
                prompt,
                ref_images,
                size=openai_size,
                quality=params.get("quality", "high"),
            )
        else:
            image_bytes = generate_openai_image(
                prompt,
                size=openai_size,
                quality=params.get("quality", "high"),
                n=params.get("n", 1),
            )
    except BadRequestError as exc:
        log_pipeline_step("Generate", "openai \u2717 " + str(exc.message)[:80])
        return {"error": f"OpenAI rejected the request: {exc.message}"}

    log_pipeline_step("Generate", "openai \u2713")
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
    ref_images = state.get("reference_images") or []

    # Map ideal size to Flux-compatible size (multiples of 64)
    w, h = _parse_size(params.get("size", "1024x1024"))
    width, height = map_size_flux(w, h)

    try:
        if ref_images:
            logger.info("Using Flux image-to-image with reference image")
            image_bytes = generate_flux_image(
                prompt,
                width=width,
                height=height,
                reference_image=ref_images[0],  # Flux supports single ref
            )
        else:
            image_bytes = generate_flux_image(
                prompt,
                width=width,
                height=height,
            )
    except Exception as exc:
        log_pipeline_step("Generate", "flux \u2717 " + str(exc)[:80])
        return {"error": f"Flux generation failed: {exc}"}

    log_pipeline_step("Generate", "flux \u2713")
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
    ref_images = state.get("reference_images") or []

    # Map ideal size to Gemini aspect ratio
    w, h = _parse_size(params.get("size", "1024x1024"))
    aspect_ratio = map_size_gemini(w, h)

    try:
        if ref_images:
            logger.info("Using Gemini multimodal with %d reference images", len(ref_images))
        image_bytes = generate_gemini_image(
            prompt,
            aspect_ratio=aspect_ratio,
            reference_images=ref_images if ref_images else None,
        )
    except Exception as exc:
        log_pipeline_step("Generate", "gemini \u2717 " + str(exc)[:80])
        return {"error": f"Gemini generation failed: {exc}"}

    log_pipeline_step("Generate", "gemini \u2713")
    return {
        "generation_metadata": {
            "provider": "gemini",
            "model": get_settings().gemini_image_model,
            "prompt_used": prompt,
            "params": params,
            "image_b64": image_to_base64(image_bytes),
        },
    }
