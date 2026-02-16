"""OpenAI gpt-image-1 provider for image generation and editing."""

from __future__ import annotations

import base64

from openai import OpenAI

from image_agent.config import get_settings


def _client() -> OpenAI:
    return OpenAI(api_key=get_settings().openai_api_key)


def generate_openai_image(
    prompt: str,
    *,
    size: str = "1024x1024",
    quality: str = "high",
    n: int = 1,
) -> bytes:
    """Generate an image using OpenAI gpt-image-1. Returns raw PNG bytes."""
    settings = get_settings()
    resp = _client().images.generate(
        model=settings.image_model,
        prompt=prompt,
        size=size,
        quality=quality,
        n=n,
    )
    b64_data = resp.data[0].b64_json
    return base64.b64decode(b64_data)


def edit_openai_image(
    prompt: str,
    image_path: str,
    *,
    size: str = "1024x1024",
) -> bytes:
    """Edit an existing image using OpenAI. Returns raw PNG bytes."""
    settings = get_settings()
    with open(image_path, "rb") as img_file:
        resp = _client().images.edit(
            model=settings.image_model,
            prompt=prompt,
            image=img_file,
            size=size,
        )
    b64_data = resp.data[0].b64_json
    return base64.b64decode(b64_data)
