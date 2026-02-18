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


def generate_openai_image_with_refs(
    prompt: str,
    reference_images: list[dict],
    *,
    size: str = "1024x1024",
    quality: str = "high",
) -> bytes:
    """Generate an image using OpenAI images.edit with reference images.

    Uses the edit endpoint which supports up to 16 input images for
    visual conditioning alongside the text prompt.
    """
    import io
    settings = get_settings()
    client = _client()

    # Convert base64 reference images to file-like objects
    image_files = []
    for ref in reference_images:
        img_bytes = base64.b64decode(ref["image_b64"])
        buf = io.BytesIO(img_bytes)
        buf.name = "reference.png"
        image_files.append(buf)

    # Use first image as the primary, pass rest as additional
    resp = client.images.edit(
        model=settings.image_model,
        image=image_files[0],
        prompt=f"Using the reference image(s) for visual accuracy: {prompt}",
        size=size,
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
