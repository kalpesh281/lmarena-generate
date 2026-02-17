"""Gemini 2.5 Flash image generation via google-genai SDK."""

from __future__ import annotations

from image_agent.config import get_settings


def generate_gemini_image(
    prompt: str,
    *,
    aspect_ratio: str = "1:1",
) -> bytes:
    """Generate an image using Gemini 2.5 Flash. Returns raw PNG bytes."""
    from google import genai
    from google.genai import types

    settings = get_settings()
    client = genai.Client(api_key=settings.gemini_api_key)

    response = client.models.generate_content(
        model=settings.gemini_image_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(
                aspect_ratio=aspect_ratio,
            ),
        ),
    )

    for part in response.parts:
        if part.inline_data is not None:
            return part.inline_data.data

    raise RuntimeError("Gemini returned no image data in the response")
