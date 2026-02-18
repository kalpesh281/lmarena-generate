"""Gemini 2.5 Flash image generation via google-genai SDK."""

from __future__ import annotations

import base64
import io

from image_agent.config import get_settings


def generate_gemini_image(
    prompt: str,
    *,
    aspect_ratio: str = "1:1",
    reference_images: list[dict] | None = None,
) -> bytes:
    """Generate an image using Gemini 2.5 Flash. Returns raw PNG bytes.

    If reference_images are provided, builds a multimodal request with
    PIL Image objects alongside the text prompt.
    """
    from google import genai
    from google.genai import types
    from PIL import Image

    settings = get_settings()
    client = genai.Client(api_key=settings.gemini_api_key)

    # Build contents: multimodal if we have reference images, text-only otherwise
    if reference_images:
        contents: list = []
        for ref in reference_images:
            img_bytes = base64.b64decode(ref["image_b64"])
            img = Image.open(io.BytesIO(img_bytes))
            contents.append(img)
        contents.append(
            f"Using the reference images above for visual accuracy, generate: {prompt}"
        )
    else:
        contents = prompt

    response = client.models.generate_content(
        model=settings.gemini_image_model,
        contents=contents,
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
