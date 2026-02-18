"""Flux image generation via Hugging Face Inference API."""

from __future__ import annotations

import base64
import io

from image_agent.config import get_settings


def generate_flux_image(
    prompt: str,
    *,
    width: int = 1024,
    height: int = 1024,
    num_inference_steps: int = 25,
    reference_image: dict | None = None,
) -> bytes:
    """Generate an image using Flux on Hugging Face Inference API. Returns raw bytes.

    If reference_image is provided, uses image_to_image for visual conditioning
    (Flux only supports a single reference image).
    """
    from huggingface_hub import InferenceClient
    from PIL import Image

    settings = get_settings()
    client = InferenceClient(token=settings.huggingface_api_key)

    if reference_image:
        # Use image-to-image with the reference
        img_bytes = base64.b64decode(reference_image["image_b64"])
        ref_img = Image.open(io.BytesIO(img_bytes))

        image = client.image_to_image(
            ref_img,
            prompt=prompt,
            model=settings.flux_model,
        )
    else:
        image = client.text_to_image(
            prompt,
            model=settings.flux_model,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
        )

    # Convert PIL Image to bytes
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()
