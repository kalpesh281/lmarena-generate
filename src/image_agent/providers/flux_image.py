"""Flux image generation via Hugging Face Inference API."""

from __future__ import annotations

from image_agent.config import get_settings


def generate_flux_image(
    prompt: str,
    *,
    width: int = 1024,
    height: int = 1024,
    num_inference_steps: int = 25,
) -> bytes:
    """Generate an image using Flux on Hugging Face Inference API. Returns raw bytes."""
    from huggingface_hub import InferenceClient

    settings = get_settings()
    client = InferenceClient(token=settings.huggingface_api_key)

    image = client.text_to_image(
        prompt,
        model=settings.flux_model,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
    )

    # Convert PIL Image to bytes
    import io
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()
