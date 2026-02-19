"""Reference images node: download, validate, and analyze reference images."""

from __future__ import annotations

import base64
import io
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

from PIL import Image
from openai import OpenAI

from image_agent.config import get_settings
from image_agent.prompts.templates import REFERENCE_IMAGE_ANALYSIS_PROMPT
from image_agent.providers.image_utils import download_image
from image_agent.state import ImageAgentState
from image_agent.utils.logger import log_pipeline_step

logger = logging.getLogger(__name__)

# Patterns that indicate a low-quality or irrelevant image URL.
_BAD_URL_PATTERNS = [
    # YouTube thumbnails
    re.compile(r"/hqdefault\.jpg", re.IGNORECASE),
    re.compile(r"/maxresdefault\.jpg", re.IGNORECASE),
    re.compile(r"/sddefault\.jpg", re.IGNORECASE),
    re.compile(r"/mqdefault\.jpg", re.IGNORECASE),
    re.compile(r"i\.ytimg\.com", re.IGNORECASE),
    # Avatars, favicons, logos, icons, banners
    re.compile(r"/(avatar|favicon|logo|icon|banner)[s_\-./]", re.IGNORECASE),
    # Tiny wiki thumbnails (e.g. /thumb/...120px-Something.png)
    re.compile(r"/thumb/.+\d{1,3}px", re.IGNORECASE),
    # SVG and GIF files (not useful as generation references)
    re.compile(r"\.svg(\?|$)", re.IGNORECASE),
    re.compile(r"\.gif(\?|$)", re.IGNORECASE),
    # Ad images, spinners, placeholders
    re.compile(r"/(ads?|spinner|placeholder|loading|spacer)[_\-./]", re.IGNORECASE),
]


def _is_quality_url(url: str) -> bool:
    """Return True if the URL is likely to be a useful reference image."""
    # Reject URLs matching known bad patterns
    for pattern in _BAD_URL_PATTERNS:
        if pattern.search(url):
            return False

    # Reject URLs with very short paths (generic / stub images)
    parsed = urlparse(url)
    # Strip leading slash and query; path like "/" or "/image" is too short
    path = parsed.path.strip("/")
    if len(path) < 8:
        return False

    return True


def _download_and_validate(url: str, max_size: tuple[int, int] = (1024, 1024)) -> dict | None:
    """Download a single image, validate with PIL, resize, and return as dict."""
    try:
        raw_bytes = download_image(url, timeout=15.0)

        # Validate with PIL
        img = Image.open(io.BytesIO(raw_bytes))
        img.verify()

        # Re-open after verify (verify closes the image)
        img = Image.open(io.BytesIO(raw_bytes))

        # Resize if needed
        img.thumbnail(max_size, Image.LANCZOS)

        # Convert to base64
        buf = io.BytesIO()
        fmt = img.format or "PNG"
        mime_type = f"image/{fmt.lower()}"
        if fmt.upper() == "JPEG":
            mime_type = "image/jpeg"
        elif fmt.upper() == "PNG":
            mime_type = "image/png"
        elif fmt.upper() == "WEBP":
            mime_type = "image/webp"
        else:
            # Convert to PNG for unsupported formats
            fmt = "PNG"
            mime_type = "image/png"

        img.save(buf, format=fmt)
        image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return {
            "url": url,
            "image_b64": image_b64,
            "mime_type": mime_type,
        }
    except Exception as exc:
        logger.warning("Failed to download/validate reference image %s: %s", url, exc)
        return None


def _analyze_with_vision(
    images: list[dict],
    subject: str,
    model: str,
    api_key: str,
) -> str:
    """Analyze reference images using GPT-4o vision."""
    client = OpenAI(api_key=api_key)

    # Build multimodal message content
    content: list[dict] = [
        {
            "type": "text",
            "text": f"Subject: {subject}\n\nAnalyze these reference images of the subject for accurate image generation:",
        }
    ]

    for img in images:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{img['mime_type']};base64,{img['image_b64']}",
                "detail": "low",
            },
        })

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": REFERENCE_IMAGE_ANALYSIS_PROMPT},
            {"role": "user", "content": content},
        ],
        max_tokens=600,
        temperature=0.3,
    )

    return response.choices[0].message.content or ""


def ref_images_node(state: ImageAgentState) -> dict:
    """Download reference images, validate, and analyze with GPT-4o vision."""
    settings = get_settings()

    # Feature flag check
    if not settings.ref_images_enabled:
        return {}

    urls = state.get("reference_image_urls") or []
    if not urls:
        return {}

    analysis = state.get("prompt_analysis", {})
    subject = analysis.get("subject", state.get("original_prompt", ""))

    # Filter out low-quality URLs before downloading
    quality_urls = [u for u in urls if _is_quality_url(u)]
    filtered_out = len(urls) - len(quality_urls)
    if filtered_out:
        logger.info("Filtered out %d low-quality reference image URLs", filtered_out)

    # Download top N images in parallel
    max_download = settings.ref_images_max_download
    urls_to_download = quality_urls[:max_download]
    downloaded: list[dict] = []

    with ThreadPoolExecutor(max_workers=max_download) as executor:
        futures = {
            executor.submit(_download_and_validate, url): url
            for url in urls_to_download
        }
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                downloaded.append(result)

    if not downloaded:
        logger.info("No reference images could be downloaded, continuing without.")
        return {}

    logger.info("Downloaded %d reference images", len(downloaded))

    # Analyze with GPT-4o vision
    analysis_text = ""
    try:
        analysis_text = _analyze_with_vision(
            downloaded,
            subject=subject,
            model=settings.ref_image_analysis_model,
            api_key=settings.openai_api_key,
        )
        logger.info("Reference image analysis complete (%d chars)", len(analysis_text))
    except Exception as exc:
        logger.warning("Vision analysis failed, continuing without: %s", exc)

    # Limit images passed forward to model
    max_pass = settings.ref_images_max_pass_to_model
    images_for_model = downloaded[:max_pass]

    log_pipeline_step(
        "Ref Images",
        f"downloaded={len(downloaded)}  analyzed={len(images_for_model)}"
        f"  filtered_out={filtered_out}"
        f'  "{(analysis_text or "")[:50]}..."',
    )
    return {
        "reference_images": images_for_model,
        "reference_image_analysis": analysis_text if analysis_text else None,
    }
