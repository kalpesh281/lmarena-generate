"""Image utility functions: download, base64 encode/decode, resize."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import httpx
from PIL import Image


def download_image(url: str, timeout: float = 60.0) -> bytes:
    """Download an image from a URL and return raw bytes."""
    resp = httpx.get(url, timeout=timeout, follow_redirects=True)
    resp.raise_for_status()
    return resp.content


def image_to_base64(image_bytes: bytes) -> str:
    """Encode raw image bytes to a base64 string."""
    return base64.b64encode(image_bytes).decode("utf-8")


def base64_to_image(b64_string: str) -> bytes:
    """Decode a base64 string back to raw image bytes."""
    return base64.b64decode(b64_string)


def save_image(image_bytes: bytes, path: Path) -> Path:
    """Save raw image bytes to a file. Returns the resolved path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(image_bytes)
    return path.resolve()


def resize_image(image_bytes: bytes, max_size: tuple[int, int] = (1024, 1024)) -> bytes:
    """Resize an image to fit within max_size, preserving aspect ratio."""
    img = Image.open(io.BytesIO(image_bytes))
    img.thumbnail(max_size, Image.LANCZOS)
    buf = io.BytesIO()
    fmt = img.format or "PNG"
    img.save(buf, format=fmt)
    return buf.getvalue()


def load_image_as_base64(path: str | Path) -> str:
    """Load an image from disk and return its base64 representation."""
    return image_to_base64(Path(path).read_bytes())
