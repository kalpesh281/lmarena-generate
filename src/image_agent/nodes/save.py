"""Save node: persist image and metadata to disk."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from image_agent.config import get_settings
from image_agent.providers.image_utils import base64_to_image, save_image
from image_agent.state import ImageAgentState
from image_agent.utils.logger import log_pipeline_step


def save_node(state: ImageAgentState) -> dict:
    """Save the generated image and a JSON metadata sidecar."""
    settings = get_settings()
    metadata = state.get("generation_metadata") or {}

    image_b64 = metadata.get("image_b64")
    if not image_b64:
        return {"error": "No image data to save."}

    image_bytes = base64_to_image(image_b64)
    image_id = uuid.uuid4().hex[:12]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{image_id}.png"

    output_dir = settings.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    image_path = save_image(image_bytes, output_dir / filename)

    # Build metadata sidecar (strip the large b64 blob)
    # Collect reference image URLs (just URLs, not the b64 data)
    ref_urls = []
    for ref in (state.get("reference_images") or []):
        if ref.get("url"):
            ref_urls.append(ref["url"])

    sidecar = {
        "image_id": image_id,
        "timestamp": timestamp,
        "original_prompt": state.get("original_prompt", ""),
        "enhanced_prompt": state.get("enhanced_prompt"),
        "action": state.get("action"),
        "provider": metadata.get("provider"),
        "model": metadata.get("model"),
        "params": metadata.get("params"),
        "prompt_analysis": state.get("prompt_analysis"),
        "research_context": state.get("research_context"),
        "reference_image_urls": ref_urls if ref_urls else None,
        "reference_image_analysis": state.get("reference_image_analysis"),
        "image_path": str(image_path),
    }
    sidecar_path = output_dir / f"{timestamp}_{image_id}.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=2, default=str))

    # Remove b64 from metadata flowing forward
    clean_metadata = {k: v for k, v in metadata.items() if k != "image_b64"}

    log_pipeline_step("Save", f"{image_path}")
    return {
        "image_path": str(image_path),
        "image_id": image_id,
        "generation_metadata": clean_metadata,
    }
