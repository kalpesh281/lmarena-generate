"""JSON-based generation history stored alongside images."""

from __future__ import annotations

import json
from pathlib import Path

from image_agent.config import get_settings


def list_history(limit: int = 20) -> list[dict]:
    """Return recent generation records, newest first."""
    output_dir = get_settings().output_dir
    if not output_dir.exists():
        return []

    json_files = sorted(output_dir.glob("*.json"), reverse=True)
    records = []
    for f in json_files[:limit]:
        try:
            records.append(json.loads(f.read_text()))
        except (json.JSONDecodeError, OSError):
            continue
    return records


def get_record(image_id: str) -> dict | None:
    """Look up a single generation record by image_id."""
    output_dir = get_settings().output_dir
    if not output_dir.exists():
        return None

    for f in output_dir.glob("*.json"):
        try:
            data = json.loads(f.read_text())
            if data.get("image_id") == image_id:
                return data
        except (json.JSONDecodeError, OSError):
            continue
    return None


def clear_history() -> int:
    """Delete all generated images and metadata. Returns count deleted."""
    output_dir = get_settings().output_dir
    if not output_dir.exists():
        return 0

    count = 0
    for f in output_dir.iterdir():
        if f.name == ".gitkeep":
            continue
        f.unlink(missing_ok=True)
        count += 1
    return count
