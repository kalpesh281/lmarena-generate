"""Pipeline trace logger — prints concise Rich panels for each pipeline stage."""

from __future__ import annotations

import os

from rich.console import Console
from rich.text import Text

_console = Console(stderr=True)

# Stage → color mapping
_STAGE_COLORS: dict[str, str] = {
    "Router": "cyan",
    "Research": "green",
    "Ref Images": "yellow",
    "Suggest": "magenta",
    "Enhance": "blue",
    "Provider": "bright_cyan",
    "Generate": "bright_green",
    "Save": "bright_white",
}

_STAGE_WIDTH = 12  # fixed width for alignment


def _is_enabled() -> bool:
    """Check if pipeline logging is enabled via settings or env var."""
    # Env var override takes precedence
    env = os.environ.get("PIPELINE_LOGGING")
    if env is not None:
        return env.lower() not in ("0", "false", "no", "off")

    try:
        from image_agent.config import get_settings
        return get_settings().pipeline_logging
    except Exception:
        return True


def log_pipeline_step(stage: str, details: str) -> None:
    """Print a single trace line for a pipeline stage.

    Args:
        stage: Stage name (e.g. "Router", "Research").
        details: Concise key=value summary string.
    """
    if not _is_enabled():
        return

    color = _STAGE_COLORS.get(stage, "white")
    label = Text(f" {stage:<{_STAGE_WIDTH}}", style=f"bold {color}")
    body = Text(details, style="dim")

    _console.print(label + body)
