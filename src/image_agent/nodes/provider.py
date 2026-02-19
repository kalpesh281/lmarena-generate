"""Provider selection node: deterministic style-to-provider mapping."""

from __future__ import annotations

from image_agent.state import ImageAgentState
from image_agent.utils.logger import log_pipeline_step

# Styles routed to Flux only when explicitly requested via --provider flux
FLUX_STYLES: set[str] = set()

# OpenAI only used when explicitly requested via --provider openai (or for edits)
OPENAI_STYLES: set[str] = set()

# Ideal social-media-friendly sizes per orientation
ORIENTATION_SIZES: dict[str, str] = {
    "landscape": "1200x630",
    "portrait": "1080x1350",
    "square": "1080x1080",
}


def provider_select_node(state: ImageAgentState) -> dict:
    """Select the image generation provider based on detected style.

    Priority: explicit override > style mapping > gemini (default).
    """
    analysis = state.get("prompt_analysis", {})

    # Resolve orientation → ideal size (CLI --size override takes precedence)
    existing_params = state.get("generation_params") or {}
    if existing_params.get("size"):
        size = existing_params["size"]
    else:
        orientation = analysis.get("orientation", "square")
        size = ORIENTATION_SIZES.get(orientation, ORIENTATION_SIZES["square"])

    # Honour explicit provider override (e.g. --provider openai)
    explicit = state.get("provider")
    if explicit:
        log_pipeline_step("Provider", f"{explicit} ({size}) [explicit]")
        return {
            "provider": explicit,
            "generation_params": {"size": size, "quality": "high", "n": 1},
        }

    style = analysis.get("style", "").lower().strip()

    if style in FLUX_STYLES:
        provider = "flux"
    elif style in OPENAI_STYLES:
        provider = "openai"
    else:
        provider = "gemini"

    generation_params = {
        "size": size,
        "quality": "high",
        "n": 1,
    }

    log_pipeline_step("Provider", f"{provider} ({size})")
    return {
        "provider": provider,
        "generation_params": generation_params,
    }
