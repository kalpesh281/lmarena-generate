"""Provider selection node: deterministic style-to-provider mapping."""

from __future__ import annotations

from image_agent.state import ImageAgentState

# Styles routed to Flux only when explicitly requested via --provider flux
FLUX_STYLES: set[str] = set()

# OpenAI only used when explicitly requested via --provider openai (or for edits)
OPENAI_STYLES: set[str] = set()


def provider_select_node(state: ImageAgentState) -> dict:
    """Select the image generation provider based on detected style.

    Priority: explicit override > style mapping > gemini (default).
    """
    # Honour explicit provider override (e.g. --provider openai)
    explicit = state.get("provider")
    if explicit:
        return {
            "provider": explicit,
            "generation_params": {"size": "1024x1024", "quality": "high", "n": 1},
        }

    analysis = state.get("prompt_analysis", {})
    style = analysis.get("style", "").lower().strip()

    if style in FLUX_STYLES:
        provider = "flux"
    elif style in OPENAI_STYLES:
        provider = "openai"
    else:
        provider = "gemini"

    # Default generation params
    generation_params = {
        "size": "1024x1024",
        "quality": "high",
        "n": 1,
    }

    return {
        "provider": provider,
        "generation_params": generation_params,
    }
