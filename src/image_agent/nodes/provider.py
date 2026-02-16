"""Provider selection node: deterministic style-to-provider mapping."""

from __future__ import annotations

from image_agent.state import ImageAgentState

# Styles routed to Flux (photorealistic strengths)
FLUX_STYLES = {
    "photorealistic",
    "photography",
    "photo",
    "realistic",
    "cinematic",
    "portrait",
    "landscape",
    "architectural",
    "product",
    "fashion",
    "documentary",
    "street",
}

# Everything else goes to OpenAI (artistic strengths)
# Explicitly listed for clarity, but the default is OpenAI
OPENAI_STYLES = {
    "anime",
    "cartoon",
    "illustration",
    "digital-art",
    "oil-painting",
    "watercolor",
    "pencil-sketch",
    "3d-render",
    "pixel-art",
    "comic",
    "fantasy",
    "abstract",
    "surreal",
    "pop-art",
    "concept-art",
}


def provider_select_node(state: ImageAgentState) -> dict:
    """Select the image generation provider based on detected style."""
    analysis = state.get("prompt_analysis", {})
    style = analysis.get("style", "").lower().strip()

    if style in FLUX_STYLES:
        provider = "flux"
    else:
        provider = "openai"

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
