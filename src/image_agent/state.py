"""State schema for the image generation agent graph."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated


class GenerationParams(TypedDict, total=False):
    """Parameters passed to the image generation provider."""

    size: str
    quality: str
    style: str
    n: int


class ImageAgentState(TypedDict, total=False):
    """Full state flowing through the image generation graph."""

    messages: Annotated[list[BaseMessage], add_messages]

    # User input
    original_prompt: str
    source_image_path: str | None
    last_image_path: str | None
    last_prompt: str | None  # description of what the last image contained

    # Router output
    action: Literal["generate", "edit", "enhance_only"]
    prompt_analysis: dict[str, Any]  # {style, mood, subject, subject_type, complexity}

    # Research output
    research_context: dict[str, Any]  # {synthesized, style_refs, factual_context, trending_techniques}
    reference_image_urls: list[str] | None  # URLs extracted from Tavily image results

    # Reference image analysis output
    reference_images: list[dict] | None  # Downloaded images: [{url, image_b64, mime_type}]
    reference_image_analysis: str | None  # GPT-4o vision description of reference images

    # Enhancement output
    enhanced_prompt: str | None

    # Provider selection
    provider: Literal["openai", "flux", "gemini"]
    generation_params: GenerationParams

    # Creative suggestions (chat mode)
    suggestions: list[dict] | None  # 3 suggestion dicts from suggest node
    selected_suggestion: str | None  # User's choice (formatted text) or None
    skip_suggestions: bool  # True for one-shot commands
    suggestion_phase_complete: bool  # True after Phase 1 (set by CLI before Phase 2)

    # Output
    image_path: str | None
    image_id: str
    generation_metadata: dict[str, Any]
    error: str | None
    retry_count: int
