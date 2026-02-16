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

    # Router output
    action: Literal["generate", "edit", "enhance_only"]
    prompt_analysis: dict[str, Any]  # {style, mood, subject, complexity}

    # Research output
    research_context: dict[str, Any]  # {synthesized, style_refs, factual_context, trending_techniques}

    # Enhancement output
    enhanced_prompt: str | None

    # Provider selection
    provider: Literal["openai", "flux"]
    generation_params: GenerationParams

    # Output
    image_path: str | None
    image_id: str
    generation_metadata: dict[str, Any]
    error: str | None
    retry_count: int
