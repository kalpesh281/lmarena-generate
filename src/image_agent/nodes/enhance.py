"""Enhance node: creates a detailed image prompt using research context."""

from __future__ import annotations

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from image_agent.config import get_settings
from image_agent.prompts.templates import ENHANCE_SYSTEM_PROMPT
from image_agent.state import ImageAgentState
from image_agent.utils.logger import log_pipeline_step


def enhance_node(state: ImageAgentState) -> dict:
    """Enhance the user prompt using research context from the internet."""
    settings = get_settings()
    llm = ChatOpenAI(
        model=settings.enhance_model,
        api_key=settings.openai_api_key,
        temperature=0.7,
    )

    research = state.get("research_context", {})
    prompt = state["original_prompt"]
    selected_suggestion = state.get("selected_suggestion")

    suggestion_block = ""
    if selected_suggestion:
        suggestion_block = f"""
Selected creative direction:
{selected_suggestion}

Follow this creative direction closely — match its style, mood, and key elements."""

    # Include visual analysis from reference images (takes priority for appearance details)
    visual_analysis_block = ""
    ref_analysis = state.get("reference_image_analysis")
    if ref_analysis:
        visual_analysis_block = f"""

Visual analysis from reference images (HIGH PRIORITY for character appearance, clothing, colors):
{ref_analysis}

Use these visual details as the primary source for how characters and subjects should look.
Text research provides context, but visual analysis should take priority for appearance, \
clothing, colors, and iconographic details."""

    user_msg = f"""\
=== WHAT TO SHOW (Compositional Blueprint — preserve ALL elements) ===
Original prompt: {prompt}
First, mentally list every key noun from the original prompt. Every one MUST appear as a \
visible element in your output — not just as a modifier of another noun.

=== HOW IT SHOULD LOOK (Visual Details & Style) ===
Research context:
{research.get('synthesized', 'No research available')}
{visual_analysis_block}
{suggestion_block}

Enhance this into a detailed image prompt. The original prompt defines WHAT must be in the scene. \
Research and visual analysis define HOW it should look. Do not drop any scene elements."""

    response = llm.invoke([
        SystemMessage(content=ENHANCE_SYSTEM_PROMPT),
        HumanMessage(content=user_msg),
    ])

    log_pipeline_step(
        "Enhance",
        f'"{(response.content or "")[:150]}..."'
        f"  visual_analysis={'yes' if ref_analysis else 'no'}",
    )
    return {"enhanced_prompt": response.content}
