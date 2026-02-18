"""Suggest node: generates creative direction suggestions using GPT-4o."""

from __future__ import annotations

import json

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from image_agent.config import get_settings
from image_agent.prompts.templates import SUGGEST_SYSTEM_PROMPT
from image_agent.state import ImageAgentState
from image_agent.utils.logger import log_pipeline_step


def suggest_node(state: ImageAgentState) -> dict:
    """Generate 3 creative direction suggestions based on prompt and research."""
    settings = get_settings()
    llm = ChatOpenAI(
        model=settings.enhance_model,
        api_key=settings.openai_api_key,
        temperature=0.9,
    )

    prompt = state["original_prompt"]
    analysis = state.get("prompt_analysis", {})
    research = state.get("research_context", {})

    # Include visual analysis if available
    visual_block = ""
    ref_analysis = state.get("reference_image_analysis")
    if ref_analysis:
        visual_block = f"""
Visual analysis from reference images:
{ref_analysis}
"""

    user_msg = f"""Original prompt: {prompt}

Prompt analysis:
- Style: {analysis.get('style', 'not specified')}
- Mood: {analysis.get('mood', 'not specified')}
- Subject: {analysis.get('subject', 'not specified')}
- Complexity: {analysis.get('complexity', 'moderate')}

Research context:
{research.get('synthesized', 'No research available')}
{visual_block}
Generate 3 distinct creative directions for this image."""

    response = llm.invoke([
        SystemMessage(content=SUGGEST_SYSTEM_PROMPT),
        HumanMessage(content=user_msg),
    ])

    try:
        data = json.loads(response.content)
        suggestions = data.get("suggestions", [])
        if not suggestions or not isinstance(suggestions, list):
            raise ValueError("No suggestions in response")
    except (json.JSONDecodeError, ValueError):
        # Fallback: return a single generic suggestion that will be auto-selected
        suggestions = [
            {
                "number": 1,
                "title": "Enhanced Default",
                "description": f"A polished, high-quality rendering of: {prompt}",
                "style": analysis.get("style", "photorealistic"),
                "mood": analysis.get("mood", "neutral"),
                "key_elements": [analysis.get("subject", prompt)],
            }
        ]

    titles = [f"{i+1}: {s.get('title', '?')}" for i, s in enumerate(suggestions[:3])]
    log_pipeline_step("Suggest", "  ".join(titles))
    return {"suggestions": suggestions}
