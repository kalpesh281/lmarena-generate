"""Router node: classifies intent and analyzes the prompt."""

from __future__ import annotations

import json

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from image_agent.config import get_settings
from image_agent.prompts.templates import ROUTER_SYSTEM_PROMPT
from image_agent.state import ImageAgentState


def router_node(state: ImageAgentState) -> dict:
    """Classify the user prompt into action + style/mood/subject analysis."""
    settings = get_settings()
    llm = ChatOpenAI(
        model=settings.router_model,
        api_key=settings.openai_api_key,
        temperature=0,
    )

    prompt = state["original_prompt"]
    last_image = state.get("last_image_path")

    last_prompt = state.get("last_prompt")

    system_prompt = ROUTER_SYSTEM_PROMPT
    if last_image and last_prompt:
        system_prompt += (
            f"\n\nIMPORTANT: A previous image exists from the conversation. "
            f'The previous image was: "{last_prompt}"\n'
            "If the user's request refers to the previous image using pronouns "
            '(e.g. "he", "she", "it", "make it darker", "add clouds", "change the colors") '
            "or asks to modify/adjust/tweak it, use action \"edit\".\n"
            "If the user is requesting an entirely new, unrelated image subject, use action \"generate\"."
        )

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt),
    ])

    try:
        analysis = json.loads(response.content)
    except json.JSONDecodeError:
        analysis = {
            "action": "generate",
            "style": "photorealistic",
            "mood": "neutral",
            "subject": prompt,
            "complexity": "moderate",
        }

    action = analysis.pop("action", "generate")

    result: dict = {
        "action": action,
        "prompt_analysis": analysis,
    }

    if action == "edit" and last_image and not state.get("source_image_path"):
        result["source_image_path"] = last_image

    return result
