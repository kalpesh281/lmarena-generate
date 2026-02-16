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
    response = llm.invoke([
        SystemMessage(content=ROUTER_SYSTEM_PROMPT),
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
    return {
        "action": action,
        "prompt_analysis": analysis,
    }
