"""Enhance node: creates a detailed image prompt using research context."""

from __future__ import annotations

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from image_agent.config import get_settings
from image_agent.prompts.templates import ENHANCE_SYSTEM_PROMPT
from image_agent.state import ImageAgentState


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

    user_msg = f"""Original prompt: {prompt}

Research context:
{research.get('synthesized', 'No research available')}
{suggestion_block}

Enhance this into a detailed image prompt using the research context above."""

    response = llm.invoke([
        SystemMessage(content=ENHANCE_SYSTEM_PROMPT),
        HumanMessage(content=user_msg),
    ])

    return {"enhanced_prompt": response.content}
