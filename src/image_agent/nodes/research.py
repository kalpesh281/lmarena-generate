"""Research node: searches the internet for visual context via Tavily."""

from __future__ import annotations

from tavily import TavilyClient
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from image_agent.config import get_settings
from image_agent.prompts.templates import RESEARCH_SYNTHESIS_PROMPT
from image_agent.state import ImageAgentState


def _extract_key_points(results: dict) -> list[str]:
    """Pull the most useful snippets from Tavily search results."""
    points: list[str] = []
    for item in results.get("results", []):
        content = item.get("content", "").strip()
        if content:
            points.append(content[:300])
    return points


def _format_search_results(
    style_results: dict,
    factual_results: dict,
    trending_results: dict,
) -> str:
    """Combine raw search results into a text block for synthesis."""
    sections = []

    sections.append("=== Style References ===")
    for item in style_results.get("results", []):
        sections.append(f"- {item.get('content', '')[:200]}")

    sections.append("\n=== Factual Context ===")
    for item in factual_results.get("results", []):
        sections.append(f"- {item.get('content', '')[:200]}")

    sections.append("\n=== Trending Techniques ===")
    for item in trending_results.get("results", []):
        sections.append(f"- {item.get('content', '')[:200]}")

    return "\n".join(sections)


def research_node(state: ImageAgentState) -> dict:
    """Search the internet for context to enrich image generation."""
    settings = get_settings()
    analysis = state["prompt_analysis"]
    subject = analysis.get("subject", state["original_prompt"])
    style = analysis.get("style", "photorealistic")

    tavily = TavilyClient(api_key=settings.tavily_api_key)
    max_results = settings.tavily_max_results

    # Run 3 searches for comprehensive context
    style_results = tavily.search(
        f"{subject} {style} art visual style reference",
        max_results=max_results,
    )
    factual_results = tavily.search(
        f"{subject} details characteristics appearance",
        max_results=max_results,
    )
    trending_results = tavily.search(
        f"AI art {style} techniques trending 2025",
        max_results=min(max_results, 2),
    )

    # Synthesize with GPT-4o
    raw_context = _format_search_results(style_results, factual_results, trending_results)
    llm = ChatOpenAI(
        model=settings.enhance_model,
        api_key=settings.openai_api_key,
        temperature=0.3,
    )
    synthesis = llm.invoke([
        SystemMessage(content=RESEARCH_SYNTHESIS_PROMPT),
        HumanMessage(
            content=f"Subject: {subject}\nStyle: {style}\n\nSearch Results:\n{raw_context}"
        ),
    ])

    return {
        "research_context": {
            "synthesized": synthesis.content,
            "style_refs": _extract_key_points(style_results),
            "factual_context": _extract_key_points(factual_results),
            "trending_techniques": _extract_key_points(trending_results),
        }
    }
