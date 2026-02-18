"""Research node: searches the internet for visual context via Tavily."""

from __future__ import annotations

from tavily import TavilyClient
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from image_agent.config import get_settings
from image_agent.prompts.templates import RESEARCH_SYNTHESIS_PROMPT
from image_agent.state import ImageAgentState
from image_agent.utils.logger import log_pipeline_step


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
    composition_results: dict | None = None,
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

    if composition_results:
        sections.append("\n=== Scene Composition & Spatial Layout ===")
        for item in composition_results.get("results", []):
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

    # Run 3 searches for comprehensive context (include images for reference)
    style_results = tavily.search(
        f"{subject} {style} art visual style reference",
        max_results=max_results,
        include_images=True,
    )
    factual_results = tavily.search(
        f"{subject} details characteristics appearance",
        max_results=max_results,
        include_images=True,
    )
    trending_results = tavily.search(
        f"AI art {style} techniques trending 2025",
        max_results=min(max_results, 2),
    )

    # 4th search: scene composition (for moderate/complex or cultural/scene subjects)
    complexity = analysis.get("complexity", "simple")
    subject_type = analysis.get("subject_type", "")
    composition_results = None
    if complexity in ("moderate", "complex") or subject_type in ("cultural", "scene"):
        composition_results = tavily.search(
            f"{subject} scene description composition layout spatial arrangement",
            max_results=max_results,
            include_images=True,
        )

    # Synthesize with GPT-4o
    raw_context = _format_search_results(
        style_results, factual_results, trending_results, composition_results
    )
    llm = ChatOpenAI(
        model=settings.enhance_model,
        api_key=settings.openai_api_key,
        temperature=0.3,
    )
    original_prompt = state["original_prompt"]
    synthesis = llm.invoke([
        SystemMessage(content=RESEARCH_SYNTHESIS_PROMPT),
        HumanMessage(
            content=(
                f"Original prompt: {original_prompt}\n"
                f"Subject: {subject}\nStyle: {style}\n\n"
                f"Search Results:\n{raw_context}"
            )
        ),
    ])

    # Extract image URLs from search results
    image_urls: list[str] = []
    search_pools = [style_results, factual_results]
    if composition_results:
        search_pools.append(composition_results)
    for results in search_pools:
        for url in results.get("images", []):
            if isinstance(url, str) and url not in image_urls:
                image_urls.append(url)

    composition_points = (
        _extract_key_points(composition_results) if composition_results else []
    )
    result = {
        "research_context": {
            "synthesized": synthesis.content,
            "style_refs": _extract_key_points(style_results),
            "factual_context": _extract_key_points(factual_results),
            "trending_techniques": _extract_key_points(trending_results),
            "composition_context": composition_points,
        },
        "reference_image_urls": image_urls if image_urls else None,
    }
    log_pipeline_step(
        "Research",
        f"style_refs={len(result['research_context']['style_refs'])}"
        f"  factual={len(result['research_context']['factual_context'])}"
        f"  trending={len(result['research_context']['trending_techniques'])}"
        f"  composition={len(composition_points)}"
        f"  image_urls={len(image_urls)}",
    )
    return result
