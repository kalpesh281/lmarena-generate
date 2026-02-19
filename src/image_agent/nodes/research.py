"""Research node: searches the internet for visual context via Tavily."""

from __future__ import annotations

from tavily import TavilyClient
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from image_agent.config import get_settings
from image_agent.prompts.templates import RESEARCH_SYNTHESIS_PROMPT
from image_agent.state import ImageAgentState
from image_agent.utils.logger import log_pipeline_step

# Domains that return low-quality reference images (AI-generated, stock vectors,
# thumbnails, social media crops). Passed to Tavily's exclude_domains parameter.
_EXCLUDED_DOMAINS = [
    # AI image generators
    "stablediffusionweb.com",
    "lexica.art",
    "civitai.com",
    "pixabay.com",
    "playground.ai",
    "midjourney.com",
    "unsplash.com",
    "nightcafe.studio",
    "craiyon.com",
    # Stock / vector sites
    "shutterstock.com",
    "istockphoto.com",
    "gettyimages.com",
    "depositphotos.com",
    "123rf.com",
    "dreamstime.com",
    "vecteezy.com",
    "freepik.com",
    "vectorstock.com",
    "pngtree.com",
    "pngwing.com",
    "cleanpng.com",
    "pngitem.com",
    "stocksnap.io",
    "i.ytimg.com",
    "twitter.com",
    "youtube.com",
    "instagram.com",
    "pinterest.com",
]


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
    canonical_results: dict | None = None,
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

    if canonical_results:
        sections.append("\n=== Canonical Reference ===")
        for item in canonical_results.get("results", []):
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

    # Run searches for comprehensive context (include images for reference)
    # All searches exclude low-quality domains (stock vectors, AI generators, etc.)
    style_results = tavily.search(
        f"{subject} {style} art visual style reference",
        max_results=max_results,
        include_images=True,
        search_depth="advanced",
        exclude_domains=_EXCLUDED_DOMAINS,
    )
    factual_results = tavily.search(
        f"{subject} details characteristics appearance",
        max_results=max_results,
        include_images=True,
        exclude_domains=_EXCLUDED_DOMAINS,
    )
    trending_results = tavily.search(
        f"AI art {style} techniques trending 2025",
        max_results=min(max_results, 2),
        exclude_domains=_EXCLUDED_DOMAINS,
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
            exclude_domains=_EXCLUDED_DOMAINS,
        )

    # 5th search: canonical reference images (wiki, fandom, official art)
    _CANONICAL_QUERIES = {
        "fictional_character": '"{subject}" character official art wiki fandom',
        "cultural": '"{subject}" traditional art iconography wiki',
        "real_person": '"{subject}" photo portrait official',
        "landmark": '"{subject}" photo high resolution',
    }
    canonical_query = _CANONICAL_QUERIES.get(
        subject_type,
        '"{subject}" reference image high quality',
    ).format(subject=subject)

    canonical_results = tavily.search(
        canonical_query,
        max_results=max_results,
        include_images=True,
        search_depth="advanced",
        exclude_domains=_EXCLUDED_DOMAINS,
    )

    # Synthesize with GPT-4o
    raw_context = _format_search_results(
        style_results,
        factual_results,
        trending_results,
        composition_results,
        canonical_results,
    )
    llm = ChatOpenAI(
        model=settings.enhance_model,
        api_key=settings.openai_api_key,
        temperature=0.3,
    )
    original_prompt = state["original_prompt"]
    synthesis = llm.invoke(
        [
            SystemMessage(content=RESEARCH_SYNTHESIS_PROMPT),
            HumanMessage(
                content=(
                    f"Original prompt: {original_prompt}\n"
                    f"Subject: {subject}\nStyle: {style}\n\n"
                    f"Search Results:\n{raw_context}"
                )
            ),
        ]
    )

    # Extract image URLs from search results.
    # Canonical results are prepended so they get download priority.
    image_urls: list[str] = []
    canonical_image_count = 0
    for url in canonical_results.get("images", []):
        if isinstance(url, str) and url not in image_urls:
            image_urls.append(url)
            canonical_image_count += 1

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
        f"  image_urls={len(image_urls)}"
        f"  canonical_images={canonical_image_count}",
    )
    return result
