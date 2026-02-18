"""LangGraph StateGraph definition and compilation."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from image_agent.state import ImageAgentState
from image_agent.nodes.router import router_node
from image_agent.nodes.research import research_node
from image_agent.nodes.ref_images import ref_images_node
from image_agent.nodes.enhance import enhance_node
from image_agent.nodes.suggest import suggest_node
from image_agent.nodes.provider import provider_select_node
from image_agent.nodes.generate import openai_generate_node, flux_generate_node, gemini_generate_node
from image_agent.nodes.edit import edit_node
from image_agent.nodes.save import save_node
from image_agent.nodes.response import response_node


def _route_from_start(state: ImageAgentState) -> str:
    """Route at graph entry: Phase 2 skips straight to enhance."""
    if state.get("suggestion_phase_complete") and state.get("research_context"):
        return "enhance"
    return "router"


def _route_after_router(state: ImageAgentState) -> str:
    """Route based on the classified action."""
    action = state.get("action", "generate")
    if action == "edit":
        return "edit"
    if action == "enhance_only":
        return "research"
    return "research"


def _route_after_ref_images(state: ImageAgentState) -> str:
    """After ref images, decide whether to suggest or skip to enhance."""
    if state.get("skip_suggestions") or state.get("action") == "enhance_only":
        return "enhance"
    return "suggest"


def _route_after_enhance(state: ImageAgentState) -> str:
    """After enhancement, decide whether to generate or just return."""
    if state.get("action") == "enhance_only":
        return "response"
    return "provider_select"


def _route_provider(state: ImageAgentState) -> str:
    """Route to the correct image provider node."""
    provider = state.get("provider", "gemini")
    if provider == "flux":
        return "flux_generate"
    if provider == "openai":
        return "openai_generate"
    return "gemini_generate"


def build_graph() -> StateGraph:
    """Construct the image agent state graph."""
    graph = StateGraph(ImageAgentState)

    # Add all nodes
    graph.add_node("router", router_node)
    graph.add_node("research", research_node)
    graph.add_node("ref_images", ref_images_node)
    graph.add_node("suggest", suggest_node)
    graph.add_node("enhance", enhance_node)
    graph.add_node("provider_select", provider_select_node)
    graph.add_node("openai_generate", openai_generate_node)
    graph.add_node("flux_generate", flux_generate_node)
    graph.add_node("gemini_generate", gemini_generate_node)
    graph.add_node("edit", edit_node)
    graph.add_node("save", save_node)
    graph.add_node("response", response_node)

    # START → conditional: Phase 2 re-entry or Phase 1
    graph.add_conditional_edges(START, _route_from_start, {
        "enhance": "enhance",
        "router": "router",
    })

    # Router → conditional branch
    graph.add_conditional_edges("router", _route_after_router, {
        "research": "research",
        "edit": "edit",
    })

    # Research → ref_images (always)
    graph.add_edge("research", "ref_images")

    # Ref images → conditional: suggest or skip to enhance
    graph.add_conditional_edges("ref_images", _route_after_ref_images, {
        "suggest": "suggest",
        "enhance": "enhance",
    })

    # Suggest → END (Phase 1 terminates so CLI can show suggestions)
    graph.add_edge("suggest", END)

    # Enhance → conditional (generate or response-only)
    graph.add_conditional_edges("enhance", _route_after_enhance, {
        "provider_select": "provider_select",
        "response": "response",
    })

    # Provider select → conditional provider
    graph.add_conditional_edges("provider_select", _route_provider, {
        "openai_generate": "openai_generate",
        "flux_generate": "flux_generate",
        "gemini_generate": "gemini_generate",
    })

    # Both generators → save → response → END
    graph.add_edge("openai_generate", "save")
    graph.add_edge("flux_generate", "save")
    graph.add_edge("gemini_generate", "save")
    graph.add_edge("edit", "save")
    graph.add_edge("save", "response")
    graph.add_edge("response", END)

    return graph


def compile_graph(checkpointer=None):
    """Build and compile the graph, ready to invoke."""
    if checkpointer is None:
        checkpointer = MemorySaver()
    return build_graph().compile(checkpointer=checkpointer)
