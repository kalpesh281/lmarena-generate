"""LangGraph StateGraph definition and compilation."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from image_agent.state import ImageAgentState
from image_agent.nodes.router import router_node
from image_agent.nodes.research import research_node
from image_agent.nodes.enhance import enhance_node
from image_agent.nodes.provider import provider_select_node
from image_agent.nodes.generate import openai_generate_node, flux_generate_node
from image_agent.nodes.edit import edit_node
from image_agent.nodes.save import save_node
from image_agent.nodes.response import response_node


def _route_after_router(state: ImageAgentState) -> str:
    """Route based on the classified action."""
    action = state.get("action", "generate")
    if action == "edit":
        return "edit"
    if action == "enhance_only":
        return "research"
    return "research"


def _route_after_enhance(state: ImageAgentState) -> str:
    """After enhancement, decide whether to generate or just return."""
    if state.get("action") == "enhance_only":
        return "response"
    return "provider_select"


def _route_provider(state: ImageAgentState) -> str:
    """Route to the correct image provider node."""
    provider = state.get("provider", "openai")
    if provider == "flux":
        return "flux_generate"
    return "openai_generate"


def build_graph() -> StateGraph:
    """Construct the image agent state graph."""
    graph = StateGraph(ImageAgentState)

    # Add all nodes
    graph.add_node("router", router_node)
    graph.add_node("research", research_node)
    graph.add_node("enhance", enhance_node)
    graph.add_node("provider_select", provider_select_node)
    graph.add_node("openai_generate", openai_generate_node)
    graph.add_node("flux_generate", flux_generate_node)
    graph.add_node("edit", edit_node)
    graph.add_node("save", save_node)
    graph.add_node("response", response_node)

    # Edges: START → router
    graph.add_edge(START, "router")

    # Router → conditional branch
    graph.add_conditional_edges("router", _route_after_router, {
        "research": "research",
        "edit": "edit",
    })

    # Research → enhance
    graph.add_edge("research", "enhance")

    # Enhance → conditional (generate or response-only)
    graph.add_conditional_edges("enhance", _route_after_enhance, {
        "provider_select": "provider_select",
        "response": "response",
    })

    # Provider select → conditional provider
    graph.add_conditional_edges("provider_select", _route_provider, {
        "openai_generate": "openai_generate",
        "flux_generate": "flux_generate",
    })

    # Both generators → save → response → END
    graph.add_edge("openai_generate", "save")
    graph.add_edge("flux_generate", "save")
    graph.add_edge("edit", "save")
    graph.add_edge("save", "response")
    graph.add_edge("response", END)

    return graph


def compile_graph(checkpointer=None):
    """Build and compile the graph, ready to invoke."""
    if checkpointer is None:
        checkpointer = MemorySaver()
    return build_graph().compile(checkpointer=checkpointer)
