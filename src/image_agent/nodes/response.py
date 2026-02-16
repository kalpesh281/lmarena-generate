"""Response node: format the final output for the user."""

from __future__ import annotations

from image_agent.state import ImageAgentState


def response_node(state: ImageAgentState) -> dict:
    """Build a human-readable response summarizing the generation."""
    error = state.get("error")
    if error:
        return {
            "messages": [{"role": "assistant", "content": f"Error: {error}"}],
        }

    image_path = state.get("image_path", "unknown")
    action = state.get("action", "generate")
    provider = state.get("generation_metadata", {}).get("provider", "unknown")
    enhanced = state.get("enhanced_prompt", "")
    research = state.get("research_context", {})
    synthesis = research.get("synthesized", "")

    parts = [f"Image saved to: {image_path}"]
    parts.append(f"Provider: {provider}")
    parts.append(f"Action: {action}")

    if enhanced:
        parts.append(f"\nEnhanced prompt:\n{enhanced}")

    if synthesis:
        # Truncate research summary for display
        preview = synthesis[:500] + ("..." if len(synthesis) > 500 else "")
        parts.append(f"\nResearch summary:\n{preview}")

    content = "\n".join(parts)
    return {
        "messages": [{"role": "assistant", "content": content}],
    }
