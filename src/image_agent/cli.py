"""Typer CLI for the image generation agent."""

from __future__ import annotations

import uuid
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from image_agent.graph import compile_graph
from image_agent.history import list_history, clear_history

app = typer.Typer(
    name="image-agent",
    help="GenAI Image Generation Agent powered by LangGraph",
    no_args_is_help=True,
)
console = Console()


@app.command()
def generate(
    prompt: str = typer.Argument(..., help="Image generation prompt"),
    provider: Optional[str] = typer.Option(None, help="Force provider: gemini, openai, or flux"),
    size: str = typer.Option("1024x1024", help="Image size"),
):
    """Generate an image from a text prompt with internet research."""
    console.print(Panel(f"[bold]Prompt:[/bold] {prompt}", title="Image Agent"))
    console.print("[dim]Routing → Researching → Enhancing → Generating...[/dim]\n")

    graph = compile_graph()
    initial_state = {"original_prompt": prompt, "skip_suggestions": True}
    if provider:
        initial_state["provider"] = provider
    if size != "1024x1024":
        initial_state["generation_params"] = {"size": size}
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    with console.status("[bold green]Working..."):
        result = graph.invoke(initial_state, config)

    if result.get("error"):
        console.print(f"[red]Error: {result['error']}[/red]")
        raise typer.Exit(1)

    _print_result(result)


@app.command()
def edit(
    prompt: str = typer.Argument(..., help="Edit instruction"),
    image: str = typer.Option(..., "--image", "-i", help="Path to source image"),
    size: str = typer.Option("1024x1024", help="Image size"),
):
    """Edit an existing image with a text instruction."""
    console.print(Panel(f"[bold]Edit:[/bold] {prompt}\n[bold]Image:[/bold] {image}", title="Image Agent"))

    graph = compile_graph()
    initial_state = {
        "original_prompt": prompt,
        "source_image_path": image,
    }
    if size != "1024x1024":
        initial_state["generation_params"] = {"size": size}
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    with console.status("[bold green]Editing..."):
        result = graph.invoke(initial_state, config)

    if result.get("error"):
        console.print(f"[red]Error: {result['error']}[/red]")
        raise typer.Exit(1)

    _print_result(result)


@app.command()
def enhance(
    prompt: str = typer.Argument(..., help="Prompt to enhance"),
):
    """Research and enhance a prompt without generating an image."""
    console.print(Panel(f"[bold]Prompt:[/bold] {prompt}", title="Enhance Only"))

    graph = compile_graph()
    initial_state = {
        "original_prompt": prompt,
        "action": "enhance_only",
    }

    # Manually run just the research + enhance steps
    from image_agent.nodes.router import router_node
    from image_agent.nodes.research import research_node
    from image_agent.nodes.enhance import enhance_node

    with console.status("[bold green]Researching & enhancing..."):
        state = {"original_prompt": prompt}
        state.update(router_node(state))
        state.update(research_node(state))
        state.update(enhance_node(state))

    console.print("\n[bold]Enhanced prompt:[/bold]")
    console.print(Panel(state.get("enhanced_prompt", ""), border_style="green"))

    research = state.get("research_context", {})
    if research.get("synthesized"):
        console.print("\n[bold]Research summary:[/bold]")
        console.print(Panel(research["synthesized"], border_style="blue"))


@app.command()
def history(
    limit: int = typer.Option(20, help="Max records to show"),
    clear: bool = typer.Option(False, "--clear", help="Delete all history"),
):
    """Show generation history."""
    if clear:
        count = clear_history()
        console.print(f"[yellow]Cleared {count} files.[/yellow]")
        return

    records = list_history(limit=limit)
    if not records:
        console.print("[dim]No generation history found.[/dim]")
        return

    table = Table(title="Generation History")
    table.add_column("ID", style="cyan")
    table.add_column("Timestamp", style="green")
    table.add_column("Prompt", max_width=40)
    table.add_column("Provider", style="magenta")
    table.add_column("Action", style="yellow")

    for rec in records:
        table.add_row(
            rec.get("image_id", "?"),
            rec.get("timestamp", "?"),
            (rec.get("original_prompt", "")[:37] + "...") if len(rec.get("original_prompt", "")) > 40 else rec.get("original_prompt", ""),
            rec.get("provider", "?"),
            rec.get("action", "?"),
        )

    console.print(table)


@app.command()
def chat():
    """Interactive REPL for generating images."""
    # Friendly greeting
    console.print()
    console.print(Panel(
        "[bold bright_cyan]Hey there! Welcome to Image Agent[/bold bright_cyan]\n\n"
        "I'm your AI-powered image creation assistant. I can help you with:\n\n"
        "  [green]1.[/green] Generate images from your descriptions\n"
        "  [green]2.[/green] Research and enhance your prompts for better results\n"
        "  [green]3.[/green] Edit existing images with instructions\n\n"
        "[dim]Commands: /history  /clear  /quit[/dim]",
        border_style="bright_cyan",
        title="[bold]Image Agent[/bold]",
        subtitle="[dim]Powered by LangGraph + AI[/dim]",
    ))
    console.print()
    console.print("[bold bright_cyan]What would you like me to create for you today?[/bold bright_cyan]")
    console.print("[dim]Just describe the image you have in mind and I'll bring it to life![/dim]\n")

    graph = compile_graph()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    image_count = 0
    last_image_path = None
    last_prompt = None
    last_suggestions = None  # Remember suggestions from previous turn
    last_phase1_result = None  # Remember Phase 1 result for re-picking

    while True:
        try:
            user_input = console.input("[bold bright_cyan]You:[/bold bright_cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[bright_cyan]It was great creating with you! See you next time![/bright_cyan]")
            break

        if not user_input:
            continue
        if user_input.lower() in ("/quit", "/exit", "/q", "quit", "exit", "bye", "goodbye", "see you", "thanks bye"):
            console.print(f"\n[bright_cyan]Thanks for hanging out! I created {image_count} image{'s' if image_count != 1 else ''} for you today. See you soon![/bright_cyan]")
            break
        if user_input == "/history":
            history(limit=10)
            continue
        if user_input == "/clear":
            count = clear_history()
            console.print(f"[yellow]Done! Cleared {count} files from history.[/yellow]\n")
            continue

        # --- Check if user is picking a previous suggestion ---
        prev_pick = _match_previous_option(user_input, last_suggestions)
        if prev_pick is not None and last_phase1_result is not None:
            console.print(f"\n[bright_cyan]Agent:[/bright_cyan] Going with [bold]Option {prev_pick + 1}[/bold] from before!")
            console.print("[dim]   Enhancing prompt and generating your image...[/dim]\n")

            selected_suggestion = _format_suggestion(last_suggestions[prev_pick])
            phase2_state = {
                "original_prompt": last_phase1_result.get("original_prompt", user_input),
                "last_image_path": last_image_path,
                "last_prompt": last_prompt,
                "suggestion_phase_complete": True,
                "selected_suggestion": selected_suggestion,
                "research_context": last_phase1_result.get("research_context"),
                "prompt_analysis": last_phase1_result.get("prompt_analysis"),
                "action": last_phase1_result.get("action"),
                # Carry reference image data from Phase 1
                "reference_images": last_phase1_result.get("reference_images"),
                "reference_image_analysis": last_phase1_result.get("reference_image_analysis"),
                "enhanced_prompt": None,
                "error": None,
                "image_path": None,
                "generation_metadata": None,
            }
            phase2_config = {"configurable": {"thread_id": str(uuid.uuid4())}}

            with console.status("[bold green]Creating your image..."):
                result = graph.invoke(phase2_state, phase2_config)

            if result.get("error"):
                console.print(f"[bright_cyan]Agent:[/bright_cyan] [red]Oops, something went wrong: {result['error']}[/red]\n")
            else:
                image_count += 1
                if result.get("image_path"):
                    last_image_path = result["image_path"]
                last_prompt = result.get("enhanced_prompt") or user_input
                _print_result(result)
                console.print(f"\n[bright_cyan]Agent:[/bright_cyan] There you go! Your image is ready.")
                console.print("[bright_cyan]Agent:[/bright_cyan] Want me to create something else? Just describe it!\n")
            continue

        console.print(f"\n[bright_cyan]Agent:[/bright_cyan] Great idea! Let me work on that for you...")
        console.print("[dim]   Researching style references and preparing creative directions...[/dim]\n")

        # --- Phase 1: Route → Research → Suggest (or Edit path) ---
        initial_state = {
            "original_prompt": user_input,
            "last_image_path": last_image_path,
            "last_prompt": last_prompt,
            # Reset stale fields from previous turn
            "enhanced_prompt": None,
            "research_context": None,
            "source_image_path": None,
            "error": None,
            "image_path": None,
            "generation_metadata": None,
            "suggestions": None,
            "selected_suggestion": None,
            "skip_suggestions": False,
            "suggestion_phase_complete": False,
        }
        with console.status("[bold green]Researching and analyzing..."):
            result = graph.invoke(initial_state, config)

        if result.get("error"):
            console.print(f"[bright_cyan]Agent:[/bright_cyan] [red]Oops, something went wrong: {result['error']}[/red]")
            console.print("[bright_cyan]Agent:[/bright_cyan] Want to try a different prompt?\n")
            continue

        # If this was an edit action (or enhance_only), the full pipeline already ran
        # (edit→save→response→END), so we have a final image — skip Phase 2.
        if result.get("image_path"):
            image_count += 1
            last_image_path = result["image_path"]
            last_prompt = result.get("enhanced_prompt") or user_input
            _print_result(result)
            console.print(f"\n[bright_cyan]Agent:[/bright_cyan] There you go! Your image is ready.")
            console.print("[bright_cyan]Agent:[/bright_cyan] Want me to create something else? Just describe it!\n")
            continue

        # We have suggestions — display them and get user choice
        suggestions = result.get("suggestions") or []
        last_suggestions = suggestions if len(suggestions) > 1 else None
        last_phase1_result = result
        if not suggestions:
            # No suggestions returned (shouldn't happen), proceed without
            selected_suggestion = None
        elif len(suggestions) == 1:
            # Fallback single suggestion — auto-select it
            selected_suggestion = _format_suggestion(suggestions[0])
            console.print("[dim]   Using default creative direction...[/dim]\n")
        else:
            _display_suggestions(suggestions)
            selected_suggestion = _get_user_selection(suggestions)

        # --- Phase 2: Enhance → Provider → Generate → Save → Response ---
        console.print()
        console.print("[dim]   Enhancing prompt and generating your image...[/dim]\n")

        phase2_state = {
            "original_prompt": user_input,
            "last_image_path": last_image_path,
            "last_prompt": last_prompt,
            "suggestion_phase_complete": True,
            "selected_suggestion": selected_suggestion,
            # Carry over from Phase 1
            "research_context": result.get("research_context"),
            "prompt_analysis": result.get("prompt_analysis"),
            "action": result.get("action"),
            # Carry reference image data from Phase 1
            "reference_images": result.get("reference_images"),
            "reference_image_analysis": result.get("reference_image_analysis"),
            # Reset output fields
            "enhanced_prompt": None,
            "error": None,
            "image_path": None,
            "generation_metadata": None,
        }

        # Use a fresh thread for Phase 2 so we don't collide with Phase 1 checkpoint
        phase2_config = {"configurable": {"thread_id": str(uuid.uuid4())}}

        with console.status("[bold green]Creating your image..."):
            result = graph.invoke(phase2_state, phase2_config)

        if result.get("error"):
            console.print(f"[bright_cyan]Agent:[/bright_cyan] [red]Oops, something went wrong: {result['error']}[/red]")
            console.print("[bright_cyan]Agent:[/bright_cyan] Want to try a different prompt?\n")
        else:
            image_count += 1
            if result.get("image_path"):
                last_image_path = result["image_path"]
            last_prompt = result.get("enhanced_prompt") or user_input
            _print_result(result)
            console.print(f"\n[bright_cyan]Agent:[/bright_cyan] There you go! Your image is ready.")
            console.print("[bright_cyan]Agent:[/bright_cyan] Want me to create something else? Just describe it!\n")


import re

# Patterns that match references to previous options (e.g. "1st one", "option 1", "first one", "the first")
_OPTION_PATTERNS = [
    (re.compile(r"^(?:the\s+)?(?:1st|first)\s*(?:one|option)?$", re.I), 0),
    (re.compile(r"^(?:the\s+)?(?:2nd|second)\s*(?:one|option)?$", re.I), 1),
    (re.compile(r"^(?:the\s+)?(?:3rd|third)\s*(?:one|option)?$", re.I), 2),
    (re.compile(r"^option\s*([123])$", re.I), None),  # dynamic index
]


def _match_previous_option(text: str, suggestions: list[dict] | None) -> int | None:
    """Return a 0-based suggestion index if the text references a previous option, else None."""
    if not suggestions:
        return None
    text = text.strip()
    for pattern, idx in _OPTION_PATTERNS:
        m = pattern.match(text)
        if m:
            if idx is not None:
                return idx if idx < len(suggestions) else None
            # Dynamic group capture
            return int(m.group(1)) - 1
    return None


def _format_suggestion(suggestion: dict) -> str:
    """Convert a suggestion dict into a text block for the enhance node."""
    parts = [
        f"Title: {suggestion.get('title', 'Untitled')}",
        f"Description: {suggestion.get('description', '')}",
        f"Style: {suggestion.get('style', '')}",
        f"Mood: {suggestion.get('mood', '')}",
    ]
    elements = suggestion.get("key_elements", [])
    if elements:
        parts.append(f"Key elements: {', '.join(elements)}")
    return "\n".join(parts)


def _display_suggestions(suggestions: list[dict]) -> None:
    """Render suggestion panels with Rich."""
    console.print("[bold bright_cyan]Agent:[/bold bright_cyan] Here are 3 creative directions for your image:\n")
    colors = ["green", "yellow", "magenta"]
    for i, s in enumerate(suggestions):
        elements = s.get("key_elements", [])
        elements_str = ", ".join(elements) if elements else "—"
        body = (
            f"[bold]{s.get('description', '')}[/bold]\n\n"
            f"[dim]Style:[/dim]  {s.get('style', '—')}   "
            f"[dim]Mood:[/dim]  {s.get('mood', '—')}\n"
            f"[dim]Key elements:[/dim]  {elements_str}"
        )
        color = colors[i % len(colors)]
        console.print(Panel(
            body,
            title=f"[bold {color}]Option {i + 1}: {s.get('title', '')}[/bold {color}]",
            border_style=color,
            padding=(1, 2),
        ))

    console.print(
        "[dim]Enter [bold]1[/bold], [bold]2[/bold], or [bold]3[/bold] to pick a direction  |  "
        "Type [bold]skip[/bold] to generate without a direction  |  "
        "Or type your own creative feedback[/dim]\n"
    )


def _get_user_selection(suggestions: list[dict]) -> str | None:
    """Prompt the user to select a suggestion and return formatted text or None."""
    try:
        choice = console.input("[bold bright_cyan]Your choice:[/bold bright_cyan] ").strip()
    except (KeyboardInterrupt, EOFError):
        console.print("\n[dim]Skipping suggestions...[/dim]")
        return None

    if not choice or choice.lower() == "skip":
        return None

    # Check for numeric selection
    if choice in ("1", "2", "3"):
        idx = int(choice) - 1
        if idx < len(suggestions):
            console.print(f"[bright_cyan]Agent:[/bright_cyan] Great pick! Going with [bold]Option {choice}[/bold].")
            return _format_suggestion(suggestions[idx])

    # Treat as custom creative direction
    console.print(f"[bright_cyan]Agent:[/bright_cyan] Got it! I'll use your custom direction.")
    return f"Custom creative direction: {choice}"


def _print_result(result: dict) -> None:
    """Pretty-print a generation result."""
    image_path = result.get("image_path", "?")
    provider = (result.get("generation_metadata") or {}).get("provider", "?")
    enhanced = result.get("enhanced_prompt") or ""

    console.print(f"\n[bold green]Image saved:[/bold green] {image_path}")
    console.print(f"[bold]Provider:[/bold] {provider}")

    if enhanced:
        console.print(f"\n[bold]Enhanced prompt:[/bold]")
        console.print(Panel(enhanced, border_style="green"))

    research = result.get("research_context") or {}
    if research.get("synthesized"):
        preview = research["synthesized"][:400]
        if len(research["synthesized"]) > 400:
            preview += "..."
        console.print(f"\n[bold]Research:[/bold]")
        console.print(Panel(preview, border_style="blue"))


if __name__ == "__main__":
    app()
