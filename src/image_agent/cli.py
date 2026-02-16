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
    provider: Optional[str] = typer.Option(None, help="Force provider: openai or flux"),
    size: str = typer.Option("1024x1024", help="Image size"),
):
    """Generate an image from a text prompt with internet research."""
    console.print(Panel(f"[bold]Prompt:[/bold] {prompt}", title="Image Agent"))
    console.print("[dim]Routing → Researching → Enhancing → Generating...[/dim]\n")

    graph = compile_graph()
    initial_state = {"original_prompt": prompt}
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

        console.print(f"\n[bright_cyan]Agent:[/bright_cyan] Great idea! Let me work on that for you...")
        console.print("[dim]   Researching style references, enhancing your prompt, and generating...[/dim]\n")

        initial_state = {
            "original_prompt": user_input,
            "last_image_path": last_image_path,
            # Reset stale fields from previous turn's checkpoint
            "enhanced_prompt": None,
            "research_context": None,
            "source_image_path": None,
            "error": None,
            "image_path": None,
            "generation_metadata": None,
        }
        with console.status("[bold green]Creating your image..."):
            result = graph.invoke(initial_state, config)

        if result.get("error"):
            console.print(f"[bright_cyan]Agent:[/bright_cyan] [red]Oops, something went wrong: {result['error']}[/red]")
            console.print("[bright_cyan]Agent:[/bright_cyan] Want to try a different prompt?\n")
        else:
            image_count += 1
            if result.get("image_path"):
                last_image_path = result["image_path"]
            _print_result(result)
            console.print(f"\n[bright_cyan]Agent:[/bright_cyan] There you go! Your image is ready.")
            console.print("[bright_cyan]Agent:[/bright_cyan] Want me to create something else? Just describe it!\n")


def _print_result(result: dict) -> None:
    """Pretty-print a generation result."""
    image_path = result.get("image_path", "?")
    provider = result.get("generation_metadata", {}).get("provider", "?")
    enhanced = result.get("enhanced_prompt", "")

    console.print(f"\n[bold green]Image saved:[/bold green] {image_path}")
    console.print(f"[bold]Provider:[/bold] {provider}")

    if enhanced:
        console.print(f"\n[bold]Enhanced prompt:[/bold]")
        console.print(Panel(enhanced, border_style="green"))

    research = result.get("research_context", {})
    if research.get("synthesized"):
        preview = research["synthesized"][:400]
        if len(research["synthesized"]) > 400:
            preview += "..."
        console.print(f"\n[bold]Research:[/bold]")
        console.print(Panel(preview, border_style="blue"))


if __name__ == "__main__":
    app()
