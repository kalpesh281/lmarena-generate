# Image Agent

AI-powered image generation CLI built with **LangGraph**. Takes a simple text prompt, researches visual context from the web, enhances the prompt with real-world details, and generates high-quality images using OpenAI or Flux.

## How It Works

```
User Prompt → Router → Research → Enhance → Provider Select → Generate → Save
```

1. **Router** - Classifies intent (generate / edit / enhance-only) and analyzes style, mood, subject, and complexity using GPT-4o-mini
2. **Research** - Searches the web via Tavily for style references, factual context, and trending AI art techniques, then synthesizes findings with GPT-4o
3. **Enhance** - Transforms your simple prompt into a detailed, vivid image generation prompt enriched with research context
4. **Provider Select** - Routes photorealistic styles to **Flux** and artistic styles (anime, illustration, watercolor, etc.) to **OpenAI**
5. **Generate** - Calls OpenAI `gpt-image-1` or Flux (via Hugging Face Inference API)
6. **Save** - Writes the image (PNG) and a JSON metadata sidecar to `output/`

## Setup

### Prerequisites

- Python 3.11+
- API keys for OpenAI, Hugging Face, and Tavily

### Install

```bash
# Clone the repo
git clone <repo-url>
cd image-generation

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .
```

### Configure

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

```env
OPENAI_API_KEY=sk-...
HUGGINGFACE_API_KEY=hf_...
TAVILY_API_KEY=tvly-...
```

## Usage

### Interactive Chat (Recommended)

```bash
python app.py
```

Opens an interactive REPL where you can describe images and get them generated. Supports commands: `/history`, `/clear`, `/quit`.

### CLI Commands

```bash
# Generate an image from a prompt
image-agent generate "A serene Japanese garden at sunset"

# Edit an existing image
image-agent edit "Add cherry blossoms" --image output/my_image.png

# Enhance a prompt without generating
image-agent enhance "A futuristic city"

# View generation history
image-agent history
image-agent history --clear
```

## Project Structure

```
.
├── app.py                          # Entry point (starts interactive chat)
├── pyproject.toml                  # Project config and dependencies
├── .env.example                    # API key template
├── output/                         # Generated images + JSON metadata
└── src/image_agent/
    ├── cli.py                      # Typer CLI (generate, edit, enhance, history, chat)
    ├── graph.py                    # LangGraph StateGraph definition
    ├── state.py                    # State schema (TypedDict)
    ├── config.py                   # Settings from .env
    ├── history.py                  # JSON-based generation history
    ├── nodes/
    │   ├── router.py               # Intent classification + prompt analysis
    │   ├── research.py             # Web research via Tavily
    │   ├── enhance.py              # Prompt enhancement with research context
    │   ├── provider.py             # Style-based provider routing
    │   ├── generate.py             # OpenAI + Flux generation nodes
    │   ├── edit.py                 # Image editing via OpenAI
    │   ├── save.py                 # Save image + metadata sidecar
    │   └── response.py             # Format final output
    ├── providers/
    │   ├── openai_image.py         # OpenAI gpt-image-1 API wrapper
    │   ├── flux_image.py           # Flux via Hugging Face Inference API
    │   └── image_utils.py          # Download, base64, resize utilities
    └── prompts/
        └── templates.py            # System prompts for router, research, enhance
```

## Provider Routing

| Style | Provider |
|-------|----------|
| Photorealistic, photography, cinematic, portrait, landscape, architectural, product, fashion | **Flux** (Hugging Face) |
| Anime, cartoon, illustration, digital-art, oil-painting, watercolor, 3D-render, pixel-art, fantasy, abstract | **OpenAI** (gpt-image-1) |

## Output Format

Each generation produces two files in `output/`:

- **`{timestamp}_{id}.png`** - The generated image
- **`{timestamp}_{id}.json`** - Metadata including original prompt, enhanced prompt, research context, provider, and generation parameters

## Tech Stack

- **[LangGraph](https://github.com/langchain-ai/langgraph)** - Agentic workflow orchestration
- **[LangChain](https://github.com/langchain-ai/langchain)** - LLM integration (OpenAI)
- **[Tavily](https://tavily.com)** - Web search API for visual research
- **[OpenAI](https://platform.openai.com)** - GPT-4o-mini (routing), GPT-4o (enhancement), gpt-image-1 (generation)
- **[Hugging Face](https://huggingface.co)** - Flux model for photorealistic generation
- **[Typer](https://typer.tiangolo.com)** + **[Rich](https://rich.readthedocs.io)** - CLI interface
