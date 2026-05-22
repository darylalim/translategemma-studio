# TranslateGemma Pipeline

Translate text with the [Google TranslateGemma 4B model](https://huggingface.co/google/translategemma-4b-it) on Apple Silicon with MLX.

## Features

- **Text translation** — translate text between supported languages
- **Token counter** — live input usage against the model's context window, with translation blocked when the input is over budget
- **Swap languages** — swap source and target languages, moving translation output to source input
- **Download as text** — download translation output as a `.txt` file

## Supported Languages

295 languages from the [TranslateGemma Technical Report](https://arxiv.org/pdf/2601.09012):

- **225 bidirectional** — paired with English in both directions (e.g., French, Japanese, Swahili)
- **70 from-English-only** — can only receive translations from English (e.g., Albanian, Finnish, Tamil)

Quality varies by language — 55 of the 295 have published WMT24++ benchmark scores in the technical report, while the rest are trained but not formally evaluated.

## Requirements

- Python 3.12+
- Apple Silicon Mac (M-series)

## Setup

1. Install dependencies:

   ```bash
   uv sync
   ```

2. Run the application:

   ```bash
   uv run streamlit run streamlit_app.py
   ```

## Development

```bash
uv run ruff check .   # lint
uv run ruff format .  # format
uv run ty check       # typecheck
uv run pytest         # run tests
```
