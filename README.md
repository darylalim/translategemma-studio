# TranslateGemma Studio

[![CI](https://github.com/darylalim/translategemma-studio/actions/workflows/ci.yml/badge.svg)](https://github.com/darylalim/translategemma-studio/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/darylalim/translategemma-studio)](https://github.com/darylalim/translategemma-studio/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.12+-blue.svg)

Translate text between 295 languages entirely on your Mac — no cloud, no API keys, nothing leaves your machine. A Streamlit app that runs Google's [TranslateGemma](https://huggingface.co/google/translategemma-4b-it) locally on Apple Silicon via MLX.

<p align="center">
  <img src="assets/screenshot-dark.png" width="100%" alt="TranslateGemma Studio in dark mode" />
</p>

## Features

- **Text translation** — translate text between supported languages
- **Streaming output** — translation streams in token-by-token as the model generates
- **Token budget** — a 1,024-token prompt cap; over it, Translate is disabled and a badge shows the count to trim to
- **Swap languages** — swap source and target languages, moving translation output to source input
- **Download as text** — download translation output as a `.txt` file
- **Light and dark mode** — a paired light/dark theme with a native Mac look, with an in-app theme switcher

## Supported Languages

295 languages from the [TranslateGemma Technical Report](https://arxiv.org/pdf/2601.09012):

- **225 bidirectional** — paired with English in both directions (e.g., French, Japanese, Swahili)
- **70 from-English-only** — can only receive translations from English (e.g., Albanian, Finnish, Tamil)

Quality varies. 55 of the 295 have published WMT24++ benchmark scores in the technical report; the rest are trained but not formally evaluated.

## Model

Runs the 8-bit MLX quant [`mlx-community/translategemma-4b-it-8bit`](https://huggingface.co/mlx-community/translategemma-4b-it-8bit) (~4B parameters). On first launch it downloads 3.9 GB from the Hugging Face Hub into `~/.cache/huggingface/hub`; later runs load from that cache. The title and language selectors render, then a "Loading model..." spinner sits below them until the download finishes; the text panels appear only after it. All inference is local.

## Requirements

- Python 3.12+
- Apple Silicon Mac
- ~4–5 GB free disk for the model, and 8 GB+ unified memory recommended

## Setup

```bash
uv sync                                  # install dependencies
uv run streamlit run streamlit_app.py    # run the app
```

## Development

```bash
uv run ruff check .            # lint
uv run ruff format .           # format
uv run ty check                # typecheck
uv run pytest                  # run tests
uv run pytest --cov            # run tests with coverage
uv run pytest -m live          # run the live-model test (loads the real quant)
```

The live test is deselected by default, since it loads the full 3.9 GB model. Run it after any `mlx` or `mlx-lm` upgrade — the rest of the suite mocks `mlx_lm`, so it cannot see a stack that loads fine but generates nothing.

CI (`.github/workflows/ci.yml`) runs lint, format check, typecheck, and tests on every push to `main` and PR — on `macos-latest`, since `mlx-lm` only depends on `mlx` on macOS and `mlx`'s macOS wheels are arm64-only.

## Releases

Bumping `version` in `pyproject.toml` is the whole release process. Push the bump to `main` and, once CI passes, the workflow tags the commit and publishes a GitHub Release with generated notes.

## License

This project's code is released under the [MIT License](LICENSE).

The TranslateGemma model it downloads at runtime is subject to Google's [Gemma Terms of Use](https://ai.google.dev/gemma/terms) — the MIT License covers the code in this repository, not the model weights.
