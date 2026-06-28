# TranslateGemma Studio

Streamlit application for translation using the [Google TranslateGemma model](https://huggingface.co/google/translategemma-4b-it) on Apple Silicon with MLX.

## Commands

- `uv sync` — install dependencies
- `uv run streamlit run streamlit_app.py` — run application
- `uv run ruff check .` — lint
- `uv run ruff format .` — format
- `uv run ty check` — typecheck
- `uv run pytest` — run tests
- `uv run pytest tests/path_to_test.py::test_name -v` — run single test
- `uv run pytest --cov` — run tests with coverage (sources configured in `pyproject.toml`)

## Code Style

- `snake_case` for functions and variables, `PascalCase` for classes
- Type annotations on all parameters and returns
- Formatting and import sorting handled by ruff
- Ruff lint rules beyond the defaults are set in `[tool.ruff.lint]` via `extend-select`: `I` (import sorting), `UP` (pyupgrade), `B` (bugbear), `C4` (comprehensions), `RUF` (ruff-specific), `SIM` (simplify) — enforced by `uv run ruff check .` and CI
- When working with Python, invoke the relevant `/astral:<skill>` for uv, ty, and ruff to ensure best practices are followed

## Dependencies

- `streamlit` — web UI
- `mlx-lm` — model loading and inference on Apple Silicon

## Architecture

### Languages

Two dicts in `languages.py` from the TranslateGemma Technical Report (Tables 5 and 6):

- `BIDIRECTIONAL` (225) — pair with English in both directions
- `FROM_ENGLISH_ONLY` (70) — receive translations from English only

Derived constants: `ALL_LANGUAGES` (merged for name → code lookup), `SOURCE_LANGS` (sorted bidirectional names), `TARGET_LANGS_FOR_ENGLISH` (sorted non-English names from both dicts).

Directionality: bidirectional languages pair only with English (not with each other). The swap button is disabled when swapping would produce an invalid pair.

### Model Loading

`load_model()` returns `(model, tokenizer)`, cached with `@st.cache_resource`. Loads `mlx-community/translategemma-4b-it-8bit` via `mlx_lm.load()` and registers `<end_of_turn>` as an EOS token so generation stops early instead of running to the `max_tokens` cap.

### Translation

`_prepare_generation()` builds the prompt, loads the model, enforces the token budget, and returns `(model, tokenizer, prompt, max_tokens)` — shared by both entry points:

- `translate(...)` — runs `mlx_lm.generate()`, returns `str`
- `translate_stream(...)` — generator running `mlx_lm.stream_generate()`, yields segment-by-segment

`_strip_eos_token()` removes `<end_of_turn>` from the output as a safety net for the rare case it leaks past the registered EOS.

### Context window

- `CONTEXT_WINDOW = 2048` — total context, shared by prompt and output
- `MAX_PROMPT_TOKENS = 1024` — prompt cap; `_prepare_generation()` raises `ValueError` when exceeded
- `max_tokens = CONTEXT_WINDOW - prompt_tokens` — translation gets all remaining room (EOS still stops it early)
- `MAX_INPUT_CHARS = 5000` — coarse text-area backstop; the token counter is the real, language-aware limit

`count_prompt_tokens(prompt, tokenizer)` returns the token length of the wrapped prompt — the Gemma chat scaffold (`<start_of_turn>user...`) is included, since that's what `build_prompt()` returns. The UI shows a live token count under the input and disables Translate when over budget.

### UI

- **Header** — `st.title` plus `st.caption` linking the model card
- **Language selectors** — `[10, 1, 10]` column layout with the swap button (`:material/swap_horiz:`) in the middle; labels collapsed
- **Swap button** — calls `_swap_languages()` to swap source/target and move the previous translation into the source area; disabled when target is `FROM_ENGLISH_ONLY` (the only invalid swap, since non-English sources always pair with English)
- **Body** — two side-by-side columns:
  - **Left** — `st.text_area` (`key="source_text"`, height 300, `max_chars=MAX_INPUT_CHARS`); live token counter caption with a red over-budget `st.badge`; Translate button (primary, full-width, disabled when over budget)
  - **Right** — `st.empty()` placeholder holding either the disabled output `st.text_area` (height 300) or the streaming container during generation; alignment-spacer caption; Download button (secondary, `mime="text/plain"`, disabled when no result)
- **Streaming** — Translate feeds `translate_stream()` into a fixed-height (300) `st.container`, updated token-by-token via `st.text` (raw text, not markdown — matches the text area and the `text/plain` download). On completion the result is saved to `st.session_state["translation_result"]` and `st.rerun()` reverts the placeholder to the settled text area.
- **Session state keys** — `source_lang`, `target_lang`, `translation_result`, `source_text`, `text_output`
- **State seeding** — output text areas are populated via session state (not the `value=` parameter) to avoid stale widget state
- **1.58 conventions** — buttons size with `width="stretch"` (replacing the deprecated `use_container_width`, which 1.58 still accepts but plans to remove); the page icon and the `st.error`/`st.warning` callouts use Material Symbols (`:material/...:`)

### Theme

`.streamlit/config.toml` applies a Material Design 3 theme (violet `#6750A4` primary, Roboto via Google Fonts) with `[theme.light]` and `[theme.dark]` variants, which gives the in-app light/dark switcher. The file is git-tracked — `.gitignore` keeps `config.toml` while ignoring `secrets.toml` and the rest of `.streamlit/`.

## Testing

Two layers plus a config guard, ~1s combined for 87 tests at 100% coverage:

- **Import-time tests** — swap `sys.modules["streamlit"]` and `sys.modules["mlx_lm"]` for `MagicMock`s, import `streamlit_app.py`, then assert on captured `st.*` calls. No Streamlit runtime runs. Covers pure functions, layout, token counting, EOS stripping.
- **End-to-end tests** (`TestStreamingClickPath`) — drive the real script via `streamlit.testing.v1.AppTest` with only `mlx_lm` mocked. Reaches branches the import-time tests can't: streaming click path, model-load failure, runtime target filtering, swap-button wiring, empty-text warning.
- **Theme-config guard** (`TestThemeConfig`) — validates `.streamlit/config.toml` keys against Streamlit's option template (`config._config_options_template`), the same lookup the runtime uses. Catches invalid theme keys (e.g. a per-variant `base`) that Streamlit only *logs* a warning for, so they'd otherwise slip past the suite.

**Fixtures (`tests/conftest.py`):**

- `_clear_streamlit_caches` (autouse) — clears `st.cache_resource` before each test; required because Streamlit's resource cache is process-global
- `app_module` (session) — mocked-import setup for the import-time tests
- `mock_tokenizer` — `encode()` returns 50 tokens, under the budget cap
- `patched_translate` — patches `load_model`, `generate`, `stream_generate`; exposes the mocks for per-test configuration
- `fake_mlx_lm` — `mlx_lm` mock injected into `sys.modules` for AppTest fixtures
- `app_test` — AppTest pre-run to its settled state
- `app_test_unrun` — AppTest not yet run; for tests that configure mocks before the first `.run()` (e.g. load failure)

**Pytest config (`pyproject.toml`):** `addopts = ["-ra", "--strict-markers", "--strict-config"]`, `xfail_strict = true`, `filterwarnings = ["error"]`. Coverage sources in `[tool.coverage.run]`.

**CI (`.github/workflows/ci.yml`):** ruff + ty + pytest on `macos-14` (required for `mlx-lm`) for every push to `main` and PR.

## Known Issues

### Do NOT use `tokenizer.apply_chat_template`

TranslateGemma's chat template requires `content` as a list with exactly one structured mapping (`type`, `source_lang_code`, `target_lang_code`, `text`). A plain string trips the `content | length != 1` guard:

```
jinja2.exceptions.TemplateError: User role must provide `content` as an
iterable with exactly one item.
```

The structured form works, but this app builds the prompt as a raw string instead — keeping it explicit and independent of the MLX quant's bundled template:

```python
prompt = f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n"
```

### `<end_of_turn>` safety-net strip

The registered EOS token usually stops generation before `<end_of_turn>` appears. `_strip_eos_token()` is kept as a safety net for the rare case the token leaks into the decoded output.

### Chinese uses `zh-CN`, not `zh`

The locale code matches the TranslateGemma Technical Report (Table 5). Since prompts are built manually, the code is inserted as text — and the model was trained with these locale codes.

### Theme variant keys are top-level-only

`base` and `chartCategoricalColors` are valid only in the top-level `[theme]` section, not inside `[theme.light]`/`[theme.dark]`. Streamlit only *logs* a warning for an invalid config key rather than raising, so `TestThemeConfig` validates every `config.toml` key against Streamlit's option template to catch regressions.

## Prompt Template

```
You are a professional {source_lang} ({src_lang_code}) to {target_lang}
({tgt_lang_code}) translator. Your goal is to accurately convey the meaning and
nuances of the original {source_lang} text while adhering to {target_lang} grammar,
vocabulary, and cultural sensitivities.\nProduce only the {target_lang}
translation, without any additional explanations or commentary. Please translate
the following {source_lang} text into {target_lang}:\n\n\n{text}
```

## Resources

- [Technical Report](https://arxiv.org/pdf/2601.09012)
- [Gemma Cookbook](https://colab.research.google.com/github/google-gemini/gemma-cookbook/blob/main/Research/[TranslateGemma]Example.ipynb)
- [Streamlit AppTest reference](https://docs.streamlit.io/develop/api-reference/app-testing)
