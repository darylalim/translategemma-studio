# TranslateGemma Pipeline

Translate text with the [Google TranslateGemma 4B model](https://huggingface.co/google/translategemma-4b-it) on Apple Silicon with MLX.

## Commands

- `uv sync` — install dependencies
- `uv run streamlit run streamlit_app.py` — run application
- `uv run ruff check .` — lint
- `uv run ruff format .` — format
- `uv run ty check` — typecheck
- `uv run pytest` — run tests
- `uv run pytest tests/path_to_test.py::test_name -v` — run single test

## Code Style

- snake_case for functions/variables, PascalCase for classes
- Type annotations for all parameters and returns
- Formatting and import sorting handled by ruff

## Dependencies

- `streamlit` — web UI
- `mlx-lm` — model loading and inference on Apple Silicon

## Architecture

### Languages

Language data lives in `languages.py` with two dicts from the TranslateGemma Technical Report (Tables 5 and 6):

- `BIDIRECTIONAL` — 225 languages paired with English in both directions (can be source or target)
- `FROM_ENGLISH_ONLY` — 70 languages that can only be targets when English is the source

Derived constants: `ALL_LANGUAGES` (merged dict for lookups), `SOURCE_LANGS` (sorted bidirectional names), `TARGET_LANGS_FOR_ENGLISH` (sorted non-English names from both dicts).

Directionality rules: bidirectional languages pair only with English (not with each other). From-English-only languages can only receive translations from English. The swap button is disabled when swapping would create an invalid pair.

### Model Loading

`load_model()` returns `(model, tokenizer)`. Cached with `@st.cache_resource`. Loads `mlx-community/translategemma-4b-it-8bit` via `mlx_lm.load()`. Registers `<end_of_turn>` as an EOS token via `tokenizer.add_eos_token()` so generation stops early instead of running to `MAX_NEW_TOKENS`.

### Translation

`translate(text, src_lang, src_code, tgt_lang, tgt_code)` builds the prompt, loads the model, and runs `mlx_lm.generate()`. Generation stops at `<end_of_turn>` via the registered EOS token. A safety-net split on `<end_of_turn>` strips the token if it leaks into the output string. Returns a `str` — the translated text. `MAX_NEW_TOKENS` (512) limits the maximum output length.

### UI

- Caption under the title: `st.caption` with a markdown link to the Google TranslateGemma model card
- Language selectors: 3-column `[10, 1, 10]` layout with swap button (`:material/swap_horiz:`) in the middle, labels hidden via `label_visibility="collapsed"`
- Swap button moves translation output to source input and clears the result; disabled when target is a from-English-only language (the only case where swap is invalid, since non-English sources always target English which is always swappable)
- 2-column side-by-side; `st.text_area` (no placeholder, `max_chars=5000`, height 300) for input, disabled `st.text_area` (placeholder "Translation", height 300) for output
- Output text areas use `st.session_state` to set value (not the `value` parameter) to avoid stale widget state
- Left panel (inside `left_col`): translate button (primary, `use_container_width=True`)
- Right panel (inside `right_col`): download button (secondary, `use_container_width=True`), `disabled` when no translation
- Download uses `st.download_button` with `mime="text/plain"`
- `st.session_state` keys: `source_lang`, `target_lang`, `translation_result`, `source_text`, `text_output`

## Known Issues

### Do NOT use `tokenizer.apply_chat_template` for text translation

TranslateGemma ships a large custom chat template that requires `content` to be a list of exactly one structured mapping (`type`, `source_lang_code`, `target_lang_code`, `text`) — not a plain string. The standard mlx-lm / model-card boilerplate (`content="hello"`) trips the template's `content | length != 1` guard and raises at runtime:

```
jinja2.exceptions.TemplateError: User role must provide `content` as an
iterable with exactly one item.
```

Given the structured format the template works (it builds the prompt in "Prompt Template" below), but this app constructs that string manually instead — keeping the prompt explicit and independent of the MLX quant's bundled template.

```python
prompt = f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n"
```

### Strip `<end_of_turn>` from model output

`load_model()` registers `<end_of_turn>` as an EOS token so generation stops early. The `translate()` split on `<end_of_turn>` is kept as a safety net in case the token leaks into the output string.

### Chinese uses `zh-CN` (not `zh`)

The app uses `zh-CN` as the language code for Chinese, matching the TranslateGemma Technical Report (Table 5). This is correct because the app constructs prompts manually (not via `apply_chat_template`), and the locale code is inserted as text in the prompt string. The model was trained with these locale codes.

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
