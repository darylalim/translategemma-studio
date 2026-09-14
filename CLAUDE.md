# TranslateGemma Studio

Streamlit application for translation using [Google TranslateGemma](https://huggingface.co/google/translategemma-4b-it) on Apple Silicon with MLX.

Two source files at the repo root, no package layout: `streamlit_app.py` (prompt building, model loading, translation, and the entire UI, top to bottom) and `languages.py` (the two language dicts and their derived constants). Everything else is tests, CI, and config.

## Commands

- `uv sync` — install dependencies
- `uv run streamlit run streamlit_app.py` — run application
- `uv run ruff check .` — lint
- `uv run ruff format .` — format
- `uv run ty check` — typecheck
- `uv run pytest` — run tests
- `uv run pytest tests/path_to_test.py::test_name -v` — run single test
- `uv run pytest -m live` — run the live-model test (loads the real quant; deselected by default)
- `uv run pytest --cov` — run tests with coverage (sources configured in `pyproject.toml`); roughly doubles the runtime, ~1.5s → ~3s

The four CI gates as one command — this is "the gate" referred to below, and what to run before pushing:

```sh
uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest
```

**A cold cache downloads the model first.** Both `uv run streamlit run streamlit_app.py` and `uv run pytest -m live` block on a ~3.9 GB pull from the Hugging Face Hub into `~/.cache/huggingface/hub` before anything else happens — the app shows its title, the language selectors and a "Loading model..." spinner, the live test shows nothing — which looks like a hang. Check whether it is already there:

```sh
du -sh ~/.cache/huggingface/hub/models--mlx-community--translategemma-4b-it-8bit
```

## Do not touch

Each of these is explained in full further down; they are collected here because every one of them is a trap that looks like an improvement.

- **`uv.lock`** — change it through uv (`uv add` / `uv lock` / `uv sync`). A `PreToolUse` hook denies direct writes.
- **`tokenizer.apply_chat_template`** — the prompt is built as a raw string on purpose. See Known Issues.
- **`Stop` hooks** — two existed and were removed deliberately. See Hooks.
- **`use_container_width`** — deprecated by Streamlit; use `width="stretch"`.
- **`[theme.light]` and `[theme.dark]` in `.streamlit/config.toml`** — both must stay, each non-empty. Streamlit keeps the in-app switcher with either one, but the mode that is missing silently becomes the stock palette under this theme's font, radius and borders — a half-designed mode. And `base` is valid only directly under `[theme]`, never inside a mode section. `TestThemeConfig` fails on either. See Architecture → Theme.
- **The target-language filter's position** — it must stay above the target selectbox. Below it the filter is dead code, not a crash: `selectbox` silently resets a stored value that is not among its options to the default and writes it back, so the condition is never true and the target stays valid only by that internal (reproduced 2026-09-13: every test green, nothing logged). `test_target_filter_runs_before_the_target_selectbox` fails on the move. See Architecture → UI.
- **Nothing between a panel and its button** — Translate directly follows the text area and Download directly follows the output box; the over-budget badge renders *below* Translate, and nothing else — no live token counter, no spacer — is added to either column. See Architecture → UI → Button row.
- **The model load's place and its spinner** — `load_model()` is called once, between the selector row and `st.columns(2)`, with no `st.spinner` around it: the decorator's `show_spinner` is the loading indicator, and a wrapper stacks a second spinner ("Running `load_model()`.") under it on every cold load. Above the selectors it hides them behind the spinner; inside a column its spinner lands between a panel and its button. `test_cache_decorator_owns_the_loading_spinner` catches the wrapper; `test_model_load_failure_logs_and_shows_error` pins the place from the failure state: the page column's children must be exactly `title`, the selector row's `flex_container`, `error` — a load above the row drops the row, one inside any column nests the error, one past `st.columns(2)` adds a second block — and the row must be complete (two selectboxes, the swap button), which catches a load slipped in between the second selectbox and the swap button. Every placement but the intended one fails at least one of those (checked by mutation, 2026-09-13). See Architecture → Model Loading and → UI.
- **`max_chars` on the text area** — there is none on purpose. The token budget is the only input limit; a character cap froze the area after a long swap and silently truncated it on the next rerun. `test_text_area_has_no_character_cap` fails on the kwarg and on the positional form. See Architecture → Context window.
- **The page column** — `layout="wide"` and the two wrapper containers (`st.container(horizontal_alignment="center")` holding `st.container(width=PAGE_WIDTH)`) go together: drop the kwarg and the page is back to Streamlit's 736 px column, drop the outer wrapper and the column sits at the left edge, drop the inner one and everything goes full-bleed. Everything after `set_page_config` renders *inside* the `with` block — a top-level `st.*` call outside it is full-bleed under `"wide"`. Both panels read `height=PANEL_HEIGHT`; do not give either its own number. See Architecture → UI → Layout.

## Code Style

- `snake_case` for functions and variables, `PascalCase` for classes
- Type annotations on all parameters and returns in `streamlit_app.py` and `languages.py`; in `tests/`, the conftest helpers and fixtures are annotated and test methods are not (`ty` checks the tests but requires no annotations, and no `ANN` rules are selected)
- Formatting and import sorting handled by ruff
- Ruff lint rules beyond the defaults are set in `[tool.ruff.lint]` via `extend-select`: `I` (import sorting), `UP` (pyupgrade), `B` (bugbear), `C4` (comprehensions), `RUF` (ruff-specific), `SIM` (simplify) — enforced by `uv run ruff check .` and CI
- Invoke `/astral:uv`, `/astral:ruff`, or `/astral:ty` before changing dependency floors, lint configuration, or type-checker settings — not on routine edits

## Dependencies

- `streamlit>=1.58` — web UI; the floor is the oldest release verified to work, while `uv.lock` resolves 1.63.0. The UI's widget conventions are recorded under Architecture → UI and are not tied to a specific release.
- `mlx-lm>=0.31.3` — model loading and inference on Apple Silicon; see Known Issues for why the floor is a patch version

Python is floored at `>=3.12` in `pyproject.toml` and pinned to `3.12` by `.python-version`; uv otherwise picks the newest interpreter it can find.

Dev-group floors (`pytest>=8.4`, `pytest-cov>=7.1.0`, `ruff>=0.16`, `ty>=0.0.69`) pin the tools whose output *is* the CI contract — a lower `ruff` can format differently and a lower `ty` can emit different diagnostics, either of which fails the build. `pytest-cov` is what makes the documented `uv run pytest --cov` work.

Floors record the oldest version verified to work; `uv.lock` still pins the exact resolution, and CI installs from the lock (`uv sync --locked`) — so the floors are never the versions CI actually exercises. Most installed versions sit well above them. Verify a floor change with `uv lock --resolution lowest-direct && uv sync --frozen`, then run the gate with `--frozen` on every command (`uv run --frozen ruff check . && uv run --frozen ruff format --check . && uv run --frozen ty check && uv run --frozen pytest`) — a plain `uv run` re-resolves to highest and reinstalls before the first command runs, so the unmodified gate never exercises the floors. (The `PostToolUse` hook runs `uv run --no-sync`, so editing a `.py` file mid-verification does not re-lock either.) Undo it afterwards: plain `uv sync` (or any `uv run` without `--frozen`) re-resolves to highest and discards the floor lock (it prints a one-line notice but never fails), and uv stamps the strategy into the lockfile as an `[options] resolution-mode` block — this repo's `uv.lock` has no `[options]` block, which is the at-a-glance proof it was resolved at the default `highest`. `uv lock --check` verifies the lock matches `pyproject.toml` locally, before CI does.

## Architecture

### Languages

Two dicts in `languages.py` from the TranslateGemma Technical Report (Appendix C, Tables 5 and 6):

- `BIDIRECTIONAL` (225) — pair with English in both directions
- `FROM_ENGLISH_ONLY` (70) — receive translations from English only

Derived constants: `ALL_LANGUAGES` (merged for name → code lookup), `SOURCE_LANGS` (sorted bidirectional names), `TARGET_LANGS_FOR_ENGLISH` (sorted non-English names from both dicts).

Directionality: bidirectional languages pair only with English (not with each other). The swap button is disabled when swapping would produce an invalid pair.

`ALL_LANGUAGES` is a `{**BIDIRECTIONAL, **FROM_ENGLISH_ONLY}` merge, which would silently last-wins a duplicate key. Two tested invariants make that safe: the dicts share no keys, and every code across the merge is unique — both matter when adding a regional variant. The counts are also asserted in nine separate places across `tests/test_languages.py` and `tests/test_streamlit_app.py`, so adding a language means updating more than one number.

### Model Loading

`load_model()` returns `(model, tokenizer)`, cached with `@st.cache_resource(show_spinner="Loading model...")`. Loads `mlx-community/translategemma-4b-it-8bit` via `mlx_lm.load()` and registers `<end_of_turn>` as an EOS token so generation stops early instead of running to the `max_tokens` cap. The decorator's own cache-miss spinner is the one loading indicator: `show_spinner` defaults to `True`, so wrapping the call in `st.spinner` as well stacked a second spinner ("Running `load_model()`.") under the first on every cold load, and `cache_utils` suppresses its spinner only inside *nested* cached functions, never under an enclosing `st.spinner`. The keyword form is also why conftest's `cache_resource` stand-in accepts both the bare and the called decorator. A consequence of the load sitting below the selector row (see UI): when it fails, the two selectboxes and the swap button are live above the `st.error`, and every interaction with them reruns the script and re-attempts `mlx_lm.load()` — `st.cache_resource` does not cache exceptions — spinner and callout each time. Before the move nothing interactive rendered in that state and the only retry was a page refresh; this is accepted as a retry affordance, not a bug.

`mlx_lm.load()` is annotated as a `tuple[Module, TokenizerWrapper] | tuple[Module, TokenizerWrapper, dict]` union (the 3-tuple is the `return_config=True` branch) with no `Literal`-keyed overloads, so a bare two-name unpack fails `ty check`. `load_model()` narrows it with a length check and raises on anything else rather than carrying a `# ty: ignore` — a suppression would turn into an `unused-ignore-comment` warning, which fails the gate, the day `mlx-lm` adds overloads.

The module configures `logging.basicConfig(INFO)` (silencing `httpx` to `WARNING`); both the model-load and translation failure paths call `logger.exception(...)` alongside their `st.error` callouts.

### Translation

`_prepare_generation()` builds the prompt, loads the model, enforces the token budget, and returns `(model, tokenizer, prompt, max_tokens)` — shared by both entry points:

- `translate(...)` — runs `mlx_lm.generate()`, returns `str`
- `translate_stream(...)` — generator running `mlx_lm.stream_generate()`, yields segment-by-segment

`_strip_eos_token()` removes `<end_of_turn>` from the output as a safety net for the rare case it leaks past the registered EOS.

### Context window

- `CONTEXT_WINDOW = 2048` — this app's self-imposed budget for prompt and output combined, not the model's ceiling (the quant reports a far larger `max_position_embeddings`)
- `MAX_PROMPT_TOKENS = 1024` — prompt cap; `_prepare_generation()` raises `ValueError` when exceeded
- `max_tokens = CONTEXT_WINDOW - prompt_tokens` — translation gets all remaining room (EOS still stops it early)
- **No character cap.** The text area had `max_chars=5000` until 2026-09-13 as a coarse backstop on the string handed to `encode()` each rerun. It was removed for two measured reasons. The cost it guarded is negligible: the real tokenizer encodes 5,000 characters in about 1–3 ms, 500,000 in ~150 ms, 2 MB in under a second, linearly. And it had a failure mode the token budget does not: the frontend drops any edit whose *result* is still over `max_chars`, deletions included, so a swapped translation longer than the cap — English→Spanish comes back longer; 5,084 characters of formal English measured 5,424 on the real model, 2026-09-13 — landed in the area in full, showed the correct over-budget badge, and could not be trimmed except by cutting to 5,000 in one stroke; worse, any rerun that re-sent the widget's state (a selectbox change) echoed the value through `TextAreaSerde.deserialize`, which truncates to `max_chars` silently (reproduced under AppTest). Characters were never the right unit anyway: whether 5,000 of them fit under the token cap depends on the register — 4,990 characters of conversational English measured 1,377 tokens, formal long-word English 430–1,030, 5,000 Chinese characters ~4,075. The token budget is the one, language-aware limit; `test_text_area_has_no_character_cap` keeps `max_chars` out

`count_prompt_tokens(prompt, tokenizer)` returns the token length of the wrapped prompt — the Gemma chat scaffold (`<start_of_turn>user...`) is included, since that's what `build_prompt()` returns, as is the `<bos>` the tokenizer prepends. The UI is silent under budget — there is no live counter — and over budget it disables Translate and shows a red badge below it carrying the count (`Too long: 1377 / 1024 tokens`) so the user knows how much to trim — the only place the number appears on the locked Streamlit 1.63.0. `_prepare_generation`'s own `ValueError` (which would repeat the count through `st.error`) is a backstop the UI cannot reach there: since 1.61 `register_widget` enforces `disabled` server-side and discards a click on a button that registers disabled in the same rerun, so `translate_clicked` is `False`. On the 1.58–1.60 floor that click went through and the count showed twice.

### UI

`streamlit_app.py` runs top to bottom: page config → the page column, and inside it: title → session defaults → language selectors → model load → the two columns → the translate branch. The load sits below the selector row so a cold start paints the selectors before the spinner (only the token count depends on the tokenizer), and above the columns because the cache spinner renders in the current container — inside `left_col` it would sit between the text area and Translate. `test_model_load_failure_logs_and_shows_error` pins the order from the failure state: the page column holds exactly `title`, the selector row, `error`, with two selectboxes and the swap button rendered and no text area. The two UI helpers (`_swap_languages`, `_show_settled`) are defined above `set_page_config` so the `with` block is one contiguous flow. Open the file for widget kwargs; what follows is only the load-bearing parts.

- **Layout** — `layout="wide"` with the whole UI inside a centred column of `PAGE_WIDTH = 1200`, built from two native containers: `st.container(horizontal_alignment="center")` holding `st.container(width=PAGE_WIDTH)`. Two because the cap and the centring are separate — a fixed-width child sits at the left edge of its parent unless the parent centres its elements. This replaced (2026-09-12) the default centred layout, and it keeps that layout's rationale — a readable-width cap — while re-sizing it: Streamlit's 736 px column was sized for one column of 16 px prose, and halved into two panels it gave 344 px panels at 46 / 41 characters per line (text area / output), below the 45–75 reading range; the 1200 column gives 592 px panels at 85 / 73 (measured with the theme's 14 px text area and 16 px `st.text`). 1300 was measured at 91–97 / 80–81 and rejected; 1100 gives 542 px panels at 76 / 67 and is the one-constant retreat. Measured geometry on 1.63.0: 592 px panels from ~1360 wide up and at 2560+ (plain `"wide"` would give ~1190 px panels and 160+ cpl there); at 1200 wide the column clamps to 1040 (512 px panels, 73 / 62) because `"wide"` pads 80 px per side at ≥ 864 and 16 px at ≤ 800 — a frontend constant a Streamlit bump can move; `"wide"` is never narrower than centred (equal at 864 and 700); columns stack at ≤ 640 with no horizontal scroll. Levers measured and declined, so nobody re-tries them: no config option controls the main block's width or padding (`96px 16px 160px`); `ui.hideTopBar` has no layout effect; `?embed=true` saves 60 px top / 144 px bottom but removes the main menu and with it the theme switcher; `st.columns(gap=None)` gains 8 px per panel and butts the two buttons together; `height="stretch"` fills only a bounded parent, and the main block is content-height. One lever left that list on 2026-09-13: `st.header` measured *taller* than `st.title` (75.2 vs 74.4 px) only because `headingFontSizes` pinned h1 alone and left h2 at the 2.25rem stock size; with h2 at 1.5rem it would be shorter, and has not been re-measured. Guarded by `test_page_layout_is_wide`, `test_page_column_wraps_the_ui` (both wrappers precede `st.title` in the mock call order) and `test_page_column_is_the_main_blocks_only_child` (the AppTest main block has exactly one direct child).
- **Panel height** — `PANEL_HEIGHT = 400` is the one height both the text area and the output box read (`test_panels_share_one_height`), so their bottoms and the two buttons below them sit level. The vertical offsets are measured on 1.63.0 and independent of the height: the buttons' bottom is at 282.4 px + `PANEL_HEIGHT`, the over-budget badge ends 41.6 px below the buttons, and a "Translation failed" `st.error` row 74.0 px below them — the two never coexist, since over budget disables Translate (682.4 / 724.0 / 756.4 at 400, at every width). 400 fits every state on 1440×900 and 1920×1080 with no content scroll (at 1440×900 the error state scrolls `stMain` by 16 px of the block's 160 px bottom padding, not content) and shows ~14 lines of 16 px output (~19 of 14 px input); a 1366×768-class laptop (~680 px viewport) needs ≤ 320 for every state or ≤ 350 for buttons + badge — a one-number change. Stay at or under 500, the `st.container` docstring's ceiling for scrolling containers (`test_panel_height_under_the_scrolling_container_ceiling`).
- **Language selectors** — `[10, 1, 10]` columns with the swap button between them. The runtime filter that rewrites `st.session_state["target_lang"]` to a valid target **must stay above the target selectbox**. Below it the filter is dead code: `selectbox` resets a stored value outside its options to the default index and writes it back to session state (`resolve_value_against_options` on 1.63.0, `validate_and_sync_value_with_options` on the 1.58 floor — silently, no log line), so by the time the filter ran the value would already be valid and the condition never true. The app would keep working, but by a widget internal that was already reworked between those two versions rather than by its own three lines. Assigning to a widget key after its widget exists does raise `StreamlitAPIException`, but that assignment never runs. AppTest only sees the post-reset tree; the import-time `app_module_non_english_source` fixture (a French source, a target selectbox mock that does not repair) is what catches the move. Note the filter silently discards the user's target selection when the source moves off English.
- **Swap button** — `_swap_languages()` swaps source/target and, when the previous translation is non-empty, moves it into the source area and resets `translation_result` to `""` (the key is seeded with the other defaults and never deleted, so the script reads it without a fallback); disabled when target is `FROM_ENGLISH_ONLY` (the only invalid swap, since non-English sources always pair with English). The callback re-checks that condition itself and returns early — that is the backstop for a stale click, not dead code.
- **Output box** — one `st.container(height=PANEL_HEIGHT)` holding one `st.empty()`, in every state. The settled result and the stream both render through `st.text` (raw text, not markdown — matching the `text/plain` download), so the settled panel is full-strength text rather than a `disabled` text area, which Streamlit paints at 40% alpha (a disabled text area *is* selectable and copyable in Chrome — that was checked — so legibility is the whole reason). Empty state shows a muted `st.caption("Translation")` as the placeholder. On Translate the box shows a shimmering "Translating…" — `st.markdown(":shimmer[…]")`, covering the prefill wait (seconds for a long input) until the first chunk replaces it; it is markdown at full text strength rather than a caption because a caption's 0.6 opacity stacked on the shimmer's ~55% rest alpha measured 3.01:1 dark / 2.30:1 light, below AA. If generation fails before any chunk arrives, `_show_settled()` restores the previous content so the box matches what Download still offers; a mid-stream failure leaves the partial output in place. The `st.error` renders below the two-column row in both cases. Two asymmetries with the input are accepted, not bugs to fix with CSS: `st.text` is 1rem where a text area is 0.875rem (neither has a theme option), and the container is bordered but unfilled, so the output sits on `backgroundColor` while the input sits on the `secondaryBackgroundColor` well. Streaming does not auto-scroll; output past the box's height streams below the fold. On completion the result is saved to `st.session_state["translation_result"]` and `st.rerun()` re-renders the box settled and enables Download.
- **Button row** — Translate is the element directly after the `PANEL_HEIGHT` text area and Download the element directly after the `PANEL_HEIGHT` output box, in every state, so the two buttons sit level and never move (measured against the app: 642.4 px top / 682.4 bottom at 1200, 1280, 1366, 1440 and 1920 wide in all four states — empty, under budget, over budget, with a result; 542.4 top before the panels grew from 300 to 400 on 2026-09-12). The only conditional element in either column is the over-budget badge, which renders *below* Translate; under budget nothing renders at all. This replaced (2026-09-12) a live `N / 1024 tokens` caption above Translate, mirrored by an invisible `&nbsp;` caption in the right column: the pair dropped 38 px together when text was committed — the moment the user reaches for the button — and the badge had no mirror, so over budget Translate sat 42 px below Download. Alternatives were measured before settling here: an unconditional counter row (works only hosted in `st.markdown`; a `:red-badge[...]` inside `st.caption` measured +1.11 px and paints the badge at 60% alpha, ~3.5:1), a horizontal button row (the counter beside Translate, but the button's width then varied 344 → 221 → 114 px with state, at the 344 px panels of the time), and a fixed-height slot (a 101 px blank band in the empty state). The always-on counter itself was dropped the same day: a typical input is 20–300 tokens, for which the number is jargon; `st.text_area` commits on blur or ⌘-Enter, so it updated only after the user stopped typing; and the number matters in exactly one state — over budget, as the amount to trim — where the badge now carries it. Only this ordering is invariant to line-height, badge metrics, wrapping and Streamlit bumps, because nothing conditional precedes either button. `st.columns(2)` must keep its default `vertical_alignment="top"`: the left column is taller than the right when the badge shows, and `"center"` or `"bottom"` would slide Download. Guarded by `TestButtonLayout` (adjacency in the import-time call order), `TestTokenBudget` (nothing rendered under budget) and `test_buttons_directly_follow_their_panels_in_every_state` (column shapes in AppTest across all four states).
- **Session state keys the script reads or writes** — `source_lang`, `target_lang`, `translation_result`, `source_text`; the first three are seeded together in the defaults block (`translation_result` to `""`), and `source_text` is the text area's own key. The button widget keys `translate_text` and `download_text` also exist in session state but are never read
- **Widget conventions** — buttons size with `width="stretch"`; `use_container_width` is deprecated and still accepted, with no removal release named (the docstring says only "a future release"), so do not reintroduce it. The page column relies on `st.container`'s `width=<int>` and `horizontal_alignment` kwargs; both exist on the 1.58 floor (checked 2026-09-12 with `uv run --no-project --with streamlit==1.58.0`, where the suite's only failures are the four pre-existing `AppTest.download_button` ones — 1.58's AppTest has no such accessor — so the app's floor holds even though the documented `--frozen` gate does not). The page icon and the `st.error`/`st.warning` callouts use Material Symbols (`:material/...:`).

### Theme

`.streamlit/config.toml` is the "Native" theme: system font, macOS window and text-field neutrals, apple.com blue for the one primary button, with both a light and a dark palette. Thirty-nine lines of config, no code. The anchor colours are Apple HIG and apple.com constants; the callout text colours, the light yellow and the dark well are tuned from them for contrast, and the comments name the rendering fact each tuned value answers — read the file for the values; what follows is why it exists and what it relies on.

It replaced the stock themes on 2026-09-11 for two problems specific to this UI. Stock primary is `#FF4B4B`, the same hue as the red "Too long…" badge and every `st.error`, so the call to action and the failure state looked alike — and white on it is only 3.30:1. And the translation was rendered as a `disabled=True` text area, which Streamlit paints as `textColor` at 40% alpha over `secondaryBackgroundColor` (`fadedText40` in the frontend); stock light mode landed at 2.17:1 there, and no config can get a disabled text area past 3.85:1 dark / ~2.86:1 light. That one was fixed in the app, not the theme: the settled result now renders through the same bordered `st.container(height=PANEL_HEIGHT)` + `st.text` path as the stream (see Architecture → UI → Output box), at 17:1 dark / 21:1 light. The pure white / pure black ink and the wells sunk just below the page remain from that constraint and still serve the input well and the selectboxes.

Facts the file relies on, all verified against Streamlit 1.63.0:

- **Both `[theme.light]` and `[theme.dark]` are present and non-empty by policy, not by Streamlit's rule.** Streamlit builds the Light / Dark / System switcher whenever *either* mode section has a value and hides it only for a `[theme]`-only block. But a mode with no section falls back to the stock palette (`#0E1117` for dark) under the shared `[theme]` font, radius and borders — so both stay designed. Shared typography and shape in `[theme]` are inherited by both.
- **`base` is only valid directly under `[theme]`.** Inside a mode section it is an unknown option: Streamlit logs it once at startup and ignores it. `TestThemeConfig` checks every key against `streamlit.config.get_config_options()`, which registers `theme.dark.primaryColor` but not `theme.dark.base`.
- **`--theme.base dark` is ignored once either mode section is non-empty.** The mode follows the browser's `prefers-color-scheme` or the user's switcher choice — which is why the screenshot recipe emulates the colour scheme in Playwright instead of passing a flag.
- **`st.text` renders in the body font and text colour**, not the code font, so the streamed and the settled output share a face with the input text area and there are no `codeFont`/`codeTextColor` keys. If a Streamlit bump moves `st.text` back to the code font, streaming turns green and monospace (`codeTextColor` defaults to `greenTextColor`); add `codeTextColor = textColor` then.
- **`redColor`/`yellowColor` are overridden and `redTextColor`/`yellowTextColor` pinned.** On this theme's `#1C1C1E` window and `#FFFFFF` page the stock callout text clears AA by only 0.02 (dark) and 0.08 (light), so the semantic colours are Apple's; and the text colours are written out rather than left to Streamlit's ±15% lightness derivation — dark red text lifted to 7:1 on its tint, light red text darkened to widen the red/yellow gap under deuteranopia, yellow pinned at exactly the derived value so a change to the derivation cannot move it. The callouts and the red badge are the only things that paint these (the text area's `max_chars` counter did too, until the cap was removed).
- **A shorter `headingFontSizes` array pins only the levels it names.** The server forwards the list as given and the frontend overrides `h{i}FontSize` per entry over the stock 2.75 / 2.25 / 1.75 / 1.5 / 1.25 / 1rem, so the earlier `["2rem"]` left h2 at 2.25rem — larger than h1. All six are pinned, decreasing, with h1 unchanged at 2rem, although only `st.title` is used. `headingFontWeights` is different: the server pads a short list to six with 600, so `[600]` is already a full scale. Numeric sizes and `baseRadius` are in rem, as both option descriptions recommend; the radius is 0.375rem, the 6 px macOS control radius at the 16 px root.
- **`font` is passed through verbatim.** `theme.font`'s documented forms are the three keywords, a `[[theme.fontFaces]]` family, or `name:url`; a plain family list like `-apple-system, system-ui` works because the frontend concatenates the string as given and appends `"Source Sans", sans-serif`. A release that validates font names would drop the value and fall back to Source Sans with every test green — check the computed `font-family` after a Streamlit bump.
- **Apple's own system blues fail AA under white text** (`#007AFF` 4.02:1, `#0A84FF` 3.65:1); the apple.com web blues (`#0066CC` / `#0071E3`) are tuned for exactly that and are what the file uses.
- The stock palettes are still untranscribable — `config.py` declares the colour options with no default values and the real ones live in the frontend bundle — so this file is a full custom theme on purpose, not an attempt to reproduce a default. Do not add a `[theme]` "restoring" a stock value; it would be hand-transcribed and drift.

`.gitignore` ignores `.streamlit/*` and un-ignores `config.toml`, so a local edit to the theme shows in `git status` rather than being silently dropped.

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

This is safe only because the raw string reproduces the trained format exactly. `build_prompt()`'s instruction text is byte-identical to the quant's own `chat_template.jinja`; the sole difference across the whole prompt is that `apply_chat_template` emits a leading `<bos>` and `build_prompt()` does not — the tokenizer supplies it instead. The mechanism is the quant's `tokenizer.json`, which ships a `TemplateProcessing` post-processor (`single = [<bos>, $A]`) that `encode()` applies whenever `add_special_tokens=True` — the default, and what mlx-lm passes for any prompt that does not already start with `<bos>`. The `add_bos_token: true` in `tokenizer_config.json` is inert: transformers v5 discards that key whenever a `tokenizer.json` is present, so `tokenizer.add_bos_token` reads `False` even though `<bos>` is prepended — the attribute is not evidence either way. Anything that bypasses the post-processor (`encode(..., add_special_tokens=False)`, a different runtime) silently drops `<bos>`. Re-verify after a quant, `transformers`, or `tokenizers` bump by rendering both and diffing the instruction text, and by checking `encode()` yields exactly one `<bos>`.

### Chinese uses `zh-CN`, not `zh`

The locale code matches the TranslateGemma Technical Report (Table 5). Since prompts are built manually, the code is inserted as text — and the model was trained with these locale codes.

### `mlx-lm` below 0.31.3 breaks on Streamlit's thread

With `mlx` 0.32, `mlx-lm` 0.31.1 and 0.31.2 raise `RuntimeError: There is no Stream(gpu, 0) in current thread` from `wired_limit()` in `mlx_lm/generate.py` — generation runs on Streamlit's ScriptRunner thread, not the main thread. Translation returns empty; the app itself loads fine. Hence the `mlx-lm>=0.31.3` floor.

The mocked layers cannot catch this: they replace `mlx_lm` with a `MagicMock`, so no real generation ever runs. `tests/test_live_model.py` exists for exactly this failure — **run `uv run pytest -m live` after any `mlx`/`mlx-lm` bump.**

## Testing

Two mocked layers, a plain unit layer, and a config guard, ~1.5s combined for 107 tests at 100% coverage, plus one opt-in live test that runs against the real model:

- **Import-time tests** — swap `sys.modules["streamlit"]` and `sys.modules["mlx_lm"]` for `MagicMock`s, import `streamlit_app.py`, then assert on captured `st.*` calls. No Streamlit runtime runs. Covers pure functions, layout (`layout="wide"`, the two page-column wrappers preceding `st.title`, `st.container` called exactly three times in order with the output box the only fixed-height one, both panels sharing `PANEL_HEIGHT`), panel/button adjacency in the `st.*` call order, the target filter running before the target selectbox (a second import with a French source), token counting, EOS stripping. `test_st_empty_is_created_inside_the_container` indexes from `call.container(height=PANEL_HEIGHT)`, not the first `container().__enter__()` — that one is the page column's.
- **End-to-end tests** (`TestStreamingClickPath`) — drive the real script via `streamlit.testing.v1.AppTest` with only `mlx_lm` mocked. Reaches branches the import-time tests can't: streaming click path and the settled `st.text` re-render after it, the content columns' shape in every state, the page column being the main block's only direct child (checked in the empty, warning and failure states — the last two live in the `if translate_clicked:` tail, the block a dedent at the end of the file would strand), model-load failure (which also pins the render order: the selector row survives it, the panels are never reached), runtime target filtering, swap-button wiring, empty-text warning. Every `.run()` asserts `not at.exception` afterwards — structurally: the fixtures build a `_CheckedAppTest` subclass that overrides `AppTest._run`, the one method both `at.run()` and a widget's `.run()` reach (`Element.run` → `ElementTree.run` → `AppTest._run`; `AppTest.run` itself is *skipped* by widget runs, so overriding it would miss 21 of the 23 sites — that was checked the hard way), so a test written with the plain `.click().run()` idiom cannot bypass it. `_run` is private harness API; `TestCheckedAppTest` drives a `tmp_path` script that raises on a button click through a `_CheckedAppTest` and expects the widget `.run()` to fail, so a rename fails one test loudly rather than silently dropping the check from all of them. A crash the script does not catch would otherwise pass any test whose assertions on session state or the element tree still hold (checked: with a `RuntimeError` appended to the end of the script, `test_non_english_source_restricts_target_to_english`, `test_swap_button_swaps_source_and_target` and `test_translate_click_streams_into_session_state` all still passed on their own assertions). The AppTest tree has the two page-column `flex_container` wrappers above the columns; the helpers search `app_test.columns`, which is recursive, so they are unaffected. The `_is_output_box` helper matches the box structurally — the only element inside the columns with a pixel height — so the AppTest layer needs no copy of `PANEL_HEIGHT` (it cannot import `streamlit_app`; that would run the script); the exact number is pinned by `TestOutputBox`.
- **Language-table tests** (`tests/test_languages.py`) — 19 tests across 6 classes, mocking nothing; a bare `from languages import ...` inside each test. Assert the 225 / 70 / 295 / 294 counts, per-code samples (including `Chinese` → `zh-CN`), key non-overlap, code uniqueness, and sort order.
- **Theme-config guard** (`TestThemeConfig`) — tests on the *shape* of `.streamlit/config.toml`, not its palette: it parses with a `[theme]` table; `[theme.light]` and `[theme.dark]` are both present and non-empty (this repo's policy — see Architecture → Theme for why one alone is a trap); every key under `[theme]` is in `streamlit.config.get_config_options()`, so a `base` inside a mode section or a typo fails here instead of being logged once at startup and ignored (other sections of the file are not walked); every `*Color` value is `#RRGGBB`, because the frontend drops a malformed colour with a console warning and paints the stock palette; and `headingFontSizes` names all six levels in decreasing rem values — in `[theme]` and in any mode section that carries it, since the key is registered per mode too — because a shorter array pins only the levels it names (see Architecture → Theme). Five tests. It reads the real `streamlit.config` — the import-time mocks are restored before any test runs.
- **Live-model test** (`tests/test_live_model.py`, `@pytest.mark.live`) — the only test with `mlx_lm` unmocked; drives AppTest against the real 3.9 GB quant and asserts a non-empty, EOS-free translation. **Deselected by default** via `-m "not live"` in `addopts`, so neither `uv run pytest` nor CI touches it.

Because the app catches load and generation failures and renders `st.error`, the live test asserts on `at.error` as well as `at.exception` after the initial run and after the Translate click (nothing after the `set_value` rerun, which cannot fail on its own); checking only the latter misreports a caught failure — after the first run as a `KeyError` from the text-area lookup (`st.stop()` ran before the text area), after the second as an empty translation, the `wired_limit()` signature the test looks for next — and the real cause is lost.

**Fixtures (`tests/conftest.py`):**

- `_clear_streamlit_caches` (autouse) — clears `st.cache_resource` before each test; required because Streamlit's resource cache is process-global
- `app_module` (session) — mocked-import setup for the import-time tests, via the `_import_app(source_lang)` helper; `app_module_non_english_source` (session) is the same import with the source selectbox returning French, for the target-filter order test. The target selectbox mock returns whatever `session_state["target_lang"]` holds when it is called and records it on `st.target_lang_at_target_selectbox`. Its `cache_resource` stand-in is an identity decorator in both forms (bare and `@st.cache_resource(show_spinner=...)`) that records the decorator kwargs for `test_cache_decorator_owns_the_loading_spinner`. The `sys.modules` swap is restored in a `finally`: if the mocked import raises, a leaked `MagicMock` streamlit would make every AppTest and theme test error in `_clear_streamlit_caches`, burying the one failure that matters. Its `st.columns` mock is **positional**: a hard-coded iterator matching the `[10, 1, 10]` selector row then the `columns(2)` content row, falling back to fresh `MagicMock`s once exhausted. Adding, removing, or reordering an `st.columns(...)` call in `streamlit_app.py` hands the wrong mock to the wrong region and fails in a way that looks unrelated to the edit. `st.container` (three calls: the two page-column wrappers, then the output box) is not sequenced — every call returns the same `MagicMock` — so `TestOutputBox` pins that order itself. The app's one Streamlit submodule import (`DeltaGenerator`, annotation only) sits under `if TYPE_CHECKING:` so that nothing real has to be cached in `sys.modules` for the swap to hold — a runtime `from streamlit.delta_generator import ...` resolves against a `MagicMock` parent only if the submodule is already cached, which a full run got by accident from `test_live_model.py`'s `streamlit.testing.v1` import at collection, and `uv run pytest tests/test_streamlit_app.py` alone did not (78 collection errors). conftest now imports `streamlit.testing.v1` at module level for `_CheckedAppTest`, so the submodule is cached in every run; the guard in the app is still what keeps nothing depending on that.
- `mock_tokenizer` — `encode()` returns 50 tokens, under the budget cap
- `patched_translate` — patches `load_model`, `generate`, `stream_generate`; exposes the mocks for per-test configuration
- `fake_mlx_lm` — `mlx_lm` mock injected into `sys.modules` for AppTest fixtures
- `app_test` — a `_CheckedAppTest` pre-run to its settled state; that first run and every later `.run()`, widget runs included, assert `not at.exception` through the `_run` override. `_CheckedAppTest` is built through `AppTest`'s own constructor because `from_file` hardcodes `AppTest(...)`; with an absolute path `from_file` only adds an eager `is_file()` check, so a wrong `_APP_PATH` would fail at the first `.run()` instead of at construction
- `app_test_unrun` — AppTest not yet run; for tests that configure mocks before the first `.run()` (e.g. load failure)
- `crashing_app_test` — a `_CheckedAppTest` over a three-line `tmp_path` script that raises only on its button click; not the app, it exists for `TestCheckedAppTest`

AppTest fixtures use `default_timeout=10` (the live test uses `600`, to cover a cold model load), so a hung mocked test fails at 10s rather than pytest's default.

**Pytest config (`pyproject.toml`):** `addopts = ["-ra", "--strict-markers", "--strict-config", "-m", "not live"]`, `xfail_strict = true`, `filterwarnings = ["error"]`. Coverage sources in `[tool.coverage.run]`.

100% coverage is currently true but **unenforced** — there is no `fail_under`, and CI never runs `--cov`. A change that drops coverage still goes green.

## CI

`.github/workflows/ci.yml` — two jobs, `test` then `release`.

`test` — `uv sync --locked`, then `ruff check` + `ruff format --check` + `ty` + `pytest` — on `macos-latest` (an Apple Silicon image is required for `mlx-lm`) for every push to `main` and PR. `--locked` is the gate that catches a `pyproject.toml` floor bump landing without a matching `uv.lock`; `enable-cache: false` is deliberate, since `setup-uv` v9 defaults it to `"auto"` with `prune-cache` off and the unpruned cache is ~620 MB against a ~4 s install. `setup-uv` publishes no bare major tag past v7, so the version is pinned in full (`@v9.0.0`).

## Releases

Releases are cut by the `release` job in `.github/workflows/ci.yml`. **Bumping `version` in `pyproject.toml` is the entire release action** — push the bump to `main` and, once `test` passes, the job tags the commit and publishes a public GitHub Release with auto-generated notes. There is nothing to run by hand, which is the point: pushes via GitHub Desktop skip tags, so tagging had to move into CI.

- **Runs on `ubuntu-latest`**, unlike `test` — no MLX is needed on this path and macOS runners bill at 10×. It installs no Python and never runs uv, so the version-reading step depends on the runner image's system `python3` being ≥3.11 for `tomllib`.
- **Trigger** — `needs: test` plus `if: github.event_name == 'push' && github.ref == 'refs/heads/main'`. PRs reach the job and skip it, and a red `test` blocks the release entirely.
- **Bump detection is tag existence, not a diff.** The job reads `[project].version` with `tomllib` and asks the remote whether `v$VERSION` is already tagged (`git ls-remote`, since checkout is shallow and fetches no tags). Diffing `pyproject.toml` against the parent commit would break on workflow re-runs, on squash merges, and on the bumps here that ride along with unrelated changes in a single commit. Tag existence answers the real question — *is this version released?* — and is idempotent, so every push to `main` between bumps is a no-op.
- **Version/lockfile drift is already covered.** `uv.lock` records the project's own version, so a bump without a matching `uv lock` fails `uv sync --locked` in `test`; the release job needs no check of its own.
- **`permissions: contents: write` is required at the job level.** The repo's `default_workflow_permissions` is `read`, so the token is read-only unless a job asks for more; without it `gh release create` fails with a 403.
- **`concurrency: {group: release, cancel-in-progress: false}`** queues rather than cancels, so two pushes landing together cannot race to create the same tag and a half-finished release is never killed.
- **`gh release create --target "$GITHUB_SHA"` creates the tag as part of the release** — a lightweight tag, matching `v0.13.1`/`v0.14.0`. `--generate-notes` builds the body from merged PRs since the previous release — with direct pushes, as here, that is just the compare link.
- The tag value reaches the shell through `env:` rather than `${{ }}` interpolation, so a crafted `pyproject.toml` version cannot break out into the run script.

The publish path first ran for real on the `v0.15.0` bump (2026-08-10): `github-actions[bot]` created the release with generated notes. `v0.13.1` and `v0.14.0` were created by hand before the job landed.

To reword a release afterwards, `gh release edit vX.Y.Z --notes "..."` (or the GitHub UI) — the job never touches a release that already exists.

## Screenshots

`assets/screenshot-dark.png` is the README's only screenshot: 2400×1450, a 1200×725 viewport at `device_scale_factor=2`. There is no capture script in the repo — this section is the recipe, because every step of it is a trap.

Run the app. Streamlit reads `.streamlit/config.toml` from beside `streamlit_app.py` as well as from the cwd, so any cwd works with a path to the script. Do not pass `--theme.base dark`: with a mode section defined the flag is ignored (see Architecture → Theme), and the mode is chosen by the browser's colour scheme, which Playwright emulates below.

```sh
uv run streamlit run streamlit_app.py --server.port 8501 --server.headless true
```

Then drive it with Playwright via `uv run --with playwright python`, using `channel="chrome"`:

This block is complete and runnable as written — keep it that way, and keep it
ruff-formatted, because `ruff format --check .` formats Python inside Markdown
fences and CI gate #2 fails on a hand-aligned snippet.

```python
from playwright.sync_api import sync_playwright

TEXT = (
    "Good morning! I would like to book a table for two at seven o'clock. "
    "If possible we would prefer a table by the window, and we would like to "
    "know whether you have vegetarian options on tonight's menu."
)
HIDE = """
document.querySelectorAll(
    '[data-testid="stToolbar"],[data-testid="stStatusWidget"]'
).forEach((e) => (e.style.display = "none"))
"""

with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome")
    ctx = browser.new_context(
        viewport={"width": 1200, "height": 725},
        device_scale_factor=2,
        color_scheme="dark",  # selects [theme.dark]; --theme.base cannot
    )
    page = ctx.new_page()
    # Both waits must cover the cold model load, not just the selector one.
    page.goto("http://localhost:8501", wait_until="networkidle", timeout=180_000)
    page.wait_for_selector("textarea", timeout=180_000)

    page.locator("textarea").first.fill(TEXT)
    page.keyboard.press("Tab")  # commit the text_area
    page.wait_for_timeout(2_000)  # let the rerun settle
    page.get_by_role("button", name="Translate").click()

    # Streaming ends when st.rerun() enables the Download button — valid for
    # the first translation of a session only; a prior result keeps Download
    # enabled while the next one streams.
    page.wait_for_function(
        "() => [...document.querySelectorAll('button')]"
        ".some(b => /Download/.test(b.textContent) && !b.disabled)",
        timeout=300_000,
    )

    page.evaluate(HIDE)
    page.evaluate("document.activeElement && document.activeElement.blur()")
    page.mouse.move(0, 0)
    page.wait_for_timeout(800)
    page.screenshot(path="assets/screenshot-dark.png")
    browser.close()
```

- **`device_scale_factor=2` is not optional.** The dev machine is a non-retina 1920×1080 display reporting `devicePixelRatio: 1`, so macOS `screencapture` and the Chrome extension's screenshot both yield 1× — half the asset's resolution. Playwright synthesizes 2× regardless of the physical display.
- **`channel="chrome"` avoids a browser download.** The cached Playwright build drifts from whatever version `uvx`/`--with` resolves (1228 vs 1234 at time of writing), and the mismatch triggers a ~150 MB `playwright install`. Driving the installed Chrome sidesteps it.
- **Blur *and* move the mouse.** After the click the button keeps focus (focus ring) and the virtual mouse stays parked on it (`:hover` darkens the button). Both survive into the still. A correct capture samples `#0071E3` on the Translate button and `#1C1C1E` on the background — the theme's own `[theme.dark]` values, which proves both that the config was picked up and that `color_scheme="dark"` selected the dark palette. A `#FF4B4B` button means the config was not found.
- **Re-measure the height after any layout change.** The viewport height is tuned to end ~42 CSS px below the buttons (their bottom is at 682.4 CSS px in the screenshot state); it is not a stable constant. Dropping the Material theme grew the page by ~45 CSS px at the buttons (its `baseFontSize = 14` and 36px title against Streamlit's 16 and 2.75rem), the Native theme then shrank it by ~14 (a 2rem title against the 2.75rem default), removing the live token caption (2026-09-12) took out one caption row plus its gap (~38), and the page column landed the same day with `PANEL_HEIGHT` 300 → 400 (+100), which is why the height went 676 → 663 → 625 → 725. Measure the buttons' `getBoundingClientRect().bottom` at 1200 wide before trusting the number.
- **The capture is 1200 wide on purpose, and shows 512 px panels, not the designed 592.** At 1200 the page column clamps to 1040 (see Architecture → UI → Layout). A 1440×725 capture (2880×1450) would show the 592 px panels, but GitHub scales the README image down to its content column (narrower than either asset), so the wider asset renders the UI text 2400/2880 ≈ 17% smaller; legibility won. The error state (756.4) does not fit a 725 viewport (the badge, at 724, does by a pixel) — the recipe only captures the result state, so that is fine.
- **`section.stMain` is the scroll container, not the document.** It has `overflow-y: auto` and the main block carries 160 px of bottom padding, so `document.documentElement.scrollHeight > innerHeight` is a false negative for "does the page scroll" (the old 625 viewport was already scrolling `stMain` by 117 px of padding). Check `stMain.scrollHeight - stMain.clientHeight` instead; a screenshot is unaffected either way because it captures the viewport at scroll 0.
- **Streamlit commits a `text_area` on blur**, so `fill()` then `Tab`, and wait for the rerun before clicking Translate — clicking too early lands on a stale widget tree.

## Hooks

`.claude/settings.json` is git-tracked, so its hooks apply to every clone rather than one machine. Two hooks, both sub-100ms. A file watcher picks up edits to the file mid-session; `/hooks` shows what is actually live and which settings file it came from.

- **`PreToolUse` on `Edit|Write`** — denies writes to `uv.lock`, `.env`, and `.streamlit/secrets.toml`. Change `uv.lock` through uv (`uv add` / `uv lock` / `uv sync`); the two gitignored secret files are edited by hand. The `case` matches the bare filename with an optional directory prefix, so `.env.example` and `uv.lock.bak` pass through. A deny is signalled by a JSON payload on **stdout with exit 0**, not by `exit 2`.
- **`PostToolUse` on `Edit|Write`** — runs `ruff format` then `ruff check --fix` on the edited file when it ends in `.py`, via `uv run --no-sync` so it never syncs or re-locks the environment (see Dependencies for why that matters mid floor-verification). It is a convenience, not a gate: the command ends in `|| true`, so failures are printed and swallowed, and `--fix` only repairs *fixable* rules. `uv run ruff check .` still has to pass before pushing.

**No hook runs the tests or the type checker.** Two `Stop` hooks used to, and were removed deliberately: `Stop` fires once per *turn* rather than once per *change*, so conversational turns ran the full suite and a whole-project `ty check` against code nobody touched — and `exit 2` on `Stop` prevents the turn from ending, letting an unrelated or pre-existing failure hijack the conversation. Run the gate explicitly after changing Python; otherwise CI is the first thing that sees a failure. Do not reinstate them as `Stop` hooks.

Hooks are the one part of this repo with no test and no CI signal — nothing validates the shell embedded in `settings.json`, and it survives two layers of escaping. After editing one, replay it from the file rather than from the string you meant to write:

```sh
CMD=$(jq -r '.hooks.PreToolUse[0].hooks[0].command' .claude/settings.json)
printf '{"tool_input":{"file_path":"uv.lock"}}' | sh -c "$CMD"   # expect a deny payload
printf '{"tool_input":{"file_path":"pyproject.toml"}}' | sh -c "$CMD"  # expect no output
```

## Prompt Template

`build_prompt()` in `streamlit_app.py` is the source of truth — this is a rendering of it, with the function's own parameter names as placeholders. The `<start_of_turn>user` / `<end_of_turn>` / `<start_of_turn>model` scaffold wraps what follows; `\n` below is a literal newline in the string, and the line breaks are cosmetic.

```
You are a professional {src_lang} ({src_code}) to {tgt_lang}
({tgt_code}) translator. Your goal is to accurately convey the meaning and
nuances of the original {src_lang} text while adhering to {tgt_lang} grammar,
vocabulary, and cultural sensitivities.\nProduce only the {tgt_lang}
translation, without any additional explanations or commentary. Please translate
the following {src_lang} text into {tgt_lang}:\n\n\n{text}
```

One known divergence from the model's own template: it applies `| trim` to the user text, while `build_prompt()` interpolates `{text}` raw, so pasted leading/trailing whitespace reaches the model and inflates the token count.

## Resources

- [Technical Report](https://arxiv.org/pdf/2601.09012)
- [Gemma Cookbook](https://colab.research.google.com/github/google-gemini/gemma-cookbook/blob/main/Research/[TranslateGemma]Example.ipynb)
- [Streamlit AppTest reference](https://docs.streamlit.io/develop/api-reference/app-testing)
