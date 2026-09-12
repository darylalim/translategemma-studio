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
- `uv run pytest --cov` — run tests with coverage (sources configured in `pyproject.toml`); roughly doubles the runtime, ~1s → ~2s

The four CI gates as one command — this is "the gate" referred to below, and what to run before pushing:

```sh
uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest
```

**A cold cache downloads the model first.** Both `uv run streamlit run streamlit_app.py` and `uv run pytest -m live` block on a ~3.9 GB pull from the Hugging Face Hub into `~/.cache/huggingface/hub` before anything happens, which looks indistinguishable from a hang. Check whether it is already there:

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
- **The target-language filter's position** — it must stay above the target selectbox. See Architecture → UI.

## Code Style

- `snake_case` for functions and variables, `PascalCase` for classes
- Type annotations on all parameters and returns
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

`load_model()` returns `(model, tokenizer)`, cached with `@st.cache_resource`. Loads `mlx-community/translategemma-4b-it-8bit` via `mlx_lm.load()` and registers `<end_of_turn>` as an EOS token so generation stops early instead of running to the `max_tokens` cap.

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
- `MAX_INPUT_CHARS = 5000` — coarse text-area backstop; the token counter is the real, language-aware limit

`count_prompt_tokens(prompt, tokenizer)` returns the token length of the wrapped prompt — the Gemma chat scaffold (`<start_of_turn>user...`) is included, since that's what `build_prompt()` returns, as is the `<bos>` the tokenizer prepends. The UI shows a live token count under the input and disables Translate when over budget.

### UI

`streamlit_app.py` runs top to bottom: page config → session defaults → model load → language selectors → the two columns → the translate branch. Open the file for widget kwargs; what follows is only the load-bearing parts.

- **Layout** — default centered (no `layout=` kwarg); a focused two-panel tool relies on the readable-width cap, guarded by `test_page_layout_is_centered`
- **Language selectors** — `[10, 1, 10]` columns with the swap button between them. The runtime filter that rewrites `st.session_state["target_lang"]` to a valid target **must stay above the target selectbox** — assigning to a widget key after its widget exists raises `StreamlitAPIException`, and only the AppTest layer catches it. Note the filter silently discards the user's target selection when the source moves off English.
- **Swap button** — `_swap_languages()` swaps source/target and moves the previous translation into the source area; disabled when target is `FROM_ENGLISH_ONLY` (the only invalid swap, since non-English sources always pair with English). The callback re-checks that condition itself and returns early — that is the backstop for a stale click, not dead code.
- **Output placeholder** — one `st.empty()` holds either the streaming container or the settled, disabled text area. Streaming writes through `st.text` (raw text, not markdown — matching the text area and the `text/plain` download); on completion the result is saved to `st.session_state["translation_result"]` and `st.rerun()` reverts the slot.
- **Captions** — the left column's token counter and the right column's `&nbsp;` spacer are both gated on `text.strip()`, so they appear and disappear together. Rendering the spacer unconditionally misaligns the Translate and Download buttons whenever the input is empty — the exact bug it exists to prevent.
- **State seeding** — output text areas are populated via session state (not the `value=` parameter) to avoid stale widget state
- **Session state keys** — `source_lang`, `target_lang`, `translation_result`, `source_text`, `text_output`
- **Widget conventions** — buttons size with `width="stretch"`; `use_container_width` is deprecated and still accepted, with no removal release named (the docstring says only "a future release"), so do not reintroduce it. The page icon and the `st.error`/`st.warning` callouts use Material Symbols (`:material/...:`).

### Theme

`.streamlit/config.toml` is the "Native" theme: system font, macOS window and text-field neutrals, apple.com blue for the one primary button, with both a light and a dark palette. Thirty-five lines of config, no code. The anchor colours are Apple HIG and apple.com constants; the callout text colours, the light yellow and the dark well are tuned from them for contrast, and the comments name the rendering fact each tuned value answers — read the file for the values; what follows is why it exists and what it relies on.

It replaced the stock themes on 2026-09-11 for two problems specific to this UI. Stock primary is `#FF4B4B`, the same hue as the red "Too long to translate" badge and every `st.error`, so the call to action and the failure state looked alike — and white on it is only 3.30:1. And the translation is rendered as a `disabled=True` text area, which Streamlit paints as `textColor` at 40% alpha over `secondaryBackgroundColor` (`fadedText40` in the frontend); stock light mode lands at 2.17:1 there. Pure white / pure black ink with the wells sunk just below the page reaches 3.83:1 dark and 2.82:1 light — 0.02 and 0.04 under the ceiling any config can reach (3.85 / ~2.86). **No theme can get the light-mode output panel to 3:1.** The fix is app code — rendering the settled result through the bordered `st.container(height=300)` + `st.text` path the streaming branch already uses (21:1 light, 17:1 dark, and selectable) — and it has not been done.

Facts the file relies on, all verified against Streamlit 1.63.0:

- **Both `[theme.light]` and `[theme.dark]` are present and non-empty by policy, not by Streamlit's rule.** Streamlit builds the Light / Dark / System switcher whenever *either* mode section has a value and hides it only for a `[theme]`-only block. But a mode with no section falls back to the stock palette (`#0E1117` for dark) under the shared `[theme]` font, radius and borders — so both stay designed. Shared typography and shape in `[theme]` are inherited by both.
- **`base` is only valid directly under `[theme]`.** Inside a mode section it is an unknown option: Streamlit logs it once at startup and ignores it. `TestThemeConfig` checks every key against `streamlit.config.get_config_options()`, which registers `theme.dark.primaryColor` but not `theme.dark.base`.
- **`--theme.base dark` is ignored once either mode section is non-empty.** The mode follows the browser's `prefers-color-scheme` or the user's switcher choice — which is why the screenshot recipe emulates the colour scheme in Playwright instead of passing a flag.
- **`st.text` renders in the body font and text colour**, not the code font, so streaming and the settled text area share a face and there are no `codeFont`/`codeTextColor` keys. If a Streamlit bump moves `st.text` back to the code font, streaming turns green and monospace (`codeTextColor` defaults to `greenTextColor`); add `codeTextColor = textColor` then.
- **`redColor`/`yellowColor` are overridden and `redTextColor`/`yellowTextColor` pinned.** On this theme's `#1C1C1E` window and `#FFFFFF` page the stock callout text clears AA by only 0.02 (dark) and 0.08 (light), so the semantic colours are Apple's; and the text colours are written out rather than left to Streamlit's ±15% lightness derivation — dark red text lifted to 7:1 on its tint, light red text darkened to widen the red/yellow gap under deuteranopia, yellow pinned at exactly the derived value so a change to the derivation cannot move it. The callouts, the red badge and the text area's `max_chars` counter (which blinks `redTextColor` at 5000/5000) are the only things that paint these.
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

Two mocked layers, a plain unit layer, and a config guard, ~1s combined for 88 tests at 100% coverage, plus one opt-in live test that runs against the real model:

- **Import-time tests** — swap `sys.modules["streamlit"]` and `sys.modules["mlx_lm"]` for `MagicMock`s, import `streamlit_app.py`, then assert on captured `st.*` calls. No Streamlit runtime runs. Covers pure functions, layout, token counting, EOS stripping.
- **End-to-end tests** (`TestStreamingClickPath`) — drive the real script via `streamlit.testing.v1.AppTest` with only `mlx_lm` mocked. Reaches branches the import-time tests can't: streaming click path, model-load failure, runtime target filtering, swap-button wiring, empty-text warning.
- **Language-table tests** (`tests/test_languages.py`) — 19 tests across 6 classes, mocking nothing; a bare `from languages import ...` inside each test. Assert the 225 / 70 / 295 / 294 counts, per-code samples (including `Chinese` → `zh-CN`), key non-overlap, code uniqueness, and sort order.
- **Theme-config guard** (`TestThemeConfig`) — three tests on the *shape* of `.streamlit/config.toml`, not its colours: it parses with a `[theme]` table; `[theme.light]` and `[theme.dark]` are both present and non-empty (this repo's policy — see Architecture → Theme for why one alone is a trap); and every key under `[theme]` is in `streamlit.config.get_config_options()`, so a `base` inside a mode section or a typo fails here instead of being logged once at startup and ignored (other sections of the file are not walked). It reads the real `streamlit.config` — the import-time mocks are restored before any test runs.
- **Live-model test** (`tests/test_live_model.py`, `@pytest.mark.live`) — the only test with `mlx_lm` unmocked; drives AppTest against the real 3.9 GB quant and asserts a non-empty, EOS-free translation. **Deselected by default** via `-m "not live"` in `addopts`, so neither `uv run pytest` nor CI touches it.

Because the app catches generation failures and renders `st.error`, the live test asserts on `at.error` as well as `at.exception`; checking only the latter turns a real failure into a downstream `KeyError`.

**Fixtures (`tests/conftest.py`):**

- `_clear_streamlit_caches` (autouse) — clears `st.cache_resource` before each test; required because Streamlit's resource cache is process-global
- `app_module` (session) — mocked-import setup for the import-time tests. Its `st.columns` mock is **positional**: a hard-coded iterator matching the `[10, 1, 10]` selector row then the `columns(2)` content row, falling back to fresh `MagicMock`s once exhausted. Adding, removing, or reordering an `st.columns(...)` call in `streamlit_app.py` hands the wrong mock to the wrong region and fails in a way that looks unrelated to the edit.
- `mock_tokenizer` — `encode()` returns 50 tokens, under the budget cap
- `patched_translate` — patches `load_model`, `generate`, `stream_generate`; exposes the mocks for per-test configuration
- `fake_mlx_lm` — `mlx_lm` mock injected into `sys.modules` for AppTest fixtures
- `app_test` — AppTest pre-run to its settled state
- `app_test_unrun` — AppTest not yet run; for tests that configure mocks before the first `.run()` (e.g. load failure)

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

`assets/screenshot-dark.png` is the README's only screenshot: 2400×1326, a 1200×663 viewport at `device_scale_factor=2`. There is no capture script in the repo — this section is the recipe, because every step of it is a trap.

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

TEXT = "Good morning! I would like to book a table for two at seven o'clock."
HIDE = """
document.querySelectorAll(
    '[data-testid="stToolbar"],[data-testid="stStatusWidget"]'
).forEach((e) => (e.style.display = "none"))
"""

with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome")
    ctx = browser.new_context(
        viewport={"width": 1200, "height": 663},
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

    # Streaming ends when st.rerun() restores the disabled output text area.
    page.wait_for_function(
        "() => { const t = [...document.querySelectorAll('textarea')];"
        " return t.length > 1 && t[1].value.trim().length > 0; }",
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
- **Re-measure the height after any layout change.** The viewport height is tuned to end ~42 CSS px below the buttons; it is not a stable constant. Dropping the Material theme grew the page by ~45 CSS px at the buttons (its `baseFontSize = 14` and 36px title against Streamlit's 16 and 2.75rem), and the Native theme then shrank it by ~14 (a 2rem title against the 2.75rem default), which is why the height went 676 → 663. Measure the buttons' `getBoundingClientRect().bottom` at 1200 wide before trusting the number.
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
