import logging
from collections.abc import Iterator
from typing import Any

import streamlit as st
from mlx_lm import generate, load, stream_generate
from streamlit.delta_generator import DeltaGenerator

from languages import (
    ALL_LANGUAGES,
    FROM_ENGLISH_ONLY,
    SOURCE_LANGS,
    TARGET_LANGS_FOR_ENGLISH,
)

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

APP_TITLE = "TranslateGemma Studio"
MODEL_ID = "mlx-community/translategemma-4b-it-8bit"
# Self-imposed prompt + output budget. The model card lists a 2K *input* context;
# the quant's max_position_embeddings (131072) allows far more.
CONTEXT_WINDOW = 2048
MAX_PROMPT_TOKENS = 1024  # prompt cap; leaves >=1024 tokens for the translation
MAX_INPUT_CHARS = 5000  # coarse backstop; the token budget is the real gate
# The page renders inside one centred column of this width (see "Page
# column" below), so the two panels are 592px each from ~1360 wide up and a
# line of 16px output holds ~75 characters, the top of the readable range.
# Measured: 1300 gives 80-81 (text area 91-97); 1100 gives 542px panels at
# 67 / 76. A narrower window clamps the column to the page.
PAGE_WIDTH = 1200
# One height for the text area and the output box, so their bottoms — and
# the two buttons directly below them — sit level. The offsets below are
# measured on 1.63.0 and independent of the height: the buttons' bottom is
# at 282 px + this, the over-budget badge ends 42 px lower, and a
# "Translation failed" st.error row 74 px lower still. 400 fits every state
# on a 1440x900 display (error row at 756) and shows ~14 lines of output; a
# 680 px laptop viewport needs <=320 for every state, <=350 for buttons and
# badge. Stay at or under 500, the st.container docstring's ceiling for
# scrolling containers. Re-measure before changing it.
PANEL_HEIGHT = 400


def build_prompt(
    text: str,
    src_lang: str,
    src_code: str,
    tgt_lang: str,
    tgt_code: str,
) -> str:
    instruction = (
        f"You are a professional {src_lang} ({src_code}) to {tgt_lang} "
        f"({tgt_code}) translator. Your goal is to accurately convey the meaning and "
        f"nuances of the original {src_lang} text while adhering to {tgt_lang} grammar, "
        f"vocabulary, and cultural sensitivities.\nProduce only the {tgt_lang} "
        f"translation, without any additional explanations or commentary. Please translate "
        f"the following {src_lang} text into {tgt_lang}:\n\n\n{text}"
    )
    return f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n"


@st.cache_resource
def load_model() -> tuple[Any, Any]:
    # mlx_lm.load() is annotated as a 2-tuple | 3-tuple union (return_config=True
    # yields the 3-tuple); the length check narrows it without a suppression.
    loaded = load(MODEL_ID)
    if len(loaded) != 2:
        raise RuntimeError(f"mlx_lm.load() returned {len(loaded)} values, expected 2")
    model, tokenizer = loaded
    tokenizer.add_eos_token("<end_of_turn>")
    return model, tokenizer


def count_prompt_tokens(prompt: str, tokenizer: Any) -> int:
    return len(tokenizer.encode(prompt))


def _prepare_generation(
    text: str,
    src_lang: str,
    src_code: str,
    tgt_lang: str,
    tgt_code: str,
) -> tuple[Any, Any, str, int]:
    # Build the prompt, load the model, enforce the token budget, and size
    # the output. Shared by translate() and translate_stream().
    prompt = build_prompt(text, src_lang, src_code, tgt_lang, tgt_code)
    model, tokenizer = load_model()
    prompt_tokens = count_prompt_tokens(prompt, tokenizer)
    if prompt_tokens > MAX_PROMPT_TOKENS:
        raise ValueError(
            f"Input is too long: {prompt_tokens} prompt tokens "
            f"(limit {MAX_PROMPT_TOKENS})."
        )
    # Hand the translation every token left in the context window.
    return model, tokenizer, prompt, CONTEXT_WINDOW - prompt_tokens


def _strip_eos_token(text: str) -> str:
    # Safety net: strip <end_of_turn> and any trailing content in case
    # the token leaks into the decoded output as literal text.
    return text.split("<end_of_turn>", 1)[0].strip()


def translate(
    text: str,
    src_lang: str,
    src_code: str,
    tgt_lang: str,
    tgt_code: str,
) -> str:
    model, tokenizer, prompt, max_tokens = _prepare_generation(
        text, src_lang, src_code, tgt_lang, tgt_code
    )
    result = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens)
    return _strip_eos_token(result)


def translate_stream(
    text: str,
    src_lang: str,
    src_code: str,
    tgt_lang: str,
    tgt_code: str,
) -> Iterator[str]:
    # Yield the translation segment-by-segment as the model generates it.
    # <end_of_turn> is a registered EOS token, so stream_generate stops
    # before emitting it; callers still strip it as a safety net.
    model, tokenizer, prompt, max_tokens = _prepare_generation(
        text, src_lang, src_code, tgt_lang, tgt_code
    )
    for response in stream_generate(
        model, tokenizer, prompt=prompt, max_tokens=max_tokens
    ):
        yield response.text


def _swap_languages() -> None:
    state = st.session_state
    if state["target_lang"] in FROM_ENGLISH_ONLY:
        return
    state["source_lang"], state["target_lang"] = (
        state["target_lang"],
        state["source_lang"],
    )
    if "translation_result" in state:
        state["source_text"] = state.pop("translation_result")


def _show_settled(box: DeltaGenerator, result: str) -> None:
    # The output box at rest: the last translation, or a muted placeholder.
    if result:
        box.text(result)
    else:
        box.caption("Translation")


st.set_page_config(
    page_title=APP_TITLE, page_icon=":material/translate:", layout="wide"
)

# --- Page column ---
# "wide" lifts the centred layout's 736px cap; the page renders inside one
# centred PAGE_WIDTH column instead, which keeps the readable-width cap the
# centred layout used to provide. Two containers because the cap and the
# centring are separate: a fixed-width child sits at the left edge of its
# parent unless the parent centres its elements. Everything below must stay
# inside this block — anything rendered outside it is full-bleed under
# "wide".
with st.container(horizontal_alignment="center"), st.container(width=PAGE_WIDTH):
    st.title(APP_TITLE)

    # --- Session state defaults ---
    st.session_state.setdefault("source_lang", "English")
    st.session_state.setdefault("target_lang", "Spanish")

    # --- Model loading ---
    try:
        with st.spinner("Loading model..."):
            _, tokenizer = load_model()
    except Exception as e:
        logger.exception("Failed to load model")
        st.error(f"Failed to load model: {e}", icon=":material/error:")
        st.stop()

    # --- Language selectors ---
    col1, col_swap, col2 = st.columns([10, 1, 10], vertical_alignment="center")
    source = col1.selectbox(
        "Source language",
        SOURCE_LANGS,
        key="source_lang",
        label_visibility="collapsed",
    )

    valid_targets = TARGET_LANGS_FOR_ENGLISH if source == "English" else ["English"]
    if st.session_state["target_lang"] not in valid_targets:
        st.session_state["target_lang"] = valid_targets[0]

    target = col2.selectbox(
        "Target language",
        valid_targets,
        key="target_lang",
        label_visibility="collapsed",
    )

    with col_swap:
        can_swap = st.session_state["target_lang"] not in FROM_ENGLISH_ONLY
        st.button(
            ":material/swap_horiz:",
            type="tertiary",
            width="stretch",
            on_click=_swap_languages,
            help="Swap languages",
            disabled=not can_swap,
        )

    # --- Text areas and buttons ---
    left_col, right_col = st.columns(2)

    with left_col:
        text = st.text_area(
            "Source text",
            height=PANEL_HEIGHT,
            max_chars=MAX_INPUT_CHARS,
            key="source_text",
            label_visibility="collapsed",
        )

        # Token usage against the prompt cap, computed here so the button can be
        # disabled on it. Surfaced only when over budget — the badge below the
        # button carries the count — so nothing sits between the text area and
        # Translate, and nothing renders under budget.
        prompt_tokens = 0
        if text.strip():
            # `tokenizer` is already bound from the module-level load above.
            preview = build_prompt(
                text,
                source,
                ALL_LANGUAGES[source],
                target,
                ALL_LANGUAGES[target],
            )
            prompt_tokens = count_prompt_tokens(preview, tokenizer)
        over_budget = prompt_tokens > MAX_PROMPT_TOKENS

        # Translate directly follows the PANEL_HEIGHT text area in every state,
        # so it stays level with Download, which directly follows the
        # PANEL_HEIGHT output box.
        translate_clicked = st.button(
            "Translate",
            type="primary",
            key="translate_text",
            width="stretch",
            disabled=over_budget,
        )

        if over_budget:
            st.badge(
                f"Too long: {prompt_tokens} / {MAX_PROMPT_TOKENS} tokens",
                icon=":material/error:",
                color="red",
            )

    prev_response = st.session_state.get("translation_result", "")

    with right_col:
        # One bordered, fixed-height box holds the translation in every state.
        # The settled result is st.text — the same element streaming writes
        # through — rather than a disabled text area, which Streamlit paints at
        # 40% alpha.
        with st.container(height=PANEL_HEIGHT):
            output_box = st.empty()
            _show_settled(output_box, prev_response)

        # Nothing between the box and Download — see the left column.
        st.download_button(
            label="Download",
            type="secondary",
            data=prev_response if prev_response else "",
            file_name="translation.txt",
            mime="text/plain",
            key="download_text",
            disabled=not prev_response,
            width="stretch",
        )

    if translate_clicked:
        if not text.strip():
            st.warning("Please enter text to translate.", icon=":material/warning:")
        else:
            # Stream into the output box as the model generates; the rerun then
            # re-renders it settled and enables Download.
            chunks: list[str] = []
            try:
                output_box.caption("Translating…")  # covers the prefill wait
                for chunk in translate_stream(
                    text,
                    source,
                    ALL_LANGUAGES[source],
                    target,
                    ALL_LANGUAGES[target],
                ):
                    chunks.append(chunk)
                    output_box.text("".join(chunks))
                st.session_state["translation_result"] = _strip_eos_token(
                    "".join(chunks)
                )
                st.rerun()
            except Exception as e:
                logger.exception("Translation failed")
                st.error(f"Translation failed: {e}", icon=":material/error:")
                if not chunks:
                    # Nothing streamed: put the box back to match what Download
                    # still offers. A partial stream is left in place.
                    _show_settled(output_box, prev_response)
