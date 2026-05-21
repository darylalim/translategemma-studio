import logging
from typing import Any

import streamlit as st
from mlx_lm import generate, load

from languages import (
    ALL_LANGUAGES,
    FROM_ENGLISH_ONLY,
    SOURCE_LANGS,
    TARGET_LANGS_FOR_ENGLISH,
)

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

MODEL_ID = "mlx-community/translategemma-4b-it-8bit"
MAX_NEW_TOKENS = 512


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
    model, tokenizer = load(MODEL_ID)
    tokenizer.add_eos_token("<end_of_turn>")
    return model, tokenizer


def translate(
    text: str,
    src_lang: str,
    src_code: str,
    tgt_lang: str,
    tgt_code: str,
) -> str:
    prompt = build_prompt(text, src_lang, src_code, tgt_lang, tgt_code)
    model, tokenizer = load_model()
    result = generate(model, tokenizer, prompt=prompt, max_tokens=MAX_NEW_TOKENS)
    # Safety net: strip <end_of_turn> and any trailing content in case
    # the token leaks into the decoded output string.
    return result.split("<end_of_turn>", 1)[0].strip()


st.set_page_config(page_title="TranslateGemma Translate", page_icon="\U0001f310")
st.title("TranslateGemma Translate")

# --- Session state defaults ---
st.session_state.setdefault("source_lang", "English")
st.session_state.setdefault("target_lang", "Spanish")

# --- Model loading ---
try:
    with st.spinner("Loading model..."):
        model, tokenizer = load_model()
except Exception as e:
    logger.exception("Failed to load model")
    st.error(f"Failed to load model: {e}")
    st.stop()


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


# --- Language selectors ---
col1, col_swap, col2 = st.columns([10, 1, 10])
source = col1.selectbox(
    "Source language",
    SOURCE_LANGS,
    key="source_lang",
    label_visibility="collapsed",
)

if source == "English":
    valid_targets = TARGET_LANGS_FOR_ENGLISH
else:
    valid_targets = ["English"]
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
        use_container_width=True,
        on_click=_swap_languages,
        help="Swap languages",
        disabled=not can_swap,
    )

# --- Text areas and buttons ---
left_col, right_col = st.columns(2)

with left_col:
    text = st.text_area(
        "Source text",
        height=300,
        max_chars=5000,
        key="source_text",
        label_visibility="collapsed",
    )

    translate_clicked = st.button(
        "Translate",
        type="primary",
        key="translate_text",
        use_container_width=True,
    )

prev_response = st.session_state.get("translation_result", "")

with right_col:
    st.session_state["text_output"] = prev_response
    st.text_area(
        "Translation output",
        placeholder="Translation",
        disabled=True,
        height=300,
        label_visibility="collapsed",
        key="text_output",
    )

    st.download_button(
        label="Download",
        type="secondary",
        data=prev_response if prev_response else "",
        file_name="translation.txt",
        mime="text/plain",
        key="download_text",
        disabled=not prev_response,
        use_container_width=True,
    )

if translate_clicked:
    if not text.strip():
        st.warning("Please enter text to translate.")
    else:
        try:
            result = translate(
                text,
                source,
                ALL_LANGUAGES[source],
                target,
                ALL_LANGUAGES[target],
            )
            st.session_state["translation_result"] = result
            st.rerun()
        except Exception as e:
            logger.exception("Translation failed")
            st.error(f"Translation failed: {e}")
