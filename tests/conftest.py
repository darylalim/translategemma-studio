import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Token count that mock tokenizers report from encode(); well under the
# app's MAX_PROMPT_TOKENS budget so translate() accepts the prompt.
_MOCK_PROMPT_TOKENS = 50

# Absolute path to streamlit_app.py for AppTest.from_file().
_APP_PATH = str(Path(__file__).parent.parent / "streamlit_app.py")


def _import_app(source_lang: str = "English"):
    """Import streamlit_app against a MagicMock streamlit and mlx_lm.

    `source_lang` is what the source selectbox returns. The target selectbox
    returns whatever session_state["target_lang"] holds when it is created and
    records that value on `st.target_lang_at_target_selectbox`, so a test can
    check the runtime target filter ran first — a real selectbox silently
    resets a stored value outside its options, so AppTest cannot see this.
    """
    mock_st = MagicMock()

    cache_resource_kwargs: list[dict] = []

    def _identity_cache(func=None, **kwargs):
        # A bare @st.cache_resource passes the function; the keyword form
        # @st.cache_resource(show_spinner=...) is called first and decorates
        # with what it returns. Both must leave the function untouched; the
        # kwargs are kept so TestLoadModel can pin the spinner message.
        cache_resource_kwargs.append(kwargs)
        return func if func is not None else (lambda f: f)

    mock_st.cache_resource = _identity_cache
    mock_st.cache_resource_kwargs = cache_resource_kwargs
    mock_st.session_state = {}

    col1, col_swap, col2 = MagicMock(), MagicMock(), MagicMock()
    col1.selectbox.return_value = source_lang

    def _target_selectbox(*args, **kwargs):
        mock_st.target_lang_at_target_selectbox = mock_st.session_state["target_lang"]
        return mock_st.target_lang_at_target_selectbox

    col2.selectbox.side_effect = _target_selectbox

    # Content columns
    left_col, right_col = MagicMock(), MagicMock()

    # Column calls are position-dependent — update this list if st.columns
    # calls are added, removed, or reordered in streamlit_app.py:
    # 1. Language selectors [10, 1, 10]
    # 2. Content columns [2]
    _columns_calls = iter(
        [
            (col1, col_swap, col2),
            (left_col, right_col),
        ]
    )

    def _mock_columns(*args, **kwargs):
        try:
            return next(_columns_calls)
        except StopIteration:
            n = args[0] if args else 2
            if isinstance(n, list):
                n = len(n)
            return tuple(MagicMock() for _ in range(n))

    mock_st.columns = MagicMock(side_effect=_mock_columns)
    mock_st.button.return_value = False

    mock_mlx_lm = MagicMock()
    # load() returns a (model, tokenizer) pair; the tokenizer's encode()
    # must yield a real sequence so count_prompt_tokens() can len() it.
    module_tokenizer = MagicMock()
    module_tokenizer.encode.return_value = list(range(_MOCK_PROMPT_TOKENS))
    mock_mlx_lm.load.return_value = (MagicMock(), module_tokenizer)

    patches = {
        "streamlit": mock_st,
        "mlx_lm": mock_mlx_lm,
    }

    originals = {}
    for mod_name, mock_obj in patches.items():
        originals[mod_name] = sys.modules.get(mod_name)
        sys.modules[mod_name] = mock_obj

    if "streamlit_app" in sys.modules:
        del sys.modules["streamlit_app"]
    try:
        module = importlib.import_module("streamlit_app")
    finally:
        # Always put the real modules back: if the import raises, a leaked
        # MagicMock streamlit would make every AppTest and theme test error
        # in _clear_streamlit_caches, burying the one failure that matters.
        for mod_name, orig in originals.items():
            if orig is None:
                sys.modules.pop(mod_name, None)
            else:
                sys.modules[mod_name] = orig

    return module


@pytest.fixture(scope="session")
def app_module():
    """Import streamlit_app with all heavy dependencies mocked."""
    return _import_app()


@pytest.fixture(scope="session")
def app_module_non_english_source():
    """The same mocked import with the source selectbox returning French.

    Only English is a valid target then, so the runtime filter has to fire;
    the default import never exercises it.
    """
    return _import_app(source_lang="French")


@pytest.fixture(autouse=True)
def _clear_streamlit_caches():
    """Reset st.cache_resource between tests.

    Streamlit's resource cache is process-global, so without this every test
    after the first would see the prior test's cached load_model() result.
    """
    import streamlit as st

    st.cache_resource.clear()
    yield


@pytest.fixture()
def mock_tokenizer():
    """A mock tokenizer whose encode() returns a short, countable token list."""
    tokenizer = MagicMock()
    tokenizer.encode.return_value = list(range(_MOCK_PROMPT_TOKENS))
    return tokenizer


@pytest.fixture()
def patched_translate(app_module, mock_tokenizer):
    """Patch load_model + generate + stream_generate for translation tests.

    Returns a dict with the bound translate/translate_stream callables plus
    the underlying mock objects so individual tests can override return
    values or set side effects per case.
    """
    mock_model = MagicMock()
    with (
        patch.object(
            app_module,
            "load_model",
            return_value=(mock_model, mock_tokenizer),
        ),
        patch.object(
            app_module,
            "generate",
            return_value="translated text",
        ) as mock_generate,
        patch.object(app_module, "stream_generate") as mock_stream_generate,
    ):
        yield {
            "translate": app_module.translate,
            "translate_stream": app_module.translate_stream,
            "model": mock_model,
            "tokenizer": mock_tokenizer,
            "generate": mock_generate,
            "stream_generate": mock_stream_generate,
        }


@pytest.fixture()
def fake_mlx_lm(mock_tokenizer):
    """A mlx_lm module mock with load() wired up; per-test customizable."""
    fake = MagicMock()
    fake.load.return_value = (MagicMock(), mock_tokenizer)
    return fake


def _build_app_test(fake_mlx_lm):
    """Common setup for app_test variants: patch sys.modules and build AppTest."""
    from streamlit.testing.v1 import AppTest

    saved_mlx = sys.modules.get("mlx_lm")
    sys.modules["mlx_lm"] = fake_mlx_lm
    # Evict any cached streamlit_app so it re-imports against the mock.
    saved_app = sys.modules.pop("streamlit_app", None)
    at = AppTest.from_file(_APP_PATH, default_timeout=10)
    return at, saved_mlx, saved_app


def _restore_modules(saved_mlx, saved_app):
    if saved_mlx is None:
        sys.modules.pop("mlx_lm", None)
    else:
        sys.modules["mlx_lm"] = saved_mlx
    if saved_app is None:
        sys.modules.pop("streamlit_app", None)
    else:
        sys.modules["streamlit_app"] = saved_app


@pytest.fixture()
def app_test(fake_mlx_lm):
    """AppTest pre-run to its settled initial state.

    Use this for tests that interact with the UI after a normal cold start
    (text entry, button clicks, etc.). For tests that need to control what
    happens during the first run (e.g. simulating a load_model() failure),
    use app_test_unrun.
    """
    at, saved_mlx, saved_app = _build_app_test(fake_mlx_lm)
    try:
        at.run()
        assert not at.exception, [e.value for e in at.exception]
        yield at
    finally:
        _restore_modules(saved_mlx, saved_app)


@pytest.fixture()
def app_test_unrun(fake_mlx_lm):
    """AppTest that has NOT yet been run; caller controls the first .run()."""
    at, saved_mlx, saved_app = _build_app_test(fake_mlx_lm)
    try:
        yield at
    finally:
        _restore_modules(saved_mlx, saved_app)
