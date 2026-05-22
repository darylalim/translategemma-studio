import importlib
import sys
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(scope="session")
def app_module():
    """Import streamlit_app with all heavy dependencies mocked."""
    mock_st = MagicMock()
    mock_st.cache_resource = lambda f: f
    mock_st.session_state = {}

    col1, col_swap, col2 = MagicMock(), MagicMock(), MagicMock()
    col1.selectbox.return_value = "English"
    col2.selectbox.return_value = "Spanish"

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
    module_tokenizer.encode.return_value = list(range(50))
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
    module = importlib.import_module("streamlit_app")

    for mod_name, orig in originals.items():
        if orig is None:
            sys.modules.pop(mod_name, None)
        else:
            sys.modules[mod_name] = orig

    return module


@pytest.fixture()
def mock_tokenizer():
    """A mock tokenizer whose encode() returns a short, countable token list."""
    tokenizer = MagicMock()
    tokenizer.encode.return_value = list(range(50))
    return tokenizer


@pytest.fixture()
def patched_translate(app_module, mock_tokenizer):
    """Patch load_model and generate for translation tests."""
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
        ),
    ):
        yield {
            "translate": app_module.translate,
            "model": mock_model,
            "tokenizer": mock_tokenizer,
        }
