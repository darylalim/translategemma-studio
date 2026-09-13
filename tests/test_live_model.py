"""Live end-to-end check against the real model.

Deselected by default (`-m "not live"` in `addopts`) because it loads the full
3.9 GB quant and runs real inference. Run it with `uv run pytest -m live`.

No other test loads the real `mlx_lm` (the ones that touch it swap in a
`MagicMock`), so the suite cannot see runtime breakage in the MLX stack: an
`mlx-lm`/`mlx` pairing that returns an empty translation still passes every
other test at 100% coverage. This is the only test that would notice, which
makes it worth running by hand after any `mlx` or `mlx-lm` bump. See Known
Issues in CLAUDE.md for the failure that prompted it.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

_APP_PATH = str(Path(__file__).parent.parent / "streamlit_app.py")

# Generous enough to cover a cold model load; generation of one short
# sentence is well under a second once the weights are resident.
_LIVE_TIMEOUT = 600

_SOURCE_TEXT = "The quick brown fox jumps over the lazy dog."


@pytest.mark.live
def test_real_translation_round_trip() -> None:
    """Translate English -> Spanish through the real model and UI path."""
    at = AppTest.from_file(_APP_PATH, default_timeout=_LIVE_TIMEOUT)
    at.run()
    assert not at.exception, f"app failed to start: {at.exception}"
    # A load failure is caught and rendered as st.error before st.stop(), so
    # without this the text_area lookup below fails with KeyError instead.
    assert not at.error, f"app reported: {[e.value for e in at.error]}"

    at.text_area(key="source_text").set_value(_SOURCE_TEXT).run()
    at.button(key="translate_text").click().run()
    assert not at.exception, f"translation raised: {at.exception}"
    # translate_stream() failures are caught and rendered as st.error, so the
    # unhandled-exception check above sails past them -- check the callout too,
    # otherwise the real cause is misreported by the empty-translation assert
    # below as the wired_limit() failure.
    assert not at.error, f"app reported: {[e.value for e in at.error]}"

    result = at.session_state["translation_result"]
    # An empty result is the signature of the wired_limit() thread failure:
    # the app loads and the click succeeds, but nothing is generated.
    assert result.strip(), "model returned an empty translation"
    assert result != _SOURCE_TEXT, "output is the untranslated source"
    assert "<end_of_turn>" not in result, "EOS token leaked into the output"

    # The settled output box should mirror what landed in session state.
    assert [t.value for t in at.text] == [result]
