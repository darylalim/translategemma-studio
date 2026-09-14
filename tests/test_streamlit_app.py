import re
import tomllib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest


def _caption_texts(app_module):
    return [c.args[0] for c in app_module.st.caption.call_args_list if c.args]


def _top_level_calls(app_module):
    # Direct st.<name>(...) calls, in order. Calls on returned mocks
    # (text_area().strip(), container().__enter__()) carry a dot in their
    # name and are dropped, so adjacency here means "no direct st.* call
    # between"; writes through a column handle are not recorded here, and the
    # AppTest walk covers those.
    return [c for c in app_module.st.mock_calls if "." not in c[0]]


def _fake_stream(*segments):
    # Stand in for mlx-lm's stream of GenerationResponse objects; only the
    # .text attribute is read by translate_stream().
    return [SimpleNamespace(text=s) for s in segments]


def _is_output_box(child):
    # The output box is the only element inside the columns with a pixel
    # height; the exact number is pinned in TestOutputBox, not here, so the
    # AppTest helpers need no copy of PANEL_HEIGHT (importing streamlit_app
    # here would run the script against the real model).
    return child.type == "flex_container" and child.proto.height_config.pixel_height > 0


def _output_box(app_test):
    # The one fixed-height container on the page, found structurally so this
    # does not repeat conftest's positional column-order knowledge.
    boxes = [
        child
        for column in app_test.columns
        for child in column.children.values()
        if _is_output_box(child)
    ]
    assert len(boxes) == 1, (
        f"expected one fixed-height output container, found {len(boxes)}"
    )
    return boxes[0]


def _box_contents(box):
    return [(child.type, child.value) for child in box.children.values()]


def _content_columns(app_test):
    # The two content columns, identified by what they hold — the source
    # text_area and the fixed-height output box — rather than by position.
    def holding(predicate, what):
        columns = [
            column
            for column in app_test.columns
            if any(predicate(child) for child in column.children.values())
        ]
        assert len(columns) == 1, (
            f"expected one column holding {what}, found {len(columns)}"
        )
        return columns[0]

    left = holding(lambda child: child.type == "text_area", "the source text_area")
    right = holding(_is_output_box, "the fixed-height output box")
    return left, right


def _column_shapes(app_test):
    left, right = _content_columns(app_test)
    return (
        [child.type for child in left.children.values()],
        [child.type for child in right.children.values()],
    )


class TestConstants:
    def test_model_id(self, app_module):
        assert app_module.MODEL_ID == "mlx-community/translategemma-4b-it-8bit"

    def test_context_window(self, app_module):
        assert app_module.CONTEXT_WINDOW == 2048

    def test_max_prompt_tokens(self, app_module):
        assert app_module.MAX_PROMPT_TOKENS == 1024

    def test_text_area_has_no_character_cap(self, app_module):
        # The token budget is the one limit. A max_chars cap froze the text
        # area whenever a swapped translation exceeded it (the frontend drops
        # any edit whose result is still over the cap, deletions included).
        # max_chars is also st.text_area's fourth positional parameter.
        assert app_module.st.text_area.call_args.args == ("Source text",)
        assert "max_chars" not in app_module.st.text_area.call_args.kwargs

    def test_panel_height_under_the_scrolling_container_ceiling(self, app_module):
        # st.container's docstring: avoid scrolling heights over 500 pixels.
        assert app_module.PANEL_HEIGHT <= 500

    def test_page_width(self, app_module):
        assert app_module.PAGE_WIDTH == 1200

    def test_prompt_budget_leaves_room_for_output(self, app_module):
        # The prompt cap must leave room within the context window
        # for the model to generate a translation.
        assert app_module.MAX_PROMPT_TOKENS < app_module.CONTEXT_WINDOW


class TestLanguageImports:
    def test_all_languages_available(self, app_module):
        assert len(app_module.ALL_LANGUAGES) == 295

    def test_source_langs_available(self, app_module):
        assert len(app_module.SOURCE_LANGS) == 225

    def test_target_langs_for_english_available(self, app_module):
        assert len(app_module.TARGET_LANGS_FOR_ENGLISH) == 294

    def test_from_english_only_available(self, app_module):
        assert len(app_module.FROM_ENGLISH_ONLY) == 70


class TestBuildPrompt:
    def test_contains_language_names(self, app_module):
        prompt = app_module.build_prompt("Hello", "English", "en", "Spanish", "es")
        assert "English" in prompt
        assert "Spanish" in prompt

    def test_contains_language_codes(self, app_module):
        prompt = app_module.build_prompt("Hello", "English", "en", "Spanish", "es")
        assert "(en)" in prompt
        assert "(es)" in prompt

    def test_contains_source_text(self, app_module):
        prompt = app_module.build_prompt(
            "Translate me", "English", "en", "French", "fr"
        )
        assert "Translate me" in prompt

    def test_uses_gemma_chat_format(self, app_module):
        prompt = app_module.build_prompt("Hello", "English", "en", "Spanish", "es")
        assert prompt.startswith("<start_of_turn>user\n")
        assert "<end_of_turn>\n<start_of_turn>model\n" in prompt

    def test_returns_string(self, app_module):
        prompt = app_module.build_prompt("Hello", "English", "en", "Spanish", "es")
        assert isinstance(prompt, str)

    def test_newline_before_produce(self, app_module):
        # The trained chat template puts a newline (not a space) after
        # "cultural sensitivities." — keep build_prompt() aligned with it.
        prompt = app_module.build_prompt("Hello", "English", "en", "Spanish", "es")
        assert "cultural sensitivities.\nProduce only the" in prompt
        assert "cultural sensitivities. Produce" not in prompt


class TestSwapLanguages:
    def test_swaps_source_and_target(self, app_module):
        mock_state = {"source_lang": "English", "target_lang": "Spanish"}
        with patch.object(app_module.st, "session_state", mock_state):
            app_module._swap_languages()
        assert mock_state == {"source_lang": "Spanish", "target_lang": "English"}

    def test_swap_is_reversible(self, app_module):
        mock_state = {"source_lang": "English", "target_lang": "French"}
        with patch.object(app_module.st, "session_state", mock_state):
            app_module._swap_languages()
            app_module._swap_languages()
        assert mock_state == {"source_lang": "English", "target_lang": "French"}

    def test_swap_copies_translation_to_source_text(self, app_module):
        mock_state = {
            "source_lang": "English",
            "target_lang": "Spanish",
            "translation_result": "hola",
        }
        with patch.object(app_module.st, "session_state", mock_state):
            app_module._swap_languages()
        assert mock_state == {
            "source_lang": "Spanish",
            "target_lang": "English",
            "source_text": "hola",
            "translation_result": "",
        }

    def test_double_swap_with_translation_is_not_reversible(self, app_module):
        mock_state = {
            "source_lang": "English",
            "target_lang": "Spanish",
            "translation_result": "hola",
        }
        with patch.object(app_module.st, "session_state", mock_state):
            app_module._swap_languages()
            app_module._swap_languages()
        assert mock_state == {
            "source_lang": "English",
            "target_lang": "Spanish",
            "source_text": "hola",
            "translation_result": "",
        }

    def test_swap_without_translation_does_not_set_source_text(self, app_module):
        mock_state = {"source_lang": "English", "target_lang": "Spanish"}
        with patch.object(app_module.st, "session_state", mock_state):
            app_module._swap_languages()
        assert mock_state == {"source_lang": "Spanish", "target_lang": "English"}

    def test_swap_with_empty_translation_keeps_source_text(self, app_module):
        # The app seeds translation_result to "" with the other defaults, so
        # the key is always present; only a non-empty result moves across.
        mock_state = {
            "source_lang": "English",
            "target_lang": "Spanish",
            "translation_result": "",
            "source_text": "hello",
        }
        with patch.object(app_module.st, "session_state", mock_state):
            app_module._swap_languages()
        assert mock_state == {
            "source_lang": "Spanish",
            "target_lang": "English",
            "translation_result": "",
            "source_text": "hello",
        }


class TestSwapDisabled:
    def test_swap_enabled_for_bidirectional_target(self, app_module):
        mock_state = {"target_lang": "Spanish"}
        can_swap = mock_state["target_lang"] not in app_module.FROM_ENGLISH_ONLY
        assert can_swap is True

    def test_swap_disabled_for_from_english_only_target(self, app_module):
        mock_state = {"target_lang": "Albanian"}
        can_swap = mock_state["target_lang"] not in app_module.FROM_ENGLISH_ONLY
        assert can_swap is False

    def test_swap_enabled_for_english_target(self, app_module):
        mock_state = {"target_lang": "English"}
        can_swap = mock_state["target_lang"] not in app_module.FROM_ENGLISH_ONLY
        assert can_swap is True

    def test_swap_guard_blocks_from_english_only(self, app_module):
        mock_state = {
            "source_lang": "English",
            "target_lang": "Albanian",
        }
        with patch.object(app_module.st, "session_state", mock_state):
            app_module._swap_languages()
        assert mock_state == {"source_lang": "English", "target_lang": "Albanian"}

    def test_swap_guard_allows_bidirectional(self, app_module):
        mock_state = {
            "source_lang": "English",
            "target_lang": "French",
        }
        with patch.object(app_module.st, "session_state", mock_state):
            app_module._swap_languages()
        assert mock_state == {"source_lang": "French", "target_lang": "English"}


class TestTargetFilter:
    def test_target_filter_runs_before_the_target_selectbox(
        self, app_module_non_english_source
    ):
        # With a non-English source the only valid target is English, so the
        # runtime filter must have rewritten session_state["target_lang"]
        # before the target selectbox was created. Below the widget it is
        # dead code, not a crash: a real selectbox silently resets a stored
        # value outside its options and writes it back, so the filter's
        # condition is never true and every AppTest stays green (see
        # CLAUDE.md → Do not touch). The mock selectbox does not repair,
        # which is what lets this layer see the order.
        st = app_module_non_english_source.st
        assert st.target_lang_at_target_selectbox == "English"


class TestCountPromptTokens:
    def test_counts_encoded_tokens(self, app_module, mock_tokenizer):
        count = app_module.count_prompt_tokens("a prompt", mock_tokenizer)
        assert count == len(mock_tokenizer.encode.return_value)

    def test_encodes_the_given_prompt(self, app_module, mock_tokenizer):
        app_module.count_prompt_tokens("a prompt", mock_tokenizer)
        mock_tokenizer.encode.assert_called_once_with("a prompt")


class TestTranslate:
    def test_returns_string(self, patched_translate):
        result = patched_translate["translate"](
            "Hello", "English", "en", "Spanish", "es"
        )
        assert isinstance(result, str)

    def test_returns_generated_text(self, patched_translate):
        result = patched_translate["translate"](
            "Hello", "English", "en", "Spanish", "es"
        )
        assert result == "translated text"

    def test_generate_called_with_correct_args(self, app_module, patched_translate):
        patched_translate["translate"]("Hello", "English", "en", "Spanish", "es")
        expected_prompt = app_module.build_prompt(
            "Hello", "English", "en", "Spanish", "es"
        )
        prompt_tokens = len(patched_translate["tokenizer"].encode.return_value)
        patched_translate["generate"].assert_called_once_with(
            patched_translate["model"],
            patched_translate["tokenizer"],
            prompt=expected_prompt,
            max_tokens=app_module.CONTEXT_WINDOW - prompt_tokens,
        )

    def test_generate_called_exactly_once(self, patched_translate):
        patched_translate["translate"]("Hello", "English", "en", "Spanish", "es")
        assert patched_translate["generate"].call_count == 1

    @pytest.mark.parametrize(
        "generated,expected",
        [
            pytest.param("hola mundo<end_of_turn>", "hola mundo", id="single_eos"),
            pytest.param(
                "hola mundo<end_of_turn><end_of_turn><end_of_turn>",
                "hola mundo",
                id="repeated_eos",
            ),
            pytest.param("hola mundo", "hola mundo", id="clean_output"),
            pytest.param("  hola mundo  <end_of_turn>", "hola mundo", id="whitespace"),
            pytest.param(
                "hola mundo<end_of_turn>extra garbage",
                "hola mundo",
                id="garbage_after_eos",
            ),
        ],
    )
    def test_strips_eos_from_generated_output(
        self, patched_translate, generated, expected
    ):
        patched_translate["generate"].return_value = generated
        result = patched_translate["translate"](
            "hello world", "English", "en", "Spanish", "es"
        )
        assert result == expected

    def test_raises_when_prompt_exceeds_budget(
        self, app_module, patched_translate, mock_tokenizer
    ):
        mock_tokenizer.encode.return_value = list(
            range(app_module.MAX_PROMPT_TOKENS + 1)
        )
        with pytest.raises(ValueError, match="too long"):
            patched_translate["translate"]("text", "English", "en", "Spanish", "es")
        patched_translate["generate"].assert_not_called()

    def test_allows_prompt_at_budget_limit(
        self, app_module, patched_translate, mock_tokenizer
    ):
        mock_tokenizer.encode.return_value = list(range(app_module.MAX_PROMPT_TOKENS))
        patched_translate["generate"].return_value = "ok"
        result = patched_translate["translate"](
            "text", "English", "en", "Spanish", "es"
        )
        assert result == "ok"


class TestTranslateStream:
    def test_yields_text_segments(self, patched_translate):
        patched_translate["stream_generate"].return_value = _fake_stream(
            "hola", " ", "mundo"
        )
        segments = list(
            patched_translate["translate_stream"](
                "hello world", "English", "en", "Spanish", "es"
            )
        )
        assert segments == ["hola", " ", "mundo"]

    def test_segments_join_to_full_translation(self, patched_translate):
        patched_translate["stream_generate"].return_value = _fake_stream(
            "hola", " mundo"
        )
        full = "".join(
            patched_translate["translate_stream"](
                "hello world", "English", "en", "Spanish", "es"
            )
        )
        assert full == "hola mundo"

    def test_stream_generate_called_with_correct_args(
        self, app_module, patched_translate
    ):
        patched_translate["stream_generate"].return_value = _fake_stream("hola")
        list(
            patched_translate["translate_stream"](
                "Hello", "English", "en", "Spanish", "es"
            )
        )
        expected_prompt = app_module.build_prompt(
            "Hello", "English", "en", "Spanish", "es"
        )
        prompt_tokens = len(patched_translate["tokenizer"].encode.return_value)
        patched_translate["stream_generate"].assert_called_once_with(
            patched_translate["model"],
            patched_translate["tokenizer"],
            prompt=expected_prompt,
            max_tokens=app_module.CONTEXT_WINDOW - prompt_tokens,
        )

    def test_is_a_lazy_generator(self, app_module):
        # The generator body must not run until iteration begins.
        with patch.object(app_module, "load_model") as mock_load:
            gen = app_module.translate_stream("Hello", "English", "en", "Spanish", "es")
        mock_load.assert_not_called()
        assert iter(gen) is gen

    def test_raises_when_prompt_exceeds_budget(
        self, app_module, patched_translate, mock_tokenizer
    ):
        mock_tokenizer.encode.return_value = list(
            range(app_module.MAX_PROMPT_TOKENS + 1)
        )
        with pytest.raises(ValueError, match="too long"):
            list(
                patched_translate["translate_stream"](
                    "text", "English", "en", "Spanish", "es"
                )
            )
        patched_translate["stream_generate"].assert_not_called()

    def test_allows_prompt_at_budget_limit(
        self, app_module, patched_translate, mock_tokenizer
    ):
        mock_tokenizer.encode.return_value = list(range(app_module.MAX_PROMPT_TOKENS))
        patched_translate["stream_generate"].return_value = _fake_stream("ok")
        segments = list(
            patched_translate["translate_stream"](
                "text", "English", "en", "Spanish", "es"
            )
        )
        assert segments == ["ok"]


class TestHeader:
    def test_page_title(self, app_module):
        app_module.st.set_page_config.assert_called_once()
        kwargs = app_module.st.set_page_config.call_args.kwargs
        assert kwargs["page_title"] == "TranslateGemma Studio"

    def test_page_icon(self, app_module):
        kwargs = app_module.st.set_page_config.call_args.kwargs
        assert kwargs["page_icon"] == ":material/translate:"

    def test_page_layout_is_wide(self, app_module):
        # Layout must stay wide: the readable-width cap moved from the
        # centered layout to the PAGE_WIDTH page column, and dropping this
        # kwarg would silently restore the 736px page.
        kwargs = app_module.st.set_page_config.call_args.kwargs
        assert kwargs.get("layout") == "wide"

    def test_page_column_wraps_the_ui(self, app_module):
        # Both page-column containers are opened before the first element,
        # so everything renders inside the centred column; anything outside
        # it is full-bleed under wide.
        calls = app_module.st.mock_calls
        outer = calls.index(call.container(horizontal_alignment="center"))
        inner = calls.index(call.container(width=app_module.PAGE_WIDTH))
        title = calls.index(call.title(app_module.APP_TITLE))
        assert outer < inner < title

    def test_title(self, app_module):
        app_module.st.title.assert_called_once_with("TranslateGemma Studio")


class TestButtonLayout:
    def test_columns_called_twice(self, app_module):
        calls = app_module.st.columns.call_args_list
        assert len(calls) == 2

    def test_language_selector_columns(self, app_module):
        calls = app_module.st.columns.call_args_list
        assert calls[0] == call([10, 1, 10], vertical_alignment="center")

    def test_content_columns(self, app_module):
        calls = app_module.st.columns.call_args_list
        assert calls[1] == call(2)

    # Each button is the element right after its PANEL_HEIGHT panel. Anything
    # conditional between a panel and its button moves the button — and,
    # unless mirrored in the other column, misaligns the pair. A live token
    # counter used to sit there; only the over-budget badge renders now,
    # below Translate.

    def test_translate_directly_follows_the_text_area(self, app_module):
        calls = _top_level_calls(app_module)
        text_area = [c[0] for c in calls].index("text_area")
        name, args, _ = calls[text_area + 1]
        assert (name, args) == ("button", ("Translate",))

    def test_download_directly_follows_the_output_box(self, app_module):
        # The box holds exactly one st.empty(), so the top-level call after
        # it is the first element outside the box.
        calls = _top_level_calls(app_module)
        empty = [c[0] for c in calls].index("empty")
        assert calls[empty + 1][0] == "download_button"

    def test_panels_share_one_height(self, app_module):
        # Hard rule: the text area and the output box read the same height,
        # so their bottoms — and the two buttons below them — sit level.
        text_area_height = app_module.st.text_area.call_args.kwargs["height"]
        # Select the box's call by its kwarg rather than by position, so an
        # st.container added after the box fails this assertion, not a
        # KeyError.
        [box_call] = [
            c for c in app_module.st.container.call_args_list if "height" in c.kwargs
        ]
        assert text_area_height == box_call.kwargs["height"] == app_module.PANEL_HEIGHT

    def test_no_spacer_caption(self, app_module):
        # The right column once mirrored the counter with an invisible
        # "&nbsp;" caption to keep the buttons level; with nothing between a
        # panel and its button on either side there is nothing to mirror.
        assert "&nbsp;" not in _caption_texts(app_module)


class TestTokenBudget:
    def test_nothing_rendered_under_budget(self, app_module):
        # The import-time tokenizer reports 50 tokens, well under the cap.
        # The budget surfaces only when exceeded: no live counter caption
        # ("93 / 1024 tokens" is jargon for a one-sentence input) and no
        # badge. The over-budget badge is covered in TestStreamingClickPath.
        # Pin the precondition first, so the negatives cannot pass vacuously:
        # the budget branch ran and came in under the cap.
        assert 0 < app_module.prompt_tokens <= app_module.MAX_PROMPT_TOKENS
        assert app_module.over_budget is False
        assert not any("tokens" in text for text in _caption_texts(app_module))
        app_module.st.badge.assert_not_called()


class TestOutputBox:
    def test_output_box_is_the_only_fixed_height_container(self, app_module):
        # st.container is called three times: the two page-column wrappers,
        # then the one bordered, fixed-height output box (PANEL_HEIGHT, the
        # same as the source text_area).
        assert app_module.st.container.call_args_list == [
            call(horizontal_alignment="center"),
            call(width=app_module.PAGE_WIDTH),
            call(height=app_module.PANEL_HEIGHT),
        ]

    def test_st_empty_is_created_inside_the_container(self, app_module):
        # The single st.empty() must be opened inside the output box's
        # `with` block — st.empty() called before __enter__ or after __exit__
        # would render the translation outside the bordered box. Index from
        # the box's own call: the page column's __enter__ comes first.
        calls = app_module.st.mock_calls
        box = calls.index(call.container(height=app_module.PANEL_HEIGHT))
        assert calls[box + 1] == call.container().__enter__()
        empty = calls.index(call.empty())
        leave = calls.index(call.container().__exit__(None, None, None), box)
        assert box < empty < leave
        app_module.st.empty.assert_called_once()

    def test_empty_state_shows_placeholder_caption_not_text(self, app_module):
        # Nothing has been translated at import time, so the box shows a
        # muted placeholder and no st.text — and never a disabled text_area.
        box = app_module.st.empty.return_value
        box.caption.assert_called_once_with("Translation")
        box.text.assert_not_called()
        assert app_module.st.text_area.call_count == 1  # the source only


class TestLoadModel:
    def test_returns_model_and_tokenizer_from_load(self, app_module):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        with patch.object(
            app_module, "load", return_value=(mock_model, mock_tokenizer)
        ):
            model, tokenizer = app_module.load_model()
        assert model is mock_model
        assert tokenizer is mock_tokenizer

    def test_load_called_with_correct_model_id(self, app_module):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        with patch.object(
            app_module, "load", return_value=(mock_model, mock_tokenizer)
        ) as mock_load:
            app_module.load_model()
        mock_load.assert_called_once_with("mlx-community/translategemma-4b-it-8bit")

    def test_registers_end_of_turn_as_eos_token(self, app_module):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        with patch.object(
            app_module, "load", return_value=(mock_model, mock_tokenizer)
        ):
            app_module.load_model()
        mock_tokenizer.add_eos_token.assert_called_once_with("<end_of_turn>")

    def test_cache_decorator_owns_the_loading_spinner(self, app_module):
        # The cache's cache-miss spinner is the one loading indicator. A
        # st.spinner wrapper around load_model() stacks a second spinner
        # ("Running `load_model()`.") under it on every cold load.
        assert app_module.st.cache_resource_kwargs == [
            {"show_spinner": "Loading model..."}
        ]
        app_module.st.spinner.assert_not_called()


class TestStreamingClickPath:
    """End-to-end tests using Streamlit's AppTest harness.

    Covers UI branches that import-time MagicMock fixtures can't reach:
    the streaming click path and the settled re-render after it, the
    output box's contents in each state, the button row's shape in each
    state, the model-load error handler, runtime target-list filtering,
    the swap button, and the empty-text warning. Every `.run()` — the
    fixture's first and each widget's — asserts the script raised nothing
    uncaught (`_CheckedAppTest` in conftest overrides `AppTest._run`, the
    funnel both reach), so a crash the script does not catch fails the test
    even when the assertions that follow would still hold.
    """

    def test_translate_click_streams_into_session_state(self, app_test, fake_mlx_lm):
        fake_mlx_lm.stream_generate.return_value = [
            SimpleNamespace(text="Hola"),
            SimpleNamespace(text=" "),
            SimpleNamespace(text="mundo"),
        ]
        app_test.text_area(key="source_text").input("Hello").run()
        app_test.button(key="translate_text").click().run()

        assert app_test.session_state["translation_result"] == "Hola mundo"
        fake_mlx_lm.stream_generate.assert_called_once()

    def test_settled_translation_renders_as_text_inside_the_box(
        self, app_test, fake_mlx_lm
    ):
        fake_mlx_lm.stream_generate.return_value = _fake_stream("Hola", " ", "mundo")
        app_test.text_area(key="source_text").input("Hello").run()
        app_test.button(key="translate_text").click().run()

        # After the post-stream rerun the bordered box holds the result as
        # st.text — full text colour — not a disabled text_area, and nothing
        # else renders as st.text anywhere on the page.
        box = _output_box(app_test)
        assert _box_contents(box) == [("text", "Hola mundo")]
        assert [t.value for t in app_test.text] == ["Hola mundo"]
        assert len(app_test.text_area) == 1  # the source only
        assert app_test.download_button(key="download_text").disabled is False

    def test_stream_writes_into_the_box_before_the_rerun(self, app_test, fake_mlx_lm):
        # A generator that yields then raises: the error path skips
        # st.rerun(), so the mid-stream tree survives for inspection. This
        # is the only test that sees the streaming write itself.
        def _yield_then_raise():
            yield SimpleNamespace(text="Hola")
            raise RuntimeError("boom")

        fake_mlx_lm.stream_generate.return_value = _yield_then_raise()
        app_test.text_area(key="source_text").input("Hello").run()
        app_test.button(key="translate_text").click().run()

        assert _box_contents(_output_box(app_test)) == [("text", "Hola")]
        assert any("boom" in e.value for e in app_test.error)

    def test_failure_before_first_chunk_restores_the_settled_box(
        self, app_test, fake_mlx_lm
    ):
        # A previous result is on screen and Download offers it; a second
        # translation that dies before streaming anything must not leave
        # "Translating…" (or nothing) in the box.
        fake_mlx_lm.stream_generate.return_value = _fake_stream("Hola", " ", "mundo")
        app_test.text_area(key="source_text").input("Hello").run()
        app_test.button(key="translate_text").click().run()
        assert _box_contents(_output_box(app_test)) == [("text", "Hola mundo")]

        fake_mlx_lm.stream_generate.side_effect = RuntimeError("boom")
        app_test.button(key="translate_text").click().run()

        assert _box_contents(_output_box(app_test)) == [("text", "Hola mundo")]
        assert any("boom" in e.value for e in app_test.error)
        assert app_test.download_button(key="download_text").disabled is False

    def test_page_column_is_the_main_blocks_only_child(self, app_test, fake_mlx_lm):
        # Everything renders inside the centred page column; a top-level
        # st.* call outside the with block would appear here as a sibling
        # and render full-bleed under wide. Checked in the two states that
        # render outside the columns — the empty-text warning and the
        # failure callout — because those live in the `if translate_clicked:`
        # tail, the block a dedent at the end of the file would strand.
        def main_children():
            return [c.type for c in app_test.main.children.values()]

        assert main_children() == ["flex_container"]  # empty

        app_test.button(key="translate_text").click().run()
        assert app_test.warning  # precondition: the warning branch ran
        assert main_children() == ["flex_container"]

        fake_mlx_lm.stream_generate.side_effect = RuntimeError("boom")
        app_test.text_area(key="source_text").input("Hello").run()
        app_test.button(key="translate_text").click().run()
        assert app_test.error  # precondition: the failure branch ran
        assert main_children() == ["flex_container"]

    def test_empty_output_box_shows_placeholder_caption(self, app_test):
        assert _box_contents(_output_box(app_test)) == [("caption", "Translation")]
        assert not app_test.text
        assert app_test.download_button(key="download_text").disabled is True

    def test_over_budget_input_disables_translate_button(
        self, app_test, mock_tokenizer
    ):
        # Force the cached tokenizer to report > MAX_PROMPT_TOKENS (1024).
        mock_tokenizer.encode.return_value = list(range(2000))
        app_test.text_area(key="source_text").set_value("text").run()

        assert app_test.button(key="translate_text").disabled is True
        # The over-budget indicator renders as a red badge (a markdown
        # element) carrying the error icon and the count to trim to — the
        # only place the number appears; there is no caption counter.
        assert any(
            "red-badge[" in m.value
            and ":material/error:" in m.value
            and "Too long: 2000 / 1024 tokens" in m.value
            for m in app_test.markdown
        )
        assert not any("tokens" in c.value for c in app_test.caption)

    def test_buttons_directly_follow_their_panels_in_every_state(
        self, app_test, fake_mlx_lm, mock_tokenizer
    ):
        # Nothing renders under budget, the badge only ever renders below
        # Translate, and the right column has nothing between the box and
        # Download, so the two buttons sit level in every state. See
        # TestButtonLayout.
        left = ["text_area", "button"]
        right = ["flex_container", "download_button"]
        assert _column_shapes(app_test) == (left, right)  # empty

        app_test.text_area(key="source_text").input("Hello").run()
        assert _column_shapes(app_test) == (left, right)  # under budget

        fake_mlx_lm.stream_generate.return_value = _fake_stream("Hola")
        app_test.button(key="translate_text").click().run()
        assert app_test.download_button(key="download_text").disabled is False
        assert _column_shapes(app_test) == (left, right)  # with a result

        mock_tokenizer.encode.return_value = list(range(2000))
        app_test.text_area(key="source_text").set_value("text").run()
        assert _column_shapes(app_test) == ([*left, "markdown"], right)  # badge

    def test_translation_exception_logs_and_shows_error(
        self, app_test, fake_mlx_lm, caplog
    ):
        fake_mlx_lm.stream_generate.side_effect = RuntimeError("model crashed")
        app_test.text_area(key="source_text").input("Hello").run()
        with caplog.at_level("ERROR"):
            app_test.button(key="translate_text").click().run()

        assert any("model crashed" in e.value for e in app_test.error)
        assert any(e.icon == ":material/error:" for e in app_test.error)
        assert any("Translation failed" in r.message for r in caplog.records)

    def test_model_load_failure_logs_and_shows_error(
        self, app_test_unrun, fake_mlx_lm, caplog
    ):
        fake_mlx_lm.load.side_effect = RuntimeError("model gone")
        with caplog.at_level("ERROR"):
            app_test_unrun.run()

        assert any("Failed to load model" in e.value for e in app_test_unrun.error)
        assert any(e.icon == ":material/error:" for e in app_test_unrun.error)
        assert any("Failed to load model" in r.message for r in caplog.records)
        # The failure state pins where the load sits: the page column holds
        # the title, the completed selector row and then the error at its own
        # level — a load above the row drops the row, one inside any column
        # nests the error, one past st.columns(2) adds a second block — and
        # the whole row rendered first (two selectboxes, the swap button).
        page_column = next(
            iter(next(iter(app_test_unrun.main.children.values())).children.values())
        )
        assert [c.type for c in page_column.children.values()] == [
            "title",
            "flex_container",
            "error",
        ]
        assert len(app_test_unrun.selectbox) == 2
        assert len(app_test_unrun.button) == 1  # the swap button
        assert not app_test_unrun.text_area

    def test_model_load_unexpected_shape_logs_and_shows_error(
        self, app_test_unrun, fake_mlx_lm, caplog
    ):
        # load(return_config=True) yields a 3-tuple; load_model() must refuse
        # anything but (model, tokenizer) rather than unpack it by accident.
        fake_mlx_lm.load.return_value = (MagicMock(), MagicMock(), {})
        with caplog.at_level("ERROR"):
            app_test_unrun.run()

        assert any("expected 2" in e.value for e in app_test_unrun.error)
        assert any("Failed to load model" in r.message for r in caplog.records)

    def test_non_english_source_restricts_target_to_english(self, app_test):
        # Default state: source=English, target=Spanish.
        # Switching source to a bidirectional non-English language must
        # collapse valid targets to ["English"] and reset target_lang.
        app_test.selectbox(key="source_lang").select("French").run()

        assert app_test.session_state["source_lang"] == "French"
        assert app_test.session_state["target_lang"] == "English"

    def test_empty_text_translate_click_shows_warning(self, app_test):
        # Default source_text is empty; clicking Translate should warn,
        # not invoke the model.
        app_test.button(key="translate_text").click().run()

        assert any(
            "Please enter text to translate" in w.value for w in app_test.warning
        )
        assert any(w.icon == ":material/warning:" for w in app_test.warning)

    def test_swap_button_swaps_source_and_target(self, app_test):
        assert app_test.session_state["source_lang"] == "English"
        assert app_test.session_state["target_lang"] == "Spanish"
        # The swap button renders before the translate button, so it's button[0].
        app_test.button[0].click().run()

        assert app_test.session_state["source_lang"] == "Spanish"
        assert app_test.session_state["target_lang"] == "English"


class TestCheckedAppTest:
    def test_widget_run_that_raises_fails_the_test(self, crashing_app_test):
        # The crash check overrides the private AppTest._run, the one method
        # both at.run() and a widget's .run() reach. If a Streamlit bump
        # renames it, the override becomes dead code and every AppTest test
        # silently loses the check — this is the test that notices.
        crashing_app_test.run()  # the script is clean until the click
        with pytest.raises(AssertionError, match="boom"):
            crashing_app_test.button[0].click().run()


class TestThemeConfig:
    """Guard the shape of .streamlit/config.toml, not its colours.

    The app ships a custom theme with both ``[theme.light]`` and
    ``[theme.dark]`` designed. Streamlit keeps the in-app switcher with
    either section alone, but the mode that has no section silently falls
    back to the stock palette under the shared ``[theme]`` font, radius and
    borders — a half-designed mode, with every other check green.
    ``.gitignore`` un-ignores exactly this path, so a stray local edit
    commits by default.

    The option-name check is the other trap turned into a test: ``base``
    is a top-level ``[theme]`` key only, and Streamlit logs an unknown key
    inside a mode section once at startup rather than failing, so a
    misplaced ``base`` or a typo would otherwise be caught by nobody.
    """

    config_path = Path(__file__).parent.parent / ".streamlit" / "config.toml"

    def _theme(self) -> dict:
        assert self.config_path.exists(), (
            f"{self.config_path} must exist — the app ships a custom theme. "
            "See CLAUDE.md → Architecture → Theme."
        )
        theme = tomllib.loads(self.config_path.read_text()).get("theme")
        assert isinstance(theme, dict), f"{self.config_path} has no [theme] table"
        return theme

    def test_config_exists_and_has_a_theme_table(self):
        self._theme()

    def test_both_mode_sections_are_defined(self):
        theme = self._theme()
        for mode in ("light", "dark"):
            assert theme.get(mode), (
                f"[theme.{mode}] is missing or empty — that mode would fall "
                "back to Streamlit's stock palette under the shared [theme] "
                "keys. Both modes are designed on purpose."
            )

    def test_every_key_is_a_registered_streamlit_option(self):
        import streamlit.config

        registered = streamlit.config.get_config_options()

        def dotted(table: dict, prefix: str) -> list[str]:
            keys = []
            for name, value in table.items():
                if isinstance(value, dict):
                    keys.extend(dotted(value, f"{prefix}.{name}"))
                else:
                    keys.append(f"{prefix}.{name}")
            return keys

        unknown = [k for k in dotted(self._theme(), "theme") if k not in registered]
        assert not unknown, (
            f"Not Streamlit config options: {unknown}. `base` is only valid "
            "directly under [theme], never inside [theme.light]/[theme.dark]."
        )

    def test_heading_sizes_are_a_full_decreasing_scale(self):
        # A shorter array pins only the levels it names and leaves the rest
        # at stock, so ["2rem"] alone left h2 at 2.25rem — larger than h1.
        # The key is also registered per mode, so a mode section carrying it
        # is held to the same shape. Shape only: the values themselves are
        # the theme's business; rem because the file's other sizes are rem.
        theme = self._theme()
        tables = {"[theme]": theme}
        for mode in ("light", "dark"):
            if "headingFontSizes" in theme.get(mode, {}):
                tables[f"[theme.{mode}]"] = theme[mode]
        assert "headingFontSizes" in theme, "[theme] must pin headingFontSizes"
        for where, table in tables.items():
            sizes = table["headingFontSizes"]
            assert isinstance(sizes, list) and len(sizes) == 6, (
                f"{where} headingFontSizes must name all six levels, got {sizes!r}"
            )
            matches = [re.fullmatch(r"(\d+(?:\.\d+)?)rem", str(v)) for v in sizes]
            assert all(matches), f"{where} headingFontSizes must be rem, got {sizes!r}"
            rems = [float(m.group(1)) for m in matches if m]
            assert rems == sorted(rems, reverse=True) and len(set(rems)) == 6, (
                f"{where} headingFontSizes must decrease from h1 to h6, got {sizes!r}"
            )

    def test_every_colour_value_is_six_digit_hex(self):
        # The frontend drops a malformed colour with only a console warning
        # and paints the stock palette instead — the theme's whole purpose,
        # silently undone. Streamlit accepts other formats; this file uses
        # six-digit hex throughout, so anything else here is a typo.
        def colours(table: dict) -> list[tuple[str, object]]:
            found = []
            for name, value in table.items():
                if isinstance(value, dict):
                    found.extend(colours(value))
                elif name.endswith("Color"):
                    found.append((name, value))
            return found

        bad = [
            (name, value)
            for name, value in colours(self._theme())
            if not (isinstance(value, str) and re.fullmatch(r"#[0-9A-Fa-f]{6}", value))
        ]
        assert not bad, f"Colour values that are not #RRGGBB: {bad}"
