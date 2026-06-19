import tomllib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest


def _caption_texts(app_module):
    return [c.args[0] for c in app_module.st.caption.call_args_list if c.args]


def _fake_stream(*segments):
    # Stand in for mlx-lm's stream of GenerationResponse objects; only the
    # .text attribute is read by translate_stream().
    return [SimpleNamespace(text=s) for s in segments]


class TestConstants:
    def test_model_id(self, app_module):
        assert app_module.MODEL_ID == "mlx-community/translategemma-4b-it-8bit"

    def test_context_window(self, app_module):
        assert app_module.CONTEXT_WINDOW == 2048

    def test_max_prompt_tokens(self, app_module):
        assert app_module.MAX_PROMPT_TOKENS == 1024

    def test_max_input_chars(self, app_module):
        assert app_module.MAX_INPUT_CHARS == 5000

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
        assert mock_state["source_lang"] == "Spanish"
        assert mock_state["target_lang"] == "English"

    def test_swap_is_reversible(self, app_module):
        mock_state = {"source_lang": "English", "target_lang": "French"}
        with patch.object(app_module.st, "session_state", mock_state):
            app_module._swap_languages()
            app_module._swap_languages()
        assert mock_state["source_lang"] == "English"
        assert mock_state["target_lang"] == "French"

    def test_swap_copies_translation_to_source_text(self, app_module):
        mock_state = {
            "source_lang": "English",
            "target_lang": "Spanish",
            "translation_result": "hola",
        }
        with patch.object(app_module.st, "session_state", mock_state):
            app_module._swap_languages()
        assert mock_state["source_text"] == "hola"
        assert "translation_result" not in mock_state

    def test_double_swap_with_translation_is_not_reversible(self, app_module):
        mock_state = {
            "source_lang": "English",
            "target_lang": "Spanish",
            "translation_result": "hola",
        }
        with patch.object(app_module.st, "session_state", mock_state):
            app_module._swap_languages()
            app_module._swap_languages()
        assert mock_state["source_lang"] == "English"
        assert mock_state["target_lang"] == "Spanish"
        assert mock_state["source_text"] == "hola"
        assert "translation_result" not in mock_state

    def test_swap_without_translation_does_not_set_source_text(self, app_module):
        mock_state = {"source_lang": "English", "target_lang": "Spanish"}
        with patch.object(app_module.st, "session_state", mock_state):
            app_module._swap_languages()
        assert "source_text" not in mock_state


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
        assert mock_state["source_lang"] == "English"
        assert mock_state["target_lang"] == "Albanian"

    def test_swap_guard_allows_bidirectional(self, app_module):
        mock_state = {
            "source_lang": "English",
            "target_lang": "French",
        }
        with patch.object(app_module.st, "session_state", mock_state):
            app_module._swap_languages()
        assert mock_state["source_lang"] == "French"
        assert mock_state["target_lang"] == "English"


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

    def test_title(self, app_module):
        app_module.st.title.assert_called_once_with("TranslateGemma Studio")

    def test_caption_links_the_model(self, app_module):
        captions = _caption_texts(app_module)
        assert any(
            "[Google TranslateGemma 4B model]" in text
            and "https://huggingface.co/google/translategemma-4b-it" in text
            for text in captions
        )


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


class TestTokenCounter:
    def test_token_count_caption_rendered(self, app_module):
        captions = _caption_texts(app_module)
        token_budget = f"/ {app_module.MAX_PROMPT_TOKENS} tokens"
        assert any(token_budget in text for text in captions)

    def test_right_column_has_alignment_spacer(self, app_module):
        app_module.st.space.assert_called_once_with("small")


class TestOutputPlaceholder:
    def test_output_slot_uses_st_empty_placeholder(self, app_module):
        # The output slot in the right column is an st.empty() placeholder
        # so the streaming handler can swap content in without re-rendering
        # surrounding elements.
        app_module.st.empty.assert_called_once()

    def test_text_area_rendered_into_placeholder(self, app_module):
        # When not streaming (the default path during import), the disabled
        # translation text_area is rendered inside the placeholder — not at
        # the top level. The args match the prior settled-view styling.
        placeholder = app_module.st.empty.return_value
        placeholder.text_area.assert_called_once_with(
            "Translation output",
            placeholder="Translation",
            disabled=True,
            height=300,
            label_visibility="collapsed",
            key="text_output",
        )


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


class TestStreamingClickPath:
    """End-to-end tests using Streamlit's AppTest harness.

    Covers UI branches that import-time MagicMock fixtures can't reach:
    the streaming click path, the model-load error handler, runtime
    target-list filtering, and the empty-text warning.
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

    def test_over_budget_input_disables_translate_button(
        self, app_test, mock_tokenizer
    ):
        # Force the cached tokenizer to report > MAX_PROMPT_TOKENS (1024).
        mock_tokenizer.encode.return_value = list(range(2000))
        # set_value bypasses max_chars so we can stage any prompt length.
        app_test.text_area(key="source_text").set_value("text").run()

        assert app_test.button(key="translate_text").disabled is True
        # The over-budget indicator renders as a red badge (a markdown
        # element) carrying the error icon, not inline caption text.
        assert any(
            "red-badge[" in m.value
            and ":material/error:" in m.value
            and "Too long to translate" in m.value
            for m in app_test.markdown
        )

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


class TestThemeConfig:
    """Guard .streamlit/config.toml against invalid theme option keys.

    Streamlit only *logs* a warning for an unknown config option, so an
    invalid key (e.g. a per-variant ``base``) slips past the rest of the
    suite — the running server is the only place it surfaces. These tests
    check every key against Streamlit's own option template, the same lookup
    the runtime uses.
    """

    _CONFIG_PATH = Path(__file__).parent.parent / ".streamlit" / "config.toml"

    def _load(self):
        return tomllib.loads(self._CONFIG_PATH.read_text())

    def _flatten(self, mapping, prefix=""):
        for key, value in mapping.items():
            dotted = f"{prefix}{key}"
            if isinstance(value, dict):
                yield from self._flatten(value, f"{dotted}.")
            else:
                yield dotted

    def test_config_file_exists(self):
        assert self._CONFIG_PATH.is_file()

    def test_defines_light_and_dark_variants(self):
        theme = self._load()["theme"]
        assert "light" in theme
        assert "dark" in theme

    def test_all_keys_are_valid_streamlit_options(self):
        from streamlit import config

        valid = set(config._config_options_template)
        invalid = [k for k in self._flatten(self._load()) if k not in valid]
        assert invalid == [], f"Invalid config options: {invalid}"
