from unittest.mock import MagicMock, call, patch

import pytest


def _caption_texts(app_module):
    return [c.args[0] for c in app_module.st.caption.call_args_list if c.args]


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


class TestTargetFiltering:
    def test_english_source_includes_bidirectional_targets(self, app_module):
        targets = set(app_module.TARGET_LANGS_FOR_ENGLISH)
        assert "French" in targets
        assert "Japanese" in targets
        assert "Swahili" in targets

    def test_english_source_includes_from_english_only_targets(self, app_module):
        targets = set(app_module.TARGET_LANGS_FOR_ENGLISH)
        assert "Albanian" in targets
        assert "Ukrainian" in targets
        assert "Tamil" in targets

    def test_english_source_excludes_english(self, app_module):
        assert "English" not in app_module.TARGET_LANGS_FOR_ENGLISH

    def test_non_english_source_targets_only_english(self, app_module):
        # A non-English source can only translate to English, which must
        # itself be a valid (bidirectional) source/target language.
        non_english_sources = [s for s in app_module.SOURCE_LANGS if s != "English"]
        assert non_english_sources
        assert "English" in app_module.SOURCE_LANGS

    def test_from_english_only_not_in_source_langs(self, app_module):
        for name in app_module.FROM_ENGLISH_ONLY:
            assert name not in app_module.SOURCE_LANGS, (
                f"{name} is from-English-only but appears in SOURCE_LANGS"
            )


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
        app_module.generate.assert_called_once_with(
            patched_translate["model"],
            patched_translate["tokenizer"],
            prompt=expected_prompt,
            max_tokens=app_module.CONTEXT_WINDOW - prompt_tokens,
        )

    def test_generate_called_exactly_once(self, app_module, patched_translate):
        patched_translate["translate"]("Hello", "English", "en", "Spanish", "es")
        assert app_module.generate.call_count == 1

    def test_strips_end_of_turn_token(self, app_module, mock_tokenizer):
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
                return_value="hola mundo<end_of_turn>",
            ),
        ):
            result = app_module.translate(
                "hello world", "English", "en", "Spanish", "es"
            )
        assert result == "hola mundo"

    def test_strips_repeated_end_of_turn_tokens(self, app_module, mock_tokenizer):
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
                return_value="hola mundo<end_of_turn><end_of_turn><end_of_turn>",
            ),
        ):
            result = app_module.translate(
                "hello world", "English", "en", "Spanish", "es"
            )
        assert result == "hola mundo"

    def test_clean_output_unchanged(self, app_module, mock_tokenizer):
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
                return_value="hola mundo",
            ),
        ):
            result = app_module.translate(
                "hello world", "English", "en", "Spanish", "es"
            )
        assert result == "hola mundo"

    def test_strips_whitespace_around_translation(self, app_module, mock_tokenizer):
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
                return_value="  hola mundo  <end_of_turn>",
            ),
        ):
            result = app_module.translate(
                "hello world", "English", "en", "Spanish", "es"
            )
        assert result == "hola mundo"

    def test_strips_content_after_end_of_turn(self, app_module, mock_tokenizer):
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
                return_value="hola mundo<end_of_turn>extra garbage",
            ),
        ):
            result = app_module.translate(
                "hello world", "English", "en", "Spanish", "es"
            )
        assert result == "hola mundo"

    def test_raises_when_prompt_exceeds_budget(self, app_module, mock_tokenizer):
        mock_tokenizer.encode.return_value = list(
            range(app_module.MAX_PROMPT_TOKENS + 1)
        )
        with (
            patch.object(
                app_module,
                "load_model",
                return_value=(MagicMock(), mock_tokenizer),
            ),
            patch.object(app_module, "generate") as mock_generate,
        ):
            with pytest.raises(ValueError, match="too long"):
                app_module.translate("text", "English", "en", "Spanish", "es")
        mock_generate.assert_not_called()

    def test_allows_prompt_at_budget_limit(self, app_module, mock_tokenizer):
        mock_tokenizer.encode.return_value = list(range(app_module.MAX_PROMPT_TOKENS))
        with (
            patch.object(
                app_module,
                "load_model",
                return_value=(MagicMock(), mock_tokenizer),
            ),
            patch.object(app_module, "generate", return_value="ok"),
        ):
            result = app_module.translate("text", "English", "en", "Spanish", "es")
        assert result == "ok"


class TestHeader:
    def test_page_title(self, app_module):
        app_module.st.set_page_config.assert_called_once()
        kwargs = app_module.st.set_page_config.call_args.kwargs
        assert kwargs["page_title"] == "TranslateGemma Pipeline"

    def test_title(self, app_module):
        app_module.st.title.assert_called_once_with("TranslateGemma Pipeline")

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
        assert calls[0] == call([10, 1, 10])

    def test_content_columns(self, app_module):
        calls = app_module.st.columns.call_args_list
        assert calls[1] == call(2)


class TestTokenCounter:
    def test_token_count_caption_rendered(self, app_module):
        captions = _caption_texts(app_module)
        token_budget = f"/ {app_module.MAX_PROMPT_TOKENS} tokens"
        assert any(token_budget in text for text in captions)

    def test_right_column_has_alignment_spacer(self, app_module):
        captions = _caption_texts(app_module)
        assert "&nbsp;" in captions


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
