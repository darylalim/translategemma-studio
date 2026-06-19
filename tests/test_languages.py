class TestBidirectional:
    def test_english_in_bidirectional(self):
        from languages import BIDIRECTIONAL

        assert "English" in BIDIRECTIONAL
        assert BIDIRECTIONAL["English"] == "en"

    def test_bidirectional_count(self):
        from languages import BIDIRECTIONAL

        assert len(BIDIRECTIONAL) == 225

    def test_all_values_are_nonempty_strings(self):
        from languages import BIDIRECTIONAL

        for name, code in BIDIRECTIONAL.items():
            assert isinstance(code, str) and code, f"{name}: empty or non-string code"

    def test_sample_bidirectional_codes(self):
        from languages import BIDIRECTIONAL

        samples = {
            "French": "fr",
            "German": "de",
            "Japanese": "ja",
            "Chinese": "zh-CN",
            "Dari": "fa-AF",
            "Spanish": "es",
            "Thai": "th",
            "Swahili": "sw",
        }
        for name, expected_code in samples.items():
            assert BIDIRECTIONAL[name] == expected_code, f"{name} code mismatch"


class TestFromEnglishOnly:
    def test_from_english_only_count(self):
        from languages import FROM_ENGLISH_ONLY

        assert len(FROM_ENGLISH_ONLY) == 70

    def test_all_values_are_nonempty_strings(self):
        from languages import FROM_ENGLISH_ONLY

        for name, code in FROM_ENGLISH_ONLY.items():
            assert isinstance(code, str) and code, f"{name}: empty or non-string code"

    def test_sample_from_english_only_codes(self):
        from languages import FROM_ENGLISH_ONLY

        samples = {
            "Albanian": "sq",
            "Arabic (Egypt)": "ar-EG",
            "Portuguese (Brazil)": "pt-BR",
            "Ukrainian": "uk",
            "Tamil": "ta",
        }
        for name, expected_code in samples.items():
            assert FROM_ENGLISH_ONLY[name] == expected_code, f"{name} code mismatch"

    def test_english_not_in_from_english_only(self):
        from languages import FROM_ENGLISH_ONLY

        assert "English" not in FROM_ENGLISH_ONLY


class TestNoOverlap:
    def test_no_overlapping_keys(self):
        from languages import BIDIRECTIONAL, FROM_ENGLISH_ONLY

        overlap = set(BIDIRECTIONAL) & set(FROM_ENGLISH_ONLY)
        assert not overlap, f"Overlapping keys: {overlap}"

    def test_all_codes_are_unique(self):
        from languages import ALL_LANGUAGES

        codes = list(ALL_LANGUAGES.values())
        assert len(codes) == len(set(codes)), "Duplicate language codes found"


class TestAllLanguages:
    def test_all_languages_count(self):
        from languages import ALL_LANGUAGES

        assert len(ALL_LANGUAGES) == 295

    def test_contains_bidirectional_and_from_english(self):
        from languages import ALL_LANGUAGES, BIDIRECTIONAL, FROM_ENGLISH_ONLY

        for name in BIDIRECTIONAL:
            assert name in ALL_LANGUAGES
        for name in FROM_ENGLISH_ONLY:
            assert name in ALL_LANGUAGES


class TestSourceLangs:
    def test_source_langs_is_sorted(self):
        from languages import SOURCE_LANGS

        assert sorted(SOURCE_LANGS) == SOURCE_LANGS

    def test_source_langs_only_contains_bidirectional(self):
        from languages import BIDIRECTIONAL, SOURCE_LANGS

        assert set(SOURCE_LANGS) == set(BIDIRECTIONAL.keys())

    def test_source_langs_count(self):
        from languages import SOURCE_LANGS

        assert len(SOURCE_LANGS) == 225


class TestTargetLangsForEnglish:
    def test_target_langs_for_english_is_sorted(self):
        from languages import TARGET_LANGS_FOR_ENGLISH

        assert sorted(TARGET_LANGS_FOR_ENGLISH) == TARGET_LANGS_FOR_ENGLISH

    def test_english_not_in_target_langs_for_english(self):
        from languages import TARGET_LANGS_FOR_ENGLISH

        assert "English" not in TARGET_LANGS_FOR_ENGLISH

    def test_target_langs_for_english_count(self):
        from languages import TARGET_LANGS_FOR_ENGLISH

        assert len(TARGET_LANGS_FOR_ENGLISH) == 294

    def test_contains_both_bidirectional_and_from_english(self):
        from languages import (
            BIDIRECTIONAL,
            FROM_ENGLISH_ONLY,
            TARGET_LANGS_FOR_ENGLISH,
        )

        targets = set(TARGET_LANGS_FOR_ENGLISH)
        for name in BIDIRECTIONAL:
            if name != "English":
                assert name in targets, f"{name} missing from targets"
        for name in FROM_ENGLISH_ONLY:
            assert name in targets, f"{name} missing from targets"
