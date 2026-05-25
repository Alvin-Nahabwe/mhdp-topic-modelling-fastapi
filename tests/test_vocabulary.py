"""
Unit tests for MHDP clinical vocabulary and stop words modules.

These tests run WITHOUT loading any ML models — they validate the
vocabulary/stop-word integrity that guards the inference pipeline.

Run: pytest tests/test_vocabulary.py -v
"""

import re
import pytest


class TestClinicalVocabulary:
    """Tests for clinical_vocabulary.py module integrity."""

    def test_hitop_vocabulary_not_empty(self):
        from clinical_vocabulary import HITOP_VOCABULARY
        assert len(HITOP_VOCABULARY) > 0, "HITOP_VOCABULARY should not be empty"

    def test_hitop_vocabulary_has_expected_categories(self):
        from clinical_vocabulary import HITOP_VOCABULARY
        expected = {
            "Hallucination/Delusions",
            "Low mood/Anhedonia",
            "Anxiety spectrum",
            "Insomnia/Hypersomnia",
            "Suicidal behaviour",
        }
        actual = set(HITOP_VOCABULARY.keys())
        missing = expected - actual
        assert not missing, f"Missing expected categories: {missing}"

    def test_all_terms_are_strings(self):
        from clinical_vocabulary import HITOP_VOCABULARY
        for category, terms in HITOP_VOCABULARY.items():
            assert isinstance(terms, (list, tuple, set)), \
                f"Category '{category}' terms should be iterable"
            for term in terms:
                assert isinstance(term, str), \
                    f"Term '{term}' in '{category}' is not a string"

    def test_minimum_term_count(self):
        """We documented 213 terms — ensure no accidental truncation."""
        from clinical_vocabulary import HITOP_VOCABULARY
        total = sum(len(terms) for terms in HITOP_VOCABULARY.values())
        assert total >= 200, \
            f"Expected ≥200 clinical terms, got {total}. Vocabulary may be truncated."

    def test_get_seed_topics_returns_list(self):
        from clinical_vocabulary import get_seed_topics
        seeds = get_seed_topics()
        # BERTopic guided mode expects list of lists
        assert isinstance(seeds, list), "get_seed_topics() should return a list"
        assert len(seeds) > 0, "Seed topics should not be empty"
        assert all(isinstance(s, list) for s in seeds), "Each seed should be a list of terms"

    def test_no_clinical_overlap_with_stop_words(self):
        """Critical: clinical terms must never be filtered as stop words."""
        from clinical_vocabulary import HITOP_VOCABULARY
        from stop_words import ALL_STOP_WORDS

        all_clinical = set()
        for terms in HITOP_VOCABULARY.values():
            all_clinical.update(t.lower() for t in terms)

        overlap = all_clinical & ALL_STOP_WORDS
        assert not overlap, \
            f"CRITICAL: {len(overlap)} clinical terms found in stop words: {overlap}"


class TestStopWords:
    """Tests for stop_words.py module integrity."""

    def test_stop_words_not_empty(self):
        from stop_words import ALL_STOP_WORDS
        assert len(ALL_STOP_WORDS) > 0

    def test_stop_words_is_set(self):
        from stop_words import ALL_STOP_WORDS
        assert isinstance(ALL_STOP_WORDS, (set, frozenset)), \
            "ALL_STOP_WORDS should be a set for O(1) lookup"

    def test_critical_clinical_terms_not_in_stop_words(self):
        """These specific terms were previously causing signal loss."""
        from stop_words import ALL_STOP_WORDS
        # These 6 terms were removed from stop words during Phase 3 audit
        must_keep = ["feel", "things", "lot", "going", "thing", "alot"]
        leaked = [t for t in must_keep if t in ALL_STOP_WORDS]
        assert not leaked, \
            f"Clinical terms incorrectly in stop words: {leaked}"

    def test_clinical_exceptions_preserved(self):
        """sklearn stop words that have clinical meaning must be kept."""
        from stop_words import ALL_STOP_WORDS
        # These are clinically relevant and should NOT be in stop words
        clinical_exceptions = ["alone", "down", "interest"]
        leaked = [t for t in clinical_exceptions if t in ALL_STOP_WORDS]
        assert not leaked, \
            f"Clinical exception terms incorrectly in stop words: {leaked}"


class TestClinicalRegex:
    """Tests for the compiled clinical vocabulary regex pattern."""

    @pytest.fixture
    def clinical_pattern(self):
        from clinical_vocabulary import HITOP_VOCABULARY
        all_terms = set()
        for terms in HITOP_VOCABULARY.values():
            all_terms.update(t.lower() for t in terms)
        sorted_terms = sorted(all_terms, key=len, reverse=True)
        return re.compile(
            r'\b(?:' + '|'.join(re.escape(t) for t in sorted_terms) + r')\b',
            re.IGNORECASE
        )

    @pytest.mark.parametrize("text,expected_match", [
        ("I feel sad and hopeless", True),
        ("hearing voices at night", True),
        ("I have been feeling anxious", True),
        ("I cannot sleep at all", True),
        ("I want to kill myself", True),
        ("I feel worthless and guilty", True),
    ])
    def test_clinical_text_detected(self, clinical_pattern, text, expected_match):
        has_match = bool(clinical_pattern.search(text))
        assert has_match == expected_match, \
            f"Expected clinical match={expected_match} for: '{text}'"

    @pytest.mark.parametrize("text", [
        "hello how are you today",
        "what time is it",
        "the weather is nice",
    ])
    def test_non_clinical_text_rejected(self, clinical_pattern, text):
        matches = clinical_pattern.findall(text)
        assert len(matches) == 0, \
            f"Non-clinical text should not match, but found: {matches}"
