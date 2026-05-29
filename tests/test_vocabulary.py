"""
Unit tests for MHDP clinical vocabulary and stop words modules.

These tests run WITHOUT loading any ML models — they validate the
vocabulary/stop-word integrity that guards the inference pipeline.

Run: pytest tests/test_vocabulary.py -v
"""

import pytest


class TestClinicalVocabulary:
    """Tests for clinical_vocabulary.py module integrity."""

    def test_hitop_vocabulary_not_empty(self):
        from clinical_vocabulary import HITOP_VOCABULARY
        assert len(HITOP_VOCABULARY) > 0

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
        assert not missing, (
            f"Missing expected categories: {missing}"
        )

    def test_all_terms_are_strings(self):
        from clinical_vocabulary import HITOP_VOCABULARY
        for category, terms in HITOP_VOCABULARY.items():
            assert isinstance(terms, (list, tuple, set)), (
                f"'{category}' terms should be iterable"
            )
            for term in terms:
                assert isinstance(term, str), (
                    f"'{term}' in '{category}' is not a str"
                )

    def test_safe_unigrams_not_empty(self):
        from clinical_vocabulary import SAFE_CLINICAL_UNIGRAMS
        assert len(SAFE_CLINICAL_UNIGRAMS) >= 30, (
            "Expected >= 30 safe clinical unigrams"
        )

    def test_clinical_phrases_not_empty(self):
        from clinical_vocabulary import CLINICAL_PHRASES
        total = sum(
            len(phrases)
            for phrases in CLINICAL_PHRASES.values()
        )
        assert total >= 100, (
            f"Expected >= 100 clinical phrases, got {total}"
        )

    def test_clinical_phrases_are_multi_word(self):
        """All phrases should be multi-word (>= 2 words)."""
        from clinical_vocabulary import CLINICAL_PHRASES
        for category, phrases in CLINICAL_PHRASES.items():
            for phrase in phrases:
                word_count = len(phrase.split())
                assert word_count >= 2, (
                    f"'{phrase}' in '{category}' is a"
                    f" single word, not a phrase"
                )

    def test_get_seed_topics_returns_list(self):
        from clinical_vocabulary import get_seed_topics
        seeds = get_seed_topics()
        assert isinstance(seeds, list)
        assert len(seeds) > 0
        assert all(isinstance(s, list) for s in seeds)

    def test_seed_topics_exclude_common_words(self):
        """Seed topics should not contain high-risk words."""
        from clinical_vocabulary import get_seed_topics
        high_risk = {
            "food", "eat", "eating", "sleep", "sleeping",
            "work", "job", "talk", "talking", "fight",
            "heart", "alone", "slow", "fast", "rapid",
            "down", "edge", "happy", "hearing", "seeing",
            "energy", "focus", "pressure", "interest",
            "pleasure", "social", "relationship", "attack",
            "dream", "school", "fired", "scared", "afraid",
            "fear", "angry", "irritable", "hide", "hiding",
            "avoid", "strange", "weak", "hurt", "harm",
            "die", "dead", "kill", "relax", "concentrate",
        }
        seeds = get_seed_topics()
        all_seed_words = set()
        for topic in seeds:
            all_seed_words.update(w.lower() for w in topic)
        overlap = all_seed_words & high_risk
        assert not overlap, (
            f"High-risk words in seed topics: {overlap}"
        )


class TestSafeUnigrams:
    """Safe unigrams must be 'without a doubt' clinical."""

    def test_no_common_english_words_in_unigrams(self):
        """Common English words should NOT be safe unigrams."""
        from clinical_vocabulary import SAFE_CLINICAL_UNIGRAMS
        forbidden_common = {
            "food", "eat", "eating", "sleep", "sleeping",
            "work", "job", "talk", "talking", "fight",
            "heart", "alone", "slow", "fast", "rapid",
            "down", "edge", "happy", "hearing", "seeing",
            "energy", "focus", "pressure", "interest",
            "pleasure", "social", "relationship", "attack",
            "dream", "afraid", "angry", "scared", "fear",
            "irritable", "worried", "worry", "nervous",
            "restless", "confused", "guilty", "guilt",
            "shame", "panic", "terrified", "annoyed",
            "hide", "hiding", "avoid", "strange", "weak",
            "hurt", "harm", "die", "dead", "kill",
            "school", "fired", "relax", "concentrate",
        }
        overlap = SAFE_CLINICAL_UNIGRAMS & forbidden_common
        assert not overlap, (
            f"Common words in safe unigrams: {overlap}"
        )


class TestStopWords:
    """Tests for stop_words.py module integrity."""

    def test_stop_words_not_empty(self):
        from stop_words import ALL_STOP_WORDS
        assert len(ALL_STOP_WORDS) > 0

    def test_stop_words_is_set(self):
        from stop_words import ALL_STOP_WORDS
        assert isinstance(ALL_STOP_WORDS, (set, frozenset))

    def test_critical_clinical_terms_not_in_stop_words(self):
        from stop_words import ALL_STOP_WORDS
        must_keep = [
            "feel", "things", "lot", "going",
            "thing", "alot",
        ]
        leaked = [t for t in must_keep if t in ALL_STOP_WORDS]
        assert not leaked, (
            f"Clinical terms in stop words: {leaked}"
        )

    def test_safe_unigrams_not_in_stop_words(self):
        """Safe clinical unigrams must not be stop words."""
        from clinical_vocabulary import SAFE_CLINICAL_UNIGRAMS
        from stop_words import ALL_STOP_WORDS
        overlap = SAFE_CLINICAL_UNIGRAMS & ALL_STOP_WORDS
        assert not overlap, (
            f"Safe unigrams in stop words: {overlap}"
        )


class TestClinicalGate:
    """Tests for the phrase-based clinical gate function."""

    @pytest.mark.parametrize("text", [
        "I have been hearing voices and seeing things",
        "I feel depressed and hopeless all the time",
        "I want to kill myself and I have insomnia",
        "I have hallucination and depression",
        "I am suicidal and feeling anxious",
    ])
    def test_clinical_text_detected(self, text):
        from clinical_vocabulary import has_clinical_content
        assert has_clinical_content(text), (
            f"Should be clinical: '{text}'"
        )

    @pytest.mark.parametrize("text", [
        "hello how are you today",
        "what time is it",
        "the weather is nice",
        "I went to work and had food for lunch",
        "I was talking to my friend about school",
        "my heart was racing during the football game",
        "I need to focus on my studies at school",
        "I was alone at home watching television",
        "the bus arrived fast and I had to fight",
        "I had a dream about traveling to Europe",
    ])
    def test_non_clinical_text_rejected(self, text):
        from clinical_vocabulary import has_clinical_content
        assert not has_clinical_content(text), (
            f"Should NOT be clinical: '{text}'"
        )

    def test_single_clinical_term_insufficient(self):
        """A single clinical match should NOT pass the gate."""
        from clinical_vocabulary import has_clinical_content
        # Only one clinical unigram — should fail >= 2 check
        assert not has_clinical_content(
            "I feel depressed today but otherwise fine"
        )

    def test_two_clinical_terms_sufficient(self):
        """Two distinct clinical matches should pass the gate."""
        from clinical_vocabulary import has_clinical_content
        assert has_clinical_content(
            "I feel depressed and I have insomnia"
        )

    def test_phrase_matching_works(self):
        """Multi-word phrases should be detected."""
        from clinical_vocabulary import has_clinical_content
        # "hearing voices" is a phrase, "depression" is a
        # safe unigram — 2 matches
        assert has_clinical_content(
            "I have been hearing voices and depression"
        )

    def test_extract_clinical_terms(self):
        """extract_clinical_terms should return matched items."""
        from clinical_vocabulary import extract_clinical_terms
        terms = extract_clinical_terms(
            "I have hallucination and depression"
        )
        assert len(terms) >= 2
        assert "hallucination" in terms
        assert "depression" in terms
