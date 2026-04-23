"""
Shared stop words and filler words for the MHDP clinical topic pipeline.
Used in both training (vectorizer) and inference (keyword extraction).
"""

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Luganda function words (pronouns, conjunctions, prepositions, etc.)
LUGANDA_STOP_WORDS = [
    "nga", "nti", "kale", "bulungi", "aah", "eeh", "mmh", "hmm",
    "ssebo", "nnyabo", "wangi", "yee", "nedda", "ddala", "nnyo", "nze", "ggwe",
    "gwe", "ye", "ffe", "mmwe", "mwe", "bo", "wange", "kino", "ekyo", "bino",
    "ebyo", "ne", "mu", "ku", "wa", "nti", "bwe", "eri", "okuva", "paka", "ko",
    "ki", "kiki", "ani", "ddi", "lwaki", "otya", "mutya", "ndi", "oli", "ali",
    "tuli", "bali", "nina", "alina"
]

# Conversational fillers and domain-specific noise
CONVERSATIONAL_FILLERS = [
    "like", "just", "hmm", "eeh", "eh", "ah", "don", "t", "s",
    "kati", "awo", "waliwo", "na", "kuba", "umm", "mmh", "mpozi",
    # Additional fillers identified from topic representations and transcripts
    "lot", "alot", "dont", "feel", "yes", "got", "went", "going",
    "things", "says", "currently", "times", "time", "day", "days",
    "previous", "know", "think", "thing", "come", "came", "way",
    "said", "say", "told", "tell", "really", "actually", "maybe",
    "something", "somebody", "someone", "everything", "nothing",
    "okay", "ok", "yeah", "yea", "right", "well", "kind", "kinda",
    "sort", "oh", "uh", "um", "huh", "mhm", "aha",
    # Luganda conversational fillers
    "oba", "kwe", "era", "naye", "ate", "kyokka", "olwo", "bwe",
    "nze", "ggwe", "gwe", "mbu", "anti", "kuba", "kubanga"
]

# Combined stop word set (lowercased for matching)
ALL_STOP_WORDS = set(
    list(ENGLISH_STOP_WORDS)
    + [w.lower() for w in LUGANDA_STOP_WORDS]
    + [w.lower() for w in CONVERSATIONAL_FILLERS]
)
