"""
Shared stop words and filler words for the MHDP clinical topic pipeline.
Used in both training (vectorizer) and inference (keyword extraction).

AUDIT NOTES (v2):
- "feel" REMOVED from fillers: appears 65x in clinical transcripts
  ("I feel tired", "I feel nervous", "I feel like dying")
- "things" REMOVED: appears in clinical context ("seeing things", "hearing things")
- "lot" REMOVED: appears in "talk a lot", "worry a lot"
- "going" REMOVED: appears in "going to die", "going crazy"
- Added: "hospital", "butabika", "kaale" (high non-clinical frequency)
- Validated: no overlap with clinical_vocabulary.py terms
"""

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Luganda function words (pronouns, conjunctions, prepositions, etc.)
# These carry grammatical rather than clinical meaning.
LUGANDA_STOP_WORDS = [
    "nga", "nti", "kale", "bulungi", "aah", "eeh", "mmh", "hmm",
    "ssebo", "nnyabo", "wangi", "yee", "nedda", "ddala", "nnyo", "nze", "ggwe",
    "gwe", "ye", "ffe", "mmwe", "mwe", "bo", "wange", "kino", "ekyo", "bino",
    "ebyo", "ne", "mu", "ku", "wa", "nti", "bwe", "eri", "okuva", "paka", "ko",
    "ki", "kiki", "ani", "ddi", "lwaki", "otya", "mutya", "ndi", "oli", "ali",
    "tuli", "bali", "nina", "alina"
]

# Conversational fillers and domain-specific noise.
# IMPORTANT: Do NOT add words that carry clinical meaning.
# See clinical_vocabulary.py for the clinical term registry.
CONVERSATIONAL_FILLERS = [
    "like", "just", "hmm", "eeh", "eh", "ah", "don", "t", "s",
    "kati", "awo", "waliwo", "na", "kuba", "umm", "mmh", "mpozi",
    # General conversational words (verified non-clinical)
    "dont", "yes", "got", "went",
    "says", "currently", "times", "time", "day", "days",
    "previous", "know", "think", "come", "came", "way",
    "said", "say", "told", "tell", "really", "actually", "maybe",
    "somebody", "someone", "everything", "nothing",
    "okay", "ok", "yeah", "yea", "right", "well", "kind", "kinda",
    "sort", "oh", "uh", "um", "huh", "mhm", "aha",
    # Luganda conversational fillers
    "oba", "kwe", "era", "naye", "ate", "kyokka", "olwo", "bwe",
    "nze", "ggwe", "gwe", "mbu", "anti", "kuba", "kubanga",
    # Non-clinical domain words (high frequency in agent/logistics speech)
    "hospital", "butabika", "kaale",
]

# --- REMOVED from fillers (v2 audit) ---
# "feel"       → clinical: "I feel tired/nervous/like dying" (65x in clinical text)
# "things"     → clinical: "seeing things", "hearing things" (hallucination context)
# "lot"        → clinical: "talk a lot", "worry a lot" (symptom severity)
# "alot"       → same as "lot"
# "going"      → clinical: "going to die", "going crazy" (suicidal/psychosis)
# "something"  → clinical: "something tells me", "seeing something"
# "thing"      → same reasoning as "things"

# Words from ENGLISH_STOP_WORDS that are clinically significant and must be preserved.
# These appear in validated screening instruments (PHQ-9, GAD-7):
#   "alone"    → isolation/withdrawal indicator
#   "down"     → "feeling down" is PHQ-9 item 1
#   "interest" → "loss of interest" is PHQ-9 item 1
CLINICAL_EXCEPTIONS = {"alone", "down", "interest"}

# Combined stop word set (lowercased for matching)
ALL_STOP_WORDS = (
    set(list(ENGLISH_STOP_WORDS))
    | set(w.lower() for w in LUGANDA_STOP_WORDS)
    | set(w.lower() for w in CONVERSATIONAL_FILLERS)
) - CLINICAL_EXCEPTIONS
