"""
Clinical vocabulary for MHDP symptom classification.

Provides a structured, research-informed vocabulary organized by HiTOP spectra
and symptom classes. Used by both the training pipeline (vectorizer) and
inference API (clinical vocabulary gate).

Architecture (Three-Tier):
  Tier 1 — Safe Unigrams: Psychiatric/medical terms that are unambiguously
           clinical in isolation (e.g., "suicidal", "hallucination").
  Tier 2 — Required Phrases: Common English words that MUST appear in a
           clinical context (e.g., "can't sleep", "hearing voices").
  Luganda terms are classified as safe unigrams because they are
  domain-specific and do not false-positive in English transcripts.

The clinical gate requires >= 2 matches to classify text as clinical,
preventing single-word false positives from casual conversation.

Sources:
- PHQ-9 (Patient Health Questionnaire-9) item wordings
- GAD-7 (Generalized Anxiety Disorder-7) item wordings
- PSQ (Psychosis Screening Questionnaire) probe wordings
- Bottom-up: Keywords from BERTopic topic representations (data-driven)
- Corpus: Luganda clinical terms extracted from training data

NOTE: This module should be reviewed by a clinician. The vocabulary was
assembled from validated screening instruments and training corpus analysis.
"""

import re

# ===================================================================
# TIER 1: SAFE UNIGRAMS
# Psychiatric/medical terms that are "without a doubt" clinical in
# isolation. These have no common non-clinical usage in conversational
# transcripts.
# ===================================================================

SAFE_CLINICAL_UNIGRAMS = {
    # --- Psychosis ---
    "hallucination", "hallucinate", "hallucinations",
    "delusion", "delusional", "delusions",
    "paranoid", "paranoia",
    "psychosis", "psychotic",
    "persecuted", "persecution",
    "incoherent",

    # --- Depression (clinical terms) ---
    "depressed", "depression",
    "hopeless", "hopelessness",
    "worthless", "worthlessness",
    "anhedonia",
    "suicidal", "suicide",

    # --- Anxiety (clinical terms) ---
    "anxiety", "anxious",
    "palpitations",

    # --- Sleep disorders (clinical terms) ---
    "insomnia",
    "hypersomnia",
    "nightmare", "nightmares",

    # --- Behavioral/motor (clinical terms) ---
    "lethargic", "lethargy",
    "agitated", "agitation",
    "hyperactive",
    "fidgety",

    # --- Self-harm (clinical terms) ---
    "overdose",
    "self-harm",

    # --- Additional clinical terms ---
    "sleepless",
    "overeating",
    "nausea",
    "forgetful",
    "mumbling",
    "isolating",
    "withdrawn",
    "withdrawing",

    # --- Luganda Clinical Terms ---
    # Sleep (Insomnia/Hypersomnia)
    "tulo", "okwebaka", "sifuna", "ssebaka",
    "teyebaka", "kubaka", "okusumagila", "linebassa",
    # Mood / Depression
    "ennaku", "okwennyama", "nafu", "bunafu", "obuzibu",
    "okwekawa", "yekyaye", "akyawe",
    # Anxiety / Fear
    "okutya", "entiisa", "okweraliikirira", "natiisa",
    # Fatigue / Energy
    "amanyi", "omubiri", "obunafu", "mangyi",
    "tayena", "munafu",
    # Appetite
    "okulya", "emmere", "ademye", "enjala",
    # Suicide
    "okwetta", "kufaa", "kufa", "ngenda",
    # Speech / Behavior
    "ayogera", "yogeera", "yogera", "bingi", "binjibinji",
    "ebigambo", "byesulise",
    # Violence / Disorganization
    "adduka", "olwaana", "okukuba", "okwesulya",
    "natandika", "kurwana",
    # Hallucination
    "okulaba", "okuwulira", "eddoboozi",
    # Social withdrawal
    "yekweka", "yeesibye", "okwekweka",
}


# ===================================================================
# TIER 2: CLINICAL PHRASES
# Common words that require clinical context to be meaningful.
# Organized by HiTOP symptom class. Each entry is a phrase (bigram
# or trigram) that captures the clinical intent.
# ===================================================================

CLINICAL_PHRASES = {
    "Insomnia/Hypersomnia": [
        # PHQ-9 #3: "Trouble falling or staying asleep, or sleeping
        #            too much"
        "can't sleep", "cannot sleep",
        "trouble sleeping", "difficulty sleeping",
        "not sleeping", "poor sleep", "no sleep",
        "sleep too much", "sleeping too much",
        "sleepless nights",
        "keeps waking", "waking up at night",
        "staying awake", "can't stay asleep",
        "bad dreams", "disturbing dreams",
        # Luganda
        "teyebaka kiro", "tafuna tulo",
    ],
    "Low mood/Anhedonia": [
        # PHQ-9 #1: "Little interest or pleasure in doing things"
        # PHQ-9 #2: "Feeling down, depressed, or hopeless"
        "lost interest", "no interest", "lack of interest",
        "no pleasure", "lost pleasure",
        "don't enjoy", "doesn't enjoy",
        "feeling down", "feeling low", "feeling sad",
        "feeling empty", "feeling numb",
        "no motivation", "lost motivation",
        "crying all the time", "always crying",
        "want to cry", "feels like crying",
        "lost meaning", "life is meaningless",
    ],
    "Fatigue/Low energy": [
        # PHQ-9 #4: "Feeling tired or having little energy"
        "feeling tired", "always tired",
        "no energy", "low energy", "little energy",
        "lack of energy", "lost energy",
        "feeling weak", "feeling exhausted",
        "feeling drained", "so tired",
        "can't get up", "too tired",
    ],
    "Appetite disturbance": [
        # PHQ-9 #5: "Poor appetite or overeating"
        "poor appetite", "lost appetite", "no appetite",
        "not eating", "can't eat", "won't eat",
        "refusing food", "stopped eating",
        "eating too much",
        "lost weight", "gaining weight",
        "feeling nauseous",
    ],
    "Concentration problems": [
        # PHQ-9 #7: "Trouble concentrating on things"
        "can't focus", "cannot focus",
        "can't concentrate", "cannot concentrate",
        "trouble concentrating", "difficulty focusing",
        "hard to concentrate", "difficulty concentrating",
        "can't think",
        "losing memory", "memory problems",
    ],
    "Psychomotor disturbance": [
        # PHQ-9 #8: "Moving or speaking so slowly" /
        #           "being so fidgety or restless"
        "moving slowly", "speaking slowly",
        "slowed down", "feeling slow",
        "very restless", "so restless",
        "can't sit still", "pacing around",
        "feeling restless",
        "too much energy", "excessive energy",
        "unusually happy", "too happy", "overly happy",
        "excessively happy",
    ],
    "Suicidal behaviour": [
        # PHQ-9 #9: "Thoughts that you would be better off dead,
        #            or of hurting yourself"
        "kill myself", "kill self", "kill himself",
        "kill herself",
        "want to die", "wants to die", "wanted to die",
        "better off dead", "wish I was dead",
        "thoughts of dying", "thinking of dying",
        "hurting myself", "hurting himself",
        "hurting herself",
        "harming myself", "harming himself",
        "harming herself",
        "self harm",
        "attempted suicide", "suicide attempt",
        "took poison", "drank poison",
        "cutting myself", "cutting herself",
    ],
    "Anxiety spectrum": [
        # GAD-7 #1: "Feeling nervous, anxious, or on edge"
        # GAD-7 #2: "Not being able to stop worrying"
        # GAD-7 #4: "Trouble relaxing"
        # GAD-7 #7: "Feeling afraid as if something awful
        #            might happen"
        "feeling nervous", "very nervous",
        "feeling anxious", "very anxious",
        "on edge", "feeling on edge",
        "can't stop worrying", "worrying too much",
        "always worrying", "constantly worried",
        "excessive worry",
        "trouble relaxing", "can't relax", "unable to relax",
        "feeling afraid", "very afraid",
        "feeling scared", "very scared",
        "feeling terrified",
        "panic attack", "panic attacks",
        "heart racing", "heart pounding",
        "rapid heartbeat",
        "something bad will happen",
        "something awful",
    ],
    "Hallucination/Delusions": [
        # PSQ: "Have you heard or saw things that other people
        #       couldn't?"
        "hearing voices", "hears voices",
        "hearing things", "hearing sounds",
        "seeing things", "sees things",
        "seeing visions", "seeing people who aren't there",
        "people watching me", "people following me",
        "people against me", "out to get me",
        "plotting against me",
        "thoughts being controlled",
        "voices telling me", "voices told me",
    ],
    "Disorganized speech": [
        "talking nonsense", "making no sense",
        "incoherent speech", "rambling speech",
        "can't understand what he says",
        "can't understand what she says",
        "talking to self", "talks to self",
        "talks to himself", "talks to herself",
        "speaking incoherently",
    ],
    "Over talkative": [
        "talking too much", "talks too much",
        "can't stop talking", "won't stop talking",
        "pressured speech", "talking fast",
        "rapid speech", "racing thoughts",
        "speaks very fast", "talking nonstop",
    ],
    "Disorganized behaviors": [
        "wants to fight", "fighting people",
        "getting into fights", "violent behavior",
        "aggressive behavior", "acting violent",
        "acting aggressive", "attacking people",
        "destroying things", "breaking things",
        "taking off clothes", "removing clothes",
        "walking naked", "wandering around",
        "acting strange", "strange behavior",
        "bizarre behavior",
    ],
    "Functional impairment": [
        # PHQ-9 functional item: "How difficult have these
        # problems made it for you to do your work"
        "can't work", "cannot work", "unable to work",
        "stopped working", "lost job", "lost his job",
        "lost her job", "fired from",
        "dropped out of school", "can't go to school",
        "suspended from school",
        "can't take care", "unable to function",
        "can't do anything",
        "relationships suffering", "relationship problems",
        "can't socialize", "avoiding people",
        "can't leave the house",
    ],
    "Isolation/Withdrawal": [
        "wants to be alone", "always alone",
        "staying alone",
        "avoiding people", "avoids everyone",
        "won't come out", "locked in room",
        "hiding from people", "refuses to go out",
        "doesn't talk to anyone",
        "stopped socializing",
    ],
}


# ===================================================================
# HITOP VOCABULARY (for BERTopic seed topics and topic labeling)
# Contains clinically meaningful terms per symptom class.
# Excludes high-risk common words. Uses safe unigrams + key
# discriminative words extracted from clinical phrases.
# ===================================================================

HITOP_VOCABULARY = {
    "Insomnia/Hypersomnia": {
        "insomnia", "hypersomnia", "sleepless", "nightmare",
        "nightmares", "awake",
        "tulo", "okwebaka", "sifuna", "ssebaka",
        "teyebaka", "kubaka", "okusumagila", "linebassa",
    },
    "Disorganized behaviors": {
        "violent", "aggressive", "bizarre",
        "adduka", "olwaana", "okukuba", "okwesulya",
    },
    "Functional impairment": {
        "impaired", "unable", "suspended",
        "emirembe", "obuzibu", "kyimazeko",
    },
    "Suicidal behaviour": {
        "suicide", "suicidal", "overdose", "self-harm",
        "okwetta", "kufaa", "kufa", "ngenda", "nzitte",
    },
    "Hallucination/Delusions": {
        "hallucination", "hallucinations", "delusion",
        "delusional", "paranoid", "persecuted",
        "okulaba", "okuwulira", "eddoboozi",
    },
    "Anxiety spectrum": {
        "anxiety", "anxious", "palpitations",
        "okutya", "entiisa", "okweraliikirira", "natiisa",
    },
    "Appetite disturbance": {
        "appetite", "overeating", "nausea",
        "okulya", "emmere", "ademye", "enjala",
    },
    "Disorganized speech": {
        "incoherent", "rambling", "mumbling",
        "ayogera", "ebigambo", "byesulise",
    },
    "Concentration problems": {
        "forgetful",
    },
    "Fatigue/Low energy": {
        "lethargic", "lethargy", "exhausted",
        "amanyi", "nafu", "omubiri", "obunafu", "mangyi",
        "tayena", "munafu",
    },
    "Over talkative": {
        "talkative",
        "ayogera", "yogeera", "yogera", "bingi", "binjibinji",
    },
    "Low mood/Anhedonia": {
        "depressed", "depression", "hopeless",
        "worthless", "anhedonia",
        "ennaku", "okwennyama", "okwekawa",
        "yekyaye", "akyawe",
    },
    "Psychomotor disturbance": {
        "agitated", "hyperactive", "fidgety",
        "lethargic", "lethargy",
        "natandika", "kurwana",
    },
    "Isolation/Withdrawal": {
        "withdrawn", "isolating", "withdrawal",
        "yekweka", "yeesibye", "okwekweka",
    },
}


# ===================================================================
# REGEX PATTERN BUILDER
# Constructs a single compiled regex that matches BOTH safe unigrams
# and clinical phrases. Used by the API clinical gate.
# ===================================================================

def build_clinical_pattern():
    """Build a compiled regex matching safe unigrams and clinical
    phrases.

    Returns a compiled regex. Matches are case-insensitive. Unigrams
    use word-boundary anchors; phrases use word-boundary on the outer
    edges only.

    The gate requires >= 2 distinct matches to classify text as
    clinical.
    """
    # Collect all phrases from CLINICAL_PHRASES
    all_phrases = []
    for phrases in CLINICAL_PHRASES.values():
        all_phrases.extend(p.lower() for p in phrases)

    # Combine safe unigrams and phrases
    all_patterns = list(SAFE_CLINICAL_UNIGRAMS) + all_phrases

    # Sort by length (longest first) for greedy matching
    all_patterns.sort(key=len, reverse=True)

    # Escape for regex, join with OR, require word boundaries
    pattern = re.compile(
        r'\b(?:'
        + '|'.join(re.escape(t) for t in all_patterns)
        + r')\b',
        re.IGNORECASE,
    )
    return pattern


# Pre-built pattern for import convenience
CLINICAL_PATTERN = build_clinical_pattern()

# Minimum number of distinct clinical matches required to classify
# text as clinical. Prevents single-word false positives.
MIN_CLINICAL_MATCHES = 2


def has_clinical_content(text, min_matches=MIN_CLINICAL_MATCHES):
    """Check whether text contains sufficient clinical vocabulary.

    Args:
        text: Input text to check.
        min_matches: Minimum number of distinct clinical term/phrase
            matches required. Defaults to MIN_CLINICAL_MATCHES (2).

    Returns:
        True if text contains >= min_matches clinical terms/phrases.
    """
    matches = CLINICAL_PATTERN.findall(text.lower())
    # Deduplicate (same term appearing twice counts as 1)
    unique_matches = set(matches)
    return len(unique_matches) >= min_matches


def extract_clinical_terms(text):
    """Extract all clinical terms/phrases found in text.

    Returns a deduplicated list, capped at 20 terms.
    """
    matches = CLINICAL_PATTERN.findall(text.lower())
    unique = list(dict.fromkeys(matches))[:20]
    return unique


# ===================================================================
# UTILITY FUNCTIONS
# ===================================================================

def validate_no_clinical_overlap(stop_words):
    """Validate that no safe clinical unigrams appear in stop words.

    Returns a set of overlapping terms that should be removed from
    stop words.
    """
    overlap = SAFE_CLINICAL_UNIGRAMS & stop_words
    if overlap:
        print(
            f"WARNING: {len(overlap)} clinical terms"
            f" found in stop words: {sorted(overlap)}"
        )
        print(
            "These should be removed from"
            " stop_words.py to avoid filtering"
            " clinical signal."
        )
    return overlap


def get_class_vocabulary(symptom_class):
    """Get the clinical vocabulary for a specific symptom class."""
    return HITOP_VOCABULARY.get(symptom_class, set())


def get_seed_topics():
    """Get seed topic lists for Guided BERTopic training.

    Returns a list of lists, where each inner list contains
    clinically meaningful seed keywords for one symptom class.
    Only includes terms that are unambiguously clinical.
    """
    return [list(terms)[:10] for terms in HITOP_VOCABULARY.values()]
