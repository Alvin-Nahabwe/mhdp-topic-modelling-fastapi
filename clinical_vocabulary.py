"""
Clinical vocabulary for MHDP symptom classification.

Provides a structured, research-informed vocabulary organized by HiTOP spectra
and symptom classes. Used by both the training pipeline (vectorizer) and
inference API (clinical vocabulary gate).

Sources:
- Bottom-up: Keywords from BERTopic topic representations (data-driven)
- Top-down: Terms from PHQ-9, GAD-7, psychosis screening instruments
- Corpus: Luganda clinical terms extracted from training data

NOTE: This module should be reviewed by a clinician. The vocabulary was
assembled from validated screening instruments and training corpus analysis.
"""

# --- PHQ-9 Depression Screening Keywords ---
# Based on the Patient Health Questionnaire-9 items
PHQ9_KEYWORDS = {
    "interest", "pleasure", "anhedonia",
    "depressed", "hopeless", "sad", "down", "miserable", "sorrow",
    "sleep", "insomnia", "sleeping", "oversleep",
    "tired", "fatigue", "energy", "exhausted", "weak",
    "appetite", "overeating", "eat", "eating", "hungry", "diet",
    "failure", "worthless", "guilt", "guilty", "shame",
    "concentrate", "concentration", "focus", "distracted",
    "slow", "fidgety", "restless", "agitated",
    "dead", "dying", "suicide", "suicidal", "kill", "harm", "hurt",
}

# --- GAD-7 Anxiety Screening Keywords ---
# Based on the Generalized Anxiety Disorder-7 items
GAD7_KEYWORDS = {
    "nervous", "anxious", "anxiety", "edge",
    "worry", "worrying", "worried", "worries",
    "relax", "relaxing",
    "restless", "uneasy",
    "irritable", "annoyed", "angry",
    "afraid", "fear", "scared", "terrified", "panic", "panicking",
    "palpitations", "heartbeat", "heart",
}

# --- Psychosis Screening Keywords ---
# Based on PSQ and clinical psychosis indicators
PSYCHOSIS_KEYWORDS = {
    "voices", "hearing", "seeing", "visions", "hallucination", "hallucinate",
    "paranoid", "persecuted", "watching", "followed",
    "beliefs", "delusion", "delusional",
    "confused", "disorganized", "incoherent", "rambling",
    "violent", "aggressive", "destroy", "fight", "attack",
    "naked", "undress", "wander",
}

# --- Luganda Clinical Terms ---
# Extracted from training corpus and cross-referenced with Luganda medical vocabulary
LUGANDA_CLINICAL = {
    # Sleep (Insomnia/Hypersomnia)
    "tulo", "okwebaka", "sifuna", "ssebaka", "teyebaka", "kubaka",
    "okusumagila", "linebassa",
    # Mood / Depression
    "ennaku", "okwennyama", "nafu", "bunafu", "obuzibu",
    "okwekawa", "yekyaye", "akyawe",
    # Anxiety / Fear
    "okutya", "entiisa", "okweraliikirira", "natiisa",
    # Fatigue / Energy
    "amanyi", "nafu", "omubiri", "obunafu", "mangyi",
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

# --- HiTOP-Organized Clinical Vocabulary ---
# Maps each symptom class to its clinical keywords (English + Luganda)
HITOP_VOCABULARY = {
    "Insomnia/Hypersomnia": {
        "sleep", "insomnia", "sleeping", "sleepy", "sleepless", "awake",
        "dream", "nightmare", "oversleep", "hypersomnia", "restful",
        "tulo", "okwebaka", "sifuna", "ssebaka", "teyebaka", "kubaka",
        "okusumagila", "linebassa",
    },
    "Disorganized behaviors": {
        "violent", "aggressive", "destroy", "fight", "attack", "wander",
        "naked", "undress", "bizarre", "strange",
        "adduka", "olwaana", "okukuba", "okwesulya",
    },
    "Functional impairment": {
        "function", "work", "job", "school", "suspended", "fired",
        "unable", "impaired", "impact", "social", "relationship",
        "emirembe", "obuzibu", "kyimazeko",
    },
    "Suicidal behaviour": {
        "suicide", "suicidal", "kill", "die", "dying", "dead",
        "harm", "hurt", "attempt", "overdose", "poison",
        "okwetta", "kufaa", "kufa", "ngenda", "nzitte",
    },
    "Hallucination/Delusions": {
        "voices", "hearing", "seeing", "visions", "hallucination",
        "paranoid", "persecuted", "delusion", "beliefs",
        "okulaba", "okuwulira", "eddoboozi",
    },
    "Anxiety spectrum": {
        "nervous", "anxious", "anxiety", "worry", "worried", "panic",
        "afraid", "fear", "scared", "terrified", "palpitations",
        "relax", "uneasy", "edge", "heartbeat",
        "okutya", "entiisa", "okweraliikirira", "natiisa",
    },
    "Appetite disturbance": {
        "appetite", "eat", "eating", "hungry", "overeating",
        "diet", "food", "starve", "vomit", "nausea",
        "okulya", "emmere", "ademye", "enjala",
    },
    "Disorganized speech": {
        "talk", "talking", "speech", "incoherent", "rambling",
        "nonsense", "confused", "mumbling",
        "ayogera", "ebigambo", "byesulise",
    },
    "Concentration problems": {
        "concentrate", "concentration", "focus", "attention",
        "distracted", "forgetful", "memory", "confused",
    },
    "Fatigue/Low energy": {
        "tired", "fatigue", "energy", "exhausted", "weak",
        "lethargic", "drained",
        "amanyi", "nafu", "omubiri", "obunafu", "mangyi",
        "tayena", "munafu",
    },
    "Over talkative": {
        "talk", "talking", "talkative", "chatter",
        "fast", "rapid", "pressure",
        "ayogera", "yogeera", "yogera", "bingi", "binjibinji",
    },
    "Low mood/Anhedonia": {
        "sad", "sadness", "depressed", "depression", "hopeless",
        "worthless", "guilt", "guilty", "shame", "crying",
        "pleasure", "interest", "enjoy", "anhedonia", "moody",
        "ennaku", "okwennyama", "okwekawa", "yekyaye", "akyawe",
    },
    "Psychomotor disturbance": {
        "restless", "agitated", "hyperactive", "fidgety",
        "slow", "sluggish", "retardation",
        "energy", "energetic", "excessive", "happy", "happiness",
        "natandika", "kurwana",
    },
    "Isolation/Withdrawal": {
        "isolation", "isolated", "withdraw", "withdrawal", "withdrawn",
        "alone", "lonely", "avoid", "hiding", "passive",
        "yekweka", "yeesibye", "okwekweka",
    },
}

# --- Flat Clinical Vocabulary ---
# Combined set of ALL clinical terms for the vocabulary gate
CLINICAL_VOCABULARY = set()
for terms in HITOP_VOCABULARY.values():
    CLINICAL_VOCABULARY.update(terms)
CLINICAL_VOCABULARY.update(PHQ9_KEYWORDS)
CLINICAL_VOCABULARY.update(GAD7_KEYWORDS)
CLINICAL_VOCABULARY.update(PSYCHOSIS_KEYWORDS)
CLINICAL_VOCABULARY.update(LUGANDA_CLINICAL)


def validate_no_clinical_overlap(clinical_vocab, stop_words):
    """Validate that no clinical vocabulary terms appear in stop words.

    Returns a set of overlapping terms that should be removed from stop words.
    """
    overlap = clinical_vocab & stop_words
    if overlap:
        print(f"WARNING: {len(overlap)} clinical terms found in stop words: {sorted(overlap)}")
        print("These should be removed from stop_words.py to avoid filtering clinical signal.")
    return overlap


def get_class_vocabulary(symptom_class):
    """Get the clinical vocabulary for a specific symptom class."""
    return HITOP_VOCABULARY.get(symptom_class, set())


def get_seed_topics():
    """Get seed topic lists for Guided BERTopic training.

    Returns a list of lists, where each inner list contains seed keywords
    for one symptom class.
    """
    return [list(terms)[:10] for terms in HITOP_VOCABULARY.values()]
