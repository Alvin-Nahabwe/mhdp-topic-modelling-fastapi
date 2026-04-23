"""
MHDP Clinical Symptom Inference API

FastAPI service that accepts call transcripts and returns aggregated
symptom predictions using a semi-supervised BERTopic model trained
on Butabika Hospital mental health call centre data.

Embedding model: Davlan/afro-xlmr-base (cross-lingual English/Luganda)
Topic model: Semi-supervised BERTopic with outlier class for non-symptom speech
"""

import pandas as pd
from collections import Counter, defaultdict
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from stop_words import ALL_STOP_WORDS

ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load BERTopic model and symptom mappings at startup."""
    print("Loading embedding model and BERTopic artifacts...")
    embedder = SentenceTransformer("Davlan/afro-xlmr-base")
    topic_model = BERTopic.load("./bertopic_semi_supervised/model", embedding_model=embedder)

    topic_info = pd.read_csv("./bertopic_semi_supervised/topic_info.csv")
    custom_labels = topic_info.set_index('Topic')['Symptom_Label'].to_dict()

    # Build topic keyword lookup (topic_id -> list of representation words)
    topic_keywords = {}
    for _, row in topic_info.iterrows():
        tid = row['Topic']
        rep = row.get('Representation', '[]')
        if isinstance(rep, str):
            words = [w.strip().strip("'\"") for w in rep.strip("[]").split(",") if w.strip().strip("'\"")]
        else:
            words = rep if isinstance(rep, list) else []
        topic_keywords[tid] = words

    ml_models["topic_model"] = topic_model
    ml_models["custom_labels"] = custom_labels
    ml_models["topic_keywords"] = topic_keywords

    print("Inference endpoint ready.")
    yield
    ml_models.clear()


app = FastAPI(lifespan=lifespan, title="MHDP Clinical Symptoms API")


# --- Request / Response Schemas ---

class Segment(BaseModel):
    end: float
    language: str
    speaker: str
    speaker_role_id: str
    start: float
    text: str


class Payload(BaseModel):
    speaker_segments: list[Segment]


class SymptomSummary(BaseModel):
    symptom_label: str
    symptom_representation: float
    confidence_score: float
    keywords: list[str]


class CallSummary(BaseModel):
    total_transcripts: int
    classified_transcripts: int
    undefined_transcripts: int
    classification_rate: float
    symptoms: list[SymptomSummary]


# --- Keyword Extraction ---

def extract_conversation_keywords(texts: list[str], topic_rep_words: list[str], top_n: int = 3) -> list[str]:
    """
    Extract keywords from actual conversation transcripts that match the
    topic representation words. Returns conversation-specific evidence
    rather than static model keywords.

    Matching strategy:
        1. Exact match against topic representation words
        2. Substring match for Luganda morphological variants
        3. Fallback to most frequent content words from the transcripts
    """
    # Tokenize and count, excluding stop words upfront
    word_counts = Counter()
    for text in texts:
        tokens = [w for w in text.lower().split() if w not in ALL_STOP_WORDS and len(w) > 1]
        word_counts.update(tokens)

    topic_rep_lower = [w.lower() for w in topic_rep_words]

    # Pass 1: exact matches with topic representation words
    matched = []
    for word in topic_rep_lower:
        if word in word_counts:
            matched.append((word, word_counts[word]))

    # Pass 2: substring matches for morphological variants
    # e.g., "amaloboozi" matching "maloboozi" (Luganda prefix variation)
    for transcript_word, count in word_counts.items():
        if transcript_word in [m[0] for m in matched]:
            continue
        for rep_word in topic_rep_lower:
            if rep_word in transcript_word or transcript_word in rep_word:
                matched.append((transcript_word, count))
                break

    # Rank by frequency, take top_n
    matched.sort(key=lambda x: x[1], reverse=True)
    result = [w for w, _ in matched[:top_n]]

    # Fallback: most frequent content words (stop words already excluded)
    if len(result) < top_n:
        for word, count in word_counts.most_common():
            if word not in result and len(word) > 2:
                result.append(word)
                if len(result) >= top_n:
                    break

    return result


# --- Inference Endpoint ---

@app.post("/predict_symptoms", response_model=CallSummary)
async def predict_symptoms(payload: Payload):
    """
    Classify caller transcripts and return an aggregated clinical summary.

    Only transcripts with speaker_role_id='caller' are processed.
    Each transcript is assigned a symptom label or marked as undefined.
    Results are aggregated per symptom with prevalence, confidence, and keywords.
    """
    caller_texts = [seg.text for seg in payload.speaker_segments if seg.speaker_role_id == 'caller']

    if not caller_texts:
        return CallSummary(
            total_transcripts=0,
            classified_transcripts=0,
            undefined_transcripts=0,
            classification_rate=0.0,
            symptoms=[]
        )

    topic_model = ml_models["topic_model"]
    custom_labels = ml_models["custom_labels"]
    topic_keywords_map = ml_models["topic_keywords"]

    predicted_topics, probabilities = topic_model.transform(caller_texts)

    # Group transcripts by symptom label
    symptom_groups = defaultdict(lambda: {"texts": [], "confidences": [], "topic_ids": []})
    undefined_count = 0
    total = len(caller_texts)

    for text, topic_id, prob in zip(caller_texts, predicted_topics, probabilities):
        label = custom_labels.get(topic_id)

        # Topic -1 or unmapped topics = no symptom detected
        if topic_id == -1 or label is None or (isinstance(label, float) and pd.isna(label)):
            undefined_count += 1
            continue

        symptom_groups[label]["texts"].append(text)
        symptom_groups[label]["confidences"].append(float(prob))
        symptom_groups[label]["topic_ids"].append(topic_id)

    # Build aggregated symptom summaries
    classified_count = total - undefined_count
    symptoms = []

    for label, group in symptom_groups.items():
        count = len(group["texts"])
        avg_confidence = sum(group["confidences"]) / count
        representation = round((count / total) * 100, 1)

        # Collect topic representation words across all topics for this symptom
        all_rep_words = []
        for tid in set(group["topic_ids"]):
            all_rep_words.extend(topic_keywords_map.get(tid, []))

        keywords = extract_conversation_keywords(group["texts"], all_rep_words, top_n=3)

        symptoms.append(SymptomSummary(
            symptom_label=label,
            symptom_representation=representation,
            confidence_score=round(avg_confidence, 4),
            keywords=keywords
        ))

    # Most prevalent symptom first
    symptoms.sort(key=lambda s: s.symptom_representation, reverse=True)

    return CallSummary(
        total_transcripts=total,
        classified_transcripts=classified_count,
        undefined_transcripts=undefined_count,
        classification_rate=round((classified_count / total) * 100, 1) if total > 0 else 0.0,
        symptoms=symptoms
    )
