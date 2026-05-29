"""
MHDP Clinical Symptom Inference API (v2)

FastAPI service that accepts call transcripts and returns aggregated
symptom predictions using a semi-supervised BERTopic model trained
on Butabika Hospital mental health call centre data.

v2 changes:
- Pydantic field validation with clinical constraints
- Clinical vocabulary gate using compiled regex (clinical_vocabulary.py)
- Affect risk endpoint for ordinal likelihood scoring
- Input sanitization and length limits
- Structured error responses

Embedding model: Davlan/afro-xlmr-base (cross-lingual English/Luganda)
Topic model: Semi-supervised BERTopic with outlier class for non-symptom speech
"""

import os
import re
import json
import time
import logging
import pandas as pd
from collections import Counter, defaultdict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from stop_words import ALL_STOP_WORDS
from clinical_vocabulary import HITOP_VOCABULARY

ml_models = {}

# --- Structured JSON Logging ---

class JSONFormatter(logging.Formatter):
    """Structured JSON log format for production observability."""
    def format(self, record):
        log_entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Include extra fields if present
        if hasattr(record, "endpoint"):
            log_entry["endpoint"] = record.endpoint
        if hasattr(record, "latency_ms"):
            log_entry["latency_ms"] = record.latency_ms
        if hasattr(record, "status_code"):
            log_entry["status_code"] = record.status_code
        if hasattr(record, "extra_data"):
            log_entry.update(record.extra_data)
        return json.dumps(log_entry)


def _setup_logging():
    logger = logging.getLogger("mhdp")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    # Use JSON format if MHDP_LOG_FORMAT=json (default in production)
    if os.getenv("MHDP_LOG_FORMAT", "json") == "json":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        ))
    logger.addHandler(handler)
    return logger

logger = _setup_logging()

# --- Request Metrics ---

_metrics = {
    "requests_total": 0,
    "requests_by_endpoint": defaultdict(int),
    "latency_sum_ms": 0.0,
    "latency_count": 0,
    "predictions_total": 0,
    "affect_risk_total": 0,
    "errors_total": 0,
}

# Pre-compile clinical vocabulary regex for fast gate matching
# Matches any clinical term (word boundary) from the HITOP vocabulary
_all_clinical_terms = set()
for terms in HITOP_VOCABULARY.values():
    _all_clinical_terms.update(t.lower() for t in terms)

# Sort by length (longest first) for greedy matching
_sorted_terms = sorted(_all_clinical_terms, key=len, reverse=True)
# Escape for regex, join with OR, require word boundaries
_clinical_pattern = re.compile(
    r'\b(?:'
    + '|'.join(re.escape(t) for t in _sorted_terms)
    + r')\b',
    re.IGNORECASE,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load BERTopic model, classifier, and affect risk
    models at startup.
    """
    print("Loading embedding model and BERTopic artifacts...")
    embedder = SentenceTransformer("Davlan/afro-xlmr-base")
    topic_model = BERTopic.load(
        "./bertopic_semi_supervised/model",
        embedding_model=embedder,
    )

    topic_info = pd.read_csv(
        "./bertopic_semi_supervised/topic_info.csv"
    )
    custom_labels = (
        topic_info.set_index('Topic')['Symptom_Label'].to_dict()
    )

    # Build topic keyword lookup (topic_id -> list of representation words)
    topic_keywords = {}
    for _, row in topic_info.iterrows():
        tid = row['Topic']
        rep = row.get('Representation', '[]')
        if isinstance(rep, str):
            words = [
                w.strip().strip("'\"")
                for w in rep.strip("[]").split(",")
                if w.strip().strip("'\"")
            ]
        else:
            words = rep if isinstance(rep, list) else []
        topic_keywords[tid] = words

    ml_models["topic_model"] = topic_model
    ml_models["embedder"] = embedder
    ml_models["custom_labels"] = custom_labels
    ml_models["topic_keywords"] = topic_keywords

    # Build global clinical vocabulary from ALL topics' representation words.
    # Used by the confidence gate to distinguish clinical content from noise.
    clinical_vocab = set()
    for tid, words in topic_keywords.items():
        if tid == -1:
            continue
        for w in words:
            w_lower = w.lower().strip()
            if len(w_lower) >= 3 and w_lower not in ALL_STOP_WORDS:
                clinical_vocab.add(w_lower)
    ml_models["clinical_vocab"] = clinical_vocab
    print(
        f"Clinical vocabulary: {len(clinical_vocab)} "
        f"topic terms + {len(_all_clinical_terms)} "
        f"HiTOP terms."
    )

    # Load classifier model if available (preferred over BERTopic for classification)
    classifier_path = "./classifier_model/classifier.joblib"
    le_path = "./classifier_model/label_encoder.joblib"
    if os.path.exists(classifier_path) and os.path.exists(le_path):
        import joblib
        ml_models["classifier"] = joblib.load(classifier_path)
        ml_models["label_encoder"] = joblib.load(le_path)
        print(
            "Classifier loaded: "
            f"{len(ml_models['label_encoder'].classes_)}"
            " classes."
        )

    # Load affect risk models if available
    affect_risk_dir = "./affect_risk_model"
    if os.path.exists(affect_risk_dir):
        import joblib
        import ordinal_classifier  # noqa: F401 — required for joblib
        for category in ["psychosis", "depression", "anxiety"]:
            model_path = os.path.join(
                affect_risk_dir, category, "model.joblib"
            )
            if os.path.exists(model_path):
                key = f"affect_{category}"
                ml_models[key] = joblib.load(model_path)
                print(f"Affect risk model loaded: {category}")

    print("Inference endpoint ready.")
    yield
    ml_models.clear()


app = FastAPI(
    lifespan=lifespan,
    title="MHDP Clinical Symptoms API",
    description=(
        "Clinical symptom classification and affect risk"
        " assessment for mental health call transcripts."
    ),
    version="2.0.0",
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with latency and status for observability."""
    start_time = time.time()
    response = await call_next(request)
    latency_ms = round((time.time() - start_time) * 1000, 1)

    _metrics["requests_total"] += 1
    _metrics["requests_by_endpoint"][request.url.path] += 1
    _metrics["latency_sum_ms"] += latency_ms
    _metrics["latency_count"] += 1

    if response.status_code >= 400:
        _metrics["errors_total"] += 1

    logger.info(
        f"{request.method} {request.url.path}"
        f" {response.status_code} {latency_ms}ms",
        extra={
            "endpoint": request.url.path,
            "latency_ms": latency_ms,
            "status_code": response.status_code,
        }
    )
    return response


@app.get("/health")
async def health():
    """Health check endpoint for Docker HEALTHCHECK and monitoring."""
    model_loaded = "topic_model" in ml_models
    classifier_loaded = "classifier" in ml_models
    affect_risk_loaded = any(k.startswith("affect_") for k in ml_models)
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded,
        "classifier_loaded": classifier_loaded,
        "affect_risk_loaded": affect_risk_loaded,
        "symptom_classes": (
            len(
                set(
                    ml_models.get(
                        "custom_labels", {}
                    ).values()
                )
                - {None, ""}
            )
            if model_loaded
            else 0
        ),
        "clinical_vocab_size": len(_all_clinical_terms),
    }


@app.get("/metrics")
async def metrics():
    """Operational metrics for monitoring dashboards."""
    avg_latency = (
        round(_metrics["latency_sum_ms"] / _metrics["latency_count"], 1)
        if _metrics["latency_count"] > 0 else 0.0
    )
    return {
        "requests_total": _metrics["requests_total"],
        "requests_by_endpoint": dict(_metrics["requests_by_endpoint"]),
        "avg_latency_ms": avg_latency,
        "predictions_total": _metrics["predictions_total"],
        "affect_risk_total": _metrics["affect_risk_total"],
        "errors_total": _metrics["errors_total"],
    }


# --- Request / Response Schemas ---

class Segment(BaseModel):
    end: float = Field(..., ge=0, description="Segment end time in seconds")
    language: str = Field(
        ...,
        min_length=1,
        max_length=10,
        description="Language code (en, lg)",
    )
    speaker: str = Field(..., min_length=1, max_length=100)
    speaker_role_id: str = Field(..., min_length=1, max_length=50,
                                 description="Role: 'caller' or 'agent'")
    start: float = Field(..., ge=0, description="Segment start time in seconds")
    text: str = Field(..., min_length=1, max_length=5000,
                      description="Transcript text (max 5000 chars)")

    @field_validator('text')
    @classmethod
    def sanitize_text(cls, v):
        # Strip control characters, normalize whitespace
        v = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', v)
        v = re.sub(r'\s+', ' ', v).strip()
        if not v:
            raise ValueError("Text cannot be empty after sanitization")
        return v

    @field_validator('end')
    @classmethod
    def end_must_be_reasonable(cls, v):
        if v > 7200:  # 2 hours max
            raise ValueError("Segment end time exceeds 2 hours (7200s)")
        return v


class Payload(BaseModel):
    speaker_segments: list[Segment] = Field(
        ..., min_length=1, max_length=500,
        description="List of speaker segments (max 500)"
    )


class AffectRiskRequest(BaseModel):
    """Request for affect risk assessment on a full call transcript."""
    transcript: str = Field(..., min_length=10, max_length=50000,
                            description="Full call transcript text")

    @field_validator('transcript')
    @classmethod
    def sanitize_transcript(cls, v):
        v = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', v)
        v = re.sub(r'\s+', ' ', v).strip()
        if len(v) < 10:
            raise ValueError("Transcript too short after sanitization")
        return v


class SymptomSummary(BaseModel):
    symptom_label: str
    symptom_representation: float
    confidence_score: float
    keywords: list[str]


class CallSummary(BaseModel):
    total_transcripts: int
    classified_transcripts: int
    undefined_transcripts: int
    non_clinical_transcripts: int = Field(
        default=0,
        description="Segments with no clinical vocabulary detected"
    )
    classification_rate: float
    model_used: str = "bertopic"
    symptoms: list[SymptomSummary]


class AffectRiskScore(BaseModel):
    category: str = Field(..., description="psychosis, depression, or anxiety")
    score: int = Field(
        ..., ge=0, le=3,
        description="0=Non-Clinical, 1=Unlikely, 2=Possible, 3=Likely"
    )
    label: str = Field(..., description="Human-readable likelihood label")
    probabilities: dict[str, float] = Field(
        ..., description="Probability for each ordinal level"
    )


class AffectRiskResponse(BaseModel):
    transcript_length: int
    is_clinical: bool = Field(
        ...,
        description="Whether the transcript contained clinical vocabulary"
    )
    risk_scores: list[AffectRiskScore]
    clinical_terms_found: list[str] = Field(
        default_factory=list,
        description="Clinical terms detected in the transcript"
    )


# --- Keyword Extraction ---

def extract_conversation_keywords(
    texts: list[str],
    topic_rep_words: list[str],
    top_n: int = 3,
) -> list[str]:
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
        tokens = [
            w for w in text.lower().split()
            if w not in ALL_STOP_WORDS and len(w) > 1
        ]
        word_counts.update(tokens)

    topic_rep_lower = [w.lower() for w in topic_rep_words]

    # Pass 1: exact matches with topic representation words
    matched = []
    for word in topic_rep_lower:
        if word in word_counts:
            matched.append((word, word_counts[word]))

    # Pass 2: substring matches for morphological variants
    # e.g., "amaloboozi" matching "maloboozi" (Luganda prefix variation)
    # Minimum 4 chars to avoid false positives
    for transcript_word, count in word_counts.items():
        already = [m[0] for m in matched]
        if transcript_word in already:
            continue
        if len(transcript_word) < 4:
            continue
        for rep_word in topic_rep_lower:
            if (
                len(rep_word) >= 4
                and (
                    rep_word in transcript_word
                    or transcript_word in rep_word
                )
            ):
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


def has_clinical_content(text: str) -> bool:
    """
    Fast clinical content gate using compiled regex from HiTOP vocabulary.

    Returns True if the text contains any clinical term. This separates
    'is this clinical content?' from 'which specific symptom?' — the model
    may assign the wrong topic, but as long as the text is clinical, the
    assignment is a reasonable attempt.
    """
    return bool(_clinical_pattern.search(text))


# --- Inference Endpoints ---

@app.post("/predict_symptoms", response_model=CallSummary)
async def predict_symptoms(payload: Payload):
    """
    Classify caller transcripts and return an aggregated clinical summary.

    Only transcripts with speaker_role_id='caller' are processed.
    Each transcript is assigned a symptom label or marked as undefined.
    Results are aggregated per symptom with prevalence, confidence, and keywords.

    If the fine-tuned classifier is available, it is used instead of BERTopic
    for more accurate classification (covers all 15 classes).
    """
    caller_texts = [
        seg.text
        for seg in payload.speaker_segments
        if seg.speaker_role_id == 'caller'
    ]

    if not caller_texts:
        return CallSummary(
            total_transcripts=0,
            classified_transcripts=0,
            undefined_transcripts=0,
            classification_rate=0.0,
            symptoms=[]
        )

    # Decide which model to use
    use_classifier = "classifier" in ml_models and "label_encoder" in ml_models

    if use_classifier:
        result = _predict_with_classifier(caller_texts)
    else:
        result = _predict_with_bertopic(caller_texts)

    _metrics["predictions_total"] += 1
    logger.info(
        "Symptom prediction: "
        f"{result.classified_transcripts}/"
        f"{result.total_transcripts} classified",
        extra={"extra_data": {
            "model_used": result.model_used,
            "total_transcripts": result.total_transcripts,
            "classified": result.classified_transcripts,
            "symptoms_found": len(result.symptoms),
        }}
    )
    return result


def _predict_with_classifier(caller_texts: list[str]) -> CallSummary:
    """Classification using fine-tuned LogReg on afro-xlmr-base embeddings."""
    embedder = ml_models["embedder"]
    classifier = ml_models["classifier"]
    label_encoder = ml_models["label_encoder"]

    embeddings = embedder.encode(caller_texts)
    predicted_labels = label_encoder.inverse_transform(
        classifier.predict(embeddings)
    )
    probabilities = classifier.predict_proba(embeddings)

    symptom_groups = defaultdict(lambda: {"texts": [], "confidences": []})
    non_clinical_count = 0
    total = len(caller_texts)

    for text, label, proba in zip(caller_texts, predicted_labels, probabilities):
        max_prob = float(proba.max())

        # Clinical vocabulary gate: skip non-clinical content
        if not has_clinical_content(text):
            non_clinical_count += 1
            continue

        # Non-Clinical class is an actual label from the classifier
        if label == "Non-Clinical":
            non_clinical_count += 1
            continue

        symptom_groups[label]["texts"].append(text)
        symptom_groups[label]["confidences"].append(max_prob)

    classified_count = total - non_clinical_count
    symptoms = []

    for label, group in symptom_groups.items():
        count = len(group["texts"])
        avg_confidence = sum(group["confidences"]) / count
        representation = round((count / total) * 100, 1)

        # Use clinical vocabulary terms as keywords
        keywords = _extract_clinical_keywords(group["texts"], top_n=3)

        symptoms.append(SymptomSummary(
            symptom_label=label,
            symptom_representation=representation,
            confidence_score=round(avg_confidence, 4),
            keywords=keywords
        ))

    symptoms.sort(key=lambda s: s.symptom_representation, reverse=True)

    return CallSummary(
        total_transcripts=total,
        classified_transcripts=classified_count,
        undefined_transcripts=0,
        non_clinical_transcripts=non_clinical_count,
        classification_rate=(
            round((classified_count / total) * 100, 1)
            if total > 0
            else 0.0
        ),
        model_used="classifier",
        symptoms=symptoms
    )


def _extract_clinical_keywords(texts: list[str], top_n: int = 3) -> list[str]:
    """Extract clinical vocabulary terms found in the texts."""
    found_terms = Counter()
    for text in texts:
        matches = _clinical_pattern.findall(text.lower())
        found_terms.update(matches)
    return [term for term, _ in found_terms.most_common(top_n)]


def _predict_with_bertopic(caller_texts: list[str]) -> CallSummary:
    """Original BERTopic-based classification (fallback)."""
    topic_model = ml_models["topic_model"]
    custom_labels = ml_models["custom_labels"]
    topic_keywords_map = ml_models["topic_keywords"]
    clinical_vocab = ml_models["clinical_vocab"]

    predicted_topics, probabilities = topic_model.transform(caller_texts)

    symptom_groups = defaultdict(
        lambda: {
            "texts": [],
            "confidences": [],
            "topic_ids": [],
        }
    )
    undefined_count = 0
    total = len(caller_texts)

    for text, topic_id, prob in zip(caller_texts, predicted_topics, probabilities):
        label = custom_labels.get(topic_id)

        # Unmapped topics = no symptom detected
        if label is None or (isinstance(label, float) and pd.isna(label)):
            undefined_count += 1
            continue

        # Clinical vocabulary gate (compiled regex — fast)
        if not has_clinical_content(text):
            # Fallback: check against topic representation words
            transcript_tokens = (
                set(text.lower().split()) - ALL_STOP_WORDS
            )
            has_topic_term = False
            for token in transcript_tokens:
                if len(token) < 2:
                    continue
                if token in clinical_vocab:
                    has_topic_term = True
                    break
                if len(token) >= 4:
                    for vocab_word in clinical_vocab:
                        if (
                            len(vocab_word) >= 4
                            and (
                                vocab_word in token
                                or token in vocab_word
                            )
                        ):
                            has_topic_term = True
                            break
                if has_topic_term:
                    break

            if not has_topic_term:
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
        classification_rate=(
            round((classified_count / total) * 100, 1)
            if total > 0
            else 0.0
        ),
        model_used="bertopic",
        symptoms=symptoms
    )


# --- Affect Risk Endpoint ---

# Minimum max-class probability required to trust the model prediction.
# Below this threshold, predictions default to Unlikely (score=1).
AFFECT_CONFIDENCE_THRESHOLD = 0.6


@app.post("/predict_affect_risk", response_model=AffectRiskResponse)
async def predict_affect_risk(request: AffectRiskRequest):
    """
    Predict ordinal likelihood scores for psychosis, depression, and anxiety.

    Returns a score of 0 (Non-Clinical), 1 (Unlikely), 2 (Possible), or
    3 (Likely) for each category, along with the probability distribution
    across all levels.

    Safety guardrails:
    - Clinical gate: transcripts without clinical vocabulary return
      score=0 / label="Non-Clinical" for all categories.
    - Confidence threshold: if the model's max probability is below 0.6,
      the prediction defaults to score=1 / label="Unlikely".
    """
    # Check if any affect risk models are loaded
    loaded_categories = [
        k.replace("affect_", "")
        for k in ml_models
        if k.startswith("affect_")
    ]
    if not loaded_categories:
        raise HTTPException(
            status_code=503,
            detail=(
                "Affect risk models not loaded."
                " Run train_affect_risk.py first."
            )
        )

    # Detect clinical terms in the transcript
    clinical_matches = _clinical_pattern.findall(
        request.transcript.lower()
    )
    # dedupe, cap at 20
    unique_terms = list(
        dict.fromkeys(clinical_matches)
    )[:20]

    is_clinical = len(unique_terms) > 0

    # Clinical gate: non-clinical text gets score=0 for all categories
    if not is_clinical:
        zero_probs = {"Unlikely": 0.0, "Possible": 0.0, "Likely": 0.0}
        risk_scores = [
            AffectRiskScore(
                category=cat,
                score=0,
                label="Non-Clinical",
                probabilities=zero_probs
            )
            for cat in ["psychosis", "depression", "anxiety"]
            if f"affect_{cat}" in ml_models
        ]

        _metrics["affect_risk_total"] += 1
        logger.info(
            "Affect risk assessment: non-clinical transcript",
            extra={"extra_data": {
                "is_clinical": False,
                "transcript_length": len(request.transcript),
            }}
        )

        return AffectRiskResponse(
            transcript_length=len(request.transcript),
            is_clinical=False,
            risk_scores=risk_scores,
            clinical_terms_found=[]
        )

    # Clinical transcript — run model inference
    embedder = ml_models["embedder"]
    embedding = embedder.encode([request.transcript])

    level_labels = {1: "Unlikely", 2: "Possible", 3: "Likely"}

    risk_scores = []
    for category in ["psychosis", "depression", "anxiety"]:
        model_key = f"affect_{category}"
        if model_key not in ml_models:
            continue

        model = ml_models[model_key]
        pred = model.predict(embedding)[0]
        proba = model.predict_proba(embedding)[0]

        max_prob = float(proba.max())

        # Confidence threshold: uncertain predictions default to Unlikely
        if max_prob < AFFECT_CONFIDENCE_THRESHOLD:
            final_score = 1
            final_label = "Unlikely"
        else:
            final_score = int(pred)
            final_label = level_labels.get(int(pred), "Unknown")

        risk_scores.append(AffectRiskScore(
            category=category,
            score=final_score,
            label=final_label,
            probabilities={
                level_labels[i+1]: round(float(p), 4)
                for i, p in enumerate(proba)
            }
        ))

    _metrics["affect_risk_total"] += 1
    logger.info(
        f"Affect risk assessment: {len(risk_scores)} categories scored",
        extra={"extra_data": {
            "is_clinical": True,
            "categories_scored": len(risk_scores),
            "scores": {r.category: r.score for r in risk_scores},
            "clinical_terms_count": len(unique_terms),
        }}
    )

    return AffectRiskResponse(
        transcript_length=len(request.transcript),
        is_clinical=True,
        risk_scores=risk_scores,
        clinical_terms_found=unique_terms
    )
