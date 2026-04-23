import pandas as pd
from collections import Counter, defaultdict
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from stop_words import ALL_STOP_WORDS

# Define a global dictionary to hold our models securely
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing Lifespan: Loading specialized NLP Architecture...")
    # 1. Fetch our state-of-the-art African Dialect Model
    embedder = SentenceTransformer("Davlan/afro-xlmr-base")
    
    # 2. Load the lightweight semi-supervised BERTopic architecture
    topic_model = BERTopic.load("./bertopic_semi_supervised/model", embedding_model=embedder)
    
    # 3. Load symptom mapping (topic_id -> symptom_label)
    topic_info = pd.read_csv("./bertopic_semi_supervised/topic_info.csv")
    custom_labels = topic_info.set_index('Topic')['Symptom_Label'].to_dict()
    
    # 4. Build topic keyword lookup (topic_id -> list of representation words)
    topic_keywords = {}
    for _, row in topic_info.iterrows():
        tid = row['Topic']
        rep = row.get('Representation', '[]')
        if isinstance(rep, str):
            # Parse the string representation of the list
            words = [w.strip().strip("'\"") for w in rep.strip("[]").split(",") if w.strip().strip("'\"")]
        else:
            words = rep if isinstance(rep, list) else []
        topic_keywords[tid] = words
    
    ml_models["topic_model"] = topic_model
    ml_models["custom_labels"] = custom_labels
    ml_models["topic_keywords"] = topic_keywords
    
    print("Models successfully deployed. Inference Endpoints Ready.")
    yield
    # Clean up on shutdown
    ml_models.clear()

app = FastAPI(lifespan=lifespan, title="MHDP Clinical Symptoms API")

# --- Pydantic Data Validation Schemas ---

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

# --- Keyword Extraction Logic ---

def extract_conversation_keywords(texts: list[str], topic_rep_words: list[str], top_n: int = 3) -> list[str]:
    """
    Extract keywords from the actual conversation transcripts that match or
    are present in the topic representation words. This provides clinician-relevant
    evidence from the specific call rather than generic model keywords.
    """
    # Tokenize all texts and count word frequencies (excluding stop words)
    word_counts = Counter()
    for text in texts:
        tokens = [w for w in text.lower().split() if w not in ALL_STOP_WORDS and len(w) > 1]
        word_counts.update(tokens)
    
    # Score words: prioritize words that appear in the topic representation
    topic_rep_lower = [w.lower() for w in topic_rep_words]
    
    # First pass: exact matches with topic representation words
    matched = []
    for word in topic_rep_lower:
        if word in word_counts:
            matched.append((word, word_counts[word]))
    
    # Second pass: words from transcripts that contain topic representation as substring
    # (handles morphological variations common in Luganda, e.g., "amaloboozi" matching "maloboozi")
    for transcript_word, count in word_counts.items():
        if transcript_word in [m[0] for m in matched]:
            continue
        for rep_word in topic_rep_lower:
            if rep_word in transcript_word or transcript_word in rep_word:
                matched.append((transcript_word, count))
                break
    
    # Sort by frequency descending, take top_n
    matched.sort(key=lambda x: x[1], reverse=True)
    result = [w for w, _ in matched[:top_n]]
    
    # If we don't have enough matches, fall back to the most frequent
    # clinically relevant tokens from the transcripts
    # Stop words are already filtered out during tokenization
    if len(result) < top_n:
        for word, count in word_counts.most_common():
            if word not in result and len(word) > 2:
                result.append(word)
                if len(result) >= top_n:
                    break
    
    return result

# --- Core Inference Endpoint ---

@app.post("/predict_symptoms", response_model=CallSummary)
async def predict_symptoms(payload: Payload):
    # 1. Extract purely the conversational segments classified as 'caller' (the patient)
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

    # 2. Run inference on all caller transcripts
    predicted_topics, probabilities = topic_model.transform(caller_texts)

    # 3. Group transcripts by symptom label
    symptom_groups = defaultdict(lambda: {"texts": [], "confidences": [], "topic_ids": []})
    undefined_count = 0
    total = len(caller_texts)

    for text, topic_id, prob in zip(caller_texts, predicted_topics, probabilities):
        label = custom_labels.get(topic_id)
        
        # Topic -1 or unmapped topics are "undefined" (no symptom detected)
        if topic_id == -1 or label is None or (isinstance(label, float) and pd.isna(label)):
            undefined_count += 1
            continue
        
        symptom_groups[label]["texts"].append(text)
        symptom_groups[label]["confidences"].append(float(prob))
        symptom_groups[label]["topic_ids"].append(topic_id)

    # 4. Build aggregated symptom summaries
    classified_count = total - undefined_count
    symptoms = []

    for label, group in symptom_groups.items():
        count = len(group["texts"])
        avg_confidence = sum(group["confidences"]) / count
        representation = round((count / total) * 100, 1)
        
        # Collect all topic representation words for this symptom's topics
        all_rep_words = []
        for tid in set(group["topic_ids"]):
            all_rep_words.extend(topic_keywords_map.get(tid, []))
        
        # Extract conversation-specific keywords
        keywords = extract_conversation_keywords(group["texts"], all_rep_words, top_n=3)

        symptoms.append(SymptomSummary(
            symptom_label=label,
            symptom_representation=representation,
            confidence_score=round(avg_confidence, 4),
            keywords=keywords
        ))

    # 5. Sort by symptom_representation descending (most prevalent first)
    symptoms.sort(key=lambda s: s.symptom_representation, reverse=True)

    return CallSummary(
        total_transcripts=total,
        classified_transcripts=classified_count,
        undefined_transcripts=undefined_count,
        classification_rate=round((classified_count / total) * 100, 1) if total > 0 else 0.0,
        symptoms=symptoms
    )
