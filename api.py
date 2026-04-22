import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# Define a global dictionary to hold our models securely
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing Lifespan: Loading specialized NLP Architecture...")
    # 1. Fetch our state-of-the-art African Dialect Model
    embedder = SentenceTransformer("Davlan/afro-xlmr-base")
    
    # 2. Load the lightweight BERTopic architecture
    topic_model = BERTopic.load("./bertopic_supervised/model", embedding_model=embedder)
    
    # 3. Load symptom mapping
    custom_labels = pd.read_csv("./bertopic_supervised/topic_info.csv").set_index('Topic')['Symptom_Label'].to_dict()
    
    ml_models["topic_model"] = topic_model
    ml_models["custom_labels"] = custom_labels
    
    print("Models successfully deployed. Inference Endpoints Ready.")
    yield
    # Clean up on shutdown
    ml_models.clear()

app = FastAPI(lifespan=lifespan, title="BERTopic Clinical Symptoms API")

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

class SymptomResult(BaseModel):
    patient_text: str
    symptom_label: str
    confidence_score: float
    topic_keywords: list[str]

class APIResponse(BaseModel):
    results: list[SymptomResult]

# --- Core Inference Endpoint ---

@app.post("/predict_symptoms", response_model=APIResponse)
async def predict_symptoms(payload: Payload):
    # 1. Extract purely the conversational segments classified as 'caller' (the patient)
    caller_texts = [seg.text for seg in payload.speaker_segments if seg.speaker_role_id == 'caller']
    
    if not caller_texts:
        # Return an empty array if no caller text exists
        return APIResponse(results=[])

    topic_model = ml_models["topic_model"]
    custom_labels = ml_models["custom_labels"]

    # 2. Generate matrix inference
    predicted_topics, probabilities = topic_model.transform(caller_texts)

    results = []
    # 3. Zip and format outgoing structure
    for text, topic_id, score in zip(caller_texts, predicted_topics, probabilities):
        symptom_name = custom_labels.get(topic_id, "Unknown Symptom")
        
        # Safely pull the keyword tuples and parse strings
        keywords_tuples = topic_model.get_topic(topic_id)
        clean_keywords = [w for w, _ in keywords_tuples] if keywords_tuples else []

        results.append(SymptomResult(
            patient_text=text,
            symptom_label=symptom_name,
            confidence_score=float(score),
            topic_keywords=clean_keywords
        ))
        
    return APIResponse(results=results)
