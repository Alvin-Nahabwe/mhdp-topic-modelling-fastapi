# MHDP Topic Modelling API

Clinical symptom classification service for the Mental Health Data Platform (MHDP) at Butabika National Referral Mental Hospital, Uganda. Classifies call centre transcripts into mental health symptom categories using semi-supervised topic modelling.

## Architecture

```
Call Transcript → BERTopic Inference → Aggregated Symptom Summary
                        │
       Davlan/afro-xlmr-base embeddings
       (cross-lingual English/Luganda)
```

The service receives diarized call transcripts, classifies each caller segment into a clinical symptom category (or marks it as undefined for non-symptom speech), and returns an aggregated per-call summary with symptom prevalence, confidence scores, and conversation-specific keywords.

## Model

**Architecture:** Semi-supervised BERTopic with UMAP dimensionality reduction, HDBSCAN clustering, and MMR-diversified representations.

| Parameter | Value |
|---|---|
| Embedding model | `Davlan/afro-xlmr-base` |
| `min_topic_size` | 2 |
| `nr_topics` | 50 |
| Topic Coherence (C_v) | 0.5343 |
| Topic Diversity (PUW) | 0.8503 |
| Symptom classes captured | 16 / 25 |
| Outlier rate | 10.2% |

The semi-supervised approach passes clinician-annotated symptom labels as soft targets during training, allowing the model to discover natural topic structure while being guided by known symptom categories. An outlier class (topic -1) captures non-symptom conversational speech.

## API

**Endpoint:** `POST /predict_symptoms`

**Request:**
```json
{
  "speaker_segments": [
    {
      "end": 7.45,
      "language": "lug",
      "speaker": "SPEAKER_00",
      "speaker_role_id": "caller",
      "start": 3.01,
      "text": "sifuna tulo buli kiro"
    }
  ]
}
```

**Response:**
```json
{
  "total_transcripts": 10,
  "classified_transcripts": 7,
  "undefined_transcripts": 3,
  "classification_rate": 70.0,
  "symptoms": [
    {
      "symptom_label": "Insomnia/Hypersomnia",
      "symptom_representation": 40.0,
      "confidence_score": 0.7823,
      "keywords": ["tulo", "sleep", "night"]
    }
  ]
}
```

Only segments with `speaker_role_id: "caller"` are classified. Results are sorted by prevalence (most dominant symptom first).

## Project Structure

```
├── api.py                          # FastAPI inference service
├── stop_words.py                   # Shared English/Luganda stop word lists
├── train_bertopic.py               # Semi-supervised training pipeline (production)
├── train_supervised_bertopic.py    # Supervised training pipeline (comparison)
├── create_dataset.py               # JSON annotation → CSV conversion
├── test_api.py                     # API integration test
├── bertopic_semi_supervised/       # Model artifacts (deployed)
│   ├── model/                      # Safetensors model files
│   └── topic_info.csv              # Topic-to-symptom mapping
├── Dockerfile                      # Production container
├── requirements.txt                # Python dependencies
└── .github/workflows/deploy.yml    # CI/CD pipeline
```

## Deployment

The service is containerized and deployed via GitHub Actions to a Tailscale-secured VM.

**Pipeline:** Push to `main` → Build Docker image → Push to GHCR → SSH deploy to VM

**Requirements:**
- GitHub Secrets: `SSH_PRIVATE_KEY`, `TS_AUTHKEY`, `MODEL_VM_IP`, `SUDO_PASS`
- The VM must have Docker installed and Tailscale connected

**Local development:**
```bash
pip install -r requirements.txt
uvicorn api:app --host 0.0.0.0 --port 8000
```

## Training

To retrain the model:

```bash
# 1. Generate training CSV from annotated JSON
python create_dataset.py

# 2. Train semi-supervised model
python train_bertopic.py

# 3. (Optional) Train supervised model for comparison
python train_supervised_bertopic.py
```

Training requires `clinical_v3_final.json` (clinician annotations) in the project root.

## Symptom Classes

The model captures 16 of 25 annotated symptom categories. The 9 missing classes have insufficient training data (1-16 examples) for density-based clustering.

**Captured:** Insomnia/Hypersomnia, Impairment in functioning, Hallucination, Disorganized speech, Disorganized behaviors, Anxiety, Fatigue, Low mood, Concentration difficulties, Over talkative, Appetite changes, Worthlessness/guilt, Worry, Anhedonia, Suicidal behaviour, Psychomotor changes.

**Missing (data-limited):** Delusions, Excessive energy, Excessive happiness, Fear of dying, Isolation, Palpitations, Passivity phenomena, Restlessness, Trouble relaxing.
