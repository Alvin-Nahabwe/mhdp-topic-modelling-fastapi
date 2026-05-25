# MHDP Clinical Symptom Classification API

Clinical symptom classification and affect risk assessment for the Mental Health Data Prize (MHDP) project at Butabika National Referral Mental Hospital, Uganda. Classifies call centre transcripts into mental health symptom categories using a multi-model pipeline grounded in the HiTOP dimensional taxonomy.

## Architecture

```
                          ┌──────────────────────┐
                          │   Call Transcript     │
                          │  (diarized segments)  │
                          └──────────┬─────────────┘
                                     │
                          ┌──────────▼─────────────┐
                          │  Davlan/afro-xlmr-base  │
                          │  (768-dim embeddings)   │
                          │  English + Luganda      │
                          └──────────┬─────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    │                                  │
         ┌──────────▼──────────┐           ┌──────────▼──────────┐
         │  Symptom Classifier │           │  Affect Risk Model  │
         │  (LogReg, 15 HiTOP  │           │  (Ordinal, 3 cats)  │
         │   classes)          │           │  Psychosis/Dep/Anx  │
         └──────────┬──────────┘           └──────────┬──────────┘
                    │                                  │
         ┌──────────▼──────────┐           ┌──────────▼──────────┐
         │  Clinical Vocab     │           │  Likelihood Scores  │
         │  Gate (regex)       │           │  1=Unlikely          │
         │  195 HiTOP terms    │           │  2=Possible          │
         └──────────┬──────────┘           │  3=Likely            │
                    │                      └──────────────────────┘
         ┌──────────▼──────────┐
         │  Aggregated Summary │
         │  per symptom class  │
         └─────────────────────┘
```

The service receives diarized call transcripts, classifies each caller segment into a clinical symptom category, and returns an aggregated per-call summary with symptom prevalence, confidence scores, and conversation-specific keywords. A separate endpoint provides ordinal risk assessment for psychosis, depression, and anxiety.

## Models

### Symptom Classifier (Production)

Fine-tuned Logistic Regression on frozen `Davlan/afro-xlmr-base` embeddings. **Recommended** for production.

| Metric | Score |
|---|---|
| Macro F1 (5-fold CV) | 0.526 |
| Weighted F1 | 0.622 |
| Classes captured | **15/15** |
| Best per-class | Insomnia/Hypersomnia (0.81) |

### BERTopic (Fallback)

Semi-supervised BERTopic with guided seed topics, UMAP + HDBSCAN.

| Metric | Score |
|---|---|
| Macro F1 | 0.339 |
| Classes captured | 11/15 |
| Topic Coherence (C_v) | 0.465 |
| Outlier rate | 12.9% |

### Affect Risk (Ordinal)

Cumulative threshold ordinal regression for likelihood estimation.

| Category | Samples | MAE | Adjacent Accuracy |
|---|---|---|---|
| Psychosis | 138 | 0.754 | 81.2% |
| Depression | 88 | 0.546 | 92.0% |
| Anxiety | 42 | 0.381 | 88.1% |

## HiTOP Symptom Taxonomy (15 classes)

The classification uses a HiTOP-informed dimensional taxonomy that consolidates 25 original fragmented labels into 15 clinically meaningful classes:

| Class | Description | Training Examples |
|---|---|---|
| Insomnia/Hypersomnia | Sleep disturbance | 130 |
| Disorganized behaviors | Behavioral dysregulation | 104 |
| Functional impairment | Occupational/social dysfunction | 97 |
| Hallucination/Delusions | Perceptual/belief disturbance | 62 |
| Suicidal behaviour | Self-harm ideation/attempts | 55 |
| Anxiety spectrum | Worry, nervousness, panic | 55 |
| Low mood/Anhedonia | Depressed mood, loss of interest | 49 |
| Appetite disturbance | Eating changes | 34 |
| Concentration problems | Cognitive difficulty | 33 |
| Disorganized speech | Communication disruption | 32 |
| Fatigue/Low energy | Physical exhaustion | 26 |
| Over talkative | Pressured speech | 26 |
| Isolation/Withdrawal | Social avoidance | 24 |
| Psychomotor disturbance | Agitation/retardation | 14 |
| Non-Clinical | Greetings, logistics, filler | 300 |

## API Endpoints

### `POST /predict_symptoms`

Classify caller transcripts into symptom categories.

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
  "model_used": "classifier",
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

### `POST /predict_affect_risk`

Ordinal likelihood assessment for psychosis, depression, and anxiety.

**Request:**
```json
{
  "transcript": "Full call transcript text..."
}
```

**Response:**
```json
{
  "transcript_length": 1234,
  "risk_scores": [
    {
      "category": "psychosis",
      "score": 2,
      "label": "Possible",
      "probabilities": {
        "Unlikely": 0.15,
        "Possible": 0.55,
        "Likely": 0.30
      }
    }
  ],
  "clinical_terms_found": ["voices", "hearing", "paranoid"]
}
```

### `GET /health`

Container health check. Returns model load status.

### `GET /metrics`

Operational metrics: request counts, average latency, prediction/affect risk totals.

## Project Structure

```
├── api.py                          # FastAPI inference service (v2)
├── clinical_vocabulary.py          # 213 HiTOP-aligned clinical terms
├── stop_words.py                   # Audited English/Luganda stop words
├── train_classifier.py             # LogReg classifier (production model)
├── train_bertopic.py               # BERTopic v2 (seed topics, guided)
├── train_affect_risk.py            # Ordinal affect risk model
├── create_dataset.py               # JSON annotation → CSV pipeline (v2)
├── test_api.py                     # Manual API integration test
├── tests/                          # pytest test suite (37 tests)
│   ├── test_api.py                 # API endpoint tests (mocked models)
│   └── test_vocabulary.py          # Vocabulary integrity tests
├── classifier_model/               # Production classifier artifacts
│   ├── classifier.joblib
│   └── label_encoder.joblib
├── affect_risk_model/              # Ordinal risk models
│   ├── psychosis/model.joblib
│   ├── depression/model.joblib
│   └── anxiety/model.joblib
├── bertopic_semi_supervised/       # BERTopic model (fallback)
├── Dockerfile                      # Multi-stage production container
├── requirements.txt                # Python dependencies
├── pyproject.toml                  # pytest configuration
└── .github/workflows/ci.yml       # CI/CD pipeline
```

## Deployment

The service is containerized with a multi-stage Docker build (builder + production) and deployed via GitHub Actions.

**Pipeline:** Push to `main` → Lint → Test (37 tests) → Build Docker → Smoke test → Deploy

**Local development:**
```bash
pip install -r requirements.txt
pip install pytest pytest-asyncio httpx

# Run tests
python -m pytest tests/ -v

# Start server
uvicorn api:app --host 0.0.0.0 --port 8000
```

## Training

To retrain models:

```bash
# 1. Generate training CSV from annotated JSON
python create_dataset.py

# 2. Train classifier (recommended)
python train_classifier.py

# 3. Train affect risk model
python train_affect_risk.py

# 4. (Optional) Train BERTopic for comparison
python train_bertopic.py
```

Training requires `clinical_v3_final.json` (clinician annotations) in the project root.

## Security

> **Note:** Early commits in this repository's git history contain deployment credentials (`deployment/` directory). These files are now gitignored and removed from the working tree, but remain in git history. The repository MUST remain private. If the repository is ever made public, all deployment SSH keys must be rotated first.
