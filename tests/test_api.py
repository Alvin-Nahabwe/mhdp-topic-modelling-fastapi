"""
API integration tests using FastAPI TestClient.

These tests validate request/response schemas, input validation,
and endpoint behavior WITHOUT requiring a running server.

The model loading is mocked to avoid the ~30s startup cost and
1GB Davlan model download in CI.

Run: pytest tests/test_api.py -v
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


# --- Mock model fixtures ---

def _make_mock_models():
    """Create mock ML models dict for lifespan injection."""
    # Mock BERTopic model
    mock_topic_model = MagicMock()
    mock_topic_model.transform.return_value = (
        [0, 1, 0],  # topic ids
        [0.85, 0.72, 0.91]  # probabilities
    )

    # Mock classifier
    mock_classifier = MagicMock()
    mock_classifier.predict.return_value = np.array([0, 1, 2])
    mock_classifier.predict_proba.return_value = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.7, 0.2],
        [0.05, 0.15, 0.8],
    ])

    # Mock label encoder
    mock_le = MagicMock()
    mock_le.inverse_transform.return_value = np.array([
        "Hallucination/Delusions",
        "Low mood/Anhedonia",
        "Anxiety spectrum"
    ])
    mock_le.classes_ = np.array([
        "Hallucination/Delusions", "Low mood/Anhedonia", "Anxiety spectrum"
    ])

    # Mock embedder
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value = np.random.randn(3, 768)

    # Mock affect risk models
    mock_affect = MagicMock()
    mock_affect.predict.return_value = np.array([2])
    mock_affect.predict_proba.return_value = np.array([[0.1, 0.3, 0.6]])
    mock_affect.classes_ = np.array([1, 2, 3])

    return {
        "topic_model": mock_topic_model,
        "custom_labels": {0: "Hallucination/Delusions", 1: "Low mood/Anhedonia"},
        "topic_keywords": {
            0: ["hallucination", "voices", "seeing"],
            1: ["sad", "hopeless", "mood"],
        },
        "clinical_vocab": {"hallucination", "voices", "sad", "hopeless", "mood"},
        "embedder": mock_embedder,
        "classifier": mock_classifier,
        "label_encoder": mock_le,
        "affect_psychosis": mock_affect,
        "affect_depression": mock_affect,
        "affect_anxiety": mock_affect,
    }


@pytest.fixture
def client():
    """TestClient with mocked model loading."""
    # Patch the ml_models dict at module level
    import api
    original_models = api.ml_models.copy()

    # Override lifespan to skip model loading
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def mock_lifespan(app):
        api.ml_models.update(_make_mock_models())
        yield
        api.ml_models.clear()

    app = api.app
    app.router.lifespan_context = mock_lifespan

    with TestClient(app) as c:
        yield c

    # Restore
    api.ml_models.clear()
    api.ml_models.update(original_models)


def _make_segment(text="I hear voices telling me things", role="caller", lang="eng"):
    return {
        "end": 10.0,
        "language": lang,
        "speaker": "SPEAKER_00",
        "speaker_role_id": role,
        "start": 5.0,
        "text": text,
    }


# --- Health endpoint ---

class TestHealth:
    def test_health_returns_200(self, client):
        res = client.get("/health")
        assert res.status_code == 200

    def test_health_reports_models(self, client):
        data = res = client.get("/health").json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["classifier_loaded"] is True
        assert data["affect_risk_loaded"] is True


# --- Input validation ---

class TestInputValidation:
    def test_empty_segments_rejected(self, client):
        res = client.post("/predict_symptoms", json={"speaker_segments": []})
        assert res.status_code == 422

    def test_missing_text_rejected(self, client):
        seg = _make_segment()
        del seg["text"]
        res = client.post("/predict_symptoms", json={"speaker_segments": [seg]})
        assert res.status_code == 422

    def test_empty_text_rejected(self, client):
        seg = _make_segment(text="")
        res = client.post("/predict_symptoms", json={"speaker_segments": [seg]})
        assert res.status_code == 422

    def test_excessively_long_text_rejected(self, client):
        seg = _make_segment(text="a" * 5001)
        res = client.post("/predict_symptoms", json={"speaker_segments": [seg]})
        assert res.status_code == 422

    def test_excessive_end_time_rejected(self, client):
        seg = _make_segment()
        seg["end"] = 99999
        res = client.post("/predict_symptoms", json={"speaker_segments": [seg]})
        assert res.status_code == 422

    def test_negative_start_time_rejected(self, client):
        seg = _make_segment()
        seg["start"] = -1.0
        res = client.post("/predict_symptoms", json={"speaker_segments": [seg]})
        assert res.status_code == 422

    def test_control_characters_sanitized(self, client):
        seg = _make_segment(text="I hear\x00 voices\x01 telling me things")
        res = client.post("/predict_symptoms", json={"speaker_segments": [seg]})
        # Should succeed — control chars stripped but text remains valid
        assert res.status_code == 200


# --- Symptom prediction endpoint ---

class TestPredictSymptoms:
    def test_basic_prediction(self, client):
        res = client.post("/predict_symptoms", json={
            "speaker_segments": [
                _make_segment("I hear voices telling me things"),
                _make_segment("I feel very sad and hopeless"),
                _make_segment("I cannot sleep at night"),
            ]
        })
        assert res.status_code == 200
        data = res.json()
        assert "total_transcripts" in data
        assert "symptoms" in data
        assert data["total_transcripts"] == 3
        assert data["model_used"] in ("classifier", "bertopic")

    def test_only_caller_segments_processed(self, client):
        res = client.post("/predict_symptoms", json={
            "speaker_segments": [
                _make_segment("I hear voices", role="caller"),
                _make_segment("Tell me more", role="agent"),
                _make_segment("The voices scare me", role="caller"),
            ]
        })
        assert res.status_code == 200
        data = res.json()
        # Only 2 caller segments should be processed
        assert data["total_transcripts"] == 2

    def test_no_caller_returns_empty(self, client):
        res = client.post("/predict_symptoms", json={
            "speaker_segments": [
                _make_segment("Tell me about your symptoms", role="agent"),
            ]
        })
        assert res.status_code == 200
        data = res.json()
        assert data["total_transcripts"] == 0
        assert data["symptoms"] == []

    def test_symptom_summary_structure(self, client):
        res = client.post("/predict_symptoms", json={
            "speaker_segments": [
                _make_segment("I hear voices at night"),
            ]
        })
        assert res.status_code == 200
        data = res.json()
        for sym in data["symptoms"]:
            assert "symptom_label" in sym
            assert "symptom_representation" in sym
            assert "confidence_score" in sym
            assert "keywords" in sym
            assert isinstance(sym["keywords"], list)

    def test_classification_rate_in_range(self, client):
        res = client.post("/predict_symptoms", json={
            "speaker_segments": [
                _make_segment("I feel very anxious all the time"),
                _make_segment("I hear voices"),
            ]
        })
        data = res.json()
        assert 0.0 <= data["classification_rate"] <= 100.0


# --- Affect risk endpoint ---

class TestAffectRisk:
    def test_basic_affect_risk(self, client):
        res = client.post("/predict_affect_risk", json={
            "transcript": "I have been hearing voices for two weeks and I feel very sad and hopeless"
        })
        assert res.status_code == 200
        data = res.json()
        assert "risk_scores" in data
        assert "transcript_length" in data
        assert "clinical_terms_found" in data

    def test_affect_risk_scores_structure(self, client):
        res = client.post("/predict_affect_risk", json={
            "transcript": "I cannot sleep and I feel anxious all the time with hearing voices"
        })
        data = res.json()
        categories_found = {r["category"] for r in data["risk_scores"]}
        assert "psychosis" in categories_found
        assert "depression" in categories_found
        assert "anxiety" in categories_found

        for risk in data["risk_scores"]:
            assert 1 <= risk["score"] <= 3
            assert risk["label"] in ("Unlikely", "Possible", "Likely")
            assert "probabilities" in risk
            # Probabilities should roughly sum to 1
            total = sum(risk["probabilities"].values())
            assert 0.95 <= total <= 1.05, f"Probabilities sum to {total}"

    def test_affect_risk_short_transcript_rejected(self, client):
        res = client.post("/predict_affect_risk", json={
            "transcript": "short"
        })
        assert res.status_code == 422

    def test_affect_risk_clinical_terms_detected(self, client):
        res = client.post("/predict_affect_risk", json={
            "transcript": "I have been hearing voices and feeling very anxious with insomnia"
        })
        data = res.json()
        # Should find at least some clinical terms
        assert len(data["clinical_terms_found"]) > 0
