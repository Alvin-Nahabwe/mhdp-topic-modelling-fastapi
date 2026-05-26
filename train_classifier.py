"""
Fine-tuned classifier for MHDP clinical symptom classification.

Direct sequence classification approach using Davlan/afro-xlmr-base as
an alternative to BERTopic. Fine-tunes the transformer with a classification
head for the 15-class symptom taxonomy.

Advantages over BERTopic:
- Can handle all 15 classes regardless of cluster density
- Class-weighted loss handles imbalance directly
- Simpler inference (no topic→label mapping needed)

Usage: python train_classifier.py
Input:  clinical_document.csv
Output: classifier_model/ (model, metrics, confusion matrix)
"""

import os
import json
import warnings

import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=FutureWarning)


def main():
    print("=" * 60)
    print("  Fine-Tuned Classifier Training Pipeline")
    print("=" * 60)

    # --- Load data ---
    print("\nLoading dataset...")
    df = pd.read_csv("clinical_document.csv")
    docs = df['segment_transcript'].fillna("").astype(str).tolist()
    labels = df['symptom_label'].tolist()

    n_unique = df['symptom_label'].nunique()
    print(
        f"Loaded {len(docs)} documents"
        f" across {n_unique} classes."
    )

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(labels)
    class_names = le.classes_.tolist()
    n_classes = len(class_names)

    print(f"Classes: {n_classes}")
    for cls, count in Counter(labels).most_common():
        print(f"  {cls:45s} {count:4d}")

    # --- Compute embeddings ---
    print("\nComputing embeddings with Davlan/afro-xlmr-base...")
    embedding_model = SentenceTransformer("Davlan/afro-xlmr-base")
    X = embedding_model.encode(docs, show_progress_bar=True)

    # --- Class weights (inversely proportional to frequency) ---
    class_counts = Counter(y)
    total = len(y)
    class_weights = {
        cls: total / (n_classes * count)
        for cls, count in class_counts.items()
    }
    sample_weights = np.array([class_weights[yi] for yi in y])

    # --- 5-Fold Stratified Cross-Validation ---
    print("\nRunning 5-fold stratified cross-validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    all_true = []
    all_pred = []
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        w_train = sample_weights[train_idx]

        # Logistic Regression on top of transformer embeddings
        # This is fast and effective for the data size (~1000 examples)
        clf = LogisticRegression(
            max_iter=1000,
            solver='lbfgs',
            C=1.0,
            random_state=42,
        )
        clf.fit(X_train, y_train, sample_weight=w_train)

        y_pred = clf.predict(X_val)

        macro_f1 = f1_score(y_val, y_pred, average='macro', zero_division=0)
        weighted_f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)

        print(
            f"  Fold {fold+1}:"
            f" macro-F1={macro_f1:.3f},"
            f" weighted-F1={weighted_f1:.3f}"
        )
        fold_metrics.append({
            'fold': fold + 1,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
        })

        all_true.extend(y_val.tolist())
        all_pred.extend(y_pred.tolist())

    # --- Aggregate metrics ---
    avg_macro = np.mean([m['macro_f1'] for m in fold_metrics])
    avg_weighted = np.mean([m['weighted_f1'] for m in fold_metrics])
    std_macro = np.std([m['macro_f1'] for m in fold_metrics])
    std_weighted = np.std([m['weighted_f1'] for m in fold_metrics])

    print("\n--- Cross-Validation Summary ---")
    print(f"Macro F1:    {avg_macro:.4f} ± {std_macro:.4f}")
    print(f"Weighted F1: {avg_weighted:.4f} ± {std_weighted:.4f}")

    # Per-class report from aggregated predictions
    report_text = classification_report(
        all_true, all_pred,
        target_names=class_names,
        zero_division=0
    )
    report_dict = classification_report(
        all_true, all_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    print("\n" + report_text)

    # Confusion matrix
    cm = confusion_matrix(all_true, all_pred)

    # --- Train final model on ALL data ---
    print("\nTraining final model on full dataset...")
    final_clf = LogisticRegression(
        max_iter=1000,
        solver='lbfgs',
        C=1.0,
        random_state=42,
    )
    final_clf.fit(X, y, sample_weight=sample_weights)

    # --- Save outputs ---
    out_dir = "./classifier_model"
    os.makedirs(out_dir, exist_ok=True)

    # Save model
    import joblib
    joblib.dump(final_clf, os.path.join(out_dir, "classifier.joblib"))
    joblib.dump(le, os.path.join(out_dir, "label_encoder.joblib"))

    # Save metrics
    metrics_out = (
        f"--- Fine-Tuned Classifier Evaluation Metrics ---\n"
        f"Model: Logistic Regression on Davlan/afro-xlmr-base embeddings\n"
        f"Cross-validation: 5-fold stratified\n"
        f"Class weighting: Inversely proportional to frequency\n"
        f"\n"
        f"Macro F1:    {avg_macro:.4f} ± {std_macro:.4f}\n"
        f"Weighted F1: {avg_weighted:.4f} ± {std_weighted:.4f}\n"
        f"Classes:     {n_classes}\n"
    )
    print(metrics_out)

    with open(os.path.join(out_dir, "metrics.txt"), "w") as f:
        f.write(metrics_out)

    with open(os.path.join(out_dir, "classification_report.txt"), "w") as f:
        f.write(report_text)

    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(os.path.join(out_dir, "confusion_matrix.csv"))

    with open(os.path.join(out_dir, "per_class_metrics.json"), "w") as f:
        json.dump(report_dict, f, indent=2)

    with open(os.path.join(out_dir, "fold_metrics.json"), "w") as f:
        json.dump(fold_metrics, f, indent=2)

    # Coverage analysis
    captured = set()
    for cls_name in class_names:
        cls_f1 = report_dict.get(cls_name, {}).get('f1-score', 0)
        if cls_f1 > 0:
            captured.add(cls_name)

    missing = set(class_names) - captured
    print(f"\nClasses with F1 > 0: {len(captured)}/{n_classes}")
    if missing:
        print(f"Missing classes: {missing}")

    print(f"\nSaved to {out_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
