"""
Dataset creation pipeline for MHDP clinical topic modelling.

Converts clinician annotations from Label Studio JSON export into a clean
CSV suitable for BERTopic training.

Handles:
- Multi-annotator overlap: same transcript annotated by multiple people
  is deduplicated to a single row per transcript+label
- Label conflicts: when the same transcript has multiple different labels,
  the majority label is kept (ties broken alphabetically)
- Conflicting annotations are logged for clinician review

Usage: python create_dataset.py
Input:  clinical_v3_final.json
Output: clinical_document.csv, label_conflicts.csv (if conflicts found)
"""

import json
import csv
import os
from collections import Counter

INPUT_FILE = "clinical_v3_final.json"
OUTPUT_FILE = "clinical_document.csv"
CONFLICTS_FILE = "label_conflicts.csv"

# --- Step 1: Extract all annotations from JSON ---

with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

raw_records = []

for task in data:
    file_upload = task.get("file_upload", "")
    filename = file_upload.split("-", 1)[-1] if "-" in file_upload else file_upload

    annotations = task.get("annotations", [])
    for ann in annotations:
        author_data = ann.get("completed_by")
        annotator_id = author_data.get("id") if isinstance(author_data, dict) else author_data

        results = ann.get("result", [])

        # Group results by segment ID
        segments = {}
        for res in results:
            segment_id = res.get("id")
            if not segment_id:
                continue

            if segment_id not in segments:
                segments[segment_id] = {
                    "annotator_id": annotator_id,
                    "filename": filename,
                    "segment_id": segment_id,
                    "transcript": "",
                    "labels": []
                }

            res_type = res.get("type")
            if res_type == "labels":
                labels = res.get("value", {}).get("labels", [])
                segments[segment_id]["labels"].extend(labels)
            elif res_type == "textarea":
                texts = res.get("value", {}).get("text", [])
                segments[segment_id]["transcript"] = " ".join(texts)

        for segment_id, seg_data in segments.items():
            transcript = seg_data["transcript"].strip()
            labels = seg_data["labels"]

            if not transcript or not labels:
                continue

            for label in labels:
                clean_label = label.strip()
                if clean_label:
                    raw_records.append({
                        "annotator_id": seg_data["annotator_id"],
                        "filename": seg_data["filename"],
                        "segment_id": seg_data["segment_id"],
                        "segment_transcript": transcript,
                        "symptom_label": clean_label
                    })

print(f"Extracted {len(raw_records)} raw annotation records.")

# --- Step 2: Deduplicate by transcript + label ---
# If multiple annotators assigned the same label to the same transcript,
# that is one training example, not multiple.

seen = set()
deduped_records = []
duplicates_removed = 0

for r in raw_records:
    key = (r["segment_transcript"], r["symptom_label"])
    if key not in seen:
        seen.add(key)
        deduped_records.append(r)
    else:
        duplicates_removed += 1

print(f"Removed {duplicates_removed} duplicate transcript+label rows.")

# --- Step 3: Resolve label conflicts ---
# If the same transcript has been assigned multiple different labels,
# keep only the majority label (most frequently assigned). Ties broken alphabetically.

transcript_labels = {}
for r in deduped_records:
    t = r["segment_transcript"]
    if t not in transcript_labels:
        transcript_labels[t] = []
    transcript_labels[t].append(r["symptom_label"])

# Find transcripts with conflicting labels
conflicts = {t: labels for t, labels in transcript_labels.items() if len(set(labels)) > 1}
print(f"Found {len(conflicts)} transcripts with conflicting labels.")

# Resolve: keep majority label for each transcript
majority_label = {}
conflict_log = []
for transcript, labels in conflicts.items():
    counts = Counter(labels)
    # Sort by count descending, then alphabetically for ties
    winner = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[0][0]
    majority_label[transcript] = winner
    conflict_log.append({
        "segment_transcript": transcript[:100],
        "labels": str(labels),
        "resolved_to": winner
    })

# Log conflicts for clinician review
if conflict_log:
    with open(CONFLICTS_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["segment_transcript", "labels", "resolved_to"])
        writer.writeheader()
        writer.writerows(conflict_log)
    print(f"Conflict details saved to {CONFLICTS_FILE}")

# Build final records: one row per transcript, using majority label for conflicts
final_seen = set()
final_records = []
conflict_rows_removed = 0

for r in deduped_records:
    transcript = r["segment_transcript"]

    # If this transcript had conflicts, only keep the majority label
    if transcript in majority_label:
        resolved_label = majority_label[transcript]
        if r["symptom_label"] != resolved_label:
            conflict_rows_removed += 1
            continue

    # Final dedup: one row per transcript
    if transcript not in final_seen:
        final_seen.add(transcript)
        final_records.append(r)

print(f"Removed {conflict_rows_removed} conflicting label rows.")

# --- Step 4: Encode labels ---

unique_labels = sorted(set(r["symptom_label"] for r in final_records))
label_to_id = {label: idx for idx, label in enumerate(unique_labels)}

for r in final_records:
    r["target"] = label_to_id[r["symptom_label"]]

# --- Step 5: Save ---

fieldnames = ['annotator_id', 'filename', 'segment_id', 'segment_transcript', 'symptom_label', 'target']

with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(final_records)

print(f"\n=== Dataset Summary ===")
print(f"Raw records:               {len(raw_records)}")
print(f"After transcript+label dedup: {len(deduped_records)}")
print(f"After conflict resolution: {len(final_records)}")
print(f"Unique symptom labels:     {len(unique_labels)}")
print(f"Saved to {OUTPUT_FILE}")
