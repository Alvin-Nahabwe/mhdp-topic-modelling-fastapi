"""
Dataset creation pipeline for MHDP clinical topic modelling (v2).

Converts clinician annotations from Label Studio JSON export into a clean
CSV suitable for BERTopic training. Also prepares affect risk dataset.

Enhancements over v1:
- HiTOP-informed label merging (25 original → 15 classes)
- Non-clinical class from unlabeled clinical files + transcription_v2 data
- Inter-annotator agreement weighting
- Timestamp extraction for temporal alignment analysis
- Affect risk dataset creation from psychosis/depression/anxiety scores

Handles:
- Multi-annotator overlap: same transcript annotated by multiple people
  is deduplicated to a single row per transcript+label
- Label conflicts: when the same transcript has multiple different labels,
  the majority label is kept (ties broken alphabetically)
- Conflicting annotations are logged for clinician review

Usage: python create_dataset.py
Input:  clinical_v3_final.json, transcription_v2.json (optional)
Output: clinical_document.csv, label_conflicts.csv, affect_risk_dataset.csv
"""

import json
import csv
import os
import random
from collections import Counter

INPUT_FILE = "clinical_v3_final.json"
V2_FILE = "transcription_v2.json"
OUTPUT_FILE = "clinical_document.csv"
CONFLICTS_FILE = "label_conflicts.csv"
AFFECT_RISK_FILE = "affect_risk_dataset.csv"

# Seed for reproducible non-clinical sampling
random.seed(42)

# Target number of non-clinical examples (start with ~300, evaluate and adjust)
NON_CLINICAL_TARGET = 300

# --- HiTOP-Informed Label Mapping ---
# Merges 25 original symptom labels into 14 clinical classes + 1 non-clinical.
# Rationale: Based on the Hierarchical Taxonomy of Psychopathology (HiTOP)
# framework which organizes symptoms into spectra (Internalizing, Thought
# Disorder, Externalizing, Detachment). Rare classes (< 5 examples) are
# merged with clinically related parent categories.
#
# NOTE: This mapping should be reviewed by a clinician. Document generated
# for clinical review is in the project's implementation plan.

HITOP_LABEL_MAP = {
    # Internalizing spectrum — Sleep
    "Insomnia/Hypersomnia": "Insomnia/Hypersomnia",

    # Thought Disorder spectrum — Disorganization
    "Disorganized behaviors": "Disorganized behaviors",

    # Cross-cutting
    "Impairment in functioning socially, occupationally, and healthwise": "Functional impairment",

    # Internalizing spectrum — Suicidality
    "Suicidal behaviour": "Suicidal behaviour",

    # Thought Disorder spectrum — Positive symptoms (MERGE: Hallucination + Delusions)
    "Hallucination": "Hallucination/Delusions",
    "Delusions": "Hallucination/Delusions",

    # Internalizing spectrum — Appetite
    "Poor appetite or overeating": "Appetite disturbance",

    # Thought Disorder spectrum — Speech
    "Disorganized speech": "Disorganized speech",

    # Internalizing spectrum — Fear (MERGE: Anxiety cluster from GAD-7 items)
    "Feeling nervous, anxious or on edge": "Anxiety spectrum",
    "Worry": "Anxiety spectrum",
    "Trouble relaxing": "Anxiety spectrum",
    "Palpitations": "Anxiety spectrum",
    "Fear of dying": "Anxiety spectrum",

    # Internalizing spectrum — Cognition
    "Trouble concentrating on things": "Concentration problems",

    # Internalizing spectrum — Energy
    "Feeling tired or having low energy": "Fatigue/Low energy",

    # Thought Disorder spectrum — Pressured speech
    "Over talkative": "Over talkative",

    # Internalizing spectrum — Distress (MERGE: Core depressive symptoms from PHQ-9)
    "Low mood": "Low mood/Anhedonia",
    "Anhedonia": "Low mood/Anhedonia",
    "Feelings of worthlessness or guilt": "Low mood/Anhedonia",

    # Externalizing spectrum — Motor (MERGE: Motor/activation symptoms)
    "Psychomotor retardation or agitation": "Psychomotor disturbance",
    "Restlessness/Hyperactive": "Psychomotor disturbance",
    "Restlesness/Hyperactive": "Psychomotor disturbance",  # Typo in original annotations
    "Excessive energy": "Psychomotor disturbance",
    "Excessive happiness": "Psychomotor disturbance",

    # Detachment spectrum (MERGE: Social withdrawal + reduced agency)
    "Isolation and withdrawal": "Isolation/Withdrawal",
    "Passivity phenomena": "Isolation/Withdrawal",
}


def extract_clinical_annotations(data):
    """Extract symptom-labeled segments from clinical annotation JSON.

    Returns:
        raw_records: list of annotation dicts
        file_segments: dict mapping filename -> list of all segment transcripts
        file_annotators: dict mapping filename -> set of annotator IDs
    """
    raw_records = []
    file_segments = {}
    file_annotators = {}

    for task in data:
        file_upload = task.get("file_upload", "")
        filename = file_upload.split("-", 1)[-1] if "-" in file_upload else file_upload

        if filename not in file_segments:
            file_segments[filename] = []
            file_annotators[filename] = set()

        annotations = task.get("annotations", [])
        for ann in annotations:
            author_data = ann.get("completed_by")
            annotator_id = author_data.get("id") if isinstance(author_data, dict) else author_data
            file_annotators[filename].add(annotator_id)

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
                        "labels": [],
                        "start": None,
                        "end": None,
                    }

                res_type = res.get("type")
                if res_type == "labels":
                    from_name = res.get("from_name", "")
                    if from_name == "symptom_labels":
                        labels = res.get("value", {}).get("labels", [])
                        segments[segment_id]["labels"].extend(labels)
                        # Extract timestamps
                        segments[segment_id]["start"] = res.get("value", {}).get("start")
                        segments[segment_id]["end"] = res.get("value", {}).get("end")
                elif res_type == "textarea":
                    from_name = res.get("from_name", "")
                    if from_name == "segment_transcript":
                        texts = res.get("value", {}).get("text", [])
                        segments[segment_id]["transcript"] = " ".join(texts)

            for segment_id, seg_data in segments.items():
                transcript = seg_data["transcript"].strip()
                labels = seg_data["labels"]

                if transcript:
                    file_segments[filename].append(transcript)

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
                            "symptom_label": clean_label,
                            "start_time": seg_data["start"],
                            "end_time": seg_data["end"],
                        })

    return raw_records, file_segments, file_annotators


def extract_non_clinical_from_unlabeled(data, file_segments):
    """Extract transcripts from clinical files that had NO symptom labels.

    These are clinician-reviewed files where no symptoms were identified —
    the strongest possible negative examples.
    """
    files_with_symptoms = set()
    all_filenames = set()

    for task in data:
        file_upload = task.get("file_upload", "")
        filename = file_upload.split("-", 1)[-1] if "-" in file_upload else file_upload
        all_filenames.add(filename)

        for ann in task.get("annotations", []):
            for res in ann.get("result", []):
                if (res.get("from_name") == "symptom_labels" and
                        res.get("type") == "labels"):
                    files_with_symptoms.add(filename)
                    break

    files_without_symptoms = all_filenames - files_with_symptoms

    # For unlabeled files, extract transcripts directly from JSON
    # (file_segments only captures transcripts found during annotation extraction)
    non_clinical = []
    for task in data:
        file_upload = task.get("file_upload", "")
        filename = file_upload.split("-", 1)[-1] if "-" in file_upload else file_upload

        if filename not in files_without_symptoms:
            continue

        for ann in task.get("annotations", []):
            for res in ann.get("result", []):
                if (res.get("from_name") == "segment_transcript" and
                        res.get("type") == "textarea"):
                    texts = res.get("value", {}).get("text", [])
                    transcript = " ".join(texts).strip()
                    if len(transcript) >= 10:
                        non_clinical.append({
                            "source": "clinical_unlabeled",
                            "filename": filename,
                            "transcript": transcript,
                        })

    print(f"Non-clinical from unlabeled clinical files: {len(non_clinical)} segments "
          f"from {len(files_without_symptoms)} files")
    return non_clinical


def extract_non_clinical_from_v2(v2_path):
    """Extract agent segments from transcription_v2.json as non-clinical examples.

    Agent/counsellor speech is by definition non-clinical (questions,
    instructions, empathy statements). These are the cleanest non-clinical
    examples from a different audio set.
    """
    if not os.path.exists(v2_path):
        print(f"  {v2_path} not found, skipping V2 non-clinical extraction.")
        return []

    with open(v2_path, 'r', encoding='utf-8') as f:
        v2_data = json.load(f)

    agent_segments = []
    for entry in v2_data:
        file_upload = entry.get("file_upload", "")
        filename = file_upload.split("-", 1)[-1] if "-" in file_upload else file_upload

        for ann in entry.get("annotations", []):
            results = ann.get("result", [])
            # Group by segment ID to pair labels with transcripts
            segments = {}
            for res in results:
                seg_id = res.get("id")
                if not seg_id:
                    continue
                if seg_id not in segments:
                    segments[seg_id] = {"label": None, "transcript": None}

                if res.get("from_name") == "speaker_labels" and res["type"] == "labels":
                    labels = res.get("value", {}).get("labels", [])
                    if labels:
                        segments[seg_id]["label"] = labels[0]
                elif res.get("from_name") == "segment_transcript" and res["type"] == "textarea":
                    texts = res.get("value", {}).get("text", [])
                    if texts:
                        segments[seg_id]["transcript"] = " ".join(texts).strip()

            for seg_id, seg in segments.items():
                if (seg["label"] == "Agent" and
                        seg["transcript"] and
                        len(seg["transcript"]) >= 10):
                    agent_segments.append({
                        "source": "v2_agent",
                        "filename": filename,
                        "transcript": seg["transcript"],
                    })

    print(f"Non-clinical from V2 agent segments: {len(agent_segments)} segments")
    return agent_segments


def build_non_clinical_class(clinical_unlabeled, v2_agent, target_count):
    """Sample and balance non-clinical examples to target count.

    Priority order: clinical unlabeled (clinician-verified) > v2 agent segments.
    """
    # Take all clinical unlabeled (highest quality)
    selected = list(clinical_unlabeled)

    if len(selected) < target_count and v2_agent:
        # Fill remaining with v2 agent segments
        remaining = target_count - len(selected)
        if len(v2_agent) <= remaining:
            selected.extend(v2_agent)
        else:
            selected.extend(random.sample(v2_agent, remaining))

    # If we have too many, sample down (prefer clinical unlabeled)
    if len(selected) > target_count:
        # Keep all clinical, reduce v2
        clinical = [s for s in selected if s["source"] == "clinical_unlabeled"]
        v2 = [s for s in selected if s["source"] == "v2_agent"]
        if len(clinical) >= target_count:
            selected = random.sample(clinical, target_count)
        else:
            selected = clinical + random.sample(v2, target_count - len(clinical))

    print(f"Non-clinical class: {len(selected)} examples "
          f"(clinical: {sum(1 for s in selected if s['source'] == 'clinical_unlabeled')}, "
          f"v2: {sum(1 for s in selected if s['source'] == 'v2_agent')})")
    return selected


def compute_agreement_weights(raw_records, file_annotators):
    """Compute confidence weights based on inter-annotator agreement.

    Weighting scheme:
    - 3 annotators agree on label: weight = 1.0 (high confidence)
    - 2/3 annotators agree (majority vote): weight = 0.8
    - Single annotator only: weight = 0.6 (no validation possible)
    """
    # Group records by (transcript, label) to count annotators agreeing
    transcript_label_annotators = {}
    for r in raw_records:
        key = (r["segment_transcript"], r["symptom_label"])
        if key not in transcript_label_annotators:
            transcript_label_annotators[key] = set()
        transcript_label_annotators[key].add(r["annotator_id"])

    # For each transcript, count total annotators and agreeing annotators
    transcript_all_labels = {}
    for r in raw_records:
        t = r["segment_transcript"]
        if t not in transcript_all_labels:
            transcript_all_labels[t] = []
        transcript_all_labels[t].append(r["symptom_label"])

    weights = {}
    for transcript, labels in transcript_all_labels.items():
        unique_labels = set(labels)
        total_annotations = len(labels)
        if len(unique_labels) == 1 and total_annotations >= 3:
            # Full agreement across 3+ annotators
            weights[transcript] = 1.0
        elif len(unique_labels) == 1 and total_annotations == 2:
            # Two annotators agree
            weights[transcript] = 0.9
        elif len(unique_labels) == 1:
            # Single annotator
            weights[transcript] = 0.6
        elif total_annotations >= 3:
            # Disagreement but majority exists
            weights[transcript] = 0.8
        else:
            # Two annotators disagree
            weights[transcript] = 0.5

    return weights


def extract_affect_risk(data, output_path):
    """Extract file-level affect risk scores and clinician notes.

    Scores represent likelihood that the condition is present:
    1=unlikely, 2=possible, 3=likely
    """
    file_data = {}

    for task in data:
        file_upload = task.get("file_upload", "")
        filename = file_upload.split("-", 1)[-1] if "-" in file_upload else file_upload

        if filename not in file_data:
            file_data[filename] = {
                "filename": filename,
                "psychosis_scores": [],
                "depression_scores": [],
                "anxiety_scores": [],
                "psychosis_confirmed": False,
                "depression_confirmed": False,
                "anxiety_confirmed": False,
                "psychosis_notes": [],
                "depression_notes": [],
                "anxiety_notes": [],
                "other_notes": [],
                "transcripts": [],
            }

        for ann in task.get("annotations", []):
            for res in ann.get("result", []):
                fn = res.get("from_name", "")
                val = res.get("value", {})

                if fn == "psychosis_score" and res["type"] == "rating":
                    file_data[filename]["psychosis_scores"].append(val.get("rating"))
                elif fn == "depression_score" and res["type"] == "rating":
                    file_data[filename]["depression_scores"].append(val.get("rating"))
                elif fn == "anxiety_score" and res["type"] == "rating":
                    file_data[filename]["anxiety_scores"].append(val.get("rating"))
                elif fn == "psychosis_confirmed" and res["type"] == "choices":
                    file_data[filename]["psychosis_confirmed"] = True
                elif fn == "depression_confirmed" and res["type"] == "choices":
                    file_data[filename]["depression_confirmed"] = True
                elif fn == "anxiety_confirmed" and res["type"] == "choices":
                    file_data[filename]["anxiety_confirmed"] = True
                elif fn == "psychosis_note" and res["type"] == "textarea":
                    file_data[filename]["psychosis_notes"].extend(val.get("text", []))
                elif fn == "depression_note" and res["type"] == "textarea":
                    file_data[filename]["depression_notes"].extend(val.get("text", []))
                elif fn == "anxiety_note" and res["type"] == "textarea":
                    file_data[filename]["anxiety_notes"].extend(val.get("text", []))
                elif fn == "other_note" and res["type"] == "textarea":
                    file_data[filename]["other_notes"].extend(val.get("text", []))
                elif fn == "segment_transcript" and res["type"] == "textarea":
                    file_data[filename]["transcripts"].extend(val.get("text", []))

    # Write affect risk dataset
    rows = []
    for filename, fd in file_data.items():
        # Aggregate scores (take max across annotators for ordinal likelihood)
        psy_score = max(fd["psychosis_scores"]) if fd["psychosis_scores"] else None
        dep_score = max(fd["depression_scores"]) if fd["depression_scores"] else None
        anx_score = max(fd["anxiety_scores"]) if fd["anxiety_scores"] else None

        # Skip files with no scores at all
        if psy_score is None and dep_score is None and anx_score is None:
            continue

        full_transcript = " ".join(t.strip() for t in fd["transcripts"] if t.strip())

        rows.append({
            "filename": filename,
            "full_transcript": full_transcript[:5000],  # Cap length
            "psychosis_score": psy_score,
            "depression_score": dep_score,
            "anxiety_score": anx_score,
            "psychosis_confirmed": fd["psychosis_confirmed"],
            "depression_confirmed": fd["depression_confirmed"],
            "anxiety_confirmed": fd["anxiety_confirmed"],
            "psychosis_notes": " | ".join(fd["psychosis_notes"])[:1000],
            "depression_notes": " | ".join(fd["depression_notes"])[:1000],
            "anxiety_notes": " | ".join(fd["anxiety_notes"])[:1000],
            "other_notes": " | ".join(fd["other_notes"])[:1000],
        })

    fieldnames = [
        "filename", "full_transcript",
        "psychosis_score", "depression_score", "anxiety_score",
        "psychosis_confirmed", "depression_confirmed", "anxiety_confirmed",
        "psychosis_notes", "depression_notes", "anxiety_notes", "other_notes",
    ]

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    psy_count = sum(1 for r in rows if r["psychosis_score"] is not None)
    dep_count = sum(1 for r in rows if r["depression_score"] is not None)
    anx_count = sum(1 for r in rows if r["anxiety_score"] is not None)
    print(f"\nAffect risk dataset: {len(rows)} files")
    print(f"  Psychosis scored: {psy_count} | Depression scored: {dep_count} | Anxiety scored: {anx_count}")
    print(f"  Saved to {output_path}")


def main():
    # --- Step 1: Load clinical annotations ---
    print("=" * 60)
    print("  MHDP Dataset Creation Pipeline (v2)")
    print("=" * 60)
    print()

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} annotated audio files from {INPUT_FILE}")

    # --- Step 2: Extract all annotations ---
    raw_records, file_segments, file_annotators = extract_clinical_annotations(data)
    print(f"Extracted {len(raw_records)} raw annotation records.")

    # --- Step 3: Apply HiTOP label merging ---
    unmapped_labels = set()
    for r in raw_records:
        original = r["symptom_label"]
        if original in HITOP_LABEL_MAP:
            r["symptom_label"] = HITOP_LABEL_MAP[original]
        else:
            unmapped_labels.add(original)

    if unmapped_labels:
        print(f"WARNING: {len(unmapped_labels)} labels not in HiTOP map: {unmapped_labels}")

    merged_labels = set(r["symptom_label"] for r in raw_records)
    print(f"After HiTOP merging: {len(merged_labels)} clinical classes (from 25 original)")

    # --- Step 4: Deduplicate by transcript + label ---
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

    # --- Step 5: Resolve label conflicts ---
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

    # Build final clinical records
    final_seen = set()
    final_records = []
    conflict_rows_removed = 0

    for r in deduped_records:
        transcript = r["segment_transcript"]

        if transcript in majority_label:
            resolved_label = majority_label[transcript]
            if r["symptom_label"] != resolved_label:
                conflict_rows_removed += 1
                continue

        if transcript not in final_seen:
            final_seen.add(transcript)
            final_records.append(r)

    print(f"Removed {conflict_rows_removed} conflicting label rows.")

    # --- Step 6: Compute agreement weights ---
    weights = compute_agreement_weights(raw_records, file_annotators)
    for r in final_records:
        r["agreement_weight"] = weights.get(r["segment_transcript"], 0.6)

    # Count annotators per file for each record
    for r in final_records:
        r["annotator_count"] = len(file_annotators.get(r["filename"], set()))

    # --- Step 7: Build non-clinical class ---
    print("\n--- Building Non-Clinical Class ---")
    clinical_unlabeled = extract_non_clinical_from_unlabeled(data, file_segments)
    v2_agent = extract_non_clinical_from_v2(V2_FILE)
    non_clinical_examples = build_non_clinical_class(
        clinical_unlabeled, v2_agent, NON_CLINICAL_TARGET
    )

    # Convert non-clinical to same format as clinical records
    for nc in non_clinical_examples:
        final_records.append({
            "annotator_id": "system",
            "filename": nc["filename"],
            "segment_id": f"nc_{nc['source']}",
            "segment_transcript": nc["transcript"],
            "symptom_label": "Non-Clinical",
            "start_time": None,
            "end_time": None,
            "agreement_weight": 0.7,  # Moderate confidence (inferred, not annotated)
            "annotator_count": 0,
        })

    # --- Step 8: Encode labels ---
    unique_labels = sorted(set(r["symptom_label"] for r in final_records))
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}

    for r in final_records:
        r["target"] = label_to_id[r["symptom_label"]]

    # --- Step 9: Save symptom classification dataset ---
    fieldnames = [
        'annotator_id', 'filename', 'segment_id', 'segment_transcript',
        'symptom_label', 'target', 'agreement_weight', 'annotator_count',
        'start_time', 'end_time',
    ]

    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_records)

    # --- Step 10: Create affect risk dataset ---
    print("\n--- Creating Affect Risk Dataset ---")
    extract_affect_risk(data, AFFECT_RISK_FILE)

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print(f"  Dataset Summary")
    print(f"{'=' * 60}")
    print(f"Raw records:                {len(raw_records)}")
    print(f"After transcript+label dedup: {len(deduped_records)}")
    print(f"After conflict resolution:  {len(final_records) - len(non_clinical_examples)}")
    print(f"Non-clinical added:         {len(non_clinical_examples)}")
    print(f"Total training examples:    {len(final_records)}")
    print(f"Unique classes:             {len(unique_labels)}")
    print()
    print("Class distribution:")
    label_counts = Counter(r["symptom_label"] for r in final_records)
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        weight_avg = sum(r["agreement_weight"] for r in final_records if r["symptom_label"] == label) / count
        print(f"  {label:45s} {count:4d}  (avg weight: {weight_avg:.2f})")
    print()
    print(f"Label encoding: {label_to_id}")
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
