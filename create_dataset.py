import json
import csv
import os

input_file = "clinical_v3_final.json"
output_file = "clinical_document.csv"

# Load JSON
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

records = []
seen = set()

for task in data:
    # 1. Filename extraction & prefix removal
    file_upload = task.get("file_upload", "")
    filename = file_upload.split("-", 1)[-1] if "-" in file_upload else file_upload
    
    annotations = task.get("annotations", [])
    for ann in annotations:
        # 2. Extract annotator_id
        author_data = ann.get("completed_by")
        annotator_id = author_data.get("id") if isinstance(author_data, dict) else author_data
        
        results = ann.get("result", [])
        
        # Group by segment ID
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

        # 3. Data Quality Checks & Structure Building
        for segment_id, seg_data in segments.items():
            transcript = seg_data["transcript"].strip()
            labels = seg_data["labels"]
            
            # Missing Value check: transcript and label must exist
            if not transcript or not labels:
                continue
                
            for label in labels:
                # Normalization
                clean_label = label.strip()
                if clean_label:
                    # Create signature for unique row checking
                    row_tuple = (
                        str(seg_data["annotator_id"]),
                        seg_data["filename"],
                        seg_data["segment_id"],
                        transcript,
                        clean_label
                    )
                    
                    # Duplicate handling
                    if row_tuple not in seen:
                        seen.add(row_tuple)
                        records.append({
                            "annotator_id": seg_data["annotator_id"],
                            "filename": seg_data["filename"],
                            "segment_id": seg_data["segment_id"],
                            "segment_transcript": transcript,
                            "symptom_label": clean_label
                        })

# 4. Encoding: Extract unique labels and sort them to ensure deterministic label encoding
unique_labels = sorted(list(set(r["symptom_label"] for r in records)))
label_to_id = {label: idx for idx, label in enumerate(unique_labels)}

# Add encoded target to records
for r in records:
    r["target"] = label_to_id[r["symptom_label"]]
    
# 5. Save: Write to CSV strictly defining columns requested
fieldnames = ['annotator_id', 'filename', 'segment_id', 'segment_transcript', 'symptom_label', 'target']

with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(records)

print(f"Processed {len(records)} unique records.")
print(f"Total dropped due to duplicates: {len(seen) - len(records)}")
print(f"Unique symptom labels (targets encoded): {len(unique_labels)}")
print(f"File successfully saved to {output_file}")
