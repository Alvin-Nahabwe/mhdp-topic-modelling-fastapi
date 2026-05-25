# HiTOP Taxonomy Rationale for MHDP Clinical Classification

## Purpose

This document provides the clinical rationale for the symptom taxonomy used in the MHDP call centre classification system at Butabika National Referral Mental Hospital, Uganda. It explains why 25 original clinician annotation labels were consolidated into 15 HiTOP-informed classes, the evidence supporting each merge decision, and the limitations that should be considered during clinical review.

> **Status:** Assembled from validated screening instruments (PHQ-9, GAD-7, PSQ) and HiTOP dimensional framework. Pending formal clinical review.

---

## Background

### The Problem

The original annotation scheme used 25 symptom labels derived from clinical screening instruments. During model training, 9 of these labels had insufficient examples for density-based clustering (1-16 examples each), and several labels represented overlapping constructs (e.g., "Hallucination" and "Delusions" are both positive symptoms of psychosis). This fragmentation caused:

1. **Model failure on rare classes** — BERTopic could not form stable clusters for labels with <15 examples
2. **Semantic overlap** — Embeddings for "Low mood" and "Anhedonia" occupy nearly identical regions in the Davlan/afro-xlmr-base embedding space
3. **Clinical redundancy** — Several labels map to the same screening instrument items (e.g., "Worry", "Trouble relaxing", "Palpitations" are all GAD-7 items measuring generalized anxiety)

### Why HiTOP

The Hierarchical Taxonomy of Psychopathology (HiTOP; Kotov et al., 2017, 2021) provides a dimensional framework that organizes psychopathology into empirically-derived spectra. Unlike categorical systems (DSM-5, ICD-11), HiTOP explicitly addresses the comorbidity and boundary problems that make fine-grained symptom labels unreliable in real-world clinical data.

HiTOP organizes symptoms into six spectra:
- **Internalizing** — distress (depression, anxiety) and fear (phobias, panic)
- **Thought Disorder** — psychosis, disorganization
- **Externalizing** — disinhibition, antagonism
- **Detachment** — social withdrawal, restricted affect
- **Somatoform** — somatic complaints
- **Mania** — elevated mood, grandiosity

Our 15-class taxonomy maps to four of these spectra relevant to the Butabika call centre population.

---

## Merge Decisions

### Merge 1: Hallucination + Delusions → Hallucination/Delusions

| Detail | Value |
|---|---|
| HiTOP Spectrum | Thought Disorder |
| Original counts | Hallucination: ~45, Delusions: ~17 |
| Merged count | 62 |
| Rationale | Both are positive symptoms of psychosis. In the PSQ (Psychosis Screening Questionnaire), hallucinations and persecutory beliefs are assessed together as indicators of psychotic experience. In clinical practice, callers rarely present with isolated delusions without perceptual disturbance — the distinction is diagnostically meaningful but not separable from call transcript language alone. |
| Screening source | PSQ items 1-4; PANSS positive subscale |
| Luganda coverage | `okulaba` (seeing), `okuwulira` (hearing), `eddoboozi` (voices) |
| Risk | Low. Both constructs contribute to the same clinical action (psychosis assessment referral). |

### Merge 2: Low mood + Anhedonia + Worthlessness/guilt → Low mood/Anhedonia

| Detail | Value |
|---|---|
| HiTOP Spectrum | Internalizing — Distress |
| Original counts | Low mood: ~30, Anhedonia: ~10, Worthlessness/guilt: ~9 |
| Merged count | 49 |
| Rationale | These three constructs correspond to PHQ-9 items 1, 2, 6 which together define the core depressive syndrome. The PHQ-9 was designed with these as a unitary construct — "little interest or pleasure" (anhedonia), "feeling down, depressed, or hopeless" (low mood), and "feeling bad about yourself" (worthlessness). In the embedding space, callers expressing "I don't enjoy anything anymore" and "I feel worthless" produce nearly identical representations because the conversational context (distress, loss, hopelessness) dominates the semantic signal. |
| Screening source | PHQ-9 items 1, 2, 6 |
| Luganda coverage | `ennaku` (sadness), `okwennyama`, `okwekawa` (self-deprecation), `akyawe` (no longer enjoys) |
| Risk | Moderate. In research contexts, distinguishing anhedonia from low mood has prognostic value (anhedonia predicts treatment resistance). However, transcript-level classification cannot reliably separate these without structured assessment. |

### Merge 3: Anxiety + Worry + Trouble relaxing + Palpitations + Fear of dying → Anxiety spectrum

| Detail | Value |
|---|---|
| HiTOP Spectrum | Internalizing — Fear |
| Original counts | Anxiety: ~25, Worry: ~15, Trouble relaxing: ~8, Palpitations: ~4, Fear of dying: ~3 |
| Merged count | 55 |
| Rationale | These five labels correspond directly to GAD-7 items 1-7, which measure a single latent construct (generalized anxiety). The GAD-7 was validated as a unidimensional scale — "feeling nervous, anxious, or on edge" (item 1), "not being able to stop or control worrying" (item 2), "trouble relaxing" (item 5), and the somatic accompaniments (palpitations, fear of dying). Keeping these separate with 3-8 examples each produced zero-valued clusters in BERTopic. |
| Screening source | GAD-7 items 1-7 |
| Luganda coverage | `okutya` (fear), `entiisa` (anxiety), `okweraliikirira` (worrying), `natiisa` (scared) |
| Risk | Low. All five labels trigger the same clinical response (anxiety assessment, possible referral to counselling). The somatic symptoms (palpitations, fear of dying) may indicate panic disorder specifically, but transcript-level data cannot distinguish GAD from panic without structured assessment. |

### Merge 4: Psychomotor retardation/agitation + Restlessness + Excessive energy + Excessive happiness → Psychomotor disturbance

| Detail | Value |
|---|---|
| HiTOP Spectrum | Externalizing / Mania |
| Original counts | Psychomotor: ~5, Restlessness: ~4, Excessive energy: ~3, Excessive happiness: ~2 |
| Merged count | 14 |
| Rationale | These labels capture motor and activation symptoms that span the psychomotor spectrum. In clinical practice, distinguishing agitation from restlessness from excessive energy requires direct observation — call transcripts provide only indirect evidence (speech rate, reported behavior). "Excessive happiness" (elevated mood/euphoria) is grouped here rather than with mood because it co-occurs with motor activation in mania. This remains the smallest clinical class (14 examples) and should be monitored for classifier performance. |
| Screening source | PHQ-9 item 8 (motor); YMRS items 2, 6 (activation) |
| Luganda coverage | `natandika` (restless), `kurwana` (fighting/agitated) |
| Risk | High. This merge conflates depressive psychomotor retardation with manic activation. These have opposite clinical implications. With more data, this class should be revisited and potentially split into "psychomotor retardation" (depression) and "psychomotor activation" (mania/agitation). |

### Merge 5: Isolation and withdrawal + Passivity phenomena → Isolation/Withdrawal

| Detail | Value |
|---|---|
| HiTOP Spectrum | Detachment |
| Original counts | Isolation: ~20, Passivity phenomena: ~4 |
| Merged count | 24 |
| Rationale | Passivity phenomena (feeling controlled by external forces, thought insertion/withdrawal) overlap behaviorally with social withdrawal — in both cases, the caller reports reduced agency and social engagement. In the HiTOP framework, both map to the Detachment spectrum. Passivity is a psychotic symptom that might better belong with Hallucination/Delusions, but the transcript-level language ("I don't go out", "I can't control it") clusters with withdrawal rather than with positive psychotic symptoms. |
| Screening source | PANSS negative subscale (blunted affect, withdrawal); PSQ item 5 (passivity) |
| Luganda coverage | `yekweka` (hiding), `yeesibye` (isolated), `okwekweka` (withdrawing) |
| Risk | Moderate. Passivity phenomena have different diagnostic implications (they suggest psychosis, not just social avoidance). With more annotated passivity examples, this should be re-evaluated. |

---

## Unchanged Classes (9 classes)

These labels had sufficient training examples and represent clinically distinct constructs that do not benefit from merging:

| Class | Examples | HiTOP Spectrum | Rationale for keeping separate |
|---|---|---|---|
| Insomnia/Hypersomnia | 130 | Internalizing | Distinct symptom domain (PHQ-9 item 3). High prevalence in the call population. Clear Luganda vocabulary (`tulo`, `okwebaka`). |
| Disorganized behaviors | 104 | Thought Disorder | Observable behavioral disruption distinct from speech/thought content. Includes violence, wandering, public nudity. |
| Functional impairment | 97 | Cross-cutting | Not a symptom but a consequence — measures impact on work, school, relationships. Kept as a separate dimension following the HiTOP principle that impairment is orthogonal to symptom severity. |
| Suicidal behaviour | 55 | Internalizing | Safety-critical. Must be tracked independently regardless of statistical considerations. Any merge would reduce clinical utility. |
| Appetite disturbance | 34 | Internalizing | Distinct somatic domain (PHQ-9 item 5). Sufficient data for stable clustering. |
| Disorganized speech | 32 | Thought Disorder | Distinct from disorganized behavior and over-talkativeness. Captures incoherence, rambling, tangentiality. |
| Concentration problems | 33 | Internalizing | Distinct cognitive domain (PHQ-9 item 7). Overlaps with disorganized speech but captures self-reported difficulty rather than observed communication disruption. |
| Fatigue/Low energy | 26 | Internalizing | Distinct somatic/energy domain (PHQ-9 item 4). Luganda has rich vocabulary for this (`obunafu`, `munafu`, `tayena`). |
| Over talkative | 26 | Thought Disorder / Mania | Pressured speech — distinct from disorganized speech (rapid but coherent vs. incoherent). Important mania indicator. |

---

## Non-Clinical Class

| Detail | Value |
|---|---|
| Examples | 300 (target) |
| Source | Agent-only segments from annotated files + transcription_v2 unlabeled data |
| Content | Greetings, logistics, call administration, counsellor advice, filler speech |
| Sampling | Random, seed=42, balanced across source files |

The Non-Clinical class serves two purposes:
1. **Training**: Gives the classifier a negative class to learn "this is not a symptom"
2. **Inference gate**: Combined with the clinical vocabulary regex, prevents the model from forcing symptom labels onto conversational speech

---

## Clinical Vocabulary Sources

The clinical vocabulary used for the inference gate and BERTopic seed topics draws from three sources:

| Source | Terms | Purpose |
|---|---|---|
| PHQ-9 (Kroenke et al., 2001) | 29 | Depression screening — validated across cultures including East Africa |
| GAD-7 (Spitzer et al., 2006) | 21 | Anxiety screening — unidimensional anxiety measure |
| PSQ (Bebbington & Nayani, 1995) | 26 | Psychosis screening — hallucinations, delusions, thought interference |
| Luganda clinical corpus | 50+ | Extracted from training data, cross-referenced with Luganda medical vocabulary |

Total unique terms after deduplication: **213** (organized by HiTOP class) + topic representation words from BERTopic.

---

## Limitations and Review Points

> [!WARNING]
> The following decisions should be specifically examined during clinical review:

### 1. Psychomotor Merge (High Priority)
The psychomotor disturbance class conflates retardation (depression) with activation (mania). These have **opposite treatment implications**. With 14 total examples, splitting is not currently feasible, but this should be the first class to split when more data is collected.

### 2. Passivity Phenomena Placement
Passivity phenomena were merged with Isolation/Withdrawal rather than Hallucination/Delusions. This was a data-driven decision (embedding similarity), not a clinical one. Passivity is diagnostically a first-rank psychotic symptom (Schneider's criteria) and may warrant reclassification.

### 3. Functional Impairment as a Class
The HiTOP framework treats impairment as a separate dimension, not a symptom. Some clinical reviewers may prefer to remove this as a classification target and instead use it as a severity modifier. Keeping it as a class was driven by its high prevalence (97 examples) and the clinical utility of detecting when callers report functional impact.

### 4. Luganda Vocabulary Completeness
The Luganda clinical terms were extracted from the training corpus rather than from a validated Luganda mental health instrument. A systematic review of Luganda clinical language for mental health, possibly using the Luganda PHQ-9 validation (Akena et al., 2013), would strengthen the vocabulary.

### 5. Ordinal Affect Risk Model
The affect risk scores (1=Unlikely, 2=Possible, 3=Likely) were assigned by clinicians without a standardized rubric. Inter-rater agreement on these scores was not explicitly measured. The model's adjacency accuracy (81-92%) should be interpreted in this context — the model may be learning rater behavior as much as clinical ground truth.

---

## References

- Kotov, R., et al. (2017). The Hierarchical Taxonomy of Psychopathology (HiTOP): A dimensional alternative to traditional nosologies. *Journal of Abnormal Psychology*, 126(4), 454-477.
- Kotov, R., et al. (2021). The Hierarchical Taxonomy of Psychopathology (HiTOP): A quantitative nosology based on consensus of evidence. *Annual Review of Clinical Psychology*, 17, 83-108.
- Kroenke, K., Spitzer, R.L., & Williams, J.B. (2001). The PHQ-9: Validity of a brief depression severity measure. *Journal of General Internal Medicine*, 16(9), 606-613.
- Spitzer, R.L., Kroenke, K., Williams, J.B., & Lowe, B. (2006). A brief measure for assessing generalized anxiety disorder: The GAD-7. *Archives of Internal Medicine*, 166(10), 1092-1097.
- Bebbington, P.E., & Nayani, T. (1995). The Psychosis Screening Questionnaire. *International Journal of Methods in Psychiatric Research*, 5(1), 11-19.
- Akena, D., et al. (2013). Sensitivity and specificity of the PHQ-9 for major depressive disorder in a Ugandan primary care setting. *BMC Psychiatry*, 13(1), 104.
