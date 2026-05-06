# Synthetic Note Fidelity Review
**Method:** Adversarial comparison of 8 synthetic notes against 3 real de-identified clinical transcriptions pulled from MTSamples (Hematology-Oncology category). Sources:
- `real_001_breast_cancer_consult.txt` — surgical consult, invasive ductal carcinoma
- `real_002_hematology_consult_dvt_pe.txt` — hematology consult, DVT/PE with renal infarct
- `real_003_discharge_mesothelioma.txt` — discharge summary, epithelioid malignant mesothelioma

---

## Verdict summary

| Finding | Severity | All 8 notes affected? |
|---|---|---|
| Hyper-structured template format | High | Yes |
| No comorbidities or unrelated medications | High | Yes |
| No allergies section | Medium | Yes (7/8) |
| Staging cited with AJCC edition inline | Medium | Yes |
| No uncertainty / hedge language | High | Yes |
| Drug doses as textbook mg/m² — no BSA calc | Low-Medium | Yes |
| Biomarkers present at wrong visit type | High | 3/8 |
| No narrative physical exam | Medium | Yes |
| No social history detail | Medium | Yes |
| Note type monoculture (all initial consults) | High | Yes |
| Missing prior surgical/treatment history | High | Yes |
| ECOG distribution unrealistically skewed 0-1 | Medium | Yes |

**Bottom line:** The synthetic notes are good for testing whether an extractor can identify named entities in a clean note. They are poor for testing whether an extractor handles the noise, ambiguity, and missing data that dominates real clinical documentation. An extractor trained or validated against these notes only will likely fail on actual Ontada notes.

---

## Finding 1 — Format is a textbook template, not a clinic note

**What real notes look like:**

The MTSamples breast cancer consult has no `DIAGNOSIS:` or `TREATMENT PLAN:` header. The diagnosis appears inline inside `IMPRESSION:` as an abbreviated string: *"T1c, Nx, M0 left breast carcinoma."* The plan appears inside `RECOMMENDATIONS:` as a paragraph, not a structured field. The hematology consult lists its impression as a numbered differential, not a single confirmed entity.

**What synthetic notes look like:**

Every synthetic note has: `DIAGNOSIS:` → staged + biomarker block → `PERFORMANCE STATUS:` → `TREATMENT PLAN:` with full regimen → `LABS:`. This is a test scaffold, not a clinic note. No real practitioner dictates in this format.

**Why it matters for the extractor:**
Entity extraction on clean structured headers is a solved problem. The Ontada challenge is extracting staging from narrative impressions, from discharge summaries where the cancer is mentioned incidentally, from follow-up notes that assume you read the prior note. None of that appears here.

**Fix:** At minimum, 3–4 notes should use narrative impressions without labeled sections. 2 notes should be follow-up visits that reference a prior staging workup without restating it.

---

## Finding 2 — No comorbidities, no unrelated medications

**What real notes look like:**

- Breast cancer consult patient: prior benign breast biopsy (1972), three laparotomies, right oophorectomy, ganglion cyst removal. Current medication: omeprazole. No allergies.
- Mesothelioma discharge: Coumadin 1 mg daily (INR 2.3), amiodarone 100 mg daily. Prior pericardectomy, pericardial window, mesenteric thrombosis. Three prior surgeries listed by date.

The mesothelioma note's anticoagulation is clinically significant for the case — it delayed the thoracentesis and drove the Lovenox bridging decision. Real notes have this kind of entanglement between comorbidity and cancer management.

**What synthetic notes look like:**

Labs listed are exclusively cancer-relevant (CEA, CA 19-9, LDH, PSA). Medications section is absent or lists only the cancer treatment drugs. No prior surgeries. No chronic disease management. The patients are conceptually healthy except for their cancer.

**Why it matters:**
The extractor needs to distinguish "current treatment regimen" from "medications for other conditions." It needs to handle a note where the cancer appears 400 words in, after resolving a Coumadin interaction. These notes don't test that at all.

**Fix:** Add comorbidities to at least 4 synthetic notes. Good realistic additions: hypertension + metformin, COPD on tiotropium, prior DVT on apixaban, hypothyroidism on levothyroxine. The mesothelioma patient (SYN-007 pancreas) is a natural fit for diabetes given pancreatic involvement.

---

## Finding 3 — No uncertainty or hedge language

**What real notes look like:**

From the breast cancer consult: *"The MRI scan showed some close involvement possibly involving the left pectoralis muscle, although thought to also possibly represent biopsy artifact."*

From the mesothelioma note: *"A CT of the abdomen and pelvis revealed normal-appearing liver, spleen, and pancreas; however, the right kidney appeared smaller compared to left and suggesting possibility of renal infarct."*

Real clinical language is hedged — "possible," "suggesting," "thought to represent," "cannot exclude," "consistent with." These qualifiers are diagnostically meaningful (they affect staging confidence and treatment decisions) and are a genuine extraction challenge.

**What synthetic notes look like:**

Every finding is definitive: *"KRAS G12D mutation. NRAS wild-type. BRAF wild-type."* No qualifier language anywhere across all 8 notes.

**Why it matters:**
An extractor that sees "EGFR mutation negative" in a synthetic note and learns to return `egfr_status: negative` will fail on *"EGFR mutation status pending — reflexed to external lab"* or *"EGFR status equivocal, repeat testing recommended."* These are common real-world note states.

**Fix:** Add at least one "pending" biomarker field per note (e.g., staging complete but molecular testing pending). Add hedge language to at least 3 imaging findings.

---

## Finding 4 — AJCC edition citation is textbook language, not clinical language

**What real notes look like:**

The breast cancer consult: *"T1c, Nx, M0 left breast carcinoma"* — just the TNM string, no edition. The mesothelioma discharge: *"stage III disease"* with no staging system cited at all. The hematology consult doesn't stage anything.

**What synthetic notes look like:**

Every note: *"T2bN2M0 per AJCC 8th edition"* or similar. The "per AJCC 8th edition" suffix appears verbatim in 7 of 8 notes.

**Why it matters:**
Low practical severity for extraction — the TNM string itself is extractable regardless. But it signals that all 8 notes were written by the same process (a prompted LLM completing a template), which means they share whatever failure modes that process has. Notes that look identical in structure will expose the same extractor bugs and hide others.

**Fix:** Minor. Remove the "per AJCC Xth edition" suffix in half the notes. Use "stage III" without TNM in 2 notes (common in hematologic malignancies, where it refers to Ann Arbor or Lugano).

---

## Finding 5 — Note type monoculture

**What real notes look like:**

MTSamples has: surgical consults, discharge summaries, follow-up SOAP notes, hematology consults, procedure notes (mediport placement). The breast cancer consult doesn't include a treatment plan — it ends at surgical recommendation, with "medical oncologist has discussed with her issues about adjuvant or neoadjuvant chemotherapy" as a handoff.

**What synthetic notes look like:**

All 8 are the same note type: a single-visit summary that includes HPI, diagnosis with full staging + biomarkers, treatment plan with specific doses, and labs. No follow-ups. No discharge summaries. No procedure notes. No handoffs between specialties.

**Why it matters:**
In a real Ontada pipeline, many notes will be follow-up visits that say: *"Patient presents for cycle 4 of FOLFOX. Tolerating well. CA 19-9 down from 88 to 32. Continuing per plan."* The staging doesn't appear. The extractor has to either retrieve it from a prior note or mark it as missing. These synthetic notes cannot surface that challenge at all.

**Fix:** Replace 2–3 synthetic notes with: one follow-up visit (restating treatment in progress but not re-stating diagnosis), one discharge summary (narrative hospital course), one tumor board note (multi-disciplinary discussion without a single clinician's plan).

---

## Finding 6 — Biomarkers front-loaded at wrong visit

**What real notes look like:**

The 2007 breast cancer consult has no ER/PR/HER2, no Ki-67, no Oncotype DX. Why? The consult is pre-surgical. Those results come from the surgical specimen, 1–2 weeks post-op. A real initial consult often has only the core biopsy histology grade and perhaps ER/PR if run on biopsy tissue. Oncotype DX takes another week after that.

**What synthetic notes look like:**

SYN-002 (breast): "ER 95% positive, PR 80% positive, HER2 1+ by IHC (negative). Ki-67 22%. Oncotype DX recurrence score 18" — all in the same note as the surgical plan. This is a composite of 3 different reports collapsed into one note.

**Why it matters:**
In real notes, biomarker fields are frequently absent or partially present. "ER positive, HER2 pending" is the common pre-treatment state. An extractor that sees fully populated biomarker blocks in training will over-confidently return nulls when it should return "pending" or "unknown."

**Fix:** In SYN-002, remove Oncotype DX (it wouldn't be ordered until post-surgical ER+ confirmation). Mark HER2 as "IHC 2+, FISH reflex pending" — a common real-world ambiguous state.

---

## Finding 7 — Missing physical exam and social history

**What real notes look like:**

Real notes have full physical exams: vitals, head-to-toe with findings. The mesothelioma note has vitals block with BP, HR, temp, O2 sat. The breast cancer consult has a 7-section PE. Social history includes employment (school teacher), marital status, specific occupational exposure (post office — relevant for mesothelioma).

**What synthetic notes look like:**

Most have no physical exam at all. Social history is absent in all 8 notes. Performance status appears as a single-line ECOG score, not derived from the note narrative.

**Why it matters:**
For real Ontada extraction, ECOG is often inferred from functional status language in the note ("patient ambulates independently, employed full-time" → ECOG 0) rather than explicitly labeled. Notes without a PE and social history train an extractor that will only fire on explicit ECOG labels.

**Fix:** Add a 3-5 line PE block and 2-sentence social history to at least 4 synthetic notes. Include at least one case where ECOG is inferrable from narrative but not explicitly labeled.

---

## Recommended action

**Immediate (improves eval quality significantly, 2–3 hours of work):**
1. Add comorbidity + unrelated medications to SYN-003, SYN-004, SYN-007
2. Add "pending" biomarker state to SYN-001, SYN-002, SYN-007
3. Convert SYN-005 (Hodgkin) to a follow-up after cycle 2 PET assessment
4. Convert SYN-008 (GBM) to a discharge summary (natural fit — surgery just happened)

**Longer term (more realistic corpus):**
5. Add 4 real MTSamples notes to the corpus (already in `data/real_notes/`)
6. Add `data/gold_standard/` JSON files for the real notes with manually labeled entities — these become your true adversarial eval set where the extractor has no home-field advantage from training on similar synthetic text

---

## Real notes now in corpus

Saved to `data/real_notes/`:
- `real_001_breast_cancer_consult.txt` — surgical consult, no staging edition, no biomarkers, rich PMH
- `real_002_hematology_consult_dvt_pe.txt` — not a cancer note; tests extractor discrimination (should return nulls for staging/regimen)
- `real_003_discharge_mesothelioma.txt` — discharge summary, narrative course, anticoagulation complexity, stage III no TNM

Source: [MTSamples Hematology-Oncology](https://www.mtsamples.com/site/pages/browse.asp?Type=96-Hematology+-+Oncology) — contributed de-identified transcription examples, free for educational use.
