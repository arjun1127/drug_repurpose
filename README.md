# PrimeKG Drug Repurposing with a Bias-Aware Residual GCN

## Project Focus
This README is intentionally ML-first and documents exactly what we implemented in the GCN pipeline.

Backend/frontend exist for serving and visualization, but the core work is:
- leakage-safe graph construction
- disease-aware ranking with GCN embeddings
- hybrid ranking + calibration losses
- explicit hub-bias diagnostics and mitigation

## End-to-End Architecture
```text
PrimeKG CSV
  -> column normalization + entity mapping
  -> extract drug-disease relations by type
      positives: indication, off-label use
      typed negatives: contraindication
  -> split positives/contra into train/val/test
  -> build signed training graph
      +1: non-drug-disease edges
      +1: train therapeutic edges
      -1: train contraindication edges
  -> normalized sparse adjacency (for message passing)
  -> Residual GCN encoder (node + type embeddings)
  -> degree-aware link scorer
  -> loss = BPR + BCE + degree-correlation regularization
  -> eval: AUC/AP + ranking + bias/diversity metrics
  -> FastAPI inference (optional debias reranking)
```

## Dataset and Labeling Strategy
Source: Harvard Dataverse PrimeKG (`https://dataverse.harvard.edu/api/access/datafile/6180620`).

We do not treat all drug-disease links as the same. We explicitly separate relation types:
- Positive supervision: `indication`, `off-label use`
- Negative supervision: `contraindication`
- Unknown pairs: used as additional evaluation negatives (50% mix by default in val/test)

Latest run statistics from `models/training_metrics.json`:

| Item | Value |
| --- | --- |
| Therapeutic positives (total across splits) | 11,708 |
| Contraindication edges (total across splits) | 30,552 |
| Conflicting treat+contra pairs removed | 123 |
| Therapeutic drug whitelist | 2,074 drugs |
| Total drug nodes in graph | 7,898 drugs |
| Train/Val/Test positives | 9,368 / 1,170 / 1,170 |
| Train/Val/Test negatives | 24,442 / 3,510 / 3,510 |

## How We Tackled Problems (Problem -> Solution)
This is the actual decision path reflected in code.

| Problem observed | Why it mattered | What we changed |
| --- | --- | --- |
| Label leakage risk in message passing | If val/test drug-disease edges are in adjacency, model can indirectly "see answers" | Built adjacency from non-drug-disease edges + train-only therapeutic/contra edges |
| Generic hubs dominating rankings | High-degree drugs get high scores for many diseases | Added degree-aware scorer features, degree-correlation penalty, and inference-time prior subtraction |
| Binary classification alone was weak for ranking | AUC can look okay while top-k disease-specific ranking stays poor | Added BPR ranking loss as primary objective plus BCE for probability calibration |
| Easy negatives gave weak learning signal | Model needs hard "looks plausible but wrong" negatives | Added periodic hard negative mining and mixed hard + random BPR pairs |
| Non-therapeutic compounds surfaced as candidates | Many graph drug nodes are not practical therapeutics | Candidate whitelist built from indication/off-label/contra drug participation |
| Contradictory labels exist in raw KG | Same pair marked as treat and contraindication creates noisy supervision | Detected and removed 123 conflicting pairs from both sets |

## GCN Model Details
Implementation: `gnn_drug_repurposing_improved.py`

### 1. Node encoding
- Learned node embedding: `Embedding(num_nodes, hidden_dim)`
- Learned type embedding: `Embedding(num_types, hidden_dim)`
- Initial feature: `x0 = node_embed + type_embed`

### 2. Graph encoder
- `GraphConv(hidden_dim -> hidden_dim)` + ReLU
- 2 x ResidualGCN blocks:
  - GraphConv(hidden_dim -> hidden_dim)
  - LayerNorm
  - ReLU + Dropout
  - skip connection: `x + h`
- Final `GraphConv(hidden_dim -> embedding_dim)`

Default dimensions:
- `hidden_dim = 128`
- `embedding_dim = 64`
- `dropout = 0.2`

### 3. Signed normalized adjacency
Training graph is signed (+1/-1). Normalization uses absolute values in degree term for stability:
- signed edge weights are preserved in message passing
- self-loops are added

### 4. Link scorer
For pair `(drug, disease)` features are:
- drug embedding `z_src`
- disease embedding `z_dst`
- elementwise interaction `z_src * z_dst`
- `log(deg_src)` and `log(deg_dst)`

MLP:
- `Linear(3*embedding_dim + 2 -> embedding_dim)`
- `ReLU -> BatchNorm1d -> Dropout`
- `Linear(embedding_dim -> 1)` (logit)

## Loss Functions and Training Objective
We train with a hybrid objective:

`L = w_bpr * L_bpr + w_bce * L_bce + lambda_deg * L_deg_corr`

Current weights:
- `w_bpr = 1.0`
- `w_bce = 0.3`
- `lambda_deg = 0.02`

### A) BPR ranking loss (primary)
For each disease, enforce positive drug score > negative drug score:

`L_bpr = - mean(log(sigmoid(s_pos - s_neg)))`

This directly optimizes ranking quality (MRR/Hits@k behavior), not just classification.

### B) BCE calibration loss
Standard binary cross-entropy on train positive + typed negative pairs:

`L_bce = BCEWithLogits(logits, labels)`

This keeps output scores calibrated and usable as confidence-like probabilities.

### C) Degree-correlation regularizer
Compute absolute correlation between predicted scores and log drug degree, then penalize it:

`L_deg_corr = abs(corr(sigmoid(logits), log(degree_src)))`

This pushes the model away from using degree as a shortcut.

### Optimizer and schedule
- Optimizer: AdamW (`lr=1e-3`, `weight_decay=1e-5`)
- LR scheduler: ReduceLROnPlateau on validation MRR
- Gradient clipping: `max_norm=1.0`
- Early stopping patience: `15` eval steps
- Eval frequency: every `5` epochs

### Hard negative mining
After warmup (`hard_neg_start_epoch=20`), every `10` epochs:
- score candidate negatives for disease
- pick top false-positive drugs (hard negatives)
- mix with random negatives (`hard_neg_fraction=0.5`)

## Hub Bias: Solved or Not?
Short answer: partially mitigated, not fully solved.

Latest bias metrics:

| Bias metric | Value | Interpretation |
| --- | --- | --- |
| Spearman rho(degree, mean score) | `-0.2123` | No longer strongly positive hub correlation |
| Mean top-k Jaccard across diseases | `0.1591` | Better diversity overall |
| P90 top-k Jaccard | `0.5385` | Some disease pairs still share too many drugs |
| Top-1 mode fraction | `0.2667` | Same top drug still appears for ~26.7% diseases |
| Most common top-1 drug | `Zinc acetate` | Residual popularity bias remains |

Why "partial":
- We reduced direct degree-score coupling.
- But top-1 concentration is still high, so disease-specificity is not fully reliable at the very top.

Inference also applies optional debias reranking in backend:

`rank_score = raw_score - alpha * global_prior_centered`

where `global_prior_centered` is average drug score across sampled diseases, centered by global mean.

## Results (Latest Saved Run)
### Main metrics
| Metric | Value |
| --- | --- |
| Best validation MRR | `0.0250` at epoch `60` |
| Test loss | `0.6286` |
| Test AUC | `0.7727` |
| Test AP | `0.4937` |
| Test MRR | `0.0299` |
| Test Hits@1 | `0.0094` |
| Test Hits@5 | `0.0368` |
| Test Hits@10 | `0.0667` |
| Test Precision@10 | `0.0126` |
| Test Recall@10 | `0.0636` |

### Degree-stratified test performance
| Bucket | Count | AUC | AP |
| --- | --- | --- | --- |
| Low degree (`<= q33`, q33=290) | 1,597 | 0.7553 | 0.3542 |
| Medium (`q33 < deg <= q66`, q66=1140.44) | 1,346 | 0.7663 | 0.5053 |
| High (`> q66`) | 1,737 | 0.8437 | 0.7002 |

Interpretation:
- Performance is still better for high-degree drugs.
- This confirms remaining long-tail difficulty even after debiasing work.

## Sample Testing (Qualitative)
Provided sample output table:

| Disease | Top 1 | Top 2 | Top 3 |
| --- | --- | --- | --- |
| Colonic Neoplasm | Butacaine | Zinc gluconate | Isoleucine |
| Chronic Lymphocytic Leukemia / SLL | Cerliponase alfa | Tenonitrozole | Tavaborole |
| Malignant Hypertension | Prednisolone | Cortisone acetate | Procarbazine |
| Type 2 Diabetes Mellitus | Cortisone acetate | Prednisolone | Dexamethasone |
| Sitosterolemia | Darbepoetin alfa | Cerliponase alfa | Tenonitrozole |
| Insomnia | Darbepoetin alfa | Cortisone acetate | Prednisolone |
| Lyme Disease | Cortisone acetate | Prednisolone | Procarbazine |
| Multidrug-Resistant Tuberculosis | Quizartinib | Ampicillin | Uracil mustard |
| HIV Infectious Disease | Dexamethasone | Cortisone acetate | Triamcinolone |
| Acute Lymphoblastic Leukemia | Darbepoetin alfa | Procarbazine | Doxycycline |

## Why Some Predictions Look Wrong
This is important and expected in KG-based repurposing.

### 1) Dataset incompleteness (major reason)
In PrimeKG, missing edge does not mean true negative. Many drug-disease pairs are simply unlabeled.

Effect:
- model may rank biologically plausible but unvalidated or noisy candidates
- evaluation with unknown negatives can penalize true-but-missing positives

### 2) Relation granularity mismatch
We train on relation types (`indication`, `off-label`, `contraindication`) but not dosage, disease stage, subtype, or patient context.

Effect:
- drugs useful in one subtype/context can be over-generalized to another

### 3) Strong neighborhood transfer from hubs
Even after debiasing, high-connectivity anti-inflammatory/oncology-adjacent drugs can still transfer to many diseases.

Effect:
- repeated steroid-like or broad-acting drugs in unrelated conditions

### 4) Candidate filtering changes what appears
In API inference, defaults are:
- `exclude_contraindicated = true`
- `exclude_known_treatments = true`

So approved first-line treatments are intentionally removed to surface novel candidates.

Effect:
- top predictions can look "wrong" clinically because known correct drugs were filtered out on purpose

### 5) Limited therapeutic drug whitelist
Only 2,074/7,898 drug nodes are used as therapeutic candidates. This improves realism but also narrows candidate diversity.

Effect:
- long-tail diseases can map to suboptimal remaining candidates

### 6) Weak supervision per disease
Many diseases have few positive edges, making disease-specific ranking hard.

Effect:
- easier to overfit to broad global priors than fine disease signals

## Backend and Frontend (Brief)
- Backend: FastAPI serves `/predict`, `/metrics`, and plot endpoints; loads model once and precomputes embeddings.
- Frontend: React/Vite app with discovery, performance, and bias tabs for interactive inspection.

## Reproduce
### Activate the env in the backend folder 
```linux
souce backend/venv/bin/activate
```
### Train
```bash
python3 gnn_drug_repurposing_improved.py --device auto --epochs 200
```

### Backend
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

## Artifacts
Training outputs are saved under `models/`:
- `gnn_drug_repurposing.pt`
- `adjacency.pt`
- `degrees.pt`
- `metadata.pkl`
- `training_metrics.json`
- `plots/*.png`
