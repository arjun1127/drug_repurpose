# Graph Neural Network for Drug Repurposing

## Project Overview

A full-stack **Graph Neural Network (GNN)** drug repurposing system built on the **PrimeKG Knowledge Graph**. The pipeline trains a leakage-safe, bias-aware GCN model to predict novel drug-disease associations, served through a **FastAPI** backend and visualized via a **React + Vite** frontend with interactive diagnostics.

> **Key Contribution**: This project identifies and fixes critical hub bias in GNN-based drug repurposing — where high-degree drugs (like Dexamethasone) dominate predictions regardless of disease — through inverse-degree negative sampling, degree-aware scoring, and correlation regularization.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Anti-Bias Pipeline](#anti-bias-pipeline)
5. [Training & Evaluation](#training--evaluation)
6. [Backend API](#backend-api)
7. [Frontend Dashboard](#frontend-dashboard)
8. [Installation & Setup](#installation--setup)
9. [Usage](#usage)
10. [Results](#results)
11. [File Structure](#file-structure)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   GNN Drug Repurposing System                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. DATA PIPELINE                                                │
│     └─ PrimeKG Knowledge Graph (Harvard Dataverse)               │
│        ├─ ~130K nodes (drug, disease, protein, gene, etc.)       │
│        ├─ ~4M edges (biomedical relationships)                   │
│        └─ Standardized column mapping                            │
│                                                                  │
│  2. LEAKAGE-SAFE GRAPH CONSTRUCTION                              │
│     ├─ Remove val/test drug-disease edges from adjacency         │
│     ├─ Symmetric normalization: Â = D^(-½) A D^(-½)             │
│     └─ Degree-aware inverse-sqrt negative sampling               │
│                                                                  │
│  3. RESIDUAL GCN MODEL (3-layer)                                 │
│     ├─ Input GCN: hidden_dim → hidden_dim                        │
│     ├─ 2× Residual GCN + LayerNorm + Dropout                    │
│     ├─ Output GCN: hidden_dim → embedding_dim                   │
│     └─ Link Predictor: [src, dst, src⊙dst, log_deg] → score    │
│                                                                  │
│  4. TRAINING LOOP                                                │
│     ├─ BCE loss + degree correlation regularizer (λ=0.1)         │
│     ├─ ReduceLROnPlateau scheduler + gradient clipping           │
│     └─ Early stopping on validation MRR (patience=15)            │
│                                                                  │
│  5. EVALUATION SUITE                                             │
│     ├─ AUC, AP, Hits@K, MRR, Precision@K, Recall@K              │
│     ├─ Degree-stratified metrics (low/medium/high)               │
│     ├─ Spearman ρ (degree vs. score correlation)                 │
│     └─ Jaccard diversity (top-K overlap across diseases)         │
│                                                                  │
│  6. DEPLOYMENT                                                   │
│     ├─ FastAPI backend (model inference + metrics API)            │
│     └─ React frontend (3-tab dashboard)                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Dataset

### PrimeKG Knowledge Graph

| Property | Value |
|----------|-------|
| **Source** | Harvard Dataverse |
| **URL** | `https://dataverse.harvard.edu/api/access/datafile/6180620` |
| **Format** | CSV (auto-downloaded on first run) |
| **Nodes** | ~129K (drug, disease, protein, gene, anatomy, etc.) |
| **Edges** | ~4M biomedical relationships |
| **Drug-Disease Positives** | ~42,383 unique pairs |

### Entity Types
- **Drug**: Chemical compounds with known pharmacological effects
- **Disease**: Medical conditions and pathologies
- **Protein/Gene**: Molecular targets and genetic information

### Column Standardization
The pipeline auto-detects column names across PrimeKG versions:
- `x_id` / `x_index` → `source_id`
- `x_type` / `source_type` → `source_type`
- `y_id` / `y_index` → `target_id`
- `y_type` / `target_type` → `target_type`

---

## Model Architecture

### 3-Layer Residual GCN

```
Node Index ──→ Embedding(num_nodes, 128)  ─┐
                                            ├──→ Add ──→ GCN_in(128→128) + ReLU
Node Type ──→ Embedding(num_types, 128)  ──┘               │
                                                            ▼
                                                  ResidualGCN × 2
                                                  ├─ GCN(128→128)
                                                  ├─ LayerNorm
                                                  ├─ ReLU + Dropout(0.2)
                                                  └─ Skip Connection (x + h)
                                                            │
                                                            ▼
                                                    GCN_out(128→64)
                                                            │
                                                     Node Embeddings (64-dim)
```

### Degree-Aware Link Predictor

```
Input: [z_src, z_dst, z_src ⊙ z_dst, log(deg_src), log(deg_dst)]
       ─── 64 + 64 + 64 + 1 + 1 = 194 features ───
                        │
                Linear(194 → 64) + ReLU
                        │
                 BatchNorm1d(64)
                        │
                   Dropout(0.2)
                        │
                  Linear(64 → 1)
                        │
                     Logit Score
```

**Why degree features?** By giving the model explicit access to `log(degree)`, it can learn to discount the influence of node popularity rather than using it as a shortcut.

### Configuration

| Parameter | Value |
|-----------|-------|
| Hidden Dimension | 128 |
| Embedding Dimension | 64 |
| GCN Layers | 3 (1 input + 2 residual) |
| Dropout | 0.2 |
| Learning Rate | 1e-3 (AdamW) |
| Weight Decay | 1e-5 |
| Negative Ratio | 3:1 |
| Early Stopping Patience | 15 epochs |
| Gradient Clip Norm | 1.0 |
| Degree Correlation λ | 0.1 |

---

## Anti-Bias Pipeline

### Critical Bugs Fixed from Original Model

| Bug | Problem | Fix |
|-----|---------|-----|
| **Test Edge Leakage** | Adjacency included test drug-disease edges during training — model saw answers | Removed val/test edges from message-passing graph |
| **Uniform Negative Sampling** | `random.choice()` gave high-degree drugs too few negatives | Inverse-sqrt degree-weighted sampling: P(drug) ∝ degree^(-0.5) |
| **1:1 Negative Ratio** | Equal positives and negatives; real space is >99% negative | 3:1 negative ratio |
| **No Over-Smoothing Prevention** | 2-layer GCN without residuals; embeddings collapse | 3-layer GCN with skip connections + LayerNorm |
| **No Early Stopping** | Fixed 100 epochs, no validation monitoring | Early stopping on val MRR with patience=15 |
| **No LR Scheduling** | Constant 1e-3 for all epochs | ReduceLROnPlateau (factor=0.5, patience=5) |
| **AUC-Only Evaluation** | Single metric misses hub bias | Full suite: AUC, AP, MRR, Hits@K, Spearman ρ, Jaccard |

### Countermeasures Applied

1. **Inverse-degree negative sampling** — High-degree drugs get proportionally more negatives
2. **Degree correlation loss** — Penalty term: λ·|corr(scores, log_degree)| added to BCE
3. **Degree-aware scoring** — Link predictor receives `log(degree)` as explicit features
4. **Residual GCN** — Skip connections prevent over-smoothing of embeddings
5. **Leakage-safe splits** — Val/test drug-disease edges removed from training adjacency

---

## Training & Evaluation

### Data Splits

| Split | Positives | Negatives (3:1) |
|-------|-----------|-----------------|
| Train | 33,907 | 101,721 |
| Validation | 4,238 | 12,714 |
| Test | 4,238 | 12,714 |

### Training Process

```
For each epoch:
  1. Forward: encode(node_type_ids, adjacency) → embeddings
  2. Score: link_predictor(src, dst, src⊙dst, log_deg) → logits
  3. Loss: BCE(logits, labels) + λ·|corr(scores, degree)|
  4. Backward: gradient clipping at norm=1.0
  5. Optimize: AdamW step
  6. Every 5 epochs: validate on held-out set
  7. Track best val MRR → save checkpoint
  8. Early stop if no improvement for 15 eval cycles
```

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **ROC-AUC** | Link prediction discrimination |
| **Average Precision** | Ranking quality under class imbalance |
| **MRR** | Mean reciprocal rank of true drugs |
| **Hits@1/5/10** | Fraction of true drugs in top-K |
| **Precision@K / Recall@K** | Per-disease ranking precision & recall |
| **Spearman ρ** | Degree vs. score correlation (bias metric) |
| **Jaccard Diversity** | Top-K overlap across diseases (diversity metric) |
| **Degree-Stratified AUC** | AUC breakdown by drug degree bucket |

### Diagnostic Plots (saved to `models/plots/`)

| Plot | Purpose |
|------|---------|
| `training_curves.png` | Train/val loss + MRR over epochs |
| `degree_distribution.png` | Drug degree histogram in PrimeKG |
| `degree_vs_score.png` | Scatter of degree vs. mean predicted score |
| `roc_pr_curves.png` | ROC and Precision-Recall curves on test set |
| `degree_stratified_metrics.png` | AUC by degree bucket (low/medium/high) |
| `topk_diversity.png` | Jaccard similarity distribution across disease pairs |

---

## Backend API

### Technology: FastAPI + PyTorch

The backend loads the trained model checkpoint, precomputes embeddings once at startup, and serves predictions via REST endpoints.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check + model loaded status |
| `GET` | `/metrics` | Full training metrics JSON (config, test results, bias analysis, history) |
| `GET` | `/plots-list` | List of available diagnostic plot filenames |
| `GET` | `/plots/{filename}` | Static serving of plot images |
| `POST` | `/predict` | Drug predictions for a disease query |

### Prediction Request

```json
POST /predict
{
  "disease": "cancer",
  "top_k": 12
}
```

### Prediction Response

```json
{
  "disease": "malignant neoplasm of breast",
  "targets": [
    {"name": "TP53", "pdb_id": "1TUP", "url": "http://..."}
  ],
  "predictions": [
    {
      "drug_name": "Dexamethasone",
      "gnn_score": 0.9821,
      "degree": 742,
      "degree_bucket": "high",
      "ligand_url": null
    }
  ]
}
```

---

## Frontend Dashboard

### Technology: React 19 + Vite + TypeScript + Framer Motion

A 3-tab dashboard with glassmorphism design and animated transitions:

### Tab 1: Drug Discovery
- Search bar for disease queries
- Drug prediction cards with GNN score + degree info
- Target protein visualization (3Dmol.js PDB viewer)
- Optional 3D ligand structure viewer (SDF format)

### Tab 2: Model Performance
- Key metrics hero cards (Test AUC, AP, Best Epoch, MRR, Hits@10)
- Training configuration table
- All 6 diagnostic plots displayed in a responsive grid

### Tab 3: Bias Analysis
- Spearman ρ correlation card with severity coloring
- Top-1 mode fraction (same drug ranking #1 across diseases)
- Mean/P90 Jaccard similarity metrics
- Degree-stratified AUC bar chart (low/medium/high)
- Changelog of all anti-bias countermeasures implemented

---

## Installation & Setup

### Prerequisites

```
Python 3.10+
Node.js 18+
```

### 1. Clone & Setup Backend

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Train the Model

```bash
# From project root, with venv activated:
python3 gnn_drug_repurposing_improved.py --device auto --epochs 200

# Options:
#   --hidden-dim 128     (default)
#   --embedding-dim 64   (default)
#   --negative-ratio 3.0 (default)
#   --eval-every 5       (default)
#   --run-tsne           (optional, uses extra memory)
```

This produces artifacts in `models/`:
- `gnn_drug_repurposing.pt` — Model checkpoint
- `adjacency.pt` — Normalized adjacency matrix
- `degrees.pt` — Node degree tensor
- `metadata.pkl` — Node maps and entity names
- `training_metrics.json` — All metrics and training history
- `plots/` — 6 diagnostic PNG plots

### 3. Start Backend

```bash
cd backend
source venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 4. Start Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` in browser.

### Backend Dependencies (`backend/requirements.txt`)

```
fastapi
uvicorn
torch
numpy
pandas
pydantic
scikit-learn
python-dotenv
```

### Frontend Dependencies

```
react, react-dom, axios, framer-motion, lucide-react, 3dmol, vite, typescript
```

---

## Usage

### 1. Drug Discovery

Open the frontend → **Drug Discovery** tab → type a disease name → click **Discover**.

Example queries: `cancer`, `diabetes`, `anemia`, `leukemia`, `parkinson`

### 2. Review Model Performance

Click the **Model Performance** tab to see:
- Test AUC, Average Precision, Best Epoch, MRR, Hits@10
- Full training configuration
- 6 diagnostic plots

### 3. Analyze Bias

Click **Bias Analysis** tab to see:
- Spearman ρ (degree-score correlation)
- Top-1 mode fraction
- Jaccard diversity metrics
- Degree-stratified AUC breakdown
- Complete list of anti-bias countermeasures

---

## Results

### Training Results (200 epochs, early stopped at epoch 125)

| Metric | Value |
|--------|-------|
| Test ROC-AUC | 0.994 |
| Test Average Precision | 0.989 |
| Test MRR | 0.025 |
| Test Hits@10 | 5.8% |
| Best Validation MRR | 0.029 (epoch 125) |

### Bias Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Spearman ρ | 0.884 | < 0.40 | ⚠️ Remaining bias |
| Mean Jaccard | 0.543 | < 0.25 | ⚠️ Moderate overlap |
| Top-1 Mode Fraction | 36.7% | < 10% | ⚠️ Dexamethasone dominates |
| P90 Jaccard | 0.818 | < 0.50 | ⚠️ High overlap at P90 |

### Degree-Stratified AUC

| Bucket | AUC | Count |
|--------|-----|-------|
| Low (degree ≤ 1) | 0.694 | 9,446 |
| Medium (1 < degree ≤ 255) | 0.980 | 3,685 |
| High (degree > 255) | 0.962 | 3,821 |

> **Note**: Hub bias remains significant despite countermeasures. The low-degree AUC of 0.694 vs high-degree AUC of 0.962 shows the model still performs better for well-connected drugs. Further work on graph augmentation, attention-based aggregation, or contrastive learning could improve this.

---

## File Structure

```
majorProj/
├── gnn_drug_repurposing_improved.py   # Training pipeline (memory-optimized)
├── gnn_drug_repurposing_old.py        # Original script (backup)
├── evaluate_model_bias.py             # Standalone bias evaluation
├── prepare_10_diseases.py             # Pre-compute disease assets
├── README.md                          # This file
│
├── backend/
│   ├── main.py                        # FastAPI server
│   ├── requirements.txt               # Python dependencies
│   ├── .env                           # BASE_URL config
│   ├── run.sh                         # Startup script
│   └── venv/                          # Python virtual environment
│
├── frontend/
│   ├── src/
│   │   ├── App.tsx                    # Main 3-tab dashboard
│   │   ├── index.css                  # Design system (glassmorphism)
│   │   ├── main.tsx                   # Entry point
│   │   └── components/
│   │       └── MolecularViewer.tsx    # 3Dmol.js wrapper
│   ├── package.json
│   ├── vite.config.ts
│   └── .env                          # VITE_API_URL config
│
├── data/
│   ├── primekg.csv                    # PrimeKG dataset (auto-downloaded)
│   ├── pdb/                           # Protein structure files
│   └── ligands/                       # Drug SDF files
│
├── models/
│   ├── gnn_drug_repurposing.pt        # Trained model checkpoint
│   ├── adjacency.pt                   # Sparse adjacency matrix
│   ├── degrees.pt                     # Node degree tensor
│   ├── metadata.pkl                   # Node maps + entity names
│   ├── training_metrics.json          # Full metrics + history
│   └── plots/                         # Diagnostic visualizations
│       ├── training_curves.png
│       ├── degree_distribution.png
│       ├── degree_vs_score.png
│       ├── roc_pr_curves.png
│       ├── degree_stratified_metrics.png
│       └── topk_diversity.png
│
└── backup/                            # Project backup
```

---

## Technical Notes

### Memory Optimization
The training script is optimized for systems with **8GB RAM**:
- Raw DataFrame freed immediately after name extraction (~600MB saved)
- Standardized DataFrame freed after node artifact construction
- Edge index arrays freed after adjacency is built
- Explicit `gc.collect()` at strategic points
- Static adjacency used during training (no per-epoch rebuild)

### Reproducibility
- Seed: 42 (set for Python, NumPy, and PyTorch)
- Fixed train/val/test splits via seeded RNG
- Deterministic negative sampling

---

## References

- **PrimeKG**: Chandak et al., "Building a knowledge graph to enable precision medicine" (Scientific Data, 2023)
- **GCN**: Kipf & Welling, "Semi-Supervised Classification with Graph Convolutional Networks" (ICLR 2017)
- **Link Prediction**: Zhang & Chen, "Link Prediction Based on Graph Neural Networks" (NeurIPS 2018)
- **DropEdge**: Rong et al., "DropEdge: Towards Deep Graph Convolutional Networks on Node Classification" (ICLR 2020)

---

## License & Attribution

This project uses publicly available datasets (PrimeKG from Harvard Dataverse) and open-source libraries. Ensure proper attribution when publishing results.
