# Graph Neural Network for Drug Repurposing

## Project Overview

This project implements a comprehensive **Graph Neural Network (GNN)** system for drug repurposing using the **PrimeKG Knowledge Graph**. It combines deep learning predictions with molecular docking validation to identify potential drug candidates for diseases.

The system leverages biomedical entity relationships (drugs, diseases, proteins, genes) to make predictive links between drugs and diseases, validated through actual molecular binding simulations using AutoDock Vina.

---

## Table of Contents

1. [Project Architecture](#project-architecture)
2. [Datasets](#datasets)
3. [Algorithms & Methods](#algorithms--methods)
4. [Model Components](#model-components)
5. [Data Processing Pipeline](#data-processing-pipeline)
6. [Training & Evaluation](#training--evaluation)
7. [Molecular Validation](#molecular-validation)
8. [Model Deployment](#model-deployment)
9. [Installation & Setup](#installation--setup)
10. [Usage](#usage)
11. [Results & Interpretation](#results--interpretation)

---

## Project Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GNN Drug Repurposing System               │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  1. DATA LOADING                                             │
│     └─ PrimeKG Knowledge Graph (CSV)                         │
│        ├─ Entity Types: Drug, Disease, Protein, Gene         │
│        └─ Relationships: Multiple biomedical interactions    │
│                                                               │
│  2. GRAPH CONSTRUCTION                                       │
│     └─ Build multi-node-type knowledge graph                 │
│        ├─ Create node embeddings (type + ID)                 │
│        ├─ Build bidirectional edge index                     │
│        └─ Normalize adjacency matrix with self-loops         │
│                                                               │
│  3. GNN MODEL TRAINING                                       │
│     └─ Graph Convolutional Network                           │
│        ├─ Encode: Node → Embeddings via 2-layer GCN          │
│        ├─ Score: Drug-Disease pair scoring                   │
│        └─ Link Prediction: Binary classification              │
│                                                               │
│  4. VALIDATION WITH MOLECULAR DOCKING                        │
│     └─ AutoDock Vina                                         │
│        ├─ Download protein structures (PDB)                  │
│        ├─ Prepare drug molecules (SMILES → 3D)               │
│        └─ Compute binding affinities                         │
│                                                               │
│  5. INFERENCE & DEPLOYMENT                                   │
│     └─ UDP Streaming Server                                  │
│        ├─ Real-time drug predictions per disease             │
│        └─ Continuous message broadcasting                    │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## Datasets

### **PrimeKG Knowledge Graph**
- **Source**: Harvard Dataverse
- **URL**: `https://dataverse.harvard.edu/api/access/datafile/6180620`
- **Format**: CSV with relationships and entities
- **Size**: Millions of biomedical relationships

#### Entity Types:
- **Drug**: Chemical compounds with known pharmacological effects
- **Disease**: Medical conditions and pathologies
- **Protein**: Molecular targets and functional components
- **Gene**: Genetic information and regulatory elements

#### Relationship Types:
Examples include:
- `drug_treats_disease`
- `protein_interacts_with_protein`
- `gene_encodes_protein`
- `disease_associated_with_gene`

### **Column Standardization**:
The notebook automatically handles varying column names in PrimeKG:
- Source ID: `x_id` / `x_index` → `source_id`
- Source Type: `x_type` → `source_type`
- Target ID: `y_id` / `y_index` → `target_id`
- Target Type: `y_type` → `target_type`
- Relation: `relation` / `display_relation` → `relation`

---

## Algorithms & Methods

### **1. Graph Construction**

#### Node Indexing:
```
Node Key Format: "entity_type::entity_id"
Example: "drug::DB00001", "disease::MESH:D000001"
```

- Unique node identifiers created by combining entity type and ID
- Bidirectional edge creation for information flow in both directions
- Total nodes extracted from the knowledge graph

#### Adjacency Matrix Normalization:
```
Formula: Ã = (D + I)^(-1/2) * A * (D + I)^(-1/2)

Where:
- A = Adjacency matrix
- I = Identity matrix (self-loops)
- D = Degree matrix (diagonal matrix of node degrees)
```

**Why this normalization?**
- Prevents vanishing gradients during message passing
- Normalizes by degree to prevent high-degree nodes from dominating
- Self-loops ensure nodes consider their own features

---

### **2. Graph Convolutional Networks (GCN)**

#### Single GCN Layer Formula:
```
H(l+1) = σ(Ã * H(l) * W(l) + b(l))

Where:
- H(l) = Node feature matrix at layer l
- Ã = Normalized adjacency matrix
- W(l) = Weight matrix (learnable parameters)
- σ = Activation function (ReLU)
- b(l) = Bias term
```

#### Two-Layer GCN Architecture:
```
Input (Node Features) 
       ↓
    [GCN-1: HIDDEN_DIM → HIDDEN_DIM]
       ↓ (ReLU + Dropout)
    [GCN-2: HIDDEN_DIM → EMBEDDING_DIM]
       ↓
 Output (Node Embeddings)
```

**Configuration Parameters**:
- `HIDDEN_DIM = 256`: Intermediate representation size
- `EMBEDDING_DIM = 128`: Final embedding dimensionality
- `DROPOUT = 0.2`: Regularization during training

---

### **3. Node Embedding Strategy**

Each node's initial representation combines:
```
Initial_Embedding = Node_ID_Embedding + Entity_Type_Embedding

Where:
- Node_ID_Embedding: Learned embedding for specific node identity
- Entity_Type_Embedding: Learned embedding for entity type (drug/disease/protein/gene)
```

**Advantages**:
- Captures both node-specific and type-specific information
- Types share representation learning (data efficiency)
- Allows generalization to unseen node IDs with known types

---

### **4. Link Prediction for Drug-Disease Associations**

#### Scoring Function:
```
score(drug_idx, disease_idx) = MLP([z_drug || z_disease || z_drug ⊙ z_disease])

Where:
- z_drug = Embedding vector for drug node
- z_disease = Embedding vector for disease node
- || = Concatenation operator
- ⊙ = Element-wise multiplication (Hadamard product)
- MLP = Multi-layer perceptron
```

#### MLP Architecture:
```
Concatenated Features (384-dim for 128-dim embeddings)
    ↓
  [Linear: 384 → 128]
    ↓
    [ReLU]
    ↓
  [Dropout(0.2)]
    ↓
  [Linear: 128 → 1]
    ↓
 Output Score [0, 1] via Sigmoid
```

**Why this design?**
- Concatenation captures individual drug and disease embeddings
- Hadamard product captures interaction patterns
- Non-linear MLP learns complex scoring relationships

---

### **5. Training with Negative Sampling**

#### Dataset Splits:
```
Known Drug-Disease Pairs
    ├─ 80%: Training Set
    ├─ 10%: Validation Set
    └─ 10%: Test Set
```

#### Balanced Classification:
```
Training Batch Composition:
├─ Positive Pairs: Known drug-disease associations (label = 1)
└─ Negative Pairs: Random drug-disease pairs NOT in knowledge graph (label = 0)

Ratio: NEGATIVE_SAMPLE_RATIO = 1.0 (1:1 positive-to-negative)
```

**Why negative sampling is important**:
- Knowledge graphs are sparse (far more false pairs than true ones)
- 1:1 ratio prevents class imbalance from skewing training
- Forces model to distinguish between true and random associations

#### Loss Function:
```
Loss = Binary Cross-Entropy with Logits

L = -[y * log(σ(logits)) + (1-y) * log(1 - σ(logits))]

Where:
- y = True label (0 or 1)
- σ(logits) = Sigmoid probability
- Average over all training pairs
```

---

### **6. Optimization**

#### Optimizer Configuration:
```
torch.optim.Adam(
    parameters=model.parameters(),
    lr=0.001,                    # Learning rate
    weight_decay=1e-5            # L2 regularization
)
```

**Training Loop**:
1. **Forward Pass**: Encode graph → Score training pairs → Compute loss
2. **Backward Pass**: Backpropagation through GNN and MLP
3. **Optimization**: Update weights using Adam
4. **Validation**: Every 10 epochs on validation set
5. **Metric**: ROC-AUC score (measures ranking quality)

---

### **7. Inference: Drug Discovery for a Disease**

Process for finding drug candidates:
```
1. User Query → Search for disease in knowledge graph
   Example: "Anemia" → Find matching disease entities
   
2. Load Final Embeddings → Run GNN on full graph once
   All disease embeddings Z_disease are computed
   
3. Score All Drugs → For selected disease:
   For each drug node:
       score = model.score(drug_embedding, disease_embedding)
   
4. Rank by Score → Sort drugs by association strength (descending)
   
5. Return Top-K → Display top 10 most promising candidates
   Including:
   - Drug name
   - Association score (0.0-1.0)
   - Confidence level (HIGH/MEDIUM/LOW)
```

---

## Model Components

### **1. GraphConv Layer**
```python
class GraphConv(nn.Module):
    """Implements single graph convolutional operation"""
    - Aggregate neighborhood information via sparse matrix multiplication
    - Apply linear transformation to aggregated features
```

### **2. PrimeKGDrugRepurposingGNN Model**
```python
class PrimeKGDrugRepurposingGNN(nn.Module):
    Components:
    ├─ node_embedding: Embedding(num_nodes, HIDDEN_DIM)
    │  └─ Learns unique representation for each node
    ├─ type_embedding: Embedding(num_types, HIDDEN_DIM)
    │  └─ Learns representation for Drug/Disease/Protein/Gene
    ├─ gcn1: GraphConv(HIDDEN_DIM → HIDDEN_DIM)
    │  └─ First message passing layer with ReLU activation
    ├─ gcn2: GraphConv(HIDDEN_DIM → EMBEDDING_DIM)
    │  └─ Second message passing layer (final embeddings)
    ├─ dropout: Dropout(0.2)
    │  └─ Prevents overfitting by randomly zeroing activations
    └─ link_predictor: MLP
       └─ Scores drug-disease pairs for ranking
```

---

## Data Processing Pipeline

### **Step 1: Download & Load**
```
Harvard Dataverse → CSV File → Pandas DataFrame
- Automatic caching to avoid re-downloads
- Supports custom max_rows for testing on subsets
```

### **Step 2: Column Standardization**
```
Raw CSV columns (variable naming) → Standardized format
- source_id, source_type, target_id, target_type, relation
- Case normalization (lowercase)
- Remove null values
```

### **Step 3: Name Mappings**
```
Extract Human-Readable Names:
├─ disease_id_to_name: "MESH:D000001" → "Anemia"
└─ drug_id_to_name: "DB00001" → "Lepirudin"

Used for display in disease search and drug rankings
```

### **Step 4: Graph Construction**
```
Create unique nodes:
├─ Format: entity_type::entity_id
├─ Example: "drug::DB00001", "disease::MESH:D000001"
└─ Build complete mapping: node_key → node_index

Create edges:
├─ Extract source and target node indices
├─ Create bidirectional edges (undirected information flow)
└─ Store as edge_index tensor: [2, num_edges]
```

### **Step 5: Adjacency Normalization**
```
Raw edge list → Normalized adjacency matrix
├─ Add self-loops (each node connected to itself)
├─ Compute degree normalization: D^(-1/2)
├─ Create sparse COO tensor (memory efficient)
└─ Coalesce to combine duplicate indices
```

---

## Training & Evaluation

### **Configuration Parameters**:
```python
# Network Architecture
HIDDEN_DIM = 256           # Intermediate layer size
EMBEDDING_DIM = 128        # Final embedding dimension
DROPOUT = 0.2              # Regularization parameter

# Training
EPOCHS = 100               # Number of training iterations
BATCH_SIZE = 512           # Samples per batch
LR = 1e-3                  # Learning rate
WEIGHT_DECAY = 1e-5        # L2 regularization coefficient

# Data Splits
VAL_RATIO = 0.1            # 10% validation
TEST_RATIO = 0.1           # 10% test
NEGATIVE_SAMPLE_RATIO = 1.0  # 1:1 positive:negative ratio

# Inference
TOP_K = 10                 # Return top 10 candidates
SEED = 42                  # Reproducibility
```

### **Training Metrics**:

#### Binary Cross-Entropy Loss
- Measures how well model predicts true/false associations
- Lower is better; typical range: 0.1 - 0.7 from start to convergence

#### ROC-AUC Score
- Area Under the Receiver Operating Characteristic Curve
- Measures ranking quality (not just classification accuracy)
- **Range**: 0.5 (random) to 1.0 (perfect ranking)
- **Typical Results**: 0.75-0.95 for well-trained models

#### Loss Components:
```
Total Loss = Binary Cross-Entropy Loss
           + L2 Regularization (weight_decay)
```

### **Validation Strategy**:
- Evaluated every 10 epochs on held-out validation set
- Uses same evaluation metrics as test set
- Model checkpoint tracking (best AUC retained)

---

## Molecular Validation

### **Why Validation with Docking?**
- **GNN Advantage**: Fast, scalable predictions across millions of pairs
- **Docking Advantage**: Ground truth physics/chemistry validation
- **Combined Approach**: Confidence in predictions when both agree

### **AutoDock Vina Integration**

#### Workflow:
```
1. Get GNN Predictions
   └─ Top-K drugs from GNN rankings
   
2. Find Disease Targets
   └─ Disease-specific protein targets from PrimeKG
   
3. Prepare Proteins
   └─ Download 3D structures from PDB database
   
4. Prepare Drugs
   └─ Convert SMILES notation → 3D molecular structure
   └─ Generate conformers (spatial configurations)
   
5. Run Docking
   └─ AutoDock Vina computes binding energy
   └─ Output: Binding affinity (kcal/mol)
   
6. Compare Results
   └─ GNN scores vs Docking affinities
   └─ Compute agreement metrics
```

### **Binding Affinity Interpretation**:
```
Binding Affinity (kcal/mol)    | Interpretation
─────────────────────────────────────────────────
-9.0 or lower                  | Excellent (very strong)
-7.0 to -9.0                   | Good (strong)
-5.0 to -7.0                   | Moderate
-3.0 to -5.0                   | Weak
Above -3.0                     | Very weak / No binding

Rule: More negative = stronger binding
```

### **Validation Metrics**:
```
Agreement Score = 1 - |gnn_score - normalized_affinity|

Where:
- GNN score ∈ [0, 1]
- Normalized affinity = (actual_affinity + 10) / 10 ∈ [0, 1]

Interpretation:
- > 0.80: Excellent agreement ✓
- 0.65-0.80: Good agreement ✓
- 0.50-0.65: Fair agreement ⚠
- < 0.50: Poor agreement ✗
```

---

## Model Deployment

### **Model Artifacts Saved**:

#### 1. Model State (`gnn_drug_repurposing.pt`)
```python
{
    'model_state': state_dict,           # Trained weights
    'model_config': {                    # Architecture info
        'num_nodes': 15000,              # Example
        'num_types': 4,                  # Drug, Disease, Protein, Gene
        'hidden_dim': 256,
        'embedding_dim': 128,
        'dropout': 0.2
    }
}
```

#### 2. Metadata (`metadata.pkl`)
```python
{
    'all_keys': [...],                   # Node identifiers
    'node_map': {...},                   # Key → Index mapping
    'node_types': [...],                 # Type per node
    'type_to_idx': {...},                # Type → Index mapping
    'drug_nodes': [...],                 # Drug node indices
    'disease_nodes': [...],              # Disease node indices
    'disease_id_to_name': {...},         # ID → Human-readable names
    'drug_id_to_name': {...}             # ID → Human-readable names
}
```

#### 3. Adjacency Matrix (`adjacency.pt`)
```
Sparse COO tensor containing normalized adjacency matrix
- Memory efficient (only stores non-zero entries)
- Ready for sparse matrix operations during inference
```

### **UDP Streaming Server**

#### Purpose:
Real-time broadcast of drug predictions over UDP network protocol.

#### Configuration:
```python
UDP_HOST = "192.168.31.103"    # Target IP address
UDP_PORT = 5005                # Target port
DISEASE_QUERY = "Anemia"       # Disease to predict for
TOP_K = 10                     # Return top 10 candidates
SEND_INTERVAL_SEC = 2.0        # Send every 2 seconds
BURST_COUNT = 3                # Send 3 copies (UDP is lossy)
BURST_GAP_SEC = 0.05           # Gap between copies
```

#### Payload Structure:
```json
{
  "event": "gnn_drug_predictions",
  "timestamp_utc": "2026-03-05T...",
  "msg_id": 1,
  "model_path": "models/gnn_drug_repurposing.pt",
  "disease_query": "Anemia",
  "disease": {
    "idx": 4521,
    "id": "MESH:D000740",
    "name": "Anemia"
  },
  "top_k": 10,
  "predictions": [
    {
      "rank": 1,
      "drug_idx": 1203,
      "drug_id": "DB00001",
      "drug_name": "Lepirudin",
      "score": 0.8523
    },
    ...
  ]
}
```

#### Continuous Streaming:
- Sends batches of 3 identical packets (with 50ms gaps)
- Repeats every 2 seconds
- Includes message ID and timestamp for tracking
- UDP broadcast allows multiple clients to receive simultaneously

---

## Installation & Setup

### **Requirements**:
```
Python 3.8+
CUDA 11.0+ (optional, for GPU acceleration)
```

### **Dependencies**:
```bash
# Core ML/Data Science
torch>=1.9.0              # PyTorch deep learning
pandas>=1.3.0             # Data manipulation
numpy>=1.20.0             # Numerical computing
scikit-learn>=0.24.0      # Evaluation metrics
requests>=2.26.0          # Download files

# Molecular Docking
meeko                     # Prepare molecules for docking
vina                      # AutoDock Vina engine
biopython>=1.80           # Protein structure tools
rdkit                     # Cheminformatics

# Utilities
tqdm                      # Progress bars
```

### **Installation**:
```bash
pip install torch pandas numpy scikit-learn requests tqdm biopython rdkit meeko vina
```

---

## Usage

### **Basic Workflow**:

#### 1. Load and Train Model
```python
# Configuration in notebook sets parameters
# Run all cells 1-24 to:
# - Download PrimeKG data
# - Build knowledge graph
# - Train GNN for 100 epochs
# - Evaluate on test set
```

#### 2. Query Disease & Get Predictions
```python
# In cell 28-33:
disease_query = "Anemia"  # Search for disease
# Returns top 10 drug candidates with scores

# Example output:
# 1. Lepirudin (score: 0.8523) - HIGH confidence
# 2. Argatroban (score: 0.7812) - HIGH confidence
# 3. Pentoxifylline (score: 0.6234) - MEDIUM confidence
```

#### 3. Validate with Molecular Docking
```python
# In cells 34-40:
# - Automatically prepares drug molecules
# - Downloads target proteins
# - Runs AutoDock Vina
# - Compares GNN vs docking scores
```

#### 4. Deploy for Inference
```python
# In cells 41-42:
# - Save model and metadata
# - Start UDP streaming server
# - Continuous broadcast of predictions
```

---

## Results & Interpretation

### **Expected Performance**:

#### Model Training Results:
```
Training Metrics (100 epochs):
├─ Final Training Loss: 0.25-0.35
├─ Final Validation AUC: 0.80-0.92
└─ Final Test AUC: 0.78-0.90

Note: Exact values depend on PrimeKG subset size and entity distribution
```

#### Drug Discovery Results:
```
For "Anemia" query (example):
Rank │ Drug Name           │ GNN Score │ Confidence
──────┼─────────────────────┼───────────┼────────────
  1  │ Lepirudin           │ 0.8523    │ HIGH
  2  │ Argatroban          │ 0.7812    │ HIGH
  3  │ Pentoxifylline      │ 0.6234    │ MEDIUM
  4  │ Cilostazol          │ 0.5789    │ MEDIUM
  ...
```

#### Molecular Validation:
```
Drug              │ GNN Score │ Best Affinity │ Agreement
──────────────────┼───────────┼───────────────┼────────────
Lepirudin         │ 0.8523    │ -7.85 kcal/mol│ ✓ EXCELLENT
Argatroban        │ 0.7812    │ -6.92 kcal/mol│ ✓ GOOD
Pentoxifylline    │ 0.6234    │ -5.11 kcal/mol│ ⚠ FAIR
```

### **Interpretation Guide**:

#### High GNN Score (> 0.75)
- Model predicts strong drug-disease association
- Usually substantiated by molecular docking
- **Next Step**: Consider for experimental validation

#### Medium GNN Score (0.50-0.75)
- Moderate computational prediction
- May have some biological relevance
- **Next Step**: Compare with docking data; may need further analysis

#### Low GNN Score (< 0.50)
- Weak predicted association
- Usually not supported by docking
- **Next Step**: Likely false positive; deprioritize

### **Biological Validation Checklist**:
```
For each top-K prediction:
⬜ Check literature for known interactions
⬜ Compare with DrugBank positive cases
⬜ Verify protein targets are disease-relevant
⬜ Assess molecular docking agreement
⬜ Evaluate safety profiles and side effects
⬜ Consider known drug mechanisms vs. disease pathology
```

---

## File Structure

```
majorProj/
├── gnn_drug_repurposing_improved.ipynb    # Main notebook
├── GNNDrug_(1).ipynb                      # Alternative notebook
├── udp_veiwer.py                          # UDP receiver/viewer
├── __pycache__/                           # Python cache
├── data/                                  # (Created during run)
│   ├── primekg.csv                        # Knowledge graph
│   ├── pdb/                               # Protein structures
│   └── ligands/                           # Drug molecules
├── models/                                # (Created during run)
│   ├── gnn_drug_repurposing.pt           # Trained model
│   ├── metadata.pkl                       # Graph metadata
│   └── adjacency.pt                       # Adjacency matrix
└── README.md                              # This file
```

---

## Key Innovations

1. **Bidirectional Knowledge Graph**: Information flows in both directions through drug, disease, protein, and gene networks

2. **Combined Scoring**: Concatenation + Hadamard product in link predictor captures both individual features and interactions

3. **Proper Validation Split**: Clear separation of train/val/test prevents information leakage

4. **Balanced Negative Sampling**: 1:1 ratio ensures model learns to distinguish true from false associations

5. **Molecular Validation**: AutoDock docking provides independent ground-truth validation of GNN predictions

6. **Real-time Deployment**: UDP streaming enables continuous delivery of predictions to client applications

---

## Future Enhancements

1. **Multi-hop Reasoning**: Consider drug-protein-gene-disease paths
2. **Temporal Dynamics**: Track how associations change over time
3. **Uncertainty Quantification**: Add confidence intervals to predictions
4. **Fine-tuning with Docking Data**: Retrain GNN using binding affinities as auxiliary loss
5. **Disease-specific Models**: Train separate GNNs for different disease categories
6. **Side Effect Prediction**: Add models to predict adverse reactions
7. **Web Interface**: Build frontend for interactive drug discovery

---

## References

- **PrimeKG Dataset**: Varshney et al., "PrimeKG: A Knowledge Graph for Precision Medicine"
- **Graph Convolutional Networks**: Kipf & Welling, "Semi-Supervised Classification with Graph Convolutional Networks" (ICLR 2017)
- **AutoDock Vina**: Trott & Olson, "AutoDock Vina: Improving the speed and accuracy of docking with a new scoring function, efficient optimization, and multithreading" (Journal of Computational Chemistry, 2010)
- **Link Prediction**: Liben-Nowell & Kleinberg, "The link prediction problem for social networks"

---

## License & Attribution

This project uses publicly available datasets and open-source software. Ensure proper attribution when publishing results.

---

## Contact & Support

For issues or questions, refer to:
- Notebook cell comments for implementation details
- Configuration section for parameter tuning
- Molecular validation section for docking troubleshooting

