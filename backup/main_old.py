from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from pathlib import Path
import random
import os
from dotenv import load_dotenv

load_dotenv()
BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000")

# Graph Neural Network Classes
class GraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = torch.sparse.mm(adj, x)
        return self.linear(x)

class PrimeKGDrugRepurposingGNN(nn.Module):
    def __init__(self, num_nodes: int, num_types: int, hidden_dim: int, embedding_dim: int, dropout: float):
        super().__init__()
        self.node_embedding = nn.Embedding(num_nodes, hidden_dim)
        self.type_embedding = nn.Embedding(num_types, hidden_dim)
        self.gcn1 = GraphConv(hidden_dim, hidden_dim)
        self.gcn2 = GraphConv(hidden_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.link_predictor = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 1)
        )
    
    def encode(self, node_type_ids: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        idx = torch.arange(len(node_type_ids), device=node_type_ids.device)
        x = self.node_embedding(idx) + self.type_embedding(node_type_ids)
        h = F.relu(self.gcn1(x, adj))
        h = self.dropout(h)
        return self.gcn2(h, adj)
    
    def score(self, z: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
        src = z[pairs[0]]
        dst = z[pairs[1]]
        feat = torch.cat([src, dst, src * dst], dim=-1)
        return self.link_predictor(feat).squeeze()

# Load Model Configuration & App Setup
app = FastAPI(title="GNN Drug Repurposing API")

from fastapi.staticfiles import StaticFiles

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the static data directory
data_dir = Path('../data')
if data_dir.exists():
    app.mount("/data", StaticFiles(directory="../data"), name="data")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

models_dir = Path('../models')
metadata = None
model = None
adj = None
z = None

class PredictionRequest(BaseModel):
    disease: str
    top_k: int = 10

@app.on_event("startup")
def load_models():
    global metadata, model, adj, z
    try:
        with open(models_dir / 'metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        checkpoint = torch.load(models_dir / 'gnn_drug_repurposing.pt', map_location=device)
        config = checkpoint['model_config']
        
        model = PrimeKGDrugRepurposingGNN(
            num_nodes=config['num_nodes'],
            num_types=config['num_types'],
            hidden_dim=config['hidden_dim'],
            embedding_dim=config['embedding_dim'],
            dropout=config['dropout']
        ).to(device)
        model.load_state_dict(checkpoint['model_state'])
        model.eval()
        
        adj = torch.load(models_dir / 'adjacency.pt', map_location=device)
        
        # Precompute embeddings
        node_types = metadata['node_types']
        type_to_idx = metadata['type_to_idx']
        node_type_ids = torch.tensor([type_to_idx[t] for t in node_types]).to(device)
        with torch.no_grad():
            z = model.encode(node_type_ids, adj)
            
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")

@app.post("/predict")
def predict(req: PredictionRequest):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
        
    query_lower = req.disease.lower()
    disease_nodes = metadata['disease_nodes']
    disease_id_to_name = metadata['disease_id_to_name']
    drug_nodes = metadata['drug_nodes']
    drug_id_to_name = metadata['drug_id_to_name']
    all_keys = metadata['all_keys']
    
    target_disease_idx = None
    target_disease_name = None
    
    # Try exact match first
    for d_idx in disease_nodes:
        d_key = all_keys[d_idx]
        d_id = d_key.split("::")[1]
        name = disease_id_to_name.get(d_id, "").lower()
        if name == query_lower:
            target_disease_idx = d_idx
            target_disease_name = disease_id_to_name.get(d_id, "")
            break
            
    # Then partial match
    if target_disease_idx is None:
        for d_idx in disease_nodes:
            d_key = all_keys[d_idx]
            d_id = d_key.split("::")[1]
            name = disease_id_to_name.get(d_id, "").lower()
            if query_lower in name:
                target_disease_idx = d_idx
                target_disease_name = disease_id_to_name.get(d_id, "")
                break
                
    if target_disease_idx is None:
        raise HTTPException(status_code=404, detail="Disease not found in knowledge graph")
        
    pairs = torch.tensor([[d, target_disease_idx] for d in drug_nodes], dtype=torch.long).T.to(device)
    
    with torch.no_grad():
        scores = torch.sigmoid(model.score(z, pairs)).cpu().numpy()
        
    # Define disease targets (matching the logic in gnn_drug_repurposing_improved.py)
    DISEASE_TARGETS = {
        "cancer": {"TP53": "1TUP", "EGFR": "1M17", "BCR_ABL": "1IEP"},
        "diabetes": {"INSULIN_RECEPTOR": "1IR3", "GLP1_RECEPTOR": "5EE7"},
        "heart": {"ACE": "1O8A", "ADRENERGIC_B1": "4GPO"},
        "leukemia": {"CD20": "2H7W", "BCL2": "2XA0"},
        "lymphoma": {"CD20": "2H7W", "BTK": "5P9J"},
        "rickets": {"VDR": "1DB1"},
        "skin disease": {"GR": "1M2Z"},
        "osteoarthritis": {"MMP13": "1XUD"},
        "long qt syndrome": {"hERG": "5VA1"},
        "escherichia coli": {"Gyra": "1EI1"},
        "peptic esophagitis": {"ATP4A": "5YLV"},
        "chronic lymphocytic leukemia": {"BTK": "5P9J"},
        "thalassemia": {"HbA": "1A3N"},
        "arteriosclerosis obliterans": {"HMGCR": "1HW9"},
        "anterior horn disease": {"SOD1": "1SPD"},
    }
    
    disease_key = None
    for key in DISEASE_TARGETS.keys():
        if key.lower() in target_disease_name.lower():
            disease_key = key
            break
            
    if not disease_key:
        disease_key = "cancer" # Default fallback
        
    targets = [{"name": prot, "pdb_id": pdb_id, "url": f"{BASE_URL}/data/pdb/{pdb_id.lower()}.pdb"} 
               for prot, pdb_id in DISEASE_TARGETS[disease_key].items()]

    top_indices = scores.argsort()[-req.top_k:][::-1]
    
    results = []
    import os
    for i, idx in enumerate(top_indices):
        drug_idx = drug_nodes[idx]
        drug_key = all_keys[drug_idx]
        d_id = drug_key.split("::")[1]
        drug_name = drug_id_to_name.get(d_id, d_id)
        gnn_score = float(scores[idx])
        
        # Simulate binding affinity and agreement
        random.seed(hash(drug_name + target_disease_name) % (2**32))
        base_affinity = -7.5 + random.uniform(-1.5, 1.5)
        affinity = round(base_affinity, 2)
        normalized_affinity = max(0, min(1, (affinity + 10) / 10))
        agreement = 1 - abs(gnn_score - normalized_affinity)
        agreement_label = "EXCELLENT" if agreement > 0.8 else "GOOD" if agreement > 0.65 else "FAIR" if agreement > 0.5 else "POOR"
        
        # Check if ligand sdf exists
        safe_drug_name = drug_name.replace(' ', '_')
        ligand_path = f"../data/ligands/{safe_drug_name}.sdf"
        ligand_url = f"{BASE_URL}/data/ligands/{safe_drug_name}.sdf" if os.path.exists(ligand_path) else None

        results.append({
            "drug_name": drug_name,
            "gnn_score": round(gnn_score, 4),
            "affinity": affinity,
            "agreement": agreement_label,
            "agreement_score": round(agreement, 4),
            "ligand_url": ligand_url
        })
        
    return {
        "disease": target_disease_name,
        "targets": targets,
        "predictions": results
    }

