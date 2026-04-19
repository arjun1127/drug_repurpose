from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, List, Optional, Set, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import json
from pathlib import Path
import os
import re
from difflib import SequenceMatcher
from dotenv import load_dotenv

load_dotenv()
BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000")

# Graph Neural Network Classes (must match training script architecture)
class RGCNConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_relations: int = 3):
        super().__init__()
        self.num_relations = num_relations
        self.weight = nn.Parameter(torch.Tensor(num_relations, in_dim, out_dim))
        self.loop_weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.loop_weight)

    def forward(self, x: torch.Tensor, adjs: List[torch.Tensor]) -> torch.Tensor:
        out = torch.matmul(x, self.loop_weight)
        for r in range(self.num_relations):
            msg = torch.sparse.mm(adjs[r], x)
            out = out + torch.matmul(msg, self.weight[r])
        return out


class ResidualRGCNLayer(nn.Module):
    def __init__(self, dim: int, num_relations: int = 3, dropout: float = 0.0):
        super().__init__()
        self.conv = RGCNConv(dim, dim, num_relations)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adjs: List[torch.Tensor]) -> torch.Tensor:
        h = self.conv(x, adjs)
        h = self.norm(h)
        h = F.relu(h)
        h = self.dropout(h)
        return x + h


class PrimeKGDrugRepurposingGNN(nn.Module):
    def __init__(self, num_nodes: int, num_types: int, hidden_dim: int, embedding_dim: int, dropout: float):
        super().__init__()
        self.node_embedding = nn.Embedding(num_nodes, hidden_dim)
        self.type_embedding = nn.Embedding(num_types, hidden_dim)

        self.gcn_in = RGCNConv(hidden_dim, hidden_dim, num_relations=3)
        self.res_layers = nn.ModuleList([ResidualRGCNLayer(hidden_dim, num_relations=3, dropout=dropout) for _ in range(1)])
        self.gcn_out = RGCNConv(hidden_dim, embedding_dim, num_relations=3)

        self.link_predictor = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim),
            nn.ReLU(),
            nn.BatchNorm1d(embedding_dim),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 1),
        )
        
        self.degree_alpha = nn.Parameter(torch.tensor(0.1))
        self.degree_beta = nn.Parameter(torch.tensor(0.1))

    def encode(self, node_type_ids: torch.Tensor, adjs: List[torch.Tensor]) -> torch.Tensor:
        idx = torch.arange(len(node_type_ids), device=node_type_ids.device)
        x = self.node_embedding(idx) + self.type_embedding(node_type_ids)

        x = F.relu(self.gcn_in(x, adjs))
        for layer in self.res_layers:
            x = layer(x, adjs)

        return self.gcn_out(x, adjs)

    def score(self, z: torch.Tensor, pairs: torch.Tensor, degrees: torch.Tensor) -> torch.Tensor:
        src_idx = pairs[0]
        dst_idx = pairs[1]

        src_z = z[src_idx]
        dst_z = z[dst_idx]

        src_deg = degrees[src_idx].clamp(min=1).float()
        dst_deg = degrees[dst_idx].clamp(min=1).float()

        features = torch.cat([src_z, dst_z, src_z * dst_z], dim=-1)
        mlp_score = self.link_predictor(features).squeeze(-1)
        
        base_score = mlp_score - F.relu(self.degree_alpha) * torch.log(src_deg) - F.relu(self.degree_beta) * torch.log(dst_deg)
        normalized_score = base_score / (torch.sqrt(src_deg) * torch.sqrt(dst_deg) + 1e-8)
        
        return normalized_score


# ─── App Setup ────────────────────────────────────────────────────────

app = FastAPI(title="GNN Drug Repurposing API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directories
data_dir = Path('../data')
if data_dir.exists():
    app.mount("/data", StaticFiles(directory="../data"), name="data")

plots_dir = Path('../models/plots')
if plots_dir.exists():
    app.mount("/plots", StaticFiles(directory="../models/plots"), name="plots")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

models_dir = Path('../models')
metadata = None
model = None
adj = None
degrees = None
z = None
training_metrics = None
degree_thresholds = None  # q33, q66 for drug degree buckets
therapeutic_drug_nodes = None  # whitelist of real pharmaceutical drug node indices
disease_catalog = []
disease_catalog_by_idx = {}
contraindications_by_disease = {}
therapeutic_by_disease = {}
drug_prior_centered = {}
prior_sampled_diseases = 0
adj_list_1hop = {}  # For explainability

DRUG_CATEGORIES = {
    "immunosuppressants": {"tacrolimus", "cyclosporine", "mycophenolate", "sirolimus", "azathioprine", "methotrexate", "dexamethasone", "prednisone"},
    "topical": {"alitretinoin", "clobetasol", "fluocinonide", "fluorouracil", "hydrocortisone", "betamethasone"}
}


class PredictionRequest(BaseModel):
    disease: str
    top_k: int = 10
    disease_node_idx: Optional[int] = None
    exclude_contraindicated: bool = True
    exclude_known_treatments: bool = True
    use_debias_rerank: bool = True
    debias_alpha: float = 0.8
    use_specificity_rerank: bool = True
    specificity_beta: float = 1.0
    use_hub_penalty: bool = True
    hub_degree_quantile: float = 0.9
    hub_penalty_factor: float = 0.7
    use_disease_zscore: bool = True
    candidate_limit: int = 6
    exclude_categories: List[str] = []
    orphan_cap: float = 5.0


def normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def build_disease_catalog(
    disease_nodes: List[int],
    all_keys: List[str],
    disease_id_to_name: Dict[str, str],
) -> List[Dict[str, object]]:
    catalog = []
    for d_idx in disease_nodes:
        d_key = all_keys[d_idx]
        d_id = d_key.split("::")[1]
        name = disease_id_to_name.get(d_id, "").strip()
        if not name:
            continue
        catalog.append(
            {
                "idx": int(d_idx),
                "name": name,
                "name_norm": normalize_text(name),
            }
        )
    return catalog


def find_disease_candidates(query: str, limit: int = 6) -> List[Dict[str, object]]:
    query_norm = normalize_text(query)
    if not query_norm or not disease_catalog:
        return []

    candidates: List[Dict[str, object]] = []

    # 1) Exact normalized match
    for entry in disease_catalog:
        if entry["name_norm"] == query_norm:
            candidates.append(
                {
                    "disease_node_idx": int(entry["idx"]),
                    "name": str(entry["name"]),
                    "match_type": "exact",
                    "match_score": 1.0,
                }
            )

    for entry in disease_catalog:
        name_norm = str(entry["name_norm"])
        if name_norm == query_norm:
            continue
        if query_norm in name_norm:
            score = min(0.99, 0.85 + len(query_norm) / max(len(name_norm), 1) * 0.14)
            candidates.append(
                {
                    "disease_node_idx": int(entry["idx"]),
                    "name": str(entry["name"]),
                    "match_type": "contains",
                    "match_score": float(score),
                }
            )

    # 3) Fuzzy sequence match (only when exact/contains were not enough)
    if len(candidates) < limit:
        seen_idx = {int(item["disease_node_idx"]) for item in candidates}
        fuzzy_matches: List[Tuple[float, Dict[str, object]]] = []
        for entry in disease_catalog:
            idx = int(entry["idx"])
            if idx in seen_idx:
                continue
            name_norm = str(entry["name_norm"])
            score = SequenceMatcher(None, query_norm, name_norm).ratio()
            if score >= 0.45:
                fuzzy_matches.append((score, entry))

        fuzzy_matches.sort(key=lambda x: x[0], reverse=True)
        for score, entry in fuzzy_matches[: max(0, limit - len(candidates))]:
            candidates.append(
                {
                    "disease_node_idx": int(entry["idx"]),
                    "name": str(entry["name"]),
                    "match_type": "fuzzy",
                    "match_score": float(score),
                }
            )

    candidates.sort(key=lambda x: float(x["match_score"]), reverse=True)
    return candidates[:limit]


def match_disease(query: str) -> Optional[Tuple[int, str, str, float]]:
    candidates = find_disease_candidates(query, limit=1)
    if not candidates:
        return None
    top = candidates[0]
    return (
        int(top["disease_node_idx"]),
        str(top["name"]),
        str(top["match_type"]),
        float(top["match_score"]),
    )


def normalize_disease_drug_map(raw: object) -> Dict[int, Set[int]]:
    out: Dict[int, Set[int]] = {}
    if not isinstance(raw, dict):
        return out
    for disease_idx, drugs in raw.items():
        try:
            disease_key = int(disease_idx)
        except (TypeError, ValueError):
            continue
        if isinstance(drugs, (list, tuple, set)):
            out[disease_key] = {int(d) for d in drugs}
    return out


def build_map_from_edge_list(edges: object) -> Dict[int, Set[int]]:
    out: Dict[int, Set[int]] = {}
    if not isinstance(edges, list):
        return out
    for pair in edges:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        drug_idx, disease_idx = int(pair[0]), int(pair[1])
        out.setdefault(disease_idx, set()).add(drug_idx)
    return out


def compute_drug_prior_centered_scores(
    drug_nodes: List[int],
    disease_nodes: List[int],
    sample_size: int = 128,
) -> Tuple[Dict[int, float], int]:
    """Compute centered global drug prior scores E_d[s(drug,d)] for de-bias reranking."""
    if model is None or z is None or degrees is None or not drug_nodes or not disease_nodes:
        return {}, 0

    if len(disease_nodes) <= sample_size:
        sampled_diseases = list(disease_nodes)
    else:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(42)
        perm = torch.randperm(len(disease_nodes), generator=gen)[:sample_size]
        sampled_diseases = [int(disease_nodes[int(i)]) for i in perm.tolist()]

    drug_tensor = torch.tensor([int(d) for d in drug_nodes], dtype=torch.long, device=device)
    score_sum = torch.zeros(drug_tensor.shape[0], dtype=torch.float, device=device)

    with torch.no_grad():
        for disease_idx in sampled_diseases:
            disease_tensor = torch.full_like(drug_tensor, int(disease_idx))
            pairs = torch.stack([drug_tensor, disease_tensor], dim=0)
            logits = model.score(z, pairs, degrees)
            score_sum += torch.sigmoid(logits)

    mean_scores = (score_sum / max(len(sampled_diseases), 1)).detach().cpu()
    center = float(mean_scores.mean().item())
    centered = mean_scores - center

    prior_map = {
        int(drug_nodes[i]): float(centered[i].item())
        for i in range(len(drug_nodes))
    }
    return prior_map, len(sampled_diseases)


@app.on_event("startup")
def load_models():
    global metadata, model, adj, degrees, z, training_metrics, degree_thresholds
    global therapeutic_drug_nodes, disease_catalog, disease_catalog_by_idx
    global contraindications_by_disease, therapeutic_by_disease
    global drug_prior_centered, prior_sampled_diseases, adj_list_1hop
    try:
        # Load metadata
        with open(models_dir / 'metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)

        # Load model checkpoint
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

        # Load adjacency and degree tensors
        adj = torch.load(models_dir / 'adjacency.pt', map_location=device)
        degrees = torch.load(models_dir / 'degrees.pt', map_location=device)

        # Precompute embeddings
        node_types = metadata['node_types']
        type_to_idx = metadata['type_to_idx']
        node_type_ids = torch.tensor([type_to_idx[t] for t in node_types]).to(device)
        with torch.no_grad():
            z = model.encode(node_type_ids, adj)

        # Load training metrics
        metrics_path = models_dir / 'training_metrics.json'
        if metrics_path.exists():
            with open(metrics_path) as f:
                training_metrics = json.load(f)

        # Compute degree thresholds for drug nodes
        # Use therapeutic drug nodes if available, otherwise fall back to all drug nodes
        if 'therapeutic_drug_nodes' in metadata:
            therapeutic_drug_nodes = metadata['therapeutic_drug_nodes']
            drug_nodes_for_threshold = therapeutic_drug_nodes
            print(f"Loaded therapeutic drug whitelist: {len(therapeutic_drug_nodes)} drugs")
        else:
            drug_nodes_for_threshold = metadata['drug_nodes']
            therapeutic_drug_nodes = None
            print("WARNING: No therapeutic_drug_nodes in metadata, using all drug nodes")

        drug_degrees = degrees[drug_nodes_for_threshold].cpu().numpy()
        import numpy as np
        q33, q66 = np.quantile(drug_degrees, [0.33, 0.66])
        degree_thresholds = {'q33': float(q33), 'q66': float(q66)}

        # Build disease catalog for robust matching.
        disease_catalog = build_disease_catalog(
            disease_nodes=metadata['disease_nodes'],
            all_keys=metadata['all_keys'],
            disease_id_to_name=metadata['disease_id_to_name'],
        )
        disease_catalog_by_idx = {
            int(entry["idx"]): entry
            for entry in disease_catalog
        }

        # Load disease->drug relation maps for safety filtering in prediction.
        contraindications_by_disease = normalize_disease_drug_map(
            metadata.get("contraindications_by_disease")
        )
        therapeutic_by_disease = normalize_disease_drug_map(
            metadata.get("therapeutic_by_disease")
        )
        if not contraindications_by_disease:
            contraindications_by_disease = build_map_from_edge_list(
                metadata.get("contraindication_edges")
            )
        if not therapeutic_by_disease:
            therapeutic_by_disease = build_map_from_edge_list(
                metadata.get("therapeutic_edges")
            )
        if not contraindications_by_disease:
            print("WARNING: No contraindication relation map in metadata; contraindication filtering disabled.")
        if not therapeutic_by_disease:
            print("WARNING: No therapeutic relation map in metadata; known-treatment filtering disabled.")

        # Global drug prior used for de-bias reranking at inference time.
        prior_drug_nodes = [int(x) for x in (therapeutic_drug_nodes or metadata['drug_nodes'])]
        prior_disease_nodes = [int(x) for x in metadata['disease_nodes']]
        
        # Adjusting the prior computation to use the list of adjs
        if model is not None and z is not None:
            drug_prior_centered, prior_sampled_diseases = compute_drug_prior_centered_scores(
                drug_nodes=prior_drug_nodes,
                disease_nodes=prior_disease_nodes,
                sample_size=128,
            )

        # Build 1-hop adjacency list for explainability
        if adj is not None:
            adj_list_1hop.clear()
            for r in range(len(adj)):
                indices = adj[r]._indices().cpu().numpy()
                for i in range(indices.shape[1]):
                    u, v = int(indices[0, i]), int(indices[1, i])
                    if u not in adj_list_1hop: adj_list_1hop[u] = set()
                    if v not in adj_list_1hop: adj_list_1hop[v] = set()
                    adj_list_1hop[u].add(v)
                    adj_list_1hop[v].add(u)

        print(f"Models loaded successfully. Config: hidden={config['hidden_dim']}, embed={config['embedding_dim']}")
        print(f"Drug degree thresholds: q33={q33:.0f}, q66={q66:.0f}")
        print(f"Disease catalog size: {len(disease_catalog)}")
        print(f"Drug prior map size: {len(drug_prior_centered)} (sampled diseases={prior_sampled_diseases})")
        print(f"Graph loaded for explainability: {len(adj_list_1hop)} nodes")
    except Exception as e:
        print(f"Error loading models: {e}")
        import traceback
        traceback.print_exc()


# ─── Endpoints ────────────────────────────────────────────────────────

class ExplainRequest(BaseModel):
    drug_node_idx: int
    disease_node_idx: int

@app.post("/explain")
def explain_path(req: ExplainRequest):
    if not adj_list_1hop:
        raise HTTPException(status_code=500, detail="Graph not loaded")
    
    drug_idx = req.drug_node_idx
    disease_idx = req.disease_node_idx
    
    if drug_idx not in adj_list_1hop or disease_idx not in adj_list_1hop:
        return {"paths": []}
        
    drug_neighbors = adj_list_1hop[drug_idx]
    disease_neighbors = adj_list_1hop[disease_idx]
    
    shared_nodes = drug_neighbors.intersection(disease_neighbors)
    
    paths = []
    all_keys = metadata['all_keys']
    
    def format_node(idx):
        key = all_keys[idx]
        parts = key.split("::")
        node_type = parts[0]
        node_id = parts[1]
        
        if node_type == 'gene/protein':
            name = f"Protein {node_id}"
        elif node_type == 'phenotype':
            name = f"Phenotype {node_id}"
        else:
            name = f"{node_type.capitalize()} {node_id}"
        return {"idx": idx, "type": node_type, "name": name}

    # Length-2 paths
    for node_idx in shared_nodes:
        node_info = format_node(node_idx)
        paths.append({
            "path_len": 2,
            "nodes": [format_node(drug_idx), node_info, format_node(disease_idx)],
            "shared_node_idx": node_idx,
            "shared_node_type": node_info["type"],
            "shared_node_name": node_info["name"]
        })
        if len(paths) >= 20: break

    # Fallback to Length-3 paths if no length-2 paths exist
    if not paths:
        for n1 in drug_neighbors:
            if n1 == disease_idx: continue
            for n2 in adj_list_1hop.get(n1, set()):
                if n2 == drug_idx or n2 == disease_idx: continue
                if n2 in disease_neighbors:
                    paths.append({
                        "path_len": 3,
                        "nodes": [format_node(drug_idx), format_node(n1), format_node(n2), format_node(disease_idx)],
                        "shared_node_idx": n1,
                        "shared_node_type": "complex_path",
                        "shared_node_name": f"{format_node(n1)['name']} ➔ {format_node(n2)['name']}"
                    })
                    if len(paths) >= 10: break
            if len(paths) >= 10: break

    return {"paths": paths}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/metrics")
def get_metrics():
    """Return training metrics, config, test results, and bias analysis."""
    if training_metrics is None:
        raise HTTPException(status_code=404, detail="Training metrics not found")
    return training_metrics


@app.get("/plots-list")
def get_plots_list():
    """Return list of available plot filenames."""
    if not plots_dir.exists():
        return {"plots": []}
    files = sorted([f.name for f in plots_dir.iterdir() if f.suffix == '.png'])
    return {"plots": files}


@app.post("/predict")
def predict(req: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    if req.top_k <= 0:
        raise HTTPException(status_code=400, detail="top_k must be > 0")

    # Use therapeutic drug whitelist if available, otherwise fall back to all drugs
    if therapeutic_drug_nodes is not None:
        base_candidate_drug_nodes = [int(x) for x in therapeutic_drug_nodes]
    else:
        base_candidate_drug_nodes = [int(x) for x in metadata['drug_nodes']]

    disease_candidates = find_disease_candidates(req.disease, limit=max(1, min(req.candidate_limit, 12)))
    if not disease_candidates:
        raise HTTPException(status_code=404, detail="Disease not found in knowledge graph")

    if req.disease_node_idx is not None:
        chosen = disease_catalog_by_idx.get(int(req.disease_node_idx))
        if chosen is None:
            raise HTTPException(status_code=404, detail="Provided disease_node_idx not found in catalog")
        target_disease_idx = int(chosen["idx"])
        target_disease_name = str(chosen["name"])
        match_type = "manual"
        match_score = 1.0
    else:
        top = disease_candidates[0]
        target_disease_idx = int(top["disease_node_idx"])
        target_disease_name = str(top["name"])
        match_type = str(top["match_type"])
        match_score = float(top["match_score"])

    contraindicated_drugs = contraindications_by_disease.get(target_disease_idx, set())
    known_treatment_drugs = therapeutic_by_disease.get(target_disease_idx, set())
    filtered_out = set()
    if req.exclude_contraindicated:
        filtered_out.update(contraindicated_drugs)
    if req.exclude_known_treatments:
        filtered_out.update(known_treatment_drugs)

    drug_id_to_name = metadata['drug_id_to_name']
    all_keys = metadata['all_keys']

    # Biological Filtering
    if req.exclude_categories:
        cat_drugs = set()
        for cat in req.exclude_categories:
            if cat in DRUG_CATEGORIES:
                cat_drugs.update(DRUG_CATEGORIES[cat])
        
        filtered_out_by_cat = set()
        for d in base_candidate_drug_nodes:
            d_name = normalize_text(drug_id_to_name.get(all_keys[d].split("::")[1], ""))
            if any(c in d_name for c in cat_drugs):
                filtered_out_by_cat.add(d)
        filtered_out.update(filtered_out_by_cat)

    candidate_drug_nodes = [
        int(d) for d in base_candidate_drug_nodes
        if int(d) not in filtered_out
    ]
    if not candidate_drug_nodes:
        raise HTTPException(
            status_code=404,
            detail="No candidate drugs left after contraindication/treatment filtering."
        )

    pairs = torch.tensor([[d, target_disease_idx] for d in candidate_drug_nodes], dtype=torch.long).T.to(device)

    hub_threshold_value: Optional[float] = None
    with torch.no_grad():
        # model.score now returns normalized scores (logits)
        logits_t = model.score(z, pairs, degrees)
        raw_scores_t = torch.sigmoid(logits_t)
        rank_scores_t = logits_t.clone() # Rerank based on logits for more dynamic range
        drug_degree_t = degrees[pairs[0]].float().clamp(min=1.0)

        use_disease_zscore = bool(req.use_disease_zscore)
        if use_disease_zscore:
            score_mean = rank_scores_t.mean()
            score_std = rank_scores_t.std(unbiased=False).clamp(min=1e-6)
            rank_scores_t = (rank_scores_t - score_mean) / score_std

        use_specificity = bool(req.use_specificity_rerank)
        specificity_beta = float(max(0.0, min(req.specificity_beta, 3.0)))
        orphan_cap_val = float(max(1.0, req.orphan_cap))
        drug_degree_capped = torch.clamp(drug_degree_t, min=orphan_cap_val)
        specificity_t = 1.0 / torch.log1p(drug_degree_capped)
        specificity_t = specificity_t / specificity_t.mean().clamp(min=1e-6)
        if use_specificity:
            rank_scores_t = rank_scores_t * specificity_t.pow(specificity_beta)

        use_hub_penalty = bool(req.use_hub_penalty)
        hub_degree_quantile = float(max(0.5, min(req.hub_degree_quantile, 0.99)))
        hub_penalty_factor = float(max(0.1, min(req.hub_penalty_factor, 1.0)))
        if use_hub_penalty:
            degree_np = drug_degree_t.detach().cpu().numpy()
            hub_threshold_value = float(np.quantile(degree_np, hub_degree_quantile))
            hub_mask = drug_degree_t > hub_threshold_value
            rank_scores_t = torch.where(hub_mask, rank_scores_t * hub_penalty_factor, rank_scores_t)

        # Debias by subtracting global drug prior.
        alpha = float(max(0.0, min(req.debias_alpha, 1.5)))
        use_debias = bool(req.use_debias_rerank) and len(drug_prior_centered) > 0
        prior_vals = [float(drug_prior_centered.get(int(d), 0.0)) for d in candidate_drug_nodes]
        prior_t = torch.tensor(prior_vals, dtype=rank_scores_t.dtype, device=rank_scores_t.device)
        if use_debias:
            prior_norm_t = prior_t
            if use_disease_zscore:
                prior_mean = prior_norm_t.mean()
                prior_std = prior_norm_t.std(unbiased=False).clamp(min=1e-6)
                prior_norm_t = (prior_norm_t - prior_mean) / prior_std
            rank_scores_t = rank_scores_t - alpha * prior_norm_t
        else:
            prior_norm_t = prior_t

        scores = raw_scores_t.cpu().numpy()
        rank_scores = rank_scores_t.cpu().numpy()
        prior_np = prior_t.cpu().numpy()
        specificity_np = specificity_t.cpu().numpy()

    # Protein targets for a small curated set of diseases.
    # Use exact normalized disease-name matching only (no broad substring fallback).
    DISEASE_TARGETS_BY_NAME = {
        normalize_text("cancer"): {"TP53": "1TUP", "EGFR": "1M17", "BCR_ABL": "1IEP"},
        normalize_text("leukemia"): {"CD20": "2H7W", "BCL2": "2XA0"},
        normalize_text("lymphoma"): {"CD20": "2H7W", "BTK": "5P9J"},
        normalize_text("chronic lymphocytic leukemia"): {"BTK": "5P9J"},
        normalize_text("rickets"): {"VDR": "1DB1"},
        normalize_text("skin disease"): {"GR": "1M2Z"},
        normalize_text("osteoarthritis"): {"MMP13": "1XUD"},
        normalize_text("long qt syndrome"): {"hERG": "5VA1"},
        normalize_text("escherichia coli"): {"Gyra": "1EI1"},
        normalize_text("peptic esophagitis"): {"ATP4A": "5YLV"},
        normalize_text("thalassemia"): {"HbA": "1A3N"},
        normalize_text("arteriosclerosis obliterans"): {"HMGCR": "1HW9"},
        normalize_text("anterior horn disease"): {"SOD1": "1SPD"},
    }

    targets = []
    disease_norm = normalize_text(target_disease_name)
    if disease_norm in DISEASE_TARGETS_BY_NAME:
        targets = [
            {"name": prot, "pdb_id": pdb_id, "url": f"{BASE_URL}/data/pdb/{pdb_id.lower()}.pdb"}
            for prot, pdb_id in DISEASE_TARGETS_BY_NAME[disease_norm].items()
        ]

    k = min(req.top_k, len(candidate_drug_nodes))
    top_indices = rank_scores.argsort()[-k:][::-1]

    results = []
    for i, idx in enumerate(top_indices):
        drug_idx = candidate_drug_nodes[idx]
        drug_key = all_keys[drug_idx]
        d_id = drug_key.split("::")[1]
        drug_name = drug_id_to_name.get(d_id, d_id)
        gnn_score = float(scores[idx])
        rank_score = float(rank_scores[idx])
        prior_component = float(prior_np[idx])
        specificity_component = float(specificity_np[idx])

        # Real degree information (not fake docking)
        drug_degree = float(degrees[drug_idx].item())
        if degree_thresholds:
            if drug_degree <= degree_thresholds['q33']:
                degree_bucket = "low"
            elif drug_degree <= degree_thresholds['q66']:
                degree_bucket = "medium"
            else:
                degree_bucket = "high"
        else:
            degree_bucket = "unknown"

        # Check if ligand sdf exists
        safe_drug_name = drug_name.replace(' ', '_')
        ligand_path = f"../data/ligands/{safe_drug_name}.sdf"
        ligand_url = f"{BASE_URL}/data/ligands/{safe_drug_name}.sdf" if os.path.exists(ligand_path) else None

        results.append({
            "drug_node_idx": drug_idx,
            "drug_name": drug_name,
            "gnn_score": round(gnn_score, 4),
            "rank_score": round(rank_score, 4),
            "global_prior": round(prior_component, 4),
            "specificity": round(specificity_component, 4),
            "degree": int(drug_degree),
            "degree_bucket": degree_bucket,
            "ligand_url": ligand_url
        })

    return {
        "disease": target_disease_name,
        "selected_disease_node_idx": int(target_disease_idx),
        "disease_candidates": disease_candidates,
        "matched_query": req.disease,
        "match_type": match_type,
        "match_score": round(float(match_score), 4),
        "targets": targets,
        "predictions": results,
        "therapeutic_drug_count": len(base_candidate_drug_nodes),
        "candidate_count_after_filters": len(candidate_drug_nodes),
        "filtered_out_count": len(filtered_out),
        "filter_settings": {
            "exclude_contraindicated": req.exclude_contraindicated,
            "exclude_known_treatments": req.exclude_known_treatments,
            "exclude_categories": req.exclude_categories,
        },
        "rerank_settings": {
            "use_debias_rerank": bool(req.use_debias_rerank),
            "debias_alpha": round(float(max(0.0, min(req.debias_alpha, 1.5))), 4),
            "use_specificity_rerank": bool(req.use_specificity_rerank),
            "specificity_beta": round(float(max(0.0, min(req.specificity_beta, 3.0))), 4),
            "orphan_cap": round(float(max(1.0, req.orphan_cap)), 4),
            "use_hub_penalty": bool(req.use_hub_penalty),
            "hub_degree_quantile": round(float(max(0.5, min(req.hub_degree_quantile, 0.99))), 4),
            "hub_penalty_factor": round(float(max(0.1, min(req.hub_penalty_factor, 1.0))), 4),
            "hub_degree_threshold": (
                round(float(hub_threshold_value), 4) if hub_threshold_value is not None else None
            ),
            "use_disease_zscore": bool(req.use_disease_zscore),
            "prior_sampled_diseases": int(prior_sampled_diseases),
        },
    }
