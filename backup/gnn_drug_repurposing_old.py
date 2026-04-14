import os
import random
import re
import math
import json
import pickle
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from scipy.stats import spearmanr
from tqdm import tqdm

# ===== CONFIGURATION =====
PRIMEKG_URL = "https://dataverse.harvard.edu/api/access/datafile/6180620"
DATA_DIR = Path("data")
MODELS_DIR = Path("models")
PLOTS_DIR = MODELS_DIR / "plots"
DATASET_PATH = DATA_DIR / "primekg.csv"

# Training Settings
EPOCHS = 200
HIDDEN_DIM = 256
EMBEDDING_DIM = 128
DROPOUT = 0.2
DROPEDGE_RATE = 0.2
LR = 1e-3
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 512

# Data Settings
VAL_RATIO = 0.1
TEST_RATIO = 0.1
NEGATIVE_SAMPLE_RATIO = 5.0  # 5:1 negative to positive to reflect sparsity

SEED = 42

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure expected dirs exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ===== DATA PREPROCESSING =====

def download_primekg(url: str, destination: Path) -> Path:
    if destination.exists():
        return destination
    print(f"Downloading PrimeKG from {url}...")
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        with tqdm(total=total_size, unit='B', unit_scale=True) as pbar, destination.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                pbar.update(len(chunk))
    return destination

def load_and_standardize_primekg() -> Tuple[pd.DataFrame, pd.DataFrame]:
    path = download_primekg(PRIMEKG_URL, DATASET_PATH)
    print("Loading raw PrimeKG...")
    df_raw = pd.read_csv(path, sep=None, engine="python")
    
    def _norm(c): return re.sub(r"[^a-z0-9]", "", c.lower())
    cols = {_norm(c): c for c in df_raw.columns}
    
    def _pick(cands):
        for cand in cands:
            if _norm(cand) in cols: return cols[_norm(cand)]
            
    c_src_id = _pick(["x_id", "x_index", "source_id"])
    c_src_ty = _pick(["x_type", "source_type"])
    c_tgt_id = _pick(["y_id", "y_index", "target_id"])
    c_tgt_ty = _pick(["y_type", "target_type"])
    c_rel = _pick(["relation", "display_relation"])
    
    df = pd.DataFrame({
        "source_id": df_raw[c_src_id].astype(str),
        "source_type": df_raw[c_src_ty].astype(str).str.lower(),
        "target_id": df_raw[c_tgt_id].astype(str),
        "target_type": df_raw[c_tgt_ty].astype(str).str.lower(),
        "relation": df_raw[c_rel].astype(str).str.lower()
    }).dropna()
    
    return df_raw, df


# ===== GNN MODEL DEFS =====

def drop_edge(edge_index: torch.Tensor, p: float, force_training: bool = False) -> torch.Tensor:
    if p < 0. or p > 1.:
        raise ValueError(f"Drop probability has to be between 0 and 1, but got {p}")
    if not force_training:
        return edge_index
    num_edges = edge_index.size(1)
    # Mask proportion of edges
    mask = torch.rand(num_edges, device=edge_index.device) >= p
    return edge_index[:, mask]

def build_normalized_adjacency(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    src, dst = edge_index
    loop = torch.arange(num_nodes, device=edge_index.device)
    src_all = torch.cat([src, loop])
    dst_all = torch.cat([dst, loop])
    
    values = torch.ones(src_all.shape[0], device=edge_index.device)
    degree = torch.zeros(num_nodes, device=edge_index.device)
    degree.scatter_add_(0, src_all, values)
    
    deg_inv_sqrt = torch.pow(degree.clamp(min=1), -0.5)
    norm_values = deg_inv_sqrt[src_all] * values * deg_inv_sqrt[dst_all]
    
    return torch.sparse_coo_tensor(
        torch.stack([src_all, dst_all]),
        norm_values,
        (num_nodes, num_nodes)
    ).coalesce()

class GraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x_sparse = torch.sparse.mm(adj, x)
        return self.linear(x_sparse)

class ResidualGCNLayer(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.conv = GraphConv(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.conv(x, adj)
        h = self.norm(h)
        h = F.relu(h)
        h = self.dropout(h)
        return x + h  # Skip connection to prevent over-smoothing

class PrimeKGDrugRepurposingGNN(nn.Module):
    def __init__(self, num_nodes: int, num_types: int, hidden_dim: int, embedding_dim: int, dropout: float):
        super().__init__()
        self.node_embedding = nn.Embedding(num_nodes, hidden_dim)
        self.type_embedding = nn.Embedding(num_types, hidden_dim)
        
        self.gcn_in = GraphConv(hidden_dim, hidden_dim)
        self.res_layers = nn.ModuleList([
            ResidualGCNLayer(hidden_dim, dropout) for _ in range(2)
        ])
        
        self.gcn_out = GraphConv(hidden_dim, embedding_dim)
        
        # Link predictor combining src, dst, their product, AND log degrees
        self.link_predictor = nn.Sequential(
            nn.Linear(embedding_dim * 3 + 2, embedding_dim),
            nn.ReLU(),
            nn.BatchNorm1d(embedding_dim),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 1)
        )
        
    def encode(self, node_type_ids: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        idx = torch.arange(len(node_type_ids), device=node_type_ids.device)
        x = self.node_embedding(idx) + self.type_embedding(node_type_ids)
        
        x = F.relu(self.gcn_in(x, adj))
        for layer in self.res_layers:
            x = layer(x, adj)
            
        x = self.gcn_out(x, adj)
        return x
    
    def score(self, z: torch.Tensor, pairs: torch.Tensor, degrees: torch.Tensor) -> torch.Tensor:
        src_idx = pairs[0]
        dst_idx = pairs[1]
        
        src_z = z[src_idx]
        dst_z = z[dst_idx]
        
        # Log degree features (helps model discount hubs)
        src_deg = torch.log(degrees[src_idx].clamp(min=1).float()).unsqueeze(1)
        dst_deg = torch.log(degrees[dst_idx].clamp(min=1).float()).unsqueeze(1)
        
        feat = torch.cat([src_z, dst_z, src_z * dst_z, src_deg, dst_deg], dim=-1)
        return self.link_predictor(feat).squeeze(-1)


# ===== METRICS & EVALUATION =====

def compute_metrics(labels: np.ndarray, preds: np.ndarray, prefix: str = "") -> dict:
    auc = roc_auc_score(labels, preds)
    ap = average_precision_score(labels, preds)
    return {f"{prefix}auc": auc, f"{prefix}ap": ap}

def evaluate_ranking(model, z, pos_edges, drug_nodes, degree_tensor, batch_size=2048):
    """Computes MRR and Hits@K for positive test diseases."""
    model.eval()
    
    # Evaluate per disease
    disease_to_pos = defaultdict(list)
    for i in range(pos_edges.shape[1]):
        u, v = pos_edges[0, i].item(), pos_edges[1, i].item()
        # Ensure we group by disease (always assuming v is disease here due to setup, or handle safely; drug->disease)
        if v in drug_nodes.tolist():
            u, v = v, u  # swap so v is disease, u is drug
        disease_to_pos[v].append(u)
        
    mrr_sum = 0.0
    hits1, hits5, hits10 = 0, 0, 0
    total = 0
    all_drugs = drug_nodes.tolist()
    
    with torch.no_grad():
        for dis_idx, pos_drugs in disease_to_pos.items():
            # Candidate pairs: evaluate ALL drugs for this disease
            pairs = torch.stack([
                drug_nodes, 
                torch.full_like(drug_nodes, dis_idx)
            ], dim=0)
            
            # Batch inference to save memory
            scores = []
            for i in range(0, pairs.size(1), batch_size):
                batch_pairs = pairs[:, i:i+batch_size]
                batch_scores = model.score(z, batch_pairs, degree_tensor)
                scores.append(batch_scores)
            
            scores = torch.cat(scores).cpu().numpy()
            
            # Rank all drugs
            ranked_indices = scores.argsort()[::-1]
            ranked_drugs = drug_nodes[ranked_indices].cpu().numpy().tolist()
            
            for pos_drug in pos_drugs:
                rank = ranked_drugs.index(pos_drug) + 1
                mrr_sum += 1.0 / rank
                if rank <= 1: hits1 += 1
                if rank <= 5: hits5 += 1
                if rank <= 10: hits10 += 1
                total += 1
                
    mrr = mrr_sum / total if total > 0 else 0
    return {
        "mrr": mrr,
        "hits@1": hits1 / total if total > 0 else 0,
        "hits@5": hits5 / total if total > 0 else 0,
        "hits@10": hits10 / total if total > 0 else 0
    }

def print_metrics(metrics: dict, epoch: int, loss: float):
    print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | " + 
          " | ".join([f"{k}: {v:.4f}" for k,v in metrics.items() if isinstance(v, float)]))

# ===== MAIN EXECUTION =====

def main():
    df_raw, df = load_and_standardize_primekg()
    print("Extracting nodes & features...")
    
    # Metadata Name mapping
    disease_id_to_name, drug_id_to_name = {}, {}
    for _, row in df_raw.iterrows():
        if row['y_type'] == 'disease' and pd.notna(row['y_name']): disease_id_to_name[str(row['y_id'])] = str(row['y_name'])
        if row['x_type'] == 'disease' and pd.notna(row['x_name']): disease_id_to_name[str(row['x_id'])] = str(row['x_name'])
        if row['y_type'] == 'drug' and pd.notna(row['y_name']): drug_id_to_name[str(row['y_id'])] = str(row['y_name'])
        if row['x_type'] == 'drug' and pd.notna(row['x_name']): drug_id_to_name[str(row['x_id'])] = str(row['x_name'])
        
    src_keys = df["source_type"] + "::" + df["source_id"]
    tgt_keys = df["target_type"] + "::" + df["target_id"]
    all_keys = pd.Index(src_keys).append(pd.Index(tgt_keys)).unique().tolist()
    node_map = {k: i for i, k in enumerate(all_keys)}
    
    src_idx = src_keys.map(node_map).to_numpy()
    tgt_idx = tgt_keys.map(node_map).to_numpy()
    
    node_types = [k.split("::")[0] for k in all_keys]
    type_to_idx = {t: i for i, t in enumerate(set(node_types))}
    node_type_ids = torch.tensor([type_to_idx[t] for t in node_types]).to(device)
    
    drug_nodes = torch.tensor([i for i, t in enumerate(node_types) if "drug" in t]).to(device)
    disease_nodes = torch.tensor([i for i, t in enumerate(node_types) if "disease" in t]).to(device)
    
    # Calculate degree
    all_edges_full = torch.tensor([src_idx.tolist() + tgt_idx.tolist(), tgt_idx.tolist() + src_idx.tolist()], dtype=torch.long)
    degree_tensor = torch.zeros(len(all_keys), device=device)
    degree_tensor.scatter_add_(0, all_edges_full[0].to(device), torch.ones_like(all_edges_full[0], device=device, dtype=torch.float))
    
    # LEAKAGE FIX: Separate base graph edges from drug-disease prediction edges
    is_drug_disease = np.array([
        (node_types[s] == "drug" and node_types[t] == "disease") or
        (node_types[s] == "disease" and node_types[t] == "drug")
        for s, t in zip(src_idx, tgt_idx)
    ])
    
    # Edges that are strictly drug-disease
    dd_src = src_idx[is_drug_disease]
    dd_tgt = tgt_idx[is_drug_disease]
    
    # We enforce directed representation (drug -> disease) for positive pairs
    pos_pairs = set()
    for s, t in zip(dd_src, dd_tgt):
        if node_types[s] == "drug":
            pos_pairs.add((s, t))
        else:
            pos_pairs.add((t, s))
            
    pos_pairs_list = list(pos_pairs)
    random.shuffle(pos_pairs_list)
    print(f"Total positive drug-disease pairs: {len(pos_pairs_list):,}")
    
    n_val = int(len(pos_pairs_list) * VAL_RATIO)
    n_test = int(len(pos_pairs_list) * TEST_RATIO)
    test_pos = pos_pairs_list[:n_test]
    val_pos = pos_pairs_list[n_test:n_test + n_val]
    train_pos = pos_pairs_list[n_test + n_val:]
    
    # Edges used for Message Passing Adjacency (Base graph: include ALL non-drug-disease + ONLY train drug-disease)
    train_undir_s, train_undir_t = [], []
    for (s, t) in train_pos:
        train_undir_s.extend([s, t])
        train_undir_t.extend([t, s])
        
    base_src = src_idx[~is_drug_disease].tolist() + train_undir_s
    base_tgt = tgt_idx[~is_drug_disease].tolist() + train_undir_t
    
    # Graph structure for training representation (No leakage of test/val disease links!)
    base_edge_index = torch.tensor([base_src, base_tgt], dtype=torch.long).to(device)
    
    # NEGATIVE SAMPLING: Inverse-degree weighted to penalize hubs
    print("Generating degree-aware negative samples...")
    drug_deg = degree_tensor[drug_nodes].cpu().numpy()
    disease_deg = degree_tensor[disease_nodes].cpu().numpy()
    
    # Weight ~ 1/sqrt(degree). This focuses negatives more on hubs so they are penalized.
    d_weights = 1.0 / np.sqrt(np.clip(drug_deg, 1, None))
    d_probs = d_weights / d_weights.sum()
    
    def generate_negatives(num_neg, excluded_set):
        negatives = []
        # Vectorized sampling for speed
        while len(negatives) < num_neg:
            # Batch sample
            batch_sz = (num_neg - len(negatives)) * 2
            d_idx = np.random.choice(drug_nodes.cpu().numpy(), size=batch_sz, p=d_probs)
            di_idx = np.random.choice(disease_nodes.cpu().numpy(), size=batch_sz)
            
            for d, di in zip(d_idx, di_idx):
                if (d, di) not in excluded_set and (d, di) not in pos_pairs:
                    negatives.append((d, di))
                    excluded_set.add((d, di))
                    if len(negatives) >= num_neg: break
        return negatives
    
    train_neg = generate_negatives(int(len(train_pos) * NEGATIVE_SAMPLE_RATIO), set(train_pos))
    val_neg = generate_negatives(int(len(val_pos) * NEGATIVE_SAMPLE_RATIO), set(val_pos))
    test_neg = generate_negatives(int(len(test_pos) * NEGATIVE_SAMPLE_RATIO), set(test_pos))
    
    def _create_tensors(pos_list, neg_list):
        pairs = torch.tensor(pos_list + neg_list, dtype=torch.long).T
        labels = torch.cat([torch.ones(len(pos_list)), torch.zeros(len(neg_list))])
        return pairs.to(device), labels.to(device)
        
    train_pairs, train_labels = _create_tensors(train_pos, train_neg)
    val_pairs, val_labels = _create_tensors(val_pos, val_neg)
    test_pairs, test_labels = _create_tensors(test_pos, test_neg)
    
    print(f"Data Prep Complete | Train: {train_pairs.size(1)} | Val: {val_pairs.size(1)} | Test: {test_pairs.size(1)}")
    
    # ===== TRAINING =====
    model = PrimeKGDrugRepurposingGNN(len(all_keys), len(type_to_idx), HIDDEN_DIM, EMBEDDING_DIM, DROPOUT).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-5)
    
    best_val_mrr = 0.0
    patience = 15
    patience_cnt = 0
    history = defaultdict(list)
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        
        current_edge_index = drop_edge(base_edge_index, p=DROPEDGE_RATE, force_training=True)
        adj_train = build_normalized_adjacency(current_edge_index, len(all_keys)).to(device)
        
        z = model.encode(node_type_ids, adj_train)
        
        # Calculate all training edge logits at once (Full batch)
        # Randomly shuffle pairs for stochasticity in dropout
        perm = torch.randperm(train_pairs.size(1))
        batch_pairs = train_pairs[:, perm]
        batch_labels = train_labels[perm]
        
        logits = model.score(z, batch_pairs, degree_tensor)
        
        bce_loss = F.binary_cross_entropy_with_logits(logits, batch_labels)
        
        neg_mask = batch_labels == 0
        if neg_mask.sum() > 0:
            neg_srcs = batch_pairs[0, neg_mask]
            neg_scores = logits[neg_mask]
            neg_degs = torch.log(degree_tensor[neg_srcs].clamp(min=1))
            hub_penalty = 0.1 * (torch.sigmoid(neg_scores) * neg_degs).mean()
        else:
            hub_penalty = 0.0
            
        loss = bce_loss + hub_penalty
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        epoch_loss = loss.item()
        
        if epoch % 2 == 0:
            model.eval()
            with torch.no_grad():
                static_adj = build_normalized_adjacency(base_edge_index, len(all_keys)).coalesce().to(device)
                z_val = model.encode(node_type_ids, static_adj)
                val_logits = []
                for i in range(0, val_pairs.size(1), BATCH_SIZE):
                    batch_pairs = val_pairs[:, i:i+BATCH_SIZE]
                    val_logits.append(model.score(z_val, batch_pairs, degree_tensor))
                val_logits = torch.cat(val_logits)
                val_loss = F.binary_cross_entropy_with_logits(val_logits, val_labels).item()
                val_probs = torch.sigmoid(val_logits).cpu().numpy()
                
                metrics = compute_metrics(val_labels.cpu().numpy(), val_probs, prefix="val_")
                
                # Evaluate Ranking structure properly on positive pairs
                pos_val_edges = val_pairs[:, val_labels == 1]
                rank_metrics = evaluate_ranking(model, z_val, pos_val_edges, drug_nodes, degree_tensor, batch_size=2048)
                metrics.update({f"val_{k}": v for k, v in rank_metrics.items()})
                
            print_metrics(metrics, epoch, epoch_loss)
            
            scheduler.step(metrics["val_mrr"])
            
            history["loss"].append(epoch_loss)
            history["val_loss"].append(val_loss)
            history["val_auc"].append(metrics["val_auc"])
            history["val_mrr"].append(metrics["val_mrr"])
            history["val_hits10"].append(metrics["val_hits@10"])
            
            if metrics["val_mrr"] > best_val_mrr:
                best_val_mrr = metrics["val_mrr"]
                patience_cnt = 0
                torch.save({'model_state': model.state_dict(),
                            'model_config': {
                                'num_nodes': len(all_keys),
                                'num_types': len(type_to_idx),
                                'hidden_dim': HIDDEN_DIM,
                                'embedding_dim': EMBEDDING_DIM,
                                'dropout': DROPOUT
                            }}, MODELS_DIR / "gnn_drug_repurposing.pt")
                # Save static graph dependencies
                torch.save(build_normalized_adjacency(base_edge_index, len(all_keys)).coalesce().cpu(), MODELS_DIR / "adjacency.pt")
            else:
                patience_cnt += 1
                if patience_cnt >= patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break

    # ===== FINAL EVALUATION & PLOTS =====
    print("\n--- TEST EVALUATION ---")
    checkpoint = torch.load(MODELS_DIR / "gnn_drug_repurposing.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    
    with torch.no_grad():
        final_adj = build_normalized_adjacency(base_edge_index, len(all_keys)).coalesce().to(device)
        z_test = model.encode(node_type_ids, final_adj)
        
        test_logits = []
        for i in range(0, test_pairs.size(1), BATCH_SIZE):
            batch_pairs = test_pairs[:, i:i+BATCH_SIZE]
            test_logits.append(model.score(z_test, batch_pairs, degree_tensor))
        test_logits = torch.cat(test_logits)
        test_probs = torch.sigmoid(test_logits).cpu().numpy()
        
        test_metrics = compute_metrics(test_labels.cpu().numpy(), test_probs, prefix="test_")
        pos_test_edges = test_pairs[:, test_labels == 1]
        test_rank = evaluate_ranking(model, z_test, pos_test_edges, drug_nodes, degree_tensor, batch_size=2048)
        test_metrics.update({f"test_{k}": v for k, v in test_rank.items()})
        
    print_metrics(test_metrics, epoch=0, loss=0.0)
    
    # Save Metadata needed for backend
    metadata = {
        'all_keys': all_keys,
        'node_map': node_map,
        'node_types': node_types,
        'type_to_idx': type_to_idx,
        'drug_nodes': drug_nodes.cpu().tolist(),
        'disease_nodes': disease_nodes.cpu().tolist(),
        'disease_id_to_name': disease_id_to_name,
        'drug_id_to_name': drug_id_to_name,
    }
    with open(MODELS_DIR / "metadata.pkl", 'wb') as f:
        pickle.dump(metadata, f)
        
    torch.save(degree_tensor.cpu(), MODELS_DIR / "degrees.pt")
        
    with open(MODELS_DIR / "training_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    # Plotting
    try:
        # 1. Training Curves
        if len(history["loss"]) > 0:
            epochs_x = np.arange(2, 2 * len(history["loss"]) + 1, 2)
            fig, ax1 = plt.subplots(figsize=(8,5))
            ax1.plot(epochs_x, history["loss"], label='Train Loss', color='b')
            ax1.plot(epochs_x, history["val_loss"], label='Val Loss', color='c')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss', color='b')
            ax2 = ax1.twinx()
            ax2.plot(epochs_x, history["val_mrr"], label='Val MRR', color='r')
            ax2.set_ylabel('MRR', color='r')
            fig.legend(loc='center right', bbox_to_anchor=(0.85, 0.5))
            plt.title('Training Curves')
            plt.savefig(PLOTS_DIR / "training_curves.png")
            plt.close()
            
        # 2. Degree vs Score Rank Correlation (Spearman) on Test set predicting random diseases
        random_disease = disease_nodes[torch.randint(0, len(disease_nodes), (100,))]
        mean_scores = []
        degs = []
        with torch.no_grad():
            for d_node in random_disease:
                batch = torch.stack([drug_nodes, torch.full_like(drug_nodes, d_node)])
                s = torch.sigmoid(model.score(z_test, batch, degree_tensor)).cpu().numpy()
                mean_scores.append(s)
        
        mean_scores = np.mean(mean_scores, axis=0)
        degs = degree_tensor[drug_nodes].cpu().numpy()
        
        plt.figure(figsize=(6,6))
        plt.scatter(np.log10(degs + 1), mean_scores, alpha=0.3, s=5)
        rho, _ = spearmanr(degs, mean_scores)
        plt.xlabel('Log10(Degree + 1)')
        plt.ylabel('Global Mean Prediction Score')
        plt.title(f'Drug Degree vs Prediction Score ($\\rho={rho:.2f}$)')
        plt.savefig(PLOTS_DIR / "degree_vs_score.png")
        plt.close()

    except Exception as e:
        print(f"Skipping plot generation: {e}")
        
    print("\n✅ Training Complete. Models saved to ./models")

if __name__ == "__main__":
    main()
