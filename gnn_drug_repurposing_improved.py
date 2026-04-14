#!/usr/bin/env python3
"""PrimeKG GNN training pipeline with leakage-safe splits and hub-bias diagnostics."""

import argparse
import gc
import json
import math
import pickle
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.manifold import TSNE
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm


@dataclass
class TrainConfig:
    dataset_url: str = "https://dataverse.harvard.edu/api/access/datafile/6180620"
    data_dir: Path = Path("data")
    models_dir: Path = Path("models")
    dataset_path: Path = Path("data/primekg.csv")
    plots_dir: Path = Path("models/plots")

    seed: int = 42
    device: str = "auto"

    epochs: int = 200
    eval_every: int = 5

    hidden_dim: int = 128
    embedding_dim: int = 64
    dropout: float = 0.2
    dropedge_rate: float = 0.2

    lr: float = 1e-3
    weight_decay: float = 1e-5
    min_lr: float = 1e-5
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 5
    grad_clip_norm: float = 1.0

    val_ratio: float = 0.1
    test_ratio: float = 0.1
    negative_ratio: float = 3.0

    # p(drug) ∝ degree ** power ; -0.5 gives inverse-sqrt weighting.
    negative_drug_weight_power: float = -0.5

    batch_size: int = 2048
    patience: int = 15
    degree_corr_lambda: float = 0.1

    ranking_k: int = 10
    spearman_diseases: int = 30
    diversity_max_diseases: int = 30
    diversity_max_pairs: int = 500

    tsne_max_points: int = 500
    skip_tsne: bool = True


@dataclass
class NodeArtifacts:
    all_keys: List[str]
    node_map: Dict[str, int]
    node_types: List[str]
    type_to_idx: Dict[str, int]
    node_type_ids: torch.Tensor
    src_idx: np.ndarray
    tgt_idx: np.ndarray
    drug_nodes: torch.Tensor
    disease_nodes: torch.Tensor


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train leakage-safe PrimeKG GNN model")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--negative-ratio", type=float, default=3.0)
    parser.add_argument("--dropedge", type=float, default=0.2)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--run-tsne", action="store_true", help="Enable t-SNE plot (uses extra memory)")
    args = parser.parse_args()

    config = TrainConfig(
        device=args.device,
        epochs=args.epochs,
        seed=args.seed,
        negative_ratio=args.negative_ratio,
        dropedge_rate=args.dropedge,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        eval_every=args.eval_every,
        skip_tsne=not args.run_tsne,
    )
    config.dataset_path = config.data_dir / "primekg.csv"
    config.plots_dir = config.models_dir / "plots"
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no GPU is available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dirs(config: TrainConfig) -> None:
    config.data_dir.mkdir(parents=True, exist_ok=True)
    config.models_dir.mkdir(parents=True, exist_ok=True)
    config.plots_dir.mkdir(parents=True, exist_ok=True)


def _norm_col(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _pick_column(df: pd.DataFrame, candidates: Sequence[str], required: bool = True) -> Optional[str]:
    normalized = {_norm_col(c): c for c in df.columns}
    for candidate in candidates:
        key = _norm_col(candidate)
        if key in normalized:
            return normalized[key]
    if required:
        raise KeyError(f"Missing required column. Tried candidates: {list(candidates)}")
    return None


def download_primekg(url: str, destination: Path) -> Path:
    if destination.exists():
        return destination

    print(f"Downloading PrimeKG from {url}")
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        with destination.open("wb") as file_obj, tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            desc="primekg.csv",
        ) as progress:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                file_obj.write(chunk)
                progress.update(len(chunk))

    return destination


def load_and_standardize_primekg(config: TrainConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dataset_path = download_primekg(config.dataset_url, config.dataset_path)
    print(f"Loading PrimeKG from {dataset_path}")
    # Use C engine for speed and lower memory; fall back to Python engine if needed
    try:
        raw = pd.read_csv(dataset_path, low_memory=True)
    except Exception:
        raw = pd.read_csv(dataset_path, sep=None, engine="python")

    c_src_id = _pick_column(raw, ["x_id", "x_index", "source_id"])
    c_src_type = _pick_column(raw, ["x_type", "source_type"])
    c_tgt_id = _pick_column(raw, ["y_id", "y_index", "target_id"])
    c_tgt_type = _pick_column(raw, ["y_type", "target_type"])
    c_rel = _pick_column(raw, ["relation", "display_relation"], required=False)

    standardized = pd.DataFrame(
        {
            "source_id": raw[c_src_id].astype(str),
            "source_type": raw[c_src_type].astype(str).str.lower(),
            "target_id": raw[c_tgt_id].astype(str),
            "target_type": raw[c_tgt_type].astype(str).str.lower(),
            "relation": raw[c_rel].astype(str).str.lower() if c_rel else "unknown",
        }
    ).dropna()

    standardized = standardized[
        standardized["source_id"].astype(str).str.len().gt(0)
        & standardized["target_id"].astype(str).str.len().gt(0)
    ]
    standardized = standardized.reset_index(drop=True)

    return raw, standardized


def extract_entity_name_maps(df_raw: pd.DataFrame) -> Tuple[Dict[str, str], Dict[str, str]]:
    disease_id_to_name: Dict[str, str] = {}
    drug_id_to_name: Dict[str, str] = {}

    c_x_id = _pick_column(df_raw, ["x_id", "x_index", "source_id"], required=False)
    c_x_type = _pick_column(df_raw, ["x_type", "source_type"], required=False)
    c_x_name = _pick_column(df_raw, ["x_name", "source_name", "xlabel"], required=False)

    c_y_id = _pick_column(df_raw, ["y_id", "y_index", "target_id"], required=False)
    c_y_type = _pick_column(df_raw, ["y_type", "target_type"], required=False)
    c_y_name = _pick_column(df_raw, ["y_name", "target_name", "ylabel"], required=False)

    def ingest(id_col: Optional[str], type_col: Optional[str], name_col: Optional[str]) -> None:
        if id_col is None or type_col is None or name_col is None:
            return

        subset = df_raw[[id_col, type_col, name_col]].dropna()
        for ent_id, ent_type, ent_name in subset.itertuples(index=False):
            et = str(ent_type).lower()
            if "disease" in et:
                disease_id_to_name.setdefault(str(ent_id), str(ent_name))
            if "drug" in et:
                drug_id_to_name.setdefault(str(ent_id), str(ent_name))

    ingest(c_x_id, c_x_type, c_x_name)
    ingest(c_y_id, c_y_type, c_y_name)

    return disease_id_to_name, drug_id_to_name


def build_node_artifacts(df: pd.DataFrame) -> NodeArtifacts:
    src_keys = df["source_type"] + "::" + df["source_id"]
    tgt_keys = df["target_type"] + "::" + df["target_id"]

    all_keys = pd.Index(src_keys).append(pd.Index(tgt_keys)).unique().tolist()
    node_map = {key: i for i, key in enumerate(all_keys)}

    src_idx = src_keys.map(node_map).to_numpy(dtype=np.int64)
    tgt_idx = tgt_keys.map(node_map).to_numpy(dtype=np.int64)

    node_types = [key.split("::", 1)[0] for key in all_keys]
    type_to_idx = {node_type: i for i, node_type in enumerate(sorted(set(node_types)))}
    node_type_ids = torch.tensor([type_to_idx[t] for t in node_types], dtype=torch.long)

    drug_nodes = torch.tensor(
        [i for i, node_type in enumerate(node_types) if "drug" in node_type],
        dtype=torch.long,
    )
    disease_nodes = torch.tensor(
        [i for i, node_type in enumerate(node_types) if "disease" in node_type],
        dtype=torch.long,
    )

    if len(drug_nodes) == 0 or len(disease_nodes) == 0:
        raise RuntimeError("Could not identify drug and disease node types from PrimeKG.")

    return NodeArtifacts(
        all_keys=all_keys,
        node_map=node_map,
        node_types=node_types,
        type_to_idx=type_to_idx,
        node_type_ids=node_type_ids,
        src_idx=src_idx,
        tgt_idx=tgt_idx,
        drug_nodes=drug_nodes,
        disease_nodes=disease_nodes,
    )


def extract_drug_disease_edges(
    src_idx: np.ndarray,
    tgt_idx: np.ndarray,
    node_types: Sequence[str],
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    is_drug_disease = np.zeros(len(src_idx), dtype=bool)
    directed_pairs: Set[Tuple[int, int]] = set()

    for i, (src, tgt) in enumerate(zip(src_idx, tgt_idx)):
        src_type = node_types[int(src)]
        tgt_type = node_types[int(tgt)]

        src_is_drug = "drug" in src_type
        src_is_disease = "disease" in src_type
        tgt_is_drug = "drug" in tgt_type
        tgt_is_disease = "disease" in tgt_type

        if src_is_drug and tgt_is_disease:
            is_drug_disease[i] = True
            directed_pairs.add((int(src), int(tgt)))
        elif src_is_disease and tgt_is_drug:
            is_drug_disease[i] = True
            directed_pairs.add((int(tgt), int(src)))

    return sorted(directed_pairs), is_drug_disease


def split_positive_edges(
    positive_edges: List[Tuple[int, int]],
    val_ratio: float,
    test_ratio: float,
    rng: np.random.Generator,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]]]:
    edges = list(positive_edges)
    rng.shuffle(edges)

    n_total = len(edges)
    n_val = int(n_total * val_ratio)
    n_test = int(n_total * test_ratio)

    if n_total - n_val - n_test <= 0:
        raise RuntimeError("Split ratios leave no training positives. Reduce val/test ratios.")

    test_pos = edges[:n_test]
    val_pos = edges[n_test : n_test + n_val]
    train_pos = edges[n_test + n_val :]

    return train_pos, val_pos, test_pos


def build_train_base_edge_index(
    src_idx: np.ndarray,
    tgt_idx: np.ndarray,
    is_drug_disease: np.ndarray,
    train_pos_edges: Sequence[Tuple[int, int]],
    device: torch.device,
) -> torch.Tensor:
    non_dd_src = src_idx[~is_drug_disease]
    non_dd_tgt = tgt_idx[~is_drug_disease]

    train_drug = np.array([s for s, _ in train_pos_edges], dtype=np.int64)
    train_disease = np.array([t for _, t in train_pos_edges], dtype=np.int64)

    # Undirected message-passing graph: all non drug-disease edges + only TRAIN drug-disease edges.
    base_src = np.concatenate([non_dd_src, non_dd_tgt, train_drug, train_disease])
    base_tgt = np.concatenate([non_dd_tgt, non_dd_src, train_disease, train_drug])

    edge_index = torch.tensor(np.vstack([base_src, base_tgt]), dtype=torch.long, device=device)
    return edge_index


def compute_degrees(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    degrees = torch.zeros(num_nodes, dtype=torch.float, device=edge_index.device)
    ones = torch.ones(edge_index.shape[1], dtype=torch.float, device=edge_index.device)
    degrees.scatter_add_(0, edge_index[0], ones)
    return degrees


def build_negative_sampling_probs(
    degrees: torch.Tensor,
    drug_nodes: torch.Tensor,
    power: float,
) -> np.ndarray:
    drug_degrees = degrees[drug_nodes].detach().cpu().numpy()
    drug_degrees = np.clip(drug_degrees, 1.0, None)
    weights = np.power(drug_degrees, power)

    if np.any(~np.isfinite(weights)) or weights.sum() <= 0:
        weights = np.ones_like(drug_degrees)

    return weights / weights.sum()


def sample_negative_edges(
    num_samples: int,
    drug_nodes_np: np.ndarray,
    disease_nodes_np: np.ndarray,
    drug_probs: np.ndarray,
    blocked_pairs: Set[Tuple[int, int]],
    rng: np.random.Generator,
    split_name: str,
) -> List[Tuple[int, int]]:
    negatives: List[Tuple[int, int]] = []
    local_blocked: Set[Tuple[int, int]] = set()

    attempts = 0
    max_attempts = max(10000, num_samples * 200)

    while len(negatives) < num_samples:
        remaining = num_samples - len(negatives)
        batch_size = min(max(remaining * 4, 2048), 50000)

        sampled_drugs = rng.choice(drug_nodes_np, size=batch_size, replace=True, p=drug_probs)
        sampled_diseases = rng.choice(disease_nodes_np, size=batch_size, replace=True)

        for drug_idx, disease_idx in zip(sampled_drugs, sampled_diseases):
            pair = (int(drug_idx), int(disease_idx))
            attempts += 1
            if pair in blocked_pairs or pair in local_blocked:
                continue

            local_blocked.add(pair)
            negatives.append(pair)
            if len(negatives) >= num_samples:
                break

        if attempts > max_attempts and len(negatives) < num_samples:
            raise RuntimeError(
                f"Negative sampling stalled for {split_name}. "
                f"Requested {num_samples}, generated {len(negatives)}."
            )

    blocked_pairs.update(local_blocked)
    return negatives


def create_pair_tensors(
    pos_edges: Sequence[Tuple[int, int]],
    neg_edges: Sequence[Tuple[int, int]],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    pairs = torch.tensor(list(pos_edges) + list(neg_edges), dtype=torch.long, device=device).T
    labels = torch.cat(
        [
            torch.ones(len(pos_edges), dtype=torch.float, device=device),
            torch.zeros(len(neg_edges), dtype=torch.float, device=device),
        ]
    )
    return pairs, labels


def drop_edge(edge_index: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    if not training or drop_prob <= 0:
        return edge_index
    if drop_prob >= 1:
        raise ValueError("drop_prob must be < 1.0")

    mask = torch.rand(edge_index.shape[1], device=edge_index.device) >= drop_prob
    return edge_index[:, mask]


def build_normalized_adjacency(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    src, dst = edge_index
    loop = torch.arange(num_nodes, device=edge_index.device)

    src_all = torch.cat([src, loop])
    dst_all = torch.cat([dst, loop])

    values = torch.ones(src_all.shape[0], device=edge_index.device)

    degree = torch.zeros(num_nodes, device=edge_index.device)
    degree.scatter_add_(0, src_all, values)

    deg_inv_sqrt = degree.clamp(min=1).pow(-0.5)
    norm_values = deg_inv_sqrt[src_all] * values * deg_inv_sqrt[dst_all]

    return torch.sparse_coo_tensor(
        torch.stack([src_all, dst_all]),
        norm_values,
        size=(num_nodes, num_nodes),
    ).coalesce()


class GraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = torch.sparse.mm(adj, x)
        return self.linear(x)


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
        return x + h


class PrimeKGDrugRepurposingGNN(nn.Module):
    def __init__(self, num_nodes: int, num_types: int, hidden_dim: int, embedding_dim: int, dropout: float):
        super().__init__()
        self.node_embedding = nn.Embedding(num_nodes, hidden_dim)
        self.type_embedding = nn.Embedding(num_types, hidden_dim)

        # 3 hidden GCN stages with residual processing.
        self.gcn_in = GraphConv(hidden_dim, hidden_dim)
        self.res_layers = nn.ModuleList([ResidualGCNLayer(hidden_dim, dropout) for _ in range(2)])
        self.gcn_out = GraphConv(hidden_dim, embedding_dim)

        # src, dst, elementwise product, and log-degree features.
        self.link_predictor = nn.Sequential(
            nn.Linear(embedding_dim * 3 + 2, embedding_dim),
            nn.ReLU(),
            nn.BatchNorm1d(embedding_dim),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 1),
        )

    def encode(self, node_type_ids: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        idx = torch.arange(len(node_type_ids), device=node_type_ids.device)
        x = self.node_embedding(idx) + self.type_embedding(node_type_ids)

        x = F.relu(self.gcn_in(x, adj))
        for layer in self.res_layers:
            x = layer(x, adj)

        return self.gcn_out(x, adj)

    def score(self, z: torch.Tensor, pairs: torch.Tensor, degrees: torch.Tensor) -> torch.Tensor:
        src_idx = pairs[0]
        dst_idx = pairs[1]

        src_z = z[src_idx]
        dst_z = z[dst_idx]

        src_deg = torch.log(degrees[src_idx].clamp(min=1).float()).unsqueeze(1)
        dst_deg = torch.log(degrees[dst_idx].clamp(min=1).float()).unsqueeze(1)

        features = torch.cat([src_z, dst_z, src_z * dst_z, src_deg, dst_deg], dim=-1)
        return self.link_predictor(features).squeeze(-1)


def predict_logits(
    model: PrimeKGDrugRepurposingGNN,
    z: torch.Tensor,
    pairs: torch.Tensor,
    degrees: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    logits = []
    for i in range(0, pairs.shape[1], batch_size):
        batch_pairs = pairs[:, i : i + batch_size]
        logits.append(model.score(z, batch_pairs, degrees))
    return torch.cat(logits, dim=0)


def safe_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    labels_int = labels.astype(int)
    if np.unique(labels_int).shape[0] < 2:
        return float("nan")
    return float(roc_auc_score(labels_int, probs))


def safe_ap(labels: np.ndarray, probs: np.ndarray) -> float:
    labels_int = labels.astype(int)
    if np.unique(labels_int).shape[0] < 2:
        return float("nan")
    return float(average_precision_score(labels_int, probs))


def compute_binary_metrics(labels: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    return {
        "auc": safe_auc(labels, probs),
        "ap": safe_ap(labels, probs),
    }


def degree_correlation_regularizer(
    logits: torch.Tensor,
    pairs: torch.Tensor,
    degrees: torch.Tensor,
) -> torch.Tensor:
    scores = torch.sigmoid(logits)
    src_degree = torch.log(degrees[pairs[0]].clamp(min=1).float())

    scores_centered = scores - scores.mean()
    degree_centered = src_degree - src_degree.mean()

    denom = torch.sqrt(scores_centered.pow(2).mean() * degree_centered.pow(2).mean() + 1e-8)
    corr = (scores_centered * degree_centered).mean() / denom
    return corr.abs()


def score_all_drugs_for_disease(
    model: PrimeKGDrugRepurposingGNN,
    z: torch.Tensor,
    drug_nodes: torch.Tensor,
    disease_node: int,
    degrees: torch.Tensor,
    batch_size: int,
) -> np.ndarray:
    disease_tensor = torch.full_like(drug_nodes, int(disease_node))
    pairs = torch.stack([drug_nodes, disease_tensor], dim=0)
    logits = predict_logits(model, z, pairs, degrees, batch_size)
    return torch.sigmoid(logits).detach().cpu().numpy()


def evaluate_ranking_metrics(
    model: PrimeKGDrugRepurposingGNN,
    z: torch.Tensor,
    positive_edges: Sequence[Tuple[int, int]],
    drug_nodes: torch.Tensor,
    degrees: torch.Tensor,
    top_k: int,
    batch_size: int,
    collect_topk: bool = False,
) -> Tuple[Dict[str, float], Dict[int, List[int]]]:
    disease_to_pos: Dict[int, Set[int]] = defaultdict(set)
    for drug_node, disease_node in positive_edges:
        disease_to_pos[int(disease_node)].add(int(drug_node))

    drug_nodes_np = drug_nodes.detach().cpu().numpy()
    drug_pos_map = {int(node): i for i, node in enumerate(drug_nodes_np)}

    mrr_sum = 0.0
    hits1 = 0
    hits5 = 0
    hits10 = 0
    total_positive = 0

    precision_sum = 0.0
    recall_sum = 0.0
    disease_count = 0

    disease_topk: Dict[int, List[int]] = {}

    with torch.no_grad():
        for disease_node, positive_drugs in disease_to_pos.items():
            scores = score_all_drugs_for_disease(
                model=model,
                z=z,
                drug_nodes=drug_nodes,
                disease_node=disease_node,
                degrees=degrees,
                batch_size=batch_size,
            )

            ranked_positions = np.argsort(scores)[::-1]
            rank_lookup = np.empty_like(ranked_positions)
            rank_lookup[ranked_positions] = np.arange(1, len(ranked_positions) + 1)

            for drug_node in positive_drugs:
                candidate_idx = drug_pos_map.get(int(drug_node))
                if candidate_idx is None:
                    continue
                rank = int(rank_lookup[candidate_idx])
                mrr_sum += 1.0 / rank
                hits1 += int(rank <= 1)
                hits5 += int(rank <= 5)
                hits10 += int(rank <= 10)
                total_positive += 1

            topk_candidate_idx = ranked_positions[:top_k]
            topk_drug_nodes = drug_nodes_np[topk_candidate_idx]
            topk_set = set(int(x) for x in topk_drug_nodes.tolist())

            true_positives = len(positive_drugs.intersection(topk_set))
            precision_sum += true_positives / max(top_k, 1)
            recall_sum += true_positives / max(len(positive_drugs), 1)
            disease_count += 1

            if collect_topk:
                disease_topk[disease_node] = [int(x) for x in topk_drug_nodes.tolist()]

    metrics = {
        "mrr": mrr_sum / max(total_positive, 1),
        "hits@1": hits1 / max(total_positive, 1),
        "hits@5": hits5 / max(total_positive, 1),
        "hits@10": hits10 / max(total_positive, 1),
        f"precision@{top_k}": precision_sum / max(disease_count, 1),
        f"recall@{top_k}": recall_sum / max(disease_count, 1),
        "ranking_eval_diseases": float(disease_count),
        "ranking_eval_edges": float(total_positive),
    }

    return metrics, disease_topk


def evaluate_degree_stratified_metrics(
    pairs: torch.Tensor,
    labels_np: np.ndarray,
    probs_np: np.ndarray,
    degrees: torch.Tensor,
    drug_nodes: torch.Tensor,
) -> Dict[str, Dict[str, float]]:
    source_degrees = degrees[pairs[0]].detach().cpu().numpy()
    all_drug_degrees = degrees[drug_nodes].detach().cpu().numpy()

    q33, q66 = np.quantile(all_drug_degrees, [0.33, 0.66])

    masks = {
        "low": source_degrees <= q33,
        "medium": (source_degrees > q33) & (source_degrees <= q66),
        "high": source_degrees > q66,
    }

    out: Dict[str, Dict[str, float]] = {
        "thresholds": {
            "q33": float(q33),
            "q66": float(q66),
        }
    }

    for bucket_name, mask in masks.items():
        if mask.sum() == 0:
            out[bucket_name] = {
                "count": 0.0,
                "positive_rate": float("nan"),
                "auc": float("nan"),
                "ap": float("nan"),
            }
            continue

        y_true = labels_np[mask]
        y_prob = probs_np[mask]
        out[bucket_name] = {
            "count": float(mask.sum()),
            "positive_rate": float(y_true.mean()),
            "auc": safe_auc(y_true, y_prob),
            "ap": safe_ap(y_true, y_prob),
        }

    return out


def compute_degree_score_bias(
    model: PrimeKGDrugRepurposingGNN,
    z: torch.Tensor,
    drug_nodes: torch.Tensor,
    disease_nodes: torch.Tensor,
    degrees: torch.Tensor,
    batch_size: int,
    max_diseases: int,
    rng: np.random.Generator,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    disease_nodes_np = disease_nodes.detach().cpu().numpy()
    num_sample = min(max_diseases, len(disease_nodes_np))

    if num_sample == 0:
        raise RuntimeError("No disease nodes available for bias evaluation.")

    sampled_diseases = rng.choice(disease_nodes_np, size=num_sample, replace=False)

    scores_per_disease = []
    with torch.no_grad():
        for disease_node in sampled_diseases:
            disease_scores = score_all_drugs_for_disease(
                model=model,
                z=z,
                drug_nodes=drug_nodes,
                disease_node=int(disease_node),
                degrees=degrees,
                batch_size=batch_size,
            )
            scores_per_disease.append(disease_scores)

    mean_scores = np.mean(np.stack(scores_per_disease, axis=0), axis=0)
    drug_degrees = degrees[drug_nodes].detach().cpu().numpy()

    rho, p_val = spearmanr(drug_degrees, mean_scores)
    metrics = {
        "spearman_rho": float(rho),
        "spearman_p_value": float(p_val),
        "sampled_diseases": float(num_sample),
    }

    return metrics, drug_degrees, mean_scores


def compute_topk_diversity(
    model: PrimeKGDrugRepurposingGNN,
    z: torch.Tensor,
    drug_nodes: torch.Tensor,
    disease_nodes: torch.Tensor,
    degrees: torch.Tensor,
    top_k: int,
    batch_size: int,
    max_diseases: int,
    max_pairs: int,
    rng: np.random.Generator,
) -> Tuple[Dict[str, float], List[float], Dict[int, List[int]]]:
    disease_nodes_np = disease_nodes.detach().cpu().numpy()
    num_sample = min(max_diseases, len(disease_nodes_np))
    sampled_diseases = rng.choice(disease_nodes_np, size=num_sample, replace=False)

    disease_topk: Dict[int, List[int]] = {}

    with torch.no_grad():
        for disease_node in sampled_diseases:
            scores = score_all_drugs_for_disease(
                model=model,
                z=z,
                drug_nodes=drug_nodes,
                disease_node=int(disease_node),
                degrees=degrees,
                batch_size=batch_size,
            )
            ranked_positions = np.argsort(scores)[::-1]
            topk_positions = ranked_positions[:top_k]
            topk_drugs = drug_nodes.detach().cpu().numpy()[topk_positions]
            disease_topk[int(disease_node)] = [int(x) for x in topk_drugs.tolist()]

    disease_ids = list(disease_topk.keys())
    if len(disease_ids) < 2:
        return {
            "mean_jaccard": float("nan"),
            "median_jaccard": float("nan"),
            "p90_jaccard": float("nan"),
            "pairs_compared": 0.0,
            "sampled_diseases": float(len(disease_ids)),
            "top1_mode_fraction": float("nan"),
        }, [], disease_topk

    pair_indices = list(combinations(range(len(disease_ids)), 2))
    if len(pair_indices) > max_pairs:
        selected = rng.choice(len(pair_indices), size=max_pairs, replace=False)
        pair_indices = [pair_indices[int(i)] for i in selected]

    topk_sets = {
        disease_id: set(disease_topk[disease_id])
        for disease_id in disease_ids
    }

    jaccards: List[float] = []
    for i, j in pair_indices:
        a = topk_sets[disease_ids[i]]
        b = topk_sets[disease_ids[j]]
        inter = len(a.intersection(b))
        union = len(a.union(b))
        jaccards.append(inter / max(union, 1))

    top1_counter = Counter(topk[0] for topk in disease_topk.values() if topk)
    top1_mode_fraction = 0.0
    if top1_counter:
        top1_mode_fraction = top1_counter.most_common(1)[0][1] / max(len(disease_topk), 1)

    metrics = {
        "mean_jaccard": float(np.mean(jaccards)),
        "median_jaccard": float(np.median(jaccards)),
        "p90_jaccard": float(np.quantile(jaccards, 0.9)),
        "pairs_compared": float(len(jaccards)),
        "sampled_diseases": float(len(disease_topk)),
        "top1_mode_fraction": float(top1_mode_fraction),
    }

    return metrics, jaccards, disease_topk


def plot_training_curves(history: Dict[str, List[float]], output_path: Path) -> None:
    if plt is None:
        return
    if not history.get("epoch"):
        return

    epochs = history["epoch"]
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss", color="#1f77b4")
    plt.plot(epochs, history["val_loss"], label="Val Loss", color="#17becf")

    ax2 = plt.gca().twinx()
    ax2.plot(epochs, history["val_mrr"], label="Val MRR", color="#d62728")
    ax2.set_ylabel("MRR")

    plt.title("Training Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    lines, labels = plt.gca().get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(lines + lines2, labels + labels2, loc="center right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_degree_distribution(drug_degrees: np.ndarray, output_path: Path) -> None:
    if plt is None:
        return
    plt.figure(figsize=(8, 5))
    plt.hist(drug_degrees, bins=50, color="#4c78a8", alpha=0.85)
    plt.title("Drug Degree Distribution")
    plt.xlabel("Degree")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_degree_vs_score(
    drug_degrees: np.ndarray,
    mean_scores: np.ndarray,
    rho: float,
    output_path: Path,
) -> None:
    if plt is None:
        return
    plt.figure(figsize=(7, 6))
    plt.scatter(np.log10(drug_degrees + 1.0), mean_scores, s=8, alpha=0.35, color="#1f77b4")
    plt.title(f"Degree vs Score (Spearman rho={rho:.3f})")
    plt.xlabel("log10(Degree + 1)")
    plt.ylabel("Average Predicted Score")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_roc_pr_curves(labels: np.ndarray, probs: np.ndarray, output_path: Path) -> None:
    if plt is None:
        return
    labels_int = labels.astype(int)
    if np.unique(labels_int).shape[0] < 2:
        return

    fpr, tpr, _ = roc_curve(labels_int, probs)
    precision, recall, _ = precision_recall_curve(labels_int, probs)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(fpr, tpr, color="#1f77b4")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    axes[0].set_title("ROC Curve")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")

    axes[1].plot(recall, precision, color="#ff7f0e")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_degree_stratified_metrics(
    stratified_metrics: Dict[str, Dict[str, float]],
    output_path: Path,
) -> None:
    if plt is None:
        return
    buckets = ["low", "medium", "high"]
    auc_values = []

    for bucket in buckets:
        auc = stratified_metrics.get(bucket, {}).get("auc", float("nan"))
        auc_values.append(0.0 if not np.isfinite(auc) else float(auc))

    plt.figure(figsize=(7, 5))
    bars = plt.bar(buckets, auc_values, color=["#4c78a8", "#72b7b2", "#f58518"])
    plt.ylim(0.0, 1.0)
    plt.title("Degree-Stratified Test AUC")
    plt.ylabel("AUC")

    for bar, value in zip(bars, auc_values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center")

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_embedding_tsne(
    z: torch.Tensor,
    drug_nodes: torch.Tensor,
    degrees: torch.Tensor,
    output_path: Path,
    max_points: int,
    seed: int,
) -> None:
    if plt is None:
        return
    embeddings = z[drug_nodes].detach().cpu().numpy()
    drug_degrees = degrees[drug_nodes].detach().cpu().numpy()

    if embeddings.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        selected = rng.choice(embeddings.shape[0], size=max_points, replace=False)
        embeddings = embeddings[selected]
        drug_degrees = drug_degrees[selected]

    if embeddings.shape[0] < 5:
        return

    perplexity = max(5, min(30, (embeddings.shape[0] - 1) // 3))
    tsne = TSNE(
        n_components=2,
        random_state=seed,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
    )
    coords = tsne.fit_transform(embeddings)

    plt.figure(figsize=(7, 6))
    sc = plt.scatter(
        coords[:, 0],
        coords[:, 1],
        c=np.log1p(drug_degrees),
        cmap="viridis",
        s=10,
        alpha=0.8,
    )
    cbar = plt.colorbar(sc)
    cbar.set_label("log(1 + degree)")
    plt.title("Drug Embedding t-SNE")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_topk_diversity(jaccard_values: List[float], output_path: Path) -> None:
    if plt is None:
        return
    if not jaccard_values:
        return

    plt.figure(figsize=(8, 5))
    plt.hist(jaccard_values, bins=30, color="#54a24b", alpha=0.85)
    plt.title("Top-K Diversity Across Disease Pairs")
    plt.xlabel("Jaccard Similarity")
    plt.ylabel("Pair Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def json_ready(obj):
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_ready(v) for v in obj]
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def node_to_entity_name(
    node_idx: int,
    all_keys: Sequence[str],
    id_to_name: Dict[str, str],
) -> str:
    key = all_keys[int(node_idx)]
    entity_id = key.split("::", 1)[1]
    return id_to_name.get(entity_id, entity_id)


def main() -> None:
    config = parse_args()
    set_seed(config.seed)
    ensure_dirs(config)

    device = resolve_device(config.device)
    print(f"Using device: {device}")

    rng = np.random.default_rng(config.seed)

    # -------------------------------
    # Data ingestion and graph setup
    # -------------------------------
    df_raw, df = load_and_standardize_primekg(config)
    disease_id_to_name, drug_id_to_name = extract_entity_name_maps(df_raw)
    del df_raw  # Free ~600MB of raw PrimeKG data
    gc.collect()

    node_artifacts = build_node_artifacts(df)
    del df  # Free standardized DataFrame
    gc.collect()
    num_nodes = len(node_artifacts.all_keys)

    positive_edges, is_drug_disease = extract_drug_disease_edges(
        node_artifacts.src_idx,
        node_artifacts.tgt_idx,
        node_artifacts.node_types,
    )
    print(f"Total unique drug-disease positives: {len(positive_edges):,}")

    train_pos, val_pos, test_pos = split_positive_edges(
        positive_edges,
        config.val_ratio,
        config.test_ratio,
        rng,
    )
    print(
        "Split positives | "
        f"Train: {len(train_pos):,}  "
        f"Val: {len(val_pos):,}  "
        f"Test: {len(test_pos):,}"
    )

    base_edge_index = build_train_base_edge_index(
        src_idx=node_artifacts.src_idx,
        tgt_idx=node_artifacts.tgt_idx,
        is_drug_disease=is_drug_disease,
        train_pos_edges=train_pos,
        device=device,
    )
    del is_drug_disease
    node_artifacts.src_idx = None
    node_artifacts.tgt_idx = None

    static_adj = build_normalized_adjacency(base_edge_index, num_nodes).to(device)
    degree_tensor = compute_degrees(base_edge_index, num_nodes)
    del base_edge_index  # No longer needed after adjacency is built
    gc.collect()
    print(f"Graph: {num_nodes:,} nodes | Memory freed after adjacency build")

    drug_sampling_probs = build_negative_sampling_probs(
        degrees=degree_tensor,
        drug_nodes=node_artifacts.drug_nodes.to(device),
        power=config.negative_drug_weight_power,
    )

    blocked_pairs: Set[Tuple[int, int]] = set(positive_edges)
    drug_nodes_np = node_artifacts.drug_nodes.detach().cpu().numpy()
    disease_nodes_np = node_artifacts.disease_nodes.detach().cpu().numpy()

    train_neg = sample_negative_edges(
        num_samples=int(len(train_pos) * config.negative_ratio),
        drug_nodes_np=drug_nodes_np,
        disease_nodes_np=disease_nodes_np,
        drug_probs=drug_sampling_probs,
        blocked_pairs=blocked_pairs,
        rng=rng,
        split_name="train",
    )
    val_neg = sample_negative_edges(
        num_samples=int(len(val_pos) * config.negative_ratio),
        drug_nodes_np=drug_nodes_np,
        disease_nodes_np=disease_nodes_np,
        drug_probs=drug_sampling_probs,
        blocked_pairs=blocked_pairs,
        rng=rng,
        split_name="val",
    )
    test_neg = sample_negative_edges(
        num_samples=int(len(test_pos) * config.negative_ratio),
        drug_nodes_np=drug_nodes_np,
        disease_nodes_np=disease_nodes_np,
        drug_probs=drug_sampling_probs,
        blocked_pairs=blocked_pairs,
        rng=rng,
        split_name="test",
    )
    del blocked_pairs, drug_sampling_probs, drug_nodes_np, disease_nodes_np
    gc.collect()

    train_pairs, train_labels = create_pair_tensors(train_pos, train_neg, device)
    val_pairs, val_labels = create_pair_tensors(val_pos, val_neg, device)
    test_pairs, test_labels = create_pair_tensors(test_pos, test_neg, device)

    print(
        "Pair counts (pos+neg) | "
        f"Train: {train_pairs.shape[1]:,}  "
        f"Val: {val_pairs.shape[1]:,}  "
        f"Test: {test_pairs.shape[1]:,}"
    )

    # -------------------------------
    # Model and training loop
    # -------------------------------
    model = PrimeKGDrugRepurposingGNN(
        num_nodes=num_nodes,
        num_types=len(node_artifacts.type_to_idx),
        hidden_dim=config.hidden_dim,
        embedding_dim=config.embedding_dim,
        dropout=config.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=config.lr_scheduler_factor,
        patience=config.lr_scheduler_patience,
        min_lr=config.min_lr,
    )

    node_type_ids = node_artifacts.node_type_ids.to(device)
    drug_nodes_device = node_artifacts.drug_nodes.to(device)
    disease_nodes_device = node_artifacts.disease_nodes.to(device)

    history: Dict[str, List[float]] = defaultdict(list)

    best_val_mrr = -math.inf
    best_epoch = -1
    best_val_snapshot: Dict[str, float] = {}
    patience_counter = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        optimizer.zero_grad()

        # Use static adjacency — model dropout provides regularization (saves ~400MB peak memory)
        z_train = model.encode(node_type_ids, static_adj)
        train_logits = model.score(z_train, train_pairs, degree_tensor)

        bce_loss = F.binary_cross_entropy_with_logits(train_logits, train_labels)
        degree_corr_penalty = degree_correlation_regularizer(train_logits, train_pairs, degree_tensor)
        loss = bce_loss + config.degree_corr_lambda * degree_corr_penalty

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip_norm)
        optimizer.step()

        if epoch % config.eval_every != 0:
            continue

        model.eval()
        with torch.no_grad():
            z_eval = model.encode(node_type_ids, static_adj)

            val_logits = predict_logits(
                model=model,
                z=z_eval,
                pairs=val_pairs,
                degrees=degree_tensor,
                batch_size=config.batch_size,
            )
            val_loss = F.binary_cross_entropy_with_logits(val_logits, val_labels).item()

            val_probs = torch.sigmoid(val_logits).detach().cpu().numpy()
            val_labels_np = val_labels.detach().cpu().numpy()

            val_binary = compute_binary_metrics(val_labels_np, val_probs)
            val_ranking, _ = evaluate_ranking_metrics(
                model=model,
                z=z_eval,
                positive_edges=val_pos,
                drug_nodes=drug_nodes_device,
                degrees=degree_tensor,
                top_k=config.ranking_k,
                batch_size=config.batch_size,
            )

        val_mrr = float(val_ranking["mrr"])
        scheduler.step(val_mrr)

        current_lr = optimizer.param_groups[0]["lr"]
        history["epoch"].append(float(epoch))
        history["lr"].append(float(current_lr))
        history["train_loss"].append(float(loss.item()))
        history["train_bce_loss"].append(float(bce_loss.item()))
        history["train_degree_corr_penalty"].append(float(degree_corr_penalty.item()))
        history["val_loss"].append(float(val_loss))
        history["val_auc"].append(float(val_binary["auc"]))
        history["val_ap"].append(float(val_binary["ap"]))
        history["val_mrr"].append(float(val_ranking["mrr"]))
        history["val_hits@10"].append(float(val_ranking["hits@10"]))

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={loss.item():.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_auc={val_binary['auc']:.4f} "
            f"val_ap={val_binary['ap']:.4f} "
            f"val_mrr={val_ranking['mrr']:.4f} "
            f"lr={current_lr:.2e}"
        )

        if val_mrr > best_val_mrr:
            best_val_mrr = val_mrr
            best_epoch = epoch
            best_val_snapshot = {
                "val_loss": float(val_loss),
                **{f"val_{k}": float(v) for k, v in val_binary.items()},
                **{f"val_{k}": float(v) for k, v in val_ranking.items()},
            }
            patience_counter = 0

            checkpoint = {
                "model_state": model.state_dict(),
                "model_config": {
                    "num_nodes": num_nodes,
                    "num_types": len(node_artifacts.type_to_idx),
                    "hidden_dim": config.hidden_dim,
                    "embedding_dim": config.embedding_dim,
                    "dropout": config.dropout,
                },
            }
            torch.save(checkpoint, config.models_dir / "gnn_drug_repurposing.pt")
            torch.save(static_adj.cpu(), config.models_dir / "adjacency.pt")
            torch.save(degree_tensor.cpu(), config.models_dir / "degrees.pt")
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(
                    f"Early stopping at epoch {epoch}. "
                    f"Best val MRR={best_val_mrr:.4f} at epoch {best_epoch}."
                )
                break

    # -------------------------------
    # Best checkpoint evaluation
    # -------------------------------
    checkpoint = torch.load(config.models_dir / "gnn_drug_repurposing.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    with torch.no_grad():
        z_best = model.encode(node_type_ids, static_adj)

        test_logits = predict_logits(
            model=model,
            z=z_best,
            pairs=test_pairs,
            degrees=degree_tensor,
            batch_size=config.batch_size,
        )
        test_loss = F.binary_cross_entropy_with_logits(test_logits, test_labels).item()

        test_probs_np = torch.sigmoid(test_logits).detach().cpu().numpy()
        test_labels_np = test_labels.detach().cpu().numpy()

    test_binary = compute_binary_metrics(test_labels_np, test_probs_np)
    test_ranking, _ = evaluate_ranking_metrics(
        model=model,
        z=z_best,
        positive_edges=test_pos,
        drug_nodes=drug_nodes_device,
        degrees=degree_tensor,
        top_k=config.ranking_k,
        batch_size=config.batch_size,
    )
    degree_stratified = evaluate_degree_stratified_metrics(
        pairs=test_pairs,
        labels_np=test_labels_np,
        probs_np=test_probs_np,
        degrees=degree_tensor,
        drug_nodes=drug_nodes_device,
    )

    spearman_metrics, drug_degrees, mean_scores = compute_degree_score_bias(
        model=model,
        z=z_best,
        drug_nodes=drug_nodes_device,
        disease_nodes=disease_nodes_device,
        degrees=degree_tensor,
        batch_size=config.batch_size,
        max_diseases=config.spearman_diseases,
        rng=rng,
    )

    diversity_metrics, jaccard_values, disease_topk = compute_topk_diversity(
        model=model,
        z=z_best,
        drug_nodes=drug_nodes_device,
        disease_nodes=disease_nodes_device,
        degrees=degree_tensor,
        top_k=config.ranking_k,
        batch_size=config.batch_size,
        max_diseases=config.diversity_max_diseases,
        max_pairs=config.diversity_max_pairs,
        rng=rng,
    )

    top1_counter = Counter(topk[0] for topk in disease_topk.values() if topk)
    top1_summary = {}
    if top1_counter:
        most_common_node, count = top1_counter.most_common(1)[0]
        top1_summary = {
            "top1_mode_node": int(most_common_node),
            "top1_mode_drug_name": node_to_entity_name(
                int(most_common_node),
                node_artifacts.all_keys,
                drug_id_to_name,
            ),
            "top1_mode_fraction": float(count / max(len(disease_topk), 1)),
        }

    # -------------------------------
    # Persist metadata and metrics
    # -------------------------------
    metadata = {
        "all_keys": node_artifacts.all_keys,
        "node_map": node_artifacts.node_map,
        "node_types": node_artifacts.node_types,
        "type_to_idx": node_artifacts.type_to_idx,
        "drug_nodes": node_artifacts.drug_nodes.detach().cpu().tolist(),
        "disease_nodes": node_artifacts.disease_nodes.detach().cpu().tolist(),
        "disease_id_to_name": disease_id_to_name,
        "drug_id_to_name": drug_id_to_name,
    }
    with (config.models_dir / "metadata.pkl").open("wb") as file_obj:
        pickle.dump(metadata, file_obj)

    metrics_payload = {
        "config": asdict(config),
        "splits": {
            "train_pos": len(train_pos),
            "val_pos": len(val_pos),
            "test_pos": len(test_pos),
            "train_neg": len(train_neg),
            "val_neg": len(val_neg),
            "test_neg": len(test_neg),
        },
        "best": {
            "best_epoch": best_epoch,
            "best_val_mrr": best_val_mrr,
            **best_val_snapshot,
        },
        "test": {
            "test_loss": float(test_loss),
            **{f"test_{k}": float(v) for k, v in test_binary.items()},
            **{f"test_{k}": float(v) for k, v in test_ranking.items()},
        },
        "degree_stratified": degree_stratified,
        "bias": {
            **spearman_metrics,
            **diversity_metrics,
            **top1_summary,
        },
        "history": history,
    }

    metrics_path = config.models_dir / "training_metrics.json"
    with metrics_path.open("w") as file_obj:
        json.dump(json_ready(metrics_payload), file_obj, indent=2)

    # -------------------------------
    # Diagnostic plots
    # -------------------------------
    if plt is None:
        print("matplotlib is not installed; skipping all plot generation.")

    drug_degrees_np = degree_tensor[drug_nodes_device].detach().cpu().numpy()

    plot_training_curves(history, config.plots_dir / "training_curves.png")
    plot_degree_distribution(drug_degrees_np, config.plots_dir / "degree_distribution.png")
    plot_degree_vs_score(
        drug_degrees=drug_degrees,
        mean_scores=mean_scores,
        rho=float(spearman_metrics["spearman_rho"]),
        output_path=config.plots_dir / "degree_vs_score.png",
    )
    plot_roc_pr_curves(test_labels_np, test_probs_np, config.plots_dir / "roc_pr_curves.png")
    plot_degree_stratified_metrics(
        stratified_metrics=degree_stratified,
        output_path=config.plots_dir / "degree_stratified_metrics.png",
    )
    if not config.skip_tsne:
        try:
            plot_embedding_tsne(
                z=z_best,
                drug_nodes=drug_nodes_device,
                degrees=degree_tensor,
                output_path=config.plots_dir / "embedding_tsne.png",
                max_points=config.tsne_max_points,
                seed=config.seed,
            )
        except Exception as exc:
            print(f"Skipping t-SNE plot due to error: {exc}")
    plot_topk_diversity(jaccard_values, config.plots_dir / "topk_diversity.png")

    print("\nTraining complete. Saved artifacts:")
    print(f"  - {config.models_dir / 'gnn_drug_repurposing.pt'}")
    print(f"  - {config.models_dir / 'adjacency.pt'}")
    print(f"  - {config.models_dir / 'degrees.pt'}")
    print(f"  - {config.models_dir / 'metadata.pkl'}")
    print(f"  - {metrics_path}")
    print(f"  - {config.plots_dir}")


if __name__ == "__main__":
    main()
