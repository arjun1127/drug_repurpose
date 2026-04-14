"""
evaluate_model_bias.py
======================
Formal diagnostic script to detect Hub / Popularity Bias in the
PrimeKG-based Drug Repurposing GNN.

Tests performed
---------------
1. Drug Degree Census          – degree distribution of drug nodes in PrimeKG
2. Global Popularity Ranking   – average sigmoid score across N random diseases
3. Degree ↔ Score Correlation  – Spearman ρ between degree and avg score
4. Per-Disease Diversity (Jaccard) – do top-k lists differ across diseases?
5. Embedding Similarity        – cosine similarity between drug embeddings and
                                  the "ideal hub" direction
6. Hub-Filtered Re-Ranking     – predictions after removing top-p% hubs

Usage
-----
    python evaluate_model_bias.py

Everything runs from the saved model artifacts in ./models and
the raw PrimeKG CSV in ./data.
"""

import sys
import pickle
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# ──────────────────────────────────────────────────────────────────────
# 0.  GNN architecture (must match training code exactly)
# ──────────────────────────────────────────────────────────────────────


class GraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = torch.sparse.mm(adj, x)
        return self.linear(x)


class PrimeKGDrugRepurposingGNN(nn.Module):
    def __init__(self, num_nodes, num_types, hidden_dim, embedding_dim, dropout):
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
            nn.Linear(embedding_dim, 1),
        )

    def encode(self, node_type_ids, adj):
        idx = torch.arange(len(node_type_ids), device=node_type_ids.device)
        x = self.node_embedding(idx) + self.type_embedding(node_type_ids)
        h = F.relu(self.gcn1(x, adj))
        h = self.dropout(h)
        return self.gcn2(h, adj)

    def score(self, z, pairs):
        src = z[pairs[0]]
        dst = z[pairs[1]]
        feat = torch.cat([src, dst, src * dst], dim=-1)
        return self.link_predictor(feat).squeeze()


# ──────────────────────────────────────────────────────────────────────
# 1.  Configuration
# ──────────────────────────────────────────────────────────────────────

SEED = 42
N_SAMPLE_DISEASES = 100        # diseases to sample for global scoring
TOP_K = 10                     # top-k list size for Jaccard / re-ranking
HUB_PERCENTILE = 5             # top p% by degree treated as "hubs"
SPEARMAN_CRITICAL_RHO = 0.40   # above this → bias is concerning

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ──────────────────────────────────────────────────────────────────────
# 2.  Load everything
# ──────────────────────────────────────────────────────────────────────

def load_assets():
    """Load model, metadata, adjacency, and raw CSV."""
    print("=" * 72)
    print("  LOADING MODEL ASSETS")
    print("=" * 72)

    meta_path = Path("models/metadata.pkl")
    model_path = Path("models/gnn_drug_repurposing.pt")
    adj_path = Path("models/adjacency.pt")
    csv_path = Path("data/primekg.csv")

    for p in (meta_path, model_path, adj_path, csv_path):
        if not p.exists():
            sys.exit(f"✗ Missing required file: {p}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Metadata
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    # Model
    ckpt = torch.load(model_path, map_location=device)
    cfg = ckpt["model_config"]
    model = PrimeKGDrugRepurposingGNN(
        cfg["num_nodes"], cfg["num_types"],
        cfg["hidden_dim"], cfg["embedding_dim"], cfg["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Adjacency
    adj = torch.load(adj_path, map_location=device)

    # Pre-compute embeddings once
    node_types = metadata["node_types"]
    type_to_idx = metadata["type_to_idx"]
    node_type_ids = torch.tensor(
        [type_to_idx[t] for t in node_types], device=device
    )
    with torch.no_grad():
        z = model.encode(node_type_ids, adj)

    # Raw CSV for degree counting
    df = pd.read_csv(csv_path, sep=None, engine="python")

    print(f"  Nodes: {len(metadata['all_keys']):,}")
    print(f"  Drugs: {len(metadata['drug_nodes']):,}")
    print(f"  Diseases: {len(metadata['disease_nodes']):,}")
    print(f"  Embeddings shape: {z.shape}")
    print()

    return metadata, model, adj, z, df, device


# ──────────────────────────────────────────────────────────────────────
# 3.  TEST 1 – Drug Degree Census
# ──────────────────────────────────────────────────────────────────────

def test_degree_census(metadata, df):
    """Count each drug's total connections in the raw PrimeKG CSV."""
    print("=" * 72)
    print("  TEST 1 ▸ Drug Degree Census (raw PrimeKG connections)")
    print("=" * 72)

    all_keys = metadata["all_keys"]
    drug_nodes = metadata["drug_nodes"]
    drug_id_to_name = metadata["drug_id_to_name"]

    # Count mentions as x or y
    mentions = (
        df.loc[df["x_type"] == "drug", "x_id"].astype(str).tolist()
        + df.loc[df["y_type"] == "drug", "y_id"].astype(str).tolist()
    )
    cnt = Counter(mentions)

    drug_degree = {}
    for d_idx in drug_nodes:
        d_key = all_keys[d_idx]
        d_id = d_key.split("::")[1]
        name = drug_id_to_name.get(d_id, d_id)
        drug_degree[d_idx] = {"name": name, "degree": cnt.get(d_id, 0)}

    # Sort by degree desc
    sorted_by_deg = sorted(
        drug_degree.values(), key=lambda x: x["degree"], reverse=True
    )

    degrees = [d["degree"] for d in sorted_by_deg]
    print(f"\n  Total drugs:   {len(degrees)}")
    print(f"  Mean degree:   {np.mean(degrees):.1f}")
    print(f"  Median degree: {np.median(degrees):.1f}")
    print(f"  Max degree:    {max(degrees)}")
    print(f"  Std deviation: {np.std(degrees):.1f}")

    print(f"\n  {'Rank':<6}{'Drug':<35}{'Connections':>12}")
    print("  " + "-" * 53)
    for i, d in enumerate(sorted_by_deg[:15], 1):
        print(f"  {i:<6}{d['name']:<35}{d['degree']:>12,}")

    return drug_degree


# ──────────────────────────────────────────────────────────────────────
# 4.  TEST 2 – Global Popularity Ranking
# ──────────────────────────────────────────────────────────────────────

def test_global_popularity(metadata, model, z, device):
    """Average sigmoid score for each drug across N random diseases."""
    print("\n" + "=" * 72)
    print(f"  TEST 2 ▸ Global Popularity Ranking (avg over {N_SAMPLE_DISEASES} diseases)")
    print("=" * 72)

    drug_nodes = metadata["drug_nodes"]
    disease_nodes = metadata["disease_nodes"]
    drug_id_to_name = metadata["drug_id_to_name"]
    all_keys = metadata["all_keys"]

    sampled = random.sample(disease_nodes, min(N_SAMPLE_DISEASES, len(disease_nodes)))

    all_scores = []
    with torch.no_grad():
        for dis_idx in sampled:
            pairs = torch.tensor(
                [[d, dis_idx] for d in drug_nodes], dtype=torch.long
            ).T.to(device)
            scores = torch.sigmoid(model.score(z, pairs)).cpu().numpy()
            all_scores.append(scores)

    avg_scores = np.mean(all_scores, axis=0)            # shape: (n_drugs,)
    score_std = np.std(all_scores, axis=0)               # per-drug std across diseases

    # Build result list
    drug_global = []
    for i, d_idx in enumerate(drug_nodes):
        d_id = all_keys[d_idx].split("::")[1]
        name = drug_id_to_name.get(d_id, d_id)
        drug_global.append({
            "idx": d_idx,
            "name": name,
            "avg_score": float(avg_scores[i]),
            "std_score": float(score_std[i]),
        })

    drug_global.sort(key=lambda x: x["avg_score"], reverse=True)

    print(f"\n  {'Rank':<6}{'Drug':<35}{'Avg Score':>10}{'Std':>10}")
    print("  " + "-" * 61)
    for i, d in enumerate(drug_global[:20], 1):
        print(f"  {i:<6}{d['name']:<35}{d['avg_score']:>10.4f}{d['std_score']:>10.4f}")

    # Score spread diagnosis
    top1 = drug_global[0]["avg_score"]
    top20 = drug_global[19]["avg_score"] if len(drug_global) >= 20 else drug_global[-1]["avg_score"]
    median_score = np.median([d["avg_score"] for d in drug_global])
    print(f"\n  Score spread:  Top-1={top1:.4f}  Top-20={top20:.4f}  Median={median_score:.4f}")

    mean_std = np.mean([d["std_score"] for d in drug_global[:20]])
    print(f"  Mean std of top-20 drugs across diseases: {mean_std:.4f}")
    if mean_std < 0.02:
        print("  ⚠  Very low variance → model gives nearly IDENTICAL scores")
        print("     regardless of disease.  Strong hub-bias signal.")
    else:
        print("  ✓  Top drugs show reasonable variance across diseases.")

    return drug_global


# ──────────────────────────────────────────────────────────────────────
# 5.  TEST 3 – Degree ↔ Score Spearman Correlation
# ──────────────────────────────────────────────────────────────────────

def test_spearman(drug_degree, drug_global):
    """Spearman ρ between PrimeKG degree and avg prediction score."""
    from scipy.stats import spearmanr

    print("\n" + "=" * 72)
    print("  TEST 3 ▸ Degree ↔ Avg-Score Spearman Rank Correlation")
    print("=" * 72)

    # Align on drug index
    idx_to_deg = {d_idx: info["degree"] for d_idx, info in drug_degree.items()}
    idx_to_score = {d["idx"]: d["avg_score"] for d in drug_global}

    common = sorted(set(idx_to_deg) & set(idx_to_score))
    degs = np.array([idx_to_deg[i] for i in common])
    scrs = np.array([idx_to_score[i] for i in common])

    rho, pval = spearmanr(degs, scrs)
    print(f"\n  Spearman ρ = {rho:.4f}   (p = {pval:.2e})")
    print(f"  N = {len(common):,} drugs")

    if abs(rho) > SPEARMAN_CRITICAL_RHO:
        verdict = "FAIL"
        msg = (
            f"  ⚠  |ρ| = {abs(rho):.4f} > {SPEARMAN_CRITICAL_RHO} threshold.\n"
            f"     The model's predictions are STRONGLY correlated with node\n"
            f"     degree. This confirms Hub / Popularity Bias."
        )
    else:
        verdict = "PASS"
        msg = (
            f"  ✓  |ρ| = {abs(rho):.4f} ≤ {SPEARMAN_CRITICAL_RHO} threshold.\n"
            f"     No significant degree-based popularity bias detected."
        )
    print(msg)

    return rho, pval, verdict


# ──────────────────────────────────────────────────────────────────────
# 6.  TEST 4 – Per-Disease Jaccard Diversity of Top-K
# ──────────────────────────────────────────────────────────────────────

def test_jaccard_diversity(metadata, model, z, device, n_pairs=50):
    """
    For n_pairs of randomly sampled diseases, compute the Jaccard similarity
    of their top-K predicted drug lists.  High Jaccard → all diseases get
    the same drugs → hub bias.
    """
    print("\n" + "=" * 72)
    print(f"  TEST 4 ▸ Per-Disease Top-{TOP_K} Jaccard Diversity ({n_pairs} disease pairs)")
    print("=" * 72)

    drug_nodes = metadata["drug_nodes"]
    disease_nodes = metadata["disease_nodes"]
    all_keys = metadata["all_keys"]
    drug_id_to_name = metadata["drug_id_to_name"]

    sampled = random.sample(disease_nodes, min(n_pairs * 2, len(disease_nodes)))

    def get_topk(dis_idx):
        pairs = torch.tensor(
            [[d, dis_idx] for d in drug_nodes], dtype=torch.long
        ).T.to(device)
        with torch.no_grad():
            sc = torch.sigmoid(model.score(z, pairs)).cpu().numpy()
        top_indices = sc.argsort()[-TOP_K:]
        return set(top_indices.tolist())

    # Cache topk lists
    topk_cache = {}
    for d in sampled:
        topk_cache[d] = get_topk(d)

    # Pairwise Jaccard
    jaccards = []
    disease_list = list(topk_cache.keys())
    for i in range(len(disease_list)):
        for j in range(i + 1, len(disease_list)):
            s1 = topk_cache[disease_list[i]]
            s2 = topk_cache[disease_list[j]]
            jac = len(s1 & s2) / len(s1 | s2) if (s1 | s2) else 0.0
            jaccards.append(jac)

    avg_jaccard = np.mean(jaccards) if jaccards else 0.0
    median_jaccard = np.median(jaccards) if jaccards else 0.0
    perfect_overlap = sum(1 for j in jaccards if j == 1.0)

    print(f"\n  Pairs compared:        {len(jaccards):,}")
    print(f"  Mean Jaccard:          {avg_jaccard:.4f}")
    print(f"  Median Jaccard:        {median_jaccard:.4f}")
    print(f"  Perfect overlaps (1.0): {perfect_overlap} / {len(jaccards)}")
    print(f"  Fraction ≥ 0.80:       {sum(1 for j in jaccards if j >= 0.8) / max(len(jaccards),1):.1%}")

    if avg_jaccard > 0.70:
        verdict = "FAIL"
        print(f"\n  ⚠  SEVERE hub bias: top-{TOP_K} lists are nearly identical")
        print(f"     across diseases (avg Jaccard = {avg_jaccard:.2f}).")
    elif avg_jaccard > 0.40:
        verdict = "WARN"
        print(f"\n  ⚠  Moderate overlap detected. Some popularity bias present.")
    else:
        verdict = "PASS"
        print(f"\n  ✓  Top-{TOP_K} lists are reasonably diverse across diseases.")

    # Show most common drugs across all top-k lists
    all_topk = []
    for s in topk_cache.values():
        all_topk.extend(s)
    most_common = Counter(all_topk).most_common(10)

    print(f"\n  Most frequently appearing drugs in top-{TOP_K} across all sampled diseases:")
    print(f"  {'Drug':<35}{'Appearances':>12}{'Fraction':>10}")
    print("  " + "-" * 57)
    for idx_pos, count in most_common:
        d_idx = drug_nodes[idx_pos]
        d_id = all_keys[d_idx].split("::")[1]
        name = drug_id_to_name.get(d_id, d_id)
        frac = count / len(topk_cache)
        print(f"  {name:<35}{count:>12}{frac:>10.0%}")

    return avg_jaccard, verdict


# ──────────────────────────────────────────────────────────────────────
# 7.  TEST 5 – Embedding Cosine Similarity Analysis
# ──────────────────────────────────────────────────────────────────────

def test_embedding_similarity(metadata, z, drug_degree, device):
    """
    Check if high-degree drugs cluster together in embedding space.
    We compute the mean embedding of the top-5% hubs, then measure the
    cosine similarity of every drug to that "hub centroid".  If high-degree
    drugs are systematically closer, the GNN is encoding popularity.
    """
    print("\n" + "=" * 72)
    print("  TEST 5 ▸ Embedding Cosine Similarity to Hub Centroid")
    print("=" * 72)

    drug_nodes = metadata["drug_nodes"]
    all_keys = metadata["all_keys"]
    drug_id_to_name = metadata["drug_id_to_name"]

    # Sort by degree
    sorted_drugs = sorted(drug_degree.items(), key=lambda x: x[1]["degree"], reverse=True)
    cutoff = max(1, int(len(sorted_drugs) * HUB_PERCENTILE / 100))

    hub_indices = [d_idx for d_idx, _ in sorted_drugs[:cutoff]]
    non_hub_indices = [d_idx for d_idx, _ in sorted_drugs[cutoff:]]

    # Hub centroid
    hub_embs = z[hub_indices]
    centroid = hub_embs.mean(dim=0, keepdim=True)

    # Cosine sim of hubs vs non-hubs to centroid
    all_drug_embs = z[drug_nodes]
    cos = F.cosine_similarity(all_drug_embs, centroid.expand_as(all_drug_embs))
    cos_np = cos.cpu().numpy()

    hub_mask = np.isin(drug_nodes, hub_indices)
    hub_cos = cos_np[hub_mask]
    non_hub_cos = cos_np[~hub_mask]

    print(f"\n  Hub drugs (top {HUB_PERCENTILE}%):  {len(hub_indices)}")
    print(f"  Non-hub drugs:         {len(non_hub_indices)}")
    print(f"\n  Cosine sim to hub centroid:")
    print(f"    Hub mean:     {hub_cos.mean():.4f}  (std {hub_cos.std():.4f})")
    print(f"    Non-hub mean: {non_hub_cos.mean():.4f}  (std {non_hub_cos.std():.4f})")
    print(f"    Gap:          {hub_cos.mean() - non_hub_cos.mean():.4f}")

    gap = hub_cos.mean() - non_hub_cos.mean()
    if gap > 0.15:
        print(f"\n  ⚠  Hub embeddings are significantly closer to each other")
        print(f"     than non-hub embeddings. Model is encoding popularity")
        print(f"     into the embedding space itself.")
    else:
        print(f"\n  ✓  Embedding space does not strongly separate hubs from non-hubs.")

    return float(gap)


# ──────────────────────────────────────────────────────────────────────
# 8.  TEST 6 – Hub-Filtered Re-Ranking
# ──────────────────────────────────────────────────────────────────────

def test_hub_filtered_reranking(metadata, model, z, drug_degree, device):
    """
    Remove top-p% hub drugs and show the new top-K for a handful of
    very different diseases.  If the filtered predictions make biological
    sense, the model has *some* disease-specific knowledge underneath
    the hub bias.
    """
    print("\n" + "=" * 72)
    print(f"  TEST 6 ▸ Hub-Filtered Re-Ranking (removing top {HUB_PERCENTILE}% hubs)")
    print("=" * 72)

    drug_nodes = metadata["drug_nodes"]
    disease_nodes = metadata["disease_nodes"]
    all_keys = metadata["all_keys"]
    drug_id_to_name = metadata["drug_id_to_name"]
    disease_id_to_name = metadata["disease_id_to_name"]

    # Identify hub drugs
    sorted_drugs = sorted(drug_degree.items(), key=lambda x: x[1]["degree"], reverse=True)
    cutoff = max(1, int(len(sorted_drugs) * HUB_PERCENTILE / 100))
    hub_set = {d_idx for d_idx, _ in sorted_drugs[:cutoff]}

    print(f"  Removed {len(hub_set)} hub drugs.\n")

    # Pick 5 specific diverse diseases (or fallback to random)
    target_queries = ["diabetes", "asthma", "leukemia", "alzheimer", "malaria"]
    selected_diseases = []

    for q in target_queries:
        for d_idx in disease_nodes:
            d_id = all_keys[d_idx].split("::")[1]
            name = disease_id_to_name.get(d_id, "").lower()
            if q in name:
                selected_diseases.append((d_idx, disease_id_to_name.get(d_id, d_id)))
                break

    # Fill with random diseases if we didn't find enough
    if len(selected_diseases) < 3:
        for d_idx in random.sample(disease_nodes, 5):
            d_id = all_keys[d_idx].split("::")[1]
            selected_diseases.append((d_idx, disease_id_to_name.get(d_id, d_id)))
        selected_diseases = selected_diseases[:5]

    for dis_idx, dis_name in selected_diseases:
        pairs = torch.tensor(
            [[d, dis_idx] for d in drug_nodes], dtype=torch.long
        ).T.to(device)
        with torch.no_grad():
            scores = torch.sigmoid(model.score(z, pairs)).cpu().numpy()

        # Build ranked list, separate hub vs non-hub
        ranked_all = sorted(
            zip(drug_nodes, scores), key=lambda x: x[1], reverse=True
        )

        print(f"  ─── {dis_name} ───")
        print(f"  {'':>4}{'UNFILTERED (original)':^40}│{'HUB-FILTERED':^40}")
        print(f"  {'Rk':>4}{'Drug':<30}{'Score':>8}   │ {'Drug':<30}{'Score':>8}")
        print("  " + "-" * 83)

        unfiltered = ranked_all[:TOP_K]
        filtered = [(d, s) for d, s in ranked_all if d not in hub_set][:TOP_K]

        for i in range(TOP_K):
            u_idx, u_sc = unfiltered[i]
            u_id = all_keys[u_idx].split("::")[1]
            u_name = drug_id_to_name.get(u_id, u_id)[:28]

            if i < len(filtered):
                f_idx, f_sc = filtered[i]
                f_id = all_keys[f_idx].split("::")[1]
                f_name = drug_id_to_name.get(f_id, f_id)[:28]
            else:
                f_name, f_sc = "—", 0.0

            print(f"  {i+1:>4} {u_name:<30}{u_sc:>8.4f}   │ {f_name:<30}{f_sc:>8.4f}")

        print()


# ──────────────────────────────────────────────────────────────────────
# 9.  Main – Run all tests & print verdict
# ──────────────────────────────────────────────────────────────────────

def main():
    metadata, model, adj, z, df, device = load_assets()

    # Test 1
    drug_degree = test_degree_census(metadata, df)

    # Test 2
    drug_global = test_global_popularity(metadata, model, z, device)

    # Test 3
    rho, pval, spearman_verdict = test_spearman(drug_degree, drug_global)

    # Test 4
    avg_jaccard, jaccard_verdict = test_jaccard_diversity(metadata, model, z, device)

    # Test 5
    emb_gap = test_embedding_similarity(metadata, z, drug_degree, device)

    # Test 6
    test_hub_filtered_reranking(metadata, model, z, drug_degree, device)

    # ── Final Report ──
    print("\n" + "=" * 72)
    print("  FINAL DIAGNOSTIC REPORT")
    print("=" * 72)
    print(f"""
  ┌──────────────────────────────────────────────────────────────────┐
  │  Test                          │  Result                        │
  ├──────────────────────────────────────────────────────────────────┤
  │  Spearman (degree↔score)       │  ρ = {rho:+.4f}  →  {spearman_verdict:<14}│
  │  Jaccard Diversity (top-{TOP_K:<2})    │  J = {avg_jaccard:.4f}  →  {jaccard_verdict:<14}│
  │  Embedding Hub Gap             │  Δ = {emb_gap:.4f}                    │
  └──────────────────────────────────────────────────────────────────┘
""")

    bias_detected = (spearman_verdict == "FAIL" or jaccard_verdict == "FAIL")

    if bias_detected:
        print("  ⚠  CONCLUSION: Hub / Popularity Bias is CONFIRMED.")
        print()
        print("  The model relies heavily on node degree (connectivity) to rank")
        print("  drugs, rather than learning disease-specific biological signals.")
        print("  Propranolol and other highly-connected drugs dominate predictions")
        print("  for almost every disease because they are statistical 'safe bets'.")
        print()
        print("  RECOMMENDED MITIGATIONS:")
        print("  ────────────────────────")
        print("  1. Degree-Penalised Scoring  – subtract a λ·log(degree) term")
        print("     from the raw logit at inference time.")
        print("  2. Inverse-Degree Sampling   – during training, sample negative")
        print("     edges with probability ∝ 1/degree to reduce hub advantage.")
        print("  3. Edge Dropout/DropEdge     – randomly mask a fraction of edges")
        print("     incident to high-degree nodes during training.")
        print("  4. Post-hoc Hub Filtering    – remove top-p% hubs from the")
        print("     candidate pool at inference (see Test 6 above).")
        print("  5. Fairness-Aware GNN Loss   – add a regularisation term that")
        print("     penalises correlation between degree and score.")
    else:
        print("  ✓  CONCLUSION: No severe hub bias detected.")
        print("     The model appears to learn disease-specific drug signals.")

    print("\n" + "=" * 72)
    print("  Evaluation complete.")
    print("=" * 72)


if __name__ == "__main__":
    main()
