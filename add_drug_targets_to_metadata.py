#!/usr/bin/env python3
"""Add drug-protein target information to existing metadata.pkl without retraining.

This script reads PrimeKG to extract drug→protein relationships (target, enzyme,
transporter, carrier) and saves them into the metadata pickle so the backend
can generate biological rationale for drug recommendations.
"""

import pickle
import pandas as pd
from collections import defaultdict
from pathlib import Path


def main():
    metadata_path = Path("models/metadata.pkl")
    primekg_path = Path("data/primekg.csv")

    print("Loading metadata...")
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    node_map = metadata["node_map"]
    drug_id_to_name = metadata["drug_id_to_name"]

    print("Loading PrimeKG...")
    df = pd.read_csv(primekg_path, low_memory=True)

    # Filter to drug_protein edges
    mask = df["relation"].astype(str).str.lower() == "drug_protein"
    dp_edges = df[mask]
    print(f"Found {len(dp_edges):,} drug-protein edges")

    drug_targets = {}
    for _, row in dp_edges.iterrows():
        drug_key = f"drug::{row['x_id']}"
        if drug_key not in node_map:
            continue
        drug_idx = node_map[drug_key]

        protein_id = str(row["y_id"])
        protein_name = str(row.get("y_name", ""))
        display_rel = str(row.get("display_relation", "target"))

        if drug_idx not in drug_targets:
            drug_targets[drug_idx] = []
        drug_targets[drug_idx].append({
            "protein_id": protein_id,
            "protein_name": protein_name,
            "relation": display_rel,
        })

    print(f"Extracted targets for {len(drug_targets):,} drugs")
    print(f"Total drug-protein edges mapped: {sum(len(v) for v in drug_targets.values()):,}")

    # Show some examples
    all_keys = metadata["all_keys"]
    for drug_idx in list(drug_targets.keys())[:5]:
        drug_key = all_keys[drug_idx]
        drug_id = drug_key.split("::")[1]
        drug_name = drug_id_to_name.get(drug_id, drug_id)
        targets = drug_targets[drug_idx]
        target_str = ", ".join(
            [f"{t['protein_name']}({t['relation']})" for t in targets[:5]]
        )
        print(f"  {drug_name}: {target_str}")

    # Add to metadata
    metadata["drug_protein_targets"] = drug_targets

    # Save updated metadata
    print(f"\nSaving updated metadata to {metadata_path}...")
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)

    print("Done! Drug-protein targets added to metadata.")


if __name__ == "__main__":
    main()
