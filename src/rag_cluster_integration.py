"""
Integration bridge: BiH Heritage DBSCAN spatial clusters -> MLST Cultural Chatbot RAG store.

What this does
---------------
1. Loads the chatbot's cleaned heritage dataset (MLSTProject/BiH_Heritage_Final_Clean.csv).
2. Runs the same DBSCAN spatial clustering used for the app's corridor map
   (micro = tight ~25km corridors, macro = ~45km regional corridors).
3. Writes cluster_micro / cluster_macro labels back into:
   - a new CSV (MLSTProject/BiH_Heritage_Clustered.csv), for inspection / notebook 2 reuse.
   - the existing Chroma collection "bih_heritage" metadata (matched by site name),
     so the chatbot's retriever can filter/boost results by geographic corridor.
4. Exposes find_nearby_sites(), a helper the chatbot can call as a RAG tool to answer
   "what else is near X?" using cluster membership instead of new embeddings.

Run:
    .venv\\Scripts\\python.exe src\\rag_cluster_integration.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import chromadb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.clustering_engine import run_spatial_dbscan  # noqa: E402
from src.data_processing import get_radial_coordinates  # noqa: E402

MLST_DIR = PROJECT_ROOT / "MLSTProject"
CLEAN_CSV = MLST_DIR / "BiH_Heritage_Final_Clean.csv"
CLUSTERED_CSV = MLST_DIR / "BiH_Heritage_Clustered.csv"
CHROMA_DB_PATH = MLST_DIR / "heritage_db"
COLLECTION_NAME = "bih_heritage"

EPS_MICRO_KM = 15.0
MIN_PTS_MICRO = 4
EPS_MACRO_KM = 45.0
MIN_PTS_MACRO = 3


def normalize_name(name: str) -> str:
    return str(name).strip().lower()


def compute_clusters(df: pd.DataFrame) -> pd.DataFrame:
    coords_rad = get_radial_coordinates(df)

    db_micro = run_spatial_dbscan(coords_rad, eps_km=EPS_MICRO_KM, min_samples=MIN_PTS_MICRO)
    df["cluster_micro"] = db_micro.labels_

    db_macro = run_spatial_dbscan(coords_rad, eps_km=EPS_MACRO_KM, min_samples=MIN_PTS_MACRO)
    df["cluster_macro"] = db_macro.labels_

    return df


def update_chroma_metadata(df: pd.DataFrame) -> dict:
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    collection = client.get_collection(COLLECTION_NAME)

    all_docs = collection.get(include=["metadatas"])
    ids = all_docs["ids"]
    metadatas = all_docs["metadatas"]

    # Build a lookup from normalized site name -> cluster labels
    lookup = {
        normalize_name(row["name"]): (int(row["cluster_micro"]), int(row["cluster_macro"]))
        for _, row in df.iterrows()
    }

    update_ids: list[str] = []
    update_metadatas: list[dict] = []
    unmatched: list[str] = []

    for doc_id, meta in zip(ids, metadatas):
        meta = dict(meta or {})
        site_name = meta.get("name") or doc_id
        key = normalize_name(site_name)

        if key in lookup:
            cluster_micro, cluster_macro = lookup[key]
            meta["cluster_micro"] = cluster_micro
            meta["cluster_macro"] = cluster_macro
            update_ids.append(doc_id)
            update_metadatas.append(meta)
        else:
            unmatched.append(site_name)

    if update_ids:
        collection.update(ids=update_ids, metadatas=update_metadatas)

    return {
        "total_docs": len(ids),
        "updated": len(update_ids),
        "unmatched": unmatched,
    }


def find_nearby_sites(site_name: str, scale: str = "macro", top_n: int = 5) -> list[str]:
    """
    RAG-tool helper: given a site name, return other sites in the same spatial
    corridor (cluster). Reads straight from the Chroma metadata written above.

    scale: "micro" (~25km, tight corridors) or "macro" (~45km, regional corridors)
    """
    if scale not in ("micro", "macro"):
        raise ValueError("scale must be 'micro' or 'macro'")

    field = f"cluster_{scale}"
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    collection = client.get_collection(COLLECTION_NAME)

    all_docs = collection.get(include=["metadatas"])
    metadatas = all_docs["metadatas"]

    target_key = normalize_name(site_name)
    target_cluster = None
    for meta in metadatas:
        if normalize_name(meta.get("name", "")) == target_key:
            target_cluster = meta.get(field)
            break

    if target_cluster is None or target_cluster == -1:
        return []

    nearby = [
        meta["name"]
        for meta in metadatas
        if meta.get(field) == target_cluster and normalize_name(meta.get("name", "")) != target_key
    ]
    return nearby[:top_n]


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    if not CLEAN_CSV.exists():
        raise FileNotFoundError(f"Chatbot dataset not found: {CLEAN_CSV}")
    if not CHROMA_DB_PATH.exists():
        raise FileNotFoundError(f"Chatbot vector DB not found: {CHROMA_DB_PATH}")

    df = pd.read_csv(CLEAN_CSV)
    df = df.dropna(subset=["name", "latitude", "longitude"]).reset_index(drop=True)

    df = compute_clusters(df)
    df.to_csv(CLUSTERED_CSV, index=False)

    n_micro = len(set(df["cluster_micro"])) - (1 if -1 in df["cluster_micro"].values else 0)
    n_macro = len(set(df["cluster_macro"])) - (1 if -1 in df["cluster_macro"].values else 0)
    print(f"Computed clusters for {len(df)} sites: {n_micro} micro corridors, {n_macro} macro corridors.")
    print(f"Saved: {CLUSTERED_CSV}")

    stats = update_chroma_metadata(df)
    print(f"Chroma collection '{COLLECTION_NAME}': {stats['updated']}/{stats['total_docs']} docs updated.")
    if stats["unmatched"]:
        print(f"Unmatched ({len(stats['unmatched'])}): {stats['unmatched'][:10]}{'...' if len(stats['unmatched']) > 10 else ''}")

    demo_site = df.iloc[0]["name"]
    nearby = find_nearby_sites(demo_site, scale="macro")
    print(f"\nDemo: sites near '{demo_site}' (macro corridor): {nearby}")


if __name__ == "__main__":
    main()
