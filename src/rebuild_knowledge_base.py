"""
Rebuilds the 'bih_heritage' Chroma collection from scratch using the current
MLSTProject/BiH_Heritage_Final_Clean.csv.

Replicates the ingestion recipe from 01_Heritage_Knowledge_Base.ipynb exactly
(same embedding model, same document/metadata/id construction), but computes
DBSCAN cluster_micro/cluster_macro labels *before* ingestion so every doc has
them from the start, instead of a separate post-hoc update pass. Needed
whenever the underlying CSV gains/loses sites (the old collection would
otherwise keep stale/orphaned docs from a previous dataset version).

Run:
    .venv\\Scripts\\python.exe src\\rebuild_knowledge_base.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import chromadb
import pandas as pd
from chromadb.utils import embedding_functions

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.rag_cluster_integration import CHROMA_DB_PATH, CLEAN_CSV, COLLECTION_NAME, compute_clusters  # noqa: E402


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    df = pd.read_csv(CLEAN_CSV)
    df = df.dropna(subset=["name", "latitude", "longitude"]).reset_index(drop=True)
    df = compute_clusters(df)

    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing '{COLLECTION_NAME}' collection.")
    except Exception:
        print(f"No existing '{COLLECTION_NAME}' collection to delete.")

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    collection = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)

    documents = (df["description"].fillna("") + " " + df["keywords"].fillna("")).tolist()
    metadatas = df.to_dict(orient="records")
    ids = df["name"].tolist()

    collection.add(documents=documents, metadatas=metadatas, ids=ids)

    n_micro = len(set(df["cluster_micro"])) - (1 if -1 in df["cluster_micro"].values else 0)
    n_macro = len(set(df["cluster_macro"])) - (1 if -1 in df["cluster_macro"].values else 0)
    print(f"Rebuilt '{COLLECTION_NAME}': {collection.count()} sites indexed.")
    print(f"Clusters: {n_micro} micro corridors, {n_macro} macro corridors.")


if __name__ == "__main__":
    main()
