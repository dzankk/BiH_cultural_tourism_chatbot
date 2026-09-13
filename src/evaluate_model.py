"""
Evaluation script for KulTur's two ML components:
1. RandomForestClassifier preference re-ranker (heritage_brain_v2.pkl)
2. DBSCAN spatial corridor clustering (micro/macro epsilon selection)

Produces plots + a text summary under evaluation_results/. Run with:
    .venv\\Scripts\\python.exe src\\evaluate_model.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix, silhouette_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "evaluation_results"
OUT_DIR.mkdir(exist_ok=True)

EARTH_RADIUS_KM = 6371.009


def encode(le, value):
    try:
        return int(le.transform([value])[0])
    except ValueError:
        return 0


def evaluate_rf():
    brain = joblib.load(DATA_DIR / "heritage_brain_v2.pkl")
    model: RandomForestClassifier = brain["model"]
    le_interest, le_season, le_region = brain["le_interest"], brain["le_season"], brain["le_region"]
    le_category, le_period = brain["le_category"], brain["le_period"]

    df = pd.read_csv(DATA_DIR / "simulated_interaction_profiles.csv")
    rows = [
        {
            "user_interest": encode(le_interest, r["user_interest"]),
            "current_season": encode(le_season, r["current_season"]),
            "user_region": encode(le_region, r["user_region_pref"]),
            "limited_time": int(r["limited_time"]),
            "site_region": encode(le_region, r["site_region"]),
            "site_category": encode(le_category, r["site_category"]),
            "site_pop": r["site_popularity"],
            "period": encode(le_period, r["historical_period"]),
        }
        for _, r in df.iterrows()
    ]
    X = pd.DataFrame(rows)[list(model.feature_names_in_)]
    y = df["label"]

    # 1. Existing saved model, evaluated on the full dataset (optimistic - not a held-out test).
    full_pred = model.predict(X)
    full_acc = accuracy_score(y, full_pred)

    # 2. Honest holdout: retrain an identical-hyperparameter RF on an 80/20 stratified split,
    #    so we have a real, defensible generalization estimate for the report.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    holdout_model = RandomForestClassifier(**{k: v for k, v in model.get_params().items() if k != "warm_start"})
    holdout_model.fit(X_train, y_train)
    holdout_pred = holdout_model.predict(X_test)
    holdout_acc = accuracy_score(y_test, holdout_pred)
    report = classification_report(y_test, holdout_pred, target_names=["No (0)", "Recommend (1)"])

    cm_full = confusion_matrix(y, full_pred)
    cm_holdout = confusion_matrix(y_test, holdout_pred)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ConfusionMatrixDisplay(cm_full, display_labels=["No", "Recommend"]).plot(ax=axes[0], colorbar=False, cmap="Blues")
    axes[0].set_title(f"Saved model on FULL data\n(optimistic, not held-out) - acc={full_acc:.3f}")
    ConfusionMatrixDisplay(cm_holdout, display_labels=["No", "Recommend"]).plot(ax=axes[1], colorbar=False, cmap="Greens")
    axes[1].set_title(f"Retrained model on 20% HELD-OUT split\n(honest generalization estimate) - acc={holdout_acc:.3f}")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "confusion_matrices.png", dpi=150)
    plt.close(fig)

    importances = sorted(zip(model.feature_names_in_, model.feature_importances_), key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh([f for f, _ in importances], [v for _, v in importances], color="#00838f")
    ax.set_xlabel("Feature importance (Gini)")
    ax.set_title("RandomForest feature importances (production model)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)

    summary = (
        "=== RandomForestClassifier evaluation ===\n"
        f"Training data: data/simulated_interaction_profiles.csv ({len(df)} rows, "
        f"label balance: {dict(y.value_counts())})\n\n"
        f"[1] Saved production model (heritage_brain_v2.pkl) on FULL dataset "
        f"(NOT a held-out test - optimistic upper bound):\n"
        f"    Accuracy: {full_acc:.4f}\n"
        f"    Confusion matrix: {cm_full.tolist()}\n\n"
        f"[2] Reproduced model, identical hyperparameters, on an 80/20 STRATIFIED HOLDOUT "
        f"split (random_state=42) - honest generalization estimate:\n"
        f"    Accuracy: {holdout_acc:.4f}\n"
        f"    Confusion matrix: {cm_holdout.tolist()}\n\n"
        f"    Classification report:\n{report}\n"
    )
    return summary


def evaluate_dbscan():
    df = pd.read_csv(DATA_DIR / "BiH_Heritage_Final_Clean.csv")
    df.columns = df.columns.str.strip()
    X_rad = np.radians(df[["latitude", "longitude"]].values)

    eps_range_km = [10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0]
    rows = []
    for eps_km in eps_range_km:
        eps_rad = eps_km / EARTH_RADIUS_KM
        db = DBSCAN(eps=eps_rad, min_samples=4, metric="haversine")
        labels = db.fit_predict(X_rad)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int((labels == -1).sum())
        if n_clusters > 1:
            score = silhouette_score(X_rad, labels, metric="haversine")
        else:
            score = None
        rows.append((eps_km, n_clusters, n_noise, score))

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    eps_vals = [r[0] for r in rows]
    scores = [r[3] if r[3] is not None else np.nan for r in rows]
    clusters = [r[1] for r in rows]
    ax1.plot(eps_vals, scores, marker="o", color="#00838f", label="Silhouette score")
    ax1.set_xlabel("Epsilon (km)")
    ax1.set_ylabel("Silhouette score", color="#00838f")
    ax2 = ax1.twinx()
    ax2.bar(eps_vals, clusters, alpha=0.25, width=2.0, color="#ff5722", label="Clusters found")
    ax2.set_ylabel("Clusters found", color="#ff5722")
    ax1.set_title("DBSCAN epsilon sensitivity (min_samples=4)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "dbscan_epsilon_sensitivity.png", dpi=150)
    plt.close(fig)

    best = max((r for r in rows if r[3] is not None), key=lambda r: r[3])
    summary = (
        "=== DBSCAN epsilon sensitivity (min_samples=4, data/BiH_Heritage_Final_Clean.csv) ===\n"
        f"{'Eps(km)':<10}{'Clusters':<10}{'Noise pts':<12}{'Silhouette':<12}\n"
        + "\n".join(f"{e:<10}{c:<10}{n:<12}{('%.4f' % s) if s is not None else 'N/A':<12}" for e, c, n, s in rows)
        + f"\n\nBest silhouette score at eps={best[0]} km ({best[1]} clusters, score={best[3]:.4f}).\n"
        f"Production config uses eps=15.0km/min_samples=4 for micro-corridors and "
        f"eps=45.0km/min_samples=3 for macro-corridors (src/rag_cluster_integration.py) - "
        f"chosen for tight, human-meaningful travel corridors, not purely for max silhouette score.\n"
    )
    return summary


if __name__ == "__main__":
    parts = [evaluate_rf(), evaluate_dbscan()]
    text = "\n".join(parts)
    print(text)
    (OUT_DIR / "summary.txt").write_text(text, encoding="utf-8")
    print(f"Saved plots + summary.txt to {OUT_DIR}")
