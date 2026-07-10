"""Helpers for in-training full-movie tracking validation."""

from __future__ import annotations

from typing import Any

import numpy as np


def is_tracking_epoch(current_epoch: int, frequency: int) -> bool:
    """Use human-readable epoch numbers: frequency 10 runs at 10, 20, ..."""
    return frequency > 0 and (current_epoch + 1) % frequency == 0


def summarize_tracking_metrics(metrics: Any) -> dict[str, float]:
    """Summarize per-movie CTC tracking metrics into Lightning scalar names."""
    movies = metrics[metrics["movie"] != "Mean"]
    summary = {
        f"val_{name}": float(movies[name].mean()) for name in ("TRA", "AOGM", "DET")
    }
    summary["val_LNK_ERR"] = float(1.0 - movies["LNK"].mean())
    for name in ("fn_div", "fp_div", "f1_div"):
        if name in movies and movies[name].notna().any():
            summary[f"val_track_{name}"] = float(np.nanmean(movies[name]))
    return summary


def summarize_lam_split_sweep(metrics: Any) -> dict[str, float]:
    """Summarize a tracking lambda sweep while keeping lambda zero as baseline."""
    if "lam_split" not in metrics or metrics["lam_split"].nunique() <= 1:
        return {}

    movies = metrics[metrics["movie"] != "Mean"]
    by_lam = (
        movies.groupby("lam_split", as_index=False)
        .agg(AOGM=("AOGM", "mean"), LNK=("LNK", "mean"), TRA=("TRA", "mean"))
        .sort_values(["AOGM", "LNK"], ascending=[True, False])
    )
    best = by_lam.iloc[0]
    summary = {
        "val_best_lam_split": float(best["lam_split"]),
        "val_best_AOGM": float(best["AOGM"]),
        "val_best_LNK_ERR": float(1.0 - best["LNK"]),
        "val_best_TRA": float(best["TRA"]),
    }
    for row in by_lam.itertuples(index=False):
        suffix = f"{row.lam_split:g}".replace(".", "p")
        summary[f"val_AOGM_lam_split_{suffix}"] = float(row.AOGM)
        summary[f"val_LNK_ERR_lam_split_{suffix}"] = float(1.0 - row.LNK)
    return summary
