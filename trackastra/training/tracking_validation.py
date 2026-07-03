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
