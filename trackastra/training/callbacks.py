"""Training callbacks used by the Trackastra trainer."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

import lightning as pl
import numpy as np
import yaml

from trackastra.model import INFERENCE_CONFIG_KEYS


def _scalar_metric(value: Any) -> int | float | None:
    """Convert a scalar metric to a CSV-safe Python value."""
    if hasattr(value, "numel"):
        if value.numel() != 1:
            return None
        value = value.detach().cpu().item()
    elif isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


class EpochMetricsCSV(pl.pytorch.callbacks.Checkpoint):
    """Write one row of aggregated scalar metrics per training epoch."""

    def __init__(self, logdir: Path | str):
        self._path = Path(logdir) / "metrics" / "metrics.csv"

    def _read_rows(self) -> list[dict[str, str]]:
        if not self._path.exists():
            return []
        with open(self._path, newline="") as f:
            return list(csv.DictReader(f))

    def _write_row(self, row: dict[str, int | float]) -> None:
        rows = self._read_rows()
        epoch = str(row["epoch"])
        rows = [existing for existing in rows if existing.get("epoch") != epoch]
        rows.append({key: str(value) for key, value in row.items()})
        rows.sort(key=lambda item: int(item["epoch"]))

        fieldnames = ["epoch", "step"]
        fieldnames.extend(
            sorted({key for item in rows for key in item} - set(fieldnames))
        )
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_suffix(".tmp")
        with open(temporary, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(self._path)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if not trainer.is_global_zero or trainer.sanity_checking:
            return

        metrics = {str(key): value for key, value in trainer.callback_metrics.items()}
        epoch_metric_bases = {
            key.removesuffix("_epoch") for key in metrics if key.endswith("_epoch")
        }
        row: dict[str, int | float] = {
            "epoch": int(trainer.current_epoch),
            "step": int(trainer.global_step),
        }
        for key, value in metrics.items():
            if key.endswith("_step") or key in epoch_metric_bases:
                continue
            scalar = _scalar_metric(value)
            if scalar is not None:
                row[key] = scalar

        for optimizer_index, optimizer in enumerate(trainer.optimizers):
            name = f"lr-{type(optimizer).__name__}"
            if len(trainer.optimizers) > 1:
                name = f"{name}-{optimizer_index}"
            for group_index, group in enumerate(optimizer.param_groups):
                key = (
                    name
                    if len(optimizer.param_groups) == 1
                    else f"{name}/pg{group_index}"
                )
                row[key] = float(group["lr"])

        self._write_row(row)


class PreciseProgressBar(pl.pytorch.callbacks.TQDMProgressBar):
    """Progress bar that shows loss metrics with higher precision."""

    def __init__(self, precision: int = 8, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._precision = precision

    def get_metrics(self, trainer, pl_module):
        items = super().get_metrics(trainer, pl_module)
        for key, value in items.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                items[key] = f"{value:.{self._precision}f}"
        return items


class TrackastraModelCheckpoint(pl.pytorch.callbacks.Callback):
    """Export Trackastra model folders during Lightning training."""

    def __init__(self, logdir: Path | str, training_args: dict, monitor: str = "val_loss"):
        self._logdir = Path(logdir)
        self._monitor = monitor
        self._best = np.inf
        self._training_args = training_args

    def on_fit_start(self, trainer, pl_module) -> None:
        if trainer.is_global_zero:
            logging.info("using logdir %s", self._logdir)
            self._logdir.mkdir(parents=True, exist_ok=True)
            with open(self._logdir / "train_config.yaml", "w") as f:
                yaml.safe_dump(self._training_args, f)
            inference_config = getattr(pl_module, "inference_config", None) or {
                k: self._training_args.get(k) for k in INFERENCE_CONFIG_KEYS
            }
            with open(self._logdir / "inference_config.yaml", "w") as f:
                yaml.safe_dump(inference_config, f)

    def on_validation_end(self, trainer, pl_module) -> None:
        if trainer.is_global_zero and not trainer.sanity_checking:
            value = trainer.logged_metrics[self._monitor]
            if value < self._best:
                self._best = value
                logging.info("saved best model with %s=%.5f", self._monitor, value)
                pl_module.model.save(self._logdir)
