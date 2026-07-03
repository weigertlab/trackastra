"""Training callbacks used by the Trackastra trainer."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import lightning as pl
import numpy as np
import yaml

from trackastra.model import INFERENCE_CONFIG_KEYS


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
