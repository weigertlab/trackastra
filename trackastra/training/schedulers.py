"""Learning-rate schedulers for Trackastra training."""

from __future__ import annotations

import logging
import warnings

import numpy as np
from torch.optim.lr_scheduler import LRScheduler


class WarmupCosineLRScheduler(LRScheduler):
    """A linear warmup plus cosine learning-rate scheduler."""

    def __init__(
        self,
        optimizer,
        warmup_epochs: int,
        max_epochs: int,
        cosine_final: float = 0.001,
        last_epoch: int = -1,
    ):
        """Use cosine_final to switch on or off the cosine annealing."""
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.cosine_final = cosine_final
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                stacklevel=2,
            )

        if self.last_epoch < self.warmup_epochs:
            initial = 1e-2
            factor = initial + (1 - initial) * self.last_epoch / self.warmup_epochs
        else:
            epoch_rel = (self.last_epoch - self.warmup_epochs) / (
                self.max_epochs - self.warmup_epochs + 1
            )
            factor = (
                0.5 * (1 + np.cos(np.pi * epoch_rel)) * (1 - self.cosine_final)
                + self.cosine_final
            )

        logging.info("LRScheduler: relative lr factor %.03f", factor)
        return [factor * base_lr for base_lr in self.base_lrs]
