"""Lightning runtime construction for Trackastra training."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from lightning.pytorch.utilities.rank_zero import rank_zero_only

import trackastra
from trackastra.training.callbacks import PreciseProgressBar, TrackastraModelCheckpoint

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LightningTrainerRuntime:
    """Lightning runtime objects derived from training config."""

    logdir: Path | None
    logger: Any
    callbacks: list[Any]
    profiler: Any
    run_name: str | None


def _lightning_trainer_class():
    import lightning as pl

    return pl.Trainer


def git_commit() -> str:
    """Return the current repository commit hash, or ``none`` outside git."""
    import git

    logging.debug("Trackastra path: %s", Path(trackastra.__path__[0]).resolve())
    try:
        return str(git.Repo(Path(trackastra.__path__[0]).resolve().parent).commit())
    except Exception:
        return "none"


def create_run_name(name: str | None, *, timestamp: bool) -> str:
    """Create the run name used for output folders and loggers."""
    if timestamp:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return f"{timestamp}_{name}"
    return str(name)


@rank_zero_only
def _init_wandb(project: str, name: str, config: dict[str, Any], save_dir: Path) -> None:
    import wandb

    _ = wandb.init(project=project, name=name, config=config, dir=save_dir)


def build_lightning_runtime(
    *,
    dry: bool,
    timestamp: bool,
    name: str | None,
    outdir: Path | str,
    resume: bool,
    logger_name: str,
    wandb_project: str,
    profile: bool,
    training_args: dict[str, Any],
    git_commit_hash: str | None = None,
) -> LightningTrainerRuntime:
    """Build callbacks, logger, profiler, and logdir from plain config values."""
    import lightning as pl
    from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
    from lightning.pytorch.profilers import PyTorchProfiler

    callbacks = []
    if not dry:
        run_name = create_run_name(name, timestamp=timestamp)
        logdir = Path(outdir) / run_name
        if logdir.exists() and not resume:
            raise ValueError(
                f'Logdir {logdir} exists, set "--resume t" if you want to resume'
            )
        callbacks.append(
            pl.pytorch.callbacks.ModelCheckpoint(
                dirpath=logdir / "checkpoints",
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                save_last=True,
            )
        )

        training_args = {
            **training_args,
            "git_commit": git_commit() if git_commit_hash is None else git_commit_hash,
        }
        callbacks.append(
            TrackastraModelCheckpoint(logdir, training_args, monitor="val_loss")
        )

        if logger_name == "tensorboard":
            train_logger = TensorBoardLogger(logdir, name="tb")
        elif logger_name == "wandb":
            train_logger = WandbLogger(
                name=run_name,
                project=wandb_project,
                save_dir=logdir,
            )
            _init_wandb(
                project=wandb_project,
                name=run_name,
                config=training_args,
                save_dir=logdir,
            )
        elif logger_name == "none":
            train_logger = False
        else:
            raise ValueError(f"Unknown logger {logger_name}")
    else:
        run_name = None
        logdir = None
        train_logger = False

    if train_logger:
        callbacks.append(
            pl.pytorch.callbacks.LearningRateMonitor(logging_interval="epoch")
        )
    callbacks.append(pl.pytorch.callbacks.Timer(interval="epoch"))
    callbacks.append(PreciseProgressBar(precision=8))

    profiler = (
        PyTorchProfiler(dirpath=".", filename="profile", skip_first=16)
        if profile
        else None
    )

    return LightningTrainerRuntime(
        logdir=logdir,
        logger=train_logger,
        callbacks=callbacks,
        profiler=profiler,
        run_name=run_name,
    )


def checkpoint_path(logdir: Path | str) -> Path | None:
    """Return the standard last checkpoint path when it exists."""
    path = Path(logdir) / "checkpoints" / "last.ckpt"
    return path if path.exists() else None


def build_or_resume_lightning_module(
    module_cls: Any,
    module_kwargs: dict[str, Any],
    *,
    logdir: Path | None,
    resume: bool,
):
    """Build a Lightning module, optionally restoring the wrapper from last.ckpt."""
    module = module_cls(**module_kwargs)
    if logdir is None or not resume or not Path(logdir).exists():
        return module

    logging.info("logdir exists, loading last state of model")
    path = checkpoint_path(logdir)
    if path is None:
        logging.warning("No checkpoint found in %s", logdir)
        return module
    return module_cls.load_from_checkpoint(path, **module_kwargs)


def resume_checkpoint_path(
    *,
    logdir: Path | None,
    resume: bool,
) -> Path | None:
    """Return the Lightning trainer ckpt_path for fit()."""
    if logdir is None or not resume:
        return None
    path = checkpoint_path(logdir)
    logger.info("Resuming from %s", path)
    return path


def configure_lightning_module_runtime_paths(
    module: Any,
    *,
    logdir: Path | None,
    debug: bool,
) -> None:
    """Set optional debug and metrics output paths on a Lightning module."""
    if debug:
        debug_root = (Path(logdir) if logdir is not None else Path(".")) / "debug"
        module.loss_spike_debug_dir = debug_root / "debug_loss_spikes"
        module.batch_provenance_path = debug_root
        module.viz_debug_dir = debug_root / "viz"
    else:
        module.loss_spike_debug_dir = None
        module.batch_provenance_path = None
        module.viz_debug_dir = None
    module.tracking_metrics_path = Path(logdir) / "metrics" if logdir is not None else None
