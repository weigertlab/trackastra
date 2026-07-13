"""Public training builders and source-loading helpers."""

from __future__ import annotations

import logging
import os
import warnings
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import joblib
import pandas as pd
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from trackastra import model as model_api
from trackastra.data import wrfeat
from trackastra.data.dataset import (
    TrackingDataset,
    association_distances,
    collate_sequence_padding,
)
from trackastra.data.distributed import (
    BalancedBatchSampler,
    BalancedDistributedSampler,
    SequenceInputSpec,
    _looks_like_ctc,
    _looks_like_geff,
    normalize_sequence_input_specs,
)
from trackastra.data.io import TrackingSequence
from trackastra.model import TrackingTransformer
from trackastra.training.runtime import (
    LightningTrainerRuntime,
    build_lightning_runtime,
    build_or_resume_lightning_module,
    configure_lightning_module_runtime_paths,
    resume_checkpoint_path,
)

SourceFormat = Literal["ctc", "geff"]
SourceSpec = dict[str, Any]
SequenceLoader = Callable[..., TrackingSequence]
logger = logging.getLogger(__name__)

_CURRENT_MODEL_CONFIG_OVERRIDES = frozenset({"causal_norm"})

SEQUENCE_LOADERS: dict[SourceFormat, SequenceLoader] = {
    "ctc": TrackingSequence.from_ctc,
    "geff": TrackingSequence.from_geff,
}


@dataclass(frozen=True)
class DataSplitConfig:
    """Resolved, inspectable training data configuration for one split."""

    split: Literal["train", "val"]
    sources: tuple[SourceSpec, ...]
    sequence_kwargs: dict[str, Any] = field(default_factory=dict)
    dataset_kwargs: dict[str, Any] = field(default_factory=dict)
    sampler_kwargs: dict[str, Any] = field(default_factory=dict)
    loader_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainConfig:
    """Training options that are not part of source loading or model construction."""

    epochs: int = 100
    warmup_epochs: int = 2
    learning_rate: float = 1e-4
    batch_size: int = 8
    distributed: bool = False
    device: Literal["auto", "cpu", "cuda"] = "auto"
    logger: Literal["tensorboard", "wandb", "none"] = "tensorboard"
    outdir: Path = Path("runs")
    name: str | None = None
    resume: bool = False
    dry: bool = False
    mixed_precision: bool = True
    compile: bool = False
    cache_dir: Path | None = None
    loss_kwargs: dict[str, Any] = field(default_factory=dict)
    tracking_kwargs: dict[str, Any] = field(default_factory=dict)
    runtime_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingRun:
    """Objects created by one trainer facade fit call."""

    trainer: Any
    lightning_module: Any
    result: Any


@dataclass(frozen=True)
class TrackingDatasetBundle:
    """Inspectable dataset plus split-specific loader settings."""

    dataset: ConcatDataset
    config: DataSplitConfig

    @property
    def split(self) -> Literal["train", "val"]:
        return self.config.split

    @property
    def datasets(self) -> list[Dataset]:
        return self.dataset.datasets

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Any:
        return self.dataset[index]

    def dataloader(self, *, distributed: bool = False) -> DataLoader:
        """Build a dataloader using the split's stored sampler and loader config."""
        loader_kwargs = deepcopy(self.config.loader_kwargs)
        loader_kwargs.setdefault("collate_fn", collate_sequence_padding)
        if self.split == "train":
            return self._train_dataloader(loader_kwargs, distributed=distributed)
        return self._val_dataloader(loader_kwargs)

    def _train_dataloader(
        self, loader_kwargs: dict[str, Any], *, distributed: bool
    ) -> DataLoader:
        if distributed:
            sampler = BalancedDistributedSampler(
                self.dataset,
                **self.config.sampler_kwargs,
            )
            batch_sampler = None
        else:
            sampler = None
            batch_sampler = BalancedBatchSampler(
                self.dataset,
                **self.config.sampler_kwargs,
            )
            if loader_kwargs.get("batch_size") != batch_sampler.batch_size:
                raise ValueError(
                    f"Batch size in loader_kwargs ({loader_kwargs.get('batch_size')}) "
                    f"and sampler_kwargs ({batch_sampler.batch_size}) must match"
                )
            loader_kwargs.pop("batch_size", None)
        return DataLoader(
            self.dataset,
            sampler=sampler,
            batch_sampler=batch_sampler,
            **loader_kwargs,
        )

    def _val_dataloader(self, loader_kwargs: dict[str, Any]) -> DataLoader:
        loader_kwargs["persistent_workers"] = False
        num_workers = loader_kwargs.get("num_workers", 0)
        loader_kwargs["num_workers"] = (
            0 if num_workers == 0 else max(1, num_workers // 2)
        )
        return DataLoader(
            self.dataset,
            shuffle=False,
            **loader_kwargs,
        )


class SequenceLoadingError(RuntimeError):
    """Raised when one or more sequence sources fail to load."""


class SourceSpecError(ValueError):
    """Raised when one or more source specs cannot be normalized."""


def _tracking_lightning_module_class():
    from trackastra.training.lightning import TrackingLightningModule

    return TrackingLightningModule


def _lightning_trainer_class():
    import lightning as pl

    return pl.Trainer


def _feature_dim(ndim: int, features: str) -> int:
    try:
        return wrfeat.feature_output_dim(features, ndim)
    except ValueError as e:
        raise ValueError(f"Unknown feature mode {features!r}") from e


def _resolve_source_format(
    spec: SequenceInputSpec,
    sequence_kwargs: Mapping[str, Any],
) -> SourceFormat:
    if spec.format != "auto":
        return spec.format

    ctc_kwargs = dict(sequence_kwargs)
    ctc_kwargs.update(spec.loader_kwargs)
    geff_ok, geff_error = _looks_like_geff(spec.path)
    ctc_ok, ctc_error = _looks_like_ctc(spec.path, ctc_kwargs)
    if geff_ok and ctc_ok:
        raise ValueError(
            f"Could not auto-detect unique format for {spec.path}: "
            "both CTC and GEFF layouts are valid"
        )
    if geff_ok:
        return "geff"
    if ctc_ok:
        return "ctc"
    raise ValueError(
        f"Could not auto-detect sequence format for {spec.path}. "
        f"CTC check failed: {ctc_error}. GEFF check failed: {geff_error}."
    )


def _ctc_source_kwargs(
    spec: SequenceInputSpec,
    sequence_kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    kwargs = {"root": spec.path, **deepcopy(dict(sequence_kwargs))}
    kwargs["ndim"] = spec.source_ndim
    if spec.spacing is not None:
        if spec.spacing == "auto":
            raise ValueError('spacing="auto" is only valid for GEFF inputs')
        kwargs["spacing"] = spec.spacing
    kwargs.update(deepcopy(spec.loader_kwargs))
    return kwargs


def _geff_source_kwargs(spec: SequenceInputSpec) -> dict[str, Any]:
    kwargs = {
        "root_or_geff": spec.path,
        "sparse_gt": spec.sparse_gt,
        "source_ndim": spec.source_ndim,
        **deepcopy(spec.loader_kwargs),
    }
    if spec.spacing is not None:
        kwargs["spacing"] = spec.spacing
    return kwargs


def _input_label(item: object) -> str:
    if isinstance(item, str | Path):
        return str(item)
    if isinstance(item, Mapping):
        if "path" in item:
            return str(item["path"])
        if "paths" in item:
            return str(item["paths"])
    return repr(item)


def tracking_inputs_from_sources(
    sources: Sequence[Mapping[str, Any]],
    tracking_frequency: int,
) -> list[dict[str, Any]]:
    """Return CTC validation roots and their spacing from normalized sources.

    The spacing is carried along so full-movie tracking validation extracts
    features in the same model units the windowed dataset trains on.
    """
    inputs = []
    skipped_geff = []
    for source in sources:
        fmt = source.get("format")
        kwargs = source.get("kwargs", {})
        if fmt == "geff":
            skipped_geff.append(str(kwargs.get("root_or_geff", source)))
            continue
        if fmt != "ctc":
            raise ValueError(f"Unknown source format {fmt!r}")
        spacing = kwargs.get("spacing")
        inputs.append(
            {
                "root": Path(kwargs["root"]),
                "source_ndim": kwargs.get("ndim", "auto"),
                "spacing": None if spacing is None else [float(s) for s in spacing],
            }
        )
    if tracking_frequency > 0 and skipped_geff:
        logger.warning(
            "full-movie tracking validation is CTC-only; skipping %d GEFF validation"
            " input(s): %s",
            len(skipped_geff),
            ", ".join(skipped_geff),
        )
    return inputs


def normalize_source_specs(
    inputs: Sequence[str | Path | Mapping[str, Any]],
    *,
    ndim: int | None = None,
    sequence_kwargs: Mapping[str, Any] | None = None,
    split_sequence_kwargs: Mapping[str, Any] | None = None,
) -> tuple[SourceSpec, ...]:
    """Normalize user inputs into plain loader specs.

    Returned specs have the form ``{"format": "ctc", "kwargs": {...}}`` and can be
    passed directly to ``load_sequence`` or inspected by users.
    """
    merged_sequence_kwargs = dict(sequence_kwargs or {})
    merged_sequence_kwargs.update(split_sequence_kwargs or {})
    sources = []
    errors = []
    for item in inputs:
        try:
            specs = normalize_sequence_input_specs([item], ndim=ndim)
        except Exception as exc:
            errors.append(f"{_input_label(item)}\n  - {exc}")
            continue
        for spec in specs:
            try:
                fmt = _resolve_source_format(spec, merged_sequence_kwargs)
                kwargs = (
                    _ctc_source_kwargs(spec, merged_sequence_kwargs)
                    if fmt == "ctc"
                    else _geff_source_kwargs(spec)
                )
            except Exception as exc:
                errors.append(f"{spec.path}\n  - {exc}")
                continue
            sources.append({"format": fmt, "kwargs": kwargs})
    if errors:
        joined = "\n\n".join(errors)
        raise SourceSpecError(
            f"Source preflight failed for {len(errors)} source(s):\n\n{joined}"
        )
    return tuple(sources)


def _source_label(source: Mapping[str, Any]) -> str:
    kwargs = source.get("kwargs", {})
    if isinstance(kwargs, Mapping):
        for key in ("root", "root_or_geff"):
            if key in kwargs:
                return str(kwargs[key])
    return repr(source)


def load_sequence(
    spec: Mapping[str, Any],
    loaders: Mapping[str, SequenceLoader] | None = None,
    *,
    cache_dir: Path | str | None = None,
) -> TrackingSequence:
    """Load one sequence from a normalized source spec."""
    fmt = spec.get("format")
    if not isinstance(fmt, str):
        raise ValueError("source spec must contain a string 'format'")
    kwargs = spec.get("kwargs")
    if not isinstance(kwargs, Mapping):
        raise ValueError("source spec must contain mapping 'kwargs'")
    registry = {**SEQUENCE_LOADERS, **(loaders or {})}
    if fmt not in registry:
        raise ValueError(f"Unknown sequence format {fmt!r}")
    loader = _cached_sequence_loader(fmt, registry[fmt], cache_dir=cache_dir)
    load_kwargs = dict(kwargs)
    if cache_dir is not None:
        hit = loader.check_call_in_cache(**load_kwargs)
        logger.info(
            "%s (%s): %s",
            _source_label(spec),
            fmt,
            "loaded from cache" if hit else "cache miss, computing",
        )
    else:
        logger.info("%s (%s): loading from disk", _source_label(spec), fmt)
    return loader(**load_kwargs)


def _cached_sequence_loader(
    fmt: str,
    loader: SequenceLoader,
    *,
    cache_dir: Path | str | None,
):
    if cache_dir is None:
        return loader
    memory = joblib.Memory(cache_dir, verbose=0)
    if fmt == "ctc":
        return memory.cache(loader, ignore=["n_workers"])
    return memory.cache(loader)


def load_sequences(
    specs: Sequence[Mapping[str, Any]],
    loaders: Mapping[str, SequenceLoader] | None = None,
    *,
    cache_dir: Path | str | None = None,
) -> tuple[TrackingSequence, ...]:
    """Load many sequences and report all failing sources together."""
    sequences = []
    errors = []
    for spec in specs:
        try:
            sequences.append(load_sequence(spec, loaders=loaders, cache_dir=cache_dir))
        except Exception as exc:
            errors.append(f"{_source_label(spec)}\n  - {exc}")
    if errors:
        joined = "\n\n".join(errors)
        raise SequenceLoadingError(
            f"Sequence loading failed for {len(errors)} source(s):\n\n{joined}"
        )
    return tuple(sequences)


def build_dataset(
    sequences: Sequence[TrackingSequence],
    data_config: DataSplitConfig,
) -> TrackingDatasetBundle:
    """Build the torch-facing dataset for a split."""
    dataset = ConcatDataset(
        [
            TrackingDataset(sequence, dataset_index=i, **data_config.dataset_kwargs)
            for i, sequence in enumerate(sequences)
        ],
    )
    return TrackingDatasetBundle(dataset=dataset, config=data_config)


def _balanced_sampler_from_loader(loader: DataLoader) -> BalancedBatchSampler | None:
    batch_sampler = getattr(loader, "batch_sampler", None)
    if isinstance(batch_sampler, BalancedBatchSampler):
        return batch_sampler
    sampler = getattr(loader, "sampler", None)
    balanced = getattr(sampler, "_balanced_batch_sampler", None)
    if isinstance(balanced, BalancedBatchSampler):
        return balanced
    return None


def _write_sampler_prob_debug(loader: DataLoader, path: Path | str | None) -> None:
    """Write the train sampler's per-window sampling probabilities to CSV."""
    if path is None or int(os.environ.get("RANK", "0")) != 0:
        return
    sampler = _balanced_sampler_from_loader(loader)
    if sampler is None:
        return

    all_indices = list(range(len(sampler.n_objects)))
    if not all_indices:
        return
    global_probs = sampler.get_probs(all_indices)
    within_dataset_probs: dict[int, float] = {}
    dataset_prob_mass: dict[int, float] = {}
    for dataset_index in sorted(set(int(x) for x in sampler.dataset_ids)):
        idxs = [
            i
            for i, current_dataset in enumerate(sampler.dataset_ids)
            if int(current_dataset) == dataset_index
        ]
        probs = sampler.get_probs(idxs)
        for i, prob in zip(idxs, probs):
            within_dataset_probs[i] = float(prob)
        dataset_prob_mass[dataset_index] = float(
            sum(float(global_probs[i]) for i in idxs)
        )

    datasets = getattr(getattr(loader, "dataset", None), "datasets", [])
    roots = {i: str(getattr(dataset, "root", "")) for i, dataset in enumerate(datasets)}
    local_index_by_dataset: dict[int, int] = {}
    rows = []
    for global_index in all_indices:
        dataset_index = int(sampler.dataset_ids[global_index])
        local_index = local_index_by_dataset.get(dataset_index, 0)
        local_index_by_dataset[dataset_index] = local_index + 1
        rows.append(
            {
                "global_index": global_index,
                "dataset_index": dataset_index,
                "dataset_root": roots.get(dataset_index, ""),
                "window_index": local_index,
                "n_objects": int(sampler.n_objects[global_index]),
                "n_divs": int(sampler.n_divs[global_index]),
                "sample_weight": float(sampler.sample_weight[global_index]),
                "sample_prob": float(global_probs[global_index]),
                "sample_prob_within_dataset": within_dataset_probs[global_index],
                "dataset_prob_mass": dataset_prob_mass[dataset_index],
                "oversample_divs": float(sampler.oversample_divs),
                "oversample_density": float(sampler.oversample_density),
                "weight_by_dataset": bool(sampler.weight_by_dataset),
            }
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _percentile(values: Any, q: float) -> float:
    """Return one percentile, or NaN for an empty array."""
    import numpy as np

    values = np.asarray(values)
    return float(np.percentile(values, q)) if values.size else float("nan")


def _positive_detection_edges(
    dataset: TrackingDataset,
    *,
    delta_cutoff: int,
) -> list[tuple[int, int, int]]:
    """Return positive detection edges as ``(segment, source, target)``."""
    import numpy as np

    relation = dataset.sequence.gt.lineage_relation
    edges = []
    for seg_index, (segment, supervision) in enumerate(
        zip(dataset.sequence.detections, dataset.sequence.supervision)
    ):
        if supervision is None:
            continue
        lineage = np.asarray(supervision.lineage_index)
        matched = lineage >= 0
        times = np.asarray(segment.timepoints)
        for source_time in np.unique(times[matched]):
            source = np.flatnonzero((times == source_time) & matched)
            for delta in range(1, delta_cutoff + 1):
                target = np.flatnonzero((times == source_time + delta) & matched)
                if not len(target):
                    continue
                rows, cols = np.nonzero(
                    relation[np.ix_(lineage[source], lineage[target])]
                )
                edges.extend(
                    (seg_index, int(source[row]), int(target[col]))
                    for row, col in zip(rows, cols)
                )
    return edges


def _candidate_link_audit(
    dataset: TrackingDataset,
    *,
    candidate_k: int,
    candidate_mode: str,
    spatial_cutoff: float,
    delta_cutoff: int,
    max_edges: int = 10_000,
) -> dict[str, float | int]:
    """Audit a bounded deterministic sample of positive links against target-frame kNN."""
    import numpy as np
    from scipy.spatial import cKDTree

    edges = _positive_detection_edges(dataset, delta_cutoff=delta_cutoff)
    n_positive = len(edges)
    if n_positive > max_edges:
        take = np.linspace(0, n_positive - 1, max_edges, dtype=np.int64)
        edges = [edges[i] for i in take]

    hits = 0
    within_cutoff = 0
    distances = []
    for seg_index in range(len(dataset.sequence.detections)):
        segment = dataset.sequence.detections[seg_index]
        coords = np.asarray(segment.coords)
        times = np.asarray(segment.timepoints)
        segment_edges = [
            (source, target) for seg, source, target in edges if seg == seg_index
        ]
        for target_time in sorted({int(times[target]) for _, target in segment_edges}):
            target_frame = np.flatnonzero(times == target_time)
            current = [
                (source, target)
                for source, target in segment_edges
                if int(times[target]) == target_time
            ]
            if not current or not len(target_frame):
                continue
            local_target = {int(index): i for i, index in enumerate(target_frame)}
            sources = np.asarray([source for source, _ in current], dtype=np.int64)
            targets = np.asarray([target for _, target in current], dtype=np.int64)
            distance = np.linalg.norm(coords[targets] - coords[sources], axis=1)
            distances.extend(distance.tolist())
            within_cutoff += int(np.sum(distance <= spatial_cutoff))

            if candidate_mode in ("per_frame", "next_frame"):
                k = min(candidate_k, len(target_frame))
                _distance, neighbors = cKDTree(coords[target_frame]).query(
                    coords[sources], k=k
                )
                neighbors = np.asarray(neighbors)
                if neighbors.ndim == 1:
                    neighbors = neighbors[:, None]
                target_local = np.asarray(
                    [local_target[int(target)] for target in targets]
                )
                hits += int(np.sum(np.any(neighbors == target_local[:, None], axis=1)))

    audited = len(edges)
    return {
        "positive_edges": n_positive,
        "candidate_edges_audited": audited,
        "candidate_recall_at_k": (
            hits / audited
            if audited and candidate_mode in ("per_frame", "next_frame")
            else float("nan")
        ),
        "cutoff_recall": within_cutoff / audited if audited else float("nan"),
        "positive_distance_p50": _percentile(distances, 50),
        "positive_distance_p90": _percentile(distances, 90),
        "positive_distance_p99": _percentile(distances, 99),
    }


def _feature_channel_stats(dataset: Any) -> dict[str, float]:
    """Mean/std of every derived feature channel over all detections of a dataset.

    Applies the dataset's own feature recipe and diameter scaling, so values are in the
    units the model consumes, but without augmentation or detection dropping. Channels
    the dataset cannot provide (missing source property, or 3D-only channels on 2D data)
    are masked out and reported as NaN.
    """
    import numpy as np

    mode = getattr(dataset, "features", "none")
    channels = wrfeat.feature_channels(mode, dataset.ndim)
    if not channels:
        return {}

    total = np.zeros(len(channels), dtype=np.float64)
    total_sq = np.zeros(len(channels), dtype=np.float64)
    count = np.zeros(len(channels), dtype=np.float64)
    for segment in dataset.sequence.detections:
        feature = wrfeat.WRFeatures(
            coords=np.asarray(segment.coords, dtype=np.float32),
            labels=np.asarray(segment.labels),
            timepoints=np.asarray(segment.timepoints, dtype=np.int32),
            features={k: np.asarray(v) for k, v in segment.features.items()},
        )
        feature = wrfeat.scale_feature_geometry(feature, dataset.scale_factor)
        values, mask = feature.stacked_with_mask(mode)
        valid = np.where(mask, values, 0.0).astype(np.float64)
        total += valid.sum(axis=0)
        total_sq += np.square(valid).sum(axis=0)
        count += mask.sum(axis=0)

    stats: dict[str, float] = {}
    for index, channel in enumerate(channels):
        n = count[index]
        if not n:
            stats[f"feat_{channel.name}_mean"] = float("nan")
            stats[f"feat_{channel.name}_std"] = float("nan")
            continue
        mean = total[index] / n
        variance = max(total_sq[index] / n - mean**2, 0.0)
        stats[f"feat_{channel.name}_mean"] = float(mean)
        stats[f"feat_{channel.name}_std"] = float(np.sqrt(variance))
    return stats


def _displacement_stats(dataset: Any, delta_cutoff: int) -> dict[str, float]:
    """Displacement between consecutively linked objects, in model units.

    Exact over every unique positive association within ``delta_cutoff`` frames,
    unlike the sampled ``positive_distance_*`` percentiles of the candidate audit.
    """
    import numpy as np

    distances = association_distances(dataset, delta_cutoff)
    if not len(distances):
        return {}
    return {
        "displacement_mean": float(distances.mean()),
        "displacement_std": float(distances.std()),
        "displacement_max": float(distances.max()),
    }


def _lineage_event_stats(gt: Any) -> dict[str, float]:
    """Fraction of GT nodes that appear, disappear, or divide.

    Nodes in the first (last) frame are excluded from the appearance (disappearance)
    count, where the event is forced by the movie boundary rather than being a real
    track birth or death. Nodes whose incoming/outgoing GT edge set is not fully known
    are excluded from the corresponding count, as they already are for divisions.
    """
    import numpy as np

    if gt is None or gt.node_in_degree is None or gt.node_out_degree is None:
        return {}
    timepoints = np.asarray(gt.timepoints)
    if not len(timepoints):
        return {}
    in_degree = np.asarray(gt.node_in_degree)
    out_degree = np.asarray(gt.node_out_degree)
    predecessor_available = (
        np.ones(len(in_degree), dtype=bool)
        if gt.node_predecessor_set_available is None
        else np.asarray(gt.node_predecessor_set_available, dtype=bool)
    )
    successor_available = (
        np.ones(len(out_degree), dtype=bool)
        if gt.node_successor_set_available is None
        else np.asarray(gt.node_successor_set_available, dtype=bool)
    )
    appear_eligible = predecessor_available & (timepoints > timepoints.min())
    disappear_eligible = successor_available & (timepoints < timepoints.max())

    def _fraction(hits: np.ndarray, eligible: np.ndarray) -> float:
        n = int(eligible.sum())
        return float(hits[eligible].sum() / n) if n else float("nan")

    return {
        "appear_fraction": _fraction(in_degree == 0, appear_eligible),
        "disappear_fraction": _fraction(out_degree == 0, disappear_eligible),
        "division_fraction": _fraction(out_degree >= 2, successor_available),
    }


def _dataset_summary_rows(
    loader: DataLoader,
    *,
    candidate_k: int,
    candidate_mode: str,
    spatial_cutoff: float,
    delta_cutoff: int,
    max_candidate_edges: int = 10_000,
) -> list[dict[str, Any]]:
    """Build one cheap diagnostic row per training dataset."""
    import numpy as np

    sampler = _balanced_sampler_from_loader(loader)
    datasets = getattr(getattr(loader, "dataset", None), "datasets", [])
    if sampler is None or not datasets:
        return []

    all_indices = np.arange(len(sampler.n_objects))
    global_probs = sampler.get_probs(all_indices)
    expected_samples = (
        int(sampler.num_samples)
        if sampler.num_samples is not None
        else len(all_indices)
    )
    rows = []
    for dataset_index, dataset in enumerate(datasets):
        window_mask = np.asarray(sampler.dataset_ids) == dataset_index
        window_objects = np.asarray(sampler.n_objects)[window_mask]
        window_divs = np.asarray(sampler.n_divs)[window_mask]
        sample_weights = np.asarray(sampler.sample_weight)[window_mask]
        probability_mass = float(global_probs[window_mask].sum())

        frame_counts = []
        feature_sets = []
        matched_count = 0
        matched_gt_nodes = set()
        predecessor_available = []
        successor_available = []
        for segment, supervision in zip(
            dataset.sequence.detections, dataset.sequence.supervision
        ):
            feature_sets.append(set(segment.features))
            frame_counts.extend(
                int(np.sum(segment.timepoints == time))
                for time in range(segment.n_frames)
            )
            if supervision is None:
                continue
            matched = (
                np.asarray(supervision.matched_gt, dtype=bool)
                if supervision.matched_gt is not None
                else np.asarray(supervision.lineage_index) >= 0
            )
            matched_count += int(matched.sum())
            if supervision.gt_node_index is not None:
                matched_gt_nodes.update(
                    int(index)
                    for index in np.asarray(supervision.gt_node_index)[matched]
                    if index >= 0
                )
            if supervision.gt_predecessor_set_available is not None:
                predecessor_available.extend(
                    np.asarray(supervision.gt_predecessor_set_available)[
                        matched
                    ].tolist()
                )
            if supervision.gt_successor_set_available is not None:
                successor_available.extend(
                    np.asarray(supervision.gt_successor_set_available)[matched].tolist()
                )

        n_detections = sum(len(segment) for segment in dataset.sequence.detections)
        feature_properties = set.intersection(*feature_sets) if feature_sets else set()
        gt = dataset.sequence.gt
        if gt.node_out_degree is None or gt.node_successor_set_available is None:
            division_nodes = 0
            eligible_division_nodes = 0
        else:
            gt_successor_available = np.asarray(
                gt.node_successor_set_available, dtype=bool
            )
            division_nodes = int(
                np.sum((gt.node_out_degree >= 2) & gt_successor_available)
            )
            eligible_division_nodes = int(gt_successor_available.sum())
        shape_properties = set(wrfeat.FEATURE_DROP_GROUPS["shape"])
        audit = _candidate_link_audit(
            dataset,
            candidate_k=candidate_k,
            candidate_mode=candidate_mode,
            spatial_cutoff=spatial_cutoff,
            delta_cutoff=delta_cutoff,
            max_edges=max_candidate_edges,
        )
        rows.append(
            {
                "dataset_index": dataset_index,
                "dataset_root": str(dataset.root),
                "source_format": "geff"
                if Path(dataset.root).suffix == ".geff"
                else "ctc",
                "n_frames": max(seg.n_frames for seg in dataset.sequence.detections),
                "n_windows": int(window_mask.sum()),
                "sample_probability_mass": probability_mass,
                "expected_samples_per_epoch": probability_mass * expected_samples,
                "n_detections": n_detections,
                "detections_per_frame_p50": _percentile(frame_counts, 50),
                "detections_per_frame_p90": _percentile(frame_counts, 90),
                "detections_per_frame_max": max(frame_counts, default=0),
                "crop_pressure_fraction": (
                    float(np.mean(np.asarray(frame_counts) > dataset.max_detections))
                    if dataset.max_detections is not None and frame_counts
                    else 0.0
                ),
                "window_objects_p50": _percentile(window_objects, 50),
                "window_objects_p90": _percentile(window_objects, 90),
                "window_divisions_mean": float(window_divs.mean()),
                "window_divisions_max": int(window_divs.max(initial=0)),
                "division_window_fraction": float(np.mean(window_divs > 0)),
                "sample_weight_mean": float(sample_weights.mean()),
                "n_gt_nodes": len(gt.coords),
                "n_matched_detections": matched_count,
                "matched_detection_fraction": matched_count / n_detections
                if n_detections
                else 0.0,
                "matched_gt_node_fraction": (
                    len(matched_gt_nodes) / len(gt.coords)
                    if len(gt.coords)
                    else float("nan")
                ),
                "predecessor_available_fraction": (
                    float(np.mean(predecessor_available))
                    if predecessor_available
                    else float("nan")
                ),
                "successor_available_fraction": (
                    float(np.mean(successor_available))
                    if successor_available
                    else float("nan")
                ),
                "division_nodes": division_nodes,
                "division_nodes_per_1000": (
                    1000 * division_nodes / eligible_division_nodes
                    if eligible_division_nodes
                    else float("nan")
                ),
                "intensity_available": wrfeat.FEATURE_INTENSITY in feature_properties,
                "shape_available": shape_properties.issubset(feature_properties),
                "feature_properties": ";".join(sorted(feature_properties)),
                "candidate_k": candidate_k,
                "candidate_mode": candidate_mode,
                "spatial_cutoff": spatial_cutoff,
                **_lineage_event_stats(gt),
                **_displacement_stats(dataset, delta_cutoff),
                **_feature_channel_stats(dataset),
                **audit,
            }
        )
    return rows


def _write_dataset_summary_debug(
    loader: DataLoader,
    path: Path | str | None,
    *,
    candidate_k: int,
    candidate_mode: str,
    spatial_cutoff: float,
    delta_cutoff: int,
    max_candidate_edges: int = 10_000,
) -> None:
    """Write one default startup diagnostic row per training dataset."""
    if path is None or int(os.environ.get("RANK", "0")) != 0:
        return
    rows = _dataset_summary_rows(
        loader,
        candidate_k=candidate_k,
        candidate_mode=candidate_mode,
        spatial_cutoff=spatial_cutoff,
        delta_cutoff=delta_cutoff,
        max_candidate_edges=max_candidate_edges,
    )
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _dataset_feature_dim(dataset: Any) -> int | None:
    if dataset is None:
        return None
    if isinstance(dataset, TrackingDatasetBundle):
        datasets = dataset.datasets
    elif isinstance(dataset, ConcatDataset):
        datasets = dataset.datasets
    else:
        datasets = (dataset,)
    dims = {int(ds.feat_dim) for ds in datasets if hasattr(ds, "feat_dim")}
    if not dims:
        return None
    if len(dims) > 1:
        raise ValueError(
            f"Training datasets have inconsistent feature dimensions: {sorted(dims)}"
        )
    return next(iter(dims))


def build_model(
    model_config: model_api.ModelConfig,
    train_dataset: Any | None = None,
) -> TrackingTransformer:
    """Construct a TrackingTransformer from a model config."""
    feat_dim = _dataset_feature_dim(train_dataset)
    if feat_dim is not None and feat_dim != model_config.feat_dim:
        raise ValueError(
            f"Model feat_dim={model_config.feat_dim} does not match "
            f"training dataset feat_dim={feat_dim}"
        )
    if model_config.model_path is not None:
        return load_model_from_path(
            model_config.model_path,
            expected_config=model_config.transformer_kwargs(),
        )
    return TrackingTransformer(**model_config.transformer_kwargs())


def pooled_node_degree_counts(bundle: Any) -> tuple[Any, Any]:
    """Pad-and-sum per-dataset node in/out degree bincounts over a dataset bundle."""
    import numpy as np

    def _pool(attr: str):
        total = np.zeros(0, dtype=np.int64)
        for ds in getattr(bundle, "datasets", []):
            counts = getattr(ds, attr, None)
            if counts is None or len(counts) == 0:
                continue
            counts = np.asarray(counts, dtype=np.int64)
            if len(counts) > len(total):
                total = np.pad(total, (0, len(counts) - len(total)))
            total[: len(counts)] += counts
        return total

    return _pool("node_in_degree_counts"), _pool("node_out_degree_counts")


def node_degree_class_weights(counts: Any, num_classes: int) -> Any:
    """Class weights ``1 / (1 + sqrt(n_c))`` normalized to mean 1.

    Counts are truncated/zero-padded to ``num_classes``; a nonzero count beyond the
    head's class range (e.g. a merge in in-degree) is warned about, not silently kept.
    """
    import numpy as np
    import torch

    counts = np.asarray(counts, dtype=np.float64)
    if len(counts) > num_classes and counts[num_classes:].sum() > 0:
        logger.warning(
            "Node-degree counts have %d samples in classes >= %d (beyond the head's "
            "%d classes); these are dropped from the class weights.",
            int(counts[num_classes:].sum()),
            num_classes,
            num_classes,
        )
    fixed = np.zeros(num_classes, dtype=np.float64)
    fixed[: min(num_classes, len(counts))] = counts[:num_classes]
    weights = 1.0 / (1.0 + np.sqrt(fixed))
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def resolve_model_checkpoint_reference(path: Path | str) -> tuple[Path, Path | None]:
    """Resolve a model folder and optional checkpoint path from a user path."""
    path = Path(path).expanduser()
    if path.is_dir():
        if not (path / "config.yaml").exists():
            raise FileNotFoundError(
                f"Could not find model config: {path / 'config.yaml'}"
            )
        return path, None
    if not path.is_file():
        raise FileNotFoundError(f"Could not find model path: {path}")

    for folder in (path.parent, *path.parents):
        if (folder / "config.yaml").exists():
            return folder, path.relative_to(folder)
    raise FileNotFoundError(
        f"Could not find model config.yaml in {path} or any parent directory"
    )


def load_model_from_path(
    path: Path | str,
    *,
    args: Any = None,
    map_location: Any = None,
    model_cls: Any = TrackingTransformer,
    expected_config: Mapping[str, Any] | None = None,
) -> TrackingTransformer:
    """Load a TrackingTransformer from a model folder or checkpoint file."""
    folder, checkpoint_path = resolve_model_checkpoint_reference(path)
    if expected_config is not None:
        _warn_model_config_mismatches(folder, expected_config)
    kwargs = {"args": args, "map_location": map_location}
    if checkpoint_path is not None:
        kwargs["checkpoint_path"] = checkpoint_path
    model = model_cls.from_folder(folder, **kwargs)
    if expected_config is not None:
        for key in _CURRENT_MODEL_CONFIG_OVERRIDES:
            if key in expected_config:
                model.config[key] = expected_config[key]
    return model


def _config_values_match(left: Any, right: Any) -> bool:
    if isinstance(left, Sequence) and not isinstance(left, str | bytes):
        if isinstance(right, Sequence) and not isinstance(right, str | bytes):
            return tuple(left) == tuple(right)
    return left == right


def _warn_model_config_mismatches(
    folder: Path, expected_config: Mapping[str, Any]
) -> None:
    import yaml

    with open(folder / "config.yaml") as f:
        loaded_config = yaml.safe_load(f) or {}

    mismatches = []
    for key, value in sorted(loaded_config.items()):
        if key in expected_config and not _config_values_match(
            value, expected_config[key]
        ):
            source = (
                "current" if key in _CURRENT_MODEL_CONFIG_OVERRIDES else "loaded"
            )
            mismatches.append(
                f"{key}: loaded={value!r}, current={expected_config[key]!r} "
                f"(using {source})"
            )
    if mismatches:
        logger.warning(
            "Loaded model config differs from current config:\n%s",
            "\n".join(f"  {mismatch}" for mismatch in mismatches),
        )


def _inference_config_from_args(args: Any) -> dict[str, Any]:
    """Build the inference config saved beside trained models."""
    from trackastra.model import INFERENCE_CONFIG_KEYS

    config = {
        key: getattr(args, key, None)
        for key in INFERENCE_CONFIG_KEYS
        if key != "feature_schema"
    }
    schema = wrfeat.feature_schema_manifest(args.features)
    if schema is not None:
        config["feature_schema"] = schema
    return config


@dataclass(frozen=True)
class TrackastraTrainer:
    """Small public facade hiding Lightning construction."""

    config: TrainConfig

    def _accelerator(self) -> str:
        if self.config.device != "auto":
            return self.config.device
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def _devices(self, accelerator: str) -> int:
        if not self.config.distributed or accelerator != "cuda":
            return 1
        import torch

        return torch.cuda.device_count()

    def _strategy(self) -> str:
        return "ddp_find_unused_parameters_true" if self.config.distributed else "auto"

    def _lightning_module_kwargs(self, model: Any) -> dict[str, Any]:
        module_kwargs = {
            "model": model,
            "warmup_epochs": self.config.warmup_epochs,
            "max_epochs": self.config.epochs,
            "learning_rate": self.config.learning_rate,
            "compile": self.config.compile,
        }
        module_kwargs.update(self.config.loss_kwargs)
        module_kwargs.update(self.config.tracking_kwargs)
        return module_kwargs

    def _runtime(self) -> LightningTrainerRuntime:
        runtime = self.config.runtime_kwargs.get("lightning_runtime")
        if runtime is not None:
            return runtime
        return build_lightning_runtime(
            dry=self.config.dry,
            timestamp=self.config.runtime_kwargs.get("timestamp", True),
            name=self.config.name,
            outdir=self.config.outdir,
            resume=self.config.resume,
            logger_name=self.config.logger,
            wandb_project=self.config.runtime_kwargs.get("wandb_project", "trackastra"),
            profile=self.config.runtime_kwargs.get("profile", False),
            training_args=self.config.runtime_kwargs.get("training_args", {}),
        )

    def _trainer(self, runtime: LightningTrainerRuntime) -> Any:
        accelerator = self._accelerator()
        trainer_kwargs = {
            "accelerator": accelerator,
            "strategy": self._strategy(),
            "devices": self._devices(accelerator),
            "gradient_clip_val": 1.0,
            "precision": "bf16-mixed" if self.config.mixed_precision else 32,
            "logger": runtime.logger,
            "default_root_dir": runtime.logdir if not self.config.dry else None,
            "num_nodes": 1,
            "max_epochs": self.config.epochs,
            "callbacks": runtime.callbacks,
            "profiler": runtime.profiler,
        }
        trainer_kwargs.update(self.config.runtime_kwargs.get("trainer_kwargs", {}))
        return _lightning_trainer_class()(**trainer_kwargs)

    def fit(
        self,
        model: Any,
        train_dataset: TrackingDatasetBundle,
        val_dataset: TrackingDatasetBundle,
        *,
        ckpt_path: Path | str | None = None,
    ) -> TrainingRun:
        """Fit a model using domain-level dataset bundles."""
        if self.config.epochs == 0:
            return TrainingRun(
                trainer=None,
                lightning_module=None,
                result=None,
            )

        runtime = self._runtime()
        module_kwargs = self._lightning_module_kwargs(model)
        if module_kwargs.get("node_loss", 0) > 0:
            in_counts, out_counts = pooled_node_degree_counts(train_dataset)
            model_config = getattr(model, "config", {}) or {}
            n_in = model_config.get("max_in_degree", 1) + 1
            n_out = model_config.get("max_out_degree", 2) + 1
            module_kwargs["node_in_weights"] = node_degree_class_weights(
                in_counts, n_in
            )
            module_kwargs["node_out_weights"] = node_degree_class_weights(
                out_counts, n_out
            )
            logger.info(
                "Node in-degree class weights: %s (counts=%s)",
                module_kwargs["node_in_weights"].tolist(),
                in_counts.tolist(),
            )
            logger.info(
                "Node out-degree class weights: %s (counts=%s)",
                module_kwargs["node_out_weights"].tolist(),
                out_counts.tolist(),
            )
        lightning_module = build_or_resume_lightning_module(
            _tracking_lightning_module_class(),
            module_kwargs,
            logdir=runtime.logdir,
            resume=self.config.resume,
        )
        configure_lightning_module_runtime_paths(
            lightning_module,
            logdir=runtime.logdir,
            debug=self.config.runtime_kwargs.get("debug", False),
        )
        trainer = self._trainer(runtime)
        resume_path = (
            ckpt_path
            if ckpt_path is not None
            else resume_checkpoint_path(
                logdir=runtime.logdir, resume=self.config.resume
            )
        )
        train_loader = train_dataset.dataloader(distributed=self.config.distributed)
        _write_sampler_prob_debug(
            train_loader,
            None
            if runtime.logdir is None
            else Path(runtime.logdir) / "diagnostics" / "sampling_probs.csv",
        )
        model_config = getattr(model, "config", {}) or {}
        max_neighbors = model_config.get("max_neighbors", (16,))
        candidate_k = (
            int(max_neighbors)
            if isinstance(max_neighbors, int)
            else max(int(value) for value in max_neighbors)
        )
        _write_dataset_summary_debug(
            train_loader,
            None
            if runtime.logdir is None
            else Path(runtime.logdir) / "diagnostics" / "dataset_summary.csv",
            candidate_k=candidate_k,
            candidate_mode=(
                str(model_config.get("sparse_knn_mode", "global"))
                if model_config.get("attn_mode") == "sparse"
                else "dense"
            ),
            spatial_cutoff=float(model_config.get("spatial_cutoff", float("inf"))),
            delta_cutoff=int(module_kwargs.get("delta_cutoff", 1)),
        )
        result = trainer.fit(
            lightning_module,
            train_dataloaders=train_loader,
            val_dataloaders=val_dataset.dataloader(),
            ckpt_path=resume_path,
        )
        return TrainingRun(
            trainer=trainer,
            lightning_module=lightning_module,
            result=result,
        )


def build_trainer(train_config: TrainConfig):
    """Build the public training facade."""
    return TrackastraTrainer(_prepare_training_config(train_config))


def _prepare_training_config(train_config: TrainConfig) -> TrainConfig:
    """Apply process-wide training setup and return config with resolved seed."""
    import numpy as np
    import torch

    from trackastra.utils import seed

    logging.basicConfig(level=logging.INFO)
    warnings.simplefilter(action="ignore", category=FutureWarning)
    torch.set_float32_matmul_precision("medium")
    np.seterr(all="ignore")

    resolved_seed = seed(train_config.runtime_kwargs.get("seed"))
    runtime_kwargs = dict(train_config.runtime_kwargs)
    runtime_kwargs["seed"] = resolved_seed
    training_args = dict(runtime_kwargs.get("training_args", {}))
    training_args["seed"] = resolved_seed
    runtime_kwargs["training_args"] = training_args
    return replace(train_config, runtime_kwargs=runtime_kwargs)


def tracking_lightning_module_class():
    """Return the current internal Lightning module class."""
    return _tracking_lightning_module_class()


def _training_config_from_args(
    args: Any,
) -> tuple[model_api.ModelConfig, DataSplitConfig, DataSplitConfig, TrainConfig]:
    """Convert parsed training CLI args into public config objects."""
    if args.model is not None and args.resume:
        raise ValueError(
            "--model/-m and --resume cannot be used together. Use --model without "
            "--resume to initialize a new run. To resume, omit --model, set "
            "--timestamp f, set --name to the exact existing run-directory name "
            "under --outdir, keep the model architecture settings unchanged, and "
            "ensure <outdir>/<name>/checkpoints/last.ckpt exists."
        )
    if not 0 <= args.feature_drop <= 1:
        raise ValueError(f"feature_drop must be in [0, 1], got {args.feature_drop}")
    if args.feature_drop > 0 and args.features not in (
        "wrfeat2",
        "wrfeat2_no_intensity",
        "wrfeat3",
    ):
        raise ValueError(
            "feature_drop requires wrfeat2, wrfeat2_no_intensity, or wrfeat3"
        )
    model_config = model_api.ModelConfig(
        coord_dim=args.ndim,
        feat_dim=_feature_dim(args.ndim, args.features),
        d_model=args.d_model,
        pos_embed_per_dim=args.pos_embed_per_dim,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dropout=args.dropout,
        window=args.window,
        spatial_cutoff=args.spatial_cutoff,
        attn_positional_bias=args.attn_positional_bias,
        attn_positional_bias_n_spatial=args.attn_positional_bias_n_spatial,
        attn_dist_mode=args.attn_dist_mode,
        attn_mode=args.attn_mode,
        max_neighbors=tuple(args.max_neighbors),
        sparse_knn_mode=args.sparse_knn_mode,
        logit_norm=args.logit_norm,
        head_mode=args.head_mode,
        causal_norm=args.causal_norm,
        architecture_version=args.architecture_version,
        data_dim_embed=args.data_dim_embed,
        disable_abs_pos=args.disable_abs_pos,
        disable_input_norm=args.disable_input_norm,
        encoder_only=args.encoder_only,
        node_head=args.node_loss > 0,
        max_in_degree=args.max_in_degree,
        max_out_degree=args.max_out_degree,
        model_path=Path(args.model) if args.model is not None else None,
    )
    sequence_kwargs = {
        "detection_folders": args.detection_folders,
        "downscale_temporal": args.downscale_temporal,
        "downscale_spatial": args.downscale_spatial,
    }
    base_dataset_kwargs = {
        "window_size": args.window,
        "model_coord_dim": args.ndim,
        "max_detections": args.max_detections,
        "features": args.features,
        "detect_drop_fraction": args.detect_drop_fraction,
        "normalize_diameter": args.normalize_diameter,
    }
    train_dataset_kwargs = {
        **base_dataset_kwargs,
        "detect_drop": args.detect_drop,
        "augment": args.augment,
        "augment_details": args.augment_details,
        "position_noise": args.spatial_cutoff,
        # Same per-window drop probability for every feature group.
        "feature_group_drop": (
            {"intensity": args.feature_drop, "shape": args.feature_drop}
            if args.feature_drop > 0
            else None
        ),
    }
    val_dataset_kwargs = {
        **base_dataset_kwargs,
        "detect_drop": 0.0,
        "augment": 0,
        "augment_details": None,
        "position_noise": 0.0,
    }
    sampler_kwargs = {
        "batch_size": args.batch_size,
        "n_pool": args.n_pool_sampler,
        "num_samples": args.train_samples if args.train_samples > 0 else None,
        "oversample_divs": args.oversample_divs,
        "oversample_density": args.oversample_density,
        "weight_by_dataset": args.weight_by_dataset,
        "balance_batch_objects": args.balance_batch_objects,
        "balance_pct": args.balance_pct,
    }
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "persistent_workers": True if args.num_workers > 0 else False,
        "pin_memory": True,
        "collate_fn": collate_sequence_padding,
    }
    train_data_config = DataSplitConfig(
        split="train",
        sources=normalize_source_specs(
            args.input_train or (),
            sequence_kwargs=sequence_kwargs,
            split_sequence_kwargs={"slice_pct": args.slice_pct_train},
        ),
        sequence_kwargs={**sequence_kwargs, "slice_pct": args.slice_pct_train},
        dataset_kwargs=train_dataset_kwargs,
        sampler_kwargs=sampler_kwargs,
        loader_kwargs=loader_kwargs,
    )
    val_data_config = DataSplitConfig(
        split="val",
        sources=normalize_source_specs(
            args.input_val or (),
            sequence_kwargs=sequence_kwargs,
            split_sequence_kwargs={"slice_pct": args.slice_pct_val},
        ),
        sequence_kwargs={**sequence_kwargs, "slice_pct": args.slice_pct_val},
        dataset_kwargs=val_dataset_kwargs,
        loader_kwargs=loader_kwargs,
    )
    inference_config = _inference_config_from_args(args)
    delta_cutoff = args.delta_cutoff if args.delta_cutoff is not None else args.window
    training_args = dict(vars(args))
    training_args.update(
        {
            "warmup_epochs": min(args.warmup_epochs, args.epochs),
            "delta_cutoff": delta_cutoff,
        }
    )
    train_config = TrainConfig(
        epochs=args.epochs,
        warmup_epochs=min(args.warmup_epochs, args.epochs),
        learning_rate=args.lr,
        batch_size=args.batch_size,
        distributed=args.distributed,
        device=args.device,
        logger=args.logger,
        outdir=Path(args.outdir),
        name=args.name,
        resume=args.resume,
        dry=args.dry,
        mixed_precision=args.mixedp,
        compile=args.compile,
        cache_dir=Path(args.cachedir) if args.cache else None,
        loss_kwargs={
            "delta_cutoff": delta_cutoff,
            "causal_norm": args.causal_norm,
            "assoc_loss": args.assoc_loss,
            "loss_norm": args.loss_norm,
            "focal_loss_gamma": args.focal_loss_gamma,
            "div_upweight": args.div_upweight,
            "node_loss": args.node_loss,
            "consistency_weight": args.consistency_weight,
        },
        tracking_kwargs={
            "tracking_frequency": args.tracking_frequency,
            "tracking_inputs": tracking_inputs_from_sources(
                val_data_config.sources,
                tracking_frequency=args.tracking_frequency,
            ),
            "tracking_detection_folder": args.detection_folders[0],
            "tracking_mode": args.tracking_mode,
            "inference_config": inference_config,
            "batch_val_tb_idx": 0,
            "grad_log_every_n_epochs": args.grad_log_every_n_epochs,
        },
        runtime_kwargs={
            "debug": args.debug,
            "profile": args.profile,
            "seed": args.seed,
            "timestamp": args.timestamp,
            "wandb_project": args.wandb_project,
            "training_args": training_args,
        },
    )
    return model_config, train_data_config, val_data_config, train_config


def parse_training_config(parser: Any = None):
    """Parse the training CLI into public config objects."""
    from trackastra.training.config import parse_train_args

    return _training_config_from_args(parse_train_args(parser))


def create_train_parser():
    """Create the training CLI parser for user extension."""
    from trackastra.training.config import create_train_parser as _create_train_parser

    return _create_train_parser()


def parse_train_args(parser: Any = None):
    """Parse training CLI args from the default or a user-extended parser."""
    from trackastra.training.config import parse_train_args as _parse_train_args

    return _parse_train_args(parser)
