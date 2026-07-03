"""Data loading and sampling utils for distributed training."""

import logging
import math
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from timeit import default_timer
from typing import Any, Literal
from tqdm import tqdm
import joblib
import numpy as np
import torch
from lightning import LightningDataModule
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from torch.utils.data import (
    BatchSampler,
    ConcatDataset,
    DataLoader,
    Dataset,
    DistributedSampler,
)

from .dataset import (
    TrackingDataset,
    association_distances,
    warn_association_distances,
)
from .io import (
    TrackingSequence,
    _resolve_detection_folder,
    _resolve_paths,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass(frozen=True)
class SequenceInputSpec:
    path: Path
    format: Literal["auto", "ctc", "geff"] = "auto"
    spacing: tuple[float, ...] | Literal["auto"] | None = None
    sparse_gt: bool = False
    loader_kwargs: dict[str, Any] = field(default_factory=dict)


_SPEC_CORE_KEYS = {"path", "paths", "format", "spacing", "sparse_gt"}
_SequenceFormat = Literal["ctc", "geff"]


def _normalize_spacing(
    spacing: object,
    ndim: int | None,
    fmt: Literal["auto", "ctc", "geff"],
) -> tuple[float, ...] | Literal["auto"] | None:
    if spacing is None:
        return None
    if spacing == "auto":
        if fmt == "ctc":
            raise ValueError('spacing="auto" is only valid for GEFF inputs')
        return "auto"
    if isinstance(spacing, str):
        raise ValueError("spacing must be a numeric sequence, None, or 'auto'")
    try:
        values = tuple(float(v) for v in spacing)  # type: ignore[union-attr]
    except TypeError as exc:
        raise ValueError("spacing must be a numeric sequence, None, or 'auto'") from exc
    if ndim is not None and len(values) != ndim:
        raise ValueError(f"spacing must have length {ndim}, got {len(values)}")
    if not values or not all(np.isfinite(values)) or any(v <= 0 for v in values):
        raise ValueError("spacing values must be finite and positive")
    return values


def _sequence_input_spec_from_mapping(
    item: Mapping[str, Any],
    ndim: int | None,
) -> SequenceInputSpec:
    if "path" not in item:
        raise ValueError("sequence input mapping must contain 'path'")
    if "paths" in item:
        raise ValueError("sequence input mapping cannot contain both 'path' and 'paths'")
    fmt = item.get("format", "auto")
    if fmt not in ("auto", "ctc", "geff"):
        raise ValueError(f"format must be one of 'auto', 'ctc', or 'geff', got {fmt!r}")
    fmt = fmt  # type: ignore[assignment]
    if fmt == "geff" and "sparse_gt" not in item:
        raise ValueError("GEFF input specs must set sparse_gt explicitly")
    loader_kwargs = {
        key: deepcopy(value)
        for key, value in item.items()
        if key not in _SPEC_CORE_KEYS
    }
    if (
        fmt == "geff"
        and "detections" in loader_kwargs
        and "match_max_distance" not in loader_kwargs
    ):
        raise ValueError("GEFF inputs with external detections require match_max_distance")
    return SequenceInputSpec(
        path=Path(item["path"]),
        format=fmt,
        spacing=_normalize_spacing(item.get("spacing"), ndim, fmt),
        sparse_gt=bool(item.get("sparse_gt", False)),
        loader_kwargs=loader_kwargs,
    )


def normalize_sequence_input_specs(
    inputs: Sequence[str | Path | Mapping[str, Any]],
    ndim: int | None = None,
) -> tuple[SequenceInputSpec, ...]:
    specs = []
    for item in inputs:
        if isinstance(item, str | Path):
            specs.append(SequenceInputSpec(path=Path(item), format="ctc"))
            continue
        if not isinstance(item, Mapping):
            raise ValueError(f"Unsupported sequence input item {item!r}")
        if ("path" in item) == ("paths" in item):
            raise ValueError("sequence input mapping must contain exactly one of path/paths")
        if "path" in item:
            specs.append(_sequence_input_spec_from_mapping(item, ndim))
            continue

        group = {key: value for key, value in item.items() if key != "paths"}
        paths = item["paths"]
        if isinstance(paths, str | Path) or not isinstance(paths, Sequence):
            raise ValueError("paths must be a sequence of paths or path mappings")
        for path_item in paths:
            if isinstance(path_item, str | Path):
                merged = {**group, "path": path_item}
            elif isinstance(path_item, Mapping):
                if "path" not in path_item or "paths" in path_item:
                    raise ValueError("path mappings inside paths must contain exactly path")
                merged = {**group, **path_item}
            else:
                raise ValueError(f"Unsupported path item {path_item!r}")
            specs.append(_sequence_input_spec_from_mapping(merged, ndim))
    return tuple(specs)


def _looks_like_geff(path: Path) -> tuple[bool, str | None]:
    path = path.expanduser()
    if path.suffix == ".geff":
        return True, None
    if not path.is_dir():
        return False, f"{path} is not a GEFF store or directory"
    stores = sorted(path.glob("*.geff"))
    if len(stores) != 1:
        return False, f"Expected exactly one .geff store in {path}, found {len(stores)}"
    return True, None


def _looks_like_ctc(path: Path, kwargs: Mapping[str, Any]) -> tuple[bool, str | None]:
    try:
        root, _, _, _ = _resolve_paths(
            path,
            kwargs.get("image_folder"),
            kwargs.get("gt_folder"),
            kwargs.get("track_file"),
            bool(kwargs.get("use_gt", True)),
        )
        for folder in kwargs.get("detection_folders", ("TRA",)):
            _resolve_detection_folder(root, folder)
    except (FileNotFoundError, ValueError) as exc:
        return False, str(exc)
    return True, None


class BalancedBatchSampler(BatchSampler):
    """samples batch indices such that the number of objects in each batch is balanced
    (so to reduce the number of paddings in the batch).


    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int,
        n_pool: int = 10,
        num_samples: int | None = None,
        weight_by_ndivs: bool = False,
        weight_by_dataset: bool = False,
        drop_last: bool = False,
        balance_batch_objects: bool = False,
        balance_pct: float = 50.0,
    ):
        """Setting n_pool =1 will result in a regular random batch sampler.

        weight_by_ndivs: if True, the probability of sampling an element is proportional to the number of divisions
        weight_by_dataset: if True, the probability of sampling an element is inversely proportional to the length of the dataset
        balance_batch_objects: if True, use a variable batch size so that the total
            number of detections per batch is held roughly constant (``batch_size``
            becomes an upper cap). Dense windows get smaller batches, sparse ones
            larger -- equalizing GPU memory/compute across differently sized data.
        balance_pct: percentile of ``n_objects`` used as the reference detection
            count ``n_ref``; the per-batch budget is ``batch_size * n_ref``. The
            median (50) anchors a full ``batch_size`` batch at the typical window.
        """
        if isinstance(dataset, TrackingDataset):
            self.n_objects = dataset.n_objects
            self.n_divs = np.array(dataset.n_divs)
            self.n_sizes = np.ones(len(dataset)) * len(dataset)
        elif isinstance(dataset, ConcatDataset):
            self.n_objects = tuple(n for d in dataset.datasets for n in d.n_objects)
            self.n_divs = np.array(tuple(n for d in dataset.datasets for n in d.n_divs))
            self.n_sizes = np.array(
                tuple(len(d) for d in dataset.datasets for _ in range(len(d)))
            )
        else:
            raise NotImplementedError(
                f"BalancedBatchSampler: Unknown dataset type {type(dataset)}"
            )
        assert len(self.n_objects) == len(self.n_divs) == len(self.n_sizes)

        self.batch_size = batch_size
        self.n_pool = n_pool
        self.drop_last = drop_last
        self.num_samples = num_samples
        self.weight_by_ndivs = weight_by_ndivs
        self.weight_by_dataset = weight_by_dataset
        self.balance_batch_objects = balance_batch_objects
        self.balance_pct = balance_pct
        logger.debug(f"{weight_by_ndivs=}")
        logger.debug(f"{weight_by_dataset=}")

        # Budget on the total number of detections per (padded) batch. Since
        # n_objects is already capped at max_detections, this reflects the
        # post-cap GPU-side size. batch_size stays the upper cap on item count.
        if balance_batch_objects and len(self.n_objects) > 0:
            n_ref = float(np.percentile(np.asarray(self.n_objects), balance_pct))
            self.object_budget = max(1.0, batch_size * n_ref)
            logger.info(
                f"BalancedBatchSampler: variable batch size, budget"
                f" {self.object_budget:.0f} detections/batch (n_ref={n_ref:.0f}"
                f" @ p{balance_pct:g}, cap {batch_size})"
            )
        else:
            self.object_budget = None

    def get_probs(self, idx):
        idx = np.array(idx)
        if self.weight_by_ndivs:
            probs = 1 + np.sqrt(self.n_divs[idx])
        else:
            probs = np.ones(len(idx))
        if self.weight_by_dataset:
            probs = probs / (self.n_sizes[idx] + 1e-6)

        probs = probs / (probs.sum() + 1e-10)
        return probs

    def _pack_sorted(self, idx_pool):
        """Greedily split an N-sorted pool into variable-size batches.

        Keeps adding items while ``len * max_n <= object_budget`` and
        ``len <= batch_size``. Because idx_pool is sorted ascending by
        n_objects, the last added item is always the batch max. Always
        emits at least one item so a single oversized window still progresses.
        """
        batches = []
        start, n = 0, len(idx_pool)
        while start < n:
            end = start + 1
            while end < n:
                new_count = end - start + 1
                n_max = self.n_objects[idx_pool[end]]
                if (
                    new_count > self.batch_size
                    or new_count * n_max > self.object_budget
                ):
                    break
                end += 1
            batches.append(idx_pool[start:end])
            start = end
        return batches

    def sample_batches(self, idx: Iterable[int]):
        # we will split the indices into pools of size n_pool
        idx = np.asarray(tuple(idx), dtype=int)
        num_samples = self.num_samples if self.num_samples is not None else len(idx)
        if num_samples <= 0 or len(idx) == 0:
            return []
        # sample from the indices with replacement and given probabilites
        idx = np.random.choice(idx, num_samples, replace=True, p=self.get_probs(idx))

        n_pool = max(self.batch_size, self.n_pool * self.batch_size)

        batches = []
        for i in range(0, len(idx), n_pool):
            # the indices in the pool are sorted by their number of objects
            idx_pool = idx[i : i + n_pool]
            idx_pool = sorted(idx_pool, key=lambda i: self.n_objects[i])

            if self.object_budget is not None:
                # variable batch size at constant detection budget; dense batches
                # are intentionally < batch_size so drop_last does not apply here
                pool_batches = self._pack_sorted(idx_pool)
                np.random.shuffle(pool_batches)
                batches.extend(pool_batches)
                continue

            # such that we can create batches where each element has a similar number of objects
            jj = np.arange(0, len(idx_pool), self.batch_size)
            np.random.shuffle(jj)

            for j in jj:
                batch = idx_pool[j : j + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                batches.append(batch)
        return batches

    def __iter__(self):
        idx = np.arange(len(self.n_objects))
        # Yield exactly len(self) batches so Lightning's num_training_batches
        # (== len of this sampler) matches the realized epoch length. With the
        # variable batch size, sample_batches() draws num_samples windows with
        # replacement and the batch count fluctuates below the __len__ estimate;
        # if the loader stops short, the epoch-end validation modulo
        # ((batch_idx + 1) % num_training_batches == 0) never fires and the
        # whole validation loop is silently skipped every epoch.
        target = len(self)
        batches = self.sample_batches(idx)
        while len(batches) < target:
            batches.extend(self.sample_batches(idx))
        return iter(batches[:target])

    def _estimate_num_batches(self) -> int:
        """Estimate epoch length under variable batch sizes.

        Simulates the greedy packing on the sorted population, scales to
        num_samples, and adds one partial batch per pool. Biased to slightly
        over-count so the Lightning epoch loop is bounded by the dataloader's
        StopIteration rather than truncated by an undercounted length.
        """
        n_sorted = np.sort(np.asarray(self.n_objects))
        N = len(n_sorted)
        cnt, start = 0, 0
        while start < N:
            end = start + 1
            while end < N:
                new_count = end - start + 1
                if (
                    new_count > self.batch_size
                    or new_count * n_sorted[end] > self.object_budget
                ):
                    break
                end += 1
            cnt += 1
            start = end
        num = self.num_samples if self.num_samples is not None else N
        n_pool = max(self.batch_size, self.n_pool * self.batch_size)
        return math.ceil(cnt * num / max(N, 1)) + math.ceil(num / n_pool)

    def __len__(self):
        n = self.num_samples if self.num_samples is not None else len(self.n_objects)
        if self.object_budget is not None:
            return self._estimate_num_batches()
        if self.drop_last:
            return n // self.batch_size
        return math.ceil(n / self.batch_size)


class BalancedDistributedSampler(DistributedSampler):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        n_pool: int,
        num_samples: int | None,
        weight_by_ndivs: bool = False,
        weight_by_dataset: bool = False,
        balance_batch_objects: bool = False,
        balance_pct: float = 50.0,
        *args,
        **kwargs,
    ) -> None:
        requested_num_samples = num_samples
        # __iter__ flattens batches into a per-index stream that the DataLoader
        # re-batches at fixed batch_size, so variable batch sizes cannot survive
        # DDP. Ignore the request here to keep ranks' step counts matched.
        if balance_batch_objects:
            logger.warning(
                "balance_batch_objects is ignored under distributed (DDP) training;"
                " batch sizes stay fixed at batch_size."
            )
        super().__init__(dataset=dataset, *args, drop_last=True, **kwargs)
        per_rank_samples = (
            self.num_samples
            if requested_num_samples is None
            else max(1, math.ceil(requested_num_samples / self.num_replicas))
        )
        self._balanced_batch_sampler = BalancedBatchSampler(
            dataset,
            batch_size=batch_size,
            n_pool=n_pool,
            num_samples=per_rank_samples,
            weight_by_ndivs=weight_by_ndivs,
            weight_by_dataset=weight_by_dataset,
        )

    def __len__(self) -> int:
        if self.num_samples is not None:
            return self._balanced_batch_sampler.num_samples
        else:
            return super().__len__()

    def __iter__(self):
        indices = list(super().__iter__())
        batches = self._balanced_batch_sampler.sample_batches(indices)
        for batch in batches:
            yield from batch


class BalancedDataModule(LightningDataModule):
    def __init__(
        self,
        input_train: list,
        input_val: list,
        cachedir: str,
        distributed: bool,
        sequence_kwargs: dict,
        tracking_data_kwargs: dict,
        sampler_kwargs: dict,
        loader_kwargs: dict,
        train_sequence_kwargs: dict | None = None,
        val_sequence_kwargs: dict | None = None,
        train_tracking_data_kwargs: dict | None = None,
        val_tracking_data_kwargs: dict | None = None,
        association_distance_cutoffs: dict[str, float] | None = None,
        association_delta_cutoff: int = 1,
    ):
        super().__init__()
        ndim = sequence_kwargs.get("ndim")
        self.input_train = normalize_sequence_input_specs(input_train, ndim=ndim)
        self.input_val = normalize_sequence_input_specs(input_val, ndim=ndim)
        self.cachedir = cachedir
        self.distributed = distributed
        self.sequence_kwargs = sequence_kwargs
        self.tracking_data_kwargs = tracking_data_kwargs
        self.train_sequence_kwargs = train_sequence_kwargs or {}
        self.val_sequence_kwargs = val_sequence_kwargs or {}
        self.train_tracking_data_kwargs = train_tracking_data_kwargs or {}
        self.val_tracking_data_kwargs = val_tracking_data_kwargs or {}
        self.sampler_kwargs = sampler_kwargs
        self.loader_kwargs = loader_kwargs
        self.association_distance_cutoffs = association_distance_cutoffs or {}
        self.association_delta_cutoff = association_delta_cutoff

    @rank_zero_only
    def _warn_association_distances(self, dataset, split: str) -> None:
        if not self.association_distance_cutoffs:
            return
        distances = association_distances(
            dataset, delta_cutoff=self.association_delta_cutoff
        )
        for cutoff_name, max_distance in self.association_distance_cutoffs.items():
            if max_distance is None:
                continue
            warn_association_distances(
                distances,
                max_distance=max_distance,
                delta_cutoff=self.association_delta_cutoff,
                cutoff_name=cutoff_name,
                dataset_name=f"{split} dataset {dataset.root}",
            )

    def _sequence_kwargs_for_split(self, split: str) -> dict:
        kwargs = self.sequence_kwargs.copy()
        kwargs.update(
            self.train_sequence_kwargs if split == "train" else self.val_sequence_kwargs
        )
        return kwargs

    def _tracking_data_kwargs_for_split(self, split: str) -> dict:
        kwargs = self.tracking_data_kwargs.copy()
        kwargs.update(
            self.train_tracking_data_kwargs
            if split == "train"
            else self.val_tracking_data_kwargs
        )
        return kwargs

    def _sequence_loader(self, fmt: _SequenceFormat):
        if fmt == "ctc":
            loader = TrackingSequence.from_ctc
        elif fmt == "geff":
            loader = TrackingSequence.from_geff
        else:
            raise ValueError(f"Unknown sequence format: {fmt!r}")
        if self.cachedir is None:
            return loader
        memory = joblib.Memory(self.cachedir, verbose=0)
        if fmt == "ctc":
            return memory.cache(loader, ignore=["n_workers"])
        return memory.cache(loader)

    def _ctc_loader_kwargs(self, inp: SequenceInputSpec, split: str) -> dict:
        kwargs = dict(root=inp.path, **self._sequence_kwargs_for_split(split))
        if inp.spacing is not None:
            if inp.spacing == "auto":
                raise ValueError('spacing="auto" is only valid for GEFF inputs')
            kwargs["spacing"] = inp.spacing
        kwargs.update(inp.loader_kwargs)
        return kwargs

    def _geff_loader_kwargs(self, inp: SequenceInputSpec) -> dict:
        kwargs = dict(
            root_or_geff=inp.path,
            sparse_gt=inp.sparse_gt,
            **inp.loader_kwargs,
        )
        if inp.spacing is not None:
            kwargs["spacing"] = inp.spacing
        return kwargs

    def _resolve_input_format(
        self,
        inp: SequenceInputSpec,
        split: str,
    ) -> _SequenceFormat:
        if inp.format != "auto":
            return inp.format
        ctc_kwargs = self._sequence_kwargs_for_split(split)
        ctc_kwargs.update(inp.loader_kwargs)
        geff_ok, geff_error = _looks_like_geff(inp.path)
        ctc_ok, ctc_error = _looks_like_ctc(inp.path, ctc_kwargs)
        if geff_ok and ctc_ok:
            raise ValueError(
                f"Could not auto-detect unique format for {inp.path}: "
                "both CTC and GEFF layouts are valid"
            )
        if geff_ok:
            return "geff"
        if ctc_ok:
            return "ctc"
        raise ValueError(
            f"Could not auto-detect sequence format for {inp.path}. "
            f"CTC check failed: {ctc_error}. GEFF check failed: {geff_error}."
        )

    def _load_sequence(self, inp: SequenceInputSpec, split: str):
        """Load one sequence, logging whether it was served from the joblib cache."""
        fmt = self._resolve_input_format(inp, split)
        loader = self._sequence_loader(fmt)
        kwargs = (
            self._ctc_loader_kwargs(inp, split)
            if fmt == "ctc"
            else self._geff_loader_kwargs(inp)
        )
        if self.cachedir is not None:
            hit = loader.check_call_in_cache(**kwargs)
            logger.info(
                f"  {split.upper()} {inp.path} ({fmt}): "
                + ("loaded from cache" if hit else "cache miss, computing")
            )
        else: 
            logger.info(f"  {split.upper()} {inp.path} ({fmt}): loading from disk")
        return loader(**kwargs)

    def prepare_data(self):
        """Loads and caches the datasets if not already done.

        Running on the main CPU process.
        """
        if self.cachedir is None:
            return
        for split, inps in zip(
            ("train", "val"),
            (self.input_train, self.input_val),
        ):
            logger.info(f"Loading {split.upper()} data")
            start = default_timer()
            sequences = tuple(self._load_sequence(inp, split) for inp in tqdm(inps, desc=f"Loading {split.upper()} data"))
            logger.info(
                f"Loaded {len(sequences)} {split.upper()} sequences (in"
                f" {(default_timer() - start):.1f} s)\n\n"
            )

    def setup(self, stage: str):
        self.datasets = dict()
        for split, inps in zip(
            ("train", "val"),
            (self.input_train, self.input_val),
        ):
            logger.info(f"Loading {split.upper()} data")
            start = default_timer()
            self.datasets[split] = torch.utils.data.ConcatDataset(
                TrackingDataset(
                    self._load_sequence(inp, split),
                    dataset_index=i,
                    **self._tracking_data_kwargs_for_split(split),
                )
                for i, inp in enumerate(inps)
            )
            for dataset in self.datasets[split].datasets:
                self._warn_association_distances(dataset, split)
            logger.info(
                f"Loaded {len(self.datasets[split])} {split.upper()} samples (in"
                f" {(default_timer() - start):.1f} s)\n\n"
            )

    def train_dataloader(self):
        loader_kwargs = self.loader_kwargs.copy()
        if self.distributed:
            sampler = BalancedDistributedSampler(
                self.datasets["train"],
                **self.sampler_kwargs,
            )
            batch_sampler = None
        else:
            sampler = None
            batch_sampler = BalancedBatchSampler(
                self.datasets["train"],
                **self.sampler_kwargs,
            )
            if not loader_kwargs["batch_size"] == batch_sampler.batch_size:
                raise ValueError(
                    f"Batch size in loader_kwargs ({loader_kwargs['batch_size']}) and sampler_kwargs ({batch_sampler.batch_size}) must match"
                )
            del loader_kwargs["batch_size"]

        loader = DataLoader(
            self.datasets["train"],
            sampler=sampler,
            batch_sampler=batch_sampler,
            **loader_kwargs,
        )
        return loader

    def val_dataloader(self):
        val_loader_kwargs = deepcopy(self.loader_kwargs)
        val_loader_kwargs["persistent_workers"] = False
        num_workers = val_loader_kwargs["num_workers"]
        val_loader_kwargs["num_workers"] = (
            0 if num_workers == 0 else max(1, num_workers // 2)
        )
        return DataLoader(
            self.datasets["val"],
            shuffle=False,
            **val_loader_kwargs,
        )
