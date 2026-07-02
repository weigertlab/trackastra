"""Runtime temporal-window dataset and collation over tracking sequences (torch).

Depends on the torch-free data model in ``io.py`` (one-way ``dataset`` -> ``io``). A
temporal window is a ``timepoints``-range slice of a flat :class:`DetectionSet`.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Literal

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from torch.utils.data import Dataset

from trackastra.data import wrfeat
from trackastra.data.io import DetectionSet, TrackingSequence
from trackastra.utils import blockwise_sum

logger = logging.getLogger(__name__)

FeatureMode = Literal["none", "intensity", "wrfeat", "wrfeat2", "wrfeat2_no_intensity"]
_FEATURE_MODES = tuple(wrfeat.FEATURE_RECIPES)


def _feature_properties_for_sequence(sequence: TrackingSequence) -> set[str]:
    feature_sets = [set(seg.features) for seg in sequence.detections]
    if not feature_sets:
        return set()
    return set.intersection(*feature_sets)


def _compatible_feature_modes(available: set[str]) -> tuple[str, ...]:
    return tuple(
        mode
        for mode in _FEATURE_MODES
        if set(wrfeat.feature_recipe_keys(mode)).issubset(available)
    )


def _validate_feature_mode(sequence: TrackingSequence, mode: FeatureMode) -> None:
    available = _feature_properties_for_sequence(sequence)
    required = set(wrfeat.feature_recipe_keys(mode))
    if required.issubset(available):
        return
    raise ValueError(
        f"Feature mode {mode!r} requires feature properties {sorted(required)}, "
        f"but the tracking sequence only has {sorted(available)}. "
        f"Compatible feature modes: {list(_compatible_feature_modes(available))}"
    )


def _association_target(
    lineage_index: np.ndarray, lineage_relation: np.ndarray
) -> np.ndarray:
    target = np.zeros((len(lineage_index), len(lineage_index)), dtype=bool)
    valid = np.flatnonzero(lineage_index >= 0)
    if len(valid):
        target[np.ix_(valid, valid)] = lineage_relation[
            np.ix_(lineage_index[valid], lineage_index[valid])
        ]
    return target


def association_supervision_mask(
    timepoints: torch.Tensor,
    gt_predecessor_set_available: torch.Tensor,
    gt_successor_set_available: torch.Tensor,
    delta_cutoff: int | None = None,
    padding_mask: torch.Tensor | None = None,
) -> torch.BoolTensor:
    """Return association pairs with annotated GT link-set supervision."""
    if delta_cutoff is not None and delta_cutoff < 1:
        raise ValueError(f"delta_cutoff must be positive, got {delta_cutoff}")
    batched = timepoints.ndim == 2
    if timepoints.ndim == 1:
        timepoints = timepoints.unsqueeze(0)
        gt_predecessor_set_available = gt_predecessor_set_available.unsqueeze(0)
        gt_successor_set_available = gt_successor_set_available.unsqueeze(0)
        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(0)
    elif timepoints.ndim != 2:
        raise ValueError(f"timepoints must be 1D or 2D, got shape {timepoints.shape}")
    if gt_predecessor_set_available.shape != timepoints.shape:
        raise ValueError("gt_predecessor_set_available must match timepoints shape")
    if gt_successor_set_available.shape != timepoints.shape:
        raise ValueError("gt_successor_set_available must match timepoints shape")

    dt = timepoints.unsqueeze(1) - timepoints.unsqueeze(2)
    pair_gt_available = gt_successor_set_available.bool().unsqueeze(
        2
    ) | gt_predecessor_set_available.bool().unsqueeze(1)
    mask_time = dt > 0
    if delta_cutoff is not None:
        mask_time = mask_time & (dt <= delta_cutoff)
    mask = mask_time & pair_gt_available
    if padding_mask is not None:
        if padding_mask.shape != timepoints.shape:
            raise ValueError("padding_mask must match timepoints shape")
        pair_padding = padding_mask.bool().unsqueeze(1) | padding_mask.bool().unsqueeze(2)
        mask = mask & ~pair_padding
    return mask if batched else mask.squeeze(0)


def _sample_neighborhood_indices(
    coords: np.ndarray,
    timepoints: np.ndarray,
    association: np.ndarray,
    max_detections: int,
) -> np.ndarray:
    last = np.flatnonzero(timepoints == timepoints.max())
    if len(last) <= max_detections:
        return np.arange(len(coords))
    anchor = last[np.random.randint(len(last))]
    distances = np.linalg.norm(coords[last] - coords[anchor], axis=1)
    seeds = last[np.argsort(distances, kind="stable")[:max_detections]]
    _, component = connected_components(
        csr_matrix(np.logical_or(association, association.T)), directed=False
    )
    return np.flatnonzero(np.isin(component, np.unique(component[seeds])))


def _sample_detection_keep_indices(
    association: np.ndarray, timepoints: np.ndarray, drop_fraction: float
) -> np.ndarray:
    if not len(association):
        return np.arange(0)
    first = np.flatnonzero(timepoints == timepoints.min())
    n_drop = int(np.floor(drop_fraction * len(first)))
    if n_drop == 0:
        return np.arange(len(association))
    n_components, component = connected_components(
        csr_matrix(np.logical_or(association, association.T)), directed=False
    )
    drop_seeds = np.random.choice(first, size=n_drop, replace=False)
    drop_components = np.unique(component[drop_seeds])
    if len(drop_components) == n_components:
        retained = np.random.choice(drop_components)
        drop_components = drop_components[drop_components != retained]
    return np.flatnonzero(~np.isin(component, drop_components))


def _subset_features(feature: wrfeat.WRFeatures, keep: np.ndarray) -> wrfeat.WRFeatures:
    return wrfeat.WRFeatures(
        coords=feature.coords[keep],
        labels=feature.labels[keep],
        timepoints=feature.timepoints[keep],
        features=OrderedDict(
            (name, values[keep]) for name, values in feature.features.items()
        ),
    )


def _normalize_diameter_factor(
    sequence: TrackingSequence, normalize_diameter: float | None
) -> float:
    features = tuple(
        wrfeat.WRFeatures(
            coords=seg.coords,
            labels=seg.labels,
            timepoints=seg.timepoints.astype(np.int32, copy=False),
            features=OrderedDict(seg.features),
        )
        for seg in sequence.detections
    )
    return wrfeat.normalize_diameter_factor(features, normalize_diameter)


def _wr_augmenter(level: int):
    common = [
        wrfeat.WRRandomFlip(p=0.5),
        wrfeat.WRRandomAffine(p=0.8, degrees=180, scale=(2 / 3, 1.5), shear=(0.1, 0.1)),
    ]
    if level == 1:
        augmentations = common
    elif level == 2:
        augmentations = [
            *common,
            wrfeat.WRRandomBrightness(p=0.8),
            wrfeat.WRRandomOffset(p=0.8, offset=(-3, 3)),
        ]
    elif level in (3, 4):
        if level == 4:
            common.append(
                wrfeat.WRRandomShapeJitter(p=0.8, scale=(0.9, 1.1), shear=0.05)
            )
        augmentations = [
            *common,
            wrfeat.WRRandomBrightness(p=0.8),
            wrfeat.WRRandomMovement(offset=(-10, 10), p=0.3),
            wrfeat.WRRandomOffset(p=0.8, offset=(-3, 3)),
        ]
    else:
        return None
    return wrfeat.WRAugmentationPipeline(augmentations)


def _division_count(association: np.ndarray, timepoints: np.ndarray) -> int:
    if not len(association):
        return 0
    block_sums = blockwise_sum(
        torch.from_numpy(association).float(), torch.from_numpy(timepoints).long()
    )
    return int((block_sums.max(dim=0)[0] == 2).sum().item())


class TrackingDataset(Dataset):
    """Runtime temporal windows over an immutable tracking sequence."""

    def __init__(
        self,
        sequence: TrackingSequence,
        window_size: int = 4,
        features: FeatureMode = "wrfeat2",
        augment: int = 0,
        position_noise: float = 0.0,
        max_detections: int | None = None,
        detect_drop: float = 0.0,
        detect_drop_fraction: float = 0.1,
        normalize_diameter: float | None = None,
        dataset_index: int = 0,
    ) -> None:
        if window_size <= 1:
            raise ValueError("window_size must be greater than one")
        if features not in _FEATURE_MODES:
            raise ValueError(f"Unsupported feature mode {features!r}")
        _validate_feature_mode(sequence, features)
        if augment not in range(5):
            raise ValueError("augment must be between 0 and 4")
        if position_noise < 0:
            raise ValueError("position_noise must be non-negative")
        if max_detections is not None and max_detections < window_size:
            raise ValueError("max_detections must be at least window_size")
        if not 0 <= detect_drop <= 1 or not 0 <= detect_drop_fraction <= 1:
            raise ValueError("Detection dropout values must be in [0, 1]")
        self.sequence = sequence
        self.root = sequence.root
        # Position of this dataset within the concatenated training set, so a
        # collated batch can be traced back to its source folder (self.root).
        self.dataset_index = dataset_index
        self.window_size = window_size
        self.features = features
        self.ndim = sequence.ndim
        self.augment = augment
        self.position_noise = position_noise
        self.max_detections = max_detections
        self.detect_drop = detect_drop
        self.detect_drop_fraction = detect_drop_fraction
        self.scale_factor = _normalize_diameter_factor(sequence, normalize_diameter)
        if normalize_diameter is not None:
            logger.info(
                "Normalizing %s: scale factor %.4g (target diameter %.4g)",
                getattr(sequence, "root", "<sequence>"),
                self.scale_factor,
                normalize_diameter,
            )
        self.augmenter = _wr_augmenter(augment)
        self.windows = tuple(
            (seg_index, start)
            for seg_index, seg in enumerate(sequence.detections)
            for start in range(seg.n_frames - window_size + 1)
        )
        self.n_objects = tuple(
            self._window_object_count(index) for index in range(len(self))
        )
        self.n_divs = tuple(
            self._window_division_count(index) for index in range(len(self))
        )
        self.feat_dim = self._infer_feature_dim()

    def __len__(self) -> int:
        return len(self.windows)

    def _window_slice(self, index: int) -> tuple[DetectionSet, slice]:
        seg_index, start = self.windows[index]
        seg = self.sequence.detections[seg_index]
        lo = int(np.searchsorted(seg.timepoints, start, side="left"))
        hi = int(np.searchsorted(seg.timepoints, start + self.window_size, side="left"))
        return seg, slice(lo, hi)

    def _window_arrays(self, index: int):
        # Slices of the stored arrays are read-only views; return writable copies so
        # downstream WRFeatures augmentation can mutate in place (matching the former
        # _concat_frames, which produced fresh np.concatenate buffers per window).
        seg, sl = self._window_slice(index)
        coords = seg.coords[sl].copy()
        labels = seg.labels[sl].copy()
        timepoints = seg.timepoints[sl].copy()
        lineage_index = seg.lineage_index[sl]
        features = OrderedDict(
            (name, values[sl].copy()) for name, values in seg.features.items()
        )
        association = _association_target(lineage_index, self.sequence.lineage_relation)
        return coords, labels, timepoints, features, association

    def _window_matched_gt(self, index: int) -> np.ndarray:
        seg, sl = self._window_slice(index)
        if seg.matched_gt is None:
            return np.ones(sl.stop - sl.start, dtype=bool)
        return seg.matched_gt[sl].copy()

    def _window_gt_link_set_availability(
        self, index: int
    ) -> tuple[np.ndarray, np.ndarray]:
        seg, sl = self._window_slice(index)
        n = sl.stop - sl.start
        predecessor = (
            np.ones(n, dtype=bool)
            if seg.gt_predecessor_set_available is None
            else seg.gt_predecessor_set_available[sl].copy()
        )
        successor = (
            np.ones(n, dtype=bool)
            if seg.gt_successor_set_available is None
            else seg.gt_successor_set_available[sl].copy()
        )
        return predecessor, successor

    def _window_object_count(self, index: int) -> int:
        _, sl = self._window_slice(index)
        count = sl.stop - sl.start
        return min(count, self.max_detections) if self.max_detections else count

    def _window_division_count(self, index: int) -> int:
        _, _, timepoints, _, association = self._window_arrays(index)
        return _division_count(association, timepoints)

    def _infer_feature_dim(self) -> int:
        for seg in self.sequence.detections:
            feature = wrfeat.WRFeatures(
                coords=seg.coords,
                labels=seg.labels,
                timepoints=seg.timepoints.astype(np.int32, copy=False),
                features=OrderedDict(seg.features),
            )
            return feature.features_stacked_for(self.features).shape[1]
        return 0

    def __getitem__(
        self,
        index: int,
        return_all: bool = False,
    ) -> dict[str, torch.Tensor]:
        seg_index, window_start = self.windows[index]
        coords, labels, timepoints, features, association = self._window_arrays(index)
        matched_gt = self._window_matched_gt(index)
        gt_predecessor_set_available, gt_successor_set_available = (
            self._window_gt_link_set_availability(index)
        )
        feature = wrfeat.WRFeatures(
            coords=coords,
            labels=labels,
            timepoints=timepoints.astype(np.int32, copy=False),
            features=features,
        )
        feature = wrfeat.scale_feature_geometry(feature, self.scale_factor)
        if self.max_detections is not None:
            keep = _sample_neighborhood_indices(
                feature.coords,
                feature.timepoints,
                association,
                self.max_detections,
            )
            if len(keep) < len(feature):
                feature = _subset_features(feature, keep)
                association = association[np.ix_(keep, keep)]
                matched_gt = matched_gt[keep]
                gt_predecessor_set_available = gt_predecessor_set_available[keep]
                gt_successor_set_available = gt_successor_set_available[keep]

        if self.detect_drop and np.random.rand() < self.detect_drop:
            keep = _sample_detection_keep_indices(
                association, feature.timepoints, self.detect_drop_fraction
            )
            if len(keep) < len(feature):
                feature = _subset_features(feature, keep)
                association = association[np.ix_(keep, keep)]
                matched_gt = matched_gt[keep]
                gt_predecessor_set_available = gt_predecessor_set_available[keep]
                gt_successor_set_available = gt_successor_set_available[keep]

        if self.augmenter is not None:
            feature = self.augmenter(feature)
        feature_values = feature.features_stacked_for(self.features)
        coords0 = torch.from_numpy(
            np.concatenate((feature.timepoints[:, None], feature.coords), axis=1)
        ).float()
        coords = coords0.clone()
        if self.augmenter is not None and self.position_noise:
            coords[:, 1:] += torch.empty((1, self.ndim)).uniform_(
                -self.position_noise, self.position_noise
            )
        result = {
            "features": torch.from_numpy(feature_values).float(),
            "coords0": coords0,
            "coords": coords,
            "assoc_matrix": torch.from_numpy(association.astype(np.float32)),
            "timepoints": torch.from_numpy(feature.timepoints).long(),
            "labels": torch.from_numpy(feature.labels).long(),
            "matched_gt": torch.from_numpy(matched_gt).bool(),
            "gt_predecessor_set_available": torch.from_numpy(
                gt_predecessor_set_available
            ).bool(),
            "gt_successor_set_available": torch.from_numpy(
                gt_successor_set_available
            ).bool(),
            "window_index": torch.tensor(index, dtype=torch.long),
            "seg_index": torch.tensor(seg_index, dtype=torch.long),
            "window_start": torch.tensor(window_start, dtype=torch.long),
            "dataset_index": torch.tensor(self.dataset_index, dtype=torch.long),
        }
        if return_all:
            result["supervision_mask"] = association_supervision_mask(
                result["timepoints"],
                result["gt_predecessor_set_available"],
                result["gt_successor_set_available"],
            )
        return result


def association_distances(dataset: TrackingDataset, delta_cutoff: int) -> np.ndarray:
    """Distances of unique positive forward associations in raw windows."""
    if delta_cutoff < 1:
        raise ValueError("delta_cutoff must be positive")
    distances = {}
    for window, (seg_index, _) in enumerate(dataset.windows):
        coords, labels, timepoints, _, association = dataset._window_arrays(window)
        rows, cols = np.nonzero(association)
        delta = timepoints[cols] - timepoints[rows]
        valid = (delta > 0) & (delta <= delta_cutoff)
        rows, cols = rows[valid], cols[valid]
        edge_distances = np.linalg.norm(coords[cols] - coords[rows], axis=1)
        for source, target, distance in zip(rows, cols, edge_distances):
            key = (
                seg_index,
                int(timepoints[source]),
                int(labels[source]),
                int(timepoints[target]),
                int(labels[target]),
            )
            distances.setdefault(key, float(distance))
    return np.fromiter(distances.values(), dtype=np.float64)


def warn_association_distances(
    distances: np.ndarray,
    max_distance: float,
    delta_cutoff: int,
    cutoff_name: str,
    dataset_name: str,
) -> None:
    """Warn when labeled associations cannot pass an inference distance cutoff."""
    exceeds = distances > max_distance
    n_exceeds = int(exceeds.sum())
    if n_exceeds == 0:
        return
    logger.warning(
        "%s: %d/%d (%.2f%%) unique matched GT forward associations within "
        "delta_cutoff=%d exceed %s=%g (p99=%.2f, max=%.2f). These associations "
        "are labeled positive but cannot be recovered with this inference cutoff.",
        dataset_name,
        n_exceeds,
        len(distances),
        100 * n_exceeds / len(distances),
        delta_cutoff,
        cutoff_name,
        max_distance,
        np.quantile(distances, 0.99),
        distances.max(),
    )


def pad_tensor(x, n_max: int, dim=0, value=0):
    n = x.shape[dim]
    if n_max < n:
        raise ValueError(f"pad_tensor: n_max={n_max} must be larger than n={n} !")
    pad_shape = list(x.shape)
    pad_shape[dim] = n_max - n
    pad = torch.full(pad_shape, fill_value=value, dtype=x.dtype)
    return torch.cat((x, pad), dim=dim)


def densify_assoc(
    assoc_coo: torch.Tensor,
    batch_size: int,
    n: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Rebuild a dense ``(B, N, N)`` association matrix from sparse COO triples.

    Inverse of the COO packing done in :func:`collate_sequence_padding`: ``assoc_coo``
    is ``(nnz, 3)`` with columns ``(batch, row, col)`` and implicit value 1 (the
    association targets are binary). Done on the GPU in the training step so the dense
    matrix never crosses the dataloader IPC boundary.
    """
    out = torch.zeros(batch_size, n, n, device=device, dtype=dtype)
    if assoc_coo.numel():
        b, r, c = assoc_coo.to(device=out.device).long().unbind(1)
        out[b, r, c] = 1.0
    return out


def collate_sequence_padding(batch):
    """Collate function that pads all sequences to the same length."""
    lens = tuple(len(x["coords"]) for x in batch)
    n_max_len = max(lens)
    normal_keys = {
        "coords": 0,
        "features": 0,
        "pretrained_feats": 0,
        "labels": 0,
        "matched_gt": False,
        "gt_predecessor_set_available": False,
        "gt_successor_set_available": False,
        # There are real timepoints with t=0; pad with -1 to distinguish from those.
        "timepoints": -1,
    }
    set_keys = {
        k: v
        for k, v in normal_keys.items()
        if k in batch[0] and batch[0][k] is not None
    }
    none_keys = [k for k in normal_keys if k in batch[0] and batch[0][k] is None]
    n_pads = tuple(n_max_len - s for s in lens)
    batch_new = {
        k: torch.stack(
            [pad_tensor(x[k], n_max=n_max_len, value=v) for x in batch], dim=0
        )
        for k, v in set_keys.items()
    }
    scalar_keys = ("window_index", "seg_index", "window_start", "dataset_index")
    for k in scalar_keys:
        if k in batch[0]:
            batch_new[k] = torch.stack([x[k] for x in batch], dim=0)
    for k in none_keys:
        batch_new[k] = None
    if "assoc_matrix" in batch[0]:
        # Ship the association targets sparsely (COO) rather than as a dense
        # (B, Nmax, Nmax) float buffer. The matrix is binary and >98% zeros, so a
        # dense collate + worker->main IPC of O(B*Nmax^2) dominated the dataloader
        # cost at large windows. ``densify_assoc`` rebuilds the dense matrix on the
        # GPU in the training step. Rows/cols are < the sample's own length <=
        # n_max_len, so the (i, row, col) triples index the padded batch directly.
        coos = []
        for i, x in enumerate(batch):
            nz = torch.nonzero(x["assoc_matrix"], as_tuple=False)  # (nnz, 2)
            if nz.numel():
                bcol = torch.full((nz.shape[0], 1), i, dtype=torch.int32)
                coos.append(torch.cat([bcol, nz.to(torch.int32)], dim=1))
        batch_new["assoc_coo"] = (
            torch.cat(coos, dim=0) if coos else torch.zeros((0, 3), dtype=torch.int32)
        )

    # boolean mask flagging padded tokens so they can be ignored downstream
    pad_mask = torch.zeros((len(batch), n_max_len), dtype=torch.bool)
    for i, n_pad in enumerate(n_pads):
        pad_mask[i, n_max_len - n_pad :] = True
    batch_new["padding_mask"] = pad_mask.bool()
    return batch_new
