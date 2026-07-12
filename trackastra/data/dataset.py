"""Runtime temporal-window dataset and collation over tracking sequences (torch).

Depends on the torch-free data model in ``io.py`` (one-way ``dataset`` -> ``io``). A
temporal window is a ``timepoints``-range slice of a flat :class:`DetectionSequence`.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from torch.utils.data import Dataset

from trackastra.data import wrfeat
from trackastra.data.io import DetectionSequence, DetectionSupervision, TrackingSequence
from trackastra.utils import blockwise_sum

logger = logging.getLogger(__name__)

FeatureMode = Literal["none", "intensity", "wrfeat2", "wrfeat2_no_intensity"]
_FEATURE_MODES = tuple(wrfeat.FEATURE_RECIPES)


@dataclass(frozen=True)
class AugmentationConfig:
    """Resolved local coordinate augmentation preset and model-unit magnitudes."""

    preset: int = 0
    jitter: float = 3.0
    drift: float = 10.0
    tilt: float = 0.0
    frame_jump: float = 0.0
    frame_jump_p: float = 0.0

    @classmethod
    def from_details(
        cls,
        preset: int,
        details: Mapping[str, float] | None = None,
    ) -> AugmentationConfig:
        if preset not in range(5):
            raise ValueError("augment must be between 0 and 4")
        values = {
            "preset": preset,
            "jitter": 3.0,
            "drift": 10.0,
            "tilt": 0.0,
            "frame_jump": 0.0,
            "frame_jump_p": 0.0,
        }
        if details is not None:
            unknown = set(details) - {
                "jitter",
                "drift",
                "tilt",
                "frame_jump",
                "frame_jump_p",
            }
            if unknown:
                raise ValueError(
                    f"Unknown augment_details keys {sorted(unknown)}; "
                    "supported keys are ['drift', 'frame_jump', 'frame_jump_p', "
                    "'jitter', 'tilt']"
                )
            values.update(details)
        jitter = float(values["jitter"])
        drift = float(values["drift"])
        tilt = float(values["tilt"])
        frame_jump = float(values["frame_jump"])
        frame_jump_p = float(values["frame_jump_p"])
        if not np.isfinite(jitter) or jitter < 0:
            raise ValueError(f"augment_details.jitter must be non-negative, got {jitter}")
        if not np.isfinite(drift) or drift < 0:
            raise ValueError(f"augment_details.drift must be non-negative, got {drift}")
        if not np.isfinite(tilt) or tilt < 0:
            raise ValueError(f"augment_details.tilt must be non-negative, got {tilt}")
        if not np.isfinite(frame_jump) or frame_jump < 0:
            raise ValueError(
                f"augment_details.frame_jump must be non-negative, got {frame_jump}"
            )
        if not np.isfinite(frame_jump_p) or not 0 <= frame_jump_p <= 1:
            raise ValueError(
                f"augment_details.frame_jump_p must be in [0, 1], got {frame_jump_p}"
            )
        return cls(
            preset=preset,
            jitter=jitter,
            drift=drift,
            tilt=tilt,
            frame_jump=frame_jump,
            frame_jump_p=frame_jump_p,
        )


def _feature_properties_for_sequence(sequence: TrackingSequence) -> set[str]:
    feature_sets = [set(seg.features) for seg in sequence.detections]
    if not feature_sets:
        return set()
    return set.intersection(*feature_sets)


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
    matched_gt: np.ndarray | None = None,
) -> np.ndarray:
    """Sample a per-frame spatial crop around a GT seed, closed under GT lineage.

    Picks a random ground-truth (``matched_gt``) detection as the seed, keeps the
    ``max_detections`` detections nearest the seed in every frame (the spatial crop,
    a density-adaptive box around the seeded cell), then completes every GT lineage
    that intersects the crop so no GT link is ever severed. Because completion pulls
    a kept parent's whole component in, divisions on the crop's lineages are kept
    intact (both daughters together); non-GT detections act as distractors/negatives.

    Unlike a directional backward/forward walk, the closure is orientation-free: a
    GT detection that only enters as a distractor still has its full in-window
    lineage restored, so the kept set never contains a partial lineage.

    ``matched_gt`` (``lineage_index >= 0``) marks the GT-annotated detections used
    for seeding; without it, or when there are none, any detection may seed.
    """
    n = len(coords)
    if n == 0:
        return np.arange(0)
    matched = (
        matched_gt.astype(bool) if matched_gt is not None else np.ones(n, dtype=bool)
    )
    gt = np.flatnonzero(matched)
    pool = gt if len(gt) else np.arange(n)
    seed = pool[np.random.randint(len(pool))]

    # GT lineage components, computed on the matched-node submatrix only: non-GT
    # detections carry no association (``_association_target`` fills GT rows/cols
    # only), so this scales with the number of GT nodes, not the total detection
    # count -- essential when sparse-GT over-detection makes a window mostly junk.
    if len(gt):
        sub = association[np.ix_(gt, gt)]
        _, gt_comp = connected_components(
            csr_matrix(np.logical_or(sub, sub.T)), directed=False
        )
        anchor = gt[gt_comp == gt_comp[np.searchsorted(gt, seed)]]
    else:
        gt_comp = np.zeros(0, dtype=np.int64)
        anchor = np.array([seed])
    anchor_mask = np.zeros(n, dtype=bool)
    anchor_mask[anchor] = True

    # Spatial crop: per frame, the max_detections detections nearest (min distance)
    # to the seed lineage's node(s) in that frame -- so after a division both
    # daughters anchor their own distractors -- falling back to the whole lineage
    # track when it is absent from a frame.
    keep_mask = np.zeros(n, dtype=bool)
    for t in np.unique(timepoints):
        idx_t = np.flatnonzero(timepoints == t)
        if len(idx_t) <= max_detections:
            keep_mask[idx_t] = True
            continue
        anchor_t = idx_t[anchor_mask[idx_t]]
        ref = coords[anchor_t] if len(anchor_t) else coords[anchor]
        d = np.linalg.norm(coords[idx_t][:, None, :] - ref[None, :, :], axis=2).min(1)
        keep_mask[idx_t[np.argpartition(d, max_detections)[:max_detections]]] = True

    # Close under GT lineage: any lineage the crop touched is kept whole, so a kept
    # detection never has an association pointing outside the kept set.
    if len(gt):
        hit = np.unique(gt_comp[keep_mask[gt]])
        keep_mask[gt[np.isin(gt_comp, hit)]] = True
    return np.flatnonzero(keep_mask)


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


def _wr_augmenter(config: AugmentationConfig):
    level = config.preset
    jitter = (-config.jitter, config.jitter)
    drift = (-config.drift, config.drift)
    frame_jump = (-config.frame_jump, config.frame_jump)
    common = [
        wrfeat.WRRandomFlip(p=0.5),
        wrfeat.WRRandomAffine(
            p=0.8,
            degrees=180,
            tilt_degrees=config.tilt,
            scale=(2 / 3, 1.5),
            shear=(0.1, 0.1),
        ),
    ]
    if level == 1:
        augmentations = common
    elif level == 2:
        augmentations = [
            *common,
            wrfeat.WRRandomBrightness(p=0.8),
            wrfeat.WRRandomOffset(p=0.8, offset=jitter),
        ]
    elif level in (3, 4):
        if level == 4:
            common.append(
                wrfeat.WRRandomShapeJitter(p=0.8, scale=(0.9, 1.1), shear=0.05)
            )
        augmentations = [
            *common,
            wrfeat.WRRandomBrightness(p=0.8),
            wrfeat.WRRandomMovement(offset=drift, p=0.3),
            wrfeat.WRRandomFrameJump(offset=frame_jump, p=config.frame_jump_p),
            wrfeat.WRRandomOffset(p=0.8, offset=jitter),
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


def _add_bincount(total: np.ndarray, values: np.ndarray) -> np.ndarray:
    counts = np.bincount(values)
    if len(counts) > len(total):
        total = np.pad(total, (0, len(counts) - len(total)))
    total[: len(counts)] += counts
    return total


def _node_degree_counts(sequence: TrackingSequence) -> tuple[np.ndarray, np.ndarray]:
    gt = sequence.gt
    if (
        gt is None
        or gt.node_in_degree is None
        or gt.node_out_degree is None
        or sequence.supervision is None
    ):
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty

    in_counts = np.zeros(0, dtype=np.int64)
    out_counts = np.zeros(0, dtype=np.int64)
    for supervision in sequence.supervision:
        if supervision is None or supervision.gt_node_index is None:
            continue
        gt_node_index = supervision.gt_node_index
        valid = gt_node_index >= 0
        valid_in = valid.copy()
        valid_out = valid.copy()
        if supervision.gt_predecessor_set_available is not None:
            valid_in &= supervision.gt_predecessor_set_available
        if supervision.gt_successor_set_available is not None:
            valid_out &= supervision.gt_successor_set_available
        in_counts = _add_bincount(in_counts, gt.node_in_degree[gt_node_index[valid_in]])
        out_counts = _add_bincount(
            out_counts, gt.node_out_degree[gt_node_index[valid_out]]
        )
    return in_counts, out_counts


class TrackingDataset(Dataset):
    """Runtime temporal windows over an immutable tracking sequence."""

    def __init__(
        self,
        sequence: TrackingSequence,
        window_size: int = 4,
        features: FeatureMode = "wrfeat2",
        augment: int = 0,
        augment_details: Mapping[str, float] | AugmentationConfig | None = None,
        position_noise: float = 0.0,
        max_detections: int | None = None,
        detect_drop: float = 0.0,
        detect_drop_fraction: float = 0.1,
        normalize_diameter: float | None = None,
        feature_group_drop: Mapping[str, float] | None = None,
        dataset_index: int = 0,
    ) -> None:
        if window_size <= 1:
            raise ValueError("window_size must be greater than one")
        if sequence.gt is None:
            raise ValueError("TrackingDataset requires a gt LineageGraph")
        if features not in _FEATURE_MODES:
            raise ValueError(f"Unsupported feature mode {features!r}")
        # A sequence may provide only a subset of a recipe's source properties: the
        # missing output columns are masked (routed through the model's null pathway)
        # instead of rejected, so datasets with different feature availability train
        # one model. Log which required properties are absent for visibility.
        missing_props = sorted(
            set(wrfeat.feature_recipe_keys(features))
            - _feature_properties_for_sequence(sequence)
        )
        if missing_props:
            logger.info(
                "Sequence %s lacks feature properties %s for mode %r; "
                "those feature columns will be masked.",
                getattr(sequence, "root", "<sequence>"),
                missing_props,
                features,
            )
        augment_config = (
            augment_details
            if isinstance(augment_details, AugmentationConfig)
            else AugmentationConfig.from_details(augment, augment_details)
        )
        if position_noise < 0:
            raise ValueError("position_noise must be non-negative")
        if max_detections is not None and max_detections < 1:
            raise ValueError("max_detections must be at least 1")
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
        self.augment_config = augment_config
        self.position_noise = position_noise
        self.max_detections = max_detections
        self.detect_drop = detect_drop
        self.detect_drop_fraction = detect_drop_fraction
        # Per-window feature-group dropout: precompute the stacked columns each group
        # occupies so __getitem__ can mask them out with the given probability. This
        # trains the model's null pathway so it stays robust to datasets missing a
        # whole feature group (e.g. intensity, or all shape).
        self._feature_group_drop: list[tuple[np.ndarray, float]] = []
        for group, prob in (feature_group_drop or {}).items():
            if not 0 <= prob <= 1:
                raise ValueError(
                    f"feature_group_drop[{group!r}] must be in [0, 1], got {prob}"
                )
            if prob > 0:
                columns = wrfeat.feature_group_columns(features, self.ndim, group)
                self._feature_group_drop.append((columns, prob))
        self.scale_factor = _normalize_diameter_factor(sequence, normalize_diameter)
        if normalize_diameter is not None:
            logger.info(
                "Normalizing %s: scale factor %.4g (target diameter %.4g)",
                getattr(sequence, "root", "<sequence>"),
                self.scale_factor,
                normalize_diameter,
            )
        self.augmenter = _wr_augmenter(augment_config)
        # With sparse ground truth, timepoints can start above 0 and contain
        # gaps, so a window's time range [start, start + window_size) may hold
        # no detections. Drop those empty windows; they carry no supervision and
        # would crash downstream sampling (e.g. timepoints.max() on empty).
        self.windows = tuple(
            (seg_index, start)
            for seg_index, seg in enumerate(sequence.detections)
            for start in range(seg.n_frames - window_size + 1)
            if np.searchsorted(seg.timepoints, start + window_size, side="left")
            > np.searchsorted(seg.timepoints, start, side="left")
        )
        self.n_objects = tuple(
            self._window_object_count(index) for index in range(len(self))
        )
        self.n_divs = tuple(
            self._window_division_count(index) for index in range(len(self))
        )
        self.node_in_degree_counts, self.node_out_degree_counts = _node_degree_counts(
            sequence
        )
        self.feat_dim = self._infer_feature_dim()

    def __len__(self) -> int:
        return len(self.windows)

    def _window_slice(
        self, index: int
    ) -> tuple[DetectionSequence, DetectionSupervision, slice]:
        seg_index, start = self.windows[index]
        seg = self.sequence.detections[seg_index]
        supervision = self.sequence.supervision[seg_index]
        if supervision is None:
            raise ValueError("TrackingDataset requires supervision for every detection")
        lo = int(np.searchsorted(seg.timepoints, start, side="left"))
        hi = int(np.searchsorted(seg.timepoints, start + self.window_size, side="left"))
        return seg, supervision, slice(lo, hi)

    def _window_arrays(self, index: int):
        # Slices of the stored arrays are read-only views; return writable copies so
        # downstream WRFeatures augmentation can mutate in place (matching the former
        # _concat_frames, which produced fresh np.concatenate buffers per window).
        seg, supervision, sl = self._window_slice(index)
        coords = seg.coords[sl].copy()
        labels = seg.labels[sl].copy()
        timepoints = seg.timepoints[sl].copy()
        lineage_index = supervision.lineage_index[sl]
        features = OrderedDict(
            (name, values[sl].copy()) for name, values in seg.features.items()
        )
        association = _association_target(lineage_index, self.sequence.gt.lineage_relation)
        return coords, labels, timepoints, features, association

    def _window_matched_gt(self, index: int) -> np.ndarray:
        _seg, supervision, sl = self._window_slice(index)
        if supervision.matched_gt is None:
            return np.ones(sl.stop - sl.start, dtype=bool)
        return supervision.matched_gt[sl].copy()

    def _window_gt_link_set_availability(
        self, index: int
    ) -> tuple[np.ndarray, np.ndarray]:
        _seg, supervision, sl = self._window_slice(index)
        n = sl.stop - sl.start
        predecessor = (
            np.ones(n, dtype=bool)
            if supervision.gt_predecessor_set_available is None
            else supervision.gt_predecessor_set_available[sl].copy()
        )
        successor = (
            np.ones(n, dtype=bool)
            if supervision.gt_successor_set_available is None
            else supervision.gt_successor_set_available[sl].copy()
        )
        return predecessor, successor

    def _window_node_degrees(self, index: int):
        """True direct in/out degrees per window node, or ``None`` if unavailable.

        Degrees are looked up from the GT lineage via ``gt_node_index``; unmatched
        detections (``gt_node_index == -1``) get ``-1`` (undefined), masked downstream.
        """
        _seg, supervision, sl = self._window_slice(index)
        gt = self.sequence.gt
        if (
            gt is None
            or gt.node_in_degree is None
            or gt.node_out_degree is None
            or supervision.gt_node_index is None
        ):
            return None
        gt_node_index = supervision.gt_node_index[sl]
        matched = gt_node_index >= 0
        idx = np.where(matched, gt_node_index, 0)
        node_in_degree = np.where(matched, gt.node_in_degree[idx], -1).astype(np.int64)
        node_out_degree = np.where(matched, gt.node_out_degree[idx], -1).astype(np.int64)
        return node_in_degree, node_out_degree

    def _window_object_count(self, index: int) -> int:
        seg, _supervision, sl = self._window_slice(index)
        timepoints = seg.timepoints[sl]
        if self.max_detections is None:
            return len(timepoints)
        _, counts = np.unique(timepoints, return_counts=True)
        return int(np.minimum(counts, self.max_detections).sum())

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
            return feature.stacked_with_mask(self.features)[0].shape[1]
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
        node_degrees = self._window_node_degrees(index)
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
                matched_gt=matched_gt,
            )
            if len(keep) < len(feature):
                feature = _subset_features(feature, keep)
                association = association[np.ix_(keep, keep)]
                matched_gt = matched_gt[keep]
                gt_predecessor_set_available = gt_predecessor_set_available[keep]
                gt_successor_set_available = gt_successor_set_available[keep]
                if node_degrees is not None:
                    node_degrees = (node_degrees[0][keep], node_degrees[1][keep])

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
                if node_degrees is not None:
                    node_degrees = (node_degrees[0][keep], node_degrees[1][keep])

        if self.augmenter is not None:
            feature = self.augmenter(feature)
        feature_values, feature_mask = feature.stacked_with_mask(self.features)
        for columns, prob in self._feature_group_drop:
            if np.random.rand() < prob:
                feature_mask[:, columns] = False
                feature_values[:, columns] = 0.0
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
            "feature_mask": torch.from_numpy(feature_mask),
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
        if node_degrees is not None:
            result["node_in_degree"] = torch.from_numpy(node_degrees[0]).long()
            result["node_out_degree"] = torch.from_numpy(node_degrees[1]).long()
        if return_all:
            result["supervision_mask"] = association_supervision_mask(
                result["timepoints"],
                result["gt_predecessor_set_available"],
                result["gt_successor_set_available"],
            )
        return result

    @staticmethod
    def write_geff(
        item: Mapping[str, torch.Tensor],
        path: str | Path,
        **kwargs,
    ) -> None:
        """Write a single ``__getitem__`` output to a GEFF file for viz/debugging.

        One node per detection with its ``(t, (z), y, x)`` coordinates and a
        ``radius`` recovered from the log1p-diameter feature (column 0 of the
        ``wrfeat2`` stack); ``assoc_matrix[i, j] > 0`` becomes a parent -> child
        edge. ``radius`` is registered as the standard GEFF sphere property.

        Args:
            item: A dict returned by :meth:`__getitem__`.
            path: Destination ``.geff`` path.
            **kwargs: Forwarded to :func:`geff.write` (e.g. ``overwrite=True``,
                ``zarr_format=3``). ``axis_names``/``axis_types``/``axis_units``
                are set from the detection geometry and cannot be overridden.
        """
        import geff
        import networkx as nx
        from geff import GeffMetadata

        coords0 = item["coords0"].numpy()
        ndim = coords0.shape[1] - 1
        if ndim == 2:
            axis_names = ["t", "y", "x"]
        elif ndim == 3:
            axis_names = ["t", "z", "y", "x"]
        else:
            raise ValueError(f"Unsupported number of spatial dimensions: {ndim}")
        axis_types = ["time"] + ["space"] * ndim
        axis_units = [None] + ["micrometer"] * ndim

        features = item["features"].numpy()
        # column 0 of the wrfeat2 stack is log1p(diameter); radius = diameter / 2
        if features.shape[1] > 0:
            radius = np.expm1(np.maximum(features[:, 0], 0.0)) / 2.0
        else:
            radius = np.zeros(len(coords0), dtype=np.float32)

        # assoc_matrix is the (symmetric, self-linked) same-lineage-tree relation,
        # so a raw forward scan would connect a detection to every descendant in
        # the window. Keep only next-frame links: child in the frame immediately
        # after the parent's (robust to sub-sampled/non-unit frame indices).
        assoc = item["assoc_matrix"].numpy()
        timepoints = coords0[:, 0]
        unique_t = np.unique(timepoints)
        next_t = dict(zip(unique_t[:-1], unique_t[1:]))
        parents, children = np.nonzero(assoc > 0)
        next_frame = np.array(
            [next_t.get(timepoints[p], np.inf) == timepoints[c]
             for p, c in zip(parents, children)],
            dtype=bool,
        )
        parents, children = parents[next_frame], children[next_frame]

        graph = nx.DiGraph()
        for node, row in enumerate(coords0):
            attrs = {name: float(v) for name, v in zip(axis_names, row)}
            attrs["t"] = int(round(row[0]))
            attrs["radius"] = float(radius[node])
            graph.add_node(node, **attrs)
        graph.add_edges_from(zip(parents.tolist(), children.tolist()))

        geff.write(
            graph,
            str(path),
            axis_names=axis_names,
            axis_types=axis_types,
            axis_units=axis_units,
            **kwargs,
        )
        meta = GeffMetadata.read(str(path))
        meta.sphere = "radius"
        meta.write(str(path))


def association_distances(dataset: TrackingDataset, delta_cutoff: int) -> np.ndarray:
    """Distances of unique positive forward associations in runtime model units."""
    if delta_cutoff < 1:
        raise ValueError("delta_cutoff must be positive")
    distances = {}
    for window, (seg_index, _) in enumerate(dataset.windows):
        coords, labels, timepoints, _, association = dataset._window_arrays(window)
        coords = coords * dataset.scale_factor
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
    spatial_cutoff: float,
    delta_cutoff: int,
    cutoff_name: str,
    dataset_name: str,
) -> None:
    """Warn when labeled associations cannot pass an inference distance cutoff."""
    exceeds = distances > spatial_cutoff
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
        spatial_cutoff,
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
        # Padded nodes get mask=False; they route through the model's null pathway
        # and are ignored anyway via padding_mask.
        "feature_mask": False,
        "pretrained_feats": 0,
        "labels": 0,
        "matched_gt": False,
        "gt_predecessor_set_available": False,
        "gt_successor_set_available": False,
        # -1 sentinel (distinct from real degrees 0/1/2); padded nodes are masked
        # out of the node loss anyway via padding_mask / availability flags.
        "node_in_degree": -1,
        "node_out_degree": -1,
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
