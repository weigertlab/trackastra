"""Immutable tracking sequences and deterministic temporal windows."""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Literal

import joblib
import networkx as nx
import numpy as np
import pandas as pd
import tifffile
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from torch.utils.data import Dataset

from trackastra.data import wrfeat
from trackastra.data._check_ctc import _check_ctc, _get_node_attributes
from trackastra.data.matching import matching
from trackastra.utils import blockwise_sum, normalize

logger = logging.getLogger(__name__)

FeatureMode = Literal["wrfeat", "wrfeat2", "wrfeat2_no_intensity"]
_FEATURE_MODES = ("wrfeat", "wrfeat2", "wrfeat2_no_intensity")


def _immutable_array(value: np.ndarray, *, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value).copy()
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"Expected {ndim} dimensions, got shape {array.shape}")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class DetectionFrame:
    """Aligned detections and canonical raw properties for one timepoint."""

    timepoint: int
    coords: np.ndarray
    labels: np.ndarray
    features: dict[str, np.ndarray]
    track_indices: np.ndarray

    def __post_init__(self) -> None:
        coords = _immutable_array(self.coords, ndim=2)
        labels = _immutable_array(self.labels, ndim=1)
        track_indices = _immutable_array(self.track_indices, ndim=1)
        n = len(coords)
        if len(labels) != n or len(track_indices) != n:
            raise ValueError("coords, labels, and track_indices must be aligned")
        if coords.shape[1] not in (2, 3):
            raise ValueError("Detection coordinates must be 2D or 3D")
        if np.any(track_indices < -1):
            raise ValueError("track_indices may only use -1 for unmatched detections")
        features = {}
        for name, values in self.features.items():
            values = _immutable_array(values)
            if values.ndim == 0 or len(values) != n:
                raise ValueError(f"Feature {name!r} is not aligned with detections")
            features[name] = values
        object.__setattr__(self, "timepoint", int(self.timepoint))
        object.__setattr__(self, "coords", coords)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "track_indices", track_indices)
        object.__setattr__(self, "features", features)

    def __len__(self) -> int:
        return len(self.labels)

    def __reduce__(self):
        return (
            type(self),
            (
                self.timepoint,
                self.coords,
                self.labels,
                self.features,
                self.track_indices,
            ),
        )


@dataclass(frozen=True)
class DetectionSeries:
    name: str
    frames: tuple[DetectionFrame, ...]
    masks: np.ndarray | None = None

    def __post_init__(self) -> None:
        frames = tuple(self.frames)
        if any(a.timepoint >= b.timepoint for a, b in pairwise(frames)):
            raise ValueError("Frame timepoints must be strictly increasing")
        if frames and len({frame.coords.shape[1] for frame in frames}) != 1:
            raise ValueError("All frames in a series must have the same dimensionality")
        object.__setattr__(self, "frames", frames)
        if self.masks is not None:
            masks = _immutable_array(self.masks)
            if len(masks) != len(frames):
                raise ValueError("masks must have one frame per detection frame")
            object.__setattr__(self, "masks", masks)

    def __len__(self) -> int:
        return len(self.frames)

    def __reduce__(self):
        return (type(self), (self.name, self.frames, self.masks))


@dataclass(frozen=True)
class TrackingSequence:
    root: Path
    ndim: int
    detection_series: tuple[DetectionSeries, ...]
    lineage_relation: np.ndarray
    lineage_parents: np.ndarray
    images: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.ndim not in (2, 3):
            raise ValueError("Only 2D and 3D tracking data is supported")
        series = tuple(self.detection_series)
        relation = _immutable_array(self.lineage_relation, ndim=2).astype(bool)
        parents = _immutable_array(self.lineage_parents, ndim=1)
        if relation.shape[0] != relation.shape[1] or len(parents) != len(relation):
            raise ValueError("Lineage arrays must describe the same tracklets")
        if not np.array_equal(relation, relation.T):
            raise ValueError("lineage_relation must be symmetric")
        if len(relation) and not np.all(np.diag(relation)):
            raise ValueError("Each tracklet must be related to itself")
        for detection in series:
            for frame in detection.frames:
                if frame.coords.shape[1] != self.ndim:
                    raise ValueError("Frame dimensionality does not match sequence")
                if np.any(frame.track_indices >= len(relation)):
                    raise ValueError("Frame references an unknown track index")
        relation.setflags(write=False)
        parents.setflags(write=False)
        object.__setattr__(self, "root", Path(self.root))
        object.__setattr__(self, "detection_series", series)
        object.__setattr__(self, "lineage_relation", relation)
        object.__setattr__(self, "lineage_parents", parents)
        if self.images is not None:
            object.__setattr__(self, "images", _immutable_array(self.images))

    def __reduce__(self):
        return (
            type(self),
            (
                self.root,
                self.ndim,
                self.detection_series,
                self.lineage_relation,
                self.lineage_parents,
                self.images,
            ),
        )

    @classmethod
    def from_ctc(
        cls,
        root: str | Path,
        ndim: int = 2,
        use_gt: bool = True,
        detection_folders: Sequence[str | Path] = ("TRA",),
        slice_pct: tuple[float, float] = (0.0, 1.0),
        downscale_spatial: int = 1,
        downscale_temporal: int = 1,
        n_workers: int = 8,
        image_folder: str | Path | None = None,
        gt_folder: str | Path | None = None,
        track_file: str | Path | None = None,
        match_threshold: float = 0.3,
        match_max_distance: float = 16,
        load_images: bool = False,
    ) -> TrackingSequence:
        return _load_ctc_sequence(
            cls,
            root,
            ndim,
            use_gt,
            detection_folders,
            slice_pct,
            downscale_spatial,
            downscale_temporal,
            n_workers,
            image_folder,
            gt_folder,
            track_file,
            match_threshold,
            match_max_distance,
            load_images,
        )


def _resolve_paths(
    root: Path,
    image_folder: str | Path | None,
    gt_folder: str | Path | None,
    track_file: str | Path | None,
    require_gt: bool,
) -> tuple[Path, Path, Path, Path]:
    root = root.expanduser()
    if root.name == "TRA":
        gt_tra = root
        root = root.parent.parent / root.parent.name.split("_")[0]
    else:
        ctc_tra = Path(f"{root}_GT") / "TRA"
        gt_tra = ctc_tra if ctc_tra.exists() else root / "TRA"
    image_path = Path(image_folder) if image_folder is not None else root
    if image_folder is None and (root / "img").exists():
        image_path = root / "img"
    if gt_folder is not None:
        gt_path = Path(gt_folder)
    elif gt_tra.parent.name.endswith("_GT"):
        gt_path = root / gt_tra.parent.name.replace("_GT", "_ST") / "SEG"
        if not gt_path.exists():
            gt_path = gt_tra
    else:
        gt_path = gt_tra
    track_path = (
        Path(track_file) if track_file is not None else gt_tra / "man_track.txt"
    )
    required_paths = [("image", image_path)]
    if require_gt:
        required_paths.append(("ground-truth", gt_path))
    for kind, path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Could not find {kind} folder: {path}")
    return root, image_path, gt_path, track_path


def _resolve_detection_folder(root: Path, folder: str | Path) -> Path:
    folder = Path(folder)
    if folder.is_absolute():
        guesses = (folder,)
    else:
        guesses = (root / folder, Path(f"{root}_{folder}"), Path(f"{root}_GT") / folder)
    for guess in guesses:
        if guess.exists():
            return guess
    raise FileNotFoundError(f"Could not find detection folder {folder!s} for {root}")


def _load_tiffs(
    folder: Path,
    start: int,
    stop: int,
    temporal_step: int,
    spatial_step: int,
    dtype: np.dtype,
) -> np.ndarray:
    files = sorted(folder.glob("*.tif"))[start:stop:temporal_step]
    if not files:
        raise ValueError(f"No TIFF frames selected from {folder}")
    values = np.stack([tifffile.imread(path).astype(dtype) for path in files])
    if spatial_step > 1:
        values = values[
            (slice(None),) + (slice(None, None, spatial_step),) * (values.ndim - 1)
        ]
    return values


def _ensure_ndim(values: np.ndarray, ndim: int) -> np.ndarray:
    if ndim == 2 and values.ndim != 3:
        raise ValueError(f"Expected 2D data, got {values.ndim - 1}D data")
    if ndim == 3 and values.ndim == 3:
        return values[:, None]
    if ndim == 3 and values.ndim != 4:
        raise ValueError(f"Expected 3D data, got {values.ndim - 1}D data")
    return values


def _correct_with_st(
    folder: Path,
    masks: np.ndarray,
    start: int,
    stop: int,
    temporal_step: int,
    spatial_step: int,
) -> np.ndarray:
    if not str(folder).endswith("_GT/TRA"):
        return masks
    st_path = folder.parent.parent / folder.parent.stem.replace("_GT", "_ST") / "SEG"
    if not st_path.exists():
        return masks
    st_masks = _load_tiffs(
        st_path, start, stop, temporal_step, spatial_step, np.dtype(np.int32)
    )
    return np.maximum(masks, st_masks)


def _filter_tracks(
    tracks: pd.DataFrame, start: int, stop: int, temporal_step: int
) -> pd.DataFrame:
    tracks = tracks[(tracks.t2 >= start) & (tracks.t1 < stop)].copy()
    tracks.t1 = (tracks.t1 - start).clip(0, stop - start - 1)
    tracks.t2 = (tracks.t2 - start).clip(0, stop - start - 1)
    tracks.loc[~tracks.parent.isin(tracks.label), "parent"] = 0
    if temporal_step > 1:
        if start % temporal_step:
            raise ValueError(
                "Selected start frame must be divisible by temporal downscale"
            )
        deleted = (
            (tracks.t1 % temporal_step != 0)
            & (tracks.t2 % temporal_step != 0)
            & (tracks.t1 // temporal_step == tracks.t2 // temporal_step)
        )
        tracks = tracks[~deleted].copy()
        tracks.loc[~tracks.parent.isin(tracks.label), "parent"] = 0
        tracks.t1 = np.ceil(tracks.t1 / temporal_step).astype(int)
        tracks.t2 = np.floor(tracks.t2 / temporal_step).astype(int)
    return tracks


def _isolated_tracks(masks: np.ndarray) -> pd.DataFrame:
    rows = [
        (int(label), t, t, 0)
        for t, mask in enumerate(masks)
        for label in np.unique(mask)
        if label != 0
    ]
    return pd.DataFrame(rows, columns=("label", "t1", "t2", "parent"))


def _lineage_arrays(
    tracks: pd.DataFrame,
) -> tuple[dict[int, int], np.ndarray, np.ndarray]:
    labels = sorted(int(label) for label in tracks.label.unique())
    index = {label: i for i, label in enumerate(labels)}
    graph = nx.DiGraph()
    graph.add_nodes_from(labels)
    graph.add_edges_from(
        (int(row.parent), int(row.label))
        for row in tracks.itertuples()
        if row.parent in index
    )
    relation = np.eye(len(labels), dtype=bool)
    for label, i in index.items():
        relatives = nx.ancestors(graph, label) | nx.descendants(graph, label)
        relation[i, [index[other] for other in relatives]] = True
    parents = np.full(len(labels), -1, dtype=np.int64)
    for row in tracks.itertuples():
        if row.parent in index:
            parents[index[int(row.label)]] = index[int(row.parent)]
    return index, relation, parents


def _load_ctc_sequence(
    sequence_type: type[TrackingSequence],
    root: str | Path,
    ndim: int,
    use_gt: bool,
    detection_folders: Sequence[str | Path],
    slice_pct: tuple[float, float],
    downscale_spatial: int,
    downscale_temporal: int,
    n_workers: int,
    image_folder: str | Path | None,
    gt_folder: str | Path | None,
    track_file: str | Path | None,
    match_threshold: float,
    match_max_distance: float,
    load_images: bool = False,
) -> TrackingSequence:
    if not 0 <= slice_pct[0] < slice_pct[1] <= 1:
        raise ValueError(f"Invalid slice_pct {slice_pct}")
    if downscale_spatial < 1 or downscale_temporal < 1:
        raise ValueError("Downscale factors must be positive integers")
    root, image_path, gt_path, track_path = _resolve_paths(
        Path(root), image_folder, gt_folder, track_file, use_gt
    )
    detection_paths = [
        _resolve_detection_folder(root, folder) for folder in detection_folders
    ]
    if not detection_paths:
        raise ValueError("At least one detection folder is required")
    reference_mask_path = gt_path if use_gt else detection_paths[0]
    n_frames = len(tuple(reference_mask_path.glob("*.tif")))
    start, stop = int(n_frames * slice_pct[0]), int(n_frames * slice_pct[1])
    gt_masks = _ensure_ndim(
        _load_tiffs(
            reference_mask_path,
            start,
            stop,
            downscale_temporal,
            downscale_spatial,
            np.dtype(np.int32),
        ),
        ndim,
    )
    gt_masks = _correct_with_st(
        reference_mask_path,
        gt_masks,
        start,
        stop,
        downscale_temporal,
        downscale_spatial,
    )
    images = _ensure_ndim(
        _load_tiffs(
            image_path,
            start,
            stop,
            downscale_temporal,
            downscale_spatial,
            np.dtype(np.float32),
        ),
        ndim,
    )
    images = np.stack([normalize(image) for image in images])
    if len(images) != len(gt_masks):
        raise ValueError("Image and ground-truth frame counts differ")

    if use_gt:
        if not track_path.exists():
            raise FileNotFoundError(f"Could not find track file: {track_path}")
        tracks = pd.read_csv(
            track_path,
            sep=r"\s+",
            names=("label", "t1", "t2", "parent"),
            dtype=int,
        )
        tracks = _filter_tracks(tracks, start, stop, downscale_temporal)
        _check_ctc(tracks, _get_node_attributes(gt_masks), gt_masks)
        track_index, lineage_relation, lineage_parents = _lineage_arrays(tracks)
        isolated_indices = None
    else:
        isolated_indices = []
        next_index = 0
        for mask in gt_masks:
            labels = [int(label) for label in np.unique(mask) if label != 0]
            isolated_indices.append(
                {label: next_index + i for i, label in enumerate(labels)}
            )
            next_index += len(labels)
        track_index = {}
        lineage_relation = np.eye(next_index, dtype=bool)
        lineage_parents = np.full(next_index, -1, dtype=np.int64)

    series = []
    for folder, detection_path in zip(detection_folders, detection_paths):
        detection_masks = _ensure_ndim(
            _load_tiffs(
                detection_path,
                start,
                stop,
                downscale_temporal,
                downscale_spatial,
                np.dtype(np.int32),
            ),
            ndim,
        )
        detection_masks = _correct_with_st(
            detection_path,
            detection_masks,
            start,
            stop,
            downscale_temporal,
            downscale_spatial,
        )
        if len(detection_masks) != len(images):
            raise ValueError(
                f"Image and detection frame counts differ for {detection_path}"
            )

        if detection_path == reference_mask_path:
            matches = [
                {int(label): int(label) for label in np.unique(mask) if label != 0}
                for mask in detection_masks
            ]
        else:
            matches = [
                {
                    int(detection): int(gt)
                    for gt, detection in matching(
                        gt_mask,
                        detection_mask,
                        threshold=match_threshold,
                        max_distance=match_max_distance,
                    )
                }
                for gt_mask, detection_mask in zip(gt_masks, detection_masks)
            ]
        frame_features = joblib.Parallel(n_jobs=n_workers)(
            joblib.delayed(wrfeat.WRFeatures.from_mask_img)(
                mask=mask[None], img=image[None], t_start=t
            )
            for t, (mask, image) in enumerate(zip(detection_masks, images))
        )
        frames = []
        for t, feature in enumerate(frame_features):
            if isolated_indices is None:
                indices = np.array(
                    [
                        track_index.get(matches[t].get(int(label), -1), -1)
                        for label in feature.labels
                    ],
                    dtype=np.int64,
                )
            else:
                indices = np.array(
                    [
                        isolated_indices[t].get(matches[t].get(int(label), -1), -1)
                        for label in feature.labels
                    ],
                    dtype=np.int64,
                )
            frames.append(
                DetectionFrame(
                    timepoint=t,
                    coords=feature.coords,
                    labels=feature.labels,
                    features=feature.features,
                    track_indices=indices,
                )
            )
        series.append(
            DetectionSeries(
                str(folder),
                tuple(frames),
                masks=detection_masks if load_images else None,
            )
        )
    return sequence_type(
        root=root,
        ndim=ndim,
        detection_series=tuple(series),
        lineage_relation=lineage_relation,
        lineage_parents=lineage_parents,
        images=images if load_images else None,
    )


def _concat_frames(
    frames: Sequence[DetectionFrame],
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, OrderedDict[str, np.ndarray]
]:
    if not frames:
        raise ValueError("Cannot concatenate an empty frame sequence")
    coords = np.concatenate([frame.coords for frame in frames])
    labels = np.concatenate([frame.labels for frame in frames])
    timepoints = np.concatenate(
        [np.full(len(frame), frame.timepoint, dtype=np.int64) for frame in frames]
    )
    track_indices = np.concatenate([frame.track_indices for frame in frames])
    names = tuple(frames[0].features)
    if any(tuple(frame.features) != names for frame in frames[1:]):
        raise ValueError("All frames in a window must have the same feature properties")
    features = OrderedDict(
        (name, np.concatenate([frame.features[name] for frame in frames]))
        for name in names
    )
    return coords, labels, timepoints, track_indices, features


def _association_target(
    track_indices: np.ndarray, lineage_relation: np.ndarray
) -> np.ndarray:
    target = np.zeros((len(track_indices), len(track_indices)), dtype=bool)
    valid = np.flatnonzero(track_indices >= 0)
    if len(valid):
        target[np.ix_(valid, valid)] = lineage_relation[
            np.ix_(track_indices[valid], track_indices[valid])
        ]
    return target


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


class TrackingData(Dataset):
    """Runtime temporal windows over an immutable tracking sequence."""

    def __init__(
        self,
        sequence: TrackingSequence,
        window_size: int = 10,
        features: FeatureMode = "wrfeat",
        augment: int = 0,
        max_detections: int | None = None,
        detect_drop: float = 0.0,
        detect_drop_fraction: float = 0.1,
    ) -> None:
        if window_size <= 1:
            raise ValueError("window_size must be greater than one")
        if features not in _FEATURE_MODES:
            raise ValueError(f"Unsupported feature mode {features!r}")
        if features != "wrfeat" and sequence.ndim != 2:
            raise ValueError(f"{features} currently supports only 2D data")
        if augment not in range(5):
            raise ValueError("augment must be between 0 and 4")
        if max_detections is not None and max_detections < window_size:
            raise ValueError("max_detections must be at least window_size")
        if not 0 <= detect_drop <= 1 or not 0 <= detect_drop_fraction <= 1:
            raise ValueError("Detection dropout values must be in [0, 1]")
        self.sequence = sequence
        self.root = sequence.root
        self.window_size = window_size
        self.features = features
        self.ndim = sequence.ndim
        self.augment = augment
        self.max_detections = max_detections
        self.detect_drop = detect_drop
        self.detect_drop_fraction = detect_drop_fraction
        self.augmenter = _wr_augmenter(augment)
        self.windows = tuple(
            (series_index, start)
            for series_index, series in enumerate(sequence.detection_series)
            for start in range(len(series.frames) - window_size + 1)
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

    def _frames(self, index: int) -> tuple[DetectionFrame, ...]:
        series_index, start = self.windows[index]
        frames = self.sequence.detection_series[series_index].frames
        return frames[start : start + self.window_size]

    def _window_arrays(self, index: int):
        coords, labels, timepoints, track_indices, features = _concat_frames(
            self._frames(index)
        )
        association = _association_target(track_indices, self.sequence.lineage_relation)
        return coords, labels, timepoints, features, association

    def _window_object_count(self, index: int) -> int:
        count = sum(len(frame) for frame in self._frames(index))
        return min(count, self.max_detections) if self.max_detections else count

    def _window_division_count(self, index: int) -> int:
        _, _, timepoints, _, association = self._window_arrays(index)
        return _division_count(association, timepoints)

    def _infer_feature_dim(self) -> int:
        for series in self.sequence.detection_series:
            for frame in series.frames:
                feature = wrfeat.WRFeatures(
                    frame.coords,
                    frame.labels,
                    np.full(len(frame), frame.timepoint, dtype=np.int32),
                    OrderedDict(frame.features),
                )
                return feature.features_stacked_for(self.features).shape[1]
        return 0

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        coords, labels, timepoints, features, association = self._window_arrays(index)
        feature = wrfeat.WRFeatures(
            coords=coords,
            labels=labels,
            timepoints=timepoints.astype(np.int32, copy=False),
            features=features,
        )
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

        if self.detect_drop and np.random.rand() < self.detect_drop:
            keep = _sample_detection_keep_indices(
                association, feature.timepoints, self.detect_drop_fraction
            )
            if len(keep) < len(feature):
                feature = _subset_features(feature, keep)
                association = association[np.ix_(keep, keep)]

        if self.augmenter is not None:
            feature = self.augmenter(feature)
        coords0 = torch.from_numpy(
            np.concatenate((feature.timepoints[:, None], feature.coords), axis=1)
        ).float()
        coords = coords0.clone()
        if self.augmenter is not None:
            coords[:, 1:] += torch.randint(0, 512, (1, self.ndim))
        result = {
            "features": torch.from_numpy(
                feature.features_stacked_for(self.features)
            ).float(),
            "coords0": coords0,
            "coords": coords,
            "assoc_matrix": torch.from_numpy(association.astype(np.float32)),
            "timepoints": torch.from_numpy(feature.timepoints).long(),
            "labels": torch.from_numpy(feature.labels).long(),
        }
        return result


def association_distances(dataset: TrackingData, delta_cutoff: int) -> np.ndarray:
    """Distances of unique positive forward associations in raw windows."""
    if delta_cutoff < 1:
        raise ValueError("delta_cutoff must be positive")
    distances = {}
    for window, (series_index, _) in enumerate(dataset.windows):
        coords, labels, timepoints, _, association = dataset._window_arrays(window)
        rows, cols = np.nonzero(association)
        delta = timepoints[cols] - timepoints[rows]
        valid = (delta > 0) & (delta <= delta_cutoff)
        rows, cols = rows[valid], cols[valid]
        edge_distances = np.linalg.norm(coords[cols] - coords[rows], axis=1)
        for source, target, distance in zip(rows, cols, edge_distances):
            key = (
                series_index,
                int(timepoints[source]),
                int(labels[source]),
                int(timepoints[target]),
                int(labels[target]),
            )
            distances.setdefault(key, float(distance))
    return np.fromiter(distances.values(), dtype=np.float64)


def _resolve_inference_paths(root: Path) -> tuple[Path, Path]:
    """Resolve the image folder and GT TRA folder for a CTC-like sequence root."""
    if root.name == "TRA":
        gt_tra = root
        root = root.parent.parent / root.parent.name.split("_")[0]
    else:
        ctc_tra = Path(f"{root}_GT") / "TRA"
        gt_tra = ctc_tra if ctc_tra.exists() else root / "TRA"
    image_path = root / "img" if (root / "img").exists() else root
    return image_path, gt_tra


def load_ctc_for_inference(
    root: str | Path,
    detection_folder: str = "TRA",
    ndim: int = 2,
    loader=None,
) -> tuple[np.ndarray, np.ndarray, Path, Path]:
    """Load images, detection masks, and GT path from a CTC-like folder.

    Canonical inference entry point: builds a detection-only ``TrackingSequence``
    with raw arrays retained (``load_images=True``) and returns the dense arrays the
    image-based inference API consumes. Pass ``loader`` (e.g. a joblib-cached
    ``TrackingSequence.from_ctc``) to reuse cached sequences across repeated calls,
    such as wandb predict-logging on the same validation movie.

    Returns:
        Normalized images, refined detection masks, image path, GT TRA path.
    """
    build = loader if loader is not None else TrackingSequence.from_ctc
    sequence = build(
        root=root,
        ndim=ndim,
        use_gt=False,
        detection_folders=(detection_folder,),
        load_images=True,
    )
    image_path, gt_tra = _resolve_inference_paths(Path(root).expanduser())
    return sequence.images, sequence.detection_series[0].masks, image_path, gt_tra


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
        "%s: %d/%d (%.2f%%) unique supervised forward associations within "
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

    if any("loss_mask" in x for x in batch):
        loss_mask = torch.zeros((len(batch), n_max_len, n_max_len), dtype=torch.bool)
        for i, x in enumerate(batch):
            n = len(x["coords"])
            if "loss_mask" in x:
                loss_mask[i, :n, :n] = x["loss_mask"]
            else:
                loss_mask[i, :n, :n] = True
        batch_new["loss_mask"] = loss_mask

    # boolean mask flagging padded tokens so they can be ignored downstream
    pad_mask = torch.zeros((len(batch), n_max_len), dtype=torch.bool)
    for i, n_pad in enumerate(n_pads):
        pad_mask[i, n_max_len - n_pad :] = True
    batch_new["padding_mask"] = pad_mask.bool()
    return batch_new
