"""Immutable tracking sequences and CTC-folder loading (torch-free).

This module holds the format-neutral data model (:class:`Segmentation`,
:class:`TrackingSequence`) and the canonical CTC-folder loaders. It imports no torch in
its own source; the torch dataset and collation live in ``dataset.py``, which depends on
this module (one-way ``dataset`` -> ``io``).
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import joblib
import networkx as nx
import numpy as np
import pandas as pd
import tifffile
from tqdm import tqdm

from trackastra.data import wrfeat
from trackastra.data._check_ctc import _check_ctc, _get_node_attributes
from trackastra.data.matching import matching
from trackastra.utils import normalize

logger = logging.getLogger(__name__)


def _immutable_array(value: np.ndarray, *, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value).copy()
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"Expected {ndim} dimensions, got shape {array.shape}")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class Segmentation:
    """One segmentation of a movie, stored as flat columnar detections.

    All detections across the movie are concatenated and sorted by ``timepoints`` so a
    temporal window is a contiguous slice (see :class:`trackastra.data.dataset`). This
    replaces the former per-frame ``DetectionFrame`` / ``DetectionSeries`` pair. ``masks``
    (optional, ``(n_frames, ...)``) is retained only for inference.
    """

    name: str
    n_frames: int
    coords: np.ndarray
    labels: np.ndarray
    timepoints: np.ndarray
    features: dict[str, np.ndarray]
    track_indices: np.ndarray
    masks: np.ndarray | None = None

    def __post_init__(self) -> None:
        coords = _immutable_array(self.coords, ndim=2)
        labels = _immutable_array(self.labels, ndim=1)
        timepoints = _immutable_array(self.timepoints, ndim=1)
        track_indices = _immutable_array(self.track_indices, ndim=1)
        n = len(coords)
        if len(labels) != n or len(timepoints) != n or len(track_indices) != n:
            raise ValueError(
                "coords, labels, timepoints, and track_indices must be aligned"
            )
        if coords.shape[1] not in (2, 3):
            raise ValueError("Detection coordinates must be 2D or 3D")
        if n and np.any(np.diff(timepoints) < 0):
            raise ValueError("timepoints must be sorted in non-decreasing order")
        if n and (timepoints.min() < 0 or timepoints.max() >= int(self.n_frames)):
            raise ValueError("timepoints must lie within [0, n_frames)")
        if np.any(track_indices < -1):
            raise ValueError("track_indices may only use -1 for unmatched detections")
        features = {}
        for name, values in self.features.items():
            values = _immutable_array(values)
            if values.ndim == 0 or len(values) != n:
                raise ValueError(f"Feature {name!r} is not aligned with detections")
            features[name] = values
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "n_frames", int(self.n_frames))
        object.__setattr__(self, "coords", coords)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "timepoints", timepoints)
        object.__setattr__(self, "track_indices", track_indices)
        object.__setattr__(self, "features", features)
        if self.masks is not None:
            masks = _immutable_array(self.masks)
            if len(masks) != int(self.n_frames):
                raise ValueError("masks must have one frame per timepoint")
            object.__setattr__(self, "masks", masks)

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def dim(self) -> int:
        return self.coords.shape[1]

    def __reduce__(self):
        return (
            type(self),
            (
                self.name,
                self.n_frames,
                self.coords,
                self.labels,
                self.timepoints,
                self.features,
                self.track_indices,
                self.masks,
            ),
        )


@dataclass(frozen=True)
class TrackingSequence:
    root: Path
    ndim: int
    segmentations: tuple[Segmentation, ...]
    lineage_relation: np.ndarray
    lineage_parents: np.ndarray
    images: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.ndim not in (2, 3):
            raise ValueError("Only 2D and 3D tracking data is supported")
        segmentations = tuple(self.segmentations)
        relation = _immutable_array(self.lineage_relation, ndim=2).astype(bool)
        parents = _immutable_array(self.lineage_parents, ndim=1)
        if relation.shape[0] != relation.shape[1] or len(parents) != len(relation):
            raise ValueError("Lineage arrays must describe the same tracklets")
        if not np.array_equal(relation, relation.T):
            raise ValueError("lineage_relation must be symmetric")
        if len(relation) and not np.all(np.diag(relation)):
            raise ValueError("Each tracklet must be related to itself")
        for seg in segmentations:
            if seg.coords.shape[1] != self.ndim:
                raise ValueError("Segmentation dimensionality does not match sequence")
            if np.any(seg.track_indices >= len(relation)):
                raise ValueError("Segmentation references an unknown track index")
        relation.setflags(write=False)
        parents.setflags(write=False)
        object.__setattr__(self, "root", Path(self.root))
        object.__setattr__(self, "segmentations", segmentations)
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
                self.segmentations,
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
        """Load a CTC-like sequence into immutable detections and lineage metadata.

        Standard CTC layouts are resolved from the movie root (for example
        ``01`` with ``01_GT/TRA`` and optional ``01_ST/SEG``), as are simple
        ``img/`` + ``TRA/`` layouts. Ground-truth ``*_GT/TRA`` masks are refined
        with ``np.maximum(TRA, ST)`` when the matching ``*_ST/SEG`` folder exists.
        Detection folders also use standard CTC resolution, so requesting
        ``detection_folders=("SEG",)`` uses ``*_ST/SEG`` when present.
        """
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
    """Resolve detection masks, including standard CTC ``*_ST/SEG`` folders."""
    folder = Path(folder)
    if folder.is_absolute():
        guesses = (folder,)
    else:
        guesses = (
            root / folder,
            Path(f"{root}_{folder}"),
            Path(f"{root}_ST") / folder,
            Path(f"{root}_GT") / folder,
        )
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


def _load_normalized_images(
    image_path: Path,
    start: int,
    stop: int,
    temporal_step: int,
    spatial_step: int,
    ndim: int,
) -> np.ndarray:
    """Load and percentile-normalize the image frames of a CTC sequence."""
    images = _ensure_ndim(
        _load_tiffs(
            image_path, start, stop, temporal_step, spatial_step, np.dtype(np.float32)
        ),
        ndim,
    )
    return np.stack(
        [normalize(image) for image in tqdm(images, desc="Normalizing", leave=False)]
    )


def _load_refined_masks(
    mask_path: Path,
    start: int,
    stop: int,
    temporal_step: int,
    spatial_step: int,
    ndim: int,
) -> np.ndarray:
    """Load a CTC mask folder, ST-refining a ``_GT/TRA`` folder with its silver SEG."""
    masks = _ensure_ndim(
        _load_tiffs(
            mask_path, start, stop, temporal_step, spatial_step, np.dtype(np.int32)
        ),
        ndim,
    )
    return _correct_with_st(mask_path, masks, start, stop, temporal_step, spatial_step)


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
    if not detection_folders:
        raise ValueError("At least one detection folder is required")
    detection_paths, resolved_folders, missing = [], [], []
    for folder in detection_folders:
        try:
            detection_paths.append(_resolve_detection_folder(root, folder))
            resolved_folders.append(folder)
        except FileNotFoundError:
            missing.append(str(folder))
    if not detection_paths:
        raise FileNotFoundError(
            f"None of the detection folders {[str(f) for f in detection_folders]} "
            f"found for {root}"
        )
    if missing:
        logger.warning(
            "%s: skipping missing detection folders %s; using %s",
            root,
            missing,
            resolved_folders,
        )
    reference_mask_path = gt_path if use_gt else detection_paths[0]
    n_frames = len(tuple(reference_mask_path.glob("*.tif")))
    start, stop = int(n_frames * slice_pct[0]), int(n_frames * slice_pct[1])
    gt_masks = _load_refined_masks(
        reference_mask_path, start, stop, downscale_temporal, downscale_spatial, ndim
    )
    images = _load_normalized_images(
        image_path, start, stop, downscale_temporal, downscale_spatial, ndim
    )
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

    segmentations = []
    for folder, detection_path in zip(resolved_folders, detection_paths):
        detection_masks = _load_refined_masks(
            detection_path, start, stop, downscale_temporal, downscale_spatial, ndim
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
                for gt_mask, detection_mask in tqdm(
                    zip(gt_masks, detection_masks),
                    total=len(detection_masks),
                    desc="Matching",
                    leave=False,
                )
            ]
        frame_features = joblib.Parallel(n_jobs=n_workers)(
            joblib.delayed(wrfeat.WRFeatures.from_mask_img)(
                mask=mask[None], img=image[None], t_start=t
            )
            for t, (mask, image) in enumerate(zip(detection_masks, images))
        )
        indices_per_frame = []
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
            indices_per_frame.append(indices)
        names = tuple(frame_features[0].features)
        coords = np.concatenate([feature.coords for feature in frame_features])
        labels = np.concatenate([feature.labels for feature in frame_features])
        timepoints = np.concatenate(
            [
                np.full(len(feature.labels), t, dtype=np.int64)
                for t, feature in enumerate(frame_features)
            ]
        )
        track_indices = (
            np.concatenate(indices_per_frame)
            if indices_per_frame
            else np.zeros(0, dtype=np.int64)
        )
        features = OrderedDict(
            (name, np.concatenate([feature.features[name] for feature in frame_features]))
            for name in names
        )
        segmentations.append(
            Segmentation(
                name=str(folder),
                n_frames=len(frame_features),
                coords=coords,
                labels=labels,
                timepoints=timepoints,
                features=features,
                track_indices=track_indices,
                masks=detection_masks if load_images else None,
            )
        )
    return sequence_type(
        root=root,
        ndim=ndim,
        segmentations=tuple(segmentations),
        lineage_relation=lineage_relation,
        lineage_parents=lineage_parents,
        images=images if load_images else None,
    )


def _resolve_inference_paths(root: Path) -> tuple[Path, Path, Path]:
    """Resolve (sequence root, image folder, GT TRA folder) for a CTC-like root."""
    if root.name == "TRA":
        gt_tra = root
        root = root.parent.parent / root.parent.name.split("_")[0]
    else:
        ctc_tra = Path(f"{root}_GT") / "TRA"
        gt_tra = ctc_tra if ctc_tra.exists() else root / "TRA"
    image_path = root / "img" if (root / "img").exists() else root
    return root, image_path, gt_tra


def load_ctc_images_masks(
    root: str | Path,
    detection_folder: str = "TRA",
    ndim: int = 2,
) -> tuple[np.ndarray, np.ndarray, Path, Path]:
    """Load normalized images and ST-refined detection masks from a CTC-like folder.

    Lean raster loader for the image-based inference path (``Trackastra.track``): it does
    NOT extract regionprops features or build a ``TrackingSequence`` (``track`` re-extracts
    features from the returned arrays). Auto-resolves the standard CTC and simple
    ``img/`` + ``TRA/`` layouts. ``TRA`` masks are refined with the matching ``*_ST/SEG``
    folder when it exists, and requesting ``detection_folder="SEG"`` resolves to
    ``*_ST/SEG`` in standard CTC layouts.

    Returns:
        Normalized images, refined detection masks, image folder, GT TRA folder.
    """
    seq_root, image_path, gt_tra = _resolve_inference_paths(Path(root).expanduser())
    detection_path = _resolve_detection_folder(seq_root, detection_folder)
    n_frames = len(tuple(detection_path.glob("*.tif")))
    images = _load_normalized_images(image_path, 0, n_frames, 1, 1, ndim)
    masks = _load_refined_masks(detection_path, 0, n_frames, 1, 1, ndim)
    if len(images) != len(masks):
        raise ValueError("Image and detection frame counts differ")
    return images, masks, image_path, gt_tra
