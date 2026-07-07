"""Immutable tracking sequences and CTC-folder loading (torch-free).

This module holds the format-neutral data model (:class:`DetectionSequence`,
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
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from skimage.measure import regionprops
from tqdm import tqdm

from trackastra.data import wrfeat
from trackastra.data._check_ctc import _check_ctc, _get_node_attributes
from trackastra.data.matching import (
    intersection_over_union,
    label_overlap,
    match_points,
    relabel_sequential,
)
from trackastra.data.utils import apply_spatial_spacing, validate_spatial_spacing
from trackastra.utils import normalize

logger = logging.getLogger(__name__)


def _resolve_detection_spacing(
    spacing: tuple[float, ...] | list[float] | np.ndarray | None,
    ndim: int,
    *,
    context: str,
) -> tuple[float, ...]:
    if spacing is None and ndim == 3:
        logger.warning(
            "%s received 3D detections without spacing; assuming unit "
            "source-to-model scale (1, 1, 1). If coordinates are pixels/voxels, "
            "pass voxel spacing in model units, normally micrometers. If coordinates "
            "are already in model units, pass spacing=(1, 1, 1) explicitly.",
            context,
        )
    return validate_spatial_spacing(spacing, ndim)


def _immutable_array(value: np.ndarray, *, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value).copy()
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"Expected {ndim} dimensions, got shape {array.shape}")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class DetectionSequence:
    """Flat columnar detections for one source across a movie.

    All detections across the movie are concatenated and sorted by ``timepoints`` so a
    temporal window is a contiguous slice (see :class:`trackastra.data.dataset`).
    ``coords`` are model-space spatial coordinates. For microscopy checkpoints, model
    units should be physical units, normally micrometers. Constructors that ingest
    masks, points, or CSV tables interpret coordinates as source pixel/voxel
    coordinates and store ``coords = source_coords * spacing``. If coordinates are
    already in model units, pass unit spacing explicitly. ``features`` stores
    precomputed per-detection arrays under canonical string keys such as ``intensity``,
    ``equivalent_diameter_area`` or ``inertia_tensor``. Dataset feature recipes query
    these keys in fixed order and apply any model-specific transforms. ``masks`` is
    optional and only needed for mask-based inference/export.
    ``source_coords`` and ``spacing`` preserve the original coordinate system when
    loaders scale source coordinates into model space. GT matching and lineage
    supervision live in :class:`DetectionSupervision`, not on this object.
    """

    name: str
    n_frames: int
    coords: np.ndarray
    labels: np.ndarray
    timepoints: np.ndarray
    features: dict[str, np.ndarray]
    masks: np.ndarray | None = None
    images: np.ndarray | None = None
    source_coords: np.ndarray | None = None
    spacing: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        coords = _immutable_array(self.coords, ndim=2)
        labels = _immutable_array(self.labels, ndim=1)
        timepoints = _immutable_array(self.timepoints, ndim=1)
        n = len(coords)
        if len(labels) != n or len(timepoints) != n:
            raise ValueError("coords, labels, and timepoints must be aligned")
        if coords.shape[1] not in (2, 3):
            raise ValueError("Detection coordinates must be 2D or 3D")
        if n and np.any(np.diff(timepoints) < 0):
            raise ValueError("timepoints must be sorted in non-decreasing order")
        if n and (timepoints.min() < 0 or timepoints.max() >= int(self.n_frames)):
            raise ValueError("timepoints must lie within [0, n_frames)")
        features = {}
        for name, values in self.features.items():
            name = wrfeat.canonical_feature_name(name)
            if name in features:
                raise ValueError(f"Duplicate canonical feature name {name!r}")
            values = _immutable_array(values)
            if values.ndim == 0 or len(values) != n:
                raise ValueError(f"Feature {name!r} is not aligned with detections")
            features[name] = values
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "n_frames", int(self.n_frames))
        object.__setattr__(self, "coords", coords)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "timepoints", timepoints)
        object.__setattr__(self, "features", features)
        if self.masks is not None:
            masks = _immutable_array(self.masks)
            if len(masks) != int(self.n_frames):
                raise ValueError("masks must have one frame per timepoint")
            object.__setattr__(self, "masks", masks)
        if self.images is not None:
            images = _immutable_array(self.images)
            if len(images) != int(self.n_frames):
                raise ValueError("images must have one frame per timepoint")
            object.__setattr__(self, "images", images)
        if self.source_coords is not None:
            source_coords = _immutable_array(self.source_coords, ndim=2)
            if source_coords.shape != coords.shape:
                raise ValueError("source_coords must have the same shape as coords")
            object.__setattr__(self, "source_coords", source_coords)
        if self.spacing is not None:
            object.__setattr__(
                self, "spacing", validate_spatial_spacing(self.spacing, self.dim)
            )

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def dim(self) -> int:
        return self.coords.shape[1]

    @classmethod
    def from_masks(
        cls,
        imgs: np.ndarray | None,
        masks: np.ndarray,
        *,
        name: str = "detections",
        ndim: int | None = None,
        spacing: tuple[float, ...] | list[float] | np.ndarray | None = None,
        normalize_imgs: bool = True,
        n_workers: int = 0,
        keep_masks: bool = True,
        keep_images: bool = False,
        progbar_class=tqdm,
    ) -> DetectionSequence:
        """Build mask-derived detections and canonical region features.

        Mask centroids and geometry are measured in pixel/voxel coordinates, then
        multiplied by ``spacing`` to produce model-space coordinates and geometry
        features. Use physical voxel size, normally micrometers per axis, when the
        model was trained in physical units. Passing ``spacing=None`` means unit
        spacing; for 3D data this logs a warning because anisotropy is common.
        """
        masks = np.asarray(masks)
        if masks.ndim not in (3, 4):
            raise ValueError(f"masks must have shape (T,(Z),Y,X), got {masks.shape}")
        source_ndim = masks.ndim - 1
        ndim = source_ndim if ndim is None else int(ndim)
        if ndim not in (2, 3):
            raise ValueError("Only 2D and 3D mask detections are supported")
        if ndim == 2 and source_ndim != 2:
            raise ValueError(f"Expected 2D masks, got {source_ndim}D masks")
        if ndim == 3 and source_ndim == 2:
            feature_masks = masks[:, np.newaxis, ...]
        elif ndim == source_ndim:
            feature_masks = masks
        else:
            raise ValueError(f"Expected {ndim}D masks, got {source_ndim}D masks")
        spacing = _resolve_detection_spacing(
            spacing, ndim, context="DetectionSequence.from_masks"
        )
        if imgs is None:
            image_frames = np.zeros(feature_masks.shape, dtype=np.float32)
            has_real_images = False
        else:
            image_frames = np.asarray(imgs)
            if image_frames.shape != masks.shape:
                raise ValueError(
                    f"imgs and masks must have matching shapes, got "
                    f"{image_frames.shape} and {masks.shape}"
                )
            if normalize_imgs:
                image_frames = normalize(image_frames)
            if ndim == 3 and source_ndim == 2:
                image_frames = image_frames[:, np.newaxis, ...]
            has_real_images = True

        def _extract(t: int, mask: np.ndarray, image: np.ndarray) -> wrfeat.WRFeatures:
            return wrfeat.WRFeatures.from_mask_img(
                mask=mask[np.newaxis, ...],
                img=image[np.newaxis, ...],
                t_start=t,
            )

        iterable = progbar_class(
            enumerate(zip(feature_masks, image_frames)),
            total=len(feature_masks),
            desc="Extracting features",
            leave=False,
        )
        if n_workers > 0:
            frame_features_raw = joblib.Parallel(n_jobs=n_workers)(
                joblib.delayed(_extract)(t, mask, image) for t, (mask, image) in iterable
            )
        else:
            frame_features_raw = [
                _extract(t, mask, image) for t, (mask, image) in iterable
            ]
        spacing_matrix = np.diag(spacing).astype(np.float32)
        frame_features = [
            wrfeat.transform_feature_geometry(feature, spacing_matrix)
            for feature in frame_features_raw
        ]
        return cls.from_wrfeatures(
            frame_features=frame_features,
            source_features=frame_features_raw,
            name=name,
            n_frames=len(feature_masks),
            masks=masks if keep_masks else None,
            images=image_frames if keep_images and has_real_images else None,
            spacing=spacing,
            drop_intensity=not has_real_images,
        )

    @classmethod
    def from_wrfeatures(
        cls,
        frame_features: Sequence[wrfeat.WRFeatures],
        *,
        source_features: Sequence[wrfeat.WRFeatures] | None = None,
        name: str = "detections",
        n_frames: int | None = None,
        masks: np.ndarray | None = None,
        images: np.ndarray | None = None,
        spacing: tuple[float, ...] | None = None,
        drop_intensity: bool = False,
    ) -> DetectionSequence:
        """Build detections from one ``WRFeatures`` object per frame."""
        frame_features = tuple(frame_features)
        source_features = tuple(source_features or frame_features)
        if len(frame_features) != len(source_features):
            raise ValueError("frame_features and source_features must have equal length")
        n_frames = len(frame_features) if n_frames is None else int(n_frames)
        if not frame_features:
            if spacing is None:
                raise ValueError("spacing is required when constructing empty detections")
            ndim = len(spacing)
            return cls(
                name=name,
                n_frames=n_frames,
                coords=np.zeros((0, ndim), dtype=np.float32),
                labels=np.zeros(0, dtype=np.int32),
                timepoints=np.zeros(0, dtype=np.int64),
                features={},
                masks=masks,
                images=images,
                source_coords=np.zeros((0, ndim), dtype=np.float32),
                spacing=spacing,
            )

        feature_names = tuple(frame_features[0].features)
        if drop_intensity:
            feature_names = tuple(
                name for name in feature_names if name != wrfeat.FEATURE_INTENSITY
            )
        features = OrderedDict(
            (
                name,
                np.concatenate([feature.features[name] for feature in frame_features]),
            )
            for name in feature_names
        )
        coords = np.concatenate([feature.coords for feature in frame_features])
        source_coords = np.concatenate([feature.coords for feature in source_features])
        labels = np.concatenate([feature.labels for feature in source_features])
        timepoints = np.concatenate(
            [
                np.full(len(feature.labels), t, dtype=np.int64)
                for t, feature in enumerate(source_features)
            ]
        )
        return cls(
            name=name,
            n_frames=n_frames,
            coords=coords,
            labels=labels,
            timepoints=timepoints,
            features=features,
            masks=masks,
            images=images,
            source_coords=source_coords,
            spacing=spacing,
        )

    @classmethod
    def from_points(
        cls,
        coords: Sequence[np.ndarray],
        *,
        name: str = "points",
        features: Sequence[np.ndarray] | dict[str, np.ndarray] | None = None,
        labels: Sequence[np.ndarray] | None = None,
        spacing: tuple[float, ...] | list[float] | np.ndarray | None = None,
    ) -> DetectionSequence:
        """Build detections from per-frame point coordinates.

        Input coordinates are interpreted as source pixel/voxel coordinates and are
        multiplied by ``spacing`` before they are fed to the model. If points are
        already in model units, normally micrometers for microscopy checkpoints, pass
        unit spacing explicitly.
        """
        coords = tuple(np.asarray(frame, dtype=np.float32) for frame in coords)
        if len(coords) < 1:
            raise ValueError("coords must contain at least one frame")
        ndim = coords[0].shape[1] if coords[0].ndim == 2 else 0
        if ndim not in (2, 3):
            raise ValueError("Point coordinates must be 2D or 3D")
        spacing = _resolve_detection_spacing(
            spacing, ndim, context="DetectionSequence.from_points"
        )
        label_frames = []
        for t, frame_coords in enumerate(coords):
            if frame_coords.ndim != 2 or frame_coords.shape[1] != ndim:
                raise ValueError(
                    f"coords[{t}] must have shape (N, {ndim}), got {frame_coords.shape}"
                )
            if labels is None:
                label_frames.append(np.arange(1, len(frame_coords) + 1, dtype=np.int32))
            else:
                frame_labels = np.asarray(labels[t])
                if frame_labels.ndim != 1 or len(frame_labels) != len(frame_coords):
                    raise ValueError(
                        f"labels[{t}] must have shape ({len(frame_coords)},), "
                        f"got {frame_labels.shape}"
                    )
                if len(np.unique(frame_labels)) != len(frame_labels):
                    raise ValueError(f"labels[{t}] must be unique within the frame")
                label_frames.append(frame_labels.astype(np.int32, copy=False))
        source_coords = np.concatenate(coords) if coords else np.zeros((0, ndim))
        model_coords = apply_spatial_spacing(source_coords, spacing)
        timepoints = np.concatenate(
            [np.full(len(frame), t, dtype=np.int64) for t, frame in enumerate(coords)]
        )
        flat_labels = np.concatenate(label_frames)
        feature_dict: OrderedDict[str, np.ndarray] = OrderedDict()
        if features is not None:
            if isinstance(features, dict):
                for name, values in features.items():
                    feature_dict[name] = np.asarray(values, dtype=np.float32)
            else:
                feature_dict[wrfeat.FEATURE_CUSTOM] = np.concatenate(
                    [np.asarray(values, dtype=np.float32) for values in features]
                )
        return cls(
            name=name,
            n_frames=len(coords),
            coords=model_coords,
            labels=flat_labels,
            timepoints=timepoints,
            features=feature_dict,
            source_coords=source_coords,
            spacing=spacing,
        )

    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        *,
        name: str = "detections",
        coord_columns: Sequence[str] = ("z", "y", "x"),
        time_column: str = "t",
        label_column: str | None = None,
        feature_columns: Sequence[str] | None = None,
        spacing: tuple[float, ...] | list[float] | np.ndarray | None = None,
    ) -> DetectionSequence:
        """Build detections from a point-detection table.

        Coordinate columns are interpreted as source pixel/voxel coordinates and are
        multiplied by ``spacing`` before tracking. If the table already stores model
        units, normally micrometers for microscopy checkpoints, pass unit spacing
        explicitly.
        """
        ndim = len(tuple(coord_columns))
        time_column, coord_columns = _resolve_point_detection_columns(
            df, time_column=time_column, coord_columns=coord_columns, ndim=ndim
        )
        feature_columns = _resolve_point_feature_columns(
            df, feature_columns, excluded=(time_column, *coord_columns)
        )
        coords, source_coords, timepoints, features = _point_detection_arrays(
            df,
            spacing=spacing,
            coord_columns=coord_columns,
            time_column=time_column,
            feature_columns=feature_columns,
        )
        order = np.argsort(df[time_column].to_numpy(dtype=np.int64), kind="stable")
        if label_column is None:
            labels = np.empty(len(coords), dtype=np.int32)
            for time in np.unique(timepoints):
                idx = np.flatnonzero(timepoints == time)
                labels[idx] = np.arange(1, len(idx) + 1, dtype=np.int32)
        else:
            if label_column not in df:
                raise ValueError(f"Point detections are missing column {label_column!r}")
            labels = df[label_column].to_numpy(dtype=np.int32)[order]
        n_frames = int(timepoints.max()) + 1 if len(timepoints) else 0
        return cls(
            name=name,
            n_frames=n_frames,
            coords=coords,
            labels=labels,
            timepoints=timepoints,
            features=features,
            source_coords=source_coords,
            spacing=_resolve_detection_spacing(
                spacing, ndim, context="DetectionSequence.from_df"
            ),
        )

    @classmethod
    def from_csv(cls, path: str | Path, **kwargs) -> DetectionSequence:
        """Build detections from a CSV point table."""
        path = Path(path).expanduser()
        return cls.from_df(pd.read_csv(path), name=kwargs.pop("name", path.stem), **kwargs)

    def to_wrfeatures(self) -> list[wrfeat.WRFeatures]:
        """Return one ``WRFeatures`` object per frame."""
        frames = []
        for t in range(self.n_frames):
            idx = np.flatnonzero(self.timepoints == t)
            frames.append(
                wrfeat.WRFeatures(
                    coords=self.coords[idx],
                    labels=self.labels[idx],
                    timepoints=self.timepoints[idx].astype(np.int32, copy=False),
                    features=OrderedDict(
                        (name, values[idx]) for name, values in self.features.items()
                    ),
                )
            )
        return frames

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
                self.masks,
                self.images,
                self.source_coords,
                self.spacing,
            ),
        )


@dataclass(frozen=True)
class DetectionSupervision:
    """Ground-truth alignment for one :class:`DetectionSequence`.

    A :class:`LineageGraph` describes the GT lineage in the abstract, but the
    model is trained on concrete detection streams (GT masks, predicted masks,
    point proposals, ...). This class is the bridge: for each detection in one
    stream it records which GT tracklet that detection corresponds to, so the
    per-tracklet ``lineage_relation`` can be projected onto detection pairs.

    Every array here is per-detection, length M = number of detections in that
    stream, aligned one-to-one with them. A stream with no GT (pure inference)
    has no ``DetectionSupervision``.

    Attributes:
        lineage_index: (M,) for each detection, the GT tracklet it matches, given
            as an index into the tracklet dimension of
            :attr:`LineageGraph.lineage_relation`. ``-1`` means the detection has
            no GT match (e.g. a predicted object with no annotated counterpart)
            and is excluded from the association target. Many detections of the
            same cell across frames share one ``lineage_index``. This is the only
            field the training target strictly requires.
        gt_node_index: (M,) for each detection, the matched GT node index in
            :class:`LineageGraph`, or ``-1`` for unmatched detections. This is
            aligned to GT-node arrays such as ``node_in_degree``.
        matched_gt: (M,) ``True`` where the detection was matched to a GT object.
            Redundant: equal to ``lineage_index >= 0``. ``None`` if unused.
        gt_predecessor_set_available: (M,) ``True`` where the complete set of
            incoming GT links is known for that detection. ``False`` at annotation
            boundaries, e.g. a track's first annotated frame under sparse GT,
            where a link just outside the annotated span cannot be ruled out.
            Lets those pairs be masked from the loss rather than scored as
            "no link". ``None`` if unused.
        gt_successor_set_available: (M,) the same for outgoing GT links (e.g.
            ``False`` at a track's last annotated frame). ``None`` if unused.
    """

    lineage_index: np.ndarray
    gt_node_index: np.ndarray | None = None
    matched_gt: np.ndarray | None = None
    gt_predecessor_set_available: np.ndarray | None = None
    gt_successor_set_available: np.ndarray | None = None

    def __post_init__(self) -> None:
        lineage_index = _immutable_array(self.lineage_index, ndim=1)
        if np.any(lineage_index < -1):
            raise ValueError("lineage_index may only use -1 for unmatched detections")
        object.__setattr__(self, "lineage_index", lineage_index)
        n = len(lineage_index)
        if self.gt_node_index is not None:
            gt_node_index = _immutable_array(self.gt_node_index, ndim=1).astype(np.int64)
            if len(gt_node_index) != n:
                raise ValueError("gt_node_index must be aligned with lineage_index")
            if np.any(gt_node_index < -1):
                raise ValueError("gt_node_index may only use -1 for unmatched detections")
            gt_node_index.setflags(write=False)
            object.__setattr__(self, "gt_node_index", gt_node_index)
        for field in (
            "matched_gt",
            "gt_predecessor_set_available",
            "gt_successor_set_available",
        ):
            values = getattr(self, field)
            if values is None:
                continue
            array = _immutable_array(values, ndim=1).astype(bool)
            if len(array) != n:
                raise ValueError(f"{field} must be aligned with lineage_index")
            array.setflags(write=False)
            object.__setattr__(self, field, array)

    def __len__(self) -> int:
        return len(self.lineage_index)

    def __reduce__(self):
        return (
            type(self),
            (
                self.lineage_index,
                self.gt_node_index,
                self.matched_gt,
                self.gt_predecessor_set_available,
                self.gt_successor_set_available,
            ),
        )


@dataclass(frozen=True)
class LineageGraph:
    """Ground-truth cell lineage for one sequence, shared by its detection streams.

    In cell tracking, objects divide over time: a mother cell splits into two
    daughters, each of which may split again. A *lineage* is the whole family
    tree descending from one founder cell. Trackastra learns to associate
    detections that belong to the same lineage, so this class holds the GT that
    supervision is built from.

    Terminology used throughout:
        - detection: one segmented object in one frame.
        - tracklet: one cell's chain of detections across consecutive frames,
          from when it appears until it divides, dies, or leaves.
        - lineage: the whole family tree of tracklets connected by divisions
          (a founder cell, its daughters, granddaughters, ...).

    This is a *tracklet-collapsed* view of the source graph, not the raw GT
    tracking graph. The loaders (``from_ctc`` / ``from_geff``) start from an
    explicit node-to-node graph, cut it into linear tracklets at divisions, and
    keep only the summary below. The explicit "which detection links to which"
    edges are NOT stored here; within a tracklet they are implicit (consecutive
    ``timepoints`` of the same tracklet form a chain) and at divisions they are
    recoverable from ``lineage_parents``. Evaluation re-loads the full edge graph
    from the source files rather than from this object.

    The arrays come at two different granularities; do not mix them.

    Per-tracklet arrays (size T = number of tracklets) hold the actual lineage
    topology and are the only part the training target uses. A detection reaches
    them via a tracklet index (``lineage_index`` in
    :class:`DetectionSupervision`), not by row position:
        lineage_relation: (T, T) symmetric bool. ``lineage_relation[i, j]`` is
            ``True`` iff tracklets i and j belong to the same lineage tree
            (ancestor/descendant across any number of divisions, plus the
            diagonal). This is the "same cell or relatives" association target.
        lineage_parents: (T,) parent tracklet index of each tracklet, or ``-1``
            for a founder with no parent. Two tracklets sharing a parent are the
            daughters of one division. This is the only record of division
            structure; note the current training loss does not consume it yet.

    Per-detection arrays (length N = number of GT detections, all row-aligned)
    are a self-contained snapshot of the GT detections. They exist so GT can be
    matched to a detection stream and traced back / visualized:
        coords: (N, ndim) object centroids, scaled by ``spacing``.
        source_coords: (N, ndim) the same centroids in raw pixel units, before
            ``spacing`` is applied. ``None`` if not provided.
        timepoints: (N,) frame index of each detection.
        node_ids: (N,) a stable identifier per detection (e.g. a ``(t, label)``
            pair or a source graph node id) to trace it back to its origin.
        node_in_degree: (N,) observed incoming direct GT edges per node.
        node_out_degree: (N,) observed outgoing direct GT edges per node.
        node_predecessor_set_available: (N,) ``True`` where the complete incoming
            GT edge set is known for that node. ``None`` if unavailable.
        node_successor_set_available: (N,) same for outgoing GT edges.

    Attributes:
        spacing: physical pixel spacing per spatial axis, or ``None`` for unit
            spacing. Only applied at construction to produce ``coords`` from
            ``source_coords``; nothing reads it afterward (it is recoverable as
            ``coords / source_coords``), so it is retained purely as provenance.
    """

    coords: np.ndarray
    timepoints: np.ndarray
    node_ids: np.ndarray
    lineage_relation: np.ndarray
    lineage_parents: np.ndarray
    spacing: tuple[float, ...] | None = None
    source_coords: np.ndarray | None = None
    node_in_degree: np.ndarray | None = None
    node_out_degree: np.ndarray | None = None
    node_predecessor_set_available: np.ndarray | None = None
    node_successor_set_available: np.ndarray | None = None

    def __post_init__(self) -> None:
        coords = _immutable_array(self.coords, ndim=2)
        timepoints = _immutable_array(self.timepoints, ndim=1)
        node_ids = _immutable_array(self.node_ids, ndim=1)
        if len(coords) != len(timepoints) or len(coords) != len(node_ids):
            raise ValueError("coords, timepoints, and node_ids must be aligned")
        relation = _immutable_array(self.lineage_relation, ndim=2).astype(bool)
        parents = _immutable_array(self.lineage_parents, ndim=1)
        if relation.shape[0] != relation.shape[1] or len(parents) != len(relation):
            raise ValueError("Lineage arrays must describe the same tracklets")
        if not np.array_equal(relation, relation.T):
            raise ValueError("lineage_relation must be symmetric")
        if len(relation) and not np.all(np.diag(relation)):
            raise ValueError("Each tracklet must be related to itself")
        relation.setflags(write=False)
        parents.setflags(write=False)
        object.__setattr__(self, "coords", coords)
        object.__setattr__(self, "timepoints", timepoints)
        object.__setattr__(self, "node_ids", node_ids)
        object.__setattr__(self, "lineage_relation", relation)
        object.__setattr__(self, "lineage_parents", parents)
        n_nodes = len(coords)
        for field in ("node_in_degree", "node_out_degree"):
            values = getattr(self, field)
            if values is None:
                continue
            array = _immutable_array(values, ndim=1).astype(np.int64)
            if len(array) != n_nodes:
                raise ValueError(f"{field} must be aligned with GT nodes")
            if np.any(array < 0):
                raise ValueError(f"{field} must be non-negative")
            array.setflags(write=False)
            object.__setattr__(self, field, array)
        for field in ("node_predecessor_set_available", "node_successor_set_available"):
            values = getattr(self, field)
            if values is None:
                continue
            array = _immutable_array(values, ndim=1).astype(bool)
            if len(array) != n_nodes:
                raise ValueError(f"{field} must be aligned with GT nodes")
            array.setflags(write=False)
            object.__setattr__(self, field, array)
        if self.spacing is not None:
            object.__setattr__(
                self, "spacing", validate_spatial_spacing(self.spacing, coords.shape[1])
            )
        if self.source_coords is not None:
            source_coords = _immutable_array(self.source_coords, ndim=2)
            if source_coords.shape != coords.shape:
                raise ValueError("source_coords must have the same shape as coords")
            object.__setattr__(self, "source_coords", source_coords)

    def __reduce__(self):
        return (
            type(self),
            (
                self.coords,
                self.timepoints,
                self.node_ids,
                self.lineage_relation,
                self.lineage_parents,
                self.spacing,
                self.source_coords,
                self.node_in_degree,
                self.node_out_degree,
                self.node_predecessor_set_available,
                self.node_successor_set_available,
            ),
        )


@dataclass(frozen=True)
class TrackingSequence:
    """Tracking data for one movie.

    ``detections`` contains one or more detection series for the same movie, for
    example GT masks, predicted masks, point proposals, or alternative detector
    outputs. ``supervision`` stores the optional GT alignment for each detection stream.
    """

    root: Path
    ndim: int
    detections: tuple[DetectionSequence, ...]
    gt: LineageGraph | None = None
    supervision: tuple[DetectionSupervision | None, ...] | None = None
    images: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.ndim not in (2, 3):
            raise ValueError("Only 2D and 3D tracking data is supported")
        detections = tuple(self.detections)
        supervision = (
            (None,) * len(detections)
            if self.supervision is None
            else tuple(self.supervision)
        )
        if len(supervision) != len(detections):
            raise ValueError("supervision must align with detections")
        for det in detections:
            if det.coords.shape[1] != self.ndim:
                raise ValueError("DetectionSequence dimensionality does not match sequence")
        if self.gt is not None:
            if self.gt.coords.shape[1] != self.ndim:
                raise ValueError("LineageGraph dimensionality does not match sequence")
            for det, sup in zip(detections, supervision):
                if sup is None:
                    continue
                if len(sup) != len(det):
                    raise ValueError("DetectionSupervision must align with detections")
                if np.any(sup.lineage_index >= len(self.gt.lineage_relation)):
                    raise ValueError("DetectionSupervision references an unknown lineage")
                if sup.gt_node_index is not None and np.any(
                    sup.gt_node_index >= len(self.gt.coords)
                ):
                    raise ValueError("DetectionSupervision references an unknown GT node")
        elif any(sup is not None for sup in supervision):
            raise ValueError("supervision requires a gt LineageGraph")
        object.__setattr__(self, "root", Path(self.root))
        object.__setattr__(self, "detections", detections)
        object.__setattr__(self, "supervision", supervision)
        if self.images is not None:
            object.__setattr__(self, "images", _immutable_array(self.images))

    def __reduce__(self):
        return (
            type(self),
            (
                self.root,
                self.ndim,
                self.detections,
                self.gt,
                self.supervision,
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
        spacing: tuple[float, ...] | list[float] | np.ndarray | None = None,
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
            spacing,
        )

    @classmethod
    def from_geff(
        cls,
        root_or_geff: str | Path,
        detections: str
        | Path
        | pd.DataFrame
        | Sequence[str | Path | pd.DataFrame]
        | None = None,
        spacing: tuple[float, ...] | str | None = None,
        coord_columns: Sequence[str] = ("z", "y", "x"),
        time_column: str = "t",
        detection_coord_columns: Sequence[str] | None = None,
        detection_time_column: str | None = None,
        feature_columns: Sequence[str] | None = None,
        match_max_distance: float | None = None,
        sparse_gt: bool = False,
    ) -> TrackingSequence:
        """Load GT point detections and lineage edges from a GEFF graph.

        ``root_or_geff`` may point directly at a ``.geff`` store or at a directory
        containing exactly one ``.geff`` store. In the directory case any ``*.csv``
        files found alongside it are loaded as proposal detections (one
        :class:`DetectionSet` each) unless ``detections`` is given explicitly.
        Explicit ``detections`` may be a single source or a sequence of sources.

        ``spacing=None`` uses unit spacing. ``spacing="auto"`` requires GEFF axis
        scales and uses them.
        """
        import geff

        geff_path, detection_sources = _resolve_geff_paths(root_or_geff, detections)
        graph, metadata = geff.read(geff_path)
        return _tracking_sequence_from_geff_graph(
            cls,
            graph,
            metadata,
            geff_path,
            detection_sources=detection_sources,
            spacing=spacing,
            coord_columns=coord_columns,
            time_column=time_column,
            detection_coord_columns=detection_coord_columns,
            detection_time_column=detection_time_column,
            feature_columns=feature_columns,
            match_max_distance=match_max_distance,
            sparse_gt=sparse_gt,
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


def _ctc_node_index(masks: np.ndarray) -> dict[tuple[int, int], int]:
    index = {}
    for t, mask in enumerate(masks):
        for prop in regionprops(mask):
            index[(t, int(prop.label))] = len(index)
    return index


def _ctc_node_degrees(
    masks: np.ndarray, tracks: pd.DataFrame | None
) -> tuple[np.ndarray, np.ndarray]:
    node_index = _ctc_node_index(masks)
    in_degree = np.zeros(len(node_index), dtype=np.int64)
    out_degree = np.zeros(len(node_index), dtype=np.int64)
    if tracks is None:
        return in_degree, out_degree

    tracks_by_label = {int(row.label): row for row in tracks.itertuples()}
    for row in tracks.itertuples():
        label = int(row.label)
        for t in range(int(row.t1), int(row.t2)):
            source = node_index.get((t, label))
            target = node_index.get((t + 1, label))
            if source is None or target is None:
                continue
            out_degree[source] += 1
            in_degree[target] += 1
        if row.parent == 0:
            continue
        parent = tracks_by_label.get(int(row.parent))
        if parent is None:
            continue
        source = node_index.get((int(parent.t2), int(row.parent)))
        target = node_index.get((int(row.t1), label))
        if source is None or target is None:
            continue
        out_degree[source] += 1
        in_degree[target] += 1
    return in_degree, out_degree


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


def _lineage_graph_from_masks(
    masks: np.ndarray,
    lineage_relation: np.ndarray,
    lineage_parents: np.ndarray,
    spacing: tuple[float, ...],
    node_in_degree: np.ndarray | None = None,
    node_out_degree: np.ndarray | None = None,
    node_predecessor_set_available: np.ndarray | None = None,
    node_successor_set_available: np.ndarray | None = None,
) -> LineageGraph:
    source_coords = []
    timepoints = []
    node_ids = []
    for t, mask in enumerate(masks):
        for prop in regionprops(mask):
            source_coords.append(prop.centroid)
            timepoints.append(t)
            node_ids.append((t, int(prop.label)))
    ndim = masks.ndim - 1
    if source_coords:
        source_coords = np.asarray(source_coords, dtype=np.float32)
        coords = apply_spatial_spacing(source_coords, spacing)
        timepoints = np.asarray(timepoints, dtype=np.int64)
        node_id_array = np.empty(len(node_ids), dtype=object)
        node_id_array[:] = node_ids
        node_ids = node_id_array
    else:
        source_coords = np.zeros((0, ndim), dtype=np.float32)
        coords = np.zeros((0, ndim), dtype=np.float32)
        timepoints = np.zeros(0, dtype=np.int64)
        node_ids = np.zeros(0, dtype=object)
    return LineageGraph(
        coords=coords,
        timepoints=timepoints,
        node_ids=node_ids,
        lineage_relation=lineage_relation,
        lineage_parents=lineage_parents,
        spacing=spacing,
        source_coords=source_coords,
        node_in_degree=node_in_degree,
        node_out_degree=node_out_degree,
        node_predecessor_set_available=node_predecessor_set_available,
        node_successor_set_available=node_successor_set_available,
    )


def _spacing_from_geff_metadata(
    metadata, coord_columns: Sequence[str]
) -> tuple[float, ...]:
    axes = getattr(metadata, "axes", None)
    if axes is None:
        raise ValueError("GEFF metadata has no axes to read spacing from")
    scales = {axis.name: getattr(axis, "scale", None) for axis in axes}
    missing = [name for name in coord_columns if name not in scales]
    if missing:
        raise ValueError(f"GEFF metadata is missing axes {missing}")
    values = tuple(
        1.0 if scales[name] is None else float(scales[name])
        for name in coord_columns
    )
    return values


def _geff_tracklet_assignments(
    graph: nx.DiGraph,
    node_times: dict,
) -> tuple[dict, np.ndarray, np.ndarray]:
    nodes = sorted(graph.nodes, key=lambda node: (node_times[node], node))
    lineage_index = {}
    lineage_parents = []

    for node in nodes:
        if node in lineage_index:
            continue
        parent_lineage = -1
        predecessors = list(graph.predecessors(node))
        if len(predecessors) == 1 and predecessors[0] in lineage_index:
            parent_lineage = lineage_index[predecessors[0]]
        lineage_id = len(lineage_parents)
        lineage_parents.append(parent_lineage)

        current = node
        while current not in lineage_index:
            lineage_index[current] = lineage_id
            successors = list(graph.successors(current))
            if len(successors) != 1:
                break
            successor = successors[0]
            if len(list(graph.predecessors(successor))) != 1:
                break
            if graph.out_degree(current) != 1:
                break
            current = successor

    parents = np.asarray(lineage_parents, dtype=np.int64)
    lineage = np.eye(len(parents), dtype=bool)
    lineage_graph = nx.DiGraph()
    lineage_graph.add_nodes_from(range(len(parents)))
    lineage_graph.add_edges_from(
        (int(parent), i) for i, parent in enumerate(parents) if parent >= 0
    )
    for lineage_id in range(len(parents)):
        relatives = nx.ancestors(lineage_graph, lineage_id) | nx.descendants(
            lineage_graph, lineage_id
        )
        lineage[lineage_id, list(relatives)] = True
    return lineage_index, lineage, parents


def _geff_gt_link_set_availability(
    graph: nx.DiGraph, node_times: dict
) -> tuple[dict, dict]:
    predecessor_available = {}
    successor_available = {}
    for component in nx.weakly_connected_components(graph):
        times = np.asarray([node_times[node] for node in component], dtype=np.int64)
        t_min = int(times.min())
        t_max = int(times.max())
        for node in component:
            time = int(node_times[node])
            predecessor_available[node] = time > t_min
            successor_available[node] = time < t_max
    return predecessor_available, successor_available


def _normalize_detection_sources(
    detections: str | Path | pd.DataFrame | Sequence | None,
) -> list:
    """Normalize the ``detections`` argument into a flat list of sources."""
    if detections is None:
        return []
    if isinstance(detections, (str, Path, pd.DataFrame)):
        return [detections]
    return list(detections)


def _resolve_geff_paths(
    root_or_geff: str | Path,
    detections: str | Path | pd.DataFrame | Sequence | None,
) -> tuple[Path, list]:
    """Resolve a ``.geff`` store and its detection sources from a path.

    ``root_or_geff`` is treated as a store when it ends in ``.geff``; otherwise,
    if it is a directory, it must hold exactly one ``.geff`` store, and its
    ``*.csv`` files become detection sources unless ``detections`` is passed
    explicitly.
    """
    path = Path(root_or_geff).expanduser()
    is_store = path.suffix == ".geff"
    if is_store or not path.is_dir():
        return path, _normalize_detection_sources(detections)
    stores = sorted(path.glob("*.geff"))
    if len(stores) != 1:
        raise FileNotFoundError(
            f"Expected exactly one .geff store in {path}, found {len(stores)}"
        )
    if detections is None:
        return stores[0], sorted(path.glob("*.csv"))
    return stores[0], _normalize_detection_sources(detections)


def _detection_source_name(source: str | Path | pd.DataFrame, index: int) -> str:
    if isinstance(source, (str, Path)):
        return Path(source).stem
    return f"detections_{index}"


def _build_geff_detection_sequence(
    name: str,
    coords: np.ndarray,
    source_coords: np.ndarray,
    timepoints: np.ndarray,
    features,
    spacing: tuple[float, ...],
) -> DetectionSequence:
    labels = np.empty(len(coords), dtype=np.int32)
    for time in np.unique(timepoints):
        idx = np.flatnonzero(timepoints == time)
        labels[idx] = np.arange(1, len(idx) + 1, dtype=np.int32)
    n_frames = int(timepoints.max()) + 1 if len(timepoints) else 0
    return DetectionSequence(
        name=name,
        n_frames=n_frames,
        coords=coords,
        labels=labels,
        timepoints=timepoints,
        features=features,
        source_coords=source_coords,
        spacing=spacing,
    )


def _tracking_sequence_from_geff_graph(
    sequence_type: type[TrackingSequence],
    graph: nx.DiGraph,
    metadata,
    root: Path,
    detection_sources: Sequence[str | Path | pd.DataFrame],
    spacing: tuple[float, ...] | str | None,
    coord_columns: Sequence[str],
    time_column: str,
    detection_coord_columns: Sequence[str] | None,
    detection_time_column: str | None,
    feature_columns: Sequence[str] | None,
    match_max_distance: float | None,
    sparse_gt: bool,
) -> TrackingSequence:
    coord_columns = tuple(coord_columns)
    ndim = len(coord_columns)
    if ndim not in (2, 3):
        raise ValueError("GEFF point coordinates must be 2D or 3D")
    if spacing is None:
        spacing = (1.0,) * ndim
    elif spacing == "auto":
        spacing = _spacing_from_geff_metadata(metadata, coord_columns)
    elif isinstance(spacing, str):
        raise ValueError("spacing must be None, a numeric tuple, or 'auto'")

    node_times = {}
    node_coords = {}
    for node, attrs in graph.nodes(data=True):
        missing = [name for name in (time_column, *coord_columns) if name not in attrs]
        if missing:
            raise ValueError(f"GEFF node {node!r} is missing properties {missing}")
        node_times[node] = int(attrs[time_column])
        node_coords[node] = [float(attrs[name]) for name in coord_columns]

    nodes = sorted(graph.nodes, key=lambda node: (node_times[node], node))
    if not nodes:
        source_coords = np.zeros((0, ndim), dtype=np.float32)
        detection_sequence = _build_geff_detection_sequence(
            name=Path(root).name,
            coords=apply_spatial_spacing(source_coords, spacing),
            source_coords=source_coords,
            timepoints=np.zeros(0, dtype=np.int64),
            features={},
            spacing=spacing,
        )
        gt = LineageGraph(
            coords=np.zeros((0, ndim), dtype=np.float32),
            timepoints=np.zeros(0, dtype=np.int64),
            node_ids=np.zeros(0, dtype=object),
            lineage_relation=np.zeros((0, 0), dtype=bool),
            lineage_parents=np.zeros(0, dtype=np.int64),
            spacing=spacing,
            source_coords=np.zeros((0, ndim), dtype=np.float32),
            node_in_degree=np.zeros(0, dtype=np.int64),
            node_out_degree=np.zeros(0, dtype=np.int64),
            node_predecessor_set_available=np.zeros(0, dtype=bool),
            node_successor_set_available=np.zeros(0, dtype=bool),
        )
        return sequence_type(
            root=root,
            ndim=ndim,
            detections=(detection_sequence,),
            gt=gt,
            supervision=(
                DetectionSupervision(
                    lineage_index=np.zeros(0, dtype=np.int64),
                    gt_node_index=np.zeros(0, dtype=np.int64),
                    matched_gt=np.zeros(0, dtype=bool),
                ),
            ),
        )

    gt_source_coords = np.asarray([node_coords[node] for node in nodes], dtype=np.float32)
    gt_coords = apply_spatial_spacing(gt_source_coords, spacing)
    gt_timepoints = np.asarray([node_times[node] for node in nodes], dtype=np.int64)
    lineage_map, lineage_relation, lineage_parents = _geff_tracklet_assignments(
        graph, node_times
    )
    gt_predecessor_available, gt_successor_available = _geff_gt_link_set_availability(
        graph, node_times
    )
    gt_node_in_degree = np.asarray(
        [graph.in_degree(node) for node in nodes], dtype=np.int64
    )
    gt_node_out_degree = np.asarray(
        [graph.out_degree(node) for node in nodes], dtype=np.int64
    )
    gt_node_predecessor_set_available = np.asarray(
        [gt_predecessor_available[node] for node in nodes], dtype=bool
    )
    gt_node_successor_set_available = np.asarray(
        [gt_successor_available[node] for node in nodes], dtype=bool
    )
    gt = LineageGraph(
        coords=gt_coords,
        timepoints=gt_timepoints,
        node_ids=np.asarray(nodes, dtype=object),
        lineage_relation=lineage_relation,
        lineage_parents=lineage_parents,
        spacing=spacing,
        source_coords=gt_source_coords,
        node_in_degree=gt_node_in_degree,
        node_out_degree=gt_node_out_degree,
        node_predecessor_set_available=gt_node_predecessor_set_available,
        node_successor_set_available=gt_node_successor_set_available,
    )

    detection_sequences = []
    supervision = []
    if not detection_sources:
        lineage_index = np.asarray([lineage_map[node] for node in nodes], dtype=np.int64)
        detection_sequences.append(
            _build_geff_detection_sequence(
                name=Path(root).name,
                coords=gt_coords,
                source_coords=gt_source_coords,
                timepoints=gt_timepoints,
                features=_geff_node_features(graph, nodes, feature_columns),
                spacing=spacing,
            )
        )
        supervision.append(
            DetectionSupervision(
                lineage_index=lineage_index,
                gt_node_index=np.arange(len(nodes), dtype=np.int64),
                matched_gt=np.ones(len(gt_coords), dtype=bool),
            )
        )
    else:
        if match_max_distance is None:
            raise ValueError(
                "match_max_distance is required when detections are provided"
            )
        for index, source in enumerate(detection_sources):
            detections_df = _read_point_detections(source)
            src_time_column, src_coord_columns = _resolve_point_detection_columns(
                detections_df,
                time_column=detection_time_column or time_column,
                coord_columns=detection_coord_columns or coord_columns,
                ndim=ndim,
            )
            src_feature_columns = _resolve_point_feature_columns(
                detections_df,
                feature_columns,
                excluded=(src_time_column, *src_coord_columns),
            )
            coords, source_coords, timepoints, features = _point_detection_arrays(
                detections_df,
                spacing=spacing,
                coord_columns=src_coord_columns,
                time_column=src_time_column,
                feature_columns=src_feature_columns,
            )
            lineage_index = np.full(len(coords), -1, dtype=np.int64)
            gt_node_index = np.full(len(coords), -1, dtype=np.int64)
            gt_predecessor_set_available = np.ones(len(coords), dtype=bool)
            gt_successor_set_available = np.ones(len(coords), dtype=bool)
            if sparse_gt:
                gt_predecessor_set_available[:] = False
                gt_successor_set_available[:] = False
            for time in np.intersect1d(np.unique(timepoints), np.unique(gt_timepoints)):
                prop_idx = np.flatnonzero(timepoints == time)
                gt_idx = np.flatnonzero(gt_timepoints == time)
                matches = match_points(
                    coords[prop_idx],
                    gt_coords[gt_idx],
                    max_distance=match_max_distance,
                )
                for prop_local, gt_local, _distance in matches:
                    prop_global = prop_idx[prop_local]
                    gt_global = gt_idx[gt_local]
                    gt_node = nodes[gt_idx[gt_local]]
                    lineage_index[prop_global] = lineage_map[gt_node]
                    gt_node_index[prop_global] = gt_global
                    if sparse_gt:
                        gt_predecessor_set_available[prop_global] = (
                            gt_predecessor_available[gt_node]
                        )
                        gt_successor_set_available[prop_global] = (
                            gt_successor_available[gt_node]
                        )
            matched_gt = (
                lineage_index >= 0 if sparse_gt else np.ones(len(coords), dtype=bool)
            )
            detection_sequences.append(
                _build_geff_detection_sequence(
                    name=_detection_source_name(source, index),
                    coords=coords,
                    source_coords=source_coords,
                    timepoints=timepoints,
                    features=features,
                    spacing=spacing,
                )
            )
            supervision.append(
                DetectionSupervision(
                    lineage_index=lineage_index,
                    gt_node_index=gt_node_index,
                    matched_gt=matched_gt,
                    gt_predecessor_set_available=gt_predecessor_set_available,
                    gt_successor_set_available=gt_successor_set_available,
                )
            )

    return sequence_type(
        root=root,
        ndim=ndim,
        detections=tuple(detection_sequences),
        gt=gt,
        supervision=tuple(supervision),
    )


def _geff_node_features(
    graph: nx.DiGraph,
    nodes: Sequence,
    feature_columns: Sequence[str] | None,
) -> OrderedDict:
    features = OrderedDict()
    columns = tuple(feature_columns or ())
    for name in columns:
        values = []
        for node in nodes:
            attrs = graph.nodes[node]
            if name not in attrs:
                raise ValueError(f"GEFF node {node!r} is missing feature {name!r}")
            values.append(float(attrs[name]))
        values = np.asarray(values, dtype=np.float32)[:, None]
        canonical = wrfeat.canonical_feature_name(name)
        if canonical in features:
            raise ValueError(f"Duplicate canonical feature name {canonical!r}")
        features[canonical] = values
    return features


def _read_point_detections(detections: str | Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(detections, pd.DataFrame):
        return detections.copy()
    return pd.read_csv(Path(detections).expanduser())


def _resolve_point_feature_columns(
    detections: pd.DataFrame,
    feature_columns: Sequence[str] | None,
    excluded: Sequence[str],
) -> tuple[str, ...]:
    if feature_columns is not None:
        return tuple(feature_columns)
    excluded = set(excluded)
    known = {
        wrfeat.FEATURE_AREA,
        wrfeat.FEATURE_BORDER_DIST,
        wrfeat.FEATURE_DIAMETER,
        wrfeat.FEATURE_INERTIA,
        wrfeat.FEATURE_INTENSITY,
    }
    columns = []
    seen = set()
    for column in detections.columns:
        if column in excluded:
            continue
        canonical = wrfeat.canonical_feature_name(column)
        if canonical not in known:
            continue
        if canonical in seen:
            raise ValueError(f"Duplicate canonical feature name {canonical!r}")
        columns.append(column)
        seen.add(canonical)
    return tuple(columns)


def _resolve_point_detection_columns(
    detections: pd.DataFrame,
    time_column: str,
    coord_columns: Sequence[str],
    ndim: int,
) -> tuple[str, tuple[str, ...]]:
    coord_columns = tuple(coord_columns)
    if time_column in detections and all(name in detections for name in coord_columns):
        return time_column, coord_columns

    axis_time = "axis-0"
    axis_coords = tuple(f"axis-{i}" for i in range(1, ndim + 1))
    if axis_time in detections and all(name in detections for name in axis_coords):
        return axis_time, axis_coords

    missing = [
        name
        for name in (time_column, *coord_columns)
        if name not in detections
    ]
    raise ValueError(
        f"Point detections are missing columns {missing}; also tried "
        f"{(axis_time, *axis_coords)}"
    )


def _point_detection_arrays(
    detections: pd.DataFrame,
    spacing: tuple[float, ...] | None,
    coord_columns: Sequence[str],
    time_column: str,
    feature_columns: Sequence[str] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, OrderedDict]:
    missing = [
        name
        for name in (time_column, *coord_columns, *(feature_columns or ()))
        if name not in detections
    ]
    if missing:
        raise ValueError(f"Point detections are missing columns {missing}")
    timepoints = detections[time_column].to_numpy(dtype=np.int64)
    order = np.argsort(timepoints, kind="stable")
    timepoints = timepoints[order]
    source_coords = detections.loc[:, coord_columns].to_numpy(dtype=np.float32)[order]
    coords = apply_spatial_spacing(source_coords, spacing)
    features = OrderedDict()
    for name in feature_columns or ():
        values = detections.loc[:, [name]].to_numpy(dtype=np.float32)[order]
        canonical = wrfeat.canonical_feature_name(name)
        if canonical in features:
            raise ValueError(f"Duplicate canonical feature name {canonical!r}")
        features[canonical] = values
    return coords, source_coords, timepoints, features


def _matching_with_spacing(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float | None,
    max_distance: float,
    spacing: tuple[float, ...],
) -> tuple[tuple[int, int], ...]:
    y_true, y_pred = y_true.astype(np.int32), y_pred.astype(np.int32)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true ({y_true.shape}) and y_pred ({y_pred.shape}) have different shapes"
        )
    if threshold is None:
        threshold = 0

    y_true, _, map_rev_true = relabel_sequential(y_true)
    y_pred, _, map_rev_pred = relabel_sequential(y_pred)
    overlap = label_overlap(y_true, y_pred, check=False)
    scores_iou = intersection_over_union(overlap)
    n_true, n_pred = scores_iou.shape
    n_matched = min(n_true, n_pred)
    if n_matched == 0:
        return ()

    true_coords = apply_spatial_spacing(
        np.asarray([r.centroid for r in regionprops(y_true)], dtype=np.float32),
        spacing,
    )
    pred_coords = apply_spatial_spacing(
        np.asarray([r.centroid for r in regionprops(y_pred)], dtype=np.float32),
        spacing,
    )
    distances = np.minimum(cdist(true_coords, pred_coords), max_distance)
    scores_dist = 1 - distances / max_distance
    scores = np.maximum(scores_iou, scores_dist)
    costs = -(scores >= float(threshold)).astype(float) - scores / (2 * n_matched)
    true_ind, pred_ind = linear_sum_assignment(costs)
    keep = scores[true_ind, pred_ind] >= float(threshold)
    true_ind = true_ind[keep]
    pred_ind = pred_ind[keep]
    return tuple(
        (int(map_rev_true[i]), int(map_rev_pred[j]))
        for i, j in zip(1 + true_ind, 1 + pred_ind)
    )


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
    spacing: tuple[float, ...] | list[float] | np.ndarray | None = None,
) -> TrackingSequence:
    if not 0 <= slice_pct[0] < slice_pct[1] <= 1:
        raise ValueError(f"Invalid slice_pct {slice_pct}")
    if downscale_spatial < 1 or downscale_temporal < 1:
        raise ValueError("Downscale factors must be positive integers")
    if spacing is not None and downscale_spatial != 1:
        raise ValueError("CTC spacing support requires downscale_spatial=1")
    spacing = validate_spatial_spacing(spacing, ndim)
    spacing_matrix = np.diag(spacing).astype(np.float32)
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
        lineage_map, lineage_relation, lineage_parents = _lineage_arrays(tracks)
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
        lineage_map = {}
        lineage_relation = np.eye(next_index, dtype=bool)
        lineage_parents = np.full(next_index, -1, dtype=np.int64)

    gt_node_index_by_id = _ctc_node_index(gt_masks)
    node_in_degree, node_out_degree = _ctc_node_degrees(
        gt_masks, tracks if use_gt else None
    )
    gt = _lineage_graph_from_masks(
        gt_masks,
        lineage_relation,
        lineage_parents,
        spacing,
        node_in_degree=node_in_degree,
        node_out_degree=node_out_degree,
        node_predecessor_set_available=np.ones(len(node_in_degree), dtype=bool),
        node_successor_set_available=np.ones(len(node_out_degree), dtype=bool),
    )
    detections = []
    supervision = []
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
                    for gt, detection in _matching_with_spacing(
                        gt_mask,
                        detection_mask,
                        threshold=match_threshold,
                        max_distance=match_max_distance,
                        spacing=spacing,
                    )
                }
                for gt_mask, detection_mask in tqdm(
                    zip(gt_masks, detection_masks),
                    total=len(detection_masks),
                    desc="Matching",
                    leave=False,
                )
            ]
        frame_features_raw = joblib.Parallel(n_jobs=n_workers)(
            joblib.delayed(wrfeat.WRFeatures.from_mask_img)(
                mask=mask[None], img=image[None], t_start=t
            )
            for t, (mask, image) in enumerate(zip(detection_masks, images))
        )
        frame_features = [
            wrfeat.transform_feature_geometry(feature, spacing_matrix)
            for feature in frame_features_raw
        ]
        detection_sequence = DetectionSequence.from_wrfeatures(
            frame_features=frame_features,
            source_features=frame_features_raw,
            name=str(folder),
            n_frames=len(frame_features),
            masks=detection_masks if load_images else None,
            images=images if load_images else None,
            spacing=spacing,
        )
        indices_per_frame = []
        gt_node_indices_per_frame = []
        for t, feature in enumerate(frame_features_raw):
            if isolated_indices is None:
                indices = np.array(
                    [
                        lineage_map.get(matches[t].get(int(label), -1), -1)
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
            gt_node_indices = np.array(
                [
                    gt_node_index_by_id.get((t, matches[t].get(int(label), -1)), -1)
                    for label in feature.labels
                ],
                dtype=np.int64,
            )
            indices_per_frame.append(indices)
            gt_node_indices_per_frame.append(gt_node_indices)
        lineage_index = (
            np.concatenate(indices_per_frame)
            if indices_per_frame
            else np.zeros(0, dtype=np.int64)
        )
        gt_node_index = (
            np.concatenate(gt_node_indices_per_frame)
            if gt_node_indices_per_frame
            else np.zeros(0, dtype=np.int64)
        )
        detections.append(detection_sequence)
        supervision.append(
            DetectionSupervision(
                lineage_index=lineage_index,
                gt_node_index=gt_node_index,
            )
        )
    return sequence_type(
        root=root,
        ndim=ndim,
        detections=tuple(detections),
        gt=gt,
        supervision=tuple(supervision),
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
