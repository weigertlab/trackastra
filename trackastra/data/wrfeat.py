"""Regionprops features and its augmentations.
WindowedRegionFeatures (WRFeatures) is a class that holds regionprops features for a windowed track region.
"""

import itertools
import logging
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from functools import reduce
from typing import TYPE_CHECKING, Literal, Optional

import joblib
import numpy as np
import pandas as pd
from edt import edt
from skimage.measure import regionprops, regionprops_table
from tqdm import tqdm

from trackastra.data.utils import load_tiff_timeseries

try:
    # Optional fast (vectorized) regionprops backend; falls back to skimage below.
    from fast_regionprops import regionprops_table_fast
    FAST_REGIONPROPS_INSTALLED = True
except ImportError:
    FAST_REGIONPROPS_INSTALLED = False

try:
    PRETRAINED_FEATS_INSTALLED = True
    if TYPE_CHECKING:
        from trackastra_pretrained_feats import FeatureExtractor
except ImportError:
    PRETRAINED_FEATS_INSTALLED = False
    if TYPE_CHECKING:
        FeatureExtractor = None  # type: ignore

logger = logging.getLogger(__name__)

if not FAST_REGIONPROPS_INSTALLED:
    logger.warning(
        "fast_regionprops not installed; using slower skimage regionprops. "
        "Install it for a speedup: pip install fast-regionprops"
    )

_PROPERTIES = {
    "regionprops": (
        "area",
        "intensity_mean",
        "intensity_max",
        "intensity_min",
        "inertia_tensor",
    ),
    "regionprops2": (
        "equivalent_diameter_area",
        "intensity_mean",
        "inertia_tensor",
        "border_dist",
    ),
}

FEATURE_AREA = "area"
FEATURE_BORDER_DIST = "border_dist"
FEATURE_DIAMETER = "equivalent_diameter_area"
FEATURE_INERTIA = "inertia_tensor"
FEATURE_INTENSITY = "intensity"
FEATURE_CUSTOM = "_custom"
FEATURE_PRETRAINED = "pretrained_feats"

FEATURE_ALIASES = {
    "intens": FEATURE_INTENSITY,
    "intensity_mean": FEATURE_INTENSITY,
    "mean_intensity": FEATURE_INTENSITY,
}

FEATURE_RECIPES = {
    "none": (),
    "intensity": (FEATURE_INTENSITY,),
    "wrfeat": (
        FEATURE_DIAMETER,
        FEATURE_INTENSITY,
        FEATURE_INERTIA,
        FEATURE_BORDER_DIST,
    ),
    "wrfeat2": (
        FEATURE_DIAMETER,
        FEATURE_INTENSITY,
        FEATURE_INERTIA,
        FEATURE_BORDER_DIST,
    ),
    "wrfeat2_no_intensity": (
        FEATURE_DIAMETER,
        FEATURE_INERTIA,
        FEATURE_BORDER_DIST,
    ),
}


def canonical_feature_name(name: str) -> str:
    return FEATURE_ALIASES.get(name, name)


def canonicalize_feature_dict(features: dict[str, np.ndarray]) -> OrderedDict:
    out = OrderedDict()
    for name, values in features.items():
        canonical = canonical_feature_name(name)
        if canonical in out:
            raise ValueError(f"Duplicate canonical feature name {canonical!r}")
        out[canonical] = values
    return out


def feature_recipe_keys(mode: str) -> tuple[str, ...]:
    if mode == FEATURE_CUSTOM:
        return (FEATURE_CUSTOM,)
    if mode not in FEATURE_RECIPES:
        raise ValueError(f"Unknown feature recipe {mode!r}")
    return FEATURE_RECIPES[mode]


def _filter_points(
    points: np.ndarray, shape: tuple[int], origin: tuple[int] | None = None
) -> np.ndarray:
    """Returns indices of points that are inside the shape extent and given origin."""
    ndim = points.shape[-1]
    if origin is None:
        origin = (0,) * ndim

    idx = tuple(
        np.logical_and(points[:, i] >= origin[i], points[:, i] < origin[i] + shape[i])
        for i in range(ndim)
    )
    idx = np.where(np.all(idx, axis=0))[0]
    return idx


def _border_dist(mask: np.ndarray, cutoff: float = 5):
    """Returns distance to border normalized to 0 (at least cutoff away) and 1 (at border)."""
    border = np.zeros_like(mask)

    # only apply to last two dimensions
    ss = tuple(
        slice(None) if i < mask.ndim - 2 else slice(1, -1)
        for i, s in enumerate(mask.shape)
    )
    border[ss] = 1
    dist = 1 - np.minimum(edt(border) / cutoff, 1)
    return tuple(r.intensity_max for r in regionprops(mask, intensity_image=dist))


def normalize_diameter_factor(
    features: Sequence["WRFeatures"], normalize_diameter: float | None
) -> float:
    """Single scale factor mapping the median equivalent diameter to a target.

    Length-based (``equivalent_diameter`` is a length in any dimension), so the
    same factor works in 2D and 3D: ``s = normalize_diameter / median(diameter)``.
    One shared scalar over all the given features preserves every relative size
    (division asymmetry, growth) while removing the absolute pixel unit.
    """
    if normalize_diameter is None:
        return 1.0
    if normalize_diameter <= 0:
        raise ValueError("normalize_diameter must be positive")
    diameters = []
    for feature in features:
        values = feature.features.get(FEATURE_DIAMETER)
        if values is not None:
            diameters.append(values[:, 0])
    if not diameters:
        raise ValueError("normalize_diameter requires equivalent_diameter_area features")
    diameters = np.concatenate(diameters)
    valid = diameters[np.isfinite(diameters) & (diameters > 0)]
    if len(valid) == 0:
        raise ValueError("normalize_diameter found no positive diameters")
    return float(normalize_diameter) / float(np.median(valid))


def scale_feature_geometry(feature: "WRFeatures", scale: float) -> "WRFeatures":
    if scale == 1.0:
        return feature
    features = OrderedDict(feature.features)
    # lengths scale as s, areas/second-moments as s**2
    for name in (FEATURE_DIAMETER, FEATURE_BORDER_DIST):
        if name in features:
            features[name] = features[name] * scale
    for name in (FEATURE_AREA, FEATURE_INERTIA):
        if name in features:
            features[name] = features[name] * scale**2
    return WRFeatures(
        coords=feature.coords * scale,
        labels=feature.labels,
        timepoints=feature.timepoints,
        features=features,
    )


def transform_feature_geometry(
    feature: "WRFeatures", matrix: np.ndarray
) -> "WRFeatures":
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape != (feature.ndim, feature.ndim):
        raise ValueError(
            f"matrix must have shape {(feature.ndim, feature.ndim)}, got {matrix.shape}"
        )
    if np.allclose(matrix, np.eye(feature.ndim)):
        return feature
    features = OrderedDict()
    for name, values in feature.features.items():
        # `border_dist` is left in source-grid units for spacing transforms.
        # A truly physical value needs recomputing from the mask with an anisotropic
        # distance transform; treating it like a generic affine feature is not correct.
        features[name] = _transform_affine(name, values, matrix)
    return WRFeatures(
        coords=feature.coords @ matrix.T,
        labels=feature.labels,
        timepoints=feature.timepoints,
        features=features,
    )


def normalize_to_diameter(
    features: Sequence["WRFeatures"], normalize_diameter: float | None
) -> list["WRFeatures"]:
    scale = normalize_diameter_factor(features, normalize_diameter)
    return [scale_feature_geometry(feature, scale) for feature in features]


def _border_dist_fast(mask: np.ndarray, cutoff: float = 5):
    cutoff = int(cutoff)
    border = np.ones(mask.shape, dtype=np.float32)
    ndim = len(mask.shape)

    for axis, size in enumerate(mask.shape):
        # only apply to last two dimensions
        if axis < ndim - 2:
            continue

        # Create fade values for the band [0, cutoff)
        band_vals = np.arange(cutoff, dtype=np.float32) / cutoff
        band_vals = band_vals[:size]
        # Build slices for the low border
        low_slices = [slice(None)] * ndim
        low_slices[axis] = slice(0, cutoff)
        border_low = border[tuple(low_slices)]
        border_low_vals = np.minimum(
            border_low, band_vals[(...,) + (None,) * (ndim - axis - 1)]
        )
        border[tuple(low_slices)] = border_low_vals
        # Build slices for the high border
        high_slices = [slice(None)] * ndim
        high_slices[axis] = slice(max(0, size - cutoff), size)
        band_vals_rev = band_vals[::-1]
        border_high = border[tuple(high_slices)]
        border_high_vals = np.minimum(
            border_high, band_vals_rev[(...,) + (None,) * (ndim - axis - 1)]
        )
        border[tuple(high_slices)] = border_high_vals

    dist = 1 - border
    return tuple(r.intensity_max for r in regionprops(mask, intensity_image=dist))


def _regionprops_table(
    mask: np.ndarray, img: np.ndarray, properties: tuple[str, ...]
) -> dict[str, np.ndarray]:
    """Region-property table for a single frame, keyed with skimage-style names.

    Uses the vectorized ``fast_regionprops`` backend when installed (much faster
    on region-dense data), otherwise falls back to ``skimage.regionprops_table``
    plus ``_border_dist_fast``. Output is identical either way.
    """
    if FAST_REGIONPROPS_INSTALLED:
        return regionprops_table_fast(mask, img, properties=properties)

    skimage_props = tuple(p for p in properties if p != "border_dist")
    table = regionprops_table(mask, intensity_image=img, properties=skimage_props)
    if "border_dist" in properties:
        table["border_dist"] = np.asarray(_border_dist_fast(mask))
    return table


class WRFeatures:
    """regionprops features for a windowed track region."""

    def __init__(
        self,
        coords: np.ndarray,
        labels: np.ndarray,
        timepoints: np.ndarray,
        features: OrderedDict[np.ndarray],
    ):
        self.ndim = coords.shape[-1]
        if self.ndim not in (2, 3):
            raise ValueError("Only 2D or 3D data is supported")

        self.coords = coords
        self.labels = labels
        self.features = canonicalize_feature_dict(features)
        self.timepoints = timepoints

    def __repr__(self):
        s = (
            f"WindowRegionFeatures(ndim={self.ndim}, nregions={len(self.labels)},"
            f" ntimepoints={len(np.unique(self.timepoints))})\n\n"
        )
        for k, v in self.features.items():
            s += f"{k:>20} -> {v.shape}\n"
        return s

    @property
    def features_stacked(self):
        # Do not include pretrained_feats here
        # They are handled separately and should not be added to shallow features
        if not self.features or (
            len(self.features) == 1 and FEATURE_PRETRAINED in self.features
        ):
            return None
        feats = np.concatenate(
            [v for k, v in self.features.items() if k != FEATURE_PRETRAINED], axis=-1
        )
        return feats

    def features_stacked_for(self, mode: str):
        """Return the configured shallow feature representation.

        ``wrfeat2`` is derived after geometric augmentation so its normalized
        inertia components remain consistent with the transformed coordinates.
        In 3D the symmetric traceless inertia tensor contributes its five
        independent components.
        """
        required = feature_recipe_keys(mode)
        missing = set(required).difference(self.features)
        if missing:
            raise ValueError(f"{mode} requires feature properties {sorted(missing)}")
        if mode == "none":
            return np.zeros((len(self), 0), dtype=np.float32)
        if mode in ("intensity", FEATURE_CUSTOM, "wrfeat"):
            return np.concatenate([self.features[name] for name in required], axis=-1)

        diameter = self.features[FEATURE_DIAMETER][:, 0]
        inertia = self.features[FEATURE_INERTIA].reshape(-1, self.ndim, self.ndim)
        border_dist = self.features[FEATURE_BORDER_DIST][:, 0]

        trace = np.trace(inertia, axis1=1, axis2=2)
        unit_ball = np.pi if self.ndim == 2 else 4 * np.pi / 3
        ball_measure = unit_ball * (np.maximum(diameter, 0) / 2) ** self.ndim
        eps = np.finfo(np.float32).eps
        compactness = trace / np.maximum(ball_measure ** (2 / self.ndim), eps)

        if self.ndim == 2:
            q1 = np.divide(
                inertia[:, 0, 0] - inertia[:, 1, 1],
                trace,
                out=np.zeros_like(trace),
                where=trace > eps,
            )
            q2 = np.divide(
                2 * inertia[:, 0, 1],
                trace,
                out=np.zeros_like(trace),
                where=trace > eps,
            )
            # A positive-semidefinite tensor gives ||q|| <= 1. Project small
            # numerical violations (or malformed inputs) back onto the unit disk.
            q_norm = np.sqrt(np.square(q1) + np.square(q2))
            q_scale = np.maximum(q_norm, 1)
            q_channels = [q1 / q_scale, q2 / q_scale]
        else:
            inertia_norm = np.divide(
                inertia,
                trace[:, None, None],
                out=np.zeros_like(inertia),
                where=trace[:, None, None] > eps,
            )
            dev = inertia_norm.copy()
            valid = trace > eps
            dev[valid] -= np.eye(self.ndim, dtype=inertia.dtype) / self.ndim
            q_scale = np.sqrt(3 / 2)
            q_norm = q_scale * np.sqrt(
                np.sum(np.square(np.diagonal(dev, axis1=1, axis2=2)), axis=1)
                + 2
                * (
                    np.square(dev[:, 0, 1])
                    + np.square(dev[:, 0, 2])
                    + np.square(dev[:, 1, 2])
                )
            )
            dev = dev * q_scale
            q_project = np.maximum(q_norm, 1)
            dev = dev / q_project[:, None, None]
            q_channels = [
                dev[:, 0, 0],
                dev[:, 1, 1],
                np.sqrt(2) * dev[:, 0, 1],
                np.sqrt(2) * dev[:, 0, 2],
                np.sqrt(2) * dev[:, 1, 2],
            ]

        channels = [
            np.log1p(np.maximum(diameter, 0)),
            np.log1p(np.maximum(compactness, 0)),
            *q_channels,
            np.log1p(np.maximum(border_dist, 0)),
        ]
        if mode == "wrfeat2":
            channels.insert(1, self.features[FEATURE_INTENSITY][:, 0])
        return np.stack(channels, axis=-1).astype(np.float32, copy=False)

    @property
    def pretrained_feats(self):
        # for compatibility with WRPretrainedFeatures
        if FEATURE_PRETRAINED in self.features:
            return self.features[FEATURE_PRETRAINED]
        return None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, key):
        if key in self.features:
            return self.features[key]
        else:
            raise KeyError(f"Key {key} not found in features")

    @classmethod
    def concat(cls, feats: Sequence["WRFeatures"]) -> "WRFeatures":
        """Concatenate multiple WRFeatures into a single one."""
        if len(feats) == 0:
            raise ValueError("Cannot concatenate empty list of features")
        return reduce(lambda x, y: x + y, feats)

    def __add__(self, other: "WRFeatures") -> "WRFeatures":
        """Concatenate two WRFeatures."""
        if self.ndim != other.ndim:
            raise ValueError("Cannot concatenate features of different dimensions")
        if self.features.keys() != other.features.keys():
            raise ValueError("Cannot concatenate features with different properties")

        coords = np.concatenate([self.coords, other.coords], axis=0)
        labels = np.concatenate([self.labels, other.labels], axis=0)
        timepoints = np.concatenate([self.timepoints, other.timepoints], axis=0)

        features = OrderedDict(
            (k, np.concatenate([v, other.features[k]], axis=0))
            for k, v in self.features.items()
        )

        return WRFeatures(
            coords=coords, labels=labels, timepoints=timepoints, features=features
        )

    @classmethod
    def from_mask_img(
        cls,
        mask: np.ndarray,
        img: np.ndarray,
        properties="regionprops2",
        t_start: int = 0,
    ):
        img = np.asarray(img)
        mask = np.asarray(mask)

        _ntime, ndim = mask.shape[0], mask.ndim - 1
        if ndim not in (2, 3):
            raise ValueError("Only 2D or 3D data is supported")

        properties = tuple(_PROPERTIES[properties])
        if "label" in properties or "centroid" in properties:
            raise ValueError(
                f"label and centroid should not be in properties {properties}"
            )

        df_properties = ("label", "centroid", *properties)
        dfs = []
        for i, (y, x) in enumerate(zip(mask, img)):
            _df = pd.DataFrame(_regionprops_table(y, x, df_properties))
            _df["timepoint"] = i + t_start
            dfs.append(_df)
        df = pd.concat(dfs)

        timepoints = df["timepoint"].values.astype(np.int32)
        labels = df["label"].values.astype(np.int32)
        coords = df[[f"centroid-{i}" for i in range(ndim)]].values.astype(np.float32)

        features = OrderedDict(
            (
                p,
                np.stack(
                    [
                        df[c].values.astype(np.float32)
                        for c in df.columns
                        if c.startswith(p)
                    ],
                    axis=-1,
                ),
            )
            for p in properties
        )

        return cls(
            coords=coords, labels=labels, timepoints=timepoints, features=features
        )


# augmentations


class WRRandomCrop:
    """windowed region random crop augmentation."""

    def __init__(
        self,
        crop_size: int | tuple[int] | None = None,
        ndim: int = 2,
        ensure_all_centers: bool = True,
    ) -> None:
        """crop_size: tuple of int
        can be tuple of length 1 (all dimensions)
                     of length ndim (y,x,...)
                     of length 2*ndim (y1,y2, x1,x2, ...).

        If ensure_all_centers is true, dimensions whose complete occupied
        extent fits within the sampled crop retain every detection center.
        """
        if isinstance(crop_size, int):
            crop_size = (crop_size,) * 2 * ndim
        elif isinstance(crop_size, Iterable):
            pass
        else:
            raise ValueError(f"{crop_size} has to be int or tuple of int")

        if len(crop_size) == 1:
            crop_size = (crop_size[0],) * 2 * ndim
        elif len(crop_size) == ndim:
            crop_size = tuple(itertools.chain(*tuple((c, c) for c in crop_size)))
        elif len(crop_size) == 2 * ndim:
            pass
        else:
            raise ValueError(f"crop_size has to be of length 1, {ndim}, or {2 * ndim}")

        crop_size = np.array(crop_size)
        self._ndim = ndim
        self._crop_bounds = crop_size[::2], crop_size[1::2]
        self.ensure_all_centers = ensure_all_centers

    def __call__(self, features: WRFeatures):
        crop_size = np.random.randint(self._crop_bounds[0], self._crop_bounds[1] + 1)
        points = features.coords

        if len(points) == 0:
            return features, np.empty(0, dtype=int)

        # sample point and corner relative to it

        _idx = np.random.randint(len(points))
        corner = (
            points[_idx]
            - crop_size
            + 1
            + np.random.randint(crop_size // 4, 3 * crop_size // 4)
        )

        # Do not cut detections along dimensions where the complete occupied
        # extent already fits in the crop. The previous unconstrained origin
        # could exclude cells even when every center fit inside crop_size.
        point_min = points.min(axis=0)
        point_max = points.max(axis=0)
        fits = self.ensure_all_centers & (point_max - point_min < crop_size)
        if np.any(fits):
            lower = point_max - crop_size + 1e-6
            corner[fits] = np.clip(corner[fits], lower[fits], point_min[fits])

        idx = _filter_points(points, shape=crop_size, origin=corner)

        return (
            WRFeatures(
                coords=points[idx],
                labels=features.labels[idx],
                timepoints=features.timepoints[idx],
                features=OrderedDict((k, v[idx]) for k, v in features.features.items()),
            ),
            idx,
        )


class WRBaseAugmentation:
    def __init__(self, p: float = 0.5) -> None:
        self._p = p

    def __call__(self, features: WRFeatures):
        # PyTorch seeds NumPy's process-local RNG separately in each DataLoader
        # worker. Do not keep a RandomState on the dataset: it is cloned with an
        # identical state into every worker and may also be frozen in cache files.
        if np.random.rand() > self._p or len(features) == 0:
            return features
        return self._augment(features)

    def _augment(self, features: WRFeatures):
        raise NotImplementedError()


class WRRandomFlip(WRBaseAugmentation):
    def _augment(self, features: WRFeatures):
        ndim = features.ndim
        flip = np.random.randint(0, 2, ndim)
        # a flip is a reflection M = diag(+-1); apply it to coords *and* features
        # (the inertia tensor's off-diagonals flip sign) for a consistent sample.
        signs = np.ones(ndim)
        for i, f in enumerate(flip):
            if f == 1:
                signs[ndim - i - 1] = -1
        M = np.diag(signs)
        points = features.coords @ M.T
        feats = OrderedDict(
            (k, _transform_affine(k, v, M)) for k, v in features.features.items()
        )
        return WRFeatures(
            coords=points,
            labels=features.labels,
            timepoints=features.timepoints,
            features=feats,
        )


def _scale_matrix(sz: float, sy: float, sx: float):
    return np.diag([sz, sy, sx])


# def _scale_matrix(sy: float, sx: float):
#     return np.array([[1, 0, 0], [0, sy, 0], [0, 0, sx]])


def _shear_matrix(shy: float, shx: float):
    return np.array([[1, 0, 0], [0, 1 + shx * shy, shy], [0, shx, 1]])


def _rotation_matrix(theta: float, axis: tuple[float, float, float] = (1, 0, 0)):
    """Return a 3D axis-angle rotation matrix in coordinate order."""
    axis = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm == 0:
        raise ValueError("rotation axis must be non-zero")
    axis = axis / norm
    a0, a1, a2 = axis
    K = np.array([
        [0, -a2, a1],
        [a2, 0, -a0],
        [-a1, a0, 0],
    ])
    eye = np.eye(3)
    return eye + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def _transform_affine(k: str, v: np.ndarray, M: np.ndarray):
    # M is either a single (ndim, ndim) map shared by all detections (global
    # geometric augmentations) or a per-detection stack (N, ndim, ndim) for
    # independent shape jitter. The two cases differ only in how M contracts.
    M = np.asarray(M)
    batched = M.ndim == 3
    ndim = M.shape[-1]
    # use |det|: area/diameter are positive magnitudes, and a reflection (det<0,
    # e.g. from WRRandomFlip) must not flip their sign (or make diameter complex).
    absdet = np.abs(np.linalg.det(M))  # scalar, or (N,) when batched
    if batched:
        absdet = absdet[:, None]
    k = canonical_feature_name(k)
    if k == FEATURE_AREA:
        v = absdet * v
    elif k == FEATURE_DIAMETER:
        v = absdet ** (1 / ndim) * v

    elif k == FEATURE_INERTIA:
        # skimage's inertia_tensor is the moment of inertia I = tr(S)*Id - S, with
        # S the covariance. Only S transforms as M S M^T; I does so only for
        # orthogonal M (rotations/reflections), and is wrong under scale/shear. So
        # convert I -> S, transform S, convert back. tr(I) = (ndim-1)*tr(S).
        v = v.reshape(-1, ndim, ndim)
        eye = np.eye(ndim)
        tr_s = np.trace(v, axis1=1, axis2=2) / (ndim - 1)
        s = tr_s[:, None, None] * eye - v
        if batched:
            s = np.einsum("rij, rjk, rlk -> ril", M, s, M)  # per-detection M S M^T
        else:
            s = np.einsum("ij, rjk, lk -> ril", M, s, M)  # M S M^T
        tr_sp = np.trace(s, axis1=1, axis2=2)
        v = (tr_sp[:, None, None] * eye - s).reshape(-1, ndim * ndim)
    elif k in (
        FEATURE_INTENSITY,
        "intensity_std",
        "intensity_max",
        "intensity_min",
        FEATURE_BORDER_DIST,
        FEATURE_CUSTOM,
    ):
        pass
    else:
        raise ValueError(f"Don't know how to affinely transform {k}")
    return v


class WRRandomAffine(WRBaseAugmentation):
    """Apply rotation, optional 3D tilt, shear, and isotropic log-uniform scale."""

    def __init__(
        self,
        degrees: float = 10,
        tilt_degrees: float = 0,
        scale: float = (0.9, 1.1),
        shear: float = (0.1, 0.1),
        p: float = 0.5,
    ):
        super().__init__(p)
        self.degrees = degrees if degrees is not None else 0
        self.tilt_degrees = tilt_degrees if tilt_degrees is not None else 0
        self.scale = scale if scale is not None else (1, 1)
        self.shear = shear if shear is not None else (0, 0)
        if self.scale[0] <= 0 or self.scale[1] < self.scale[0]:
            raise ValueError("scale bounds must satisfy 0 < min <= max")

    def _augment(self, features: WRFeatures):
        degrees = np.random.uniform(-self.degrees, self.degrees) / 180 * np.pi
        rotation = _rotation_matrix(degrees)
        if features.ndim == 3 and self.tilt_degrees:
            tilt = np.random.uniform(-self.tilt_degrees, self.tilt_degrees) / 180 * np.pi
            psi = np.random.uniform(-np.pi, np.pi)
            tilt_axis = (0, np.cos(psi), np.sin(psi))
            rotation = rotation @ _rotation_matrix(tilt, axis=tilt_axis)
        # A linear draw from (0.5, 2) is biased toward zooming in (mean 1.25).
        # A single log-uniform draw makes reciprocal zooms equally likely and
        # avoids unrealistic independent stretching of each spatial axis.
        scale = np.exp(np.random.uniform(np.log(self.scale[0]), np.log(self.scale[1])))
        shy = np.random.uniform(-self.shear[0], self.shear[0])
        shx = np.random.uniform(-self.shear[1], self.shear[1])

        self._M = (
            rotation
            @ _scale_matrix(scale, scale, scale)
            @ _shear_matrix(shy, shx)
        )

        # M is by default 3D , we need to remove the last dimension for 2D
        self._M = self._M[-features.ndim :, -features.ndim :]
        points = features.coords @ self._M.T

        feats = OrderedDict(
            (k, _transform_affine(k, v, self._M)) for k, v in features.features.items()
        )

        return WRFeatures(
            coords=points,
            labels=features.labels,
            timepoints=features.timepoints,
            features=feats,
        )


def _random_shape_jitter_matrices(
    n: int, scale: tuple[float, float], shear: float, ndim: int
) -> np.ndarray:
    """Per-detection near-identity linear maps for shape measurement noise.

    Anisotropic log-uniform scale perturbs size (diameter) and aspect ratio
    (compactness, inertia anisotropy); a small 2D shear perturbs orientation.
    Returns a stack of (n, ndim, ndim) matrices.
    """
    # process-local RNG: see WRBaseAugmentation.__call__ on worker seeding
    s = np.exp(np.random.uniform(np.log(scale[0]), np.log(scale[1]), size=(n, ndim)))
    M = s[:, :, None] * np.eye(ndim)[None]  # (n, ndim, ndim) anisotropic scale
    if shear and ndim == 2:
        shy = np.random.uniform(-shear, shear, n)
        shx = np.random.uniform(-shear, shear, n)
        Sh = np.tile(np.eye(2), (n, 1, 1))
        Sh[:, 0, 0] = 1 + shx * shy
        Sh[:, 0, 1] = shy
        Sh[:, 1, 0] = shx
        M = M @ Sh
    return M


class WRRandomShapeJitter(WRBaseAugmentation):
    """Independent per-detection shape measurement noise.

    Applies a small random linear map to each detection's *shape* features
    (size and second moments) through the same `_transform_affine` used by the
    global geometric augmentations, so diameter, compactness and the inertia
    anisotropy stay mutually consistent. Coordinates, intensity and border_dist
    are left untouched: this perturbs how a cell looks, not where it is, so the
    model does not treat shape as an exact cross-frame fingerprint (matters most
    when intensity is unused, e.g. wrfeat2_no_intensity).
    """

    def __init__(
        self,
        scale: tuple[float, float] = (0.9, 1.1),
        shear: float = 0.05,
        p: float = 0.8,
    ):
        super().__init__(p)
        if scale[0] <= 0 or scale[1] < scale[0]:
            raise ValueError("scale bounds must satisfy 0 < min <= max")
        self.scale = scale
        self.shear = shear

    def _augment(self, features: WRFeatures):
        M = _random_shape_jitter_matrices(
            len(features), self.scale, self.shear, features.ndim
        )
        feats = OrderedDict(
            (k, _transform_affine(k, v, M)) for k, v in features.features.items()
        )
        return WRFeatures(
            coords=features.coords,
            labels=features.labels,
            timepoints=features.timepoints,
            features=feats,
        )


class WRRandomBrightness(WRBaseAugmentation):
    def __init__(
        self,
        scale: tuple[float] = (0.5, 2.0),
        shift: tuple[float] = (-0.1, 0.1),
        p: float = 0.5,
    ):
        super().__init__(p)
        self.scale = scale
        self.shift = shift

    def _augment(self, features: WRFeatures):
        scale = np.random.uniform(*self.scale)
        shift = np.random.uniform(*self.shift)

        key_vals = []

        for k, v in features.features.items():
            if "intensity" in k:
                v = v * scale + shift
            key_vals.append((k, v))
        feats = OrderedDict(key_vals)
        return WRFeatures(
            coords=features.coords,
            labels=features.labels,
            timepoints=features.timepoints,
            features=feats,
        )


class WRRandomOffset(WRBaseAugmentation):
    def __init__(self, offset: float = (-3, 3), p: float = 0.5):
        super().__init__(p)
        self.offset = offset

    def _augment(self, features: WRFeatures):
        offset = np.random.uniform(*self.offset, features.coords.shape)
        coords = features.coords + offset
        return WRFeatures(
            coords=coords,
            labels=features.labels,
            timepoints=features.timepoints,
            features=features.features,
        )


class WRRandomFrameJump(WRBaseAugmentation):
    """Shift exactly one frame in a window by one shared spatial offset."""

    def __init__(self, offset: tuple[float, float] = (0, 0), p: float = 0.0):
        super().__init__(p)
        self.offset = offset

    def _augment(self, features: WRFeatures):
        frames = np.unique(features.timepoints)
        if len(frames) == 0:
            return features
        frame = np.random.choice(frames)
        frame_mask = features.timepoints == frame
        offset = np.random.uniform(*self.offset, features.coords.shape[-1])
        coords = features.coords.copy()
        coords[frame_mask] += offset
        return WRFeatures(
            coords=coords,
            labels=features.labels,
            timepoints=features.timepoints,
            features=features.features,
        )


class WRRandomMovement(WRBaseAugmentation):
    """random global linear shift."""

    def __init__(self, offset: float = (-10, 10), p: float = 0.5):
        super().__init__(p)
        self.offset = offset

    def _augment(self, features: WRFeatures):
        base_offset = np.random.uniform(*self.offset, features.coords.shape[-1])
        tmin = features.timepoints.min()
        offset = (features.timepoints[:, None] - tmin) * base_offset[None]
        coords = features.coords + offset

        return WRFeatures(
            coords=coords,
            labels=features.labels,
            timepoints=features.timepoints,
            features=features.features,
        )


class WRAugmentationPipeline:
    def __init__(self, augmentations: Sequence[WRBaseAugmentation]):
        self.augmentations = augmentations

    def __call__(self, feats: WRFeatures):
        for aug in self.augmentations:
            feats = aug(feats)
        return feats


def get_features(
    detections: np.ndarray,
    imgs: np.ndarray | None = None,
    features: Literal[
        "none",
        "intensity",
        "wrfeat",
        "wrfeat2",
        "wrfeat2_no_intensity",
        "pretrained_feats",
        "pretrained_feats_aug",
    ] = "wrfeat",
    ndim: int = 2,
    n_workers=0,
    progbar_class=tqdm,
    feature_extractor: Optional["FeatureExtractor"] | None = None,
) -> list[WRFeatures]:
    detections = _check_dimensions(detections, ndim)
    imgs = _check_dimensions(imgs, ndim)
    logger.info(f"Extracting features from {len(detections)} frames.")
    logger.info(
        "Using fast_regionprops backend"
        if FAST_REGIONPROPS_INSTALLED
        else "Using skimage regionprops backend (install trackastra[fast] to speed"
        " up feature extraction)"
    )
    if n_workers > 0:
        logger.info(f"Using {n_workers} processes for feature extraction")
        features = joblib.Parallel(n_jobs=n_workers, backend="loky")(
            joblib.delayed(WRFeatures.from_mask_img)(
                # New axis for time component
                mask=mask[np.newaxis, ...].copy(),
                img=img[np.newaxis, ...].copy(),
                t_start=t,
            )
            for t, (mask, img) in progbar_class(
                enumerate(zip(detections, imgs)),
                total=len(imgs),
                desc="Extracting features",
            )
        )
    elif features == "pretrained_feats" or features == "pretrained_feats_aug":
        feature_extractor.precompute_image_embeddings(imgs)
        from trackastra_pretrained_feats import WRPretrainedFeatures

        features = [
            WRPretrainedFeatures.from_mask_img(
                img=img[np.newaxis],
                mask=mask[np.newaxis],
                feature_extractor=feature_extractor,
                t_start=t,
                additional_properties=feature_extractor.additional_features,
            )
            for t, (mask, img) in enumerate(zip(detections, imgs))
        ]
    else:
        logger.info("Using single process for feature extraction")
        features = tuple(
            WRFeatures.from_mask_img(
                mask=mask[np.newaxis, ...],
                img=img[np.newaxis, ...],
                t_start=t,
            )
            for t, (mask, img) in progbar_class(
                enumerate(zip(detections, imgs)),
                total=len(imgs),
                desc="Extracting features",
            )
        )

    return features


def _check_dimensions(x: np.ndarray, ndim: int):
    if ndim == 2 and not x.ndim == 3:
        raise ValueError(f"Expected 2D data, got {x.ndim - 1}D data")
    elif ndim == 3:
        # if ndim=3 and data is two dimensional, it will be cast to 3D
        if x.ndim == 3:
            x = np.expand_dims(x, axis=1)
        elif x.ndim == 4:
            pass
        else:
            raise ValueError(f"Expected 3D data, got {x.ndim - 1}D data")
    return x


def build_windows(
    features: list[WRFeatures],
    window_size: int,
    progbar_class=tqdm,
    as_torch: bool = False,
    feature_mode: Literal["wrfeat", "wrfeat2", "wrfeat2_no_intensity"] = "wrfeat",
) -> list[dict]:
    if len(features) < 2:
        raise ValueError(f"Need at least 2 frames for tracking, got {len(features)}.")
    if as_torch:
        import torch
    # Clamp window size to number of frames
    window_size = min(window_size, len(features))
    windows = []
    for t1, t2 in progbar_class(
        zip(range(0, len(features)), range(window_size, len(features) + 1)),
        total=len(features) - window_size + 1,
        desc="Building windows",
    ):
        feat = WRFeatures.concat(features[t1:t2])
        try:
            pt_feats = (
                feat.pretrained_feats if feat.pretrained_feats is not None else None
            )
        except AttributeError:
            pt_feats = None
        labels = feat.labels
        timepoints = feat.timepoints
        coords = feat.coords

        if len(feat) == 0:
            coords = np.zeros((0, feat.ndim), dtype=int)

        stacked_features = feat.features_stacked_for(feature_mode)
        if as_torch and stacked_features is not None:
            stacked_features = torch.from_numpy(stacked_features)
        w = dict(
            coords=torch.from_numpy(coords) if as_torch else coords,
            t1=torch.tensor(t1, dtype=torch.int32) if as_torch else t1,
            labels=torch.from_numpy(labels) if as_torch else labels,
            timepoints=torch.from_numpy(timepoints) if as_torch else timepoints,
            features=stacked_features,
        )
        # Add pre-trained features
        if pt_feats is not None:
            w["pretrained_feats"] = torch.from_numpy(pt_feats) if as_torch else pt_feats

        windows.append(w)

    logger.debug(f"Built {len(windows)} track windows.\n")
    return windows


if __name__ == "__main__":
    imgs = load_tiff_timeseries(
        # "/scratch0/data/celltracking/ctc_2024/Fluo-C3DL-MDA231/train/01",
        "/scratch0/data/celltracking/ctc_2024/Fluo-N2DL-HeLa/train/01",
    )
    masks = load_tiff_timeseries(
        # "/scratch0/data/celltracking/ctc_2024/Fluo-C3DL-MDA231/train/01_GT/TRA",
        "/scratch0/data/celltracking/ctc_2024/Fluo-N2DL-HeLa/train/01_GT/TRA",
        dtype=int,
    )

    features = get_features(detections=masks, imgs=imgs, ndim=3)
    windows = build_windows(features, window_size=4)


# if __name__ == "__main__":
#     y = np.zeros((1, 100, 100), np.uint8)
#     y[:, 20:40, 20:60] = 1
#     x = y + np.random.normal(0, 0.1, y.shape)

#     f = WRFeatures.from_mask_img(y, x, properties=("intensity_mean", "area"))
