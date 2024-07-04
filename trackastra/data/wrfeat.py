"""Regionprops features and its augmentations.
WindowedRegionFeatures (WRFeatures) is a class that holds regionprops features for a windowed track region.
"""

import itertools
import logging
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from functools import reduce
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from edt import edt
from skimage.measure import regionprops, regionprops_table
from tqdm import tqdm

from trackastra.data.utils import load_tiff_timeseries

logger = logging.getLogger(__name__)

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
        self.features = features.copy()
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
        return np.concatenate([v for k, v in self.features.items()], axis=-1)

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
        _ntime, ndim = mask.shape[0], mask.ndim - 1
        if ndim not in (2, 3):
            raise ValueError("Only 2D or 3D data is supported")

        properties = tuple(_PROPERTIES[properties])
        if "label" in properties or "centroid" in properties:
            raise ValueError(
                f"label and centroid should not be in properties {properties}"
            )

        if "border_dist" in properties:
            use_border_dist = True
            # remove border_dist from properties
            properties = tuple(p for p in properties if p != "border_dist")
        else:
            use_border_dist = False

        df_properties = ("label", "centroid", *properties)
        dfs = []
        for i, (y, x) in enumerate(zip(mask, img)):
            _df = pd.DataFrame(
                regionprops_table(y, intensity_image=x, properties=df_properties)
            )
            _df["timepoint"] = i + t_start

            if use_border_dist:
                _df["border_dist"] = _border_dist(y)

            dfs.append(_df)
        df = pd.concat(dfs)

        if use_border_dist:
            properties = (*properties, "border_dist")

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
    ) -> None:
        """crop_size: tuple of int
        can be tuple of length 1 (all dimensions)
                     of length ndim (y,x,...)
                     of length 2*ndim (y1,y2, x1,x2, ...).
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
        self._rng = np.random.RandomState()

    def __call__(self, features: WRFeatures):

        crop_size = self._rng.randint(self._crop_bounds[0], self._crop_bounds[1] + 1)
        points = features.coords

        if len(points) == 0:
            print("No points given, cannot ensure inside points")
            return features

        # sample point and corner relative to it

        _idx = np.random.randint(len(points))
        corner = (
            points[_idx]
            - crop_size
            + 1
            + self._rng.randint(crop_size // 4, 3 * crop_size // 4)
        )

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
        self._rng = np.random.RandomState()

    def __call__(self, features: WRFeatures):
        if self._rng.rand() > self._p or len(features) == 0:
            return features
        return self._augment(features)

    def _augment(self, features: WRFeatures):
        raise NotImplementedError()


class WRRandomFlip(WRBaseAugmentation):
    def _augment(self, features: WRFeatures):
        ndim = features.ndim
        flip = self._rng.randint(0, 2, 2)
        points = features.coords.copy()
        for i, f in enumerate(flip):
            if f == 1:
                points[:, ndim - i - 1] *= -1
        return WRFeatures(
            coords=points,
            labels=features.labels,
            timepoints=features.timepoints,
            features=features.features,
        )


def _scale_matrix(sy: float, sx: float):
    return np.array([[1, 0, 0], [0, sy, 0], [0, 0, sx]])


def _shear_matrix(shy: float, shx: float):
    return np.array([[1, 0, 0], [0, 1, shy], [0, shx, 1]])


def _rotation_matrix(theta: float):
    return np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ]
    )


def _transform_affine(k: str, v: np.ndarray, M: np.ndarray):
    ndim = len(M)
    if k == "area":
        v = np.linalg.det(M) * v
    elif k == "equivalent_diameter_area":
        v = np.linalg.det(M) * (1 / len(M)) * v

    elif k == "inertia_tensor":
        # v' = M * v  * M^T
        v = v.reshape(-1, ndim, ndim)
        # v * M^T
        v = np.einsum("ijk, mk -> ijm", v, M)
        # M * v
        v = np.einsum("ij, kjm -> kim", M, v)
        v = v.reshape(-1, ndim * ndim)
    elif k in (
        "intensity_mean",
        "intensity_std",
        "intensity_max",
        "intensity_min",
        "border_dist",
    ):
        pass
    else:
        raise ValueError(f"Don't know how to affinely transform {k}")
    return v


class WRRandomAffine(WRBaseAugmentation):
    def __init__(
        self,
        degrees: float = 10,
        scale: float = (0.9, 1.1),
        shear: float = (0.1, 0.1),
        p: float = 0.5,
    ):
        super().__init__(p)
        self.degrees = degrees if degrees is not None else 0
        self.scale = scale if scale is not None else (1, 1)
        self.shear = shear if shear is not None else (0, 0)

    def _augment(self, features: WRFeatures):

        degrees = self._rng.uniform(-self.degrees, self.degrees) / 180 * np.pi
        scale = self._rng.uniform(*self.scale, 2)
        shy = self._rng.uniform(-self.shear[0], self.shear[0])
        shx = self._rng.uniform(-self.shear[1], self.shear[1])

        self._M = (
            _rotation_matrix(degrees)
            @ _scale_matrix(scale[0], scale[1])
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

        scale = self._rng.uniform(*self.scale)
        shift = self._rng.uniform(*self.shift)

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

        offset = self._rng.uniform(*self.offset, features.coords.shape)
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
    features: Literal["none", "wrfeat"] = "wrfeat",
    ndim: int = 2,
    n_workers=0,
    progbar_class=tqdm,
) -> list[WRFeatures]:
    detections = _check_dimensions(detections, ndim)
    imgs = _check_dimensions(imgs, ndim)
    logger.info(f"Extracting features from {len(detections)} detections")
    if n_workers > 0:
        features = joblib.Parallel(n_jobs=n_workers)(
            joblib.delayed(WRFeatures.from_mask_img)(
                # New axis for time component
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
    features: list[WRFeatures], window_size: int, progbar_class=tqdm
) -> list[dict]:
    windows = []
    for t1, t2 in progbar_class(
        zip(range(0, len(features)), range(window_size, len(features) + 1)),
        total=len(features) - window_size + 1,
        desc="Building windows",
    ):
        feat = WRFeatures.concat(features[t1:t2])

        labels = feat.labels
        timepoints = feat.timepoints
        coords = feat.coords

        if len(feat) == 0:
            coords = np.zeros((0, feat.ndim), dtype=int)

        w = dict(
            coords=coords,
            t1=t1,
            labels=labels,
            timepoints=timepoints,
            features=feat.features_stacked,
        )
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
