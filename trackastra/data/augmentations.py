"""#TODO: dont convert to numpy and back to torch."""

from collections.abc import Iterable, Sequence
from itertools import chain
from typing import Any

import kornia.augmentation as K
import numpy as np
import torch
from kornia.augmentation import random_generator as rg
from kornia.augmentation.utils import _range_bound
from kornia.constants import DataKey, Resample


def default_augmenter(coords: np.ndarray):
    # TODO parametrize magnitude of different augmentations
    ndim = coords.shape[1]

    assert coords.ndim == 2 and ndim in (2, 3)

    # first remove offset
    center = coords.mean(axis=0, keepdims=True)

    coords = coords - center

    # apply random flip
    coords *= 2 * np.random.randint(0, 2, (1, ndim)) - 1

    # apply rotation along the last two dimensions
    phi = np.random.uniform(0, 2 * np.pi)
    coords = _rotate(coords, phi, center=None)

    if ndim == 3:
        # rotate along the first two dimensions too
        phi2, phi3 = np.random.uniform(0, 2 * np.pi, 2)
        coords = _rotate(coords, phi2, rot_axis=(0, 1), center=None)
        coords = _rotate(coords, phi3, rot_axis=(0, 2), center=None)

    coords += center

    # translation
    trans = 128 * np.random.uniform(-1, 1, (1, ndim))
    coords += trans

    # elastic
    coords += 1.5 * np.random.normal(0, 1, coords.shape)

    return coords


def _rotate(
    coords: np.ndarray, phi: float, rot_axis=(-2, -1), center: tuple | None = None
):
    """Rotation along the last two dimensions of coords[..,:-2:]."""
    ndim = coords.shape[1]
    assert coords.ndim == 2 and ndim in (2, 3)

    if center is None:
        center = (0,) * ndim

    assert len(center) == ndim

    center = np.asarray(center)
    co, si = np.cos(phi), np.sin(phi)
    Rot = np.eye(ndim)
    Rot[np.ix_(rot_axis, rot_axis)] = np.array(((co, -si), (si, co)))
    x = coords - center
    x = x @ Rot.T
    x += center
    return x


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


class ConcatAffine(K.RandomAffine):
    """Concatenate multiple affine transformations without intermediates."""

    def __init__(self, affines: Sequence[K.RandomAffine]):
        super().__init__(degrees=0)
        self._affines = affines
        if not all([a.same_on_batch for a in affines]):
            raise ValueError("all affines must have same_on_batch=True")

    def merge_params(self, params: Sequence[dict[str, torch.Tensor]]):
        """Merge params from affines."""
        out = params[0].copy()

        def _torchmax(x, dim):
            return torch.max(x, dim=dim).values

        ops = {
            "translations": torch.sum,
            "center": torch.mean,
            "scale": torch.prod,
            "shear_x": torch.sum,
            "shear_y": torch.sum,
            "angle": torch.sum,
            "batch_prob": _torchmax,
        }
        for k, v in params[0].items():
            ps = [p[k] for p in params if len(p[k]) > 0]
            if len(ps) > 0 and k in ops:
                v_new = torch.stack(ps, dim=0).float()
                v_new = ops[k](v_new, dim=0)
                v_new = v_new.to(v.dtype)
            else:
                v_new = v
            out[k] = v_new

        return out

    def forward_parameters(
        self, batch_shape: tuple[int, ...]
    ) -> dict[str, torch.Tensor]:
        params = tuple(a.forward_parameters(batch_shape) for a in self._affines)
        # print(params)
        return self.merge_params(params)


# custom augmentations
class RandomIntensityScaleShift(K.IntensityAugmentationBase2D):
    r"""Apply a random scale and shift to the image intensity.

    Args:
        p: probability of applying the transformation.
        scale:  the scale factor to apply
        shift: the offset to apply
        clip_output: if true clip output
        same_on_batch: apply the same transformation across the batch.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                 to the batch form (False).
    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`, Optional: :math:`(B, 3, 3)`
        - Output: :math:`(B, C, H, W)`

    .. note::
        This function internally uses :func:`kornia.enhance.adjust_brightness`

    """

    def __init__(
        self,
        scale: tuple[float, float] = (0.5, 2.0),
        shift: tuple[float, float] = (-0.1, 0.1),
        clip_output: bool = True,
        same_on_batch: bool = False,
        p: float = 1.0,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self.scale = _range_bound(
            scale, "scale", center=0, bounds=(-float("inf"), float("inf"))
        )
        self.shift = _range_bound(
            shift, "shift", center=0, bounds=(-float("inf"), float("inf"))
        )
        self._param_generator = rg.PlainUniformGenerator(
            (self.scale, "scale_factor", None, None),
            (self.shift, "shift_factor", None, None),
        )

        self.clip_output = clip_output

    def apply_transform(
        self,
        input: torch.Tensor,
        params: dict[str, torch.Tensor],
        flags: dict[str, Any],
        transform: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scale_factor = params["scale_factor"].to(input)
        shift_factor = params["shift_factor"].to(input)
        scale_factor = scale_factor.view(len(scale_factor), 1, 1, 1)
        shift_factor = shift_factor.view(len(scale_factor), 1, 1, 1)
        img_adjust = input * scale_factor + shift_factor
        if self.clip_output:
            img_adjust = img_adjust.clamp(min=0.0, max=1.0)
        return img_adjust


class RandomTemporalAffine(K.RandomAffine):
    r"""Apply a random 2D affine transformation to a batch of images while
    varying the transformation across the time dimension from 0 to 1.

    Same args/kwargs as K.RandomAffine

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, same_on_batch=True, **kwargs)

    def forward_parameters(
        self, batch_shape: tuple[int, ...]
    ) -> dict[str, torch.Tensor]:
        params = super().forward_parameters(batch_shape)
        factor = torch.linspace(0, 1, batch_shape[0]).to(params["translations"])
        for key in ["translations", "center", "angle", "shear_x", "shear_y"]:
            v = params[key]
            if len(v) > 0:
                params[key] = v * factor.view(*((-1,) + (1,) * len(v.shape[1:])))

        for key in [
            "scale",
        ]:
            v = params[key]
            if len(v) > 0:
                params[key] = 1 + (v - 1) * factor.view(
                    *((-1,) + (1,) * len(v.shape[1:]))
                )
        return params

    # def compute_transformation(self, input: torch.Tensor,
    #                            params: Dict[str, torch.Tensor],
    #                            flags: Dict[str, Any]) -> torch.Tensor:
    #     factor = torch.linspace(0, 1, input.shape[0]).to(input)
    #     for key in ["translations", "center", "angle", "shear_x", "shear_y"]:
    #         v = params[key]
    #         params[key] = v * factor.view(*((-1,)+(1,)*len(v.shape[1:])))

    #     for key in ["scale", ]:
    #         v = params[key]
    #         params[key] = 1 + (v-1) * factor.view(*((-1,)+(1,)*len(v.shape[1:])))

    #     return super().compute_transformation(input, params, flags)


class BasicPipeline:
    """transforms img, mask, and points.

    Only supports 2D transformations for now (any 3D object will preserve its z coordinates/dimensions)
    """

    def __init__(self, augs: tuple, filter_points: bool = True):
        self.data_keys = ("input", "mask", "keypoints")
        self.pipeline = K.AugmentationSequential(
            *augs,
            # disable align_corners to not trigger lots of warnings from kornia
            extra_args={
                DataKey.MASK: {"resample": Resample.NEAREST, "align_corners": False}
            },
            data_keys=self.data_keys,
        )
        self.filter_points = filter_points

    def __call__(
        self,
        img: np.ndarray,
        mask: np.ndarray,
        points: np.ndarray,
        timepoints: np.ndarray,
    ):
        ndim = img.ndim - 1
        assert (
            ndim in (2, 3)
            and points.ndim == 2
            and points.shape[-1] == ndim
            and timepoints.ndim == 1
            and img.shape == mask.shape
        )

        x = torch.from_numpy(img).float()
        y = torch.from_numpy(mask.astype(np.int64)).float()

        # if 2D add dummy channel
        if ndim == 2:
            x = x.unsqueeze(1)
            y = y.unsqueeze(1)
            p = points[..., [1, 0]]
        # if 3D we use z as channel (i.e. fix augs across z)
        elif ndim == 3:
            p = points[..., [2, 1]]

        # flip as kornia expects xy and not yx
        p = torch.from_numpy(p).unsqueeze(0).float()
        # add batch by duplicating to make kornia happy
        p = p.expand(len(x), -1, -1)
        # create a mask to know which timepoint the points belong to
        ts = torch.from_numpy(timepoints).long()
        n_points = p.shape[1]
        if n_points > 0:
            x, y, p = self.pipeline(x, y, p)
        else:
            # dummy keypoints
            x, y = self.pipeline(x, y, torch.zeros((len(x), 1, 2)))[:2]

        # remove batch
        p = p[ts, torch.arange(n_points)]
        # flip back
        p = p[..., [1, 0]]

        # remove channel
        if ndim == 2:
            x = x.squeeze(1)
            y = y.squeeze(1)

        x = x.numpy()
        y = y.numpy().astype(np.uint16)
        # p = p.squeeze(0).numpy()
        p = p.numpy()
        # add back z coordinates
        if ndim == 3:
            p = np.concatenate([points[..., 0:1], p], axis=-1)
        ts = ts.numpy()
        # remove points outside of img/mask

        if self.filter_points:
            idx = _filter_points(p, shape=x.shape[-ndim:])

        else:
            idx = np.arange(len(p), dtype=int)

        p = p[idx]
        return (x, y, p), idx


class RandomCrop:
    def __init__(
        self,
        crop_size: int | tuple[int] | None = None,
        ndim: int = 2,
        ensure_inside_points: bool = False,
        use_padding: bool = True,
        padding_mode="constant",
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
            crop_size = tuple(chain(*tuple((c, c) for c in crop_size)))
        elif len(crop_size) == 2 * ndim:
            pass
        else:
            raise ValueError(f"crop_size has to be of length 1, {ndim}, or {2 * ndim}")

        crop_size = np.array(crop_size)
        self._ndim = ndim
        self._crop_bounds = crop_size[::2], crop_size[1::2]
        self._use_padding = use_padding
        self._ensure_inside_points = ensure_inside_points
        self._rng = np.random.RandomState()
        self._padding_mode = padding_mode

    def crop_img(self, img: np.ndarray, corner: np.ndarray, crop_size: np.ndarray):
        if not img.ndim == self._ndim + 1:
            raise ValueError(
                f"img has to be 1 (time) + {self._ndim} spatial dimensions"
            )

        pad_left = np.maximum(0, -corner)
        pad_right = np.maximum(
            0, corner + crop_size - np.array(img.shape[-self._ndim :])
        )

        img = np.pad(
            img,
            ((0, 0), *tuple(np.stack((pad_left, pad_right)).T)),
            mode=self._padding_mode,
        )
        slices = (
            slice(None),
            *tuple(slice(c, c + s) for c, s in zip(corner + pad_left, crop_size)),
        )
        return img[slices]

    def crop_points(
        self, points: np.ndarray, corner: np.ndarray, crop_size: np.ndarray
    ):
        idx = _filter_points(points, shape=crop_size, origin=corner)
        return points[idx] - corner, idx

    def __call__(self, img: np.ndarray, mask: np.ndarray, points: np.ndarray):
        assert (
            img.ndim == self._ndim + 1
            and points.ndim == 2
            and points.shape[-1] == self._ndim
            and img.shape == mask.shape
        )

        points = points.astype(int)

        crop_size = self._rng.randint(self._crop_bounds[0], self._crop_bounds[1] + 1)
        # print(f'{crop_size=}')

        if self._ensure_inside_points:
            if len(points) == 0:
                print("No points given, cannot ensure inside points")
                return (img, mask, points), np.zeros((0,), int)

            # sample point and corner relative to it

            _idx = np.random.randint(len(points))
            corner = (
                points[_idx]
                - crop_size
                + 1
                + self._rng.randint(crop_size // 4, 3 * crop_size // 4)
            )
        else:
            corner = self._rng.randint(
                0, np.maximum(1, np.array(img.shape[-self._ndim :]) - crop_size)
            )

        if not self._use_padding:
            corner = np.maximum(0, corner)
            crop_size = np.minimum(
                crop_size, np.array(img.shape[-self._ndim :]) - corner
            )

        img = self.crop_img(img, corner, crop_size)
        mask = self.crop_img(mask, corner, crop_size)
        points, idx = self.crop_points(points, corner, crop_size)

        return (img, mask, points), idx


class AugmentationPipeline(BasicPipeline):
    """transforms img, mask, and points."""

    def __init__(self, p=0.5, filter_points=True, level=1):
        if level == 1:
            augs = [
                # Augmentations for all images in a window
                K.RandomHorizontalFlip(p=0.5, same_on_batch=True),
                K.RandomVerticalFlip(p=0.5, same_on_batch=True),
                K.RandomAffine(
                    degrees=180,
                    shear=(-10, 10, -10, 10),  # x_min, x_max, y_min, y_max
                    translate=(0.05, 0.05),
                    scale=(0.8, 1.2),  # x_min, x_max, y_min, y_max
                    p=p,
                    same_on_batch=True,
                ),
                K.RandomBrightness(
                    (0.5, 1.5), clip_output=False, p=p, same_on_batch=True
                ),
                K.RandomGaussianNoise(mean=0.0, std=0.03, p=p, same_on_batch=False),
            ]
        elif level == 2:
            # Crafted for DeepCell crop size 256
            augs = [
                # Augmentations for all images in a window
                K.RandomHorizontalFlip(p=0.5, same_on_batch=True),
                K.RandomVerticalFlip(p=0.5, same_on_batch=True),
                K.RandomAffine(
                    degrees=180,
                    shear=(-5, 5, -5, 5),  # x_min, x_max, y_min, y_max
                    translate=(0.03, 0.03),
                    scale=(0.8, 1.2),  # isotropic
                    p=p,
                    same_on_batch=True,
                ),
                # Anisotropic scaling
                K.RandomAffine(
                    degrees=0,
                    scale=(0.9, 1.1, 0.9, 1.1),  # x_min, x_max, y_min, y_max
                    p=p,
                    same_on_batch=True,
                ),
                # Independet augmentations for each image in window
                K.RandomAffine(
                    degrees=3,
                    shear=(-2, 2, -2, 2),  # x_min, x_max, y_min, y_max
                    translate=(0.04, 0.04),
                    p=p,
                    same_on_batch=False,
                ),
                # not implemented for points in kornia 0.7.0
                # K.RandomElasticTransform(alpha=50, sigma=5, p=p, same_on_batch=False),
                # Intensity-based augmentations
                K.RandomBrightness(
                    (0.5, 1.5), clip_output=False, p=p, same_on_batch=True
                ),
                K.RandomGaussianNoise(mean=0.0, std=0.03, p=p, same_on_batch=False),
            ]
        elif level == 3:
            # Crafted for DeepCell crop size 256
            augs = [
                # Augmentations for all images in a window
                K.RandomHorizontalFlip(p=0.5, same_on_batch=True),
                K.RandomVerticalFlip(p=0.5, same_on_batch=True),
                ConcatAffine(
                    [
                        K.RandomAffine(
                            degrees=180,
                            shear=(-5, 5, -5, 5),  # x_min, x_max, y_min, y_max
                            translate=(0.03, 0.03),
                            scale=(0.8, 1.2),  # isotropic
                            p=p,
                            same_on_batch=True,
                        ),
                        # Anisotropic scaling
                        K.RandomAffine(
                            degrees=0,
                            scale=(0.9, 1.1, 0.9, 1.1),  # x_min, x_max, y_min, y_max
                            p=p,
                            same_on_batch=True,
                        ),
                    ]
                ),
                RandomTemporalAffine(
                    degrees=10,
                    translate=(0.05, 0.05),
                    p=p,
                    # same_on_batch=True,
                ),
                # Independet augmentations for each image in window
                K.RandomAffine(
                    degrees=2,
                    shear=(-2, 2, -2, 2),  # x_min, x_max, y_min, y_max
                    translate=(0.01, 0.01),
                    p=0.5 * p,
                    same_on_batch=False,
                ),
                # Intensity-based augmentations
                RandomIntensityScaleShift(
                    (0.5, 2.0), (-0.1, 0.1), clip_output=False, p=p, same_on_batch=True
                ),
                K.RandomGaussianNoise(mean=0.0, std=0.03, p=p, same_on_batch=False),
            ]
        elif level == 4:
            # debug
            augs = [
                K.RandomAffine(
                    degrees=30,
                    shear=(-0, 0, -0, 0),  # x_min, x_max, y_min, y_max
                    translate=(0.0, 0.0),
                    p=1,
                    same_on_batch=True,
                ),
            ]
        else:
            raise ValueError(f"level {level} not supported")

        super().__init__(augs, filter_points)
