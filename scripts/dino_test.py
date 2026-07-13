"""Test mask-pooled DINO features on a 2D CTC sequence.

The script loads normalized images and optionally ST-refined TRA masks through
``TrackingSequence.from_ctc``. It runs the selected DINO model on a resized frame or overlapping
original-resolution tiles, pools its spatial patch tokens over each labeled
mask, and evaluates adjacent-frame identity retrieval.

Example:
    python dino_test.py \
        -i data/ctc_2026/2d/Fluo-C2DL-Huh7/01 \
        --max-frames 20
"""

import argparse
import logging
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from scipy.spatial.distance import cdist
from trackastra.data import TrackingSequence
from transformers import AutoImageProcessor, AutoModel

logger = logging.getLogger(__name__)

MODEL_ALIASES = {
    "nearest-neighbor": "coordinates",
    "dinov3-vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "dinov3-vits16": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "dino-vits8": "facebook/dino-vits8",
    "dinov3-convnext-tiny-s8": "timm:convnext_tiny.dinov3_lvd1689m@1",
    "dinov3-convnext-tiny-s16": "timm:convnext_tiny.dinov3_lvd1689m@2",
    "dinov3-convnext-small-s8": "timm:convnext_small.dinov3_lvd1689m@1",
    "dinov3-convnext-small-s16": "timm:convnext_small.dinov3_lvd1689m@2",
    "dinov3-convnext-base-s8": "timm:convnext_base.dinov3_lvd1689m@1",
    "dinov3-convnext-base-s16": "timm:convnext_base.dinov3_lvd1689m@2",
}
DEFAULT_MODEL = "dinov3-vitb16"
DEFAULT_INPUT = Path("data/ctc_2026/2d/Fluo-C2DL-Huh7/01")
REFERENCE_BATCH_IMAGE_SIZE = 448
RANDOM_PROJECTION_SEED = 0
RESULT_COLUMNS = (
    "dataset",
    "model",
    "randpro",
    "scale",
    "resize",
    "batch_size",
    "diameter_dino_px",
    "time_per_frame_ms",
    "correct_sim",
    "incorrect_sim",
    "margin_sim",
    "top1_acc",
    "daughter_recall",
)


@dataclass(frozen=True)
class FrameFeatures:
    """Object labels, centroids, and DINO features for one frame."""

    frame: int
    labels: np.ndarray
    lineages: np.ndarray
    centroids: np.ndarray
    features: np.ndarray


@dataclass(frozen=True)
class MovieExtraction:
    """Unprojected object features and extraction metadata for one variant."""

    frames: tuple[FrameFeatures, ...]
    effective_batch_size: int
    median_dino_diameter: float
    time_per_frame_ms: float


@dataclass(frozen=True)
class PreparedFrame:
    """Scaled frame data and supervision needed for DINO extraction."""

    local_index: int
    frame: int
    image: np.ndarray
    mask: np.ndarray
    detection_indices: np.ndarray
    labels: np.ndarray
    centroids: np.ndarray
    lineages: np.ndarray
    pooling_plans: dict[tuple[Any, ...], "MaskPoolingPlan"] = field(
        default_factory=dict, compare=False, repr=False
    )


@dataclass(frozen=True)
class MaskPoolingPlan:
    """Sparse linear map from a spatial token grid to mask-object means."""

    labels: np.ndarray
    centroids: np.ndarray
    weights: torch.Tensor
    object_weights: np.ndarray


@dataclass(frozen=True)
class LoadedMovie:
    """CTC images, masks, and supervision shared by all model variants."""

    root: Path
    frame_numbers: tuple[int, ...]
    images: Sequence[np.ndarray]
    masks: Sequence[np.ndarray]
    detections: Any
    supervision: Any
    lineage_parents: np.ndarray
    object_diameters: tuple[np.ndarray, ...]
    prepared_frames: dict[float, tuple[PreparedFrame, ...]] = field(
        default_factory=dict, compare=False, repr=False
    )


@dataclass(frozen=True)
class SpatialFeatureConfig:
    """Spatial output metadata shared by transformer and convolutional models."""

    patch_size: int
    hidden_size: int


@dataclass(frozen=True)
class ProcessorOutput:
    """Minimal image-processor output compatible with the extraction path."""

    pixel_values: torch.Tensor


class TimmImageProcessor:
    """Resize and normalize image batches according to a timm pretrained config."""

    def __init__(
        self,
        mean: tuple[float, ...],
        std: tuple[float, ...],
        interpolation: str,
    ) -> None:
        if interpolation != "bicubic":
            raise ValueError(
                f"Only bicubic timm interpolation is supported, got {interpolation!r}"
            )
        self.mean = torch.tensor(mean, dtype=torch.float32)[None, :, None, None]
        self.std = torch.tensor(std, dtype=torch.float32)[None, :, None, None]

    def __call__(
        self,
        images: Sequence[np.ndarray],
        size: dict[str, int] | None = None,
        do_resize: bool = True,
        return_tensors: str = "pt",
    ) -> ProcessorOutput:
        """Return a normalized BCHW float tensor."""
        if return_tensors != "pt":
            raise ValueError(f"Expected return_tensors='pt', got {return_tensors!r}")
        try:
            arrays = np.stack(images)
        except ValueError as error:
            raise ValueError("All images in a timm batch must have the same shape") from error
        if arrays.ndim != 4 or arrays.shape[-1] != 3:
            raise ValueError(f"Expected BHWC RGB images, got shape {arrays.shape}")
        pixel_values = (
            torch.from_numpy(arrays).permute(0, 3, 1, 2).float().div_(255.0)
        )
        if do_resize:
            if size is None or set(size) != {"height", "width"}:
                raise ValueError(f"Expected height and width resize values, got {size}")
            pixel_values = F.interpolate(
                pixel_values,
                size=(size["height"], size["width"]),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
        pixel_values = (pixel_values - self.mean) / self.std
        return ProcessorOutput(pixel_values)


def _resolve_model_name(model_spec: str) -> str:
    """Resolve a short alias or an explicit timm DINOv3 ConvNeXt name."""
    model_name = MODEL_ALIASES.get(model_spec, model_spec)
    if model_name.startswith("convnext_") and ".dinov3_" in model_name:
        return f"timm:{model_name}"
    return model_name


def _load_timm_feature_model(model_spec: str) -> tuple[Any, Any, int]:
    """Load one pretrained timm model exposing one spatial feature level."""
    import timm

    model_name, separator, output_index_text = model_spec.partition("@")
    output_index = int(output_index_text) if separator else 1
    if output_index not in {0, 1, 2, 3}:
        raise ValueError(f"Timm output index must be in [0, 3], got {output_index}")
    model = timm.create_model(
        model_name,
        pretrained=True,
        features_only=True,
        out_indices=(output_index,),
    ).eval()
    reductions = model.feature_info.reduction()
    channels = model.feature_info.channels()
    if len(reductions) != 1 or len(channels) != 1:
        raise ValueError(
            f"Expected one timm feature output, got reductions={reductions}, "
            f"channels={channels}"
        )
    stride = int(reductions[0])
    feature_dim = int(channels[0])
    pretrained_config = model.pretrained_cfg
    crop_pct = float(pretrained_config.get("crop_pct", 1.0))
    if crop_pct != 1.0:
        raise ValueError(f"Expected timm crop_pct=1, got {crop_pct}")
    processor = TimmImageProcessor(
        tuple(pretrained_config["mean"]),
        tuple(pretrained_config["std"]),
        str(pretrained_config["interpolation"]),
    )
    model.config = SpatialFeatureConfig(stride, feature_dim)
    return processor, model, stride


def _parse_on_off(value: str) -> bool:
    """Parse an on/off command-line value."""
    normalized = value.lower()
    if normalized in {"on", "true", "1"}:
        return True
    if normalized in {"off", "false", "0"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected on or off, got {value!r}")


def _as_rgb_uint8(image: np.ndarray) -> np.ndarray:
    """Convert a loader-normalized 2D microscopy image to RGB uint8."""
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D image, got shape {image.shape}")
    image = np.clip(image, 0.0, 1.0)
    image = np.round(255.0 * image).astype(np.uint8)
    return np.repeat(image[..., None], 3, axis=-1)


def _centroids(mask: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Return yx centroids for the requested nonzero labels."""
    result = np.empty((len(labels), 2), dtype=np.float32)
    for i, label in enumerate(labels):
        coords = np.argwhere(mask == label)
        if len(coords) == 0:
            raise ValueError(f"Label {label} has no pixels")
        result[i] = coords.mean(axis=0)
    return result


def _scale_image_mask(
    image: np.ndarray, mask: np.ndarray, scale: float
) -> tuple[np.ndarray, np.ndarray]:
    """Scale an image bilinearly and its label mask with nearest neighbors."""
    if scale == 1.0:
        return image, mask
    output_size = tuple(max(1, round(size * scale)) for size in image.shape)
    scaled_image = F.interpolate(
        torch.from_numpy(image.astype(np.float32, copy=True))[None, None],
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )[0, 0].numpy()
    scaled_mask = F.interpolate(
        torch.from_numpy(mask.astype(np.float32, copy=True))[None, None],
        size=output_size,
        mode="nearest",
    )[0, 0].numpy()
    return scaled_image, scaled_mask.astype(mask.dtype, copy=False)


def _model_hidden_states(model, pixel_values: torch.Tensor) -> torch.Tensor:
    """Run a DINO model while preserving the batch dimension."""
    kwargs = {}
    embeddings = getattr(model, "embeddings", None)
    if hasattr(embeddings, "interpolate_pos_encoding"):
        kwargs["interpolate_pos_encoding"] = True
    with torch.inference_mode():
        if hasattr(model, "feature_info"):
            feature_maps = model(pixel_values)
            if len(feature_maps) != 1 or feature_maps[0].ndim != 4:
                shapes = [tuple(feature.shape) for feature in feature_maps]
                raise ValueError(f"Expected one BCHW timm feature map, got {shapes}")
            return feature_maps[0].flatten(2).transpose(1, 2)
        return model(pixel_values=pixel_values, **kwargs).last_hidden_state


def _spatial_tokens(
    hidden: torch.Tensor, grid_height: int, grid_width: int
) -> torch.Tensor:
    """Remove leading class/register tokens and return the spatial patch tokens."""
    expected_tokens = grid_height * grid_width
    num_special_tokens = len(hidden) - expected_tokens
    if num_special_tokens < 0:
        raise ValueError(
            f"Expected at least {expected_tokens} spatial tokens, got {len(hidden)} total"
        )
    patch_tokens = hidden[num_special_tokens:]
    if len(patch_tokens) != expected_tokens:
        raise ValueError(
            f"Expected {expected_tokens} spatial tokens, got {len(patch_tokens)}"
        )
    return patch_tokens


def _interpolation_axis_weights(
    output_size: int, input_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return source indices and weights for align_corners=False interpolation."""
    coordinates = (
        (np.arange(output_size, dtype=np.float32) + 0.5)
        * np.float32(input_size / output_size)
        - 0.5
    )
    lower_unclipped = np.floor(coordinates).astype(np.int64)
    upper_unclipped = lower_unclipped + 1
    upper_weights = coordinates - lower_unclipped
    lower_weights = 1.0 - upper_weights
    lower = np.clip(lower_unclipped, 0, input_size - 1)
    upper = np.clip(upper_unclipped, 0, input_size - 1)
    return (
        lower,
        upper,
        lower_weights.astype(np.float32),
        upper_weights.astype(np.float32),
    )


def _build_mask_pooling_plan(
    mask: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    grid_shape: tuple[int, int],
    pixel_weights: np.ndarray | None = None,
    normalize: bool = True,
    upsample_shape: tuple[int, int] | None = None,
) -> MaskPoolingPlan:
    """Build an exact sparse equivalent of upsample-then-average pooling."""
    grid_height, grid_width = grid_shape
    if upsample_shape is None:
        upsample_shape = mask.shape
    if mask.shape[0] > upsample_shape[0] or mask.shape[1] > upsample_shape[1]:
        raise ValueError(
            f"Mask shape {mask.shape} exceeds upsample shape {upsample_shape}"
        )
    if pixel_weights is not None and pixel_weights.shape != mask.shape:
        raise ValueError(
            f"Pixel weights shape {pixel_weights.shape} differs from mask shape "
            f"{mask.shape}"
        )
    foreground_y, foreground_x = np.nonzero(mask)
    if len(foreground_y) == 0:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Sparse invariant checks are implicitly disabled"
            )
            weights = torch.sparse_coo_tensor(
                torch.empty((2, 0), dtype=torch.int64),
                torch.empty(0, dtype=torch.float32),
                (0, grid_height * grid_width),
                check_invariants=False,
                is_coalesced=True,
            ).coalesce()
        return MaskPoolingPlan(
            labels, centroids, weights, np.zeros(len(labels), dtype=np.float32)
        )

    object_indices = np.searchsorted(labels, mask[foreground_y, foreground_x])
    foreground_weights = (
        np.ones(len(object_indices), dtype=np.float32)
        if pixel_weights is None
        else pixel_weights[foreground_y, foreground_x].astype(np.float32, copy=False)
    )
    object_weights = np.bincount(
        object_indices,
        weights=foreground_weights,
        minlength=len(labels),
    ).astype(np.float32)
    y0, y1, wy0, wy1 = _interpolation_axis_weights(
        upsample_shape[0], grid_height
    )
    x0, x1, wx0, wx1 = _interpolation_axis_weights(
        upsample_shape[1], grid_width
    )

    rows = np.concatenate([object_indices] * 4)
    columns = np.concatenate(
        [
            y0[foreground_y] * grid_width + x0[foreground_x],
            y0[foreground_y] * grid_width + x1[foreground_x],
            y1[foreground_y] * grid_width + x0[foreground_x],
            y1[foreground_y] * grid_width + x1[foreground_x],
        ]
    )
    values = np.concatenate(
        [
            wy0[foreground_y] * wx0[foreground_x],
            wy0[foreground_y] * wx1[foreground_x],
            wy1[foreground_y] * wx0[foreground_x],
            wy1[foreground_y] * wx1[foreground_x],
        ]
    )
    values *= np.tile(foreground_weights, 4)
    if normalize:
        values /= object_weights[rows]
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Sparse invariant checks are implicitly disabled"
        )
        weights = torch.sparse_coo_tensor(
            torch.from_numpy(np.stack((rows, columns))).long(),
            torch.from_numpy(values).float(),
            (len(labels), grid_height * grid_width),
            check_invariants=False,
            is_coalesced=False,
        ).coalesce()
    return MaskPoolingPlan(labels, centroids, weights, object_weights)


def _pool_mask_features(
    mask: np.ndarray,
    hidden: torch.Tensor,
    processed_size: tuple[int, int],
    model,
    device: torch.device,
    pooling_plans: dict[tuple[Any, ...], MaskPoolingPlan] | None = None,
    labels: np.ndarray | None = None,
    centroids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pool one frame's spatial DINO tokens over its instance masks.

    A cached sparse linear map exactly combines bilinear upsampling and
    per-instance averaging without materializing a dense full-resolution map.
    """
    if labels is None:
        labels = np.unique(mask)
        labels = labels[labels != 0]
    if len(labels) == 0:
        return labels, np.empty((0, 2), np.float32), np.empty((0, 0), np.float32)
    if centroids is None:
        centroids = _centroids(mask, labels)

    patch_size = int(model.config.patch_size)
    grid_height = processed_size[0] // patch_size
    grid_width = processed_size[1] // patch_size
    patch_tokens = _spatial_tokens(hidden, grid_height, grid_width)

    grid_shape = (grid_height, grid_width)
    if pooling_plans is not None and grid_shape in pooling_plans:
        plan = pooling_plans[grid_shape]
    else:
        plan = _build_mask_pooling_plan(mask, labels, centroids, grid_shape)
        if pooling_plans is not None:
            pooling_plans[grid_shape] = plan

    pooling_weights = plan.weights.to(device)
    pooled = torch.sparse.mm(pooling_weights, patch_tokens.float())

    features = F.normalize(pooled, dim=1).cpu().float().numpy()
    return plan.labels, plan.centroids, features


def extract_mask_features(
    image: np.ndarray,
    mask: np.ndarray,
    processor,
    model,
    device: torch.device,
    image_size: int,
    resize: bool = True,
    pooling_plans: dict[tuple[Any, ...], MaskPoolingPlan] | None = None,
    labels: np.ndarray | None = None,
    centroids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract one mask-pooled feature per object from one frame."""
    return extract_mask_features_batch(
        [image],
        [mask],
        processor,
        model,
        device,
        image_size,
        resize=resize,
        pooling_plan_caches=[pooling_plans],
        mask_labels=[labels],
        mask_centroids=[centroids],
    )[0]


def extract_mask_features_batch(
    images: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    processor,
    model,
    device: torch.device,
    image_size: int,
    resize: bool = True,
    pooling_plan_caches: Sequence[
        dict[tuple[Any, ...], MaskPoolingPlan] | None
    ]
    | None = None,
    mask_labels: Sequence[np.ndarray | None] | None = None,
    mask_centroids: Sequence[np.ndarray | None] | None = None,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Extract mask-pooled features from a batch of optionally resized frames."""
    if len(images) != len(masks):
        raise ValueError(f"Got {len(images)} images and {len(masks)} masks")
    if not images:
        return []
    if pooling_plan_caches is None:
        pooling_plan_caches = [None] * len(images)
    if mask_labels is None:
        mask_labels = [None] * len(images)
    if mask_centroids is None:
        mask_centroids = [None] * len(images)
    if len(pooling_plan_caches) != len(images):
        raise ValueError(
            f"Got {len(pooling_plan_caches)} pooling caches for {len(images)} images"
        )
    if len(mask_labels) != len(images) or len(mask_centroids) != len(images):
        raise ValueError("Mask metadata count must match the image count")
    for image, mask in zip(images, masks):
        if image.shape != mask.shape:
            raise ValueError(
                f"Image and mask shapes differ: {image.shape} versus {mask.shape}"
            )
    processor_images = [_as_rgb_uint8(image) for image in images]
    if resize:
        inputs = processor(
            images=processor_images,
            size={"height": image_size, "width": image_size},
            return_tensors="pt",
        )
    else:
        inputs = processor(
            images=processor_images,
            do_resize=False,
            return_tensors="pt",
        )
    pixel_values = inputs.pixel_values.to(device)
    hidden_states = _model_hidden_states(model, pixel_values)
    processed_size = tuple(pixel_values.shape[-2:])
    return [
        _pool_mask_features(
            mask,
            hidden,
            processed_size,
            model,
            device,
            pooling_plans,
            labels,
            centroids,
        )
        for mask, hidden, pooling_plans, labels, centroids in zip(
            masks,
            hidden_states,
            pooling_plan_caches,
            mask_labels,
            mask_centroids,
        )
    ]


def _tile_starts(size: int, tile_size: int, overlap: int) -> tuple[int, list[int]]:
    """Cover one axis with overlapping tiles, adjusting strides to fit exactly."""
    padded_size = max(size, tile_size)
    if padded_size == tile_size:
        return padded_size, [0]
    stride = tile_size - overlap
    n_tiles = int(np.ceil((padded_size - tile_size) / stride)) + 1
    starts = np.rint(np.linspace(0, padded_size - tile_size, n_tiles)).astype(int)
    return padded_size, starts.tolist()


def _tile_blend_weights(tile_size: int) -> np.ndarray:
    """Return edge-tapered weights for blending overlapping tile predictions."""
    axis = np.maximum(np.hanning(tile_size), 1e-3).astype(np.float32)
    return axis[:, None] * axis[None, :]


def _ctc_image_folder(root: Path) -> Path:
    """Resolve the raw-image folder for the CTC path forms accepted by from_ctc."""
    if root.name == "TRA":
        root = root.parent.parent / root.parent.name.split("_")[0]
    return root / "img" if (root / "img").exists() else root


def _dataset_name(root: Path) -> str:
    """Return the input path relative to its last data directory."""
    if root.name == "TRA":
        root = root.parent.parent / root.parent.name.split("_")[0]
    data_indices = [i for i, part in enumerate(root.parts) if part == "data"]
    if data_indices:
        relative_parts = root.parts[data_indices[-1] + 1 :]
        if relative_parts:
            return Path(*relative_parts).as_posix()
    return f"{root.parent.name}/{root.name}"


def _slice_fraction(index: int, n_frames: int) -> float:
    """Convert an exact frame boundary to the fraction consumed by from_ctc."""
    if index == n_frames:
        return 1.0
    fraction = index / n_frames
    if int(n_frames * fraction) < index:
        fraction = float(np.nextafter(fraction, 1.0))
    return fraction


def _frame_bounds(
    n_frames: int, start_frame: int | None, max_frames: int
) -> tuple[int, int]:
    """Return the selected half-open frame interval."""
    if start_frame is None:
        start_frame = 0 if max_frames == 0 else max(0, n_frames - max_frames)
    stop_frame = (
        n_frames if max_frames == 0 else min(start_frame + max_frames, n_frames)
    )
    return start_frame, stop_frame


def extract_mask_features_tiled(
    image: np.ndarray,
    mask: np.ndarray,
    processor,
    model,
    device: torch.device,
    tile_size: int,
    batch_size: int = 8,
    pooling_plans: dict[tuple[Any, ...], MaskPoolingPlan] | None = None,
    labels: np.ndarray | None = None,
    centroids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract mask features from overlapping tiles without resizing the image.

    Tiles overlap by at least one quarter of their size. Their upsampled patch
    features are blended at original image resolution, then averaged over each
    complete mask.
    """
    if image.shape != mask.shape:
        raise ValueError(
            f"Image and mask shapes differ: {image.shape} versus {mask.shape}"
        )
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D image, got shape {image.shape}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    if labels is None:
        labels = np.unique(mask)
        labels = labels[labels != 0]
    if len(labels) == 0:
        return labels, np.empty((0, 2), np.float32), np.empty((0, 0), np.float32)
    if centroids is None:
        centroids = _centroids(mask, labels)

    height, width = image.shape
    overlap = tile_size // 4
    padded_height, y_starts = _tile_starts(height, tile_size, overlap)
    padded_width, x_starts = _tile_starts(width, tile_size, overlap)
    padded_image = np.pad(
        image,
        ((0, padded_height - height), (0, padded_width - width)),
        mode="edge",
    )

    blend = _tile_blend_weights(tile_size)
    coverage = np.zeros((height, width), dtype=np.float32)
    tiles = []
    for y in y_starts:
        for x in x_starts:
            valid_height = min(tile_size, height - y)
            valid_width = min(tile_size, width - x)
            tiles.append((y, x, valid_height, valid_width))
            coverage[y : y + valid_height, x : x + valid_width] += blend[
                :valid_height, :valid_width
            ]
    if np.any(coverage <= 0):
        raise ValueError("Tiled prediction did not cover the complete image")

    pooled = torch.zeros(
        (len(labels), int(model.config.hidden_size)), dtype=torch.float32, device=device
    )
    object_weights = torch.zeros(len(labels), dtype=torch.float32, device=device)
    patch_size = int(model.config.patch_size)
    grid_size = tile_size // patch_size

    for batch_start in range(0, len(tiles), batch_size):
        tile_batch = tiles[batch_start : batch_start + batch_size]
        tile_images = [
            _as_rgb_uint8(padded_image[y : y + tile_size, x : x + tile_size])
            for y, x, _, _ in tile_batch
        ]
        inputs = processor(
            images=tile_images,
            do_resize=False,
            return_tensors="pt",
        )
        pixel_values = inputs.pixel_values.to(device)
        if pixel_values.shape[-2:] != (tile_size, tile_size):
            raise ValueError(
                f"Expected unresized {tile_size} x {tile_size} tiles, got "
                f"{tuple(pixel_values.shape[-2:])}"
            )
        hidden_states = _model_hidden_states(model, pixel_values)

        for (y, x, valid_height, valid_width), hidden in zip(tile_batch, hidden_states):
            patch_tokens = _spatial_tokens(hidden, grid_size, grid_size)
            local_mask = mask[y : y + valid_height, x : x + valid_width]
            if not np.any(local_mask):
                continue
            normalized_weight = (
                blend[:valid_height, :valid_width]
                / coverage[y : y + valid_height, x : x + valid_width]
            )
            plan_key = (
                "tile",
                grid_size,
                y,
                x,
                valid_height,
                valid_width,
            )
            if pooling_plans is not None and plan_key in pooling_plans:
                plan = pooling_plans[plan_key]
            else:
                plan = _build_mask_pooling_plan(
                    local_mask,
                    labels,
                    centroids,
                    (grid_size, grid_size),
                    pixel_weights=normalized_weight,
                    normalize=False,
                    upsample_shape=(tile_size, tile_size),
                )
                if pooling_plans is not None:
                    pooling_plans[plan_key] = plan
            pooled += torch.sparse.mm(plan.weights.to(device), patch_tokens.float())
            object_weights += torch.from_numpy(plan.object_weights).to(device)

    if torch.any(object_weights == 0):
        raise ValueError("At least one mask label received no tiled feature samples")
    pooled /= object_weights[:, None]
    features = F.normalize(pooled, dim=1).cpu().numpy()
    return labels, centroids, features


def _evaluate_pair(
    previous: FrameFeatures,
    current: FrameFeatures,
    coordinate_baseline: bool = False,
) -> dict[str, float]:
    """Compare each true match with its spatially nearest incorrect target."""
    target_by_label = {int(label): i for i, label in enumerate(current.labels)}
    source_indices = [
        i for i, label in enumerate(previous.labels) if int(label) in target_by_label
    ]
    if not source_indices:
        return {
            "correct_n": 0.0,
            "comparison_n": 0.0,
            "correct_similarity": 0.0,
            "compared_correct_similarity": 0.0,
            "incorrect_similarity": 0.0,
            "correct_wins": 0.0,
            "top1_correct": 0.0,
        }

    source = np.asarray(source_indices)
    truth = np.asarray(
        [target_by_label[int(previous.labels[i])] for i in source_indices],
        dtype=np.int64,
    )
    if coordinate_baseline:
        similarities = 1.0 / (
            1.0 + cdist(previous.centroids[source], current.centroids)
        )
    else:
        similarities = previous.features[source] @ current.features.T
    rows = np.arange(len(source))
    correct_similarity = similarities[rows, truth]
    if current.features.shape[0] < 2:
        return {
            "correct_n": float(len(source)),
            "comparison_n": 0.0,
            "correct_similarity": float(correct_similarity.sum()),
            "compared_correct_similarity": 0.0,
            "incorrect_similarity": 0.0,
            "correct_wins": 0.0,
            "top1_correct": 0.0,
        }

    distances = cdist(previous.centroids[source], current.centroids)
    distances[rows, truth] = np.inf
    nearest_incorrect = np.argmin(distances, axis=1)
    incorrect_similarity = similarities[rows, nearest_incorrect]

    return {
        "correct_n": float(len(source)),
        "comparison_n": float(len(source)),
        "correct_similarity": float(correct_similarity.sum()),
        "compared_correct_similarity": float(correct_similarity.sum()),
        "incorrect_similarity": float(incorrect_similarity.sum()),
        "correct_wins": float((correct_similarity > incorrect_similarity).sum()),
        "top1_correct": float((np.argmax(similarities, axis=1) == truth).sum()),
    }


def _evaluate_divisions(
    previous: FrameFeatures,
    current: FrameFeatures,
    lineage_parents: np.ndarray,
    coordinate_baseline: bool = False,
) -> dict[str, float]:
    """Evaluate whether each parent retrieves both daughters in its top two."""
    previous_by_lineage = {
        int(lineage): i for i, lineage in enumerate(previous.lineages) if lineage >= 0
    }
    daughters_by_parent: dict[int, list[int]] = {}
    for object_index, lineage in enumerate(current.lineages):
        if lineage < 0:
            continue
        parent = int(lineage_parents[int(lineage)])
        if parent in previous_by_lineage:
            daughters_by_parent.setdefault(parent, []).append(object_index)

    totals = {
        "division_n": 0.0,
        "daughter_links": 0.0,
        "daughter_hits_at2": 0.0,
        "complete_division_hits_at2": 0.0,
        "parent_daughter_similarity": 0.0,
        "division_margin": 0.0,
        "division_margin_n": 0.0,
    }
    for parent, daughters in daughters_by_parent.items():
        if len(daughters) != 2:
            continue
        parent_index = previous_by_lineage[parent]
        if coordinate_baseline:
            distances = np.linalg.norm(
                current.centroids - previous.centroids[parent_index], axis=1
            )
            similarities = 1.0 / (1.0 + distances)
        else:
            parent_feature = previous.features[parent_index]
            similarities = current.features @ parent_feature
        top2 = set(np.argsort(similarities)[-2:])
        daughter_set = set(daughters)
        daughter_hits = len(top2 & daughter_set)
        daughter_similarities = similarities[daughters]

        totals["division_n"] += 1.0
        totals["daughter_links"] += 2.0
        totals["daughter_hits_at2"] += float(daughter_hits)
        totals["complete_division_hits_at2"] += float(daughter_hits == 2)
        totals["parent_daughter_similarity"] += float(daughter_similarities.sum())

        non_daughters = np.ones(len(current.labels), dtype=bool)
        non_daughters[daughters] = False
        if non_daughters.any():
            totals["division_margin"] += float(
                daughter_similarities.min() - similarities[non_daughters].max()
            )
            totals["division_margin_n"] += 1.0
    return totals


def evaluate_sequence(
    frames: list[FrameFeatures],
    lineage_parents: np.ndarray,
    coordinate_baseline: bool = False,
) -> dict[str, float]:
    """Aggregate persistence and division retrieval metrics over a sequence."""
    totals = {
        "correct_n": 0.0,
        "comparison_n": 0.0,
        "correct_similarity": 0.0,
        "compared_correct_similarity": 0.0,
        "incorrect_similarity": 0.0,
        "correct_wins": 0.0,
        "top1_correct": 0.0,
        "division_n": 0.0,
        "daughter_links": 0.0,
        "daughter_hits_at2": 0.0,
        "complete_division_hits_at2": 0.0,
        "parent_daughter_similarity": 0.0,
        "division_margin": 0.0,
        "division_margin_n": 0.0,
    }
    for previous, current in pairwise(frames):
        if current.frame != previous.frame + 1:
            logger.warning(
                "Skipping non-adjacent frames %d and %d", previous.frame, current.frame
            )
            continue
        pair = _evaluate_pair(previous, current, coordinate_baseline)
        for key, value in pair.items():
            totals[key] += value
        divisions = _evaluate_divisions(
            previous, current, lineage_parents, coordinate_baseline
        )
        for key, value in divisions.items():
            totals[key] += value
    return totals


def _inputs_from_config(config_path: Path) -> list[Path]:
    """Read legacy and grouped CTC paths from a training config's input_train."""
    with config_path.open() as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}")

    sources = config.get("input_train")
    if not isinstance(sources, list) or not sources:
        raise ValueError(f"Config {config_path} has no nonempty input_train list")

    inputs: list[Path] = []
    for source_index, source in enumerate(sources):
        if isinstance(source, str):
            inputs.append(Path(source))
            continue
        if not isinstance(source, dict):
            raise ValueError(
                f"input_train entry {source_index} in {config_path} must be a path "
                "or source mapping"
            )

        source_format = source.get("format", "ctc")
        if source_format != "ctc":
            raise ValueError(
                f"input_train entry {source_index} in {config_path} has format "
                f"{source_format!r}; dino_test.py supports only CTC inputs"
            )
        paths = source.get("paths")
        if isinstance(paths, str):
            paths = [paths]
        if not isinstance(paths, list) or not paths:
            raise ValueError(
                f"CTC input_train entry {source_index} in {config_path} has no paths"
            )
        if not all(isinstance(path, str) for path in paths):
            raise ValueError(
                f"CTC input_train entry {source_index} in {config_path} contains "
                "a non-string path"
            )
        inputs.extend(Path(path) for path in paths)
    return inputs


def _ratio(numerator: float, denominator: float) -> float:
    """Return a ratio or NaN when its denominator is zero."""
    return numerator / denominator if denominator else float("nan")


def _make_random_projection(input_dim: int, output_dim: int) -> np.ndarray:
    """Create deterministic random unit vectors for feature projection."""
    rng = np.random.default_rng(RANDOM_PROJECTION_SEED)
    projection = rng.standard_normal((input_dim, output_dim), dtype=np.float32)
    projection /= np.linalg.norm(projection, axis=0, keepdims=True)
    return projection


def _project_features(
    features: np.ndarray, projection: np.ndarray
) -> np.ndarray:
    """Project features and restore unit length for cosine comparison."""
    if features.shape[0] == 0:
        return np.empty((0, projection.shape[1]), dtype=np.float32)
    if features.shape[1] != projection.shape[0]:
        raise ValueError(
            f"Feature dimension {features.shape[1]} differs from random projection "
            f"input dimension {projection.shape[0]}"
        )
    projected = features @ projection
    norms = np.linalg.norm(projected, axis=1, keepdims=True)
    return projected / np.maximum(norms, np.finfo(np.float32).eps)


def _results_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Build the compact, rounded result table."""
    return pd.DataFrame(rows).loc[:, RESULT_COLUMNS].round(4)


def _load_movie(root: Path, args: argparse.Namespace) -> LoadedMovie:
    """Load one selected CTC frame range for reuse by all model variants."""
    image_folder = _ctc_image_folder(root)
    image_paths = sorted((*image_folder.glob("*.tif"), *image_folder.glob("*.tiff")))
    if not image_paths:
        raise ValueError(f"No TIFF images found in {image_folder}")
    n_available_frames = len(image_paths)
    start_frame, stop_frame = _frame_bounds(
        n_available_frames, args.start_frame, args.max_frames
    )
    frame_numbers = tuple(range(start_frame, stop_frame))
    if len(frame_numbers) < 2:
        raise ValueError(
            f"Need at least two selected CTC frames, found {len(frame_numbers)}"
        )
    slice_pct = (
        _slice_fraction(start_frame, n_available_frames),
        _slice_fraction(stop_frame, n_available_frames),
    )

    logger.info(
        "Loading and caching CTC frames %d:%d through TrackingSequence.from_ctc: %s",
        start_frame,
        stop_frame,
        root,
    )
    sequence = TrackingSequence.from_ctc(
        root,
        ndim=2,
        detection_folders=("TRA",),
        slice_pct=slice_pct,
        n_workers=args.n_workers,
        load_images=True,
    )
    images = sequence.images
    detections = sequence.detections[0]
    supervision = sequence.supervision[0]
    masks = detections.masks
    if images is None or masks is None:
        raise ValueError(
            "TrackingSequence.from_ctc did not return attached images and masks"
        )
    if sequence.gt is None or supervision is None:
        raise ValueError("Division evaluation requires CTC lineage supervision")
    if len(images) != len(frame_numbers):
        raise ValueError(
            f"Requested {len(frame_numbers)} CTC frames, loader returned {len(images)}"
        )
    object_diameters = []
    for mask in masks:
        labels, counts = np.unique(mask, return_counts=True)
        object_diameters.append(2.0 * np.sqrt(counts[labels != 0] / np.pi))
    return LoadedMovie(
        root=root,
        frame_numbers=frame_numbers,
        images=images,
        masks=masks,
        detections=detections,
        supervision=supervision,
        lineage_parents=sequence.gt.lineage_parents,
        object_diameters=tuple(object_diameters),
    )


def _prepare_movie_frames(
    movie: LoadedMovie, scale: float
) -> tuple[PreparedFrame, ...]:
    """Scale and annotate frames once per input movie and scale."""
    if scale in movie.prepared_frames:
        return movie.prepared_frames[scale]

    prepared_frames = []
    for local_index, frame in enumerate(movie.frame_numbers):
        detection_indices = np.flatnonzero(
            movie.detections.timepoints == local_index
        )
        image, mask = _scale_image_mask(
            movie.images[local_index], movie.masks[local_index], scale
        )
        labels = np.unique(mask)
        labels = labels[labels != 0]
        centroids = _centroids(mask, labels)
        lineage_by_label = {
            int(movie.detections.labels[i]): int(movie.supervision.lineage_index[i])
            for i in detection_indices
        }
        lineages = np.asarray(
            [lineage_by_label.get(int(label), -1) for label in labels],
            dtype=np.int64,
        )
        prepared_frames.append(
            PreparedFrame(
                local_index=local_index,
                frame=frame,
                image=image,
                mask=mask,
                detection_indices=detection_indices,
                labels=labels,
                centroids=centroids,
                lineages=lineages,
            )
        )
    result = tuple(prepared_frames)
    movie.prepared_frames[scale] = result
    return result


def _extract_movie(
    movie: LoadedMovie,
    args: argparse.Namespace,
    processor: Any,
    model: Any,
    device: torch.device,
    feature_extractor: Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]] | None,
) -> MovieExtraction:
    """Extract unprojected object features once for one model variant."""
    frame_numbers = movie.frame_numbers
    images = movie.images
    masks = movie.masks
    prepared_frames = _prepare_movie_frames(movie, args.scale)

    effective_batch_size = args.batch_size
    if args.full_resolution:
        scaled_shapes = [
            tuple(max(1, round(size * args.scale)) for size in image.shape)
            for image in images
        ]
        largest_shape = max(scaled_shapes, key=lambda shape: shape[0] * shape[1])
        relative_area = max(
            1.0,
            largest_shape[0]
            * largest_shape[1]
            / REFERENCE_BATCH_IMAGE_SIZE**2,
        )
        effective_batch_size = max(1, int(args.batch_size / relative_area))
        logger.info(
            "Full-resolution ConvNeXt batch size: %d -> %d for maximum scaled "
            "frame size %d x %d (%.2fx the area of %d x %d)",
            args.batch_size,
            effective_batch_size,
            largest_shape[0],
            largest_shape[1],
            relative_area,
            REFERENCE_BATCH_IMAGE_SIZE,
            REFERENCE_BATCH_IMAGE_SIZE,
        )

    diameters: list[float] = []
    dino_diameters: list[float] = []
    for mask, frame_diameters in zip(masks, movie.object_diameters):
        diameters.extend(frame_diameters)
        if args.coordinate_baseline:
            dino_scale = 1.0
        elif args.explicit_tiling or args.full_resolution:
            dino_scale = args.scale
        else:
            scaled_height = round(mask.shape[0] * args.scale)
            scaled_width = round(mask.shape[1] * args.scale)
            downsampling = np.sqrt(scaled_height * scaled_width) / args.image_size
            dino_scale = args.scale / downsampling
        dino_diameters.extend(frame_diameters * dino_scale)

    median_diameter = float(np.median(diameters)) if diameters else float("nan")
    median_dino_diameter = (
        float(np.median(dino_diameters)) if dino_diameters else float("nan")
    )
    logger.info(
        "Median object equivalent diameter before scaling: %s",
        f"{median_diameter:.2f} px ({len(diameters)} instances)"
        if diameters
        else "n/a",
    )
    logger.info(
        "Median object equivalent diameter as seen by DINO: %s",
        f"{median_dino_diameter:.2f} px" if dino_diameters else "n/a",
    )

    frames = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    extraction_start = perf_counter()

    def record_frame(
        prepared: PreparedFrame,
        result: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        labels, centroids, features = result
        if not np.array_equal(labels, prepared.labels):
            raise ValueError(
                f"Extracted labels differ from cached mask labels in frame "
                f"{prepared.frame}"
            )
        frames.append(
            FrameFeatures(
                prepared.frame,
                labels,
                prepared.lineages,
                centroids,
                features,
            )
        )
        logger.info(
            "Frame %d (%d/%d): %d objects, feature shape %s",
            prepared.frame,
            prepared.local_index + 1,
            len(frame_numbers),
            len(labels),
            features.shape,
        )

    if args.coordinate_baseline:
        for local_index, frame in enumerate(frame_numbers):
            prepared = prepared_frames[local_index]
            record_frame(
                prepared,
                (prepared.labels, prepared.centroids, prepared.centroids),
            )
    elif args.explicit_tiling:
        if feature_extractor is None:
            raise ValueError("Explicit tiling requires a feature extractor")
        for local_index, frame in enumerate(frame_numbers):
            prepared = prepared_frames[local_index]
            logger.info(
                "Frame %d image size before tiling: %d x %d",
                frame,
                prepared.image.shape[0],
                prepared.image.shape[1],
            )
            result = feature_extractor(
                prepared.image,
                prepared.mask,
                processor,
                model,
                device,
                args.image_size,
                batch_size=args.batch_size,
                pooling_plans=prepared.pooling_plans,
                labels=prepared.labels,
                centroids=prepared.centroids,
            )
            record_frame(prepared, result)
    else:
        if feature_extractor is None:
            raise ValueError("DINO extraction requires a feature extractor")
        for batch_start in range(0, len(frame_numbers), effective_batch_size):
            batch_stop = min(batch_start + effective_batch_size, len(frame_numbers))
            prepared_batch = [
                prepared_frames[local_index]
                for local_index in range(batch_start, batch_stop)
            ]
            results = extract_mask_features_batch(
                [prepared.image for prepared in prepared_batch],
                [prepared.mask for prepared in prepared_batch],
                processor,
                model,
                device,
                args.image_size,
                resize=not args.full_resolution,
                pooling_plan_caches=[
                    prepared.pooling_plans for prepared in prepared_batch
                ],
                mask_labels=[prepared.labels for prepared in prepared_batch],
                mask_centroids=[prepared.centroids for prepared in prepared_batch],
            )
            for prepared, result in zip(prepared_batch, results):
                record_frame(prepared, result)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    time_per_frame_ms = (
        1_000.0 * (perf_counter() - extraction_start) / len(frame_numbers)
    )
    logger.info("Mean feature extraction time: %.2f ms/frame", time_per_frame_ms)

    return MovieExtraction(
        frames=tuple(frames),
        effective_batch_size=effective_batch_size,
        median_dino_diameter=median_dino_diameter,
        time_per_frame_ms=time_per_frame_ms,
    )


def _evaluate_movie(
    movie: LoadedMovie,
    args: argparse.Namespace,
    extraction: MovieExtraction,
    random_projection: np.ndarray | None,
) -> dict[str, Any]:
    """Evaluate one optional projection of cached object features."""
    if random_projection is None:
        frames = extraction.frames
    else:
        frames = tuple(
            FrameFeatures(
                frame=frame.frame,
                labels=frame.labels,
                lineages=frame.lineages,
                centroids=frame.centroids,
                features=_project_features(frame.features, random_projection),
            )
            for frame in extraction.frames
        )

    metrics = evaluate_sequence(
        frames, movie.lineage_parents, args.coordinate_baseline
    )
    correct_similarity = _ratio(metrics["correct_similarity"], metrics["correct_n"])
    incorrect_similarity = _ratio(
        metrics["incorrect_similarity"], metrics["comparison_n"]
    )
    similarity_margin = _ratio(
        metrics["compared_correct_similarity"] - metrics["incorrect_similarity"],
        metrics["comparison_n"],
    )
    correct_higher = _ratio(metrics["correct_wins"], metrics["comparison_n"])
    top1_accuracy = _ratio(metrics["top1_correct"], metrics["comparison_n"])
    daughter_recall_at2 = _ratio(
        metrics["daughter_hits_at2"], metrics["daughter_links"]
    )
    complete_division_recall_at2 = _ratio(
        metrics["complete_division_hits_at2"], metrics["division_n"]
    )
    parent_daughter_similarity = _ratio(
        metrics["parent_daughter_similarity"], metrics["daughter_links"]
    )
    division_margin = _ratio(metrics["division_margin"], metrics["division_margin_n"])
    similarity_name = "inverse-distance" if args.coordinate_baseline else "cosine"

    logger.info("Found %.0f persistent object transitions", metrics["correct_n"])
    logger.info(
        "Mean %s similarity, correct next-frame track: %s",
        similarity_name,
        f"{correct_similarity:.4f}" if np.isfinite(correct_similarity) else "n/a",
    )
    if metrics["comparison_n"]:
        logger.info(
            "Mean %s similarity, nearest incorrect next-frame cell: %.4f",
            similarity_name,
            incorrect_similarity,
        )
        logger.info(
            "Mean correct-minus-incorrect similarity margin: %.4f", similarity_margin
        )
        logger.info("Correct similarity is higher: %.1f%%", 100.0 * correct_higher)
        logger.info("Global top-1 accuracy: %.1f%%", 100.0 * top1_accuracy)
    else:
        logger.info("Nearest incorrect next-frame similarity: n/a")
        logger.info("No selected transition has an incorrect next-frame cell")

    logger.info("Found %.0f binary division events", metrics["division_n"])
    if metrics["division_n"]:
        logger.info("Daughter Recall@2: %.1f%%", 100.0 * daughter_recall_at2)
        logger.info(
            "Complete division Recall@2: %.1f%%",
            100.0 * complete_division_recall_at2,
        )
        logger.info(
            "Mean parent-daughter %s similarity: %.4f",
            similarity_name,
            parent_daughter_similarity,
        )
        logger.info(
            "Mean division margin, weaker daughter minus hardest non-daughter: %s",
            f"{division_margin:.4f}" if np.isfinite(division_margin) else "n/a",
        )

    return {
        "dataset": _dataset_name(movie.root),
        "batch_size": extraction.effective_batch_size,
        "diameter_dino_px": extraction.median_dino_diameter,
        "time_per_frame_ms": extraction.time_per_frame_ms,
        "correct_sim": correct_similarity,
        "incorrect_sim": incorrect_similarity,
        "margin_sim": similarity_margin,
        "top1_acc": top1_accuracy,
        "daughter_recall": daughter_recall_at2,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        nargs="+",
        help=f"One or more CTC sequence directories; default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        help="Training YAML config whose input_train CTC stacks are evaluated",
    )
    parser.add_argument(
        "-m",
        "--model",
        nargs="+",
        default=[DEFAULT_MODEL],
        help=(
            "One or more model aliases or Hugging Face names; aliases: "
            + ", ".join(sorted(MODEL_ALIASES))
        ),
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=448,
        help=(
            "Square target size when resizing or ViT tile size when preserving "
            "resolution; must be divisible by the spatial stride"
        ),
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=8,
        help="Frames per batch or tiles per explicitly tiled ViT batch",
    )
    parser.add_argument(
        "-r",
        "--randpro",
        type=int,
        nargs="+",
        default=[-1],
        metavar="K",
        help="Random projection dimensions to evaluate; -1 keeps all features",
    )
    parser.add_argument(
        "--resize",
        nargs="*",
        type=_parse_on_off,
        default=[True],
        metavar="{on,off}",
        help=(
            "Resize modes to evaluate; off preserves native resolution using ViT "
            "tiles or full-frame ConvNeXt"
        ),
    )
    parser.add_argument(
        "--scale",
        type=float,
        nargs="+",
        default=[1.0],
        help="Scale factors to evaluate before DINO extraction",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        help="First frame number; by default, select the last --max-frames frames",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=20,
        help="Maximum number of frames; 0 uses all",
    )
    parser.add_argument(
        "-j",
        "--n-workers",
        type=int,
        default=8,
        help="Workers used by TrackingSequence.from_ctc for region features",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        type=Path,
        help="Write one metrics row per movie to this CSV file",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.start_frame is not None and args.start_frame < 0:
        raise ValueError(f"start_frame must be nonnegative, got {args.start_frame}")
    if args.max_frames < 0:
        raise ValueError(f"max_frames must be nonnegative, got {args.max_frames}")
    if args.n_workers < 0:
        raise ValueError(f"n_workers must be nonnegative, got {args.n_workers}")
    if args.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {args.batch_size}")
    for randpro in args.randpro:
        if randpro == 0 or randpro < -1:
            raise ValueError(
                f"randpro values must be -1 or positive integers, got {randpro}"
            )
    if not args.resize:
        args.resize = [True]
    for scale in args.scale:
        if not np.isfinite(scale) or scale < 1.0:
            raise ValueError(f"scale must be finite and at least 1, got {scale}")

    inputs = []
    if args.config is not None:
        inputs.extend(_inputs_from_config(args.config))
    if args.input is not None:
        inputs.extend(args.input)
    if not inputs:
        inputs = [DEFAULT_INPUT]
    variants_per_movie = sum(
        1
        if _resolve_model_name(model) == "coordinates"
        else len(args.resize) * len(args.randpro) * len(args.scale)
        for model in args.model
    )
    n_evaluations = variants_per_movie * len(inputs)
    logger.info(
        "Evaluating %d model(s), %d random projection(s), %d scale(s), and "
        "%d resize mode(s) on %d movie(s): %d evaluations",
        len(args.model),
        len(args.randpro),
        len(args.scale),
        len(args.resize),
        len(inputs),
        n_evaluations,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = []
    if args.outfile is not None:
        args.outfile.parent.mkdir(parents=True, exist_ok=True)
    model_cache: dict[str, tuple[Any, Any, int]] = {}
    for movie_index, root in enumerate(inputs, start=1):
        logger.info("Movie %d/%d: %s", movie_index, len(inputs), root)
        movie = _load_movie(root, args)
        for model_index, model_spec in enumerate(args.model, start=1):
            model_name = _resolve_model_name(model_spec)
            coordinate_baseline = model_name == "coordinates"
            is_convnext = model_name.startswith("timm:")
            if model_name not in model_cache:
                logger.info(
                    "Model %d/%d: loading %s",
                    model_index,
                    len(args.model),
                    model_name,
                )
                if coordinate_baseline:
                    processor = None
                    model = None
                    spatial_stride = 1
                elif is_convnext:
                    processor, model, spatial_stride = _load_timm_feature_model(
                        model_name.removeprefix("timm:")
                    )
                else:
                    processor = AutoImageProcessor.from_pretrained(model_name)
                    model = AutoModel.from_pretrained(model_name).eval()
                    spatial_stride = int(model.config.patch_size)
                if not coordinate_baseline and (
                    args.image_size <= 0 or args.image_size % spatial_stride != 0
                ):
                    raise ValueError(
                        f"image_size must be positive and divisible by spatial stride "
                        f"{spatial_stride} for {model_name}, got {args.image_size}"
                    )
                model_cache[model_name] = (
                    processor,
                    model,
                    args.image_size // spatial_stride,
                )
            else:
                logger.info(
                    "Model %d/%d: using cached %s",
                    model_index,
                    len(args.model),
                    model_name,
                )
            processor, model, grid_size = model_cache[model_name]
            if model is not None:
                model = model.to(device).eval()
            random_projections: dict[int, np.ndarray] = {}
            scales = [1.0] if coordinate_baseline else args.scale
            resize_modes = [False] if coordinate_baseline else args.resize
            projection_dims = [-1] if coordinate_baseline else args.randpro

            for scale in scales:
                for resize in resize_modes:
                    explicit_tiling = (
                        not coordinate_baseline and not resize and not is_convnext
                    )
                    full_resolution = (
                        not coordinate_baseline and not resize and is_convnext
                    )
                    if scale != 1.0:
                        logger.info(
                            "Upsampling images and masks by %.3g before DINO extraction",
                            scale,
                        )
                    if coordinate_baseline:
                        logger.info(
                            "Coordinate nearest-neighbor baseline using centroid "
                            "distance"
                        )
                        feature_extractor = None
                    elif explicit_tiling:
                        logger.info(
                            "Tiled DINO: %d x %d tiles, at least %d px overlap, "
                            "%d x %d token grid per tile, batch size %d",
                            args.image_size,
                            args.image_size,
                            args.image_size // 4,
                            grid_size,
                            grid_size,
                            args.batch_size,
                        )
                        feature_extractor = extract_mask_features_tiled
                    elif full_resolution:
                        logger.info(
                            "Full-resolution ConvNeXt inference without explicit "
                            "tiling, stride-%d feature map, requested frame batch "
                            "size %d",
                            int(model.config.patch_size),
                            args.batch_size,
                        )
                        feature_extractor = extract_mask_features
                    else:
                        logger.info(
                            "Resized DINO input %d x %d gives a %d x %d patch-token "
                            "grid, frame batch size %d",
                            args.image_size,
                            args.image_size,
                            grid_size,
                            grid_size,
                            args.batch_size,
                        )
                        feature_extractor = extract_mask_features

                    extraction: MovieExtraction | None = None
                    for randpro in projection_dims:
                        run_args = argparse.Namespace(**vars(args))
                        run_args.scale = scale
                        run_args.resize = resize
                        run_args.explicit_tiling = explicit_tiling
                        run_args.full_resolution = full_resolution
                        run_args.coordinate_baseline = coordinate_baseline
                        run_args.randpro = randpro
                        logger.info(
                            "Variant: scale=%.3g, resize=%s, randpro=%d",
                            scale,
                            resize,
                            randpro,
                        )
                        if randpro > 0 and randpro not in random_projections:
                            if model is None:
                                raise ValueError(
                                    "Random projection requires a feature model"
                                )
                            random_projections[randpro] = _make_random_projection(
                                int(model.config.hidden_size), randpro
                            )
                        random_projection = random_projections.get(randpro)
                        if random_projection is not None:
                            logger.info(
                                "Random feature projection: %d -> %d dimensions",
                                random_projection.shape[0],
                                random_projection.shape[1],
                            )

                        if extraction is None:
                            extraction = _extract_movie(
                                movie,
                                run_args,
                                processor,
                                model,
                                device,
                                feature_extractor,
                            )
                        else:
                            logger.info(
                                "Reusing cached unprojected object features for "
                                "randpro=%d",
                                randpro,
                            )
                        row = _evaluate_movie(
                            movie,
                            run_args,
                            extraction,
                            random_projection,
                        )
                        row.update(
                            {
                                "model": model_spec,
                                "randpro": randpro,
                                "scale": scale,
                                "resize": resize,
                            }
                        )
                        rows.append(row)
                        dataframe = _results_dataframe(rows)
                        if args.outfile is not None:
                            dataframe.to_csv(args.outfile, index=False)
                            logger.info(
                                "Updated %s with %d completed evaluations",
                                args.outfile,
                                len(dataframe),
                            )
                        logger.info(
                            "Latest metrics:\n%s",
                            dataframe.tail(5).to_markdown(index=False, floatfmt=".4f"),
                        )

            if device.type == "cuda" and model is not None:
                model.to("cpu")
                torch.cuda.empty_cache()

    dataframe = _results_dataframe(rows)
    logger.info(
        "Metrics:\n%s",
        dataframe.to_markdown(index=False, floatfmt=".4f"),
    )
