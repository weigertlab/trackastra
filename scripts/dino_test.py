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
from collections.abc import Callable, Sequence
from dataclasses import dataclass
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
class PreparedFrame:
    """Scaled frame data and supervision needed for DINO extraction."""

    local_index: int
    frame: int
    image: np.ndarray
    mask: np.ndarray
    detection_indices: np.ndarray


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


def _accumulate_mask_features(
    pooled: torch.Tensor,
    feature_grid: torch.Tensor,
    mask: np.ndarray,
    labels: np.ndarray,
    upsample_size: tuple[int, int],
    pixel_weights: np.ndarray | None = None,
    channel_chunk_size: int = 64,
) -> torch.Tensor:
    """Add one dense feature grid to object sums and return object weights."""
    device = pooled.device
    flat_mask = torch.from_numpy(mask.astype(np.int64, copy=False)).to(device).view(-1)
    foreground = flat_mask != 0
    object_weights = torch.zeros(len(labels), dtype=torch.float32, device=device)
    if not foreground.any():
        return object_weights

    label_tensor = torch.from_numpy(labels.astype(np.int64, copy=False)).to(device)
    object_indices = torch.searchsorted(label_tensor, flat_mask[foreground])
    if pixel_weights is None:
        foreground_weights = torch.ones(
            len(object_indices), dtype=torch.float32, device=device
        )
    else:
        if pixel_weights.shape != mask.shape:
            raise ValueError(
                f"Pixel weights shape {pixel_weights.shape} differs from mask "
                f"shape {mask.shape}"
            )
        flat_weights = torch.from_numpy(
            pixel_weights.astype(np.float32, copy=False)
        ).to(device).view(-1)
        foreground_weights = flat_weights[foreground]
    object_weights.index_add_(0, object_indices, foreground_weights)

    for start in range(0, pooled.shape[1], channel_chunk_size):
        stop = min(start + channel_chunk_size, pooled.shape[1])
        dense_chunk = F.interpolate(
            feature_grid[:, start:stop],
            size=upsample_size,
            mode="bilinear",
            align_corners=False,
        )[0, :, : mask.shape[0], : mask.shape[1]]
        foreground_features = dense_chunk.permute(1, 2, 0).reshape(
            -1, stop - start
        )[foreground]
        pooled[:, start:stop].index_add_(
            0,
            object_indices,
            foreground_features.float() * foreground_weights[:, None],
        )
    return object_weights


def _pool_mask_features(
    mask: np.ndarray,
    hidden: torch.Tensor,
    processed_size: tuple[int, int],
    model,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pool one frame's spatial DINO tokens over its instance masks.

    Spatial tokens are bilinearly upsampled into the unchanged mask coordinate
    system and averaged per instance with a device-side indexed reduction.
    """
    labels = np.unique(mask)
    labels = labels[labels != 0]
    if len(labels) == 0:
        return labels, np.empty((0, 2), np.float32), np.empty((0, 0), np.float32)

    patch_size = int(model.config.patch_size)
    grid_height = processed_size[0] // patch_size
    grid_width = processed_size[1] // patch_size
    patch_tokens = _spatial_tokens(hidden, grid_height, grid_width)

    # (P, D) -> (1, D, grid_height, grid_width). Upsample channel chunks below
    # to bound the temporary dense feature-map memory.
    feature_grid = patch_tokens.T.reshape(1, -1, grid_height, grid_width)
    pooled = torch.zeros(
        (len(labels), patch_tokens.shape[1]), dtype=torch.float32, device=device
    )
    object_weights = _accumulate_mask_features(
        pooled, feature_grid, mask, labels, mask.shape
    )
    if torch.any(object_weights == 0):
        raise ValueError("At least one nonzero mask label has no pixels")
    pooled /= object_weights[:, None]

    features = F.normalize(pooled, dim=1).cpu().float().numpy()
    return labels, _centroids(mask, labels), features


def extract_mask_features(
    image: np.ndarray,
    mask: np.ndarray,
    processor,
    model,
    device: torch.device,
    image_size: int,
    resize: bool = True,
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
    )[0]


def extract_mask_features_batch(
    images: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    processor,
    model,
    device: torch.device,
    image_size: int,
    resize: bool = True,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Extract mask-pooled features from a batch of optionally resized frames."""
    if len(images) != len(masks):
        raise ValueError(f"Got {len(images)} images and {len(masks)} masks")
    if not images:
        return []
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
        )
        for mask, hidden in zip(masks, hidden_states)
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

    labels = np.unique(mask)
    labels = labels[labels != 0]
    if len(labels) == 0:
        return labels, np.empty((0, 2), np.float32), np.empty((0, 0), np.float32)

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
            feature_grid = patch_tokens.T.reshape(1, -1, grid_size, grid_size)

            local_mask = mask[y : y + valid_height, x : x + valid_width]
            local_labels = np.unique(local_mask)
            local_labels = local_labels[local_labels != 0]
            if len(local_labels) == 0:
                continue
            normalized_weight = (
                blend[:valid_height, :valid_width]
                / coverage[y : y + valid_height, x : x + valid_width]
            )
            object_weights += _accumulate_mask_features(
                pooled,
                feature_grid,
                local_mask,
                labels,
                (tile_size, tile_size),
                normalized_weight,
            )

    if torch.any(object_weights == 0):
        raise ValueError("At least one mask label received no tiled feature samples")
    pooled /= object_weights[:, None]
    features = F.normalize(pooled, dim=1).cpu().numpy()
    return labels, _centroids(mask, labels), features


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


def _evaluate_movie(
    root: Path,
    args: argparse.Namespace,
    processor: Any,
    model: Any,
    device: torch.device,
    feature_extractor: Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]] | None,
    random_projection: np.ndarray | None,
) -> dict[str, Any]:
    """Extract features and return one flat metrics row for a CTC movie."""
    image_folder = _ctc_image_folder(root)
    image_paths = sorted((*image_folder.glob("*.tif"), *image_folder.glob("*.tiff")))
    if not image_paths:
        raise ValueError(f"No TIFF images found in {image_folder}")
    n_available_frames = len(image_paths)
    start_frame, stop_frame = _frame_bounds(
        n_available_frames, args.start_frame, args.max_frames
    )
    frame_numbers = list(range(start_frame, stop_frame))
    if len(frame_numbers) < 2:
        raise ValueError(
            f"Need at least two selected CTC frames, found {len(frame_numbers)}"
        )
    slice_pct = (
        _slice_fraction(start_frame, n_available_frames),
        _slice_fraction(stop_frame, n_available_frames),
    )

    logger.info(
        "Loading CTC frames %d:%d through TrackingSequence.from_ctc: %s",
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
    for mask in masks:
        labels, counts = np.unique(mask, return_counts=True)
        frame_diameters = 2.0 * np.sqrt(counts[labels != 0] / np.pi)
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

    def prepare_frame(local_index: int, frame: int) -> PreparedFrame:
        detection_indices = np.flatnonzero(detections.timepoints == local_index)
        image, mask = _scale_image_mask(
            images[local_index], masks[local_index], args.scale
        )
        return PreparedFrame(
            local_index,
            frame,
            image,
            mask,
            detection_indices,
        )

    def record_frame(
        prepared: PreparedFrame,
        result: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        labels, centroids, features = result
        if random_projection is not None:
            features = _project_features(features, random_projection)
        lineage_by_label = {
            int(detections.labels[i]): int(supervision.lineage_index[i])
            for i in prepared.detection_indices
        }
        lineages = np.asarray(
            [lineage_by_label.get(int(label), -1) for label in labels], dtype=np.int64
        )
        frames.append(
            FrameFeatures(prepared.frame, labels, lineages, centroids, features)
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
            prepared = prepare_frame(local_index, frame)
            labels = np.unique(prepared.mask)
            labels = labels[labels != 0]
            centroids = _centroids(prepared.mask, labels)
            record_frame(prepared, (labels, centroids, centroids))
    elif args.explicit_tiling:
        if feature_extractor is None:
            raise ValueError("Explicit tiling requires a feature extractor")
        for local_index, frame in enumerate(frame_numbers):
            prepared = prepare_frame(local_index, frame)
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
            )
            record_frame(prepared, result)
    else:
        if feature_extractor is None:
            raise ValueError("DINO extraction requires a feature extractor")
        for batch_start in range(0, len(frame_numbers), effective_batch_size):
            batch_stop = min(batch_start + effective_batch_size, len(frame_numbers))
            prepared_batch = [
                prepare_frame(local_index, frame_numbers[local_index])
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
            )
            for prepared, result in zip(prepared_batch, results):
                record_frame(prepared, result)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    time_per_frame_ms = (
        1_000.0 * (perf_counter() - extraction_start) / len(frame_numbers)
    )
    logger.info("Mean feature extraction time: %.2f ms/frame", time_per_frame_ms)

    metrics = evaluate_sequence(
        frames, sequence.gt.lineage_parents, args.coordinate_baseline
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
        "dataset": _dataset_name(root),
        "batch_size": effective_batch_size,
        "diameter_dino_px": median_dino_diameter,
        "time_per_frame_ms": time_per_frame_ms,
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

                        row = _evaluate_movie(
                            root,
                            run_args,
                            processor,
                            model,
                            device,
                            feature_extractor,
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
