import logging
import os
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import dask.array as da
import networkx as nx
import numpy as np
import tifffile
import torch
import yaml
from tqdm import tqdm

from ..data import build_windows, get_features, load_tiff_timeseries
from ..data.wrfeat import (
    FEATURE_CUSTOM,
    FEATURE_RECIPES,
    WRFeatures,
    normalize_to_diameter,
)
from ..tracking import apply_solution_graph_to_masks, build_graph, track_greedy
from ..utils import normalize
from .model import TrackingTransformer
from .predict import predict_windows
from .pretrained import download_pretrained

try:
    from trackastra_pretrained_feats import FeatureExtractor

    PRETRAINED_FEATS_INSTALLED = True
except ImportError:
    PRETRAINED_FEATS_INSTALLED = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Keys required to reproduce feature extraction at inference time. Persisted to
# inference_config.yaml (the inference contract), separate from the free-form
# train_config.yaml provenance dump.
INFERENCE_CONFIG_KEYS = (
    "features",
    "normalize_diameter",
    "pretrained_feats_model",
    "pretrained_feats_mode",
    "pretrained_feats_additional_props",
)


def _resolve_inference_max_distance(trained: float, requested: float | None) -> float:
    """Resolve the candidate-graph distance against the model's trained radius.

    Defaults to the trained ``max_distance`` (the radius the model's attention was
    masked to). A lower request is honoured as-is (tighter linking); a higher request
    is honoured but warned, since edges beyond the trained radius were masked out during
    training and cannot be scored.
    """
    if requested is None:
        return trained
    if requested > trained:
        logger.warning(
            "Requested max_distance=%g exceeds the model's trained max_distance=%g; "
            "candidate edges beyond the trained radius were masked during training and "
            "cannot be scored.",
            requested,
            trained,
        )
    return requested


def _default_batch_size(device: str | torch.device):
    if isinstance(device, torch.device):
        device = device.type
    d = {"cpu": 1, "cuda": 4, "mps": 4}
    batch_size = d[device]
    logger.info(f"Default batch size = {batch_size} for model on {device}.")
    return batch_size


def _as_point_coords(coords, *, ndim: int, frame: int) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float32)
    if coords.size == 0 and coords.ndim == 1:
        coords = coords.reshape(0, ndim)
    if coords.ndim != 2 or coords.shape[1] != ndim:
        raise ValueError(
            f"coords[{frame}] must have shape (N, {ndim}), got {coords.shape}"
        )
    if not np.all(np.isfinite(coords)):
        raise ValueError(f"coords[{frame}] contains non-finite values")
    return np.ascontiguousarray(coords)


def _as_point_labels(labels, *, n: int, frame: int) -> np.ndarray:
    if labels is None:
        return np.arange(1, n + 1, dtype=np.int32)
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) != n:
        raise ValueError(f"labels[{frame}] must have shape ({n},), got {labels.shape}")
    if len(np.unique(labels)) != len(labels):
        raise ValueError(f"labels[{frame}] must be unique within the frame")
    return labels.astype(np.int32, copy=False)


def _as_point_features(features, *, n: int, frame: int) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or len(features) != n:
        raise ValueError(
            f"features[{frame}] must have shape (N, F) with N={n}, got "
            f"{features.shape}"
        )
    if not np.all(np.isfinite(features)):
        raise ValueError(f"features[{frame}] contains non-finite values")
    return np.ascontiguousarray(features)


class Trackastra:
    """A transformer-based tracking model for time-lapse data.

    Trackastra links segmented objects across time frames by predicting
    associations with a transformer model trained on diverse time-lapse videos.

    The model takes as input:
    - A sequence of images of shape (T,(Z),Y,X)
    - Corresponding instance segmentation masks of shape (T,(Z),Y,X)

    It supports multiple tracking modes:
    - greedy_nodiv: Fast greedy linking without division
    - greedy: Fast greedy linking with division
    - ilp: Integer Linear Programming based linking (more accurate but slower)

    Examples:
        >>> # Load example data
        >>> from trackastra.data import example_data_bacteria
        >>> imgs, masks = example_data_bacteria()
        >>>
        >>> # Load pretrained model and track
        >>> model = Trackastra.from_pretrained("general_2d", device="cuda")
        >>> track_graph = model.track(imgs, masks, mode="greedy")
    """

    def __init__(
        self,
        transformer: TrackingTransformer,
        inference_config: dict,
        device: Literal["cuda", "mps", "cpu", "automatic", None] = None,
        batch_size: int | None = None,
    ):
        """Initialize Trackastra model.

        Args:
            transformer: The underlying transformer model.
            inference_config: Curated keys required to reproduce feature extraction at
                inference time (see ``INFERENCE_CONFIG_KEYS``). This is the complete
                inference contract; training provenance is not needed for prediction.
            device: Device to run model on ("cuda", "mps", "cpu", "automatic" or None).
            batch_size: Batch size for prediction. If None, defaults to 1 on CPU and 16 on GPU.
        """
        if device == "cuda":
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                logger.info("Cuda not available, falling back to cpu.")
                self.device = "cpu"
        elif device == "mps":
            if (
                torch.backends.mps.is_available()
                and os.getenv("PYTORCH_ENABLE_MPS_FALLBACK") is not None
                and os.getenv("PYTORCH_ENABLE_MPS_FALLBACK") != "0"
            ):
                self.device = "mps"
            else:
                logger.info("Mps not available, falling back to cpu.")
                self.device = "cpu"
        elif device == "cpu":
            self.device = "cpu"
        elif device == "automatic" or device is None:
            should_use_mps = (
                torch.backends.mps.is_available()
                and os.getenv("PYTORCH_ENABLE_MPS_FALLBACK") is not None
                and os.getenv("PYTORCH_ENABLE_MPS_FALLBACK") != "0"
            )
            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else (
                    "mps"
                    if should_use_mps and os.getenv("PYTORCH_ENABLE_MPS_FALLBACK")
                    else "cpu"
                )
            )
        else:
            raise ValueError(f"Device {device} not recognized.")

        logger.info(f"Using device {self.device}")

        self.batch_size = (
            _default_batch_size(self.device) if batch_size is None else batch_size
        )

        self.transformer = transformer.to(self.device)
        self.inference_config = inference_config
        self.imgs_path = None
        self.masks_path = None
        self.feature_extractor = None

    @classmethod
    def from_folder(
        cls,
        dir: Path | str,
        device: str | None = None,
        checkpoint_path: str | None = None,
        **kwargs,
    ):
        """Load a Trackastra model from a local folder.

        Args:
            dir: Path to model folder containing:
                - model weights
                - inference_config.yaml with the feature-extraction contract
                - train_config.yaml with the full training-argument provenance dump
            device: Device to run model on.
            checkpoint_path: Path to model checkpoint file (defaults to "model.pt" in dir).

        Returns:
            Trackastra model instance.
        """
        # Always load to cpu first
        if checkpoint_path is None:
            checkpoint_path = "model.pt"
        transformer = TrackingTransformer.from_folder(
            Path(dir).expanduser(), map_location="cpu", checkpoint_path=checkpoint_path
        )
        inference_config_path = dir / "inference_config.yaml"
        if inference_config_path.exists():
            inference_config = yaml.load(
                open(inference_config_path), Loader=yaml.FullLoader
            )
        else:
            # LEGACY: pre-split model folder without inference_config.yaml. Derive the
            # contract from the train_config dump. Remove once all published/saved
            # models ship an inference_config.yaml.
            train_args = yaml.load(
                open(dir / "train_config.yaml"), Loader=yaml.FullLoader
            )
            inference_config = {k: train_args.get(k) for k in INFERENCE_CONFIG_KEYS}
            logger.info(
                "No inference_config.yaml found in %s; derived inference config from "
                "train_config.yaml (legacy model folder): %s",
                dir,
                inference_config,
            )
        return cls(
            transformer=transformer,
            inference_config=inference_config,
            device=device,
            **kwargs,
        )

    @classmethod
    def from_pretrained(
        cls,
        name: str,
        device: str | None = None,
        download_dir: Path | None = None,
        **kwargs,
    ):
        """Load a pretrained Trackastra model.

        Available pretrained models are described in detail in pretrained.json.

        Args:
            name: Name of pretrained model (e.g. "general_2d").
            device: Device to run model on ("cuda", "mps", "cpu", "automatic" or None).
            download_dir: Directory to download model. Default handled by platformdirs.

        Returns:
            Trackastra model instance.
        """
        folder = download_pretrained(name, download_dir)
        # download zip from github to location/name, then unzip
        return cls.from_folder(folder, device=device, **kwargs)

    def _maybe_setup_pretrained_extractor(self, imgs, batch_size: int | None = None):
        """Set up the pretrained feature extractor when the contract requires it.

        No-op for the standard wrfeat path. Called by every entry that extracts
        features (``track`` and the cached validation path) so they behave the same.
        """
        feat_type = self.inference_config["features"]
        if feat_type not in ("pretrained_feats", "pretrained_feats_aug"):
            return
        if not PRETRAINED_FEATS_INSTALLED:
            raise ImportError(
                "The trackastra_pretrained_feats package is required for pretrained "
                "feature extraction. Please install it with: "
                "pip install trackastra[etultra]"
            )
        save_path = "./embeddings" if self.imgs_path is None else self.imgs_path / "embeddings"
        self.feature_extractor = FeatureExtractor.from_model_name(
            self.inference_config["pretrained_feats_model"],
            imgs.shape[-2:],
            save_path=save_path,
            mode=self.inference_config["pretrained_feats_mode"],
            device=self.device,
            additional_features=self.inference_config.get(
                "pretrained_feats_additional_props", None
            ),
            batch_size=batch_size or self.batch_size,
        )
        self.feature_extractor.force_recompute = True

    def _extract_features_windows(
        self,
        imgs: np.ndarray | da.Array,
        masks: np.ndarray | da.Array,
        n_workers: int = 0,
        normalize_imgs: bool = True,
        progbar_class=tqdm,
        normalize_diameter: float | None = None,
        batch_size: int | None = None,
    ):
        """Weight-independent prefix of :meth:`_predict`.

        Image normalisation, feature extraction, diameter normalisation, and window
        construction depend only on the inputs, not the transformer weights. Callers
        that predict repeatedly on the same movie (e.g. per-epoch validation
        tracking) can build this once and reuse it across weight updates.
        """
        self._maybe_setup_pretrained_extractor(imgs, batch_size=batch_size)
        if normalize_imgs:
            if isinstance(imgs, da.Array):
                imgs = imgs.map_blocks(normalize)
            else:
                imgs = normalize(imgs)

        features = get_features(
            detections=masks,
            imgs=imgs,
            features=self.inference_config["features"],
            feature_extractor=self.feature_extractor,
            ndim=self.transformer.config["coord_dim"],
            n_workers=n_workers,
            progbar_class=progbar_class,
        )
        if normalize_diameter is None:
            normalize_diameter = self.inference_config.get("normalize_diameter")
        features = normalize_to_diameter(features, normalize_diameter)
        logger.info("Building windows")
        windows = build_windows(
            features,
            window_size=self.transformer.config["window"],
            progbar_class=progbar_class,
            as_torch=True,
            feature_mode=(
                self.inference_config["features"]
                if self.inference_config["features"]
                in FEATURE_RECIPES
                else "wrfeat"
            ),
        )
        return features, windows

    def _predict_from_windows(
        self,
        features,
        windows,
        spatial_dim: int,
        edge_threshold: float = 0.05,
        batch_size: int | None = None,
        progbar_class=tqdm,
    ):
        """Weight-dependent half of :meth:`_predict`.

        Runs the transformer over pre-built ``windows`` from
        :meth:`_extract_features_windows`. This is the only part that must re-run
        when the weights change.
        """
        self.transformer.eval()
        batch_size = batch_size or self.batch_size
        logger.info(f"Predicting windows with batch size {batch_size}")
        return predict_windows(
            windows=windows,
            features=features,
            model=self.transformer,
            edge_threshold=edge_threshold,
            spatial_dim=spatial_dim,
            progbar_class=progbar_class,
            batch_size=batch_size,
        )

    def _predict(
        self,
        imgs: np.ndarray | da.Array,
        masks: np.ndarray | da.Array,
        edge_threshold: float = 0.05,
        n_workers: int = 0,
        normalize_imgs: bool = True,
        progbar_class=tqdm,
        batch_size: int | None = None,
        normalize_diameter: float | None = None,
    ):
        logger.info("Predicting weights for candidate graph")
        features, windows = self._extract_features_windows(
            imgs,
            masks,
            n_workers=n_workers,
            normalize_imgs=normalize_imgs,
            progbar_class=progbar_class,
            normalize_diameter=normalize_diameter,
            batch_size=batch_size,
        )
        return self._predict_from_windows(
            features,
            windows,
            spatial_dim=masks.ndim - 1,
            edge_threshold=edge_threshold,
            batch_size=batch_size,
            progbar_class=progbar_class,
        )

    def _extract_point_windows(
        self,
        coords: Sequence[np.ndarray],
        features: Sequence[np.ndarray] | None = None,
        labels: Sequence[np.ndarray] | None = None,
        progbar_class=tqdm,
    ) -> tuple[list[WRFeatures], list[dict], int]:
        """Build model windows from already scaled point coordinates."""
        coords = list(coords)
        if len(coords) < 2:
            raise ValueError(f"Need at least 2 frames for tracking, got {len(coords)}.")
        ndim = int(self.transformer.config["coord_dim"])

        if features is not None:
            features = list(features)
            if len(features) != len(coords):
                raise ValueError(
                    "features must have the same number of frames as coords"
                )
        if labels is not None:
            labels = list(labels)
            if len(labels) != len(coords):
                raise ValueError("labels must have the same number of frames as coords")

        point_features = []
        feature_width = 0
        for t, frame_coords in enumerate(coords):
            frame_coords = _as_point_coords(frame_coords, ndim=ndim, frame=t)
            frame_labels = _as_point_labels(
                None if labels is None else labels[t],
                n=len(frame_coords),
                frame=t,
            )
            timepoints = np.full(len(frame_coords), t, dtype=np.int32)

            frame_features = OrderedDict()
            if features is not None:
                values = _as_point_features(features[t], n=len(frame_coords), frame=t)
                if t == 0:
                    feature_width = values.shape[1]
                elif values.shape[1] != feature_width:
                    raise ValueError(
                        "All point feature arrays must have the same width, got "
                        f"{feature_width} and {values.shape[1]}"
                    )
                frame_features[FEATURE_CUSTOM] = values

            point_features.append(
                WRFeatures(
                    coords=frame_coords,
                    labels=frame_labels,
                    timepoints=timepoints,
                    features=frame_features,
                )
            )

        expected_width = int(self.transformer.config.get("feat_dim", 0))
        if feature_width != expected_width:
            raise ValueError(
                "Point feature width does not match the model: "
                f"got {feature_width}, expected {expected_width}"
            )

        logger.info("Building point windows")
        windows = build_windows(
            point_features,
            window_size=self.transformer.config["window"],
            progbar_class=progbar_class,
            as_torch=True,
            feature_mode="none" if features is None else FEATURE_CUSTOM,
        )
        return point_features, windows, ndim

    def _track_from_predictions(
        self,
        predictions,
        mode: Literal["greedy_nodiv", "greedy", "ilp"] = "greedy",
        use_distance: bool = False,
        max_distance: int | None = None,
        max_neighbors: int | None = None,
        delta_t: int = 1,
        return_candidate: bool = False,
        **kwargs,
    ) -> nx.DiGraph | tuple[nx.DiGraph, nx.DiGraph]:
        logger.info("Running greedy tracker")
        nodes = predictions["nodes"]
        weights = predictions["weights"]

        # Spatial radius and neighbour count default to the values the model was
        # trained with (stored in its config) so inference matches training.
        config = self.transformer.config
        max_distance = _resolve_inference_max_distance(
            config["max_distance"], max_distance
        )
        if max_neighbors is None:
            # config["max_neighbors"] is a (lo, hi) pair; the tracking candidate
            # graph uses the larger hi (inference always takes the larger one).
            max_neighbors = max(config["max_neighbors"])

        candidate_graph = build_graph(
            nodes=nodes,
            weights=weights,
            use_distance=use_distance,
            max_distance=max_distance,
            max_neighbors=max_neighbors,
            delta_t=delta_t,
        )
        if mode == "greedy":
            solution = track_greedy(candidate_graph)
        elif mode == "greedy_nodiv":
            solution = track_greedy(candidate_graph, allow_divisions=False)
        elif mode == "ilp":
            from trackastra.tracking.ilp import track_ilp

            solution = track_ilp(candidate_graph, ilp_config="gt", **kwargs)
        else:
            raise ValueError(f"Tracking mode {mode} does not exist.")
        if return_candidate:
            return solution, candidate_graph
        return solution

    def track_points(
        self,
        coords: Sequence[np.ndarray],
        features: Sequence[np.ndarray] | None = None,
        mode: Literal["greedy_nodiv", "greedy", "ilp"] = "greedy",
        labels: Sequence[np.ndarray] | None = None,
        return_details: bool = False,
        edge_threshold: float = 0.05,
        batch_size: int | None = None,
        progbar_class=tqdm,
        **kwargs,
    ) -> nx.DiGraph | tuple[nx.DiGraph, dict]:
        """Track point detections whose coordinates are already in model space.

        Args:
            coords: Per-frame coordinate arrays with shape ``(N_t, ndim)``.
                Coordinates must already be scaled into the isotropic/model-space
                units expected by the checkpoint.
            features: Optional per-frame scalar feature arrays with shape
                ``(N_t, F)``. If omitted, the model must have ``feat_dim=0``.
            mode: Tracking mode, one of ``"greedy_nodiv"``, ``"greedy"``, or
                ``"ilp"``.
            labels: Optional unique per-frame labels. Defaults to ``1..N_t``.
            return_details: If True, return diagnostics with candidate graph and
                raw predictions.
            edge_threshold: Minimum association score used when collecting edges
                from window predictions.
            batch_size: Batch size for prediction. If None, uses the model default.
            progbar_class: Progress bar class to use.
            **kwargs: Additional arguments passed to the tracking algorithm.

        Returns:
            ``nx.DiGraph`` containing the point tracking result, and optionally a
            diagnostics dict with ``candidate_graph`` and ``predictions``.
        """
        point_features, windows, ndim = self._extract_point_windows(
            coords,
            features=features,
            labels=labels,
            progbar_class=progbar_class,
        )
        predictions = self._predict_from_windows(
            point_features,
            windows,
            spatial_dim=ndim,
            edge_threshold=edge_threshold,
            batch_size=batch_size,
            progbar_class=progbar_class,
        )

        if return_details:
            track_graph, candidate_graph = self._track_from_predictions(
                predictions, mode=mode, return_candidate=True, **kwargs
            )
            return track_graph, {
                "candidate_graph": candidate_graph,
                "predictions": predictions,
            }
        return self._track_from_predictions(predictions, mode=mode, **kwargs)

    def track(
        self,
        imgs: np.ndarray | da.Array,
        masks: np.ndarray | da.Array,
        mode: Literal["greedy_nodiv", "greedy", "ilp"] = "greedy",
        normalize_imgs: bool = True,
        progbar_class=tqdm,
        n_workers: int = 0,
        batch_size: int | None = None,
        normalize_diameter: float | None = None,
        return_details: bool = False,
        **kwargs,
    ) -> tuple[nx.DiGraph, np.ndarray] | tuple[nx.DiGraph, np.ndarray, dict]:
        """Track objects across time frames.

        This method links segmented objects across time frames using the specified
        tracking mode. No hyperparameters need to be chosen beyond the tracking mode.

        Args:
            imgs: Input images of shape (T,(Z),Y,X) (numpy or dask array)
            masks: Instance segmentation masks of shape (T,(Z),Y,X).
            mode: Tracking mode:
                - "greedy_nodiv": Fast greedy linking without division
                - "greedy": Fast greedy linking with division
                - "ilp": Integer Linear Programming based linking (more accurate but slower)
            progbar_class: Progress bar class to use.
            n_workers: Number of worker processes for feature extraction.
            normalize_imgs: Whether to normalize the images.
            batch_size: Batch size for prediction. If None, defaults to 1 on CPU and 16 on GPU.
            normalize_diameter: If set, scale spatial WR features so the movie's
                median equivalent diameter equals this value. If None, uses the value
                stored in the model's inference config when present.
            return_details: If True, additionally return a diagnostics dict with the
                pre-solution ``candidate_graph`` (all scored edges with weights) and
                the raw ``predictions`` (nodes + edge weights). Cheap: reuses the work
                already done, nothing is recomputed.
            **kwargs: Additional arguments passed to tracking algorithm.

        Returns:
            nx.DiGraph containing the tracking results.
            np.ndarray of tracked masks of shape (T,(Z),Y,X).
            (only if ``return_details``) dict with keys ``candidate_graph`` and
            ``predictions``.
        """
        # Pretrained-feature setup (no-op for wrfeat) now happens inside
        # _extract_features_windows, shared with the cached validation path.
        predictions = self._predict(
            imgs,
            masks,
            normalize_imgs=normalize_imgs,
            progbar_class=progbar_class,
            n_workers=n_workers,
            batch_size=batch_size or self.batch_size,
            normalize_diameter=normalize_diameter,
        )

        if return_details:
            track_graph, candidate_graph = self._track_from_predictions(
                predictions, mode=mode, return_candidate=True, **kwargs
            )
        else:
            track_graph = self._track_from_predictions(predictions, mode=mode, **kwargs)
        masks_tracked = apply_solution_graph_to_masks(track_graph, masks)
        if return_details:
            details = {
                "candidate_graph": candidate_graph,
                "predictions": predictions,
            }
            return track_graph, masks_tracked, details
        return track_graph, masks_tracked

    def track_from_disk(
        self,
        imgs_path: Path,
        masks_path: Path,
        mode: Literal["greedy_nodiv", "greedy", "ilp"] = "greedy",
        normalize_imgs: bool = True,
        **kwargs,
    ) -> tuple[nx.DiGraph, np.ndarray]:
        """Track objects directly from image and mask files on disk.

        This method supports both single tiff files and directories

        Args:
            imgs_path: Path to input images. Can be:
                - Directory containing numbered tiff files of shape (C),(Z),Y,X
                - Single tiff file with time series of shape T,(C),(Z),Y,X
            masks_path: Path to mask files. Can be:
                - Directory containing numbered tiff files of shape (Z),Y,X
                - Single tiff file with time series of shape T,(Z),Y,X
            mode: Tracking mode:
                - "greedy_nodiv": Fast greedy linking without division
                - "greedy": Fast greedy linking with division
                - "ilp": Integer Linear Programming based linking (more accurate but slower)
            normalize_imgs: Whether to normalize the images.
            **kwargs: Additional arguments passed to tracking algorithm.

        Returns:
            Tuple of (nx.DiGraph, tracked masks).
        """
        if not imgs_path.exists():
            raise FileNotFoundError(f"{imgs_path=} does not exist.")
        if not masks_path.exists():
            raise FileNotFoundError(f"{masks_path=} does not exist.")

        self.imgs_path = imgs_path
        self.masks_path = masks_path

        if imgs_path.is_dir():
            imgs = load_tiff_timeseries(imgs_path)
        else:
            imgs = tifffile.imread(imgs_path)

        if masks_path.is_dir():
            masks = load_tiff_timeseries(masks_path)
        else:
            masks = tifffile.imread(masks_path)

        if len(imgs) != len(masks):
            raise RuntimeError(
                f"#imgs and #masks do not match. Found {len(imgs)} images,"
                f" {len(masks)} masks."
            )

        if imgs.ndim - 1 == masks.ndim:
            if imgs[1] == 1:
                logger.info(
                    "Found a channel dimension with a single channel. Removing dim."
                )
                masks = np.squeeze(masks, 1)
            else:
                raise RuntimeError(
                    "Trackastra currently only supports single channel images."
                )

        if imgs.shape != masks.shape:
            raise RuntimeError(
                f"Img shape {imgs.shape} and mask shape {masks.shape} do not match."
            )

        return self.track(imgs, masks, mode, normalize_imgs=normalize_imgs, **kwargs)
