import logging
import os
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
from ..data.wrfeat import scale_to_target_diameter
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
        train_args: dict,
        device: Literal["cuda", "mps", "cpu", "automatic", None] = None,
        batch_size: int | None = None,
    ):
        """Initialize Trackastra model.

        Args:
            transformer: The underlying transformer model.
            train_args: Training configuration arguments.
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
        self.train_args = train_args
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
                - train_config.yaml with training arguments
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
        train_args = yaml.load(open(dir / "train_config.yaml"), Loader=yaml.FullLoader)
        return cls(
            transformer=transformer, train_args=train_args, device=device, **kwargs
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

    def _predict(
        self,
        imgs: np.ndarray | da.Array,
        masks: np.ndarray | da.Array,
        edge_threshold: float = 0.05,
        n_workers: int = 0,
        normalize_imgs: bool = True,
        progbar_class=tqdm,
        batch_size: int | None = None,
        scale_target_diameter: float | None = None,
    ):
        logger.info("Predicting weights for candidate graph")
        if normalize_imgs:
            if isinstance(imgs, da.Array):
                imgs = imgs.map_blocks(normalize)
            else:
                imgs = normalize(imgs)

        self.transformer.eval()

        features = get_features(
            detections=masks,
            imgs=imgs,
            features=self.train_args["features"],
            feature_extractor=self.feature_extractor,
            ndim=self.transformer.config["coord_dim"],
            n_workers=n_workers,
            progbar_class=progbar_class,
        )
        if scale_target_diameter is None:
            scale_target_diameter = self.train_args.get("scale_target_diameter")
        features = scale_to_target_diameter(features, scale_target_diameter)
        logger.info("Building windows")
        windows = build_windows(
            features,
            window_size=self.transformer.config["window"],
            progbar_class=progbar_class,
            as_torch=True,
            feature_mode=(
                self.train_args["features"]
                if self.train_args["features"]
                in ("wrfeat", "wrfeat2", "wrfeat2_no_intensity")
                else "wrfeat"
            ),
        )

        batch_size = batch_size or self.batch_size
        logger.info(f"Predicting windows with batch size {batch_size}")
        predictions = predict_windows(
            windows=windows,
            features=features,
            model=self.transformer,
            edge_threshold=edge_threshold,
            spatial_dim=masks.ndim - 1,
            progbar_class=progbar_class,
            batch_size=batch_size,
        )

        return predictions

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

    def track(
        self,
        imgs: np.ndarray | da.Array,
        masks: np.ndarray | da.Array,
        mode: Literal["greedy_nodiv", "greedy", "ilp"] = "greedy",
        normalize_imgs: bool = True,
        progbar_class=tqdm,
        n_workers: int = 0,
        batch_size: int | None = None,
        scale_target_diameter: float | None = None,
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
            scale_target_diameter: If set, scale spatial WR features so the movie's
                median equivalent diameter equals this value. If None, uses the value
                stored in the model's train config when present.
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
        # Pretrained feature extraction requires the trackastra_pretrained_feats package
        feat_type = self.train_args["features"]
        if feat_type == "pretrained_feats" or feat_type == "pretrained_feats_aug":
            if not PRETRAINED_FEATS_INSTALLED:
                raise ImportError(
                    "The trackastra_pretrained_feats package is required for pretrained feature extraction."
                    "Please install it with :"
                    "pip install trackastra[etultra]"
                )
            additional_features = self.train_args.get(
                "pretrained_feats_additional_props", None
            )
            if self.imgs_path is None:
                save_path = "./embeddings"
            else:
                save_path = self.imgs_path / "embeddings"
            self.feature_extractor = FeatureExtractor.from_model_name(
                self.train_args["pretrained_feats_model"],
                imgs.shape[-2:],
                save_path=save_path,
                mode=self.train_args["pretrained_feats_mode"],
                device=self.device,
                additional_features=additional_features,
                batch_size=batch_size or self.batch_size,
            )
            self.feature_extractor.force_recompute = True

        predictions = self._predict(
            imgs,
            masks,
            normalize_imgs=normalize_imgs,
            progbar_class=progbar_class,
            n_workers=n_workers,
            batch_size=batch_size or self.batch_size,
            scale_target_diameter=scale_target_diameter,
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
