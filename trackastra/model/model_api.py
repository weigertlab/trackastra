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
from ..tracking import apply_solution_graph_to_masks, build_graph, track_greedy
from ..utils import normalize
from .model import TrackingTransformer
from .predict import predict_windows
from .pretrained import download_pretrained

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

        self.batch_size = _default_batch_size(self.device)

        self.transformer = transformer.to(self.device)
        self.train_args = train_args

    @classmethod
    def from_folder(cls, dir: Path | str, device: str | None = None):
        """Load a Trackastra model from a local folder.

        Args:
            dir: Path to model folder containing:
                - model weights
                - train_config.yaml with training arguments
            device: Device to run model on.

        Returns:
            Trackastra model instance.
        """
        # Always load to cpu first
        transformer = TrackingTransformer.from_folder(
            Path(dir).expanduser(), map_location="cpu"
        )
        train_args = yaml.load(open(dir / "train_config.yaml"), Loader=yaml.FullLoader)
        return cls(transformer=transformer, train_args=train_args, device=device)

    @classmethod
    def from_pretrained(
        cls, name: str, device: str | None = None, download_dir: Path | None = None
    ):
        """Load a pretrained Trackastra model.

        Available pretrained models are described in detail in pretrained.json.

        Args:
            name: Name of pretrained model (e.g. "general_2d").
            device: Device to run model on ("cuda", "mps", "cpu", "automatic" or None).
            download_dir: Directory to download model to (defaults to ~/.cache/trackastra).

        Returns:
            Trackastra model instance.
        """
        folder = download_pretrained(name, download_dir)
        # download zip from github to location/name, then unzip
        return cls.from_folder(folder, device=device)

    def _predict(
        self,
        imgs: np.ndarray | da.Array,
        masks: np.ndarray | da.Array,
        edge_threshold: float = 0.05,
        n_workers: int = 0,
        normalize_imgs: bool = True,
        progbar_class=tqdm,
        batch_size: int | None = None,
    ):
        batch_size = self.batch_size if batch_size is None else batch_size

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
            ndim=self.transformer.config["coord_dim"],
            n_workers=n_workers,
            progbar_class=progbar_class,
        )
        logger.info("Building windows")
        windows = build_windows(
            features,
            window_size=self.transformer.config["window"],
            progbar_class=progbar_class,
            as_torch=True,
        )

        logger.info("Predicting windows")
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
        max_distance: int = 256,
        max_neighbors: int = 10,
        delta_t: int = 1,
        **kwargs,
    ) -> nx.DiGraph:
        logger.info("Running greedy tracker")
        nodes = predictions["nodes"]
        weights = predictions["weights"]

        candidate_graph = build_graph(
            nodes=nodes,
            weights=weights,
            use_distance=use_distance,
            max_distance=max_distance,
            max_neighbors=max_neighbors,
            delta_t=delta_t,
        )
        if mode == "greedy":
            return track_greedy(candidate_graph)
        elif mode == "greedy_nodiv":
            return track_greedy(candidate_graph, allow_divisions=False)
        elif mode == "ilp":
            from trackastra.tracking.ilp import track_ilp

            return track_ilp(candidate_graph, ilp_config="gt", **kwargs)
        else:
            raise ValueError(f"Tracking mode {mode} does not exist.")

    def track(
        self,
        imgs: np.ndarray | da.Array,
        masks: np.ndarray | da.Array,
        mode: Literal["greedy_nodiv", "greedy", "ilp"] = "greedy",
        normalize_imgs: bool = True,
        progbar_class=tqdm,
        n_workers: int = 0,
        batch_size: int | None = None,
        **kwargs,
    ) -> tuple[nx.DiGraph, np.ndarray]:
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
            **kwargs: Additional arguments passed to tracking algorithm.

        Returns:
            nx.DiGraph containing the tracking results.
        """
        predictions = self._predict(
            imgs,
            masks,
            normalize_imgs=normalize_imgs,
            progbar_class=progbar_class,
            n_workers=n_workers,
            batch_size=batch_size,
        )

        track_graph = self._track_from_predictions(predictions, mode=mode, **kwargs)
        masks_tracked = apply_solution_graph_to_masks(track_graph, masks)

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
