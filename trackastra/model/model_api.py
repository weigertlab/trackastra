import logging
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from pydantic import validate_call
from tqdm import tqdm

from ..data import build_windows, get_features
from ..tracking import build_graph, track_greedy
from ..utils import normalize
from .model import TrackingTransformer
from .predict import predict_windows
from .pretrained import download_pretrained

logger = logging.getLogger(__name__)


class Trackastra:
    def __init__(self, transformer, train_args, device="cpu"):
        if device == "mps":
            raise NotImplementedError("Trackastra on mps not supported.")
        # Hack: to(device) for some more submodules that map_location does cover
        self.transformer = transformer.to(device)
        self.train_args = train_args
        self.device = device

    @classmethod
    @validate_call
    def from_folder(cls, dir: Path, device: str = "cpu"):
        transformer = TrackingTransformer.from_folder(dir, map_location=device)
        train_args = yaml.load(open(dir / "train_config.yaml"), Loader=yaml.FullLoader)
        return cls(transformer=transformer, train_args=train_args, device=device)

    # TODO make safer
    @classmethod
    @validate_call
    def from_pretrained(
        cls, name: str, device: str = "cpu", download_dir: Path | None = None
    ):
        folder = download_pretrained(name, download_dir)
        # download zip from github to location/name, then unzip
        return cls.from_folder(folder, device=device)

    def _predict(
        self,
        imgs: np.ndarray,
        masks: np.ndarray,
        edge_threshold: float = 0.05,
        n_workers: int = 0,
        progbar_class=tqdm,
    ):
        logger.info("Predicting weights for candidate graph")
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
        )

        logger.info("Predicting windows")
        predictions = predict_windows(
            windows=windows,
            features=features,
            model=self.transformer,
            edge_threshold=edge_threshold,
            spatial_dim=masks.ndim - 1,
            progbar_class=progbar_class,
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
    ):

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
        imgs: np.ndarray,
        masks: np.ndarray,
        mode: Literal["greedy", "ilp"] = "greedy",
        progbar_class=tqdm,
        **kwargs,
    ):
        predictions = self._predict(imgs, masks, progbar_class=progbar_class)
        track_graph = self._track_from_predictions(predictions, mode=mode, **kwargs)
        return track_graph
