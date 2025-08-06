from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from timeit import default_timer
from typing import TYPE_CHECKING, ClassVar, Literal

import joblib
import lz4.frame
import networkx as nx
import numpy as np
import pandas as pd
import tifffile
import torch
import zarr
from numba import njit
from scipy import ndimage as ndi
from scipy.spatial.distance import cdist
from skimage.measure import regionprops
from skimage.segmentation import relabel_sequential
from torch.utils.data import Dataset
from tqdm import tqdm

from trackastra.data import wrfeat
from trackastra.data._check_ctc import _check_ctc, _get_node_attributes
from trackastra.data.augmentations import (
    AugmentationPipeline,
    RandomCrop,
    default_augmenter,
)
from trackastra.data.features import (
    _PROPERTIES,
    extract_features_patch,
    extract_features_regionprops,
)
from trackastra.data.matching import matching
from trackastra.data.pretrained_augmentations import PretrainedAugmentations
from trackastra.utils import blockwise_sum, masks2properties, normalize

from .utils import make_hashable

logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
logger.setLevel(logging.DEBUG)  # FIXME go back to INFO for release

if TYPE_CHECKING:
    from trackastra.data.pretrained_features import (
        PretrainedBackboneType,
        PretrainedFeatsExtractionMode,
        PretrainedFeatureExtractorConfig,
    )
    

def _filter_track_df(df, start_frame, end_frame, downscale):
    """Only keep tracklets that are present in the given time interval."""
    # only retain cells in interval
    df = df[(df.t2 >= start_frame) & (df.t1 < end_frame)]

    # shift start and end of each cell
    df.t1 = df.t1 - start_frame
    df.t2 = df.t2 - start_frame
    # set start/end to min/max
    df.t1 = df.t1.clip(0, end_frame - start_frame - 1)
    df.t2 = df.t2.clip(0, end_frame - start_frame - 1)
    # set all parents to 0 that are not in the interval
    df.loc[~df.parent.isin(df.label), "parent"] = 0

    if downscale > 1:
        if start_frame % downscale != 0:
            raise ValueError("start_frame must be a multiple of downscale")

        logger.info(f"Temporal downscaling of tracklet links by {downscale}")

        # remove tracklets that have been fully deleted by temporal downsampling

        mask = (
            # (df["t2"] - df["t1"] < downscale - 1)
            (df["t1"] % downscale != 0)
            & (df["t2"] % downscale != 0)
            & (df["t1"] // downscale == df["t2"] // downscale)
        )
        logger.info(
            f"Remove {mask.sum()} tracklets that are fully deleted by downsampling"
        )
        logger.debug(f"Remove {df[mask]}")

        df = df[~mask]
        # set parent to 0 if it has been deleted
        df.loc[~df.parent.isin(df.label), "parent"] = 0

        df["t2"] = (df["t2"] / float(downscale)).apply(np.floor).astype(int)
        df["t1"] = (df["t1"] / float(downscale)).apply(np.ceil).astype(int)

        # Correct for edge case of single frame tracklet
        assert np.all(df["t1"] == np.minimum(df["t1"], df["t2"]))

    return df


class _CompressedArray:
    """a simple class to compress and decompress a numpy arrays using lz4."""

    # dont compress float types
    def __init__(self, data):
        self._data = lz4.frame.compress(data)
        self._dtype = data.dtype.type
        self._shape = data.shape

    def decompress(self):
        s = lz4.frame.decompress(self._data)
        data = np.frombuffer(s, dtype=self._dtype).reshape(self._shape)
        return data


def debug_function(f):
    def wrapper(*args, **kwargs):
        try:
            batch = f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {f.__name__}: {e}")
            return None
        logger.info(f"XXXX {len(batch['coords'])}")
        return batch

    return wrapper


class CTCData(Dataset):
    """Cell Tracking Challenge data loader."""
    # Amount of feature per mode per dimension
    FEATURES_DIMENSIONS: ClassVar = {
        "regionprops": {
            2: 7,
            3: 11,
        },
        "regionprops2": {
            2: 6,
            3: 11,
        },
        "regionprops_full": {
            2: 9,
            3: 13,
        },
        "patch": {
            2: 256,
            3: 256,
        },
        "patch_regionprops": {
            2: 256 + 5,
            3: 256 + 8,
        },
        "none": {
            2: 0,
            3: 0,
        }
        # "wrfeat" -> defined by wrfeat
        # "pretrained_feats":{ # -> defined by PretrainedFeatureExtractorConfig.feat_dim
    }
    VALID_FEATURES: ClassVar = {
        "none",
        "regionprops",
        "regionprops2",
        "patch",
        "patch_regionprops",
        "wrfeat",
        "pretrained_feats",
        "pretrained_feats_aug",
    }
    
    def __new__(cls, *args, **kwargs):
        # Check if features is "pretrained_feats_aug"; if it is, use CTCDataAugPretrainedFeats class
        if kwargs.get("features") == "pretrained_feats_aug":
            return super().__new__(globals()["CTCDataAugPretrainedFeats"])
        return super().__new__(cls)

    def __init__(
        self,
        root: str = "",
        ndim: int = 2,
        use_gt: bool = True,
        detection_folders: list[str] = ["TRA"],
        window_size: int = 10,
        max_tokens: int | None = None,
        slice_pct: tuple = (0.0, 1.0),
        downscale_spatial: int = 1,
        downscale_temporal: int = 1,
        augment: int = 0,
        features: Literal[
            "none",
            "regionprops",
            "regionprops2",
            "patch",
            "patch_regionprops",
            "wrfeat",
            "pretrained_feats",
            "pretrained_feats_aug",
        ] = "wrfeat",
        sanity_dist: bool = False,
        crop_size: tuple | None = None,
        return_dense: bool = False,
        compress: bool = False,
        pretrained_backbone_config: PretrainedFeatureExtractorConfig | None = None,
        # pca_preprocessor: EmbeddingsPCACompression | None = None,
        rotate_features: bool = False,
        load_immediately: bool = True,
        **kwargs,
    ) -> None:
        """Args:
        root (str):
            Folder containing the CTC TRA folder.
        ndim (int):
            Number of dimensions of the data. Defaults to 2d
            (if ndim=3 and data is two dimensional, it will be cast to 3D)
        detection_folders:
            List of relative paths to folder with detections.
            Defaults to ["TRA"], which uses the ground truth detections.
        window_size (int):
            Window size for transformer.
        slice_pct (tuple):
            Slice the dataset by percentages (from, to).
        augment (int):
            if 0, no data augmentation. if > 0, defines level of data augmentation.
        features (str):
            Types of features to use.
        sanity_dist (bool):
            Use euclidian distance instead of the association matrix as a target.
        crop_size (tuple):
            Size of the crops to use for augmentation. If None, no cropping is used.
        return_dense (bool):
            Return dense masks and images in the data samples.
        compress (bool):
            Compress elements/remove img if not needed to save memory for large datasets
        pretrained_backbone_config (PretrainedFeatureExtractorConfig):
            Configuration for the pretrained backbone.
            If mode is set to "pretrained_feats", this configuration is used to extract features.
            Ignored otherwise.
        rotate_features (bool):
            Apply rotation to features based on (augmented) coordinates.
            Only valid if used with "pretrained_feats" or "pretrained_feats_aug" mode.
        load_immediately (bool):
            If True, load the data immediately. If False, load the data lazily.
            If False, you need to call `start_loading()` to load the data.
            Defaults to True.
        # pca_preprocessor (EmbeddingsPCACompression):
        #     PCA preprocessor for the pretrained features.
        #     If mode is set to "pretrained_feats", this is used to reduce the dimensionality of the features.
        #     Ignored otherwise.
        """
        super().__init__()

        self.root = Path(root)
        self.name = self.root.name
        self.use_gt = use_gt
        self.slice_pct = slice_pct
        if not 0 <= slice_pct[0] < slice_pct[1] <= 1:
            raise ValueError(f"Invalid slice_pct {slice_pct}")
        self.downscale_spatial = downscale_spatial
        self.downscale_temporal = downscale_temporal
        self.detection_folders = detection_folders
        self._ndim = ndim
        self.features = features
        self.rotate_feats = rotate_features

        if features not in self.VALID_FEATURES:
            if features not in _PROPERTIES[self._ndim] and features != "wrfeat":
                raise ValueError(
                    f"'{features}' not one of the supported {self._ndim}D features"
                    f" {tuple(_PROPERTIES[self._ndim].keys())}"
                )
        
        if features == "pretrained_feats" or features == "pretrained_feats_aug":
            try:
                if TYPE_CHECKING:
                    import transformers
                    transformers.__version__
            except ImportError as e:
                msg = """Please install pretrained_feats extra requirements to use pretrained features mode.\n
                Run :
                    pip install trackastra[pretrained_feats]
                to install the required dependencies.
                """
                raise ImportError(msg) from e        

        logger.info(f"ROOT (config): \t{self.root}")
        self.root, self.gt_tra_folder = self._guess_root_and_gt_tra_folder(self.root)
        logger.info(f"ROOT (guessed): \t{self.root}")
        logger.info(f"GT TRA (guessed):\t{self.gt_tra_folder}")
        if self.use_gt:
            self.gt_mask_folder = self._guess_mask_folder(self.root, self.gt_tra_folder)
        else:
            logger.info("Using dummy masks as GT")
            self.gt_mask_folder = self._guess_det_folder(
                self.root, self.detection_folders[0]
            )
        logger.info(f"GT MASK (guessed):\t{self.gt_mask_folder}")

        # dont load image data if not needed
        if features in ("none",):
            self.img_folder = None
        else:
            self.img_folder = self._guess_img_folder(self.root)
        logger.info(f"IMG (guessed):\t{self.img_folder}")

        self._pretrained_config = None
        if features == "pretrained_feats" or features == "pretrained_feats_aug": 
            if pretrained_backbone_config is None:
                raise ValueError("Pretrained backbone config must be provided for pretrained features mode.")
            self.pretrained_config = pretrained_backbone_config
            if self.pretrained_config.save_path is None:
                self.pretrained_config.save_path = self.img_folder
            self.FEATURES_DIMENSIONS["pretrained_feats"] = self.pretrained_config.feat_dim
        
        self.augment_level = augment
        self.crop_size = crop_size
        self.augmenter, self.cropper = self._setup_features_augs()

        if window_size <= 1:
            raise ValueError("window must be >1")
        self.window_size = window_size
        self.max_tokens = max_tokens

        self.slice_pct = slice_pct
        self.sanity_dist = sanity_dist
        self.return_dense = return_dense
        self.compress = compress
        self.start_frame = 0
        self.end_frame = None
        
        # Pretrained model attributes for feature extraction if specified
        self._pretrained_model_input_size_factor = 1
        self.feature_extractor_input_size = None
        self.feature_extractor_save_path = None
        self.pretrained_features = None
        self.feature_extractor = None
        self.pretrained_feature_augmenter = None
        # self.pca_preprocessor = pca_preprocessor
        
        if load_immediately:
            self.start_loading()
        
        if kwargs:
            logger.warning(f"Unused kwargs: {kwargs}")

    def start_loading(self):
        start = default_timer()

        self._init_features()  # loads and creates windows

        self.n_divs = self._get_ndivs(self.windows)

        # if len(self.windows) > 0:
        #     self.ndim = self.windows[0]["coords"].shape[1]
        #     self.n_objects = tuple(len(t["coords"]) for t in self.windows)
        #     logger.info(
        #         f"Found {np.sum(self.n_objects)} objects in {len(self.windows)} track"
        #         f" windows from {self.root} ({default_timer() - start:.1f}s)\n"
        #     )
        # else:
        #     self.n_objects = 0
        #     logger.warning(f"Could not load any tracks from {self.root}")
        self._get_ndim_and_nobj(start)

        if self.compress:
            self._compress_data()

    def _init_features(self):
        if self.features == "wrfeat" or self.features == "pretrained_feats":
            self.windows = self._load_wrfeat()
        else:
            self.windows = self._load()
    
    @property
    def config(self):
        return {
            "root": str(self.root),
            "ndim": self.ndim,
            "use_gt": self.use_gt,
            "detection_folders": self.detection_folders,
            "window_size": self.window_size,
            "max_tokens": self.max_tokens,
            "slice_pct": self.slice_pct,
            "downscale_spatial": self.downscale_spatial,
            "downscale_temporal": self.downscale_temporal,
            "augment": self.augment_level,
            "features": self.features,
            "sanity_dist": self.sanity_dist,
            "crop_size": self.crop_size,
            "return_dense": self.return_dense,
            "compress": self.compress,
            "pretrained_config": (
                self.pretrained_config.to_dict() if self.pretrained_config else None
            ),
            "rotate_features": self.rotate_feats,
        }
    
    @property
    def config_hash(self):
        """Returns a hash of the configuration."""
        cfg = make_hashable(self.config)
        config_str = json.dumps(cfg, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()
    
    @property
    def ndim(self):
        return self._ndim
    
    @ndim.setter
    def ndim(self, value: int):
        if value not in (2, 3):
            raise ValueError(f"ndim must be 2 or 3, got {value}")
        self._ndim = value
    
    @property
    def feat_dim(self):
        if self.pretrained_config is None:
            return self.FEATURES_DIMENSIONS[self.features][self.ndim]
        elif self.features == "wrfeat":
            return wrfeat.WRFeatures.PROPERTIES_DIMS[
                wrfeat.DEFAULT_PROPERTIES
            ][self.ndim]
        else:
            return self.pretrained_config.additional_feat_dim
    
    @property
    def pretrained_config(self):
        return self._pretrained_config
    
    @property
    def pretrained_feat_dim(self):
        if self._pretrained_config is None:
            return 0
        return self._pretrained_config.feat_dim
    
    @pretrained_config.setter
    def pretrained_config(self, config: PretrainedFeatureExtractorConfig):
        if isinstance(config, dict):
            from trackastra.data.pretrained_features import (
                PretrainedFeatureExtractorConfig,
            )
            config = PretrainedFeatureExtractorConfig.from_dict(config)
        self.update_pretrained_feat_dim(config)
        self._pretrained_config = config
    
    def update_pretrained_feat_dim(self, config):
        try:
            self.FEATURES_DIMENSIONS["pretrained_feats"] = config.feat_dim
        except AttributeError as e:
            if isinstance(config, dict):
                self.FEATURES_DIMENSIONS["pretrained_feats"] = config["feat_dim"]
            else:
                raise e
    
    @staticmethod
    def get_feat_dim(features, ndim, ):
        return CTCData.FEATURES_DIMENSIONS[features][ndim]
    
    @classmethod
    def from_arrays(cls, imgs: np.ndarray, masks: np.ndarray, train_args: dict):
        self = cls(**train_args)
    # def from_ctc
        # for key, value in train_args.items():
        #     setattr(self, key, value)

        # self.use_gt = use_gt
        # self.slice_pct = slice_pct
        # if not 0 <= slice_pct[0] < slice_pct[1] <= 1:
        # raise ValueError(f"Invalid slice_pct {slice_pct}")
        # self.downscale_spatial = downscale_spatial
        # self.downscale_temporal = downscale_temporal
        # self.detection_folders = detection_folders
        # self.ndim = ndim
        # self.features = features

        # if features not in ("none", "wrfeat") and features not in _PROPERTIES[ndim]:
        # raise ValueError(
        # f"'{features}' not one of the supported {ndim}D features {tuple(_PROPERTIES[ndim].keys())}"
        # )

        # logger.info(f"ROOT (config): {self.root}")
        # self.root, self.gt_tra_folder = self._guess_root_and_gt_tra_folder(self.root)
        # logger.info(f"ROOT: \t{self.root}")
        # logger.info(f"GT TRA:\t{self.gt_tra_folder}")
        # if self.use_gt:
        # self.gt_mask_folder = self._guess_mask_folder(self.root, self.gt_tra_folder)
        # else:
        # logger.info("Using dummy masks as GT")
        # self.gt_mask_folder = self._guess_det_folder(
        # self.root, self.detection_folders[0]
        # )
        # logger.info(f"GT MASK:\t{self.gt_mask_folder}")

        # dont load image data if not needed
        # if features in ("none",):
        # self.img_folder = None
        # else:
        # self.img_folder = self._guess_img_folder(self.root)
        # logger.info(f"IMG:\t\t{self.img_folder}")

        self.augmenter, self.cropper = self._setup_features_augs()

        start = default_timer()

        if self.features == "wrfeat" or self.features == "pretrained_feats":
            self.windows = self._load_wrfeat()
        else:
            self.windows = self._load()

        self.n_divs = self._get_ndivs(self.windows)

        if len(self.windows) > 0:
            self.ndim = self.windows[0]["coords"].shape[1]
            self.n_objects = tuple(len(t["coords"]) for t in self.windows)
            logger.info(
                f"Found {np.sum(self.n_objects)} objects in {len(self.windows)} track"
                f" windows from {self.root} ({default_timer() - start:.1f}s)\n"
            )
        else:
            self.n_objects = 0
            logger.warning(f"Could not load any tracks from {self.root}")

        if self.compress:
            self._compress_data()
    
    def _get_ndim_and_nobj(self, start):
        if len(self.windows) > 0:
            self.ndim = self.windows[0]["coords"].shape[1]
            self.n_objects = tuple(len(t["coords"]) for t in self.windows)
            logger.info(
                f"Found {np.sum(self.n_objects)} objects in {len(self.windows)} track"
                f" windows from {self.root} ({default_timer() - start:.1f}s)\n"
            )
        else:
            self.n_objects = 0
            logger.warning(f"Could not load any tracks from {self.root}")
    
    def _get_ndivs(self, windows):
        n_divs = []
        for w in tqdm(windows, desc="Counting divisions", leave=False):
            _n = (
                (
                    blockwise_sum(
                        torch.from_numpy(w["assoc_matrix"]).float(),
                        torch.from_numpy(w["timepoints"]).long(),
                    ).max(dim=0)[0]
                    == 2
                )
                .sum()
                .item()
            )
            n_divs.append(_n)
        return n_divs

    def _setup_features_augs(
        self
    ):
        if self.features in ["wrfeat", "pretrained_feats"]:
            return self._setup_features_augs_wrfeat()

        cropper = (
            RandomCrop(
                crop_size=self.crop_size,
                ndim=self.ndim,
                use_padding=False,
                ensure_inside_points=True,
            )
            if self.crop_size is not None
            else None
        )

        # Hack
        if self.features == "none":
            return default_augmenter, cropper

        augmenter = AugmentationPipeline(p=0.8, level=self.augment_level) if self.augment_level else None

        return augmenter, cropper

    def _compress_data(self):
        # compress masks and assoc_matrix
        logger.info("Compressing masks and assoc_matrix to save memory")
        for w in self.windows:
            if "mask" in w:
                w["mask"] = _CompressedArray(w["mask"])
            # dont compress full imgs (as needed for patch features)
            if "img" in w:
                w["img"] = _CompressedArray(w["img"])
            w["assoc_matrix"] = _CompressedArray(w["assoc_matrix"])
        self.gt_masks = _CompressedArray(self.gt_masks)
        self.det_masks = {k: _CompressedArray(v) for k, v in self.det_masks.items()}
        # dont compress full imgs (as needed for patch features)
        self.imgs = _CompressedArray(self.imgs)

    @staticmethod
    def _guess_root_and_gt_tra_folder(inp: Path):
        """Guesses the root and the ground truth folder from a given input path.

        Args:
            inp (Path): _description_

        Returns:
            Path: root folder,
        """
        if inp.name == "TRA":
            # 01_GT/TRA --> 01, 01_GT/TRA
            root = inp.parent.parent / inp.parent.name.split("_")[0]
            return root, inp
        elif "ERR_SEG" in inp.name:
            # 01_ERR_SEG --> 01, 01_GT/TRA. We know that the data is in CTC folder format
            num = inp.name.split("_")[0]
            return inp.parent / num, inp.parent / f"{num}_GT" / "TRA"
        else:
            ctc_tra = Path(f"{inp}_GT") / "TRA"
            tra = ctc_tra if ctc_tra.exists() else inp / "TRA"
            # 01 --> 01, 01_GT/TRA or 01/TRA
            return inp, tra
    
    @staticmethod
    def _guess_img_folder(root: Path):
        """Guesses the image folder corresponding to a root."""
        if (root / "img").exists():
            return root / "img"
        else:
            return root

    @staticmethod
    def _guess_mask_folder(root: Path, gt_tra: Path):
        """Guesses the mask folder corresponding to a root.

        In CTC format, we use silver truth segmentation masks.
        """
        f = None
        # first try CTC format
        if gt_tra.parent.name.endswith("_GT"):
            # We use the silver truth segmentation masks
            f = root / str(gt_tra.parent.name).replace("_GT", "_ST") / "SEG"
        # try our simpler 'img' format
        if f is None or not f.exists():
            f = gt_tra
        if not f.exists():
            raise ValueError(f"Could not find mask folder for {root}")
        return f

    @classmethod
    def _guess_det_folder(cls, root: Path, suffix: str):
        """Checks for the annoying CTC format with dataset numbering as part of folder names."""
        guesses = (
            (root / suffix),
            Path(f"{root}_{suffix}"),
            Path(f"{root}_GT") / suffix,
        )
        for path in guesses:
            if path.exists():
                return path

        logger.warning(f"Skipping non-existing detection folder {root / suffix}")
        return None

    def __len__(self):
        return len(self.windows)
    
    def _load_gt(self):
        logger.info("Loading ground truth")
        self.start_frame = int(
            len(list(self.gt_mask_folder.glob("*.tif"))) * self.slice_pct[0]
        )
        self.end_frame = int(
            len(list(self.gt_mask_folder.glob("*.tif"))) * self.slice_pct[1]
        )

        masks = self._load_tiffs(self.gt_mask_folder, dtype=np.int32)
        masks = self._correct_gt_with_st(self.gt_mask_folder, masks, dtype=np.int32)

        if self.use_gt:
            track_df = self._load_tracklet_links(self.gt_tra_folder)
            track_df = _filter_track_df(
                track_df, self.start_frame, self.end_frame, self.downscale_temporal
            )
        else:
            # create dummy track dataframe
            logger.info("Using dummy track dataframe")
            track_df = self._build_tracklets_without_gt(masks)

        _check_ctc(track_df, _get_node_attributes(masks), masks)

        # Build ground truth lineage graph
        self.gt_labels, self.gt_timepoints, self.gt_graph = _ctc_lineages(
            track_df, masks
        )

        return masks, track_df

    def _correct_gt_with_st(
        self, folder: Path, x: np.ndarray, dtype: str | None = None
    ):
        if str(folder).endswith("_GT/TRA"):
            st_path = (
                tuple(folder.parents)[1]
                / folder.parent.stem.replace("_GT", "_ST")
                / "SEG"
            )
            if not st_path.exists():
                logger.debug("No _ST folder found, skipping correction")
            else:
                logger.info(f"ST MASK:\t\t{st_path} for correcting with ST masks")
                st_masks = self._load_tiffs(st_path, dtype)
                x = np.maximum(x, st_masks)

        return x

    def _load_tiffs(self, folder: Path, dtype=None):
        assert isinstance(self.downscale_temporal, int)
        logger.debug(f"Loading tiffs from {folder} as {dtype}")
        logger.debug(
            f"Temporal downscaling of {folder.name} by {self.downscale_temporal}"
        )
        x = np.stack(
            [
                tifffile.imread(f).astype(dtype)
                for f in tqdm(
                    sorted(folder.glob("*.tif"))[
                        self.start_frame : self.end_frame : self.downscale_temporal
                    ],
                    leave=False,
                    desc=f"Loading [{self.start_frame}:{self.end_frame}]",
                )
            ]
        )

        # T, (Z), Y, X
        assert isinstance(self.downscale_spatial, int)
        if self.downscale_spatial > 1 or self.downscale_temporal > 1:
            # TODO make safe for label arrays
            logger.debug(
                f"Spatial downscaling of {folder.name} by {self.downscale_spatial}"
            )
            slices = (
                slice(None),
                *tuple(
                    slice(None, None, self.downscale_spatial) for _ in range(x.ndim - 1)
                ),
            )
            x = x[slices]

        logger.debug(f"Loaded array of shape {x.shape} from {folder}")
        return x

    def _masks2properties(self, masks, return_props_by_time=False):
        """Turn label masks into lists of properties, sorted (ascending) by time and label id.

        Args:
            masks (np.ndarray): T, (Z), H, W

        Returns:
            labels: List of labels
            ts: List of timepoints
            coords: List of coordinates
        """
        return masks2properties(self.imgs, masks, return_props_by_time=return_props_by_time)

    def _load_tracklet_links(self, folder: Path) -> pd.DataFrame:
        df = pd.read_csv(
            folder / "man_track.txt",
            delimiter=" ",
            names=["label", "t1", "t2", "parent"],
            dtype=int,
        )
        n_dets = (df.t2 - df.t1 + 1).sum()
        logger.debug(f"{folder} has {n_dets} detections")

        n_divs = (df[df.parent != 0]["parent"].value_counts() == 2).sum()
        logger.debug(f"{folder} has {n_divs} divisions")
        return df

    def _build_tracklets_without_gt(self, masks):
        """Create a dataframe with tracklets from masks."""
        rows = []
        for t, m in enumerate(masks):
            for c in np.unique(m[m > 0]):
                rows.append([c, t, t, 0])
        df = pd.DataFrame(rows, columns=["label", "t1", "t2", "parent"])
        return df

    def _check_dimensions(self, x: np.ndarray):
        if self.ndim == 2 and not x.ndim == 3:
            raise ValueError(f"Expected 2D data, got {x.ndim - 1}D data")
        elif self.ndim == 3:
            # if ndim=3 and data is two dimensional, it will be cast to 3D
            if x.ndim == 3:
                x = np.expand_dims(x, axis=1)
            elif x.ndim == 4:
                pass
            else:
                raise ValueError(f"Expected 3D data, got {x.ndim - 1}D data")
        return x

    def _prepare_masks_and_imgs(self, return_orig_imgs=False):
        # Load ground truth
        self.gt_masks, self.gt_track_df = self._load_gt()
        self.gt_masks = self._check_dimensions(self.gt_masks)

        # Load images
        if self.img_folder is None:
            if self.gt_masks is not None:
                self.imgs = np.zeros_like(self.gt_masks)
            else:
                raise NotImplementedError("No images and no GT masks")
        else:
            logger.info("Loading images")
            imgs = self._load_tiffs(self.img_folder, dtype=np.float32)
            self.imgs = np.stack(
                [normalize(_x) for _x in tqdm(imgs, desc="Normalizing", leave=False)]
            )
            self.imgs = self._check_dimensions(self.imgs)
            if self.compress:
                # prepare images to be compressed later (e.g. removing non masked parts for regionprops features)
                self.imgs = np.stack(
                    [
                        _compress_img_mask_preproc(im, mask, self.features)
                        for im, mask in zip(self.imgs, self.gt_masks)
                    ]
                )
                if np.any(np.isnan(self.imgs)):
                    raise ValueError("Compressed images contain NaN values")

        assert len(self.gt_masks) == len(self.imgs)
        if return_orig_imgs:
            return imgs
    
    def _load(self):
        # # Load ground truth
        # logger.info("Loading ground truth")
        # self.gt_masks, self.gt_track_df = self._load_gt()
        # self.gt_masks = self._check_dimensions(self.gt_masks)

        # # Load images
        # if self.img_folder is None:
        #     self.imgs = np.zeros_like(self.gt_masks)
        # else:
        #     logger.info("Loading images")
        #     imgs = self._load_tiffs(self.img_folder, dtype=np.float32)
        #     self.imgs = np.stack(
        #         [normalize(_x) for _x in tqdm(imgs, desc="Normalizing", leave=False)]
        #     )
        #     self.imgs = self._check_dimensions(self.imgs)
        #     if self.compress:
        #         # prepare images to be compressed later (e.g. removing non masked parts for regionprops features)
        #         self.imgs = np.stack(
        #             [
        #                 _compress_img_mask_preproc(im, mask, self.features)
        #                 for im, mask in zip(self.imgs, self.gt_masks)
        #             ]
        #         )
        self._prepare_masks_and_imgs()

        # Load each of the detection folders and create data samples with a sliding window
        windows = []
        self.properties_by_time = dict()
        self.det_masks = dict()
        for _f in self.detection_folders:
            det_folder = self.root / _f

            if det_folder == self.gt_mask_folder:
                det_masks = self.gt_masks
                logger.info("DET MASK:\tUsing GT masks")
                (
                    det_labels,
                    det_ts,
                    det_coords,
                    det_properties_by_time,
                ) = self._masks2properties(det_masks, return_props_by_time=True)

                det_gt_matching = {
                    t: {_l: _l for _l in det_properties_by_time[t]["labels"]}
                    for t in range(len(det_masks))
                }
            else:
                det_folder = self._guess_det_folder(root=self.root, suffix=_f)
                if det_folder is None:
                    continue

                logger.info(f"DET MASK:\t{det_folder}")
                det_masks = self._load_tiffs(det_folder, dtype=np.int32)
                det_masks = self._correct_gt_with_st(
                    det_folder, det_masks, dtype=np.int32
                )
                det_masks = self._check_dimensions(det_masks)
                (
                    det_labels,
                    det_ts,
                    det_coords,
                    det_properties_by_time,
                ) = self._masks2properties(det_masks, return_props_by_time=True)

                # FIXME matching can be slow for big images
                # raise NotImplementedError("Matching not implemented for 3d version")
                det_gt_matching = {
                    t: {
                        _d: _gt
                        for _gt, _d in matching(
                            self.gt_masks[t],
                            det_masks[t],
                            threshold=0.3,
                            max_distance=16,
                        )
                    }
                    for t in tqdm(range(len(det_masks)), leave=False, desc="Matching")
                }

            self.properties_by_time[_f] = det_properties_by_time
            self.det_masks[_f] = det_masks
            
            # Build windows
            _w = self._build_windows(
                det_folder,
                det_masks,
                det_labels,
                det_ts,
                det_coords,
                det_gt_matching,
            )

            windows.extend(_w)

        return windows

    def _build_windows(
        self,
        det_folder,
        det_masks,
        labels,
        ts,
        coords,
        matching,
    ):
        """_summary_.

        Args:
            det_folder (_type_): _description_
            det_masks (_type_): _description_
            labels (_type_): _description_
            ts (_type_): _description_
            coords (_type_): _description_
            matching (_type_): _description_

        Raises:
            ValueError: _description_
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        window_size = self.window_size
        windows = []

        # Creates the data samples with a sliding window
        masks = self.gt_masks
        for t1, t2 in tqdm(
            zip(range(0, len(masks)), range(window_size, len(masks) + 1)),
            total=len(masks) - window_size + 1,
            leave=False,
            desc="Building windows",
        ):
            idx = (ts >= t1) & (ts < t2)
            _ts = ts[idx]
            _labels = labels[idx]
            _coords = coords[idx]

            # Use GT
            # _labels = self.gt_labels[idx]
            # _ts = self.gt_timepoints[idx]

            if len(_labels) == 0:
                # raise ValueError(f"No detections in sample {det_folder}:{t1}")
                A = np.zeros((0, 0), dtype=bool)
                _coords = np.zeros((0, masks.ndim - 1), dtype=int)
            else:
                if len(np.unique(_ts)) == 1:
                    logger.debug(
                        "Only detections from a single timepoint in sample"
                        f" {det_folder}:{t1}"
                    )

                # build matrix from incomplete labels, but full lineage graph. If a label is missing, I should skip over it.
                A = _ctc_assoc_matrix(
                    _labels,
                    _ts,
                    self.gt_graph,
                    matching,
                )

            if self.sanity_dist:
                # # Sanity check: Can the model learn the euclidian distances?
                # c = coords - coords.mean(axis=0, keepdims=True)
                # c /= c.std(axis=0, keepdims=True)
                # A = np.einsum('id,jd',c,c)
                # A = 1 / (1 + np.exp(-A))
                A = np.exp(-0.01 * cdist(_coords, _coords))

            w = dict(
                coords=_coords,
                # TODO imgs and masks are unaltered here
                t1=t1,
                img=self.imgs[t1:t2],
                mask=det_masks[t1:t2],
                assoc_matrix=A,
                labels=_labels,
                timepoints=_ts,
            )

            windows.append(w)

        logger.debug(f"Built {len(windows)} track windows from {det_folder}.\n")
        return windows
    
    def _apply_transform_and_check(self, img, labels, mask, coords, timepoints, min_time, assoc_matrix):
        (img2, mask2, coords2), idx = self.augmenter(
                    img, mask, coords, timepoints - min_time
                )
        if len(idx) > 0:
            img, mask, coords = img2, mask2, coords2
            labels = labels[idx]
            timepoints = timepoints[idx]
            assoc_matrix = assoc_matrix[idx][:, idx]
            mask = mask.astype(int)
        else:
            logger.debug(
                "Disable augmentation as no trajectories would be left"
            )
        return img, labels, mask, coords, timepoints, assoc_matrix
    
    @staticmethod
    def decompress(data):
        if isinstance(data, _CompressedArray):
            return data.decompress()
        return data
                    
    def __getitem__(self, n: int, return_dense=None):
        # if not set, use default
        if self.features == "wrfeat" or self.features == "pretrained_feats":
            return self._getitem_wrfeat(n, return_dense)

        if return_dense is None:
            return_dense = self.return_dense

        track = self.windows[n]
        coords = track["coords"]
        assoc_matrix = track["assoc_matrix"]
        labels = track["labels"]
        img = track["img"]
        mask = track["mask"]
        timepoints = track["timepoints"]
        min_time = track["t1"]

        # if isinstance(mask, _CompressedArray):
        #     mask = mask.decompress()
        # if isinstance(img, _CompressedArray):
        #     img = img.decompress()
        # if isinstance(assoc_matrix, _CompressedArray):
        #     assoc_matrix = assoc_matrix.decompress()
        mask = CTCData.decompress(mask)
        img = CTCData.decompress(img)
        assoc_matrix = CTCData.decompress(assoc_matrix)

        # cropping
        if self.cropper is not None:
            (img2, mask2, coords2), idx = self.cropper(img, mask, coords)
            cropped_timepoints = timepoints[idx]

            # at least one detection in each timepoint to accept the crop
            if len(np.unique(cropped_timepoints)) == self.window_size:
                # at least two total detections to accept the crop
                # if len(idx) >= 2:
                img, mask, coords = img2, mask2, coords2
                labels = labels[idx]
                timepoints = timepoints[idx]
                assoc_matrix = assoc_matrix[idx][:, idx]
            else:
                logger.debug("disable cropping as no trajectories would be left")

        if self.features == "none":
            if self.augmenter is not None:
                coords = self.augmenter(coords)
            # Empty features
            features = np.zeros((len(coords), 0))

        elif self.features in ("regionprops", "regionprops2"):
            if self.augmenter is not None:
                img, labels, mask, coords, timepoints, assoc_matrix = self._apply_transform_and_check(
                    img, labels, mask, coords, timepoints, min_time, assoc_matrix
                )

            features = tuple(
                extract_features_regionprops(
                    m, im, labels[timepoints == i + min_time], properties=self.features
                )
                for i, (m, im) in enumerate(zip(mask, img))
            )
            features = np.concatenate(features, axis=0)
            # features = np.zeros((len(coords), self.feat_dim))
        elif self.features == "patch":
            if self.augmenter is not None:
                img, labels, mask, coords, timepoints, assoc_matrix = self._apply_transform_and_check(
                    img, labels, mask, coords, timepoints, min_time, assoc_matrix
                )
            features = tuple(
                extract_features_patch(
                    m,
                    im,
                    coords[timepoints == min_time + i],
                    labels[timepoints == min_time + i],
                )
                for i, (m, im) in enumerate(zip(mask, img))
            )
            features = np.concatenate(features, axis=0)
        elif self.features == "patch_regionprops":
            if self.augmenter is not None:
                img, labels, mask, coords, timepoints, assoc_matrix = self._apply_transform_and_check(
                    img, labels, mask, coords, timepoints, min_time, assoc_matrix
                )
            features1 = tuple(
                extract_features_patch(
                    m,
                    im,
                    coords[timepoints == min_time + i],
                    labels[timepoints == min_time + i],
                )
                for i, (m, im) in enumerate(zip(mask, img))
            )
            features2 = tuple(
                extract_features_regionprops(
                    m,
                    im,
                    labels[timepoints == i + min_time],
                    properties=self.features,
                )
                for i, (m, im) in enumerate(zip(mask, img))
            )

            features = tuple(
                np.concatenate((f1, f2), axis=-1)
                for f1, f2 in zip(features1, features2)
            )

            features = np.concatenate(features, axis=0)
        # MOVED as WRFeat. See wrfeat.WRPretrainedFeatures
        # elif self.features == "pretrained_feats":
        #     if self.augmenter is not None:
        #         img, labels, mask, coords, timepoints, assoc_matrix = self._apply_transform_and_check(
        #             img, labels, mask, coords, timepoints, min_time, assoc_matrix
        #         )
        #     ts, n_obj = np.unique(timepoints, return_counts=True)
        #     features = torch.zeros((n_obj.sum(), self.feat_dim))
        #     offset = 0

        #     for t, count in zip(ts, n_obj):
        #         feat = self.pretrained_features[t]  # (timepoints -> (n_regions, n_features)) for a SINGLE timepoint
        #         if feat.shape[0] != count:
        #             raise ValueError(f"Feature mismatch at time {t}: expected {count}, got {feat.shape[0]}")
        #         features[offset:offset + count] = feat
        #         offset += count

        #     if features.shape[0] != len(timepoints):
        #         raise ValueError(f"Pretrained features shape mismatch: {features.shape[0]} != {len(timepoints)}")

        # remove temporal offset and add timepoints to coords
        relative_timepoints = timepoints - track["t1"]
        coords = np.concatenate((relative_timepoints[:, None], coords), axis=-1)

        if self.max_tokens and len(timepoints) > self.max_tokens:
            time_incs = np.where(timepoints - np.roll(timepoints, 1))[0]
            n_elems = time_incs[np.searchsorted(time_incs, self.max_tokens) - 1]
            timepoints = timepoints[:n_elems]
            labels = labels[:n_elems]
            coords = coords[:n_elems]
            features = features[:n_elems]
            assoc_matrix = assoc_matrix[:n_elems, :n_elems]
            logger.info(
                f"Clipped window of size {timepoints[n_elems - 1] - timepoints.min()}"
            )

        coords0 = torch.from_numpy(coords).float()
        features = torch.from_numpy(features).float() if isinstance(features, np.ndarray) else features.float()
        assoc_matrix = torch.from_numpy(assoc_matrix.copy()).float()
        labels = torch.from_numpy(labels).long()
        timepoints = torch.from_numpy(timepoints).long()

        if self.augmenter is not None:
            coords = coords0.clone()
            coords[:, 1:] += torch.randint(-1024, 1024, (1, self.ndim))
        else:
            coords = coords0.clone()
        res = dict(
            features=features,
            coords0=coords0,
            coords=coords,
            assoc_matrix=assoc_matrix,
            timepoints=timepoints,
            labels=labels,
        )

        if return_dense:
            if all([x is not None for x in img]):
                img = torch.from_numpy(img).float()
                res["img"] = img

            mask = torch.from_numpy(mask.astype(int)).long()
            res["mask"] = mask
        return res

    # wrfeat functions...
    # TODO: refactor this as a subclass or make everything a class factory. *very* hacky this way
    # -> updated _setup_features_augs_wrfeat to use a factory instead

    def _setup_features_augs_wrfeat(
        self
    ):
        augmenter = wrfeat.AugmentationFactory.create_augmentation_pipeline(self.augment_level)
        cropper = wrfeat.AugmentationFactory.create_cropper(self.crop_size, self.ndim) if self.crop_size is not None else None

        return augmenter, cropper

    def _load_wrfeat(self):
        # # Load ground truth
        # self.gt_masks, self.gt_track_df = self._load_gt()
        # self.gt_masks = self._check_dimensions(self.gt_masks)

        # # Load images
        # if self.img_folder is None:
        #     if self.gt_masks is not None:
        #         self.imgs = np.zeros_like(self.gt_masks)
        #     else:
        #         raise NotImplementedError("No images and no GT masks")
        # else:
        #     logger.info("Loading images")
        #     imgs = self._load_tiffs(self.img_folder, dtype=np.float32)
        #     self.imgs = np.stack(
        #         [normalize(_x) for _x in tqdm(imgs, desc="Normalizing", leave=False)]
        #     )
        #     self.imgs = self._check_dimensions(self.imgs)
        #     if self.compress:
        #         # prepare images to be compressed later (e.g. removing non masked parts for regionprops features)
        #         self.imgs = np.stack(
        #             [
        #                 _compress_img_mask_preproc(im, mask, self.features)
        #                 for im, mask in zip(self.imgs, self.gt_masks)
        #             ]
        #         )
        #         if np.any(np.isnan(self.imgs)):
        #             raise ValueError("Compressed images contain NaN values")

        # assert len(self.gt_masks) == len(self.imgs)
        imgs = self._prepare_masks_and_imgs(return_orig_imgs=True)

        # Load each of the detection folders and create data samples with a sliding window
        windows = []
        self.properties_by_time = dict()
        self.det_masks = dict()
        logger.info("Loading detections")
        for _f in self.detection_folders:
            det_folder = self.root / _f

            if det_folder == self.gt_mask_folder:
                det_masks = self.gt_masks
                logger.info("DET MASK:\tUsing GT masks")
                # identity matching
                det_gt_matching = {
                    t: {_l: _l for _l in set(np.unique(d)) - {0}}
                    for t, d in enumerate(det_masks)
                }
            else:
                det_folder = self._guess_det_folder(root=self.root, suffix=_f)
                if det_folder is None:
                    continue
                logger.info(f"DET MASK (guessed):\t{det_folder}")
                det_masks = self._load_tiffs(det_folder, dtype=np.int32)
                det_masks = self._correct_gt_with_st(
                    det_folder, det_masks, dtype=np.int32
                )
                det_masks = self._check_dimensions(det_masks)
                # FIXME matching can be slow for big images
                # raise NotImplementedError("Matching not implemented for 3d version")
                det_gt_matching = {
                    t: {
                        _d: _gt
                        for _gt, _d in matching(
                            self.gt_masks[t],
                            det_masks[t],
                            threshold=0.3,
                            max_distance=16,
                        )
                    }
                    for t in tqdm(range(len(det_masks)), leave=False, desc="Matching")
                }

            self.det_masks[_f] = det_masks

            # build features
            if self.features == "pretrained_feats":
                self._setup_pretrained_feature_extractor()
                if np.all(self.imgs == 0):
                    raise ValueError("Images are empty. Images must be provided when using pretrained features")
                self.feature_extractor.precompute_image_embeddings(imgs)  # use NON_NORMALIZED images for pretrained features
                # normalization is performed in the feature extractor
                features = [
                    wrfeat.WRPretrainedFeatures.from_mask_img(
                        img=img[np.newaxis], 
                        mask=mask[np.newaxis], 
                        feature_extractor=self.feature_extractor, 
                        t_start=t, 
                        additional_properties=self.pretrained_config.additional_features
                    )
                    for t, (mask, img) in enumerate(zip(det_masks, self.imgs))
                ]
                for wrf in features:
                    feats = wrf.features_stacked
                    if feats is not None and np.any(np.isnan(wrf.features_stacked)):
                        raise ValueError("NaN in features")
                if torch.cuda.is_available():
                    self.feature_extractor.embeddings = self.feature_extractor.embeddings.cpu()
                    torch.cuda.empty_cache()
            elif self.features == "wrfeat":
                features = joblib.Parallel(n_jobs=8)(
                    joblib.delayed(wrfeat.WRFeatures.from_mask_img)(
                        mask=mask[None], img=img[None], t_start=t
                    )
                    for t, (mask, img) in enumerate(zip(det_masks, self.imgs))
                )
                # features = []
                # for t, (mask, img) in enumerate(zip(det_masks, self.imgs)):
                #     wrf = wrfeat.WRFeatures.from_mask_img(
                #         mask=mask[None], img=img[None], t_start=t
                #     )
                #     features.append(wrf)

            properties_by_time = dict()
            for _t, _feats in enumerate(features):
                properties_by_time[_t] = dict(
                    coords=_feats.coords, labels=_feats.labels
                )
            self.properties_by_time[_f] = properties_by_time

            _w = self._build_windows_wrfeat(
                features,
                det_masks,
                det_gt_matching,
            )

            windows.extend(_w)

        return windows

    def _build_windows_wrfeat(
        self,
        features: Sequence[wrfeat.WRFeatures],
        det_masks: np.ndarray,
        matching: tuple[dict],
    ):
        assert len(self.imgs) == len(det_masks)

        window_size = self.window_size
        windows = []

        # Creates the data samples with a sliding window
        for t1, t2 in tqdm(
            zip(range(0, len(det_masks)), range(window_size, len(det_masks) + 1)),
            total=len(det_masks) - window_size + 1,
            leave=False,
            desc="Building windows",
        ):

            img = self.imgs[t1:t2]
            mask = det_masks[t1:t2]
            feat = wrfeat.WRFeatures.concat(features[t1:t2])

            labels = feat.labels
            timepoints = feat.timepoints
            coords = feat.coords

            if len(feat) == 0:
                A = np.zeros((0, 0), dtype=bool)
                coords = np.zeros((0, feat.ndim), dtype=int)
            else:

                # build matrix from incomplete labels, but full lineage graph. If a label is missing, I should skip over it.
                A = _ctc_assoc_matrix(
                    labels,
                    timepoints,
                    self.gt_graph,
                    matching,
                )
            w = dict(
                coords=coords,
                # TODO imgs and masks are unaltered here
                t1=t1,
                img=img,
                mask=mask,
                assoc_matrix=A,
                labels=labels,
                timepoints=timepoints,
                wrfeat=feat,
            )
            windows.append(w)

        logger.debug(f"Built {len(windows)} track windows.\n")
        return windows

    def _getitem_wrfeat(self, n: int, return_dense=None):
        # if not set, use default

        if return_dense is None:
            return_dense = self.return_dense

        track = self.windows[n]
        # coords = track["coords"]
        assoc_matrix = track["assoc_matrix"]
        labels = track["labels"]
        img = track["img"]
        mask = track["mask"]
        timepoints = track["timepoints"]
        # track["t1"]
        feat = track["wrfeat"]

        if return_dense and isinstance(mask, _CompressedArray):
            mask = mask.decompress()
        if return_dense and isinstance(img, _CompressedArray):
            img = img.decompress()
        if isinstance(assoc_matrix, _CompressedArray):
            assoc_matrix = assoc_matrix.decompress()

        # cropping
        if self.cropper is not None:
            # Use only if there is at least one timepoint per detection
            cropped_feat, cropped_idx = self.cropper(feat)
            cropped_timepoints = timepoints[cropped_idx]
            if len(np.unique(cropped_timepoints)) == self.window_size:
                idx = cropped_idx
                feat = cropped_feat
                labels = labels[idx]
                timepoints = timepoints[idx]
                assoc_matrix = assoc_matrix[idx][:, idx]
            # else:
            # logger.debug("Skipping cropping")

        if self.augmenter is not None:
            feat = self.augmenter(feat)
        
        coords0 = np.concatenate((feat.timepoints[:, None], feat.coords), axis=-1)
        coords0 = torch.from_numpy(coords0).float()
        assoc_matrix = torch.from_numpy(assoc_matrix.astype(np.float32))
        # if self.pca_preprocessor is not None:
        # features = self.pca_preprocessor.transform(feat.features_stacked)
        # else:
        features = feat.features_stacked
        if features is not None:
            features = torch.from_numpy(features).float()
        
        labels = torch.from_numpy(feat.labels).long()
        timepoints = torch.from_numpy(feat.timepoints).long()
        
        pretrained_features = feat.pretrained_feats
        if pretrained_features is not None:
            pretrained_features = torch.from_numpy(pretrained_features).float()
    
        if self.max_tokens and len(timepoints) > self.max_tokens:
            time_incs = np.where(timepoints - np.roll(timepoints, 1))[0]
            n_elems = time_incs[np.searchsorted(time_incs, self.max_tokens) - 1]
            timepoints = timepoints[:n_elems]
            labels = labels[:n_elems]
            coords0 = coords0[:n_elems]
            if features is not None:
                features = features[:n_elems]
            if pretrained_features is not None:
                pretrained_features = pretrained_features[:n_elems]
            assoc_matrix = assoc_matrix[:n_elems, :n_elems]
            logger.debug(
                f"Clipped window of size {timepoints[n_elems - 1] - timepoints.min()}"
            )

        if self.augmenter is not None:
            coords = coords0.clone()
            coords[:, 1:] += torch.randint(0, 512, (1, self.ndim))
        else:
            coords = coords0.clone()
        
        if self.features == "pretrained_feats" and self.rotate_feats:
            if isinstance(img, _CompressedArray):
                image_shape = img._shape
            else:
                image_shape = img.shape
            # logger.debug(f"Rotating pretrained features with shape {pretrained_features.shape} for image shape {image_shape}")
            pretrained_features = CTCData.rotate_features(
                pretrained_features, coords, image_shape,
                n_rot_dims=self.pretrained_feat_dim,
            )

        res = dict(
            features=features,
            pretrained_features=pretrained_features,
            coords0=coords0,
            coords=coords,
            assoc_matrix=assoc_matrix,
            timepoints=timepoints,
            labels=labels,
        )
        
        if return_dense:
            if all([x is not None for x in img]):
                img = torch.from_numpy(img).float()
                res["img"] = img

            mask = torch.from_numpy(mask.astype(int)).long()
            res["mask"] = mask
        
        if features is not None:
            if torch.any(torch.isnan(features)):
                raise ValueError("NaN in features")
            elif torch.any(torch.all(features == 0, dim=-1)):
                raise ValueError("Empty features")
            
        if pretrained_features is not None:
            if torch.any(torch.isnan(pretrained_features)):
                raise ValueError("NaN in pretrained features")
            elif torch.any(torch.all(pretrained_features == 0, dim=-1)):
                raise ValueError("Empty pretrained features")
        
        return res
    
    def _get_pretrained_features_save_path(self):
        if self.pretrained_config is not None:
            img_folder_name = "_".join(self.root.parts[-3:]) if len(self.root.parts) >= 3 else "_".join(self.root.parts)
            img_folder_name = str(img_folder_name).replace(".", "").replace("/", "_").replace("\\", "_").replace(" ", "_")
            return self.pretrained_config.save_path / f"embeddings/{img_folder_name}"

    def _setup_pretrained_feature_extractor(self):
        if self.ndim == 3:
            raise ValueError("Pretrained model feature extraction is not implemented for 3D data")
        img_shape = self.imgs.shape[-2:]  # initial guess, replaced later if shape changes
        from trackastra.data.pretrained_features import (
            FeatureExtractor,
        )
        self.feature_extractor_save_path = self._get_pretrained_features_save_path()
        # self.feature_extractor = FeatureExtractor.from_model_name(
        #     self.pretrained_config.model_name,
        #     img_shape, 
        #     save_path=self.feature_extractor_save_path,
        #     mode=self.pretrained_config.mode,
        #     device=self.pretrained_config.device,
        #     additional_features=self.pretrained_config.additional_features,
        # )
        self.feature_extractor = FeatureExtractor.from_config(
            self.pretrained_config,
            image_shape=img_shape,
            save_path=self.feature_extractor_save_path,
        )
        self.feature_extractor_input_size = self.feature_extractor.input_size
        
    def _compute_pretrained_model_features(self):
        if self.pretrained_config.model_name is None:
            logger.warning("No pretrained model set, feature extraction not run")
            return

        try:
            self.feature_extractor.input_mul = self._pretrained_model_input_size_factor
        except Exception:
            logger.warning(f"Cannot change input size for pretrained model: {self.pretrained_config.model_name}")
        self.pretrained_features = self.feature_extractor.precompute_region_embeddings(self.imgs)
        # dict(n_frames) : torch.Tensor(n_regions_in_frame, n_features)
        self.feature_extractor = None

    def compute_pretrained_features(self, input_size_factor: int | None = None, model: PretrainedBackboneType = None, mode: PretrainedFeatsExtractionMode = "nearest_patch"):
        """Compute pretrained features for the dataset, if the model. input size factor or mode was changed.
        
        Args:
            input_size_factor (int, optional): The input size factor for the pretrained model. Defaults to None.
            model (PretrainedBackboneType, optional): The pretrained model to use. Defaults to None.
            mode (PretrainedFeatsExtractionMode, optional): The mode to use for feature extraction. Defaults to "nearest_patch".
        """
        if input_size_factor is not None:
            self._pretrained_model_input_size_factor = input_size_factor
            logger.debug(f"Setting input size factor to {input_size_factor}")
        if model is not None:
            self.pretrained_config.model_name = model
            logger.debug(f"Setting pretrained model to {model}")
        if mode is not None:
            self.pretrained_config.mode = mode
            logger.debug(f"Setting feature extraction mode to {mode}")
        if input_size_factor is None and model is None and mode is None:
            logger.warning("No changes in input size factor, model or mode. Skipping feature extraction.")
            return
        else:
            self._compute_pretrained_model_features()
        
    @staticmethod
    def rotate_features(
        features: torch.Tensor,
        coords: torch.Tensor,
        image_shape: tuple,
        n_rot_dims: int | None = None,
        skip_first: int = 0,
    ) -> torch.Tensor:
        """Applies a RoPE-style rotation to each feature vector based on the object's centroid.

        Args:
            features: (n_objects, hidden_state_size) tensor of features.
            coords: (n_objects, 2) tensor of coordinates.
            image_shape: (time, height, width) shape of the input image.
            n_rot_dims: Number of feature dimensions to apply rotation to (must be even). If None, rotate all.
            skip_first: Number of dimensions to skip at the beginning of the feature vector. No effect if 0.

        Returns:
            Rotated features: (n_objects, hidden_state_size)
        """
        import math
        N, D = features.shape
        assert skip_first < n_rot_dims, "skip_first must be less than n_rot_dims."
        if n_rot_dims is None:
            n_rot_dims = D
        if skip_first != 0:
            n_rot_dims = n_rot_dims - skip_first
        assert n_rot_dims % 2 == 0, "n_rot_dims must be even for RoPE."
        assert n_rot_dims <= D, "n_rot_dims cannot exceed feature dimension."

        # Normalize x and y to [0, 1]
        H, W = image_shape[-2], image_shape[-1]
        x_norm = coords[:, 1] / H
        y_norm = coords[:, 2] / W
        # Compute two angles for x and y
        angle_x = 2 * math.pi * x_norm
        angle_y = 2 * math.pi * y_norm
        # Interleave angles for each feature pair
        angles = torch.stack([angle_x, angle_y], dim=1).repeat(1, n_rot_dims // 2)
        angles = angles.view(N, n_rot_dims)
        # Prepare cos/sin
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        # Interleave features for rotation
        # try:
        features_rot = features[:, skip_first:n_rot_dims + skip_first].view(N, -1, 2)
        # except Exception:
        #     breakpoint()
        x_feat, y_feat = features_rot[..., 0], features_rot[..., 1]
        x_rot = x_feat * cos[:, ::2] - y_feat * sin[:, ::2]
        y_rot = x_feat * sin[:, ::2] + y_feat * cos[:, ::2]
        rotated_part = torch.stack([x_rot, y_rot], dim=-1).reshape(N, n_rot_dims)
        if n_rot_dims < D:
            rotated = torch.cat([rotated_part, features[:, n_rot_dims:]], dim=1)
        else:
            rotated = rotated_part
        return rotated  # (n_objects, d)


class CTCDataAugPretrainedFeats(CTCData):
    """CTCData with pretrained features."""

    def __init__(
        self,
        pretrained_n_augmentations: int = 3,
        n_aug_workers: int = 8,
        force_recompute=False,
        aug_pipeline: PretrainedAugmentations = None,
        load_from_disk: bool = False,
        *args, 
        **kwargs
        ):
        """Args:
        root (str):
            Folder containing the CTC TRA folder.
        ndim (int):
            Number of dimensions of the data. Defaults to 2d
            (if ndim=3 and data is two dimensional, it will be cast to 3D)
        detection_folders:
            List of relative paths to folder with detections.
            Defaults to ["TRA"], which uses the ground truth detections.
        window_size (int):
            Window size for transformer.
        slice_pct (tuple):
            Slice the dataset by percentages (from, to).
        augment (int):
            if 0, no data augmentation. if > 0, defines level of data augmentation.
        features (str):
            Types of features to use.
        sanity_dist (bool):
            Use euclidian distance instead of the association matrix as a target.
        crop_size (tuple):
            Size of the crops to use for augmentation. If None, no cropping is used.
        return_dense (bool):
            Return dense masks and images in the data samples.
        compress (bool):
            Compress elements/remove img if not needed to save memory for large datasets
        pretrained_backbone_config (PretrainedFeatureExtractorConfig):
            Configuration for the pretrained backbone.
            If mode is set to "pretrained_feats", this configuration is used to extract features.
            Ignored otherwise.
        pretrained_n_augmentations (int):
            How many augmented versions of the pretrained model embeddings to create.
        n_aug_workers (int):
            Number of workers to use for offline augmentation.
        load_from_disk (bool):
            If True, the offline augmented windows are saved to disk and sampled from there.
            If False, all windows are loaded into RAM and sampled from there.
        force_recompute (bool):
            If False, previously computed offline augmentations are loaded if available.
        # pca_preprocessor (EmbeddingsPCACompression):
        #     PCA preprocessor for the pretrained features.
        #     If mode is set to "pretrained_feats", this is used to reduce the dimensionality of the features.
        #     Ignored otherwise.
        """
        features = kwargs.get("features", None)
        
        if features is not None and not features == "pretrained_feats_aug":
            raise ValueError("This class should only be used with pretrained_feats_aug features")

        self.n_augs = pretrained_n_augmentations
        self.n_aug_workers = n_aug_workers
        self.force_recompute = force_recompute
        self.load_from_disk = load_from_disk
        
        from trackastra.data.pretrained_augmentations import (
            PretrainedMovementAugmentations,
        )
        self.pretrained_feats_augmenter = PretrainedMovementAugmentations(rng_seed=42) if aug_pipeline is None else aug_pipeline
        if not isinstance(self.pretrained_feats_augmenter, PretrainedAugmentations):
            raise ValueError(
                f"Augmentation pipeline must be of type PretrainedAugmentations, got {type(self.pretrained_feats_augmenter)}"
            )
        logger.debug(self.pretrained_feats_augmenter)
        self.augmented_feature_extractor = None
        self.augmented_image_shapes = None  # used to store the augmented image shapes, used to rotate features
        self.save_windows = True
        
        self._aug_embeds_file = None  # stores the augmented per-object embeddings
        self.delete_augs_after_loading = False
        # self.window_save_path = None
        self._last_selected = None
        self._rng = np.random.default_rng()
        self._len = None
        self._debug = False
        
        super().__init__(*args, **kwargs, load_immediately=False)
        
        if self.load_from_disk:
            self.window_save_path = self._get_pretrained_features_save_path() / "windows"
            self.window_save_path.mkdir(parents=True, exist_ok=True)
            self.window_save_path = self.window_save_path / f"{self.config_hash}.zarr"
            logger.debug(f"Windows will be saved to {self.window_save_path}")
        else:
            self.window_save_path = None
            logger.debug("Windows will be loaded into RAM")
            
        if kwargs.get("load_immediately", True):  # hook to delay loading if needed
            self.start_loading()
            start = default_timer()
        else:
            start = None
        
        logger.debug("Loading finished, clearing feature extractors...")
        
        # Clear pre-trained model
        self.augmented_feature_extractor = None
        # Clear windos as they are loaded from disk when __getitem__ is called
        if self.load_from_disk:
            self._get_ndim_and_nobj(start, self.windows)
            self.windows = None
            # Clear intermediate data
            if self.delete_augs_after_loading and self._aug_embeds_file.exists():
                try:
                    self._aug_embeds_file.close()
                except Exception as e:
                    logger.warning(f"Could not close HDF5 file: {e}")
                try:
                    self._aug_embeds_file.unlink()
                    self._aug_embeds_file = None
                except Exception as e:
                    logger.warning(f"Could not delete file {self._aug_embeds_file}: {e}")
            logger.info("Feature extractors cleared.")
        else:
            self._get_ndim_and_nobj(start, self.windows)
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    @property
    def config(self):
        cfg = super().config
        cfg["pretrained_n_augmentations"] = self.n_augs
        cfg["pretrained_augmentations"] = self.pretrained_feats_augmenter.get_signature()
        return cfg

    @property
    def feat_dim(self):
        return self.pretrained_config.feat_dim
    
    def _init_features(self):
        self.windows = self._load()
        if self.load_from_disk:
            self._save_windows()
        else:
            self._get_ndim_and_nobj(None, self.windows)
            
    def _setup_features_augs(
        self
    ):
        logger.debug(f"Creating augmentations with level {self.augment_level}")
        augmenter = wrfeat.AugmentationFactory.create_augmentation_pipeline(self.augment_level, return_type=wrfeat.WRAugPretrainedFeatures)
        cropper = wrfeat.AugmentationFactory.create_cropper(self.crop_size, self.ndim, return_type=wrfeat.WRAugPretrainedFeatures) if self.crop_size is not None else None

        return augmenter, cropper
        
    def _get_ndim_and_nobj(self, start=None, windows=None):
        if windows is not None:
            self.ndim = windows[0]["coords"][0][0].shape[0]
            self.n_objects = tuple(len(t["coords"][0]) for t in windows)
        if self.save_windows and self.load_from_disk:
            return
        if len(self.windows) > 0:
            self.ndim = self.windows[0]["coords"][0].shape[1]
            self.n_objects = tuple(len(t["coords"][0]) for t in self.windows)
            if start is None:
                logger.info(
                    f"Found {np.sum(self.n_objects)} objects in {len(self.windows)} track"
                    f" windows from {self.root}\n"
                )
            else:
                logger.info(
                    f"Found {np.sum(self.n_objects)} objects in {len(self.windows)} track"
                    f" windows from {self.root} ({default_timer() - start:.1f}s)\n"
                )
        else:
            self.n_objects = 0
            logger.warning(f"Could not load any tracks from {self.root}")
    
    @classmethod
    def from_arrays(cls, imgs: np.ndarray, masks: np.ndarray, train_args: dict):
        raise NotImplementedError()
        # self = cls(**train_args)
        # start = default_timer()
        
        # self.windows = self._load()
        # self.n_divs = self._get_ndivs()
        
        # if len(self.windows) > 0:
        #     self.ndim = self.windows[0]["coords"][0].shape[1]
        #     self.n_objects = tuple(len(t["coords"][0]) for t in self.windows)
        #     logger.info(
        #         f"Found {np.sum(self.n_objects)} objects in {len(self.windows)} track"
        #         f" windows from {self.root} ({default_timer() - start:.1f}s)\n"
        #     )
        # else:
        #     self.n_objects = 0
        #     logger.warning(f"Could not load any tracks from {self.root}")

        # if self.compress:
        #     self._compress_data() 
    
    def _load(self):        
        all_windows = []
        imgs = self._prepare_masks_and_imgs(return_orig_imgs=True)
        
        # self.properties_by_time = dict()
        self.det_masks = dict()
        logger.info("Loading detections")
        if len(self.detection_folders) > 1:
            raise NotImplementedError("Pretrained aug features with several folders is not supported yet")
        
        if self._load_windows() is not None:
            return self.windows
        
        for _f in self.detection_folders:
            det_folder = self.root / _f

            if det_folder == self.gt_mask_folder:
                det_masks = self.gt_masks
                logger.info("DET MASK:\tUsing GT masks")
                # identity matching
                (
                    det_labels,
                    det_ts,
                    _,
                ) = self._masks2properties(det_masks)
                
                det_gt_matching = {
                    t: {_l: _l for _l in set(np.unique(d)) - {0}}
                    for t, d in enumerate(det_masks)
                }
            else:
                det_folder = self._guess_det_folder(root=self.root, suffix=_f)
                if det_folder is None:
                    continue
                logger.info(f"DET MASK (guessed):\t{det_folder}")
                det_masks = self._load_tiffs(det_folder, dtype=np.int32)
                det_masks = self._correct_gt_with_st(
                    det_folder, det_masks, dtype=np.int32
                )
                det_masks = self._check_dimensions(det_masks)
                (
                    det_labels,
                    det_ts,
                    _,
                ) = self._masks2properties(det_masks)
                # FIXME matching can be slow for big images
                # raise NotImplementedError("Matching not implemented for 3d version")
                det_gt_matching = {
                    t: {
                        _d: _gt
                        for _gt, _d in matching(
                            self.gt_masks[t],
                            det_masks[t],
                            threshold=0.3,
                            max_distance=16,
                        )
                    }
                    for t in tqdm(range(len(det_masks)), leave=False, desc="Matching")
                }

            self.det_masks[_f] = det_masks
            # Setup feature extractor
            self._setup_pretrained_feature_extractor()
            
            # Build augmentation pipeline
            from trackastra.data.pretrained_features import FeatureExtractorAugWrapper
            self.augmented_feature_extractor = FeatureExtractorAugWrapper(
                extractor=self.feature_extractor,
                augmenter=self.pretrained_feats_augmenter,
                n_aug=self.n_augs,
                force_recompute=self.force_recompute,
            )
            self._aug_embeds_file = self.augmented_feature_extractor.get_save_path()
            
            # Compute features for all augmentations
            augmented_dict = self.augmented_feature_extractor.compute_all_features(
                images=imgs,
                masks=det_masks,
                clear_mem=not self.load_from_disk,
                n_workers=self.n_aug_workers,
            )
            self.augmented_image_shapes = self.augmented_feature_extractor.image_shape_reference
            # logger.debug(f"AUG DICT keys : {augmented_dict.keys()}")

            _w = self._build_windows(
                det_ts,
                det_labels,
                det_gt_matching,
                augmented_dict
            )
            all_windows.extend(_w)
            
        return all_windows

    def _build_windows(self, ts, labels, matching, augmented_dict):        
        windows = []
        window_size = self.window_size
        n_frames = len(np.unique(ts))
        n_entries = self.n_augs + 1
        # augmented_dict structure :
        #  - aug_id:
        #     - metadata: dict, record of the applied augmentations and other metadata
        #    - data: the data for aug_id
        #         - t: frame between 0 and n_frames
        #           - lab: label of the detection
        #               - coords: coordinates of the detections for (t, lab)
        #               - features: dict of features for (t, lab)
        #                   - feat_name_1: feature 1 for (t, lab)
        #                   - ...   
        #                   - feat_name_n: feature n for (t, lab)
                
        for t1, t2 in tqdm(
            zip(range(0, n_frames), range(window_size, n_frames + 1)),
            total=n_frames - window_size + 1,
            leave=False,
            desc="Building windows",
        ):
            idx = (ts >= t1) & (ts < t2)
            _ts = ts[idx]
            _labels = labels[idx]

            _coords = {aug_id: [] for aug_id in range(n_entries)}
            _features = {aug_id: {} for aug_id in range(n_entries)}
            present_labels_per_aug = [set() for _ in range(n_entries)]

            for aug_id in range(n_entries):
                for t in range(t1, t2):
                    labels_at_t = _labels[_ts == t]
                    data = augmented_dict[str(aug_id)]["data"]

                    coords_at_t = []
                    for lab in labels_at_t:
                        try:
                            coords_at_t.append(data[t][lab]["coords"])
                            present_labels_per_aug[aug_id].add((t, lab))
                        except KeyError:
                            continue
                    if len(coords_at_t) == 0:
                        coords_at_t = np.zeros((0, self.ndim), dtype=int)
                    else:
                        coords_at_t = np.stack(coords_at_t, axis=0)
                    _coords[aug_id].extend(coords_at_t)

                    features_at_t = []
                    for lab in labels_at_t:
                        try:
                            features_at_t.append(data[t][lab]["features"])
                        except KeyError:
                            continue
                    if len(features_at_t) == 0:
                        features_at_t = {}
                    else:
                        features_at_t = [dict(f) for f in features_at_t]
                    for _f in features_at_t:
                        for k, v in _f.items():
                            if k not in _features[aug_id]:
                                _features[aug_id][k] = []
                            _features[aug_id][k].append(v)

            # --- Filter labels missing in any augmentation --- #
            # (This can happen due to downsampling in pretrained features,
            # label has too few pixels to have any valid associated features.
            # If this occurs for too many labels, check the data and augmentation settings.)
            common_labels = set.intersection(*present_labels_per_aug)
            keep_mask = np.array([(t, lab) in common_labels for t, lab in zip(_ts, _labels)])
            if np.sum(~keep_mask) > 0:
                missing_labels = set(
                    (t, lab) for t, lab in zip(_ts[~keep_mask], _labels[~keep_mask])
                )
                logger.warning(
                    f"Labels were removed from window {t1} to {t2}"
                    f", as those labels are missing in some augmentations. If this occurs for too many labels,"
                    f" check the data and ensure augmentation settings are appropriate."
                )
                logger.warning(f"Removed labels: {missing_labels}")
                _labels = _labels[keep_mask]
                _ts = _ts[keep_mask]
                for aug_id in range(n_entries):
                    filtered_coords = []
                    filtered_features = {k: [] for k in _features[aug_id].keys()}
                    idx_counter = 0
                    for t in range(t1, t2):
                        labels_at_t = _labels[_ts == t]
                        n = len(labels_at_t)
                        filtered_coords.append(_coords[aug_id][idx_counter:idx_counter + n])
                        for k in _features[aug_id].keys():
                            filtered_features[k].extend(_features[aug_id][k][idx_counter:idx_counter + n])
                        idx_counter += n
                    _coords[aug_id] = np.concatenate(filtered_coords, axis=0) if filtered_coords else np.zeros((0, self.ndim), dtype=np.float32)
                    for k in filtered_features:
                        _features[aug_id][k] = np.array(filtered_features[k], dtype=np.float32)
            else:
                # No missing labels, just convert to arrays as usual
                for aug_id in range(n_entries):
                    _coords[aug_id] = np.array(_coords[aug_id], dtype=np.float32)
                    for k, v in _features[aug_id].items():
                        _features[aug_id][k] = np.array(v, dtype=np.float32)
            
            if len(_labels) == 0:
                # raise ValueError(f"No detections in sample {det_folder}:{t1}") # empty frames can happen
                A = np.zeros((0, 0), dtype=bool)
            else:
                A = _ctc_assoc_matrix(
                    _labels,
                    _ts,
                    self.gt_graph,
                    matching,
                )
                        
            w = dict(
                coords=_coords,
                t1=t1,
                # img=self.imgs[t1:t2],
                # mask=det_masks[t1:t2],
                assoc_matrix=A,
                labels=_labels,
                timepoints=_ts,
                features=_features,
            )
            if not len(_coords) == n_entries or not len(_features) == n_entries:
                raise ValueError(f"Number of coords {len(_coords)} or features {len(_features)} does not match number of augmentations {n_entries}")
            windows.append(w)
            
        logger.debug(f"Built {len(windows)} track windows.\n")
        return windows
    
    def _save_windows(self):
        if self.window_save_path is not None:
            self._len = len(self.windows)
            logger.info(f"Saving windows to {self.window_save_path}")
            mode = "w" if self.force_recompute else "a"
            root = zarr.open_group(str(self.window_save_path), mode=mode)
            for i, w in enumerate(self.windows):
                group_name = f"window_{i}"
                if group_name in root:
                    del root[group_name]
                grp = root.create_group(group_name)
                for aug_id in range(self.n_augs + 1):
                    grp.create_dataset(f"coords_{aug_id}", data=w["coords"][aug_id])
                    features_group = grp.create_group(f"features_{aug_id}")
                    for k, v in w["features"][aug_id].items():
                        features_group.create_dataset(k, data=v)
                grp.create_dataset("labels", data=w["labels"])
                grp.create_dataset("timepoints", data=w["timepoints"])
                grp.create_dataset("assoc_matrix", data=w["assoc_matrix"])
                grp.attrs["t1"] = w["t1"]
        else:
            raise ValueError("No augmented embeddings zarr file set. Cannot save windows.")

    def _load_windows(self):
        if not self.load_from_disk:
            if getattr(self, "windows", None) is not None:
                logger.debug("Windows already loaded into memory.")
                return self.windows
            return None
        if self.window_save_path.exists() and not self.force_recompute:
            self.windows = []
            logger.info(f"Loading windows from {self.window_save_path}")
            root = zarr.open_group(str(self.window_save_path), mode="r")
            group_names = sorted(
                root.keys(),
                key=lambda x: int(x.split("_")[1]) if x.startswith("window_") else x
            )
            logger.debug(f"Found {len(group_names)} windows, loading...")
            for w in group_names:
                grp = root[w]
                coords = [grp[f"coords_{aug_id}"][...] for aug_id in range(self.n_augs + 1)]
                features = {}
                for aug_id in range(self.n_augs + 1):
                    features[aug_id] = {k: grp[f"features_{aug_id}"][k][...] for k in grp[f"features_{aug_id}"].keys()}
                labels = grp["labels"][...]
                timepoints = grp["timepoints"][...]
                assoc_matrix = grp["assoc_matrix"][...]
                t1 = grp.attrs["t1"]
                self.windows.append(dict(
                    coords=coords,
                    features=features,
                    labels=labels,
                    timepoints=timepoints,
                    assoc_matrix=assoc_matrix,
                    t1=t1,
                ))
            self._len = len(self.windows)
            self._get_ndim_and_nobj(None, self.windows)
            logger.info(f"Loaded {self._len} windows from {self.window_save_path}")
            return self.windows

    @lru_cache
    def _sample_from_memory(self, n: int, aug_choice: int = 0):
        """When self.load_from_disk is False, sample a window from memory."""
        # logger.debug(f"Sampling window {n} with augmentation choice {aug_choice}")
        track = self.windows[n]
        # 0 is original, 1 to n_augs are the augmented versions
        coords = track["coords"][aug_choice]
        features = track["features"][aug_choice]
        assoc_matrix = track["assoc_matrix"]
        labels = track["labels"]
        timepoints = track["timepoints"]
        t1 = track["t1"]

        return coords, features, labels, timepoints, assoc_matrix, t1

    def _sample_from_file(self, window_id: int, aug_choice: int = 0):
        """When self.load_from_disk is True, sample a window from the saved zarr file."""
        root = zarr.open_group(str(self.window_save_path), mode="r")
        grp = root[f"window_{window_id}"]
        coords = grp[f"coords_{aug_choice}"][...]
        features = {}
        for k in grp[f"features_{aug_choice}"].keys():
            features[k] = grp[f"features_{aug_choice}"][k][...]
        labels = grp["labels"][...]
        timepoints = grp["timepoints"][...]
        assoc_matrix = grp["assoc_matrix"][...]
        t1 = grp.attrs["t1"]
        return coords, features, labels, timepoints, assoc_matrix, t1

    def _augment_item(self, item: wrfeat.WRAugPretrainedFeatures, labels, timepoints, assoc_matrix):
        """Apply augmentations to the features."""
        # FIXME some arguments are redundant
        if self.cropper is not None:
            # Use only if there is at least one timepoint per detection
            cropped_item, cropped_idx = self.cropper(item)
            cropped_timepoints = item.timepoints[cropped_idx]
            if len(np.unique(cropped_timepoints)) == self.window_size:
                idx = cropped_idx
                item = cropped_item
                labels = labels[idx]
                timepoints = timepoints[idx]
                assoc_matrix = assoc_matrix[idx][:, idx]
            # else:
            #     logger.debug("Skipping cropping")
        
        if self.augmenter is not None:
            item = self.augmenter(item)
            
        return item, assoc_matrix
    
    @lru_cache
    def get_augmented_image_shape(self, aug_choice: int):
        try:
            image_shape = self.augmented_image_shapes[aug_choice]
        except KeyError:
            root = zarr.open_group(str(self._aug_embeds_file), mode="r")
            metadata_json = root[str(aug_choice)].attrs["metadata"]
            metadata = json.loads(metadata_json)
            image_shape = metadata["image_shape"]
        return image_shape

    def __len__(self):
        if self.save_windows and self.windows is None:
            return self._len
        else:
            return len(self.windows)
    
    def __getitem__(self, n: int, return_dense=None):
        if return_dense is None:
            return_dense = self.return_dense
        
        random_aug_choice = self._rng.integers(0, self.n_augs + 1)
         
        if self.load_from_disk:
            coords, features, labels, timepoints, assoc_matrix, _ = self._sample_from_file(
                    n, random_aug_choice
                )
        else:
            coords, features, labels, timepoints, assoc_matrix, _ = self._sample_from_memory(
                n, random_aug_choice
            )

        # if return_dense and isinstance(mask, _CompressedArray):
        #     mask = CTCDataAugPretrainedFeats.decompress(mask)
        # if return_dense and isinstance(img, _CompressedArray):
        #     img = CTCDataAugPretrainedFeats.decompress(img)
        if isinstance(assoc_matrix, _CompressedArray):
            assoc_matrix = CTCDataAugPretrainedFeats.decompress(assoc_matrix)
        
        coords = np.stack(coords, axis=0)
        # features = np.stack(features, axis=0)

        augment_wrfeat = wrfeat.WRAugPretrainedFeatures.from_window(
            features=features,
            coords=coords,
            timepoints=timepoints,
            labels=labels,
        )
        augmented_data, assoc_matrix = self._augment_item(augment_wrfeat, labels, timepoints, assoc_matrix)
        if not isinstance(augmented_data, wrfeat.WRAugPretrainedFeatures):
            raise ValueError("Augmented data is not a WRAugPretrainedFeatures. Check that augmenter return type is correct.")
        features, pretrained_features, coords, timepoints, labels = augmented_data.to_window()
        
        shapes = [
            len(labels),
            len(timepoints),
            len(coords),
            len(pretrained_features),
            len(assoc_matrix),
        ]
        if features is not None:
            shapes.append(len(features))
        if len(np.unique(shapes)) != 1:
            raise ValueError(f"Shape mismatch: {shapes} (labs/timepoints/coords/features)")
        
        if coords.shape[-1] != self.ndim + 1:
            raise ValueError(f"Coords shape mismatch: {coords.shape[-1]} != {self.ndim + 1}")
        
        # coords is already including time, simply remove min_time along the first axis
        # coords[:, 0] -= min_time
        
        if self.max_tokens and len(timepoints) > self.max_tokens:
            time_incs = np.where(timepoints - np.roll(timepoints, 1))[0]
            n_elems = time_incs[np.searchsorted(time_incs, self.max_tokens) - 1]
            timepoints = timepoints[:n_elems]
            labels = labels[:n_elems]
            coords = coords[:n_elems]
            if features is not None:
                features = features[:n_elems]
            if pretrained_features is not None:
                pretrained_features = pretrained_features[:n_elems]
            assoc_matrix = assoc_matrix[:n_elems, :n_elems]
            logger.info(
                f"Clipped window of size {timepoints[n_elems - 1] - timepoints.min()}"
            )
            
        coords0 = torch.from_numpy(coords).float()
        if features is not None:
            features = torch.from_numpy(features).float()
        pretrained_features = torch.from_numpy(pretrained_features).float()
        assoc_matrix = torch.from_numpy(assoc_matrix.copy()).float()
        labels = torch.from_numpy(labels).long()
        timepoints = torch.from_numpy(timepoints).long()
        
        if self.augmenter is not None:
            coords = coords0.clone()
            coords[:, 1:] += torch.randint(0, 512, (1, self.ndim))
        else:
            coords = coords0.clone()
        
        if self.rotate_feats:
            image_shape = self.get_augmented_image_shape(random_aug_choice)
            pretrained_features = CTCData.rotate_features(
                pretrained_features, coords, image_shape,
                n_rot_dims=self.pretrained_feat_dim  # // 2
            )

        res = dict(
            features=features,
            pretrained_features=pretrained_features,
            coords0=coords0,
            coords=coords,
            assoc_matrix=assoc_matrix,
            timepoints=timepoints,
            labels=labels,
        )
        
        # if return_dense:
        #     if all([x is not None for x in img]):
        #         img = torch.from_numpy(img).float()
        #         res["img"] = img

        #     mask = torch.from_numpy(mask.astype(int)).long()
        #     res["mask"] = mask
        return res


def _ctc_lineages(df, masks, t1=0, t2=None):
    """From a ctc dataframe, create a digraph that contains all sublineages
    between t1 and t2 (exclusive t2).

    Args:
        df: pd.DataFrame with columns `label`, `t1`, `t2`, `parent` (man_track.txt)
        masks: List of masks. If t1 is not 0, then the masks are assumed to be already cropped accordingly.
        t1: Start timepoint
        t2: End timepoint (exclusive). If None, then t2 is set to len(masks)

    Returns:
        labels: List of label ids extracted from the masks, ordered by timepoint.
        ts: List of corresponding timepoints
        graph: The digraph of the lineages between t1 and t2.
    """
    if t1 > 0:
        assert t2 is not None
        assert t2 - t1 == len(masks)
    if t2 is None:
        t2 = len(masks)

    graph = nx.DiGraph()
    labels = []
    ts = []

    # get all objects that are present in the time interval
    df_sub = df[(df.t1 < t2) & (df.t2 >= t1)]

    # Correct offset
    df_sub.loc[:, "t1"] -= t1
    df_sub.loc[:, "t2"] -= t1

    # all_labels = df_sub.label.unique()
    # TODO speed up by precalculating unique values once
    # in_masks = set(np.where(np.bincount(np.stack(masks[t1:t2]).ravel()))[0]) - {0}
    # all_labels = [l for l in all_labels if l in in_masks]
    all_labels = set()

    for t in tqdm(
        range(0, t2 - t1), desc="Building and checking lineage graph", leave=False
    ):
        # get all entities at timepoint
        obs = df_sub[(df_sub.t1 <= t) & (df_sub.t2 >= t)]
        in_t = set(np.where(np.bincount(masks[t].ravel()))[0]) - {0}
        all_labels.update(in_t)
        for row in obs.itertuples():
            label, t1, t2, parent = row.label, row.t1, row.t2, row.parent
            if label not in in_t:
                continue

            labels.append(label)
            ts.append(t)

            # add label as node if not already in graph
            if not graph.has_node(label):
                graph.add_node(label)

            # Parents have been added in previous timepoints
            if parent in all_labels:
                if not graph.has_node(parent):
                    graph.add_node(parent)
                graph.add_edge(parent, label)

    labels = np.array(labels)
    ts = np.array(ts)
    return labels, ts, graph


@njit
def _assoc(A: np.ndarray, labels: np.ndarray, family: np.ndarray):
    """For each detection, associate with all detections that are."""
    for i in range(len(labels)):
        for j in range(len(labels)):
            A[i, j] = family[i, labels[j]]


def determine_ctc_class(dataset_kwargs: dict):
    if "features" not in dataset_kwargs:
        raise ValueError("features must be set in dataset_kwargs")
    if dataset_kwargs["features"] == "pretrained_feats_aug":
        return CTCDataAugPretrainedFeats
    else:
        return CTCData
        

def _ctc_assoc_matrix(detections, ts, graph, matching):
    """Create the association matrix for a list of labels and a tracklet parent -> childrend graph.

    Each detection is associated with all its ancestors and descendants, but not its siblings and their offspring.

    Args:
        detections: list of integer labels, ordered by timepoint
        ts: list of timepoints corresponding to the detections
        graph: networkx DiGraph with each ground truth tracklet id (spanning n timepoints) as a single node
            and parent -> children relationships as edges.
        matching: for each timepoint, a dictionary that maps from detection id to gt tracklet id
    """
    assert 0 not in graph
    matched_gt = []
    for i, (label, t) in enumerate(zip(detections, ts)):
        gt_tracklet_id = matching[t].get(label, 0)
        matched_gt.append(gt_tracklet_id)
    matched_gt = np.array(matched_gt, dtype=int)
    # Now we have the subset of gt nodes that is matched to any detection in the current window

    # relabel to reduce the size of lookup matrices
    # offset 0 not allowed in skimage, which makes this very annoying
    relabeled_gt, fwd_map, _inv_map = relabel_sequential(matched_gt, offset=1)
    # dict is faster than arraymap
    fwd_map = dict(zip(fwd_map.in_values, fwd_map.out_values))
    # inv_map = dict(zip(inv_map.in_values, inv_map.out_values))

    # the family relationships for each ground truth detection,
    # Maps from local detection number (0-indexed) to global gt tracklet id (1-indexed)
    family = np.zeros((len(detections), len(relabeled_gt) + 1), bool)

    # Connects each tracklet id with its children and parent tracklets (according to man_track.txt)
    for i, (label, t) in enumerate(zip(detections, ts)):
        # Get the original label corresponding to the graph
        gt_tracklet_id = matching[t].get(label, None)
        if gt_tracklet_id is not None:
            ancestors = []
            descendants = []
            # This iterates recursively through the graph
            for n in nx.descendants(graph, gt_tracklet_id):
                if n in fwd_map:
                    descendants.append(fwd_map[n])
            for n in nx.ancestors(graph, gt_tracklet_id):
                if n in fwd_map:
                    ancestors.append(fwd_map[n])

            family[i, np.array([fwd_map[gt_tracklet_id], *ancestors, *descendants])] = (
                True
            )
        else:
            pass
            # Now we match to nothing, so even the matrix diagonal will not be filled.

    # This assures that matching to 0 is always false
    assert family[:, 0].sum() == 0

    # Create the detection-to-detection association matrix
    A = np.zeros((len(detections), len(detections)), dtype=bool)

    _assoc(A, relabeled_gt, family)

    return A


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _compress_img_mask_preproc(img, mask, features):
    """Remove certain img pixels if not needed to save memory for large datasets."""
    # dont change anything if we need patch values
    if features in ("patch", "patch_regionprops"):
        # clear img pixels outside of patch_mask of size 16x16
        patch_width = 16  # TOD: hardcoded: change this if needed
        coords = tuple(np.array(r.centroid).astype(int) for r in regionprops(mask))
        img2 = np.zeros_like(img)
        if len(coords) > 0:
            coords = np.stack(coords)
            coords = np.clip(coords, 0, np.array(mask.shape)[None] - 1)
            patch_mask = np.zeros_like(img, dtype=bool)
            patch_mask[tuple(coords.T)] = True
            # retain 3*patch_width+1 around center to be safe...
            patch_mask = ndi.maximum_filter(patch_mask, 3 * patch_width + 1)
            img2[patch_mask] = img[patch_mask]

    else:
        # otherwise set img value inside masks to mean
        # FIXME: change when using other intensity based regionprops
        img2 = np.zeros_like(img)
        for reg in regionprops(mask, intensity_image=img):
            m = mask[reg.slice] == reg.label
            img2[reg.slice][m] = reg.mean_intensity
    return img2


def pad_tensor(x, n_max: int, dim=0, value=0):
    n = x.shape[dim]
    if n_max < n:
        raise ValueError(f"pad_tensor: n_max={n_max} must be larger than n={n} !")
    pad_shape = list(x.shape)
    pad_shape[dim] = n_max - n
    # pad = torch.full(pad_shape, fill_value=value, dtype=x.dtype).to(x.device)
    pad = torch.full(pad_shape, fill_value=value, dtype=x.dtype)
    return torch.cat((x, pad), dim=dim)


def collate_sequence_padding(batch):
    """Collate function that pads all sequences to the same length."""
    lens = tuple(len(x["coords"]) for x in batch)
    n_max_len = max(lens)
    # print(tuple(len(x["coords"]) for x in batch))
    # print(tuple(len(x["features"]) for x in batch))
    # print(batch[0].keys())
    tuple(batch[0].keys())
    normal_keys = {
        "coords": 0,
        "features": 0,
        "pretrained_features": 0,
        "labels": 0,  # Not needed, remove for speed.
        "timepoints": -1,  # There are real timepoints with t=0. -1 for distinction from that.
    }
    actual_keys = {
        k: v for k, v in normal_keys.items() if k in batch[0] and batch[0][k] is not None
    }
    none_keys = [
        k for k in normal_keys.keys() if k in batch[0] and batch[0][k] is None
    ]
    n_pads = tuple(n_max_len - s for s in lens)
    batch_new = dict(
        (
            k,
            torch.stack(
                [pad_tensor(x[k], n_max=n_max_len, value=v) for x in batch], dim=0
            ),
        )
        for k, v in actual_keys.items()
    )
    for k in none_keys:
        batch_new[k] = None
    batch_new["assoc_matrix"] = torch.stack(
        [
            pad_tensor(
                pad_tensor(x["assoc_matrix"], n_max_len, dim=0), n_max_len, dim=1
            )
            for x in batch
        ],
        dim=0,
    )

    # add boolean mask that signifies whether tokens are padded or not (such that they can be ignored later)
    pad_mask = torch.zeros((len(batch), n_max_len), dtype=torch.bool)
    for i, n_pad in enumerate(n_pads):
        pad_mask[i, n_max_len - n_pad :] = True

    batch_new["padding_mask"] = pad_mask.bool()
    if torch.all(pad_mask.bool()):
        raise ValueError("No valid entries for padding mask!")
    return batch_new


if __name__ == "__main__":

    dummy_data = CTCData(
        root="../../scripts/data/synthetic_cells/01",
        ndim=2,
        detection_folders=["TRA"],
        window_size=4,
        max_tokens=None,
        augment=3,
        features="none",
        downscale_temporal=1,
        downscale_spatial=1,
        sanity_dist=False,
        crop_size=(256, 256),
    )

    x = dummy_data[0]
