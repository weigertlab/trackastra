import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import joblib
import numpy as np
import torch
import torch.nn.functional as F
import zarr
from numcodecs import Blosc
from sam2.sam2_image_predictor import SAM2ImagePredictor
from skimage.measure import regionprops
from tqdm import tqdm
from transformers import (
    AutoImageProcessor,
    # Dinov2Config,
    # Dinov2Model,
    AutoModel,
    HieraConfig,
    HieraModel,
    SamModel,
    SamProcessor,
)

from trackastra.data import wrfeat
from trackastra.utils.utils import percentile_norm

if TYPE_CHECKING:
    from trackastra.data.pretrained_augmentations import PretrainedAugmentations

try:
    from micro_sam.util import get_sam_model as get_microsam_model
    MICRO_SAM_AVAILABLE = True
except ImportError:
    MICRO_SAM_AVAILABLE = False

try:
    from tarrow.models import TimeArrowNet
    from tarrow.utils import normalize as tap_normalize
    TARROW_AVAILABLE = True
except ImportError:
    TARROW_AVAILABLE = False

try:
    from cellpose import transforms as cp_transforms
    from cellpose.vit_sam import Transformer as CellposeSAM
    CELLPOSE_AVAILABLE = True
except ImportError:
    CELLPOSE_AVAILABLE = False

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Updated with actual class after each definition
# See register_backbone decorator
AVAILABLE_PRETRAINED_BACKBONES = {}

PretrainedFeatsExtractionMode = Literal[
    # "exact_patch",  # Uses the image patch centered on the detection for embedding
    "nearest_patch",  # Runs on whole image, then finds the nearest patch to the detection in the embedding
    "mean_patches_bbox",  # Runs on whole image, then averages the embeddings of all patches that intersect with the detection's bounding box
    "mean_patches_exact",  # Runs on whole image, then averages the embeddings of all patches that intersect with the detection
    "max_patches_bbox",  # Runs on whole image, then takes the maximum for each feature dimension of all patches that intersect with the detection
    "max_patches_exact",  # Runs on whole image, then takes the maximum for each feature dimension of all patches that intersect with the detection
    "median_patches_exact",  # Runs on whole image, then takes the median for each feature dimension of all patches that intersect with the detection
]

PretrainedBackboneType = Literal[  # cannot unpack this directly in python < 3.11 so it has to be copied
    "facebook/hiera-tiny-224-hf",  # 768
    "facebook/dinov2-base",  # 768
    "facebook/sam-vit-base",  # 256
    "facebook/sam2-hiera-large",  # 256
    "facebook/sam2.1-hiera-base-plus",  # 256
    "facebookresearch/co-tracker",  # 128
    "microsam/vit_b_lm",
    "microsam/vit_l_lm",
    "weigertlab/tarrow",  # arbitrary. default 32
    "mouseland/cellpose-sam",  # 192
    "facebook/sam2.1-hiera-base-plus/highres",
    "debug/random",
    "debug/encoded_labels",  # 32
]


def register_backbone(model_name, feat_dim):
    def decorator(cls):
        AVAILABLE_PRETRAINED_BACKBONES[model_name] = {
            "class": cls,
            "feat_dim": feat_dim,
        }
        return cls
    return decorator


# Feature extraction from pretrained models
# Meant to wrap any transformers model
# >NOTE : currently not applicable to 3D data
# (but aggregation-based modes may be adapted eventually)
import time
from functools import wraps


def average_time_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not hasattr(wrapper, 'total_time'):
            wrapper.total_time = 0
            wrapper.call_count = 0
        
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        
        wrapper.total_time += elapsed_time
        wrapper.call_count += 1
        average_time = wrapper.total_time / wrapper.call_count
        
        print(f"Average time taken by {func.__name__}: {average_time:.6f} seconds over {wrapper.call_count} calls")
        
        return result
    
    return wrapper


# Configs for pretrained models ###

@dataclass
class PretrainedFeatureExtractorConfig:
    """model_name (str):
        Specify the pretrained backbone to use.
    model_path (str | Path):
        Path to the pretrained model.
    save_path (str | Path):
        Specify the path to save the embeddings.
    batch_size (int):
        Specify the batch size to use for the model.
    mode (str):
        Specify the mode to use for the model.
        Currently available modes are "nearest_patch", "mean_patches_bbox", "mean_patches_exact", "max_patches_bbox", "max_patches_exact".
    normalize_embeddings (bool):
        Whether to normalize the embeddings (divide by the norm).
    device (str):
        Specify the device to use for the model.
        If not set and "pretrained_feats" is used, the device is automatically set by default to "cuda", "mps" or "cpu" as available.
    n_augmented_copies (int):
        How many augmented copies of the embeddings to create. If 0, only the original embeddings are saved. Creates n+1 embeddings entries total. 
    additional_features (str):
        Specify any additional features (from regionprops) to include in the extraction process. See WRFeat documentation for available features. Unused if None.
    pca_components (int):
        Specify the number of PCA components to use for dimensionality reduction of the features. Unused if None.
    pca_preprocessor_path (str | Path):
        Specify the path to the pickled PCA preprocessor. This is used to transform the features to a reduced PCA feature space.
    """
    model_name: PretrainedBackboneType
    model_path: str | Path = None
    save_path: str | Path = None
    batch_size: int = 4
    mode: PretrainedFeatsExtractionMode = "nearest_patch"
    normalize_embeddings: bool = True  # whether to normalize the embeddings (divide by the norm)
    device: str | None = None
    feat_dim: int = None
    additional_features: str | None = None  # for regionprops features
    additional_feat_dim: int = 0  # for regionprops features
    n_augmented_copies: int = 0  # number of augmented copies to create
    seed: int | None = None  # seed for debug/random
    # pca_components: int = None  # for PCA reduction of the features
    # pca_preprocessor_path: str | Path = None  # for PCA preprocessor path
    # apply_rope: bool = False  # whether to apply RoPE-like rotation to the features based on coordinates
    
    def __post_init__(self):
        self._guess_device()
        self.model_path = self._check_path(self.model_path)
        self.save_path = self._check_path(self.save_path)
        # self.pca_preprocessor_path = self._check_path(self.pca_preprocessor_path)
        self._check_model_availability()
        
    def _check_path(self, path):
        if path is not None and not isinstance(path, str | Path):
            raise ValueError(f"Path must be a string or Path object, got {type(path)}.")
        if isinstance(path, str):
            return Path(path).resolve()
        return path
    
    def _check_model_availability(self):
        if self.model_name not in AVAILABLE_PRETRAINED_BACKBONES.keys():
            raise ValueError(f"Model {self.model_name} is not available for feature extraction.")
        if self.model_name == "weigertlab/tarrow":
            if not TARROW_AVAILABLE:
                raise ImportError("TArrow is not available. Please install it to use this model.")
            elif self.model_path is None:
                raise ValueError("Model path must be specified for TArrow.")
            _, self.feat_dim = TAPFeatures._load_model_from_path(self.model_path)
        else:
            self.feat_dim = AVAILABLE_PRETRAINED_BACKBONES[self.model_name]["feat_dim"]
        if self.additional_features is not None:
            # TODO if this ever accepts 3D data this will be incorrect
            self.additional_feat_dim = wrfeat.WRFeatures.PROPERTIES_DIMS[
                self.additional_features
            ][2]
            if self.additional_features not in wrfeat._PROPERTIES:
                raise ValueError(f"Additional feature {self.additional_features} is not valid.")
        # if self.pca_components is not None:
        #     self.feat_dim = self.pca_components

    def _guess_device(self):
        if self.device is None:
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
        
        try:
            torch.device(self.device)  # check if device is valid
        except Exception as e:
            raise ValueError(f"Invalid device: {self.device}") from e

    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)

# Feature extractors ###


class FeatureExtractor(ABC):
    model_name = None
    _available_backbones = None

    def __init__(
        self, 
        image_size: tuple[int, int],
        save_path: str | Path,
        batch_size: int = 4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        mode: PretrainedFeatsExtractionMode = "nearest_patch",
        normalize_embeddings: bool = True,
        **kwargs,
        ):
        """ 
        Initializes a pretrained model feature extractor with the given parameters.
        Consumes images, computes embeddings of shape T, H*W, N (N the model feature dimension),
        and generates n_regions x N embeddings for each object in each frame using the specified mode.
        
        Args:
        - image_size (tuple[int, int]): Size of the input images (height, width).
        - save_path (str | Path): Path to save the embeddings.
        - batch_size (int): Batch size to use for the model.
        - device (str): Device to use for the model. Defaults to "cuda" if available, otherwise "cpu".
        - mode (str): Mode to use for the model. Defaults to "nearest_patch". See type PretrainedFeatsExtractionMode for available modes.
        - normalize_embeddings (bool): Whether to normalize the embeddings (divide by the norm). Defaults to True. For aggregation-based modes, this is applied before aggregation.
        """
        # Image processor extra args
        # Modify as needed in subclasses
        self.im_proc_kwargs = {
            "do_rescale": False,
            "do_normalize": False,
            "do_resize": True,
            "return_tensors": "pt",
            "do_center_crop": False,
        }
        # Model specs
        self.model = None
        self._input_size: tuple[int] = None
        self._final_grid_size: tuple[int] = None
        self.n_channels: int = None
        self.hidden_state_size: int = None
        self.model_patch_size: int = None
        # Data specs
        self.orig_image_size = image_size
        self.orig_n_channels = 1
        # Batch options and preprocessing
        self.do_normalize = True
        self.rescale_batches = False
        self.channel_first = True
        self.batch_return_type: Literal["list[np.ndarray]", "np.ndarray", "torch.Tensor"] = "np.ndarray"
        self.batch_size = batch_size
        self.device = device
        # Parameters for embedding extraction
        self.mode = mode
        # If running FeatureExtractor in a parallelized context, set to False to avoid overhead 
        # from spawning many threads within the parallelized context.
        self.parallel = True
        self.additional_features = None
        self.apply_rope = False  # deprecated, use "rotate_features" in CTCData # TODO remove
        self.normalize_embeddings = normalize_embeddings
        # Saving parameters
        self.save_path: str | Path = save_path
        self.do_save = True
        self.force_recompute = False
        
        self.embeddings = None
        self._debug_view = False
        # self._debug = True
        
        if not isinstance(self.save_path, Path):
            self.save_path = Path(self.save_path)
        
        if not self.save_path.exists():
            self.save_path.mkdir(parents=True, exist_ok=True)
    
    @property
    def input_size(self):
        return self._input_size
    
    @input_size.setter
    def input_size(self, value: int | tuple[int]):
        if isinstance(value, int):
            value = (value, value)
        elif isinstance(value, tuple):
            if len(value) != 2:
                raise ValueError("Input size must be a tuple of length 2.")
        else:
            raise ValueError("Input size must be an int or a tuple of ints.")
        self._input_size = value
        self._set_model_patch_size()
        
    @property
    def final_grid_size(self):
        return self._final_grid_size
    
    @final_grid_size.setter
    def final_grid_size(self, value: int | tuple[int]):
        if isinstance(value, int):
            value = (value, value)
        elif isinstance(value, tuple):
            if len(value) != 2:
                raise ValueError("Final grid size must be a tuple of length 2.")
        else:
            raise ValueError("Final grid size must be an int or a tuple of ints.")
        self._final_grid_size = value
        self._set_model_patch_size()
        
    @property
    def model_name_path(self):
        return self.model_name.replace("/", "-")
    
    @staticmethod
    def _load_model_from_path(self) -> tuple[torch.nn.Module, int]:
        """Loads the model from the specified path. Returns the model and the model feature dimension (e.g. 256 for SAM2)."""
        raise NotImplementedError("This model currently only supports being loaded from huggingface's hub.")
    
    @classmethod
    def from_model_name(cls, 
                        model_name: PretrainedBackboneType, 
                        image_shape: tuple[int, int], 
                        save_path: str | Path, 
                        device: torch.device = "cuda" if torch.cuda.is_available() else "cpu",
                        mode="nearest_patch",
                        additional_features=None,
                        model_folder=None,
                        ):
        cls._available_backbones = AVAILABLE_PRETRAINED_BACKBONES
        if model_name not in cls._available_backbones:
            raise ValueError(f"Model {model_name} is not available for feature extraction.")
        logger.info(f"Using model {model_name} with mode {mode} for pretrained feature extraction.")
        backbone = cls._available_backbones[model_name]["class"]
        backbone.model_name = model_name
        model = backbone(
            image_size=image_shape,
            save_path=save_path, 
            device=device, 
            mode=mode,
            model_folder=model_folder,
            )
        model.additional_features = additional_features
        return model
        
    @classmethod
    def from_config(cls, config: PretrainedFeatureExtractorConfig, image_shape: tuple[int, int], save_path: str | Path | None = None):
        cls._available_backbones = AVAILABLE_PRETRAINED_BACKBONES
        if config.model_name not in cls._available_backbones:
            raise ValueError(f"Model {config.model_name} is not available for feature extraction.")
        logger.info(f"Using model {config.model_name} with mode {config.mode} for pretrained feature extraction.")
        backbone = cls._available_backbones[config.model_name]["class"]
        
        parts = config.model_name.split("/")
        if len(parts) > 2:
            model_name = "/".join(parts[:2])
        else:
            model_name = config.model_name
        backbone.model_name = model_name

        model = backbone(
            image_size=image_shape, 
            save_path=save_path if save_path is not None else config.save_path,
            batch_size=config.batch_size, 
            device=config.device, 
            mode=config.mode,
            additional_features=config.additional_features,
            model_folder=config.model_path,
            normalize_embeddings=config.normalize_embeddings,
            seed=config.seed if hasattr(config, "seed") else None,
            # n_augmented_copies=config.n_augmented_copies,
            # aug_pipeline=PretrainedAugmentations() if config.n_augmented_copies > 0 else None,
        )
        model.additional_features = config.additional_features
        model.normalize_embeddings = config.normalize_embeddings
        # model.apply_rope = config.apply_rope
        return model
        
    def clear_model(self):
        """Clears the model from memory."""
        if self.model is not None:
            del self.model
            self.model = None
            torch.cuda.empty_cache()
            logger.info("Model cleared from memory.")
        else:
            logger.warning("No model to clear from memory.")
    
    def _set_model_patch_size(self):
        if self.final_grid_size is None or self.input_size is None:
            self.model_patch_size = None
        else:
            if not isinstance(self.input_size, tuple):
                raise ValueError("Input size must be a tuple of ints.")
            self.model_patch_size = (
                self.input_size[0] // self.final_grid_size[0],
                self.input_size[1] // self.final_grid_size[1],
            )
            if self.model_patch_size[0] <= 0 or self.model_patch_size[1] <= 0:
                raise ValueError("Model patch size must be greater than 0.")

    def compute_region_features(
        self, 
        coords,
        masks=None, 
        timepoints=None,
        labels=None,
        # embeddings=None
    ) -> torch.Tensor:
        feats = torch.zeros(len(coords), self.hidden_state_size, device=self.device)
        match self.mode:
            case "nearest_patch":
                feats = self._nearest_patches(coords, masks, norm=self.normalize_embeddings)
                return feats  # Return early, nothing else to do
            case mode if mode.endswith("_patches_exact"):
                if masks is None or labels is None or timepoints is None:
                    raise ValueError("Masks and labels must be provided for the chosen patch mode.")
                feats_func = partial(self._agg_patches_exact, masks, timepoints, labels, norm=self.normalize_embeddings)
            case mode if mode.endswith("_patches_bbox"):
                if masks is None or labels is None or timepoints is None:
                    raise ValueError("Masks and labels must be provided for the chosen patch mode.")
                feats_func = partial(self._agg_patches_bbox, masks, timepoints, labels, norm=self.normalize_embeddings)
            case _:
                raise NotImplementedError(f"Mode {self.mode} is not implemented.")

        # Only for aggregation modes
        if "max" in self.mode:
            feats = feats_func(agg=torch.max)
        elif "mean" in self.mode:
            feats = feats_func(agg=torch.mean)
        elif "median" in self.mode:
            feats = feats_func(agg=torch.median)
        else:
            raise NotImplementedError(f"Unknown aggregation for mode {self.mode}")

        assert feats.shape == (len(coords), self.hidden_state_size)
        return feats  # (n_regions, embedding_size)
    
    def precompute_image_embeddings(self, images, **kwargs):  # , windows, window_size):
        """Precomputes embeddings for all images."""
        missing = self._check_missing_embeddings()
        all_embeddings = torch.zeros(len(images), self.final_grid_size[0] * self.final_grid_size[1], self.hidden_state_size, device=self.device)
        if missing:
            for ts, batches in tqdm(self._prepare_batches(images), total=len(images) // self.batch_size, desc="Computing embeddings", leave=False):
                embeddings = self._run_model(batches, **kwargs)
                if torch.any(embeddings.isnan()):
                    raise RuntimeError("NaN values found in features.")
                # logger.debug(f"Embeddings shape: {embeddings.shape}")
                all_embeddings[ts] = embeddings.to(torch.float32)
                assert embeddings.shape[-1] == self.hidden_state_size
            self.embeddings = all_embeddings
            self._save_features(all_embeddings)
        # logger.debug(f"Precomputed embeddings shape: {self.embeddings.shape}")
        return self.embeddings

    def _extract_region_embeddings(self, all_frames_embeddings, window, start_index, remaining=None):
        window_coords = window["coords"]
        window_timepoints = window["timepoints"]
        window_masks = window["mask"]
        window_labels = window["labels"]
        
        n_regions_per_frame, features = self.extract_embedding(window_masks, window_timepoints, window_labels, window_coords)
        
        for i in range(remaining or len(n_regions_per_frame)):
            # if computing remaining frames' embeddings, start from the end
            obj_per_frame = n_regions_per_frame[-i - 1] if remaining else n_regions_per_frame[i]
            frame_index = start_index + i if not remaining else np.max(window_timepoints) - i
            # logger.debug(f"Frame {frame_index} has {obj_per_frame} objects.")
            all_frames_embeddings[frame_index] = features[:obj_per_frame]
            features = features[obj_per_frame:]
    
    def extract_embedding(self, masks, timepoints, labels, coords):
        # if masks.shape[-2:] != self.orig_image_size:
        # This should not be occuring since each folder is loaded as a separate CTCData
        # However when computing augmented embeddings in parallel, the input size may change
        # logger.debug(f"Input shape change detected: {masks.shape[-2:]} from {self.orig_image_size}.")
        # self.orig_image_size = masks.shape[-2:]
        n_regions_per_frame = np.unique(timepoints, return_counts=True)[1]
        tot_regions = n_regions_per_frame.sum()
        coords_txy = np.concatenate((timepoints[:, None], coords), axis=-1)
        if coords_txy.shape[0] != tot_regions:
            raise RuntimeError(f"Number of coords ({coords_txy.shape[0]}) does not match the number of coordinates ({timepoints.shape[0]}).")
        features = self.compute_region_features(
            masks=masks,
            coords=coords_txy,
            timepoints=timepoints,
            labels=labels,
        )
        if torch.isnan(features).any():
            raise RuntimeError("NaN values found in features.")
        if tot_regions != features.shape[0]:
            raise RuntimeError(f"Number of regions ({n_regions_per_frame}) does not match the number of embeddings ({features.shape[0]}).")
        return n_regions_per_frame, features
    
    @abstractmethod
    def _run_model(self, images, **kwargs) -> torch.Tensor:  # must return (B, grid_size**2, hidden_state_size)
        """Extracts embeddings from the model."""
        pass
    
    def normalize_array(self, b):
        b = percentile_norm(b)
        if self.rescale_batches:
            b = b * 255.0
        return b
    
    @staticmethod
    def get_centroids_from_masks(masks: np.ndarray) -> np.ndarray:
        """Computes the centroids of the objects in the masks.
        
        Args:
            masks: (n_objects, H, W) array of masks.

        Returns:
            Centroids: (n_objects, 2) array of (y, x) centroid coordinates, normalized to [0, 1].
        """
        centroids_df = regionprops(masks)
        centroids = np.array([region.centroid for region in centroids_df])
        centroids[:, 0] = centroids[:, 0] / masks.shape[1]
        centroids[:, 1] = centroids[:, 1] / masks.shape[2]
        return centroids
    
    def apply_rot_to_features(self, features: torch.Tensor, centroids: np.ndarray) -> torch.Tensor:
        """Applies a rotation to each feature vector based on the object's centroid.
        
        Args:
            features: (n_objects, hidden_state_size) tensor of features.
            centroids: (n_objects, 2) array of (y, x) centroid coordinates, normalized to [0, 1].

        Returns:
            Rotated features: (n_objects, hidden_state_size)
        """
        n_objects, d = features.shape
        assert d % 2 == 0, "Feature dimension must be even for rotation."
        angle_x = torch.from_numpy(2 * np.pi * centroids[:, 0]).to(features.device)
        angle_y = torch.from_numpy(2 * np.pi * centroids[:, 1]).to(features.device)

        angles = torch.stack([angle_x, angle_y], dim=1).repeat(1, d // 2)
        angles = angles.view(n_objects, d)
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        features_ = features.view(n_objects, -1, 2)
        x_feat, y_feat = features_[..., 0], features_[..., 1]
        x_rot = x_feat * cos[:, ::2] - y_feat * sin[:, ::2]
        y_rot = x_feat * sin[:, ::2] + y_feat * cos[:, ::2]
        rotated = torch.stack([x_rot, y_rot], dim=-1).reshape(n_objects, d)
        if torch.allclose(rotated, features):
            logger.warning("Rotated features are equal to original features. Rotation may not be applied correctly.")
        return rotated
    
    def _prepare_batches(self, images):
        """Prepares batches of images for embedding extraction."""
        if self.do_normalize:
            images = self.normalize_array(images)
            if self.rescale_batches:
                images = images * 255.0
        for i in range(0, len(images), self.batch_size):
            end = i + self.batch_size
            end = min(end, len(images))
            batch = np.expand_dims(images[i:end], axis=1)  # (B, C, H, W)

            timepoints = range(i, end)
            if self.n_channels > 1:  # repeat channels if needed
                if self.orig_n_channels > 1 and self.orig_n_channels != self.n_channels:
                    raise ValueError("When more than one original channel is provided, the number of channels in the model must match the number of channels in the input.")
                batch = np.repeat(batch, self.n_channels, axis=1)
            if not self.channel_first:
                batch = np.moveaxis(batch, 1, -1)
            if self.batch_return_type == "list[np.ndarray]":
                batch = list([im for im in batch])
            yield timepoints, batch
    
    @staticmethod
    def normalize_tensor(embeddings: torch.Tensor, norm: bool = True) -> torch.Tensor:
        """Normalizes the embeddings by dividing by the norm."""
        if norm:
            embeddings = embeddings / (embeddings.norm(dim=-1, keepdim=True) + 1e-8)
        return embeddings
    
    def _map_coords_to_model_grid(self, coords):
        scale_x = self.input_size[0] / self.orig_image_size[0]
        scale_y = self.input_size[1] / self.orig_image_size[1]
        coords = np.array(coords)
        patch_x = (coords[:, 1] * scale_x).astype(int)
        patch_y = (coords[:, 2] * scale_y).astype(int)
        patch_coords = np.column_stack((coords[:, 0], patch_x, patch_y))
        return patch_coords
    
    def _find_nearest_cell(self, patch_coords):
        """Finds the nearest cell in the grid for each patch coordinate."""
        x_idxs = patch_coords[:, 1] // self.model_patch_size[0]
        y_idxs = patch_coords[:, 2] // self.model_patch_size[1]
        patch_idxs = np.column_stack((patch_coords[:, 0], x_idxs, y_idxs)).astype(int)
        return patch_idxs
    
    def _find_bbox_cells(self, regions: dict, cell_height: int, cell_width: int):
        """Finds the cells in a grid that a bounding box belongs to.
        
        Args:
        - regions (dict): Dictionary from regionprops. Must contain bbox.
        - cell_height (int): Height of a cell in the grid.
        - cell_width (int): Width of a cell in the grid.
    
        Returns:
        - tuple: A tuple containing the grid cell indices that the bounding box intersects.
        """
        mask_patches = {}
        for region in regions:
            minr, minc, maxr, maxc = region.bbox
            patches = self._find_region_cells(minr, minc, maxr, maxc, cell_height, cell_width)
            mask_patches[region.label] = patches
            
        return mask_patches
    
    @staticmethod
    def _find_region_cells(minr, minc, maxr, maxc, cell_height, cell_width):
        start_patch_y = minr // cell_height
        end_patch_y = (maxr - 1) // cell_height
        start_patch_x = minc // cell_width
        end_patch_x = (maxc - 1) // cell_width
        patches = np.array([(i, j) for i in range(start_patch_y, end_patch_y + 1) for j in range(start_patch_x, end_patch_x + 1)])
        return patches
    
    def _find_patches_for_masks(self, image_mask: np.ndarray) -> dict:
        """Find which patches in a grid each mask belongs to using regionprops.
        
        Args:
        - image_masks (np.ndarray): Masks where each region has a unique label.
                
        Returns:
        - mask_patches (dict): Dictionary with region labels as keys and lists of patch indices as values.
        """
        patch_height = image_mask.shape[0] // self.final_grid_size[0]
        patch_width = image_mask.shape[1] // self.final_grid_size[1]
        regions = regionprops(image_mask)
        return self._find_bbox_cells(regions, patch_height, patch_width)
    
    def _debug_show_patches(self, embeddings, masks, coords, patch_idxs):
        import napari
        v = napari.Viewer()
        # v.add_labels(masks)
        e = embeddings.detach().cpu().numpy().swapaxes(1, 2)
        e = e.reshape(-1, self.hidden_state_size, self.final_grid_size[0], self.final_grid_size[1]).swapaxes(0, 1)

        v.add_image(
            e,
            name="Embeddings",
        )
        # add red points at patch indices for the relevant frame
        points = np.zeros((len(patch_idxs) * self.hidden_state_size, 3))
        for i, (t, y, x) in enumerate(patch_idxs):
            point = np.array([t, y, x])
            points[i * self.hidden_state_size:(i + 1) * self.hidden_state_size] = np.tile(point, (self.hidden_state_size, 1))
        
        v.add_points(points, size=1, face_color='red', name='Patch Indices')

        from skimage.transform import resize
        masks_resized = resize(masks[0], (self.final_grid_size[0], self.final_grid_size[1]), anti_aliasing=False, order=0, preserve_range=True)
        v.add_labels(masks_resized)
        logger.debug(f"Lost labels : {set(np.unique(masks)) - set(np.unique(masks_resized))}")
        
        napari.run()
            
    def _nearest_patches(self, coords, masks=None, norm=True, embs=None):
        """Finds the nearest patches to the detections in the embedding."""
        # find coordinate patches from detections
        patch_coords = self._map_coords_to_model_grid(coords)
        patch_idxs = self._find_nearest_cell(patch_coords)
        # logger.debug(f"Patch indices: {patch_idxs}")

        # load the embeddings and extract the relevant ones
        feats = torch.zeros(len(coords), self.hidden_state_size, device=self.device)
        indices = [y * self.final_grid_size[1] + x for _, y, x in patch_idxs]
        unique_timepoints = list(set(t for t, _, _ in patch_idxs))
        # logger.debug(f"Unique timepoints: {unique_timepoints}")
        embeddings = self._load_features() if embs is None else embs
        
        # try:
        #   t = coords[0][0]
        #   if t == 3:
        #       self._debug_show_patches(embeddings, masks, coords, patch_idxs)
        # except IndexError:
        #   logger.debug("No timepoint found in coords.")

        # logger.debug(f"Embeddings shape: {embeddings.shape}")
        embeddings_dict = {t: embeddings[t] for t in unique_timepoints}
        try:
            for i, (t, _, _) in enumerate(patch_idxs):
                feats[i] = embeddings_dict[t][indices[i]]
        except KeyError as e:
            logger.error(f"KeyError: {e} - Check if the timepoint exists in embeddings_dict.")
        except IndexError as e:
            # TODO improve handling of this error. Maybe check shape earlier
            logger.error(f"IndexError: {e} - Embeddings exist but do not have the correct shape. Did the model input size change ? If so, please delete saved embeddings and recompute.")
        feats = FeatureExtractor.normalize_tensor(feats, norm=norm)
        if self.apply_rope:
            centroids = FeatureExtractor.get_centroids_from_masks(masks)
            feats = self.apply_rot_to_features(feats, centroids) 
        return feats
    
    # @average_time_decorator
    def _agg_patches_bbox(self, masks, timepoints, labels, agg=torch.mean, norm=True, embs=None):
        """Averages the embeddings of all patches that intersect with the detection.
        
        Args:
            - masks (np.ndarray): Masks where each region has a unique label (t x H x W).
            - timepoints (np.ndarray): For each region, contains the corresponding timepoint. (n_regions)
            - labels (np.ndarray): Unique labels of the regions. (n_regions)
            - agg (callable): Aggregation function to use for averaging the embeddings.
        """
        try:
            n_regions = len(timepoints)
            timepoints_shifted = timepoints - timepoints.min()
        except ValueError:
            logger.error("Error: issue computing shifted timepoints.")
            logger.error(f"Regions: {len(timepoints)}")
            logger.error(f"Timepoints: {timepoints}")
            return torch.zeros(n_regions, self.hidden_state_size, device=self.device)

        feats = torch.zeros(n_regions, self.hidden_state_size, device=self.device)
        patches = []
        times = np.unique(timepoints_shifted)
        patches_res = joblib.Parallel(n_jobs=8, backend="threading")(
            joblib.delayed(self._find_patches_for_masks)(masks[t]) for t in times
        )   
        patches = {t: patch for t, patch in zip(times, patches_res)}
        # logger.debug(f"Patches : {patches}")
            
        embeddings = self._load_features() if embs is None else embs

        def process_region(i, t):
            patches_feats = []
            for patch in patches[t][labels[i]]:
                embs = embeddings[t][patch[1] * self.final_grid_size[1] + patch[0]]
                embs = FeatureExtractor.normalize_tensor(embs, norm=norm)
                patches_feats.append(embs)
            aggregated = agg(torch.stack(patches_feats), dim=0)
            # If agg is torch.max, extract only the values
            if isinstance(aggregated, torch.return_types.max):
                aggregated = aggregated.values
            return aggregated
        
        res = joblib.Parallel(n_jobs=8, backend="threading")(
            joblib.delayed(process_region)(i, t) for i, t in enumerate(timepoints_shifted)
        )

        for i, r in enumerate(res):
            feats[i] = r

        return feats
        
    def _agg_patches_debug_view(self, v, region_mask, lab=None):
        """Debug function to visualize the patches and their embeddings."""        
        # Add region mask
        v.add_labels(
            region_mask,
            name=f"Region Mask {lab}",
            opacity=0.5,
            blending="translucent",
        )

    def _view_embeddings(self, embeddings, context: dict | None = None, **kwargs):
        # If this causes issues with augmented feature computation because
        # the extractor grid size depends on image size,
        # redefine it as appropriate in the subclass.
        # Use context to pass information as needed for the use case.
        
        # Currently redefined in :
        # - CoTrackerFeatures
        embs = embeddings.view(
            -1, self.final_grid_size[0], self.final_grid_size[1], self.hidden_state_size
        )
        return embs, self.final_grid_size
    
    def _agg_patches_exact(self, masks, timepoints, labels, agg=torch.mean, norm=True, embs=None):
        """Aggregates the embeddings of all patches that strictly belong to the mask."""
        try:
            n_regions = len(timepoints)
            timepoints_shifted = timepoints - timepoints.min()
        except ValueError:
            logger.error("Error: issue computing shifted timepoints.")
            logger.error(f"Regions: {len(timepoints)}")
            logger.error(f"Timepoints: {timepoints}")
            return torch.zeros(n_regions, self.hidden_state_size, device=self.device)
        
        feats = torch.zeros(n_regions, self.hidden_state_size, device=self.device)
        embeddings = self._load_features() if embs is None else embs
        embeddings, grid_size = self._view_embeddings(embeddings, context={"masks_shape": masks.shape})

        _T, H, W = masks.shape
        # assert embeddings.shape[0] == _T, f"Embeddings times {embeddings.shape} does not match masks times {_T}."
        grid_H, grid_W = grid_size
        scale_y = grid_H / H
        scale_x = grid_W / W
        
        if self._debug_view:
            import napari
            if napari.current_viewer() is None:
                v = napari.Viewer()
            else:
                v = napari.current_viewer()
            if "Masks" not in v.layers:
                v.add_labels(masks[0], name="Masks")
            if "Embeddings" not in v.layers:
                embs = embeddings.view(
                    -1, self.final_grid_size[0], self.final_grid_size[1], self.hidden_state_size
                )
                v.add_image(
                    embs.permute(3, 0, 1, 2).cpu().numpy(),
                    name="Embeddings",
                    colormap="inferno",
                )

        def process_region(i, t, masks):
            if masks.shape[0] == 1:
                masks = masks.squeeze(0)
                mask_reg = masks == labels[i]
            else:
                mask_reg = masks[t] == labels[i]
            if not np.any(mask_reg):
                logger.warning(f"No pixels found for region {labels[i]} at timepoint {t}.")
                # return torch.zeros(self.hidden_state_size, device=self.device) # small values to avoid zero divs etc.
                return torch.fill(self.hidden_state_size, 1e-8, device=self.device)

            y_idxs, x_idxs = np.nonzero(mask_reg)
            grid_y = np.clip((y_idxs * scale_y).astype(int), 0, grid_H - 1)
            grid_x = np.clip((x_idxs * scale_x).astype(int), 0, grid_W - 1)
            patch_embeddings = embeddings[timepoints[i]][grid_y, grid_x]
            # normalizing before the mean seems most effective
            patch_embeddings = FeatureExtractor.normalize_tensor(patch_embeddings, norm=norm)

            if self._debug_view:
                mask_emb = np.zeros((grid_H, grid_W), dtype=np.uint16)
                mask_emb[grid_y, grid_x] = labels[i]
                self._agg_patches_debug_view(v, mask_emb, labels[i])
            
            if patch_embeddings.shape[0] == 0:
                logger.warning(f"No mapped pixels for region {labels[i]} at timepoint {t}.")
                return torch.zeros(self.hidden_state_size, device=self.device)
            return agg(patch_embeddings, dim=0)

        # Parallel processing
        if self.parallel:
            res = joblib.Parallel(n_jobs=8, backend="threading")(
                joblib.delayed(process_region)(i, t, masks=masks) for i, t in enumerate(timepoints_shifted)
            )
            for i, r in enumerate(res):
                # If agg is torch.max or torch.median, extract only the values
                if isinstance(r, torch.return_types.max) or isinstance(r, torch.return_types.median):
                    feats[i] = r.values
                else:
                    feats[i] = r
        else:
            for i, t in enumerate(timepoints_shifted):
                feats[i] = process_region(i, t, masks)
        if self._debug_view:
            napari.run()
        # if norm:
        #     feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats
    
    def _exact_patch(self, masks, timepoints, labels, norm=False):
        """Returns all embeddings overlapping with the mask of each object."""
        try:
            n_regions = len(timepoints)
            timepoints_shifted = timepoints - timepoints.min()
        except ValueError:
            logger.error("Error: issue computing shifted timepoints.")
            logger.error(f"Regions: {len(timepoints)}")
            logger.error(f"Timepoints: {timepoints}")
            return torch.zeros(n_regions, self.hidden_state_size, device=self.device)

        feats = torch.zeros(n_regions, self.hidden_state_size, device=self.device)
        embeddings = self._load_features()
        embeddings = embeddings.view(
            -1, self.final_grid_size[0], self.final_grid_size[1], self.hidden_state_size
        )

        _T, H, W = masks.shape
        grid_H, grid_W = self.final_grid_size
        scale_y = grid_H / H
        scale_x = grid_W / W
        
        def process_region(i, t, masks):
            if masks.shape[0] == 1:  # single timepoint
                masks = masks.squeeze(0)
                mask_reg = masks == labels[i]
            else:  # all timepoints
                mask_reg = masks[t] == labels[i]
            if not np.any(mask_reg):
                logger.warning(f"No pixels found for region {labels[i]} at timepoint {t}.")
                return torch.zeros(self.hidden_state_size, device=self.device)

            y_idxs, x_idxs = np.nonzero(mask_reg)
            grid_y = np.clip((y_idxs * scale_y).astype(int), 0, grid_H - 1)
            grid_x = np.clip((x_idxs * scale_x).astype(int), 0, grid_W - 1)
            patch_embeddings = embeddings[timepoints[i]][grid_y, grid_x]            
            if patch_embeddings.shape[0] == 0:
                logger.warning(f"No mapped pixels for region {labels[i]} at timepoint {t}.")
                return torch.zeros(self.hidden_state_size, device=self.device)
            return patch_embeddings

        # Parallel processing
        res = joblib.Parallel(n_jobs=8, backend="threading")(
            joblib.delayed(process_region)(i, t, masks=masks) for i, t in enumerate(timepoints_shifted)
        )
        for i, r in enumerate(res):
            feats[i] = r
        # for i, t in enumerate(timepoints_shifted):
        #     feats[i] = process_region(i, t, masks)
        
        if norm:
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats
    
    def _save_features(self, features):  # , timepoint):
        """Saves the features to disk."""
        # save_path = self.save_path / f"{timepoint}_{self.model_name_path}_features.npy"
        self.embeddings = features
        if not self.do_save:
            return
        save_path = self.save_path / f"{self.model_name_path}_features.npy"
        np.save(save_path, features.cpu().numpy())
        assert save_path.exists(), f"Failed to save features to {save_path}"
    
    def _load_features(self):  # , timepoint):
        """Loads the features from disk."""
        # load_path = self.save_path / f"{timepoint}_{self.model_name_path}_features.npy"
        if self.embeddings is None:
            load_path = self.save_path / f"{self.model_name_path}_features.npy"
            if load_path.exists():
                features = np.load(load_path)
                assert features is not None, f"Failed to load features from {load_path}"
                if np.any(np.isnan(features)): 
                    raise RuntimeError(f"NaN values found in features loaded from {load_path}.")
                # check feature shape consistency
                if features.shape[1] != self.final_grid_size[0] * self.final_grid_size[1] or features.shape[2] != self.hidden_state_size:
                    logger.error(f"Saved embeddings found, but shape {features.shape} does not match expected shape {('n_frames', self.final_grid_size[0] * self.final_grid_size[1], self.hidden_state_size)}.")
                    logger.error("Embeddings will be recomputed.")
                    return None
                logger.info("Saved embeddings loaded.")
                self.embeddings = torch.tensor(features).to(self.device)
                return self.embeddings
            else:
                logger.info(f"No saved embeddings found at {load_path}. Features will be computed.")
                return None
        else:
            return self.embeddings
   
    def _check_missing_embeddings(self):
        """Checks if embeddings for the model already exist or are missing.
        
        Returns whether the embeddings need to be recomputed.
        """
        if self.force_recompute:
            return True
        try:
            features = self._load_features()
        except FileNotFoundError:
            return True
        if features is None:
            return True
        else:
            logger.info(f"Embeddings for {self.model_name} already exist. Skipping embedding computation.")
        return False


class FeatureExtractorAugWrapper:
    """Wrapper for the FeatureExtractor class to apply augmentations."""
    def __init__(
            self,
            extractor: FeatureExtractor,
            augmenter: "PretrainedAugmentations", 
            n_aug: int = 1,
            force_recompute: bool = False,
        ):
        self.extractor = extractor
        self.additional_features = extractor.additional_features
        self.n_aug = n_aug
        self.aug_pipeline = augmenter
        self.all_aug_features = {}  # n_aug -> {aug_id: {metadata, data}}
        # data -> {t: {lab: {"coords": coords, "features": features}}}
        self.image_shape_reference = {}
        
        self.extractor.force_recompute = True
        self.extractor.do_save = False  # do not save intermediate features (augmented image embeddings)
        # instead, we will save the augmented features + coordinates on a per-object basis in a zarr store
        self.extractor.do_normalize = False 
        self.extractor.parallel = False 
        # already parallelized, faster this way since it avoids the overhead of spawning 
        # many small processes within the parallelized augmentation pipeline

        self._zarr_sync = zarr.ProcessSynchronizer(str(self.get_save_path()) + ".sync")
        self.force_recompute = force_recompute
        
        self._debug_view = None
        
    def get_save_path(self):
        root_path = self.extractor.save_path / "aug"
        if not root_path.exists():
            root_path.mkdir(parents=True, exist_ok=True)
        return root_path / f"{self.extractor.model_name_path}_aug.zarr"
        
    def _check_existing(self):
        save_path = self.get_save_path()
        if not save_path.exists() or self.force_recompute:
            logger.debug(f"Augmentation zarr store {save_path} does not exist or force_recompute is True. Recomputing features.")
            return False, None, None
        logger.info(f"Loading existing features from {save_path}...")
        # root = zarr.open_group(str(save_path), mode="r")
        # existing_augs = [k for k in root.keys() if k.isdigit()]
        features_dict = self.load_all_features()
        existing_augs = list(features_dict.keys())
        logger.info("Done.")
        return True, existing_augs, features_dict
    
    def _compute(self, images, masks):
        """Computes the features for the images and masks."""
        images_shape = images.shape
        if len(images_shape) != 3:
            images_shape = images_shape[1:]  # remove batch dimension if present
        embs = self.extractor.precompute_image_embeddings(images, image_shape=images_shape)
        
        if self._debug_view is not None:
            embs = embs.cpu().numpy()
            logger.debug(f"Embeddings shape: {embs.shape}")
            embs = embs.reshape(-1, self.extractor.final_grid_size[0], self.extractor.final_grid_size[1], self.extractor.hidden_state_size)
            embs = np.moveaxis(embs, 3, 0)
            self._debug_view.add_image(embs, name="Embeddings", colormap="inferno")
            self._debug_view.add_image(images.cpu().numpy(), name="Images", colormap="viridis")
            self._debug_view.add_labels(masks.cpu().numpy(), name="Masks")

        images, masks = images.cpu().numpy(), masks.cpu().numpy()
        # features = wrfeat.WRAugPretrainedFeatures.from_mask_img(
        #     img=images,
        #     mask=masks,
        #     feature_extractor=self.extractor,
        #     t_start=0,
        #     additional_properties=self.extractor.additional_features,
        # )
        features = [
            wrfeat.WRAugPretrainedFeatures.from_mask_img(
                # embeddings=embs,
                img=img[np.newaxis], 
                mask=mask[np.newaxis], 
                feature_extractor=self.extractor, 
                t_start=t, 
                additional_properties=self.extractor.additional_features
            )
            for t, (mask, img) in tqdm(
                enumerate(zip(masks, images)), desc="Computing features...", total=len(masks), leave=False
            )  # if t == 10 debug
        ]
        features_dict = {t: v for f in features for t, v in f.to_dict().items()}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return features_dict
    
    def _compute_original(self, images, masks):
        """Computes the original features for the images and masks."""
        images, masks = self.aug_pipeline.preprocess(images, masks, normalize_func=self.extractor.normalize_array)
        orig_feat_dict = self._compute(images, masks)
        self.image_shape_reference[0] = images.shape[-2:]
        return orig_feat_dict
    
    def _compute_augmented(self, images, masks, n):
        images, masks = self.aug_pipeline.preprocess(images, masks, normalize_func=self.extractor.normalize_array)
        aug_images, aug_masks, aug_record = self.aug_pipeline(images, masks)
        
        # check for NaNs
        if torch.isnan(aug_images).any() or torch.isnan(aug_masks).any():
            raise RuntimeError("NaN values found in augmented images or masks.")
        
        im_shape, masks_shape = aug_images.shape, aug_masks.shape
        assert im_shape == masks_shape, f"Augmented images shape {im_shape} does not match augmented masks shape {masks_shape}."
        if im_shape[-2:] != self.extractor.orig_image_size:
            # if isinstance(self.extractor, TAPFeatures):
            # self.extractor.final_grid_size = (im_shape[-2], im_shape[-1]) # TAP features have same dims as images 
            if isinstance(self.extractor, CoTrackerFeatures):
                stride = self.extractor.model.stride
                self.extractor.final_grid_size = (im_shape[-2] // stride, im_shape[-1] // stride)
            if im_shape[-1] == 0 or im_shape[-2] == 0:
                raise ValueError(f"Augmented images have invalid shape {im_shape}. Cannot extract features.")
            self.extractor.orig_image_size = im_shape[-2:]
        
        aug_feat_dict = self._compute(aug_images, aug_masks)
        self.image_shape_reference[n] = aug_images.shape[-2:]
        return aug_feat_dict, aug_record
    
    def _process_aug(self, n, images, masks, existing_aug_ids, existing_features_dict):
        try:
            if str(f"{n + 1}") in existing_aug_ids:
                logger.info(f"Augmentation {n + 1} already exists. Skipping computation.")
                aug_feat_dict = existing_features_dict[str(n + 1)]["data"]
                aug_record = existing_features_dict[str(n + 1)]["metadata"]
            else:
                aug_feat_dict, aug_record = self._compute_augmented(images, masks, n=n + 1)
            result = {
                "n": n,
                "metadata": aug_record,
                "data": aug_feat_dict,
                # "should_save": str(n + 1) not in existing_aug_ids,
            }
            if str(n + 1) not in existing_aug_ids:
                self._save_features(n + 1, result)
            return result
        except Exception as e:
            logger.error(f"Error processing augmentation {n + 1}: {e}")
            raise e
    
    def compute_all_features(self, images, masks, clear_mem=True, n_workers=8) -> dict:
        """Augments the images and masks, computes the embeddings, and saves features incrementally."""
        # check existing features
        present, existing_augs, existing_features_dict = self._check_existing()
        save_path = self.get_save_path()
        if present:
            logger.debug(f"Saved features found at {save_path}.")
            if len(existing_augs) == self.n_aug + 1:
                logger.info(f"All {self.n_aug} augmentations + original already exist. Loading existing features.")
                self.all_aug_features = existing_features_dict
                return self.all_aug_features
            else:
                logger.info("No existing augmentations found.")
        logger.debug(f"Existing augmentations: {existing_augs}")
        if existing_augs is None:
            existing_aug_ids = []
        else:
            existing_aug_ids = existing_augs   
        logger.debug(f"Existing augmentations IDs: {existing_aug_ids}")   
        
        if "0" not in existing_aug_ids:
            orig_feat_dict = self._compute_original(images, masks)
        else:
            logger.info("Original features already exist. Skipping computation for original features.")
            orig_feat_dict = existing_features_dict["0"]["data"]
            
        self.all_aug_features = {
            "0": {
                "data": orig_feat_dict,
                "metadata": {
                    "image_shape": images.shape,
                    }
            }
        }
        if "0" not in existing_aug_ids:
            self._save_features(0, self.all_aug_features["0"])
        
        disable_parallel = False
        if isinstance(self.extractor, CoTrackerFeatures) or isinstance(self.extractor, TAPFeatures) or isinstance(self.extractor, MicroSAMFeatures):
            # CoTrackerFeatures and TAP uses a different grid size for each image,
            # which requires a different approach to parallel processing.
            # As a quick fix, parallel processing is disabled
            # TODO make necessary changes to CoTrackerFeatures to allow parallel processing
            disable_parallel = True
            logger.debug(f"Disabling parallel processing for {self.extractor.__class__.__name__} due to variable grid size.")

        if n_workers == 0 or disable_parallel:
            for n in range(self.n_aug):
                res = self._process_aug(n, images, masks, existing_aug_ids, existing_features_dict)
                self.all_aug_features[str(n + 1)] = res
                # if res["should_save"]:
                # self._save_features(n + 1, self.all_aug_features[str(n + 1)])
        else:
            # joblib parallel processing
            results = joblib.Parallel(n_jobs=n_workers, backend="threading")(
                joblib.delayed(self._process_aug)(
                    n, images, masks, existing_aug_ids, existing_features_dict
                ) for n in range(self.n_aug)
            )
            for res in results:
                n = res["n"]
                self.all_aug_features[str(n + 1)] = {
                    "metadata": res["metadata"],
                    "data": res["data"],
                }
                # if res["should_save"]:
                # self._save_features(n + 1, self.all_aug_features[str(n + 1)])
                    
        if clear_mem:
            self.extractor.embeddings = self.extractor.embeddings.cpu()
            try:
                self.extractor.model = self.extractor.model.cpu()
            except AttributeError as e:
                logger.error(f"Model attribute not found: {e}. Skipping model transfer to CPU.")
            self.extractor = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return self.all_aug_features

    # def _create_feat_dict(self, labels, ts, coords, features):
    #     """Creates a dictionary with the augmented features."""
    #     aug_feat_dict = {}
    #     features = features.cpu().numpy()
    #     for i, (t, lab) in enumerate(zip(ts, labels)):
    #         t = int(t)
    #         lab = int(lab)
    #         if t not in aug_feat_dict:
    #             aug_feat_dict[t] = {}
    #         aug_feat_dict[t][lab] = {
    #             "coords": coords[i],
    #             "features": features[i],
    #         }
    #     return aug_feat_dict

    def _save_features(self, aug_id: int, aug_data: dict):
        """Saves the features for a specific augmentation to disk as zarr (fast, flat layout)."""
        import zarr

        save_path = self.get_save_path()
        compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)
        root = zarr.open_group(str(save_path), mode="a", synchronizer=self._zarr_sync)
        group_name = str(aug_id)
        if group_name in root:
            del root[group_name]
        group = root.create_group(group_name)
        group.attrs["metadata"] = json.dumps(aug_data.get("metadata", {}))

        coords_list = []
        t_list = []
        lab_list = []
        features_dict = {}

        for t, data in aug_data["data"].items():
            for lab, lab_data in data.items():
                coords_list.append(lab_data["coords"])
                t_list.append(t)
                lab_list.append(lab)
                for key, value in lab_data["features"].items():
                    if key not in features_dict:
                        features_dict[key] = []
                    features_dict[key].append(value)

        coords_arr = np.stack(coords_list)
        t_arr = np.array(t_list)
        lab_arr = np.array(lab_list)
        group.create_dataset("coords", data=coords_arr, compressor=compressor)
        group.create_dataset("timepoints", data=t_arr, compressor=compressor)
        group.create_dataset("labels", data=lab_arr, compressor=compressor)

        features_group = group.create_group("features")
        for key, values in features_dict.items():
            features_group.create_dataset(key, data=np.stack(values), compressor=compressor)

        # logger.debug(f"Augmented features for augmentation {aug_id} saved to {save_path}.")
    
    def load_all_features(self) -> dict:
        """Loads all features from disk."""
        save_path = self.get_save_path()
        if not save_path.exists():
            raise FileNotFoundError(f"Path {save_path} does not exist.")

        features = FeatureExtractorAugWrapper.load_features(
                save_path,
                additional_props=self.additional_features,
            )
        self.all_aug_features = features
        return features

    @staticmethod
    def load_features(path: str | Path, additional_props: str | None = None, n_jobs: int = 12) -> dict:
        """Loads the features for all augmentations from disk (flat zarr layout, parallelized)."""
        import joblib

        if not isinstance(path, Path):
            path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Path {path} does not exist.")

        if additional_props is not None:
            required_features = wrfeat._PROPERTIES[additional_props]
            if len(required_features) == 0:
                required_features = "pretrained_feats"
            else:
                required_features += ("pretrained_feats", )

        root = zarr.open_group(str(path), mode="r")
        aug_ids = [k for k in root.keys() if k.isdigit()]

        def _load_single(aug_id):
            missing = False
            if aug_id not in root:
                missing = True
            group = root[aug_id]
            try:
                metadata = json.loads(group.attrs["metadata"])
            except KeyError:
                metadata = None
            try:
                coords_arr = group["coords"][...]
                t_arr = group["timepoints"][...]
                lab_arr = group["labels"][...]
                features_group = group["features"]
                features_dict = {}
                for key in features_group.keys():
                    if additional_props is not None and key not in required_features:
                        continue
                    features_dict[key] = features_group[key][...]
                if additional_props is not None:
                    missing_keys = [k for k in required_features if k not in features_dict]
                    if missing_keys:
                        raise RuntimeError(
                            f"Missing required features {missing_keys} in augmentation {aug_id}. "
                            f"Please delete the cache at {path} and recompute the features."
                        )

                data = {}
                for i in range(len(t_arr)):
                    t = int(t_arr[i])
                    lab = int(lab_arr[i])
                    if t not in data:
                        data[t] = {}
                    feats = {k: features_dict[k][i] for k in features_dict}
                    data[t][lab] = {
                        "coords": coords_arr[i],
                        "features": feats,
                    }
                return aug_id, {"metadata": metadata, "data": data}
            except KeyError as e:
                logger.error(f"KeyError: {e} - Augmentation {aug_id} is missing some data. Skipping.")
                missing = True
            if missing:
                return aug_id, None
            raise RuntimeError(f"Augmentation {aug_id} could not be loaded. Missing data or invalid format.")

        results = joblib.Parallel(n_jobs=n_jobs, backend="threading")(
            joblib.delayed(_load_single)(aug_id) for aug_id in aug_ids
        )
        all_data = {aug_id: aug_data for aug_id, aug_data in results if aug_data is not None}
        return all_data
       

##############
@register_backbone("facebook/hiera-tiny-224-hf", 768)
class HieraFeatures(FeatureExtractor):
    model_name = "facebook/hiera-tiny-224-hf"

    def __init__(
        self, 
        image_size: tuple[int, int],
        save_path: str | Path,
        batch_size: int = 16,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        mode: PretrainedFeatsExtractionMode = "nearest_patch",
        **kwargs,
        ):
        super().__init__(image_size, save_path, batch_size, device, mode, **kwargs)
        # self.input_size = 224
        self.input_mul = 3
        self.input_size = int(self.input_mul * 224)
        self.final_grid_size = int(7 * self.input_mul)  # default is 7x7 grid
        self.n_channels = 3
        self.hidden_state_size = 768
        self.rescale_batches = False

        ##
        self.im_proc_kwargs["size"] = self.input_size
        ##
        self.image_processor = AutoImageProcessor.from_pretrained(self.model_name)
        config = HieraConfig.from_pretrained(self.model_name)
        config.image_size = [self.input_size[0], self.input_size[1]]
        # logger.debug(f"Config: {config}")
        # self.model = HieraModel.from_pretrained(self.model_name)
        # self.model.config.image_size = [self.input_size, self.input_size]
        self.model = HieraModel(config)
        self.model.to(self.device)
        # self.model.embeddings.patch_embeddings.num_patches = (self.input_size // self.model.config.patch_size[0]) ** 2
        # self.model.embeddings.position_embeddings = torch.nn.Parameter(
        #     torch.zeros(1, self.model.embeddings.patch_embeddings.num_patches + 1, self.hidden_state_size)
        # )
        # self.model.embeddings.position_ids = torch.arange(0, self.model.embeddings.patch_embeddings.num_patches + 1).unsqueeze(0)

    def _run_model(self, images, **kwargs) -> torch.Tensor:
        """Extracts embeddings from the model."""
        # images = self._normalize_batch(images)
        inputs = self.image_processor(images, **self.im_proc_kwargs).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        return outputs.last_hidden_state
    

@register_backbone("facebook/dinov2-base", 768)
class DinoV2Features(FeatureExtractor):
    model_name = "facebook/dinov2-base"

    def __init__(
        self, 
        image_size: tuple[int, int],
        save_path: str | Path,
        batch_size: int = 16,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        mode: PretrainedFeatsExtractionMode = "nearest_patch",
        **kwargs,
        ):
        super().__init__(image_size, save_path, batch_size, device, mode)
        self.input_size = 224
        self.final_grid_size = 16  # 16x16 grid
        self.n_channels = 3  # expects RGB images
        self.hidden_state_size = 768
        self.image_processor = AutoImageProcessor.from_pretrained(self.model_name)
        ##
        self.im_proc_kwargs["size"] = self.input_size
        ##
        self.model = AutoModel.from_pretrained(self.model_name)
        self.rescale_batches = False
        # config = Dinov2Config.from_pretrained(self.model_name)
        # config.image_size = self.input_size
        
        # self.model = Dinov2Model(config)
        # logger.info(f"Model from config: {self.model.config}")
        # logger.info(f"Pretrained model : {Dinov2Model.from_pretrained(self.model_name).config}")
        self.model.to(self.device)

    def _run_model(self, images, **kwargs) -> torch.Tensor:
        """Extracts embeddings from the model."""
        inputs = self.image_processor(images, **self.im_proc_kwargs).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # ignore the CLS token (not classifying)
        # this way we get only the patch embeddings
        # which are compatible with finding the relevant patches directly
        # in the rest of the code
        return outputs.last_hidden_state[:, 1:, :]
    

@register_backbone("facebook/sam-vit-base", 256)
class SAMFeatures(FeatureExtractor):
    model_name = "facebook/sam-vit-base"

    def __init__(
        self, 
        image_size: tuple[int, int],
        save_path: str | Path,
        batch_size: int = 4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        mode: PretrainedFeatsExtractionMode = "nearest_patch",
        **kwargs,
        ):
        super().__init__(image_size, save_path, batch_size, device, mode)
        self.input_size = 1024
        self.final_grid_size = 64  # 64x64 grid
        self.n_channels = 3
        self.hidden_state_size = 256
        self.image_processor = SamProcessor.from_pretrained(self.model_name)
        self.model = SamModel.from_pretrained(self.model_name)
        self.rescale_batches = False
        
        self.model.to(self.device)

    def _run_model(self, images, **kwargs) -> torch.Tensor:
        """Extracts embeddings from the model."""
        inputs = self.image_processor(images, **self.im_proc_kwargs).to(self.device)
        outputs = self.model.get_image_embeddings(inputs['pixel_values'])
        B, N, H, W = outputs.shape
        return outputs.permute(0, 2, 3, 1).reshape(B, H * W, N)  # (B, grid_size**2, hidden_state_size)
        

@register_backbone("facebook/sam2-hiera-large", 256)
@register_backbone("facebook/sam2.1-hiera-base-plus", 256)
class SAM2Features(FeatureExtractor):
    model_name = "facebook/sam2.1-hiera-base-plus"
    
    def __init__(
        self, 
        image_size: tuple[int, int],
        save_path: str | Path,
        batch_size: int = 4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        mode: PretrainedFeatsExtractionMode = "nearest_patch",
        **kwargs,
        ):
        super().__init__(image_size, save_path, batch_size, device, mode)
        self.input_size = 1024
        self.final_grid_size = 64  # 64x64 grid
        self.n_channels = 3   
        self.hidden_state_size = 256        
        self.model = SAM2ImagePredictor.from_pretrained(self.model_name, device=self.device).model
        
        self.batch_return_type = "list[np.ndarray]"
        self.channel_first = True
        self.rescale_batches = False
        
        self._bb_feat_sizes = [
            (256, 256),
            (128, 128),
            (64, 64),
        ]
        if self.rescale_batches:
            print("Rescaling batches to [0, 255] range.")
        
    @torch.no_grad()
    def _run_model(self, images: list[np.ndarray], **kwargs) -> torch.Tensor:
        """Extracts embeddings from the model."""
        with torch.autocast(device_type=self.device), torch.inference_mode():
            images_ten = torch.stack([torch.tensor(image) for image in images]).to(self.device)
            # logger.debug(f"Image dtype: {images_ten.dtype}")
            # logger.debug(f"Image shape: {images_ten.shape}")
            # logger.debug(f"Image min :  {images_ten.min()}, max: {images_ten.max()}")
            images_ten = F.interpolate(images_ten, size=(self.input_size[0], self.input_size[1]), mode="bilinear", align_corners=False)
            # from torchvision.transforms.functional import resize
            # images_ten = resize(images_ten, size=(self.input_size, self.input_size))
            # images_ten = normalize(images_ten, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # out = self.model.image_encoder(images_ten)
            out = self.model.forward_image(images_ten)
            _, vision_feats, _, _ = self.model._prepare_backbone_features(out)
            # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
            if self.model_name != "facebook/sam2.1-hiera-base-plus/highres":
                if self.model.directly_add_no_mem_embed:
                    vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed
                feats = [
                    feat.permute(1, 2, 0).view(feat.shape[1], -1, *feat_size)
                    for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
                ][::-1]
                features = feats[-1]
        # features = self.model.set_image_batch(images)
        # features = self.model._features['image_embed']
        B, N, H, W = features.shape
        return features.permute(0, 2, 3, 1).reshape(B, H * W, N)  # (B, grid_size**2, hidden_state_size)


@register_backbone("facebook/sam2.1-hiera-base-plus/highres", 32)
class SAM2HighresFeatures(SAM2Features):
    model_name = "facebook/sam2.1-hiera-base-plus"
    
    def __init__(
        self, 
        image_size: tuple[int, int],
        save_path: str | Path,
        batch_size: int = 4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        mode: PretrainedFeatsExtractionMode = "nearest_patch",
        **kwargs,
        ):
        super().__init__(image_size, save_path, batch_size, device, mode)
        self.final_grid_size = 256  # 256x256 grid
        # self.final_grid_size = 128
        self.hidden_state_size = 32
        # self.hidden_state_size = 64
    
    @property
    def model_name_path(self):
        """Returns the model name for saving."""
        p = f"{self.model_name}/highres" 
        return p.replace("/", "-")

    def _run_model(self, images: list[np.ndarray], **kwargs) -> torch.Tensor:
        """Extracts embeddings from the model."""
        with torch.autocast(device_type=self.device), torch.inference_mode():
            images_ten = torch.stack([torch.tensor(image) for image in images]).to(self.device)
            # logger.debug(f"Image dtype: {images_ten.dtype}")
            # logger.debug(f"Image shape: {images_ten.shape}")
            # logger.debug(f"Image min :  {images_ten.min()}, max: {images_ten.max()}")
            images_ten = F.interpolate(images_ten, size=(self.input_size[0], self.input_size[1]), mode="bilinear", align_corners=False)
            out = self.model.forward_image(images_ten)
            backbone_out, _, _, _ = self.model._prepare_backbone_features(out)
            features = backbone_out["backbone_fpn"][0]  # (B, N, H, W)
        
        B, N, H, W = features.shape
        return features.permute(0, 2, 3, 1).reshape(B, H * W, N)  # (B, grid_size**2, hidden_state_size)


@register_backbone("facebookresearch/co-tracker", 128)
class CoTrackerFeatures(FeatureExtractor):
    model_name = "facebookresearch/co-tracker"

    def __init__(
        self, 
        image_size: tuple[int, int],
        save_path: str | Path,
        batch_size: int = 4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        mode: PretrainedFeatsExtractionMode = "nearest_patch",
        **kwargs,
        ):
        super().__init__(image_size, save_path, batch_size, device, mode)
        self.input_size = image_size
        cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
        self.model = cotracker.model
        self.model.to(device)
        self.final_grid_size = (image_size[0] // self.model.stride, image_size[1] // self.model.stride)
        self.hidden_state_size = 128
        self.n_channels = 3
        self.fmaps_chunk_size = 8
        
        self.batch_return_type = "list[np.ndarray]"

    def _view_embeddings(self, embeddings, context: dict, **kwargs):
            image_shape = context.get("masks_shape", None)
            H, W = image_shape[-2:]
            grid_size = (H // self.model.stride, W // self.model.stride)
            embs = embeddings.view(
                -1,
                *grid_size,
                self.hidden_state_size,
            )
            return embs, grid_size
    
    def precompute_image_embeddings(self, images, image_shape=None, **kwargs):  # , windows, window_size):
        """Precomputes embeddings for all images."""
        try:
            if image_shape is None:
                _, H, W = images.shape
                if H != self.input_size[0] or W != self.input_size[1]:
                    self.input_size = (H, W)
                    self.final_grid_size = (H // self.model.stride, W // self.model.stride)
                    logger.debug(f"Updated CoTracker input size: {self.input_size}, final grid size: {self.final_grid_size}")
                grid_size = self.final_grid_size
            else:
                H, W = image_shape[-2:]
                grid_size = (H // self.model.stride, W // self.model.stride)

            missing = self._check_missing_embeddings()
            all_embeddings = torch.zeros(len(images), grid_size[0] * grid_size[1], self.hidden_state_size, device=self.device)
            if missing:
                for ts, batches in tqdm(self._prepare_batches(images), total=len(images) // self.batch_size, desc="Computing embeddings", leave=False):
                    try:
                        embeddings = self._run_model(batches, image_shape, **kwargs)
                    except Exception as e:
                        breakpoint()
                        raise e
                    if torch.any(embeddings.isnan()):
                        raise RuntimeError("NaN values found in features.")
                    # logger.debug(f"Embeddings shape: {embeddings.shape}")
                    all_embeddings[ts] = embeddings.to(torch.float32)
                    assert embeddings.shape[-1] == self.hidden_state_size
                self.embeddings = all_embeddings
                self._save_features(all_embeddings)
            # logger.debug(f"Precomputed embeddings shape: {self.embeddings.shape}")
            return self.embeddings
        except Exception as e:
            logger.error(f"Error occurred while precomputing embeddings: {e}")
            raise e

    def _run_model(self, images: list[np.ndarray], image_shape=None, **kwargs) -> torch.Tensor:
        self.model.eval()
        x = torch.stack([torch.tensor(image) for image in images]).to(self.device)
        x = x.unsqueeze(0)  # B, T, C, H, W
        with torch.no_grad():
            B = x.shape[0]
            T = x.shape[1]
            C_ = x.shape[2]
            H, W = x.shape[3], x.shape[4]
            if T > self.batch_size:
                fmaps = []
                for t in range(0, T, self.fmaps_chunk_size):
                    video_chunk = x[:, t : t + self.fmaps_chunk_size]
                    fmaps_chunk = self.model.fnet(video_chunk.reshape(-1, C_, H, W))
                    T_chunk = video_chunk.shape[1]
                    C_chunk, H_chunk, W_chunk = fmaps_chunk.shape[1:]
                    fmaps.append(fmaps_chunk.reshape(B, T_chunk, C_chunk, H_chunk, W_chunk))
                fmaps = torch.cat(fmaps, dim=1).reshape(-1, C_chunk, H_chunk, W_chunk)
            else:
                fmaps = self.model.fnet(x.reshape(-1, C_, H, W))
            fmaps = fmaps.permute(0, 2, 3, 1)
            fmaps = fmaps / torch.sqrt(
                torch.maximum(
                    torch.sum(torch.square(fmaps), axis=-1, keepdims=True),
                    torch.tensor(1e-12, device=fmaps.device),
                )
            )
            fmaps = fmaps.permute(0, 3, 1, 2).reshape(
                B, -1, self.model.latent_dim, H // self.model.stride, W // self.model.stride
            )  # B, T, N, H', W'
            # end of original code
            fmaps = fmaps.permute(0, 1, 3, 4, 2).squeeze(0)  # T, H', W', N
            fmaps = fmaps.reshape(
                fmaps.shape[0], fmaps.shape[1] * fmaps.shape[2], fmaps.shape[3]
            )  # T, H' * W', N
            return fmaps


@register_backbone("microsam/vit_b_lm", 256)
@register_backbone("microsam/vit_l_lm", 256)
class MicroSAMFeatures(FeatureExtractor):
    model_name = "microsam/vit_b_lm"
    
    def __init__(
        self, 
        image_size: tuple[int, int],
        save_path: str | Path,
        batch_size: int = 4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        mode: PretrainedFeatsExtractionMode = "nearest_patch",
        **kwargs,
        ):
        if not MICRO_SAM_AVAILABLE:
            raise ImportError("microSAM is not available. Please install it following the instructions in the documentation.")
        super().__init__(image_size, save_path, batch_size, device, mode)
        self.input_size = 1024
        self.final_grid_size = 64
        self.n_channels = 3
        self.hidden_state_size = 256
        model_name = self.model_name.split("/")[-1]
        self.model = get_microsam_model(model_name, device=self.device)
        
        self.batch_return_type = "list[np.ndarray]"
        self.channel_first = False
        self.rescale_batches = False

    def _run_model(self, images: list[np.ndarray], **kwargs) -> torch.Tensor:
        """Extracts embeddings from the model."""
        embeddings = []
        for image in images:
            # logger.debug(f"Image shape: {image.shape}")
            self.model.set_image(image)
            embedding = self.model.get_image_embedding()  # (1, hidden_state_size, grid_size, grid_size)
            B, N, H, W = embedding.shape
            embedding = embedding.permute(0, 2, 3, 1).reshape(B, H * W, N)
            embeddings.append(embedding)
        out = torch.stack(embeddings).squeeze()  # (B, grid_size**2, hidden_state_size)
        if len(out.shape) == 2:
            out = out.unsqueeze(0)
        return out


@register_backbone("weigertlab/tarrow", 32)
class TAPFeatures(FeatureExtractor):
    model_name = "weigertlab/tarrow"

    def __init__(
        self,
        model_folder: str,
        image_size: tuple[int, int],
        save_path: str | Path,
        batch_size: int = 2,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        mode: PretrainedFeatsExtractionMode = "nearest_patch",
        **kwargs,
        ):
        super().__init__(image_size, save_path, batch_size, device, mode)
        self._final_grid_size = (self.orig_image_size[0], self.orig_image_size[1])  
        self.input_size = self.final_grid_size
        self.n_channels = 1
        
        self.model_folder = model_folder
        self.full_model, self.hidden_state_size = self._load_model_from_path(model_folder)
        self.full_model.to(device)
        self.full_model.eval()
        AVAILABLE_PRETRAINED_BACKBONES["weigertlab/tarrow"]["feat_dim"] = self.hidden_state_size
        self.model = self.full_model.backbone
        
        self.batch_return_type = "torch.Tensor"
        self.channel_first = False
        self.rescale_batches = False
        self.normalize_embeddings = False
    
        # TODO clear full model from memory
    
    @property
    def final_grid_size(self) -> tuple[int, int]:
        return self._final_grid_size
    
    @final_grid_size.setter
    def final_grid_size(self, value: tuple[int, int]):
        """Sets the final grid size and updates the model's input size."""
        self._final_grid_size = value
        self.orig_image_size = value
        self.input_size = value
        self.model.input_size = value
        self.model_patch_size = (1, 1)
        
    def _set_model_patch_size(self):
        pass
        
    @staticmethod
    def _load_model_from_path(model_folder: str):
        """Loads the model from the folder."""
        if not os.path.exists(model_folder):
            raise FileNotFoundError(f"Model folder {model_folder} does not exist.")
        model = TimeArrowNet.from_folder(model_folder, from_state_dict=True)
        feat_dim = model.bb_n_feat
        return model, feat_dim
    
    def normalize_array(self, b):
        images_batch = tap_normalize(b)
        images_batch = torch.from_numpy(images_batch).to(torch.float32)  # T, H, W
        return images_batch
        
    def _prepare_batches(self, images):
        if self.do_normalize:
            images = self.normalize_array(images)
        images = images.unsqueeze(1)  # T, C, H, W
        return images

    def _run_model(self, images: list[np.ndarray], **kwargs) -> torch.Tensor:
        """Extracts embeddings from the model."""
        features = []
        ts = images.shape[0]
        im_shape = tuple(images.shape[-2:])
        self.orig_image_size = im_shape
        self.final_grid_size = im_shape
        with torch.no_grad():
            for i in tqdm(range(0, len(images), self.batch_size), desc="Computing TAP features", leave=False):
                batch = images[i : i + self.batch_size]
                batch = batch.to(self.device)
                out = self.model(batch)
                features.append(out)
        
        features = torch.cat(features, dim=0).cpu()
        
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        features = features.moveaxis(1, 3)  # (T, H, W, N)
        features = features.reshape(ts, self.final_grid_size[0] * self.final_grid_size[1], self.hidden_state_size)  # (T, grid_size**2, N)
        return features
    
    def precompute_image_embeddings(self, images, **kwargs):
        # missing = self._check_missing_embeddings()
        if images.shape[-2:] != self.orig_image_size:
            self.orig_image_size = images.shape[-2:]
            self.final_grid_size = images.shape[-2:]
        batches = self._prepare_batches(images)
        self.embeddings = self._run_model(batches, **kwargs)
        # self._save_features(self.embeddings)
        return self.embeddings


@register_backbone("mouseland/cellpose-sam", 192)
class CellposeSAMFeatures(FeatureExtractor):
    model_name = "mouseland/cellpose-sam"
    
    def __init__(
        self, 
        image_size: tuple[int, int],
        save_path: str | Path,
        batch_size: int = 4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        mode: PretrainedFeatsExtractionMode = "nearest_patch",
        **kwargs,
        ):
        if not CELLPOSE_AVAILABLE:
            raise ImportError("Cellpose is not available. Please install it following the instructions in the documentation.")
        super().__init__(image_size, save_path, batch_size, device, mode)
        self.input_size = 256
        self.final_grid_size = 32  # 32x32 grid
        self.n_channels = 3
        self.hidden_state_size = 192
        self.model = CellposeSAM()
        self.model.to(self.device)
        self.model.eval()

        self.batch_return_type = "np.ndarray"
        self.channel_first = False
        self.rescale_batches = False
        
    def normalize_array(self, images_batch):
        batch = torch.zeros(
            (*tuple(images_batch.shape), self.n_channels),  # add a channel dimension
            dtype=torch.float32
        )
        for n, b in enumerate(images_batch):
            if isinstance(b, torch.Tensor):
                b = b.cpu().numpy()
            b_ = cp_transforms.convert_image(
                b,
                channel_axis=None,
                z_axis=None,
                do_3D=False
            )
            b_ = cp_transforms.normalize_img(b_)
            b_ = torch.from_numpy(b_).to(torch.float32)
            batch[n] = b_
        logger.debug(f"Cellpose SAM batch shape: {batch.shape}")
        return batch[..., 0]  # keep only a single copy of the channel
        
    def _prepare_batches(self, images):
        if self.do_normalize:
            images = self.normalize_array(images)
        for i in range(0, len(images), self.batch_size):
            end = i + self.batch_size
            end = min(end, len(images))
            batch = images[i:end]  # (B, H, W)
            ts = range(i, end)
            # if self.do_normalize:
            # batch = self.normalize_array(batch)
            if len(batch.shape) == 3:
                batch = batch.unsqueeze(1)
                batch = batch.repeat(1, 3, 1, 1)
            batch = batch.to(self.device)
            yield ts, batch

    def _run_model(self, images_batch: np.ndarray, **kwargs) -> torch.Tensor:
        embeddings = []
        
        with torch.no_grad():
            
            b = F.interpolate(images_batch, size=(self.input_size[0], self.input_size[1]), mode="bilinear", align_corners=False)
            
            x = self.model.encoder.patch_embed(b)
            if self.model.encoder.pos_embed is not None:
                x = x + self.model.encoder.pos_embed
            for i, blk in enumerate(self.model.encoder.blocks):
                x = blk(x)
            x = self.model.encoder.neck(x.permute(0, 3, 1, 2))
            x = self.model.out(x)  # (B, N, H, W)
            embeddings.append(x)
                
        embeddings = torch.cat(embeddings, dim=0)  # (T, N, H, W)
        embeddings = embeddings.moveaxis(1, 3)  # (T, H, W, N)
        embeddings = embeddings.reshape(-1, self.final_grid_size[0] * self.final_grid_size[1], self.hidden_state_size)  # (T, grid_size**2, N)
        return embeddings


@register_backbone("debug/encoded_labels", 64)
class EncodedLabelsFeatures(FeatureExtractor):
    """Encodes labels to 32 dimensions. Should work as a "perfect" feature extractor that uses GT labels as a sanity check."""
    model_name = "debug/encoded_labels"

    def __init__(
        self, 
        image_size: tuple[int, int],
        save_path: str | Path,
        batch_size: int = 4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        mode: PretrainedFeatsExtractionMode = "nearest_patch",
        **kwargs,
        ):
        super().__init__(image_size, save_path, batch_size, device, mode)
        self.input_size = 1024
        self.final_grid_size = 32
        self.n_channels = 1
        self.hidden_state_size = 64

    def _run_model(self, images, **kwargs) -> torch.Tensor:
        """Extracts embeddings from the model."""
        pass

    def precompute_image_embeddings(self, images, **kwargs):
        pass
    
    def _encode_labels(self, labels):
        """Encodes the labels to D dimensions."""
        features = np.zeros((labels.shape[0], self.hidden_state_size), dtype=np.float32)
        for i in range(labels.shape[0]):
            label = labels[i]
            features[i] = np.array([int(x) for x in np.binary_repr(label, width=self.hidden_state_size)], dtype=np.float32)
        
        features = torch.from_numpy(features).to(self.device)
        return features
    
    def compute_region_features(self, labels=None, **kwargs):
        return self._encode_labels(labels)  # (n_labels, self.hidden_state_size)


@register_backbone("debug/random", 128)
class RandomFeatures(FeatureExtractor):
    model_name = "debug/random"

    def __init__(
        self, 
        image_size: tuple[int, int],
        save_path: str | Path,
        batch_size: int = 4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        mode: PretrainedFeatsExtractionMode = "nearest_patch",
        seed: int = 42,
        **kwargs,
        ):
        super().__init__(image_size, save_path, batch_size, device, mode)
        self.input_size = 1024
        # self.final_grid_size = self.orig_image_size
        self.final_grid_size = 128
        self.n_channels = 3
        self.hidden_state_size = 128
        
        self._seed = seed if seed is not None else 42
        self.device = "cpu"
        self._generator = torch.Generator(device="cpu").manual_seed(self._seed)
        
        self.do_save = False
        
    def _run_model(self, images, **kwargs) -> torch.Tensor:
        """Extracts embeddings from the model."""
        # Normal distribution
        # return torch.randn(len(images), self.final_grid_size**2, self.hidden_state_size, generator=self._generator).to(self.device)
        # Uniform distribution
        feats = torch.rand(
                len(images),
                self.final_grid_size[0] * self.final_grid_size[1], 
                self.hidden_state_size, 
                generator=self._generator,
                dtype=torch.float32
            )
        # feats = feats * 4 - 2  # [-2, 2]
        return feats.to("cpu")


FeatureExtractor._available_backbones = AVAILABLE_PRETRAINED_BACKBONES


# Embeddings post-processing

import pickle

from sklearn.decomposition import PCA


class EmbeddingsPCACompression:
    
    def __init__(self, original_model_name: str, n_components: int = 15, save_path: str | Path | None = None):
        self.original_model_name = original_model_name
        self.n_components = n_components
        self.save_path = save_path / "pca_model.pkl" if save_path is not None else None
        self.pca = PCA(n_components=n_components)
        self.max_frames = 1500
        self.random_state = 42
        self.pca.random_state = self.random_state
        self.generator = np.random.default_rng(self.random_state)
    
    @classmethod
    def from_pretrained_cfg(cls, cfg: PretrainedFeatureExtractorConfig):
        return cls(
            original_model_name=cfg.model_name.replace("/", "-"),
            n_components=cfg.pca_components,
            save_path=cfg.pca_preprocessor_path
        )
    
    def fit(self, X: np.ndarray):
        """Fits the PCA model to the embeddings."""
        self.pca.fit(X)
        
        if self.save_path is not None:
            if not self.save_path.parents[0].exists():
                self.save_path.parents[0].mkdir(parents=True, exist_ok=True)
            with open(self.save_path, 'wb') as f:
                pickle.dump(self.pca, f)
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transforms the embeddings using the fitted PCA model."""
        return self.pca.transform(X)
    
    def load_from_file(self, path: str | Path):
        """Loads the PCA model from a file."""
        if isinstance(path, str):
            path = Path(path)
        path = path / "pca_model.pkl" if path.suffix != ".pkl" else path
        with open(path, 'rb') as f:
            self.pca = pickle.load(f)
        logger.info(f"Loaded PCA model from {path}.")
    
    def fit_on_embeddings(self, embeddings_source_folders: list[str | Path]):
        """Fits the PCA model to the embeddings loaded from a file."""
        embeddings = []
        N_samples = 0
        
        embeddings_paths = []
        for folder in embeddings_source_folders:
            for file in Path(folder).rglob("*.npy"):
                if self.original_model_name in file.name:
                    embeddings_paths.append(file)
        
        if len(embeddings_paths) == 0:
            return
        
        embeddings_paths = self.generator.permutation(embeddings_paths)
        logger.info(f"Fitting PCA model on {len(embeddings_paths)} embeddings files.")
        logger.info("Files :")
        for p in embeddings_paths:
            logger.info(f" - {p}")
        logger.info("*" * 50)
        
        for path in embeddings_paths:
            if isinstance(path, str):
                path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"File {path} does not exist.")
            emb = np.load(path)
            N_samples += emb.shape[0]
            if N_samples > self.max_frames:
                logger.info(f"Amount of loaded frames exceeds {self.max_frames} limit for PCA computation.")
                break
            else:
                embeddings.append(emb)
        
        embeddings = np.concatenate(embeddings, axis=0)
        embeddings = embeddings.reshape(-1, embeddings.shape[-1])
        self.fit(embeddings)
        logger.info(f"Fitted PCA model with {self.n_components} components on {N_samples} frames.")
        