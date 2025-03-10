import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import torch
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

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# forward declarations for indexing
HieraFeatures = None
DinoV2Features = None
SAMFeatures = None
SAM2Features = None

# Updated with actual class after each definition
# See register_backbone decorator
AVAILABLE_PRETRAINED_BACKBONES = {}

PretrainedFeatsExtractionMode = Literal[
    # "exact_patch",  # Uses the image patch centered on the detection for embedding
    "nearest_patch",  # Runs on whole image, then finds the nearest patch to the detection in the embedding
    "mean_patches"  # Runs on whole image, then averages the embeddings of all patches that intersect with the detection
]

PretrainBackboneType = Literal[  # cannot unpack this directly in python < 3.11 so it has to be copied
    "facebook/hiera-tiny-224-hf",  # 768
    "facebook/dinov2-base",  # 768
    "facebook/sam-vit-base",  # 256
    "facebook/sam2-hiera-large",  # 256
    "facebook/sam2.1-hiera-base-plus",  # 256
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
# Currently meant to wrap any transformers model
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
        ):
        # Image processor extra args
        self.im_proc_kwargs = {
            "do_rescale": False,
            "do_normalize": False,
            "do_resize": True,
            "return_tensors": "pt",
            "do_center_crop": False,
        }
        # Model specs
        self.model = None
        self._input_size: int = None
        self._final_grid_size: int = None
        self.n_channels: int = None
        self.hidden_state_size: int = None
        self.model_patch_size: int = None
        # Data specs
        self.orig_image_size = image_size
        self.orig_n_channels = 1
        self.do_rescale = True
        self.channel_first = True
        self.batch_return_type: Literal["list[np.ndarray]", "np.ndarray"] = "np.ndarray"
        # Parameters for embedding extraction
        self.batch_size = batch_size
        self.device = device
        self.mode = mode
        # Saving parameters
        self.save_path: str | Path = save_path
        self.embeddings = None
        
        if not isinstance(self.save_path, Path):
            self.save_path = Path(self.save_path)
        
        if not self.save_path.exists():
            self.save_path.mkdir(parents=False, exist_ok=True)
    
    @property
    def input_size(self):
        return self._input_size
    
    @input_size.setter
    def input_size(self, value: int):
        self._input_size = value
        self._set_model_patch_size()
        
    @property
    def final_grid_size(self):
        return self._final_grid_size
    
    @final_grid_size.setter
    def final_grid_size(self, value: int):
        self._final_grid_size = value
        self._set_model_patch_size()
        
    @property
    def model_name_path(self):
        return self.model_name.replace("/", "-")
    
    @classmethod
    def from_model_name(cls, 
                        model_name: PretrainBackboneType, 
                        image_shape: tuple[int, int], 
                        save_path: str | Path, 
                        device: torch.device = "cuda" if torch.cuda.is_available() else "cpu",
                        mode="nearest_patch"
                        # mode="mean_patches"
                        ):
        cls._available_backbones = AVAILABLE_PRETRAINED_BACKBONES
        if model_name not in cls._available_backbones:
            raise ValueError(f"Model {model_name} is not available for feature extraction.")
        logger.info(f"Using model {model_name} with mode {mode} for pretrained feature extraction.")
        backbone = cls._available_backbones[model_name]["class"]
        backbone.model_name = model_name
        return backbone(image_shape, save_path, device=device, mode=mode)
    
    def _set_model_patch_size(self):
        if self.final_grid_size is not None and self.input_size is not None:
            self.model_patch_size = self.input_size // self.final_grid_size
            # if not self.input_size % self.final_grid_size == 0:
            # raise ValueError("The input size must be divisible by the final grid size.")

    def forward(
            self, 
            coords,
            masks=None, 
            timepoints=None,
            labels=None,
        ) -> torch.Tensor:  # (n_regions, embedding_size)
        """Extracts embeddings from the model.
        
        Args:
        - coords (np.ndarray): Coordinates of the detections.
        - masks (np.ndarray): Masks where each region has a unique label for each timepoint.
        - timepoints (np.ndarray): For each region, contains the corresponding timepoint.
        - labels (np.ndarray): Unique labels of the regions.
        """
        feats = torch.zeros(len(coords), self.hidden_state_size, device=self.device)
        
        if self.mode == "nearest_patch":
            feats = self._nearest_patches(coords)    
        elif self.mode == "mean_patches":
            if masks is None or labels is None or timepoints is None:
                raise ValueError("Masks and labels must be provided for mean patch mode.")
            feats = self._mean_patches(masks, timepoints, labels)
        else:
            raise NotImplementedError(f"Mode {self.mode} is not implemented.")
        
        assert feats.shape == (len(coords), self.hidden_state_size)
        return feats  # (n_regions, embedding_size)
    
    def precompute_region_embeddings(self, images, windows, window_size):
        """Precomputes embeddings for all images."""
        missing = self._check_existing_embeddings(len(images))
        all_embeddings = torch.zeros(len(images), self.final_grid_size**2, self.hidden_state_size, device=self.device)
        if missing:
            for ts, batches in tqdm(self._prepare_batches(images), total=len(images) // self.batch_size, desc="Computing embeddings"):
                embeddings = self._run_model(batches)
                # logger.debug(f"Embeddings shape: {embeddings.shape}")
                all_embeddings[ts] = embeddings
            self._save_features(all_embeddings)

        region_embeddings = {}
        for n in tqdm(range(0, len(windows), window_size), desc="Extracting region embeddings"):
            self._extract_region_embeddings(region_embeddings, windows[n], n)

        remaining = len(images) - len(region_embeddings)
        logger.debug(f"Remaining frames: {remaining}")
        if remaining > 0:
            self._extract_region_embeddings(region_embeddings, windows[-1], len(images) - remaining, remaining)

        self.embeddings = None  # clear embeddings from memory
        return region_embeddings

    def _extract_region_embeddings(self, region_embeddings, window, start_index, remaining=None):
        window_coords = window["coords"]
        window_timepoints = window["timepoints"]
        window_masks = window["mask"]
        window_labels = window["labels"]
        n_regions_per_frame = np.unique(window_timepoints, return_counts=True)[1]
        tot_regions = n_regions_per_frame.sum()
        coords_txy = np.concatenate((window_timepoints[:, None], window_coords), axis=-1)
        if coords_txy.shape[0] != tot_regions:
            raise RuntimeError(f"Number of coords ({coords_txy.shape[0]}) does not match the number of coordinates ({window_timepoints.shape[0]}).")
        features = self.forward(
            coords=coords_txy,
            masks=window_masks,
            timepoints=window_timepoints,
            labels=window_labels,
        )  # (n_regions, embedding_size)
        if tot_regions != features.shape[0]:
            raise RuntimeError(f"Number of regions ({n_regions_per_frame}) does not match the number of embeddings ({features.shape[0]}).")
        for i in range(remaining or len(n_regions_per_frame)):
            # if computing remaining frames' embeddings, start from the end
            obj_per_frame = n_regions_per_frame[-i - 1] if remaining else n_regions_per_frame[i]
            frame_index = start_index + i if not remaining else np.max(window_timepoints) - i
            # logger.debug(f"Frame {frame_index} has {obj_per_frame} objects.")
            region_embeddings[frame_index] = features[:obj_per_frame]
            features = features[obj_per_frame:]
    
    @abstractmethod
    def _run_model(self, images) -> torch.Tensor:  # must return (B, grid_size**2, hidden_state_size)
        """Extracts embeddings from the model."""
        pass
    
    def _rescale_batch(self, b):
        for i, im in enumerate(b):
            # b[i] = (im - im.min()) / (im.max() - im.min())
            p1, p99 = np.percentile(im, (1, 99.8))
            b[i] = (im - p1) / (p99 - p1)
            b[i] = np.clip(b[i], 0, 1)
        return b
    
    def _prepare_batches(self, images):
        """Prepares batches of images for embedding extraction."""
        for i in range(0, len(images), self.batch_size):
            end = i + self.batch_size
            end = len(images) if end > len(images) else end
            batch = np.expand_dims(images[i:end], axis=1)  # (B, C, H, W)
            
            # required by AutoImageProcessor (PIL Image needs [0, 1] range)
            if self.do_rescale:
                batch = self._rescale_batch(batch)  # TODO check if this is okay to do 
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
    
    def _map_coords_to_model_grid(self, coords):
        scale_x = self.input_size / self.orig_image_size[0]
        scale_y = self.input_size / self.orig_image_size[1]
        coords = np.array(coords)
        patch_x = (coords[:, 1] * scale_x).astype(int)
        patch_y = (coords[:, 2] * scale_y).astype(int)
        patch_coords = np.column_stack((coords[:, 0], patch_x, patch_y))
        return patch_coords
    
    def _find_nearest_cell(self, patch_coords):
        x_idxs = patch_coords[:, 1] // self.model_patch_size
        y_idxs = patch_coords[:, 2] // self.model_patch_size
        patch_idxs = np.column_stack((patch_coords[:, 0], x_idxs, y_idxs))
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
        patch_height, patch_width = image_mask.shape[0] // self.final_grid_size, image_mask.shape[1] // self.final_grid_size
        regions = regionprops(image_mask)
        return self._find_bbox_cells(regions, patch_height, patch_width)
    
    def _nearest_patches(self, coords):
        """Finds the nearest patches to the detections in the embedding."""
        # find coordinate patches from detections
        patch_coords = self._map_coords_to_model_grid(coords)
        patch_idxs = self._find_nearest_cell(patch_coords)
        # logger.debug(f"Patch indices: {patch_idxs}")

        # load the embeddings and extract the relevant ones
        feats = torch.zeros(len(coords), self.hidden_state_size, device=self.device)
        indices = [y * self.final_grid_size + x for _, y, x in patch_idxs]

        unique_timepoints = list(set(t for t, _, _ in patch_idxs))
        # logger.debug(f"Unique timepoints: {unique_timepoints}")
        embeddings = self._load_features()
        # logger.debug(f"Embeddings shape: {embeddings.shape}")
        embeddings_dict = {t: embeddings[t] for t in unique_timepoints}

        for i, (t, _, _) in enumerate(patch_idxs):
            feats[i] = embeddings_dict[t][indices[i]]
            
        return feats
    
    # @average_time_decorator
    def _mean_patches(self, masks, timepoints, labels, agg=torch.mean):
        """Averages the embeddings of all patches that intersect with the detection.
        
        Args:
            - masks (np.ndarray): Masks where each region has a unique label (t x H x W).
            - timepoints (np.ndarray): For each region, contains the corresponding timepoint. (n_regions)
            - labels (np.ndarray): Unique labels of the regions. (n_regions)
            - agg (callable): Aggregation function to use for averaging the embeddings.
        """
        n_regions = len(timepoints)
        timepoints_shifted = timepoints - timepoints.min()
        feats = torch.zeros(n_regions, self.hidden_state_size, device=self.device)
        patches = []
        times = np.unique(timepoints_shifted)
        patches_res = joblib.Parallel(n_jobs=8, backend="threading")(
            joblib.delayed(self._find_patches_for_masks)(masks[t]) for t in times
        )   
        patches = {t: patch for t, patch in zip(times, patches_res)}
        # logger.debug(f"Patches : {patches}")
            
        embeddings = self._load_features()    

        def process_region(i, t):
            patches_feats = []
            for patch in patches[t][labels[i]]:
                patches_feats.append(embeddings[t][patch[1] * self.final_grid_size + patch[0]])
            return agg(torch.stack(patches_feats), dim=0)
        
        res = joblib.Parallel(n_jobs=8, backend="threading")(
            joblib.delayed(process_region)(i, t) for i, t in enumerate(timepoints_shifted)
        )

        for i, r in enumerate(res):
            feats[i] = r

        return feats

    def _exact_patch(self, imgs, masks, coords):
        """Uses the image patch centered on the detection for embedding."""
        raise NotImplementedError()
    
    def _save_features(self, features):  # , timepoint):
        """Saves the features to disk."""
        # save_path = self.save_path / f"{timepoint}_{self.model_name_path}_features.npy"
        self.embeddings = features
        save_path = self.save_path / f"{self.model_name_path}_features.npy"
        np.save(save_path, features.cpu().numpy())
        assert save_path.exists(), f"Failed to save features to {save_path}"
    
    def _load_features(self):  # , timepoint):
        """Loads the features from disk."""
        # load_path = self.save_path / f"{timepoint}_{self.model_name_path}_features.npy"
        if self.embeddings is None:
            load_path = self.save_path / f"{self.model_name_path}_features.npy"
            features = np.load(load_path)
            assert features is not None, f"Failed to load features from {load_path}"
            self.embeddings = torch.tensor(features).to(self.device)
        return self.embeddings
   
    def _check_existing_embeddings(self, n_images):
        """Checks if embeddings for the model already exist."""
        try:
            self._load_features()
        except FileNotFoundError:
            return True
        missing = (self.embeddings.shape[0] != n_images)
        if not missing:
            logger.info(f"Embeddings for {self.model_name} already exist. Skipping embedding computation.")
        return missing


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
        mode: Literal[
            # "exact_patch",  # Uses the image patch centered on the detection for embedding
            "nearest_patch",  # Runs on whole image, then finds the nearest patch to the detection in the embedding
            "mean_patches"  # Runs on whole image, then averages the embeddings of all patches that intersect with the detection
            ] = "nearest_patch",
        ):
        super().__init__(image_size, save_path, batch_size, device, mode)
        # self.input_size = 224
        self.input_mul = 3
        self.input_size = self.input_mul * 224
        self.final_grid_size = 7 * self.input_mul  # 14x14 grid
        self.n_channels = 3
        self.hidden_state_size = 768

        ##
        self.im_proc_kwargs["size"] = (self.input_size, self.input_size)
        ##
        self.image_processor = AutoImageProcessor.from_pretrained(self.model_name)
        config = HieraConfig.from_pretrained(self.model_name)
        config.image_size = [self.input_size, self.input_size]
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

    def _run_model(self, images) -> torch.Tensor:
        """Extracts embeddings from the model."""
        images = self._rescale_batch(images)
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
        mode: Literal[
            # "exact_patch",  # Uses the image patch centered on the detection for embedding
            "nearest_patch",  # Runs on whole image, then finds the nearest patch to the detection in the embedding
            "mean_patches"  # Runs on whole image, then averages the embeddings of all patches that intersect with the detection
            ] = "nearest_patch",
        ):
        super().__init__(image_size, save_path, batch_size, device, mode)
        self.input_size = 224
        self.final_grid_size = 16  # 16x16 grid
        self.n_channels = 3  # expects RGB images
        self.hidden_state_size = 768
        self.image_processor = AutoImageProcessor.from_pretrained(self.model_name)
        ##
        self.im_proc_kwargs["size"] = (self.input_size, self.input_size)
        ##
        self.model = AutoModel.from_pretrained(self.model_name)
        # config = Dinov2Config.from_pretrained(self.model_name)
        # config.image_size = self.input_size
        
        # self.model = Dinov2Model(config)
        # logger.info(f"Model from config: {self.model.config}")
        # logger.info(f"Pretrained model : {Dinov2Model.from_pretrained(self.model_name).config}")
        self.model.to(self.device)
        
    def _run_model(self, images) -> torch.Tensor:
        """Extracts embeddings from the model."""
        inputs = self.image_processor(images, **self.im_proc_kwargs).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # ignore the CLS token (not classifying)
        # this way we get only the patch embeddings
        # which are compatible with finding the relevant patches directly
        # in the rest of the code
        self.final_grid_size = np.sqrt(504)
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
        mode: Literal[
            # "exact_patch",  # Uses the image patch centered on the detection for embedding
            "nearest_patch",  # Runs on whole image, then finds the nearest patch to the detection in the embedding
            "mean_patches"  # Runs on whole image, then averages the embeddings of all patches that intersect with the detection
            ] = "nearest_patch",
        ):
        super().__init__(image_size, save_path, batch_size, device, mode)
        self.input_size = 1024
        self.final_grid_size = 64  # 64x64 grid
        self.n_channels = 3
        self.hidden_state_size = 256
        self.image_processor = SamProcessor.from_pretrained(self.model_name)
        self.model = SamModel.from_pretrained(self.model_name)
        
        self.model.to(self.device)
        
    def _run_model(self, images) -> torch.Tensor:
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
        mode: Literal[
            # "exact_patch",  # Uses the image patch centered on the detection for embedding
            "nearest_patch",  # Runs on whole image, then finds the nearest patch to the detection in the embedding
            "mean_patches"  # Runs on whole image, then averages the embeddings of all patches that intersect with the detection
            ] = "nearest_patch",
        ):
        super().__init__(image_size, save_path, batch_size, device, mode)
        self.input_size = 1024
        self.final_grid_size = 64  # 64x64 grid
        self.n_channels = 3   
        self.hidden_state_size = 256        
        self.model = SAM2ImagePredictor.from_pretrained(self.model_name, device=self.device)
        
        self.batch_return_type = "list[np.ndarray]"
        self.channel_first = False
    
    def _run_model(self, images: list[np.ndarray]) -> torch.Tensor:
        """Extracts embeddings from the model."""
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            self.model.set_image_batch(images)
            
        features = self.model._features['image_embed']  # (B, hidden_state_size, grid_size, grid_size)
        B, N, H, W = features.shape
        return features.permute(0, 2, 3, 1).reshape(B, H * W, N)  # (B, grid_size**2, hidden_state_size)
    

FeatureExtractor._available_backbones = AVAILABLE_PRETRAINED_BACKBONES