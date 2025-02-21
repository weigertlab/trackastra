import itertools
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from skimage.measure import regionprops_table
from tqdm import tqdm
from transformers import (
    AutoImageProcessor,
    HieraModel,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# the property keys that are supported for 2 and 3 dim

_PROPERTIES = {
    2: {
        # FIXME: The only image regionprop possible now (when compressing) is mean_intensity,
        # since we store a mask with the mean intensity of each detection as the image.
        "regionprops": (
            "label",
            "area",
            "intensity_mean",
            "eccentricity",
            "solidity",
            "inertia_tensor",
        ),
        # faster
        "regionprops2": (
            "label",
            "area",
            "intensity_mean",
            "inertia_tensor",
        ),
        "patch_regionprops": (
            "label",
            "area",
            "intensity_mean",
            "inertia_tensor",
        ),
        "pretrained_feats": (),
    },
    3: {
        "regionprops2": (
            "label",
            "area",
            "intensity_mean",
            "inertia_tensor",
        ),
        "patch_regionprops": (
            "label",
            "area",
            "intensity_mean",
            "inertia_tensor",
        ),
    },
}

# forward declarations for indexing
HieraFeatures = None

_AVAILABLE_PRETRAINED_BACKBONES = {
    "facebook/hiera-tiny-224-hf": {
        "class": HieraFeatures, 
        "feat_dim": 768,
    },
}
PretrainBackboneType = Literal[  # cannot unpack this directly in python < 3.11 so it has to be copied
    "facebook/hiera-tiny-224-hf",
]
##############
##############
# Feature extraction from pretrained models
# Currently meant to wrap any transformers model


class FeatureExtractor(ABC):
    model_name = None
    def __init__(
        self, 
        image_size: tuple[int, int],
        save_path: str | Path,
        batch_size: int = 4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        mode: Literal[
            # "exact_patch",  # Uses the image patch centered on the detection for embedding
            "nearest_patch",  # Runs on whole image, then finds the nearest patch to the detection in the embedding
            # "mean_patch"  # Runs on whole image, then averages the embeddings of all patches that intersect with the detection
            ] = "nearest_patch",
        ):
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
        # Parameters for embedding extraction
        self.batch_size = batch_size
        self.device = device
        self.mode = mode
        # Saving parameters
        self.save_path: str | Path = save_path
        
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
    
    def _set_model_patch_size(self):
        if self.final_grid_size is not None and self.input_size is not None:
            self.model_patch_size = self.input_size // self.final_grid_size
            if not self.input_size % self.final_grid_size == 0:
                raise ValueError("The input size must be divisible by the final grid size.")
    
    def forward(
            self, 
            imgs, 
            coords,
            masks=None, 
        ) -> torch.Tensor:  # (n_regions, embedding_size)
        """Extracts embeddings from the model."""

        if self.mode == "nearest_patch":
            # find coordinate patches from detections
            patch_coords = self._map_coords_to_patches(coords)
            patch_idxs = self._find_nearest_patch(patch_coords)
            logger.debug(f"Coords: {coords}")
            logger.debug(f"Patch coordinates: {patch_coords}")
            logger.debug(f"Patch indices: {patch_idxs}")
            # load the embeddings and extract the relevant ones
            feats = torch.zeros(len(coords), self.hidden_state_size, device=self.device)
            indices = [y * self.final_grid_size + x for _, y, x in patch_idxs]

            unique_timepoints = list(set(t for t, _, _ in patch_idxs))
            embeddings_dict = {t: self._load_features(t) for t in unique_timepoints}
            feats = torch.zeros(len(coords), self.hidden_state_size, device=self.device)

            for i, (t, _, _) in enumerate(patch_idxs):
                feats[i] = embeddings_dict[t][indices[i]]

            return feats  # (n_regions, embedding_size)
    
    def precompute_embeddings(self, images):
        """Precomputes embeddings for all images."""
        missing = self._check_existing_embeddings(range(len(images)))
        if len(missing) > 0:
            for ts, batches in tqdm(self._prepare_batches(images[missing]), total=len(images) // self.batch_size, desc="Computing embeddings"):
                logger.debug(f"Time points: {ts}")
                logger.debug(f"Batch size: {batches.shape}")
                embeddings = self._run_model(batches)
                for t, ft in zip(ts, embeddings):
                    self._save_features(ft, t)
    
    @abstractmethod
    def _run_model(self, images) -> torch.Tensor:  # must return (B, grid_size**2, hidden_state_size)
        """Extracts embeddings from the model."""
        pass
    
    def _prepare_batches(self, images):
        """Prepares batches of images for embedding extraction."""
        # make batches if mode is exact_patch, otherwise prepare whole images and batch them
        if self.mode == "exact_patch":
            raise NotImplementedError()
        else:
            for i in range(0, len(images), self.batch_size):
                end = i + self.batch_size
                end = len(images) if end > len(images) else end
                batch = np.expand_dims(images[i:end], axis=1)  # (B, C, H, W)
                
                def normalize_batch(b):
                    for i, im in enumerate(b):
                        b[i] = (im - im.min()) / (im.max() - im.min())
                    return b
                
                batch = normalize_batch(batch)
                logger.debug(f"Batch min {batch.min()} max {batch.max()}")
                timepoints = range(i, end)
                if self.n_channels > 1:  # repeat channels if needed
                    if self.orig_n_channels > 1 and self.orig_n_channels != self.n_channels:
                        raise ValueError("When more than one original channel is provided, the number of channels in the model must match the number of channels in the input.")
                    batch = np.repeat(batch, self.n_channels, axis=1)
                yield timepoints, batch
    
    def _map_coords_to_patches(self, coords):
        """Maps the detection coordinates to the corresponding image patches."""
        scale_x = self.input_size / self.orig_image_size[0]
        scale_y = self.input_size / self.orig_image_size[1]
        patch_coords = []
        for t, x, y in coords:
            patch_x = int(x * scale_x)
            patch_y = int(y * scale_y)
            patch_coords.append((int(t), patch_x, patch_y))
        return patch_coords
    
    def _find_nearest_patch(self, patch_coords):
        """Finds the nearest patch to the detection in the embedding."""
        patch_idxs = []
        for c in patch_coords:
            x_idx = c[1] // self.model_patch_size
            y_idx = c[2] // self.model_patch_size
            patch_idxs.append((c[0], x_idx, y_idx))
        return patch_idxs
    
    def _mean_patch(self, masks):
        """Averages the embeddings of all patches that intersect with the detection."""
        raise NotImplementedError()
    
    def _exact_patch(self, imgs, masks, coords):
        """Uses the image patch centered on the detection for embedding."""
        raise NotImplementedError()
    
    def _save_features(self, features, timepoint):
        """Saves the features to disk."""
        save_path = self.save_path / f"{timepoint}_{self.model_name_path}_features.npy"
        np.save(save_path, features.cpu().numpy())
        assert save_path.exists(), f"Failed to save features to {save_path}"
    
    def _load_features(self, timepoint):
        """Loads the features from disk."""
        load_path = self.save_path / f"{timepoint}_{self.model_name_path}_features.npy"
        features = np.load(load_path)
        assert features is not None, f"Failed to load features from {load_path}"
        return torch.tensor(features).to(self.device)
   
    def _check_existing_embeddings(self, timepoints):
        """Checks if embeddings for all timepoints already exist."""
        missing = []
        for t in timepoints:
            t = int(t)
            if not (self.save_path / f"{t}_{self.model_name_path}_features.npy").exists() and t not in missing:
                missing.append(t)
        return missing

##############
class HieraFeatures(FeatureExtractor):
    model_name = "facebook/hiera-tiny-224-hf"
    def __init__(
        self, 
        image_size: tuple[int, int],
        save_path: str | Path,
        batch_size: int = 4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        mode: Literal[
            # "exact_patch",  # Uses the image patch centered on the detection for embedding
            "nearest_patch",  # Runs on whole image, then finds the nearest patch to the detection in the embedding
            # "mean_patch"  # Runs on whole image, then averages the embeddings of all patches that intersect with the detection
            ] = "nearest_patch",
        ):
        super().__init__(image_size, save_path, batch_size, device, mode)
        self.input_size = 224
        self.final_grid_size = 7  # 7x7 grid
        self.n_channels = 3  # expects RGB images
        self.hidden_state_size = 768
        self.image_processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = HieraModel.from_pretrained(self.model_name)
        
        self.model.to(self.device)
        
    def _run_model(self, images) -> torch.Tensor:
        """Extracts embeddings from the model."""
        inputs = self.image_processor(images, return_tensors="pt", use_fast=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        return outputs.last_hidden_state
    

_AVAILABLE_PRETRAINED_BACKBONES[HieraFeatures.model_name]["class"] = HieraFeatures
##############
##############


def extract_features_regionprops(
    mask: np.ndarray,
    img: np.ndarray,
    labels: np.ndarray,
    properties="regionprops2",
):
    ndim = mask.ndim
    assert ndim in (2, 3)
    assert mask.shape == img.shape

    prop_dict = _PROPERTIES[ndim]
    if properties not in prop_dict:
        raise ValueError(f"properties must be one of {prop_dict.keys()}")
    properties_tuple = prop_dict[properties]

    assert properties_tuple[0] == "label"

    labels = np.asarray(labels)

    # remove mask labels that are not present
    # not needed, remove for speed
    # mask[~np.isin(mask, labels)] = 0

    df = pd.DataFrame(
        regionprops_table(mask, intensity_image=img, properties=properties_tuple)
    )
    assert df.columns[0] == "label"
    assert df.columns[1] == "area"

    # the bnumber of inertia tensor columns depends on the dimensionality
    n_cols_inertia = ndim**2
    assert np.all(["inertia_tensor" in col for col in df.columns[-n_cols_inertia:]])

    # Hack for backwards compatibility
    if properties in ("regionprops", "patch_regionprops"):
        # Nice for conceptual clarity, but does not matter for speed
        # drop upper triangular part of symmetric inertia tensor
        for i, j in itertools.product(range(ndim), repeat=2):
            if i > j:
                df.drop(f"inertia_tensor-{i}-{j}", axis=1, inplace=True)

    table = df.to_numpy()
    table[:, 1] *= 0.001
    table[:, -n_cols_inertia:] *= 0.01
    # reorder according to labels
    features = np.zeros((len(labels), len(df.columns) - 1))

    # faster than iterating over pandas dataframe
    for row in table:
        # old version with tuple indexing, slow.
        # n = labels.index(int(row.label))
        # features[n] = row.to_numpy()[1:]

        # Only process regions present in the labels
        n = np.where(labels == int(row[0]))[0]
        if len(n) > 0:
            # Remove label column (0)!
            features[n[0]] = row[1:]

    return features


def extract_features_patch(
    mask: np.ndarray,
    img: np.ndarray,
    coords: np.ndarray,
    labels: np.ndarray,
    width_patch: int = 16,
):
    """16x16 Image patch around detection."""
    ndim = mask.ndim
    assert ndim in (2, 3) and mask.shape == img.shape
    if len(coords) == 0:
        return np.zeros((0, width_patch * width_patch))

    pads = (width_patch // 2,) * ndim

    img = np.pad(
        img,
        tuple((p, p) for p in pads),
        mode="constant",
    )

    coords = coords.astype(int) + np.array(pads)

    ss = tuple(
        tuple(slice(_c - width_patch // 2, _c + width_patch // 2) for _c in c)
        for c in coords
    )
    fs = tuple(img[_s] for _s in ss)

    # max project along z if 3D
    if ndim == 3:
        fs = tuple(f.max(0) for f in fs)

    features = np.stack([f.flatten() for f in fs])
    return features
