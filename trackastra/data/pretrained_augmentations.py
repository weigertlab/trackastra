from abc import ABC, abstractmethod
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import tv_tensors
from torchvision.transforms import v2 as transforms

from trackastra.utils.utils import percentile_norm


class BaseAugmentation(ABC):
    """Base class for windowed region augmentations."""
    def __init__(self, p: float = 0.5, rng_seed=None):
        self._p = p
        self._rng = np.random.RandomState(rng_seed)
        self.applied_record = {}
        self.signature = {}

    def __call__(self, images: torch.Tensor, masks: tv_tensors.Mask):
        if self._p is None or self._rng.rand() < self._p:
            aug = self._get_aug()
            return aug(images, masks)
        return images, masks

    @abstractmethod
    def _get_aug(self) -> transforms.Compose:
        raise NotImplementedError()


class FlipAugment(BaseAugmentation):
    def __init__(self, p_horizontal: float = 0.5, p_vertical: float = 0.5, rng_seed=None):
        super().__init__(p=None, rng_seed=rng_seed)
        self._p_horizontal = p_horizontal
        self._p_vertical = p_vertical
        self.signature = {
            "FlipAugment": {
                "horizontal": self._p_horizontal,
                "vertical": self._p_vertical
            }
        }

    def __call__(self, images: torch.Tensor, masks: tv_tensors.Mask):
        if self._rng.rand() < self._p_horizontal:
            images = transforms.functional.hflip(images)
            masks = transforms.functional.hflip(masks)
            self.applied_record["hflip"] = True
        else:
            self.applied_record["hflip"] = False
        if self._rng.rand() < self._p_vertical:
            images = transforms.functional.vflip(images)
            masks = transforms.functional.vflip(masks)
            self.applied_record["vflip"] = True
        else:
            self.applied_record["vflip"] = False
        return images, masks

    def _get_aug(self) -> transforms.Compose:
        raise NotImplementedError("Use __call__ instead.")


class RotAugment(BaseAugmentation):

    def __init__(self, p: float = 0.5, degrees: int = 15, rng_seed=None):
        super().__init__(p, rng_seed=rng_seed)
        self.degrees = degrees
        self.signature = {
            "RotAugment": {
                "p": self._p,
                "degrees": self.degrees,
            }
        }

    def _get_aug(self):
        self.applied_record["rotation"] = self.degrees
        t = transforms.RandomRotation(degrees=self.degrees)
        return t


class Rot90Augment(BaseAugmentation):

    def __init__(self, p=0.5, rng_seed=None):
        super().__init__(p, rng_seed=rng_seed)
        self.signature = {
            "Rot90Augment": {
                "p": self._p,
            }
        }

    def __call__(self, images, masks):
        if self._rng.rand() > self._p:
            return images, masks
        angle = self._get_aug()
        images = transforms.functional.rotate(images, angle, expand=True)
        masks = transforms.functional.rotate(masks, angle, expand=True)
        return images, masks

    def _get_aug(self):
        angle = self._rng.choice([90, 180, 270])
        self.applied_record["rot90"] = int(angle)
        return angle


class BrightnessJitter(BaseAugmentation):

    def __init__(self, bright_shift: float = 0.5, contrast_shift: float = 0.5, rng_seed=None):
        super().__init__(p=None, rng_seed=rng_seed)
        self._b_shift = bright_shift
        self._c_shift = contrast_shift
        self.signature = {
            "BrightnessJitter": {
                "brightness_shift": self._b_shift,
                "contrast_shift": self._c_shift
            }
        }

    def _get_aug(self):
        if self._b_shift is not None:
            bright = self._rng.uniform(0, self._b_shift)
        else:
            bright = None
        self.applied_record["brightness_jitter"] = bright
        if self._c_shift is not None:
            contrast = self._rng.uniform(0, self._c_shift)
        else:
            contrast = None
        self.applied_record["contrast_jitter"] = contrast
        return transforms.ColorJitter(brightness=bright, contrast=contrast)


class AddGaussianNoise(BaseAugmentation):
    def __init__(self, mean: float = 0.0, std: float = 0.1, rng_seed=None):
        super().__init__(p=None, rng_seed=rng_seed)
        self.mean = mean
        self.sigma = std
        self.signature = {
            "AddGaussianNoise": {
                "mean": self.mean,
                "std": self.sigma
            }
        }

    def _get_aug(self):
        # sample random mean/std
        self.applied_record["gaussian_noise"] = (self.mean, self.sigma)
        return transforms.GaussianNoise(mean=self.mean, sigma=self.sigma)

    def __call__(self, images: torch.Tensor, masks: tv_tensors.Mask):
        aug = self._get_aug()
        images = aug(images)
        return images, masks


class GaussianBlur(BaseAugmentation):
    def __init__(self, kernel_size: int = 3, sigma: tuple[float] = (0.01, 1.0), rng_seed=None):
        super().__init__(p=None, rng_seed=rng_seed)
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.signature = {
            "GaussianBlur": {
                "kernel_size": self.kernel_size,
                "sigma": self.sigma
            }
        }

    def _get_aug(self):
        self.applied_record["gaussian_blur"] = (self.kernel_size, self.sigma)
        return transforms.GaussianBlur(kernel_size=self.kernel_size, sigma=self.sigma)

    def __call__(self, images: torch.Tensor, masks: tv_tensors.Mask):
        aug = self._get_aug()
        images = aug(images)
        return images, masks


class RandomAffine(BaseAugmentation):
    def __init__(self, degrees: float = 0.0, translate: tuple[float, float] = (0.0, 0.0), scale: tuple[float, float] = (1.0, 1.0), rng_seed=None):
        super().__init__(p=None, rng_seed=rng_seed)
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.signature = {
            "RandomAffine": {
                "degrees": self.degrees,
                "translate": self.translate,
                "scale": self.scale
            }
        }

    def _get_aug(self):
        return transforms.RandomAffine(degrees=self.degrees, translate=self.translate, scale=self.scale)

    def __call__(self, images: torch.Tensor, masks: tv_tensors.Mask):
        aug = self._get_aug()
        images, masks = aug(images, masks)
        return images, masks


class ElasticTransform(BaseAugmentation):
    def __init__(self, p=0.5, alpha: float = 10.0, sigma: float = 0.5, rng_seed=None):
        super().__init__(p=p, rng_seed=rng_seed)
        self.alpha = alpha
        self.sigma = sigma
        self.signature = {
            "ElasticTransform": {
                "p": self._p,
                "alpha": self.alpha,
                "sigma": self.sigma
            }
        }

    def _get_aug(self):
        alpha = self._rng.uniform(0, self.alpha)
        sigma = self._rng.uniform(0, self.sigma)
        self.applied_record["elastic_transform"] = (alpha, sigma)
        return transforms.ElasticTransform(alpha=alpha, sigma=sigma)


class RandomScale(BaseAugmentation):
    def __init__(self, p: float = 0.9, max_scale: float = 1.0, min_scale=0.8, preserve_size=False, rng_seed=None):
        super().__init__(p=p, rng_seed=rng_seed)
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.preserve_size = preserve_size
        self.signature = {
            "RandomScale": {
                "p": self._p,
                "min_scale": self.min_scale,
                "max_scale": self.max_scale,
                "preserve_size": self.preserve_size
            }
        }

    def _get_aug(self):
        scale = self._rng.uniform(self.min_scale, self.max_scale)
        self.applied_record["random_scale"] = scale
        return scale

    def __call__(self, images: torch.Tensor, masks: tv_tensors.Mask):
        if self._p is None or self._rng.rand() < self._p:
            scale = self._get_aug()
            orig_h, orig_w = images.shape[-2], images.shape[-1]
            new_h, new_w = int(orig_h * scale), int(orig_w * scale)

            # Resize images and masks
            images_scaled = F.interpolate(images, size=(new_h, new_w), mode="bilinear", align_corners=False)
            masks_scaled = F.interpolate(masks.float(), size=(new_h, new_w), mode="nearest").long()

            if self.preserve_size:
                pad_h = max(orig_h - new_h, 0)
                pad_w = max(orig_w - new_w, 0)
                pad = [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]  # left, right, top, bottom

                images_scaled = F.pad(images_scaled, pad, mode="constant", value=0)
                masks_scaled = F.pad(masks_scaled, pad, mode="constant", value=0)

                # If scaled image is larger, crop to original size
                images_scaled = images_scaled[..., :orig_h, :orig_w]
                masks_scaled = masks_scaled[..., :orig_h, :orig_w]

            return images_scaled, masks_scaled
        return images, masks


class IdentityAugment(BaseAugmentation):
    """Identity augmentation for debugging purposes."""
    def __init__(self, p: float = 1.0, rng_seed=None):
        super().__init__(p=p, rng_seed=rng_seed)
        self.signature = {
            "IdentityAugment": {
                "p": self._p
            }
        }

    def _get_aug(self):
        self.applied_record["identity"] = True
        return transforms.Lambda(lambda x: x)

    def __call__(self, images: torch.Tensor, masks: tv_tensors.Mask):
        return images, masks


class PretrainedAugmentations:
    """Augmentation pipeline to get augmented copies of model embeddings."""
    default_normalize = percentile_norm

    def __init__(self, rng_seed=None, normalize=True, shuffle=True):
        self.aug_record = {}
        self.aug_list = [
            # IdentityAugment(rng_seed=rng_seed), # debugging
            BrightnessJitter(bright_shift=0.3, contrast_shift=0.3, rng_seed=rng_seed),
            FlipAugment(p_horizontal=0.5, p_vertical=0.5, rng_seed=rng_seed),
            # RotAugment(degrees=10, rng_seed=rng_seed),
            Rot90Augment(p=0.5, rng_seed=rng_seed),
            # AddGaussianNoise(mean=0.0, std=0.02, rng_seed=rng_seed),
            RandomScale(rng_seed=rng_seed),
            GaussianBlur(kernel_size=5, sigma=(0.01, 2.0), rng_seed=rng_seed),
            # ElasticTransform(p=0.25, alpha=10.0, sigma=0.5, rng_seed=rng_seed),
            # RandomAffine(degrees=0.0, translate=(0.1, 0.1), scale=(0.9, 1.1), rng_seed=rng_seed),
        ]
        self._aug = None
        self._rng = np.random.RandomState(rng_seed)
        self.normalize = normalize
        self.image_shape = None
        self.shuffle = shuffle

    def __call__(self, images: torch.Tensor, masks: tv_tensors.Mask, normalize_func=None) -> tuple[torch.Tensor, tv_tensors.Mask, dict]:
        """Applies the augmentations to the images."""
        images, masks = self.preprocess(images, masks, normalize_func=normalize_func)

        if self.shuffle:
            aug_list = self.aug_list.copy()
            self._rng.shuffle(aug_list)
        else:
            aug_list = self.aug_list
        self._aug = transforms.Compose(aug_list)

        images = torch.unsqueeze(images, dim=1)  # add channel dimension (T, C, H, W) for augmentation
        masks = torch.unsqueeze(masks, dim=1)  # add channel dimension (T, C, H, W) for augmentation

        images, masks = self._aug(images, masks)
        if torch.isnan(images).any() or torch.isnan(masks).any():
            raise RuntimeError("NaN values found in images or masks after augmentation.")
        self.image_shape = images.shape
        # NOTE : most models do require 3 channels, but this will be done in FeatureExtractor, so the output is squeezed
        return images.squeeze(), masks.squeeze(), self.gather_records()

    def get_signature(self):
        """Returns the signature of the augmentations."""
        if self._aug is None:
            self._aug = transforms.Compose(self.aug_list)
        signatures = OrderedDict()
        for aug in self.aug_list:
            if aug.signature:
                signatures.update(aug.signature)
        return signatures

    def __add__(self, other):
        """Combines two augmentation pipelines."""
        if not isinstance(other, PretrainedAugmentations):
            raise TypeError("Can only combine with another PretrainedAugmentations instance.")
        combined = PretrainedAugmentations(rng_seed=self._rng.seed, normalize=self.normalize, shuffle=self.shuffle)
        combined.aug_list = self.aug_list + other.aug_list
        return combined
    
    def __repr__(self):
        sig = self.get_signature()
        msg = "Augmentation pipeline"
        for aug_name, params in sig.items():
            msg += f"- {aug_name}: {params}\n"
        msg += "_" * 40 + "\n"
        return msg.strip()

    def preprocess(self, images, masks, normalize_func=None):
        if not len(images.shape) == 3:
            raise ValueError(f"Images must be tensor of shape (T, H, W), got {len(images.shape)}D tensor.")
        if not len(masks.shape) == 3:
            raise ValueError(f"Masks must be tensor of shape (T, H, W), got {len(masks.shape)}D tensor.")

        if not isinstance(images, torch.Tensor):
            try:
                images = torch.tensor(images, dtype=torch.float32)
            except Exception as e:
                raise ValueError(f"Failed to convert images to tensor: {e}")
        if not isinstance(masks, tv_tensors.Mask):
            try:
                masks = tv_tensors.Mask(masks, dtype=torch.int64)
            except Exception as e:
                raise ValueError(f"Failed to convert masks to tensor: {e}")

        if normalize_func is not None:
            if not callable(normalize_func):
                raise ValueError("normalize_func must be a callable function.")
            images = normalize_func(images)

        return images, masks

    def gather_records(self):
        """Gathers the applied augmentation records."""
        self.aug_record = {}
        for aug in self.aug_list:
            self.aug_record.update(aug.applied_record)
        self.aug_record["image_shape"] = self.image_shape
        return self.aug_record
    

class PretrainedMovementAugmentations(PretrainedAugmentations):
    """Augmentation pipeline for movement embeddings."""
    def __init__(self, rng_seed=None, normalize=True, shuffle=True):
        super().__init__(rng_seed=rng_seed, normalize=normalize, shuffle=shuffle)
        self.aug_list = [
            FlipAugment(p_horizontal=0.5, p_vertical=0.5, rng_seed=rng_seed),
            Rot90Augment(p=0.5, rng_seed=rng_seed),
            RandomScale(rng_seed=rng_seed),
        ]
        

class PretrainedIntensityAugmentations(PretrainedAugmentations):
    """Augmentation pipeline for intensity embeddings."""
    def __init__(self, rng_seed=None, normalize=True, shuffle=True):
        super().__init__(rng_seed=rng_seed, normalize=normalize, shuffle=shuffle)
        self.aug_list = [
            BrightnessJitter(bright_shift=0.05, contrast_shift=0.05, rng_seed=rng_seed),
            # FlipAugment(p_horizontal=0.5, p_vertical=0.5, rng_seed=rng_seed),
            # Rot90Augment(p=0.5, rng_seed=rng_seed),
            AddGaussianNoise(mean=0.0, std=0.02, rng_seed=rng_seed),
            # RandomScale(rng_seed=rng_seed),
        ]
        

class IdentityAugmentations(PretrainedAugmentations):
    """Identity augmentation pipeline for debugging."""
    def __init__(self, rng_seed=None, normalize=True, shuffle=True):
        super().__init__(rng_seed=rng_seed, normalize=normalize, shuffle=shuffle)
        self.aug_list = [
            IdentityAugment(p=1.0, rng_seed=rng_seed),
        ]