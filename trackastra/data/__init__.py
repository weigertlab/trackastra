
# Core data utilities (no training dependencies required)
from .dataset import (
    TrackingDataset,
    collate_sequence_padding,
    densify_assoc,
    warn_association_distances,
)
from .example_data import example_data_bacteria, example_data_fluo_3d, example_data_hela
from .features import extract_features_regionprops
from .io import TrackingSequence, load_ctc_images_masks
from .utils import filter_track_df, load_tiff_timeseries, load_tracklet_links
from .wrfeat import WRFeatures, build_windows, get_features

# Training-only classes (require lightning, kornia)
# Import these directly when needed for training:
#   from trackastra.data.augmentations import AugmentationPipeline, RandomCrop
#   from trackastra.data.distributed import BalancedDataModule, BalancedBatchSampler, BalancedDistributedSampler
