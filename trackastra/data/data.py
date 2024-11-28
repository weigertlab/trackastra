import logging
from collections.abc import Sequence
from pathlib import Path
from timeit import default_timer
from typing import Literal

import joblib
import lz4.frame
import networkx as nx
import numpy as np
import pandas as pd
import tifffile
import torch
from numba import njit
from scipy import ndimage as ndi
from scipy.spatial.distance import cdist
from skimage.measure import regionprops
from skimage.segmentation import relabel_sequential
from torch.utils.data import Dataset
from tqdm import tqdm

from ..utils import blockwise_sum, normalize
from . import wrfeat
from ._check_ctc import _check_ctc, _get_node_attributes
from .augmentations import AugmentationPipeline, RandomCrop
from .features import _PROPERTIES, extract_features_patch, extract_features_regionprops
from .matching import matching

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
    def __init__(
        self,
        root: str = "",
        ndim: int = 2,
        use_gt: bool = True,
        detection_folders: list[str] | None = None,
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
        ] = "wrfeat",
        sanity_dist: bool = False,
        crop_size: tuple | None = None,
        return_dense: bool = False,
        compress: bool = False,
        **kwargs,
    ) -> None:
        """_summary_.

        Args:
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
        self.ndim = ndim
        self.features = features

        if features not in ("none", "wrfeat") and features not in _PROPERTIES[ndim]:
            raise ValueError(
                f"'{features}' not one of the supported {ndim}D features"
                f" {tuple(_PROPERTIES[ndim].keys())}"
            )

        logger.info(f"ROOT (config): {self.root}")
        self.root, self.gt_tra_folder = self._guess_root_and_gt_tra_folder(self.root)
        logger.info(f"ROOT: \t{self.root}")
        logger.info(f"GT TRA:\t{self.gt_tra_folder}")
        if self.use_gt:
            self.gt_mask_folder = self._guess_mask_folder(self.root, self.gt_tra_folder)
        else:
            logger.info("Using dummy masks as GT")
            self.gt_mask_folder = self._guess_det_folder(
                self.root, self.detection_folders[0]
            )
        logger.info(f"GT MASK:\t{self.gt_mask_folder}")

        # dont load image data if not needed
        if features in ("none",):
            self.img_folder = None
        else:
            self.img_folder = self._guess_img_folder(self.root)
        logger.info(f"IMG:\t\t{self.img_folder}")

        self.feat_dim, self.augmenter, self.cropper = self._setup_features_augs(
            ndim, features, augment, crop_size
        )

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

        start = default_timer()

        if self.features == "wrfeat":
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

    # def from_ctc

    @classmethod
    def from_arrays(cls, imgs: np.ndarray, masks: np.ndarray, train_args: dict):
        self = cls(**train_args)
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

        self.feat_dim, self.augmenter, self.cropper = self._setup_features_augs(
            self.ndim, self.features, self.augment, self.crop_size
        )

        start = default_timer()

        if self.features == "wrfeat":
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

    def _get_ndivs(self, windows):
        n_divs = []
        for w in tqdm(windows, desc="Counting divisions", leave=True):
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
        self, ndim: int, features: str, augment: int, crop_size: tuple[int]
    ):
        if self.features == "wrfeat":
            return self._setup_features_augs_wrfeat(ndim, features, augment, crop_size)

        if ndim == 2:
            augmenter = AugmentationPipeline(p=0.8, level=augment) if augment else None
            feat_dim = {
                "none": 0,
                "regionprops": 7,
                "regionprops2": 6,
                "patch": 256,
                "patch_regionprops": 256 + 5,
            }[features]
        elif ndim == 3:
            augmenter = AugmentationPipeline(p=0.8, level=augment) if augment else None
            feat_dim = {
                "none": 0,
                "regionprops2": 11,
                "patch_regionprops": 256 + 8,
            }[features]
        cropper = (
            RandomCrop(
                crop_size=crop_size,
                ndim=ndim,
                use_padding=False,
                ensure_inside_points=True,
            )
            if crop_size is not None
            else None
        )
        return feat_dim, augmenter, cropper

    def _compress_data(self):
        # compress masks and assoc_matrix
        logger.info("Compressing masks and assoc_matrix to save memory")
        for w in self.windows:
            w["mask"] = _CompressedArray(w["mask"])
            # dont compress full imgs (as needed for patch features)
            w["img"] = _CompressedArray(w["img"])
            w["assoc_matrix"] = _CompressedArray(w["assoc_matrix"])
        self.gt_masks = _CompressedArray(self.gt_masks)
        self.det_masks = {k: _CompressedArray(v) for k, v in self.det_masks.items()}
        # dont compress full imgs (as needed for patch features)
        self.imgs = _CompressedArray(self.imgs)

    def _guess_root_and_gt_tra_folder(self, inp: Path):
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

    def _guess_img_folder(self, root: Path):
        """Guesses the image folder corresponding to a root."""
        if (root / "img").exists():
            return root / "img"
        else:
            return root

    def _guess_mask_folder(self, root: Path, gt_tra: Path):
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
                logger.info(f"GT MASK:\t{st_path} for correcting with ST masks")
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

    def _masks2properties(self, masks):
        """Turn label masks into lists of properties, sorted (ascending) by time and label id.

        Args:
            masks (np.ndarray): T, (Z), H, W

        Returns:
            labels: List of labels
            ts: List of timepoints
            coords: List of coordinates
        """
        # Get coordinates, timepoints, and labels of detections
        labels = []
        ts = []
        coords = []
        properties_by_time = dict()
        assert len(self.imgs) == len(masks)
        for _t, frame in tqdm(
            enumerate(masks),
            # total=len(detections),
            leave=False,
            desc="Loading masks and properties",
        ):
            regions = regionprops(frame)
            t_labels = []
            t_ts = []
            t_coords = []
            for _r in regions:
                t_labels.append(_r.label)
                t_ts.append(_t)
                centroid = np.array(_r.centroid).astype(int)
                t_coords.append(centroid)

            properties_by_time[_t] = dict(coords=t_coords, labels=t_labels)
            labels.extend(t_labels)
            ts.extend(t_ts)
            coords.extend(t_coords)

        labels = np.array(labels, dtype=int)
        ts = np.array(ts, dtype=int)
        coords = np.array(coords, dtype=int)

        return labels, ts, coords, properties_by_time

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

    def _load(self):
        # Load ground truth
        self.gt_masks, self.gt_track_df = self._load_gt()

        self.gt_masks = self._check_dimensions(self.gt_masks)

        # Load images
        if self.img_folder is None:
            self.imgs = np.zeros_like(self.gt_masks)
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
                    _compress_img_mask_preproc(im, mask, self.features)
                    for im, mask in zip(self.imgs, self.gt_masks)
                )

        assert len(self.gt_masks) == len(self.imgs)

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
                ) = self._masks2properties(det_masks)

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

            self.properties_by_time[_f] = det_properties_by_time
            self.det_masks[_f] = det_masks
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

    def __getitem__(self, n: int, return_dense=None):
        # if not set, use default
        if self.features == "wrfeat":
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

        if isinstance(mask, _CompressedArray):
            mask = mask.decompress()
        if isinstance(img, _CompressedArray):
            img = img.decompress()
        if isinstance(assoc_matrix, _CompressedArray):
            assoc_matrix = assoc_matrix.decompress()

        # cropping
        if self.cropper is not None:
            (img2, mask2, coords2), idx = self.cropper(img, mask, coords)
            if len(idx) > 0:
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
                        "disable augmentation as no trajectories would be left"
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
                    print("disable augmentation as no trajectories would be left")

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
                    print("disable augmentation as no trajectories would be left")

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
            logger.debug(
                f"Clipped window of size {timepoints[n_elems - 1] - timepoints.min()}"
            )

        coords0 = torch.from_numpy(coords).float()
        features = torch.from_numpy(features).float()
        assoc_matrix = torch.from_numpy(assoc_matrix.copy()).float()
        labels = torch.from_numpy(labels).long()
        timepoints = torch.from_numpy(timepoints).long()

        if self.augmenter is not None:
            coords = coords0.clone()
            coords[:, 1:] += torch.randint(0, 256, (1, self.ndim))
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

    def _setup_features_augs_wrfeat(
        self, ndim: int, features: str, augment: int, crop_size: tuple[int]
    ):
        # FIXME: hardcoded
        feat_dim = 7 if ndim == 2 else 12
        if augment == 1:
            augmenter = wrfeat.WRAugmentationPipeline(
                [
                    wrfeat.WRRandomFlip(p=0.5),
                    wrfeat.WRRandomAffine(
                        p=0.8, degrees=180, scale=(0.5, 2), shear=(0.1, 0.1)
                    ),
                    # wrfeat.WRRandomBrightness(p=0.8, factor=(0.5, 2.0)),
                    # wrfeat.WRRandomOffset(p=0.8, offset=(-3, 3)),
                ]
            )
        elif augment > 1:
            augmenter = wrfeat.WRAugmentationPipeline(
                [
                    wrfeat.WRRandomFlip(p=0.5),
                    wrfeat.WRRandomAffine(
                        p=0.8, degrees=180, scale=(0.5, 2), shear=(0.1, 0.1)
                    ),
                    wrfeat.WRRandomBrightness(p=0.8),
                    wrfeat.WRRandomOffset(p=0.8, offset=(-3, 3)),
                ]
            )
        else:
            augmenter = None

        cropper = (
            wrfeat.WRRandomCrop(
                crop_size=crop_size,
                ndim=ndim,
            )
            if crop_size is not None
            else None
        )
        return feat_dim, augmenter, cropper

    def _load_wrfeat(self):
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
                self.imgs = np.stack([
                    _compress_img_mask_preproc(im, mask, self.features)
                    for im, mask in zip(self.imgs, self.gt_masks)
                ])

        assert len(self.gt_masks) == len(self.imgs)

        # Load each of the detection folders and create data samples with a sliding window
        windows = []
        self.properties_by_time = dict()
        self.det_masks = dict()
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
                logger.info(f"DET MASK:\t{det_folder}")
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

            features = joblib.Parallel(n_jobs=8)(
                joblib.delayed(wrfeat.WRFeatures.from_mask_img)(
                    mask=mask[None], img=img[None], t_start=t
                )
                for t, (mask, img) in enumerate(zip(det_masks, self.imgs))
            )

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
        coords = track["coords"]
        assoc_matrix = track["assoc_matrix"]
        labels = track["labels"]
        img = track["img"]
        mask = track["mask"]
        timepoints = track["timepoints"]
        track["t1"]
        feat = track["wrfeat"]

        if return_dense and isinstance(mask, _CompressedArray):
            mask = mask.decompress()
        if return_dense and isinstance(img, _CompressedArray):
            img = img.decompress()
        if isinstance(assoc_matrix, _CompressedArray):
            assoc_matrix = assoc_matrix.decompress()

        # cropping
        if self.cropper is not None:
            feat, idx = self.cropper(feat)
            labels = labels[idx]
            timepoints = timepoints[idx]
            assoc_matrix = assoc_matrix[idx][:, idx]

        if self.augmenter is not None:
            feat = self.augmenter(feat)

        coords0 = np.concatenate((feat.timepoints[:, None], feat.coords), axis=-1)
        coords0 = torch.from_numpy(coords0).float()
        assoc_matrix = torch.from_numpy(assoc_matrix.copy()).float()
        features = torch.from_numpy(feat.features_stacked).float()
        labels = torch.from_numpy(feat.labels).long()
        timepoints = torch.from_numpy(feat.timepoints).long()

        if self.augmenter is not None:
            coords = coords0.clone()
            coords[:, 1:] += torch.randint(0, 256, (1, self.ndim))
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

            family[
                i, np.array([fwd_map[gt_tracklet_id], *ancestors, *descendants])
            ] = True
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
    pad = torch.full(pad_shape, fill_value=value, dtype=x.dtype).to(x.device)
    return torch.cat((x, pad), dim=dim)


def collate_sequence_padding(max_len: int | None = None):
    """Collate function that pads all sequences to the same length."""

    def collate_sequence(batch):
        lens = tuple(len(x["coords"]) for x in batch)
        n_max_len = max(lens) if max_len is None else max_len
        # print(tuple(len(x["coords"]) for x in batch))
        # print(tuple(len(x["features"]) for x in batch))
        # print(batch[0].keys())
        tuple(batch[0].keys())
        normal_keys = {
            "coords": 0,
            "features": 0,
            "labels": 0,  # Not needed, remove for speed.
            "timepoints": -1,  # There are real timepoints with t=0. -1 for distinction from that.
        }
        n_pads = tuple(n_max_len - s for s in lens)
        batch_new = dict(
            (
                k,
                torch.stack(
                    [pad_tensor(x[k], n_max=n_max_len, value=v) for x in batch], dim=0
                ),
            )
            for k, v in normal_keys.items()
        )
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
        return batch_new

    return collate_sequence
