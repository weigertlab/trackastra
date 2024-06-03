import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# from .data import CTCData
import tifffile
from tqdm import tqdm

logger = logging.getLogger(__name__)


def load_tiff_timeseries(
    dir: Path,
    dtype: str | type | None = None,
    downscale: tuple[int, ...] | None = None,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> np.ndarray:
    """Loads a folder of `.tif` or `.tiff` files into a numpy array.
    Each file is interpreted as a frame of a time series.

    Args:
        folder:
        dtype:
        downscale: One int for each dimension of the data. Avoids memory overhead.
        start_frame: The first frame to load.
        end_frame: The last frame to load.

    Returns:
        np.ndarray: The loaded data.
    """
    # TODO make safe for label arrays
    logger.debug(f"Loading tiffs from {dir} as {dtype}")
    files = sorted(list(dir.glob("*.tif")) + list(dir.glob("*.tiff")))[
        start_frame:end_frame
    ]
    shape = tifffile.imread(files[0]).shape
    if downscale:
        assert len(downscale) == len(shape)
    else:
        downscale = (1,) * len(shape)

    files = files[:: downscale[0]]

    x = []
    for f in tqdm(
        files,
        leave=False,
        desc=f"Loading [{start_frame}:{end_frame}:{downscale[0]}]",
    ):
        _x = tifffile.imread(f)
        if dtype:
            _x = _x.astype(dtype)
        assert _x.shape == shape
        slices = tuple(slice(None, None, d) for d in downscale[1:])
        _x = _x[slices]
        x.append(_x)

    x = np.stack(x)
    logger.debug(f"Loaded array of shape {x.shape} from {dir}")
    return x


def load_tracklet_links(folder: Path) -> pd.DataFrame:
    candidates = [
        folder / "man_track.txt",
        folder / "res_track.txt",
    ]
    for c in candidates:
        if c.exists():
            path = c
            break
    else:
        raise FileNotFoundError(f"Could not find tracklet links in {folder}")

    df = pd.read_csv(
        path,
        delimiter=" ",
        names=["label", "t1", "t2", "parent"],
        dtype=int,
    )
    # Remove invalid tracks with t2 > t1
    df = df[df.t1 <= df.t2]

    n_dets = (df.t2 - df.t1 + 1).sum()
    logger.debug(f"{folder} has {n_dets} detections")

    n_divs = (df[df.parent != 0]["parent"].value_counts() == 2).sum()
    logger.debug(f"{folder} has {n_divs} divisions")
    return df


def filter_track_df(
    df: pd.DataFrame,
    start_frame: int = 0,
    end_frame: int = sys.maxsize,
    downscale: int = 1,
) -> pd.DataFrame:
    """Only keep tracklets that are present in the given time interval."""
    df.columns = ["label", "t1", "t2", "parent"]
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

        logger.debug(f"Temporal downscaling of tracklet links by {downscale}")

        # remove tracklets that have been fully deleted by temporal downsampling

        mask = (
            # (df["t2"] - df["t1"] < downscale - 1)
            (df["t1"] % downscale != 0)
            & (df["t2"] % downscale != 0)
            & (df["t1"] // downscale == df["t2"] // downscale)
        )
        logger.debug(
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


# TODO fix
# def dataset_to_ctc(dataset: CTCData, path, start: int = 0, stop: int | None = None):
#     """save dataset to ctc format for debugging purposes"""
#     out = Path(path)
#     print(f"Saving dataset to {out}")
#     out_img = out / "img"
#     out_img.mkdir(exist_ok=True, parents=True)
#     out_mask = out / "TRA"
#     out_mask.mkdir(exist_ok=True, parents=True)
#     if stop is None:
#         stop = len(self)
#     lines = []
#     masks, imgs = [], []
#     t_offset = 0
#     max_mask = 0
#     n_lines = 0
#     all_coords = []
#     for i in tqdm(range(start, stop)):
#         d = dataset.__getitem__(i, return_dense=True)
#         mask = d["mask"].numpy()
#         mask[mask > 0] += max_mask
#         max_mask = max(max_mask, mask.max())
#         masks.extend(mask)
#         imgs.extend(d["img"].numpy())
#         # add vertices
#         coords = d["coords0"].numpy()
#         ts, coords = coords[:, 0].astype(int), coords[:, 1:]
#         A = d["assoc_matrix"].numpy()
#         t_unique = sorted(np.unique(ts))
#         for t1, t2 in zip(t_unique[:-1], t_unique[1:]):
#             A_sub = A[ts == t1][:, ts == t2]
#             for i, a in enumerate(A_sub):

#                 v1 = coords[ts == t1][i]
#                 for j in np.where(a > 0)[0]:
#                     v2 = coords[ts == t2][j]
#                     # lines.append(
#                     #     {
#                     #         "index": n_lines,
#                     #         "shape-type": "line",
#                     #         "vertex-index": 0,
#                     #         "axis-0": t2 + t_offset,
#                     #         "axis-1": v1[0],
#                     #         "axis-2": v1[1],
#                     #     }
#                     # )
#                     # lines.append(
#                     #     {
#                     #         "index": n_lines,
#                     #         "shape-type": "line",
#                     #         "vertex-index": 1,
#                     #         "axis-0": t2 + t_offset,
#                     #         "axis-1": v2[0],
#                     #         "axis-2": v2[1],
#                     #     }
#                     # )
#                     lines.append([n_lines, "line", 0, t2 + t_offset] + v1.tolist())
#                     lines.append([n_lines, "line", 1, t2 + t_offset] + v2.tolist())
#                     n_lines += 1

#         c = d["coords0"].numpy()
#         c[:, 0] += t_offset
#         all_coords.extend(c)
#         t_offset += len(mask)

#     ax_cols = [f"axis-{i}" for i in range(dataset.ndim + 1)]
#     df = pd.DataFrame(lines, columns=["index", "shape-type", "vertex-index"] + ax_cols)
#     df.to_csv(out / "lines.csv", index=False)

#     df_c = pd.DataFrame(all_coords, columns=ax_cols)
#     df_c.to_csv(out / "coords.csv", index=False)

#     for i, m in enumerate(imgs):
#         # tifffile.imwrite(out_img/f'img_{i:04d}.tif', m)
#         if dataset.ndim == 2:
#             imageio.imwrite(
#                 out_img / f"img_{i:04d}.jpg",
#                 np.clip(20 + 100 * m, 0, 255).astype(np.uint8),
#             )

#     for i, m in enumerate(masks):
#         tifffile.imwrite(out_mask / f"mask_{i:04d}.tif", m, compression="zstd")

#     return d
