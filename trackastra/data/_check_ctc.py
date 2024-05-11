import logging

import numpy as np
import pandas as pd
from skimage.measure import label, regionprops_table

logger = logging.getLogger(__name__)


# from https://github.com/Janelia-Trackathon-2023/traccuracy/blob/main/src/traccuracy/loaders/_ctc.py
def _check_ctc(tracks: pd.DataFrame, detections: pd.DataFrame, masks: np.ndarray):
    """Sanity checks for valid CTC format.

    Hard checks (throws exception):
    - Tracklet IDs in tracks file must be unique and positive
    - Parent tracklet IDs must exist in the tracks file
    - Intertracklet edges must be directed forward in time.
    - In each time point, the set of segmentation IDs present in the detections must equal the set
    of tracklet IDs in the tracks file that overlap this time point.

    Soft checks (prints warning):
    - No duplicate tracklet IDs (non-connected pixels with same ID) in a single timepoint.

    Args:
        tracks (pd.DataFrame): Tracks in CTC format with columns Cell_ID, Start, End, Parent_ID.
        detections (pd.DataFrame): Detections extracted from masks, containing columns
            segmentation_id, t.
        masks (np.ndarray): Set of masks with time in the first axis.

    Raises:
        ValueError: If any of the hard checks fail.
    """
    logger.debug("Running CTC format checks")
    tracks = tracks.copy()
    tracks.columns = ["Cell_ID", "Start", "End", "Parent_ID"]
    if tracks["Cell_ID"].min() < 1:
        raise ValueError("Cell_IDs in tracks file must be positive integers.")
    if len(tracks["Cell_ID"]) < len(tracks["Cell_ID"].unique()):
        raise ValueError("Cell_IDs in tracks file must be unique integers.")

    for _, row in tracks.iterrows():
        if row["Parent_ID"] != 0:
            if row["Parent_ID"] not in tracks["Cell_ID"].values:
                raise ValueError(
                    f"Parent_ID {row['Parent_ID']} is not present in tracks."
                )
            parent_end = tracks[tracks["Cell_ID"] == row["Parent_ID"]]["End"].iloc[0]
            if parent_end >= row["Start"]:
                raise ValueError(
                    f"Invalid tracklet connection: Daughter tracklet with ID {row['Cell_ID']} "
                    f"starts at t={row['Start']}, "
                    f"but parent tracklet with ID {row['Parent_ID']} only ends at t={parent_end}."
                )

    for t in range(tracks["Start"].min(), tracks["End"].max()):
        track_ids = set(
            tracks[(tracks["Start"] <= t) & (tracks["End"] >= t)]["Cell_ID"]
        )
        det_ids = set(detections[(detections["t"] == t)]["segmentation_id"])
        if not track_ids.issubset(det_ids):
            raise ValueError(f"Missing IDs in masks at t={t}: {track_ids - det_ids}")
        if not det_ids.issubset(track_ids):
            raise ValueError(
                f"IDs {det_ids - track_ids} at t={t} not represented in tracks file."
            )

    for t, frame in enumerate(masks):
        _, n_components = label(frame, return_num=True)
        n_labels = len(detections[detections["t"] == t])
        if n_labels < n_components:
            logger.warning(f"{n_components - n_labels} non-connected masks at t={t}.")


def _get_node_attributes(masks):
    """Calculates x,y,z,t,label for each detection in a movie.

    Args:
        masks (np.ndarray): Set of masks with time in the first axis

    Returns:
        pd.DataFrame: Dataframe with one detection per row. Columns
            segmentation_id, x, y, z, t
    """
    data_df = pd.concat(
        [_detections_from_image(masks, idx) for idx in range(masks.shape[0])]
    ).reset_index(drop=True)
    data_df = data_df.rename(
        columns={
            "label": "segmentation_id",
            "centroid-2": "z",
            "centroid-1": "y",
            "centroid-0": "x",
        }
    )
    data_df["segmentation_id"] = data_df["segmentation_id"].astype(int)
    data_df["t"] = data_df["t"].astype(int)
    return data_df


def _detections_from_image(stack, idx):
    """Return the unique track label, centroid and time for each track vertex.

    Args:
        stack (np.ndarray): Stack of masks
        idx (int): Index of the image to calculate the centroids and track labels

    Returns:
        pd.DataFrame: The dataframe of track data for one time step (specified by idx)
    """
    props = regionprops_table(
        np.asarray(stack[idx, ...]), properties=("label", "centroid")
    )
    props["t"] = np.full(props["label"].shape, idx)
    return pd.DataFrame(props)
