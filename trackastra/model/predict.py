import logging
import warnings

import numpy as np
import torch
from scipy.sparse import SparseEfficiencyWarning, csr_array
from tqdm import tqdm

from trackastra.data import collate_sequence_padding
from trackastra.model import TrackingTransformer

warnings.simplefilter("ignore", SparseEfficiencyWarning)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def predict(batch: list[dict], model: TrackingTransformer) -> np.ndarray:
    """Predict association scores between objects in a batch of windows.

    Args:
        batch: List of dictionaries containing:
            - features: Object features array
            - coords: Object coordinates array
            - timepoints: Time points array
        model: TrackingTransformer model to use for prediction.

    Returns:
        Array of association scores between objects.
    """
    # feats = torch.stack([torch.from_numpy(b["features"]) for b in batch])
    # coords = torch.stack([torch.from_numpy(b["coords"]) for b in batch])
    # timepoints = torch.stack([torch.from_numpy(b["timepoints"]) for b in batch]).long()

    padded_batch = collate_sequence_padding(batch)
    try:
        if padded_batch["features"] is not None:
            feats = padded_batch["features"]
        else:
            feats = None
    except KeyError:
        feats = None
    try:
        if padded_batch["pretrained_feats"] is not None:
            pretrained_feats = padded_batch["pretrained_feats"]
        else:
            pretrained_feats = None
    except KeyError:
        pretrained_feats = None

    coords = padded_batch["coords"]
    timepoints = padded_batch["timepoints"].long()
    padding_mask = padded_batch["padding_mask"]

    # Hack that assumes that all parameters of a model are on the same device
    device = next(model.parameters()).device
    if feats is not None:
        feats = feats.to(device)
    if pretrained_feats is not None:
        pretrained_feats = pretrained_feats.unsqueeze(0).to(device)
    timepoints = timepoints.to(device)
    coords = coords.to(device)
    padding_mask = padding_mask.to(device)

    # Concat timepoints to coordinates
    coords = torch.cat((timepoints.unsqueeze(2).float(), coords), dim=2)
    with torch.no_grad():
        if pretrained_feats is None:
            A = model(coords, features=feats, padding_mask=padding_mask)
        else:
            A = model(
                coords,
                features=feats,
                pretrained_features=pretrained_feats,
                padding_mask=padding_mask,
            )

        A = model.normalize_output(A, timepoints, coords)

        # # Spatially far entries should not influence the causal normalization
        # dist = torch.cdist(coords[0, :, 1:], coords[0, :, 1:])
        # invalid = dist > model.config["spatial_pos_cutoff"]
        # A[invalid] = -torch.inf

        # TODO stay on device for further computation?
        A = A.detach().cpu().numpy()

    return A


def predict_windows(
    windows: list[dict],
    # features: list[WRFeatures],
    # model: TrackingTransformer,
    features: list,
    model,
    intra_window_weight: float = 0,
    delta_t: int = 1,
    edge_threshold: float = 0.05,
    spatial_dim: int = 3,
    batch_size: int = 1,
    progbar_class=tqdm,
) -> dict:
    """Predict associations between objects across sliding windows.

    This function processes a sequence of sliding windows to predict associations
    between objects across time frames. It handles:
    - Object tracking across time
    - Weight normalization across windows
    - Edge thresholding
    - Time-based filtering

    Args:
        windows: List of window dictionaries containing:
            - timepoints: Array of time points
            - labels: Array of object labels
            - features: Object features
            - coords: Object coordinates
        features: List of feature objects containing:
            - labels: Object labels
            - timepoints: Time points
            - coords: Object coordinates
        model: TrackingTransformer model to use for prediction.
        intra_window_weight: Weight factor for objects in middle of window. Defaults to 0.
        delta_t: Maximum time difference between objects to consider. Defaults to 1.
        edge_threshold: Minimum association score to consider. Defaults to 0.05.
        spatial_dim: Dimensionality of input masks. May be less than model.coord_dim.
        batch_size: Number of windows to predict on in parallel. Defaults to 1.
        progbar_class: Progress bar class to use. Defaults to tqdm.

    Returns:
        Dictionary containing:
            - nodes: List of node properties (id, coords, time, label)
            - weights: Tuple of ((node_i, node_j), weight) pairs
    """
    # first get all objects/coords
    time_labels_to_id = dict()
    node_properties = list()
    max_id = np.sum([len(f.labels) for f in features])

    all_timepoints = np.concatenate([f.timepoints for f in features])
    all_labels = np.concatenate([f.labels for f in features])
    all_coords = np.concatenate([f.coords for f in features])
    all_coords = all_coords[:, -spatial_dim:]

    for i, (t, la, c) in enumerate(zip(all_timepoints, all_labels, all_coords)):
        time_labels_to_id[(t, la)] = i
        node_properties.append(
            dict(
                id=i,
                coords=tuple(c),
                time=t,
                # index=ix,
                label=la,
            )
        )

    # create assoc matrix between ids
    sp_weights, sp_accum = (
        csr_array((max_id, max_id), dtype=np.float32),
        csr_array((max_id, max_id), dtype=np.float32),
    )

    for t in progbar_class(
        range(0, len(windows), batch_size),
        desc="Computing associations",
    ):
        # This assumes that the samples in the dataset are ordered by time and start at 0.
        batch = windows[t : t + batch_size]
        lengths = [len(b["timepoints"]) for b in batch]
        if np.all(np.array(lengths) == 0):
            logger.warning(f"No detections in window {t} - {t + batch_size}, skipping")
            continue

        A_batch = predict(batch, model)

        for i, A in enumerate(A_batch):
            timepoints = batch[i]["timepoints"].numpy()
            labels = batch[i]["labels"].numpy()

            dt = timepoints[None, :] - timepoints[:, None]
            time_mask = np.logical_and(dt <= delta_t, dt > 0)
            A = A[: len(timepoints), : len(timepoints)]
            A[~time_mask] = 0
            ii, jj = np.where(A >= edge_threshold)

            if len(ii) == 0:
                continue

            labels_ii = labels[ii]
            labels_jj = labels[jj]
            ts_ii = timepoints[ii]
            ts_jj = timepoints[jj]
            nodes_ii = np.array(
                tuple(time_labels_to_id[(t, lab)] for t, lab in zip(ts_ii, labels_ii))
            )
            nodes_jj = np.array(
                tuple(time_labels_to_id[(t, lab)] for t, lab in zip(ts_jj, labels_jj))
            )

            # weight middle parts higher
            t_middle = t + (model.config["window"] - 1) / 2
            ddt = timepoints[:, None] - t_middle * np.ones_like(dt)
            window_weight = np.exp(-intra_window_weight * ddt**2)  # default is 1
            # window_weight = np.exp(4*A) # smooth max
            sp_weights[nodes_ii, nodes_jj] += window_weight[ii, jj] * A[ii, jj]
            sp_accum[nodes_ii, nodes_jj] += window_weight[ii, jj]

    sp_weights_coo = sp_weights.tocoo()
    sp_accum_coo = sp_accum.tocoo()
    assert np.allclose(sp_weights_coo.col, sp_accum_coo.col) and np.allclose(
        sp_weights_coo.row, sp_accum_coo.row
    )

    # Normalize weights by the number of times they were written from different sliding window positions
    weights = tuple(
        ((i, j), v / a)
        for i, j, v, a in zip(
            sp_weights_coo.row,
            sp_weights_coo.col,
            sp_weights_coo.data,
            sp_accum_coo.data,
        )
    )

    results = dict()
    results["nodes"] = node_properties
    results["weights"] = weights

    return results
