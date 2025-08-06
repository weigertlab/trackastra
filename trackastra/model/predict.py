import logging
import warnings

import numpy as np
import torch
from scipy.sparse import SparseEfficiencyWarning, csr_array
from tqdm import tqdm

# TODO fix circular import
# from .model import TrackingTransformer
# from trackastra.data import WRFeatures

warnings.simplefilter("ignore", SparseEfficiencyWarning)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def predict(batch, model):
    """Predict association scores between objects in a batch.
    
    Args:
        batch: Dictionary containing:
            - features: Object features array
            - coords: Object coordinates array
            - timepoints: Time points array
        model: TrackingTransformer model to use for prediction.
    
    Returns:
        Array of association scores between objects.
    """
    if batch["features"] is not None:
        feats = torch.from_numpy(batch["features"])
    else:
        feats = None
    if batch["pretrained_features"] is not None:
        pretrained_feats = torch.from_numpy(batch["pretrained_features"])
    else:
        pretrained_feats = None
    coords = torch.from_numpy(batch["coords"])
    timepoints = torch.from_numpy(batch["timepoints"]).long()
    # Hack that assumes that all parameters of a model are on the same device
    device = next(model.parameters()).device
    coords = coords.unsqueeze(0).to(device)
    timepoints = timepoints.unsqueeze(0).to(device)
    if feats is not None:
        feats = feats.unsqueeze(0).to(device)
    if pretrained_feats is not None:
        pretrained_feats = pretrained_feats.unsqueeze(0).to(device)

    # Concat timepoints to coordinates
    coords = torch.cat((timepoints.unsqueeze(2).float(), coords), dim=2)
    with torch.no_grad():
        A = model(coords, features=feats, pretrained_features=pretrained_feats)
        A = model.normalize_output(A, timepoints, coords)

        # # Spatially far entries should not influence the causal normalization
        # dist = torch.cdist(coords[0, :, 1:], coords[0, :, 1:])
        # invalid = dist > model.config["spatial_pos_cutoff"]
        # A[invalid] = -torch.inf

        A = A.squeeze(0).detach().cpu().numpy()

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
    progbar_class=tqdm,
    pred_func_override=None,
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
        progbar_class: Progress bar class to use. Defaults to tqdm.
        pred_func_override: Function to override the prediction function. This is useful for debugging or testing other prediction methods.

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
    sp_weights, sp_accum = csr_array((max_id, max_id), dtype=np.float32), csr_array(
        (max_id, max_id), dtype=np.float32
    )

    for t in progbar_class(
        range(len(windows)),
        desc="Computing associations",
    ):
        # This assumes that the samples in the dataset are ordered by time and start at 0.
        batch = windows[t]
        timepoints = batch["timepoints"]
        if isinstance(timepoints, torch.Tensor):
            timepoints = timepoints.cpu().numpy()
        labels = batch["labels"]
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
    
        if pred_func_override is None:
            A = predict(batch, model)
        else:
            A = pred_func_override(batch) 
        
        dt = timepoints[None, :] - timepoints[:, None]
        time_mask = np.logical_and(dt <= delta_t, dt > 0)
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
