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
    """Args:
        batch (_type_): _description_
        model (_type_): _description_.

    Returns:
        _type_: _description_
    """
    feats = torch.from_numpy(batch["features"])
    coords = torch.from_numpy(batch["coords"])
    timepoints = torch.from_numpy(batch["timepoints"]).long()
    # Hack that assumes that all parameters of a model are on the same device
    device = next(model.parameters()).device
    feats = feats.unsqueeze(0).to(device)
    timepoints = timepoints.unsqueeze(0).to(device)
    coords = coords.unsqueeze(0).to(device)

    # Concat timepoints to coordinates
    coords = torch.cat((timepoints.unsqueeze(2).float(), coords), dim=2)
    with torch.no_grad():
        A = model(coords, features=feats)
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
) -> dict:
    """_summary_.

    Args:
        windows (_type_): _description_
        features (_type_): _description_
        model (_type_): _description_
        intra_window_weight (_type_, optional): _description_. Defaults to 0.
        delta_t (_type_, optional): _description_. Defaults to 1.
        edge_threshold (_type_, optional): _description_. Defaults to 0.05.
        spatial_dim: Dimensionality of the input masks. This might be < model.coord_dim

    Returns:
        _type_: _description_
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
        labels = batch["labels"]

        A = predict(batch, model)

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
