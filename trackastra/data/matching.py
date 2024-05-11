# Adapted from https://github.com/stardist/stardist/blob/master/stardist/matching.py

import numpy as np
from numba import jit
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from skimage.measure import regionprops

matching_criteria = dict()


def label_are_sequential(y):
    """Returns true if y has only sequential labels from 1..."""
    labels = np.unique(y)
    return (set(labels) - {0}) == set(range(1, 1 + labels.max()))


def is_array_of_integers(y):
    return isinstance(y, np.ndarray) and np.issubdtype(y.dtype, np.integer)


def _check_label_array(y, name=None, check_sequential=False):
    err = ValueError(
        "{label} must be an array of {integers}.".format(
            label="labels" if name is None else name,
            integers=("sequential " if check_sequential else "")
            + "non-negative integers",
        )
    )

    if not is_array_of_integers(y):
        raise err
    if len(y) == 0:
        return True
    if check_sequential and not label_are_sequential(y):
        raise err
    else:
        if not y.min() >= 0:
            raise err
    return True


def label_overlap(x, y, check=True):
    if check:
        _check_label_array(x, "x", True)
        _check_label_array(y, "y", True)
        if not x.shape == y.shape:
            raise ValueError("x and y must have the same shape")
    return _label_overlap(x, y)


@jit(nopython=True)
def _label_overlap(x, y):
    x = x.ravel()
    y = y.ravel()
    overlap = np.zeros((1 + x.max(), 1 + y.max()), dtype=np.uint32)
    for i in range(len(x)):
        overlap[x[i], y[i]] += 1
    return overlap[1:, 1:]


def _safe_divide(x, y, eps=1e-10):
    """Computes a safe divide which returns 0 if y is zero."""
    if np.isscalar(x) and np.isscalar(y):
        return x / y if np.abs(y) > eps else 0.0
    else:
        out = np.zeros(np.broadcast(x, y).shape, np.float32)
        np.divide(x, y, out=out, where=np.abs(y) > eps)
        return out


def intersection_over_union(overlap):
    _check_label_array(overlap, "overlap")
    if np.sum(overlap) == 0:
        return overlap
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    return _safe_divide(overlap, (n_pixels_pred + n_pixels_true - overlap))


def dist_score(y_true, y_pred, max_distance: int = 10):
    """Compute distance score between centroids of regions in y_true and y_pred
    and returns a score matrix of shape (n_true, n_pred) with values in [0,1]
    where
    distance >= max_distance  -> score = 0
    distance = 0             -> score = 1.
    """
    c_true = np.stack([r.centroid for r in regionprops(y_true)], axis=0)
    c_pred = np.stack([r.centroid for r in regionprops(y_pred)], axis=0)
    dist = np.minimum(cdist(c_true, c_pred), max_distance)
    score = 1 - dist / max_distance
    return score


# copied from scikit-image master for now (remove when part of a release)
def relabel_sequential(label_field, offset=1):
    """Relabel arbitrary labels to {`offset`, ... `offset` + number_of_labels}.

    This function also returns the forward map (mapping the original labels to
    the reduced labels) and the inverse map (mapping the reduced labels back
    to the original ones).

    Parameters
    ----------
    label_field : numpy array of int, arbitrary shape
        An array of labels, which must be non-negative integers.
    offset : int, optional
        The return labels will start at `offset`, which should be
        strictly positive.

    Returns:
    -------
    relabeled : numpy array of int, same shape as `label_field`
        The input label field with labels mapped to
        {offset, ..., number_of_labels + offset - 1}.
        The data type will be the same as `label_field`, except when
        offset + number_of_labels causes overflow of the current data type.
    forward_map : numpy array of int, shape ``(label_field.max() + 1,)``
        The map from the original label space to the returned label
        space. Can be used to re-apply the same mapping. See examples
        for usage. The data type will be the same as `relabeled`.
    inverse_map : 1D numpy array of int, of length offset + number of labels
        The map from the new label space to the original space. This
        can be used to reconstruct the original label field from the
        relabeled one. The data type will be the same as `relabeled`.

    Notes:
    -----
    The label 0 is assumed to denote the background and is never remapped.

    The forward map can be extremely big for some inputs, since its
    length is given by the maximum of the label field. However, in most
    situations, ``label_field.max()`` is much smaller than
    ``label_field.size``, and in these cases the forward map is
    guaranteed to be smaller than either the input or output images.

    Examples:
    --------
    >>> from skimage.segmentation import relabel_sequential
    >>> label_field = np.array([1, 1, 5, 5, 8, 99, 42])
    >>> relab, fw, inv = relabel_sequential(label_field)
    >>> relab
    array([1, 1, 2, 2, 3, 5, 4])
    >>> fw
    array([0, 1, 0, 0, 0, 2, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5])
    >>> inv
    array([ 0,  1,  5,  8, 42, 99])
    >>> (fw[label_field] == relab).all()
    True
    >>> (inv[relab] == label_field).all()
    True
    >>> relab, fw, inv = relabel_sequential(label_field, offset=5)
    >>> relab
    array([5, 5, 6, 6, 7, 9, 8])
    """
    offset = int(offset)
    if offset <= 0:
        raise ValueError("Offset must be strictly positive.")
    if np.min(label_field) < 0:
        raise ValueError("Cannot relabel array that contains negative values.")
    max_label = int(label_field.max())  # Ensure max_label is an integer
    if not np.issubdtype(label_field.dtype, np.integer):
        new_type = np.min_scalar_type(max_label)
        label_field = label_field.astype(new_type)
    labels = np.unique(label_field)
    labels0 = labels[labels != 0]
    new_max_label = offset - 1 + len(labels0)
    new_labels0 = np.arange(offset, new_max_label + 1)
    output_type = label_field.dtype
    required_type = np.min_scalar_type(new_max_label)
    if np.dtype(required_type).itemsize > np.dtype(label_field.dtype).itemsize:
        output_type = required_type
    forward_map = np.zeros(max_label + 1, dtype=output_type)
    forward_map[labels0] = new_labels0
    inverse_map = np.zeros(new_max_label + 1, dtype=output_type)
    inverse_map[offset:] = labels0
    relabeled = forward_map[label_field]
    return relabeled, forward_map, inverse_map


def matching(y_true, y_pred, threshold=0.5, max_distance: int = 16):
    """Computes IoU and distance score between all pairs of regions in y_true and y_pred.

    returns the true/pred matching based on the higher of the two scores for each pair of regions

    Parameters
    ----------
    y_true: ndarray
        ground truth label image (integer valued)
    y_pred: ndarray
        predicted label image (integer valued)
    threshold: float
        threshold for matching criterion (default 0.5)
    max_distance: int
        maximum distance between centroids of regions in y_true and y_pred (default 16)

    Returns:
    -------
    gt_pred: tuple
        tuple of all matched region label pairs in y_true and y_pred


    """
    y_true, y_pred = y_true.astype(np.int32), y_pred.astype(np.int32)
    _check_label_array(y_true, "y_true")
    _check_label_array(y_pred, "y_pred")
    if not y_true.shape == y_pred.shape:
        raise ValueError(
            f"y_true ({y_true.shape}) and y_pred ({y_pred.shape}) have different shapes"
        )
    if threshold is None:
        threshold = 0

    threshold = float(threshold) if np.isscalar(threshold) else map(float, threshold)

    y_true, _, map_rev_true = relabel_sequential(y_true)
    y_pred, _, map_rev_pred = relabel_sequential(y_pred)

    overlap = label_overlap(y_true, y_pred, check=False)

    scores_iou = intersection_over_union(overlap)
    scores_dist = dist_score(y_true, y_pred, max_distance)
    scores = np.maximum(scores_iou, scores_dist)

    assert 0 <= np.min(scores) <= np.max(scores) <= 1

    n_true, n_pred = scores.shape
    n_matched = min(n_true, n_pred)

    # not_trivial = n_matched > 0 and np.any(scores >= thr)
    not_trivial = n_matched > 0
    if not_trivial:
        # compute optimal matching with scores as tie-breaker
        costs = -(scores >= threshold).astype(float) - scores / (2 * n_matched)
        true_ind, pred_ind = linear_sum_assignment(costs)
        assert n_matched == len(true_ind) == len(pred_ind)
        match_ok = scores[true_ind, pred_ind] >= threshold
        true_ind = true_ind[match_ok]
        pred_ind = pred_ind[match_ok]
        matched = tuple(
            (int(map_rev_true[i]), int(map_rev_pred[j]))
            for i, j in zip(1 + true_ind, 1 + pred_ind)
        )
    else:
        matched = ()

    return matched
