"""Adapted from Fast R-CNN
Written by Sergey Karayev
Licensed under The MIT License
Copyright (c) 2015 Microsoft.
"""

import numpy as np
from skimage.measure import regionprops


def _union_slice(a: tuple[slice], b: tuple[slice]):
    """Returns the union of slice tuples a and b."""
    starts = tuple(min(_a.start, _b.start) for _a, _b in zip(a, b))
    stops = tuple(max(_a.stop, _b.stop) for _a, _b in zip(a, b))
    return tuple(slice(start, stop) for start, stop in zip(starts, stops))


def get_labels_with_overlap(gt_frame, res_frame):
    """Get all labels IDs in gt_frame and res_frame whose bounding boxes
    overlap.

    Args:
        gt_frame (np.ndarray): ground truth segmentation for a single frame
        res_frame (np.ndarray): result segmentation for a given frame

    Returns:
        overlapping_gt_labels: List[int], labels of gt boxes that overlap with res boxes
        overlapping_res_labels: List[int], labels of res boxes that overlap with gt boxes
        intersections_over_gt: List[float], list of (intersection gt vs res) / (gt area)
    """
    gt_frame = gt_frame.astype(np.uint16, copy=False)
    res_frame = res_frame.astype(np.uint16, copy=False)
    gt_props = regionprops(gt_frame)
    gt_boxes = [np.array(gt_prop.bbox) for gt_prop in gt_props]
    gt_boxes = np.array(gt_boxes).astype(np.float64)
    gt_box_labels = np.asarray(
        [int(gt_prop.label) for gt_prop in gt_props], dtype=np.uint16
    )

    res_props = regionprops(res_frame)
    res_boxes = [np.array(res_prop.bbox) for res_prop in res_props]
    res_boxes = np.array(res_boxes).astype(np.float64)
    res_box_labels = np.asarray(
        [int(res_prop.label) for res_prop in res_props], dtype=np.uint16
    )
    if len(gt_props) == 0 or len(res_props) == 0:
        return [], [], []

    if gt_frame.ndim == 3:
        overlaps = compute_overlap_3D(gt_boxes, res_boxes)
    else:
        overlaps = compute_overlap(
            gt_boxes, res_boxes
        )  # has the form [gt_bbox, res_bbox]

    # Find the bboxes that have overlap at all (ind_ corresponds to box number - starting at 0)
    ind_gt, ind_res = np.nonzero(overlaps)
    ind_gt = np.asarray(ind_gt, dtype=np.uint16)
    ind_res = np.asarray(ind_res, dtype=np.uint16)
    overlapping_gt_labels = gt_box_labels[ind_gt]
    overlapping_res_labels = res_box_labels[ind_res]

    intersections_over_gt = []
    for i, j in zip(ind_gt, ind_res):
        sslice = _union_slice(gt_props[i].slice, res_props[j].slice)
        gt_mask = gt_frame[sslice] == gt_box_labels[i]
        res_mask = res_frame[sslice] == res_box_labels[j]
        area_inter = np.count_nonzero(np.logical_and(gt_mask, res_mask))
        area_gt = np.count_nonzero(gt_mask)
        intersections_over_gt.append(area_inter / area_gt)

    return overlapping_gt_labels, overlapping_res_labels, intersections_over_gt


def compute_overlap(boxes: np.ndarray, query_boxes: np.ndarray) -> np.ndarray:
    """Args:
        a: (N, 4) ndarray of float
        b: (K, 4) ndarray of float.

    Returns:
        overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=np.float64)
    for k in range(K):
        box_area = (query_boxes[k, 2] - query_boxes[k, 0] + 1) * (
            query_boxes[k, 3] - query_boxes[k, 1] + 1
        )
        for n in range(N):
            iw = (
                min(boxes[n, 2], query_boxes[k, 2])
                - max(boxes[n, 0], query_boxes[k, 0])
                + 1
            )
            if iw > 0:
                ih = (
                    min(boxes[n, 3], query_boxes[k, 3])
                    - max(boxes[n, 1], query_boxes[k, 1])
                    + 1
                )
                if ih > 0:
                    ua = np.float64(
                        (boxes[n, 2] - boxes[n, 0] + 1)
                        * (boxes[n, 3] - boxes[n, 1] + 1)
                        + box_area
                        - iw * ih
                    )
                    overlaps[n, k] = iw * ih / ua
    return overlaps


def compute_overlap_3D(boxes: np.ndarray, query_boxes: np.ndarray) -> np.ndarray:
    """Args:
        a: (N, 6) ndarray of float
        b: (K, 6) ndarray of float.

    Returns:
        overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=np.float64)
    for k in range(K):
        box_volume = (
            (query_boxes[k, 3] - query_boxes[k, 0] + 1)
            * (query_boxes[k, 4] - query_boxes[k, 1] + 1)
            * (query_boxes[k, 5] - query_boxes[k, 2] + 1)
        )
        for n in range(N):
            id_ = (
                min(boxes[n, 3], query_boxes[k, 3])
                - max(boxes[n, 0], query_boxes[k, 0])
                + 1
            )
            if id_ > 0:
                iw = (
                    min(boxes[n, 4], query_boxes[k, 4])
                    - max(boxes[n, 1], query_boxes[k, 1])
                    + 1
                )
                if iw > 0:
                    ih = (
                        min(boxes[n, 5], query_boxes[k, 5])
                        - max(boxes[n, 2], query_boxes[k, 2])
                        + 1
                    )
                    if ih > 0:
                        ua = np.float64(
                            (boxes[n, 3] - boxes[n, 0] + 1)
                            * (boxes[n, 4] - boxes[n, 1] + 1)
                            * (boxes[n, 5] - boxes[n, 2] + 1)
                            + box_volume
                            - iw * ih * id_
                        )
                        overlaps[n, k] = iw * ih * id_ / ua
    return overlaps


try:
    import numba
except ImportError:
    import os
    import warnings

    if not os.getenv("NO_JIT_WARNING", False):
        warnings.warn(
            "Numba not installed, falling back to slower numpy implementation. "
            "Install numba for a significant speedup.  Set the environment "
            "variable NO_JIT_WARNING=1 to disable this warning.",
            stacklevel=2,
        )
else:
    # compute_overlap 2d and 3d have the same signature
    signature = [
        "f8[:,::1](f8[:,::1], f8[:,::1])",
        numba.types.Array(numba.float64, 2, "C", readonly=True)(
            numba.types.Array(numba.float64, 2, "C", readonly=True),
            numba.types.Array(numba.float64, 2, "C", readonly=True),
        ),
    ]

    # variables that appear in the body of each function
    common_locals = {
        "N": numba.uint64,
        "K": numba.uint64,
        "overlaps": numba.types.Array(numba.float64, 2, "C"),
        "iw": numba.float64,
        "ih": numba.float64,
        "ua": numba.float64,
        "n": numba.uint64,
        "k": numba.uint64,
    }

    compute_overlap = numba.njit(
        signature,
        locals={**common_locals, "box_area": numba.float64},
        fastmath=True,
        nogil=True,
        boundscheck=False,
    )(compute_overlap)

    compute_overlap_3D = numba.njit(
        signature,
        locals={**common_locals, "id_": numba.float64, "box_volume": numba.float64},
        fastmath=True,
        nogil=True,
        boundscheck=False,
    )(compute_overlap_3D)
