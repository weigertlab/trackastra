"""Updated evaluation metrics computation for CTC datasets.

Built for :
- traccuracy 0.3.0
- motmetrics 1.4.0
"""
import contextlib
import logging
from contextlib import nullcontext
from copy import deepcopy
from functools import partialmethod
from itertools import product
from pathlib import Path
from timeit import default_timer

import motmetrics as mm
import numpy as np
import pandas as pd
import tqdm
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from traccuracy import TrackingGraph as TrackingData
from traccuracy import run_metrics
from traccuracy.loaders import load_ctc_data as _load_ctc_data
from traccuracy.matchers import CTCMatcher, IOUMatcher
from traccuracy.metrics import AOGMMetrics, CTCMetrics, DivisionMetrics

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class FoundTracks(Exception):
    pass


def load_ctc_graph(path: str | Path):
    """Load a ctc dataset and convert it to traccuracy TrackingData
    (that contains the tracking graph and the segmentation data).

    Returns:
        traccuracy._tracking_data.TrackingData: tracking data with graph and segmentation
    """
    path = Path(path).expanduser().resolve()
    tracks_globs = ["man_track.txt", "res_track.txt", "*.txt"]

    try:
        for _glob in tracks_globs:
            print(f"Trying to load tracks with `glob {path / _glob}`")
            for _cand in path.glob(_glob):
                tracks_file = path / _cand
                if tracks_file.exists():
                    man_track = tracks_file
                    raise FoundTracks
    except FoundTracks:
        print(f"Found tracks file {tracks_file}")
    else:
        raise ValueError(f"Did not find a .txt file with tracks in {path}.")

    graph = _load_ctc_data(path, man_track)

    return graph


def metrics(
    gt,
    pred,
    outdir=None,
    start=0,
    stop=None,
    every=1,
    n_timesteps=None,
    n_workers=8,
    dry=False,
    division_frame_buffer=1,
    which=["ctc", "divisions", "mot", "aa"],
    # which=["ctc"],
):
    """Args:
    gt (str): path to ground truth
    pred (str): path to predictions
    outdir (str): path to output directory
    start(int): Don't consider frames before this one
    stop (int): end frame
    every (int): compute metrics every n frames
    n_timesteps (int): number of timesteps to compute metrics for
    n_workers (int): number of workers for parallel processing
    dry (bool): dry run
    division_frame_buffer (int): number of frame difference to tolerate for counting division events as correct.
    """
    # if division_frame_buffer > 0:
    # raise NotImplementedError(
    # "Frame buffer still buggy in our pinned version of traccuracy"
    # )

    assert start >= 0

    if outdir is None:
        outdir = Path(pred)
    elif len(str(outdir)) == 0:
        outdir = None
    else:
        outdir = Path(outdir)

    if outdir is not None and not dry:
        outdir.mkdir(exist_ok=True, parents=True)

    gt_path = Path(gt)
    pred_path = Path(pred)

    gt = load_ctc_graph(gt_path)
    pred = load_ctc_graph(pred_path)

    if stop is None:
        stop = len(gt.segmentation)
    logging.info(f"Computing metrics on video from frame {start} to frame {stop}")

    t = default_timer()

    every = max(1, stop // n_timesteps) if n_timesteps is not None else every

    # Makes sure we compute metrics until the last frame
    untils = range(
        stop, start + 1, -every
    )  # need at least two frames to compute metrics
    print(f"Computing metrics for frames {list(untils)}")

    if n_workers > 0:
        res = Parallel(n_jobs=n_workers)(
            delayed(get_metrics)(
                gt,
                pred,
                start=start,
                stop=n,
                division_frame_buffer=division_frame_buffer,
                which_metrics=which,
            )
            for n in tqdm.tqdm(untils)
        )
    else:
        res = tuple(
            get_metrics(
                gt,
                pred,
                start=start,
                stop=n,
                division_frame_buffer=division_frame_buffer,
                which_metrics=which,
            )
            for n in tqdm.tqdm(untils)
        )

    res = sorted(res, key=lambda x: x[0])

    rows = []
    division_events = []
    edge_errors = []
    for _res in res:
        r = _res[1]["scores"]
        row = {}
        if "ctc" in which:
            # row.update(r["CTCMetrics"])
            row.update(r["CTC"])
        if "divisions" in which:
            division_events.append(_res[1]["division_events"])
            row.update(r["divisions"])
        if "edge_errors" in _res[1]:
            edge_errors.append(_res[1]["edge_errors"])

        if "mot" in which:
            row.update(r["MOTMetrics"])
        if "aa" in which:
            row.update(r["association_metrics"])
        rows.append(row)

    df = pd.DataFrame.from_records(rows)
    # TODO mot metrics are now computed twice on the whole video
    mot_events = compute_metrics_mot(gt, pred)[1] if "mot" in which else None
    edge_errors = edge_errors[-1] if "ctc" in which else None
    division_events = division_events[-1] if "divisions" in which else None

    if outdir is not None and not dry:
        print(f"Writing to {outdir}")
        df.to_csv(outdir / "metrics.csv", index=False)
        if mot_events is not None:
            mot_events.to_csv(outdir / "mot_events.csv", index=False)
        if division_events is not None:
            division_events.to_csv(outdir / "division_events.csv", index=False)
        if edge_errors is not None:
            edge_errors.to_csv(outdir / "edge_errors.csv", index=False)

    t = default_timer() - t
    print(f"Time: {t:.2f} s")

    return df, mot_events, division_events, edge_errors


def crop_graph(graph: TrackingData, start: int, stop: int):
    """Crop graph to a certain length.

    Args:
        graph (traccuracy.TrackingGraph): tracking graph
        start (int): start frame
        end (int): end frame
    Returns:
        traccuracy.TrackingGraph: copy of cropped tracking graph
    """
    g = deepcopy(graph)

    g.segmentation = g.segmentation[start:stop]
    g.start_frame = start
    g.end_frame = stop
    for n in tuple(g.graph.nodes()):
        if (
            g.graph.nodes[n]["t"] >= stop
            or g.graph.nodes[n]["t"] < start
        ):
            g.graph.remove_node(n)

    return g


@contextlib.contextmanager
def monkeypatched(obj, name, patch):
    """Temporarily monkeypatch."""
    old_attr = getattr(obj, name)
    setattr(obj, name, patch(old_attr))
    try:
        yield
    finally:
        setattr(obj, name, old_attr)


@contextlib.contextmanager
def disable_tqdm():
    """Context manager to disable tqdm."""

    def _patch(old_init):
        return partialmethod(old_init, disable=True)

    with monkeypatched(tqdm.std.tqdm, "__init__", _patch):
        yield


def compute_metrics_divisions(
    gt_graph, pred_graph, return_events=False, frame_buffer: int = 1, verbose=True
):

    context = disable_tqdm() if not verbose else nullcontext()
    # NOTE : FN Divisions are not consistent with previous versions
    # Whether this is due to a parameter change or an actual fix is unclear for now
    with context:
        logger.info("Compute DivisionMetrics")
        res, _matcher = run_metrics(
            gt_data=deepcopy(gt_graph),
            pred_data=deepcopy(pred_graph),
            matcher=IOUMatcher(one_to_one=True),
            metrics=[DivisionMetrics(max_frame_buffer=frame_buffer)],
            # metrics_kwargs=dict(frame_buffer=tuple(range(frame_buffer + 1))),
        )
        res = res[0]
    if not return_events:
        return res, None

    # TODO not clear how different frame buffers are handled, since the errors are stored on the graph
    cols_pred = ["is_tp_division", "is_fp_division"]
    cols_gt = ["is_fn_division"]
    divisions_pred = [
        v
        for k, v in pred_graph.graph.nodes().items()
        if any(v.get(c, False) for c in cols_pred)
    ]
    # FN are stored on ground truth graph
    divisions_gt = [
        v
        for k, v in gt_graph.graph.nodes().items()
        if any(v.get(c, False) for c in cols_gt)
    ]
    columns = cols_pred + cols_gt
    divisions = divisions_pred + divisions_gt

    divisions = pd.DataFrame.from_records(divisions)
    for c in columns:
        if c not in divisions.columns:
            divisions[c] = False

    if len(divisions) == 0:
        # pandas syntactic salt
        divisions["t"] = None
        divisions["segmentation_id"] = None

    columns.extend(["t", "segmentation_id"])
    divisions = divisions[columns]
    divisions.sort_values(by="t", inplace=True, ignore_index=True)
    divisions.fillna(False, inplace=True)

    return res, divisions


def compute_metrics_ctc(gt_graph, pred_graph, return_events=False, verbose=True):
    start = default_timer()
    logger.info("Computing CTCMetrics")
    context = disable_tqdm() if not verbose else nullcontext()
    with context:
        res, _matcher = run_metrics(
            gt_data=gt_graph,
            pred_data=pred_graph,
            matcher=CTCMatcher(),
            metrics=[CTCMetrics(), AOGMMetrics()],
        )
        res = res[0]
        res["CTCMetrics"] = {}
        res["CTCMetrics"]["AOGM_unweighted"] = res["results"]["AOGM"]
        # Hack
        res["CTCMetrics"]["AOGM-A"] = (
            res["results"]["fp_edges"]
            + 1.5 * res["results"]["fn_edges"]
            + res["results"]["ws_edges"]
        )
        # del res["AOGMMetrics"]

    if not return_events:
        logger.info(f"CTCMetrics computed in {default_timer() - start:.1f} seconds")
        return res, None

    # TODO include ctc node events

    # Edge errors
    edge_fp = []
    for (u, v), attrs in pred_graph.graph.edges().items():
        if attrs.get("is_fp", False):
            fp = deepcopy(attrs)
            fp["u"], fp["t_u"] = map(int, str(u).split("_"))
            fp["v"], fp["t_v"] = map(int, str(v).split("_"))
            edge_fp.append(fp)
    edge_fn = []
    for (u, v), attrs in gt_graph.graph.edges().items():
        if attrs.get("is_fn", False):
            fn = deepcopy(attrs)
            fn["u"], fn["t_u"] = map(int, str(u).split("_"))
            fn["v"], fn["t_v"] = map(int, str(v).split("_"))
            edge_fn.append(fn)
    edge_errors = edge_fp + edge_fn
    edge_errors = pd.DataFrame.from_records(edge_errors)

    if len(edge_errors) > 0:
        for c in ["is_fp", "is_fn"]:
            if c not in edge_errors.columns:
                edge_errors[c] = False
        edge_errors.sort_values(by="t_u", inplace=True, ignore_index=True)
        edge_errors.fillna(False, inplace=True)

    logger.info(f"CTCMetrics computed in {default_timer() - start:.1f} seconds")

    return res, edge_errors


def _graph_to_mo16(graph: TrackingData):
    g = graph
    rows = []
    nodes = g.nodes()
    for i, t in enumerate(range(g.start_frame, g.end_frame)):
        t_nodes = g.nodes_by_frame.get(t, [])
        for n in t_nodes:
            n = nodes[n]
            label = n["segmentation_id"]
            x, y = n["x"], n["y"]
            w = 1e-2
            bx, by = x - w / 2, y - w / 2
            # MOT 16 <frame number>, <object id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <confidence>, <x>, <y>, <z>
            rows.append([i + 1, label, bx, by, w, w, 1.0, -1, -1, -1])
    rows = np.stack(rows)
    return rows

def iou_matrix(objs, hyps, max_iou=1.):
    # Monkey patched due to np.asfarray usage (deprecated)
    """Computes 'intersection over union (IoU)' distance matrix between object and hypothesis rectangles.

    The IoU is computed as

        IoU(a,b) = 1. - isect(a, b) / union(a, b)

    where isect(a,b) is the area of intersection of two rectangles and union(a, b) the area of union. The
    IoU is bounded between zero and one. 0 when the rectangles overlap perfectly and 1 when the overlap is
    zero.

    Params
    ------
    objs : Nx4 array
        Object rectangles (x,y,w,h) in rows
    hyps : Kx4 array
        Hypothesis rectangles (x,y,w,h) in rows

    Kwargs
    ------
    max_iou : float
        Maximum tolerable overlap distance. Object / hypothesis points
        with larger distance are set to np.nan signalling do-not-pair. Defaults
        to 0.5

    Returns
    -------
    C : NxK array
        Distance matrix containing pairwise distances or np.nan.
    """

    if np.size(objs) == 0 or np.size(hyps) == 0:
        return np.empty((0, 0))

    objs = np.asarray(objs).astype(float)
    hyps = np.asarray(hyps).astype(float)
    assert objs.shape[1] == 4
    assert hyps.shape[1] == 4
    iou = mm.distances.boxiou(objs[:, None], hyps[None, :])
    dist = 1 - iou
    return np.where(dist > max_iou, np.nan, dist)


def compute_metrics_mot(gt: TrackingData, pred: TrackingData):
    """https://github.com/cheind/py-motmetrics#for-custom-dataset."""
    gt = _graph_to_mo16(gt)
    pred = _graph_to_mo16(pred)

    acc = mm.MOTAccumulator(auto_id=True)

    # Max frame number maybe different for gt and t files
    for frame in range(int(gt[:, 0].min()), int(gt[:, 0].max())):
        frame += 1
        # select id, x, y, width, height for current frame
        # required format for distance calculation is X, Y, Width, Height
        gt_dets = gt[gt[:, 0] == frame, 1:6]
        t_dets = pred[pred[:, 0] == frame, 1:6]

        C = iou_matrix(gt_dets[:, 1:], t_dets[:, 1:], max_iou=0.5)

        # Call update once for per frame.
        # format: gt object ids, t object ids, distance
        acc.update(
            gt_dets[:, 0].astype("int").tolist(), t_dets[:, 0].astype("int").tolist(), C
        )

    mh = mm.metrics.create()

    summary = mh.compute(
        acc,
        metrics=[
            "num_frames",
            "idf1",
            "idp",
            "idr",
            "recall",
            "precision",
            "num_objects",
            "mostly_tracked",
            "partially_tracked",
            "mostly_lost",
            "num_false_positives",
            "num_misses",
            "num_switches",
            "num_fragmentations",
            "mota",
            "motp",
        ],
        name="acc",
    )
    mot_event = acc.events.reset_index()
    return dict(summary.iloc[0]), mot_event


def compute_association_metrics(gt: TrackingData, pred: TrackingData):
    """According to Bise et al. MPM: Joint Representation of Motion and Position Map for Cell Tracking."""
    gt_edges = gt.graph.edges()
    pred_edges = pred.graph.edges()

    def get_coords_from_nodes(graph, nodes: list[str]):
        all_nodes = graph.nodes()
        keys = ("t", "y", "x")
        coords = np.array(tuple(tuple(all_nodes[n][k] for k in keys) for n in nodes))
        return coords

    def match(gt_nodes: list[str], pred_nodes: list[str]):
        """Matches pred nodes to gt nodes and return dict of matches."""
        if len(gt_nodes) == 0 or len(pred_nodes) == 0:
            return {}
        gt_coords = get_coords_from_nodes(gt.graph, gt_nodes)
        pred_coords = get_coords_from_nodes(pred.graph, pred_nodes)
        dist = cdist(gt_coords, pred_coords)
        ii, jj = linear_sum_assignment(dist)
        return dict((gt_nodes[i], pred_nodes[j]) for i, j in zip(ii, jj))

    t1, t2 = gt.start_frame, gt.end_frame
    gt_nodes = tuple(gt.nodes_by_frame[t] for t in range(t1, t2))

    # the matching map from gt to pred node keys per timepoint
    map_gt_to_pred = tuple(
        match(
            list(gt.nodes_by_frame[t]), list(pred.nodes_by_frame[t])
        )
        for t in range(t1, t2)
    )

    edge_count = 0
    match_count = 0
    for i, t in tqdm.tqdm(
        enumerate(range(t1, t2 - 1)), desc="association accuracy", leave=False
    ):
        map1, map2 = map_gt_to_pred[i], map_gt_to_pred[i + 1]
        for n1, n2 in product(gt_nodes[i], gt_nodes[i + 1]):
            if (n1, n2) in gt_edges:
                edge_count += 1
                if (map1.get(n1, None), map2.get(n2, None)) in pred_edges:
                    match_count += 1

    acc = match_count / edge_count
    res = dict(association_accuracy=acc, aa_match=match_count, aa_edge_count=edge_count)
    return res


def get_metrics(
    gt: TrackingData,
    pred: TrackingData,
    start=0,
    stop=None,
    division_frame_buffer=1,
    which_metrics=["ctc", "divisions", "mot", "aa"],
):
    if stop is None:
        stop = gt.segmentation.shape[0]

    gt = crop_graph(gt, start, stop)
    pred = crop_graph(pred, start, stop)
    assert len(gt.segmentation) == len(pred.segmentation)

    # CTC metrics
    # start = default_timer()
    # logger.debug("Computing CTC metrics")
    scores = {}
    out = {}
    if "ctc" in which_metrics:
        ctc_scores, edge_errors = compute_metrics_ctc(
            gt,
            pred,
            return_events=False,
        )
        scores["CTC"] = ctc_scores["results"]
        out["edge_errors"] = edge_errors
    else:
        out["edge_errors"] = None
    if "divisions" in which_metrics:
        divisions_scores, division_events = compute_metrics_divisions(
            gt,
            pred,
            return_events=False,
            frame_buffer=division_frame_buffer,
        )
        scores["divisions"] = divisions_scores["results"][f"Frame Buffer {division_frame_buffer}"]
        out["division_events"] = division_events
    else:
        out["division_events"] = None

    for k in scores.keys():
        scores[k].update(start=start, stop=stop)
        scores[k]["gt_nodes"] = len(gt.graph.nodes())
        scores[k]["pred_nodes"] = len(pred.graph.nodes())
        scores[k]["gt_edges"] = len(gt.graph.edges())
        scores[k]["pred_edges"] = len(pred.graph.edges())

    # MOT metrics
    if "mot" in which_metrics:
        start = default_timer()
        logger.debug("Computing MOT metrics")
        scores["MOTMetrics"] = compute_metrics_mot(gt, pred)[0]
        logger.debug(f"in {default_timer() - start:.2f} seconds")

    if "aa" in which_metrics:
        start = default_timer()
        logger.debug("Computing association metrics")
        scores["association_metrics"] = compute_association_metrics(gt, pred)
        logger.debug(f"in {default_timer() - start:.2f} seconds")

    out["scores"] = scores
    return stop, out