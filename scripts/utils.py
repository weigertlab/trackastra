"""Visualise tracking errors of a CTC prediction as an overlay movie.

The main entry point is :func:`viz_error`, which renders the input frames with
the ground-truth and predicted tracks overlaid and coloured by error type:

- green:   true-positive links (GT links that were correctly predicted)
- orange:  false-positive links (predicted links absent from the GT)
- magenta: false-negative links (GT links that were not predicted)
- yellow:  wrong-semantic links (correct association, wrong division semantics)

Errors are computed with ``traccuracy`` using the same CTC matcher that
``predict.py`` uses for evaluation, so the colours are consistent with the
reported TRA/AOGM metrics.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

# RGB colours per error class
COLORS = {
    "tp": (60, 220, 60),  # green
    "fp": (255, 150, 0),  # orange
    "fn": (230, 0, 230),  # magenta
    "ws": (255, 230, 0),  # yellow
}


def _load_frames(img_path: Path) -> np.ndarray:
    """Load an image time series, normalise to [0, 1] and scale to uint8 [0, 255]."""
    from trackastra.data.utils import load_tiff_timeseries
    from trackastra.utils import normalize

    img_path = Path(img_path)
    if not (list(img_path.glob("*.tif")) + list(img_path.glob("*.tiff"))):
        if (img_path / "img").exists():
            img_path = img_path / "img"
    x = load_tiff_timeseries(img_path)
    if x.ndim != 3:
        raise ValueError(f"Expected a 2D+t image series (T, Y, X), got shape {x.shape}")
    x = normalize(x)
    # dim the image so the coloured track overlays stand out
    return np.clip(x * 255 / 1.5, 0, 255).astype(np.uint8)


def _annotated_graphs(gt_path: Path, pred_path: Path):
    """Match GT and prediction and annotate edges with CTC error flags."""
    from traccuracy._tracking_graph import EdgeFlag
    from traccuracy.loaders import load_ctc_data
    from traccuracy.matchers import CTCMatcher
    from traccuracy.track_errors import evaluate_ctc_events

    gt = load_ctc_data(str(gt_path), run_checks=False)
    pred = load_ctc_data(str(pred_path), run_checks=False)
    matched = CTCMatcher().compute_mapping(gt, pred)
    evaluate_ctc_events(matched)
    pred_to_gt = {pred_id: gt_id for gt_id, pred_id in matched.mapping}
    return matched.gt_graph, matched.pred_graph, EdgeFlag, pred_to_gt


def _collect_edges(
    gt_graph, pred_graph, edge_flag, pred_to_gt: dict, scale: float
) -> list[dict]:
    """Return drawable links as dicts with target frame, endpoints and colour class.

    Each link connects a source node at time ``t`` to a target node at a later
    time. Coordinates are stored as ``(x, y)`` pixel positions already scaled to
    the resized frame.
    """

    def node_xy(graph, n):
        d = graph.nodes[n]
        return (d["x"] * scale, d["y"] * scale)

    edges = []
    wrong_gt_edges = {
        (pred_to_gt[u], pred_to_gt[v])
        for u, v, data in pred_graph.graph.edges(data=True)
        if data.get(edge_flag.WRONG_SEMANTIC, False)
        and u in pred_to_gt
        and v in pred_to_gt
    }
    # GT edges: false-negative if flagged, otherwise true-positive
    for u, v, d in gt_graph.graph.edges(data=True):
        if (u, v) in wrong_gt_edges:
            continue
        cls = "fn" if d.get(edge_flag.CTC_FALSE_NEG, False) else "tp"
        edges.append({
            "t": gt_graph.nodes[v]["t"],
            "p0": node_xy(gt_graph, u),
            "p1": node_xy(gt_graph, v),
            "cls": cls,
        })
    # Predicted errors: TP edges are already drawn from the GT graph.
    for u, v, d in pred_graph.graph.edges(data=True):
        if d.get(edge_flag.CTC_FALSE_POS, False):
            cls = "fp"
        elif d.get(edge_flag.WRONG_SEMANTIC, False):
            cls = "ws"
        else:
            continue
        edges.append({
            "t": pred_graph.nodes[v]["t"],
            "p0": node_xy(pred_graph, u),
            "p1": node_xy(pred_graph, v),
            "cls": cls,
        })
    return edges


def _draw_legend(draw: ImageDraw.ImageDraw) -> None:
    items = [
        ("TP", COLORS["tp"]),
        ("FP", COLORS["fp"]),
        ("FN", COLORS["fn"]),
        ("WS", COLORS["ws"]),
    ]
    for i, (label, color) in enumerate(items):
        y = 6 + i * 16
        draw.line([(8, y + 6), (26, y + 6)], fill=color, width=3)
        draw.text((32, y), label, fill=(255, 255, 255))


def viz_error(
    img_path: Path | str,
    gt_path: Path | str,
    pred_path: Path | str,
    out_path: Path | str = "errors.mp4",
    size: int = 512,
    fps: int = 5,
    tail: int = 10,
    line_width: int = 2,
) -> Path:
    """Render a movie overlaying TP/FP/FN/WS tracking links on the input frames.

    Args:
        img_path: Folder with the input image frames (``tXXX.tif`` or an ``img``
            subfolder).
        gt_path: Ground-truth CTC TRA folder (``man_trackXXX.tif`` + ``man_track.txt``).
        pred_path: Predicted CTC folder (``man_trackXXX.tif`` + ``man_track.txt``).
        out_path: Output movie path.
        size: Longest side of the rendered frames in pixels; frames are scaled
            to this while preserving aspect ratio.
        fps: Frames per second of the output movie.
        tail: Number of past frames a link stays visible for; each link fades
            from full opacity at its own frame to transparent ``tail`` frames later.
        line_width: Width of the link lines in pixels.

    Returns:
        The path the movie was written to.
    """
    from moviepy import ImageSequenceClip

    img_path, gt_path, pred_path = Path(img_path), Path(gt_path), Path(pred_path)
    out_path = Path(out_path)

    frames = _load_frames(img_path)
    n_frames, height, width = frames.shape
    scale = size / max(height, width)
    out_w, out_h = round(width * scale), round(height * scale)

    gt_graph, pred_graph, edge_flag, pred_to_gt = _annotated_graphs(gt_path, pred_path)
    edges = _collect_edges(gt_graph, pred_graph, edge_flag, pred_to_gt, scale)
    logger.info(
        "links: %d total (%s)",
        len(edges),
        ", ".join(
            f"{c}={sum(e['cls'] == c for e in edges)}"
            for c in ("tp", "fp", "fn", "ws")
        ),
    )

    rendered = []
    for f in range(n_frames):
        rgb = np.repeat(frames[f, :, :, None], 3, axis=2)
        im = Image.fromarray(rgb).resize((out_w, out_h), Image.BILINEAR).convert("RGBA")
        overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        # draw oldest links first so the brightest (most recent) end up on top
        for e in sorted(edges, key=lambda e: e["t"]):
            age = f - e["t"]
            # show only links from the current and past `tail` frames, fading out
            if age < 0 or age > tail:
                continue
            alpha = int(255 * (1 - age / (tail + 1)))
            color = (*COLORS[e["cls"]], alpha)
            draw.line([e["p0"], e["p1"]], fill=color, width=line_width)
        im = Image.alpha_composite(im, overlay).convert("RGB")
        _draw_legend(ImageDraw.Draw(im))
        rendered.append(np.asarray(im))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    clip = ImageSequenceClip(rendered, fps=fps)
    clip.write_videofile(str(out_path), codec="libx264", logger=None)
    logger.info("wrote %s (%d frames)", out_path, n_frames)
    return out_path


# --------------------------------------------------------------------------- #
# Window debug visualisation (training --debug)
# --------------------------------------------------------------------------- #
# Reconstruct each detection's geometry from the *consumed* feature vector and
# draw it as an ellipse, so a sampled training window can be checked for
# augmentation correctness in feature space (post-augmentation there is no
# image; only coords + geometric features exist, and they must stay mutually
# consistent under flip/rotate/scale/crop).


def _inertia_from_feature_row(feat_row: np.ndarray, mode: str) -> np.ndarray:
    """Inverse of ``WRFeatures.features_stacked_for`` for the wrfeat2 family.

    Returns the 2x2 skimage-convention inertia tensor (area-normalised second
    moments, in pixel**2). See ``trackastra/data/wrfeat.py`` (``features_stacked_for``);
    channels are ``[log1p(diam), (intensity), log1p(compactness), q1, q2,
    log1p(border)]`` with ``intensity`` absent for ``wrfeat2_no_intensity``.
    """
    if mode == "wrfeat2":
        diam, comp, q1, q2 = feat_row[0], feat_row[2], feat_row[3], feat_row[4]
    elif mode == "wrfeat2_no_intensity":
        diam, comp, q1, q2 = feat_row[0], feat_row[1], feat_row[2], feat_row[3]
    else:
        raise NotImplementedError(
            f"window debug viz only supports the wrfeat2 family, got {mode!r}"
        )
    diameter = np.expm1(float(diam))
    area = np.pi * (diameter / 2) ** 2
    trace = np.expm1(float(comp)) * area
    i00 = trace * (1 + float(q1)) / 2
    i11 = trace * (1 - float(q1)) / 2
    i01 = float(q2) * trace / 2
    return np.array([[i00, i01], [i01, i11]], dtype=float)


def _ellipse_axes(inertia: np.ndarray) -> tuple[float, float, float]:
    """Major/minor pixel radii and orientation (rad) of the equivalent ellipse.

    Eigen-decomposes the covariance ``[[I00,-I01],[-I01,I11]]`` (skimage's inertia
    tensor carries the off-diagonal with a flipped sign). Radii are ``2*sqrt(lambda)``
    (matching skimage's ``axis_length = 4*sqrt(lambda)``); orientation is the angle
    of the major eigenvector, passed to ``skimage.draw.ellipse_perimeter``.
    """
    cov = np.array(
        [[inertia[0, 0], -inertia[0, 1]], [-inertia[0, 1], inertia[1, 1]]], dtype=float
    )
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending
    eigvals = np.clip(eigvals, 0, None)
    r_major = 2.0 * np.sqrt(eigvals[1])
    r_minor = 2.0 * np.sqrt(eigvals[0])
    v_row, v_col = eigvecs[0, 1], eigvecs[1, 1]  # major eigenvector (row, col)
    orientation = float(np.arctan2(v_row, v_col))
    return float(r_major), float(r_minor), orientation


def _densify_assoc_np(assoc_coo, batch_size: int, n: int) -> np.ndarray:
    """Rebuild a dense ``(B, N, N)`` 0/1 association matrix from COO triples."""
    out = np.zeros((batch_size, n, n), dtype=np.float32)
    coo = np.asarray(assoc_coo.detach().cpu().numpy() if hasattr(assoc_coo, "detach")
                     else assoc_coo)
    if coo.size:
        b, r, c = coo[:, 0].astype(int), coo[:, 1].astype(int), coo[:, 2].astype(int)
        out[b, r, c] = 1.0
    return out


def _render_window(
    coords: np.ndarray,
    feats: np.ndarray,
    timepoints: np.ndarray,
    assoc: np.ndarray,
    *,
    mode: str,
    delta_cutoff: int,
    size: int,
    title: str | None,
) -> "Image.Image":
    """Draw one window: timepoint-coloured ellipses + GT forward-association lines."""
    from matplotlib import colormaps
    from skimage.draw import ellipse_perimeter, line

    # (row, col) centres, translated so the bounding-box centre of the window
    # sits at the image centre (no scaling; geometry stays in true pixels).
    yx = coords[:, 1:3].astype(float)
    if len(yx):
        yx = yx - (yx.min(0) + yx.max(0)) / 2 + size / 2
    canvas = np.zeros((size, size, 3), dtype=np.uint8)

    # colour by timepoint with 'turbo' (high contrast between adjacent frames, so
    # consecutive detections of the same track stay visually distinguishable):
    # earliest -> dark blue, latest -> red.
    cmap = colormaps["turbo"]
    uniq = np.unique(timepoints)
    color_of = {}
    for i, t in enumerate(uniq):
        frac = i / (len(uniq) - 1) if len(uniq) > 1 else 0.0
        color_of[int(t)] = (np.array(cmap(frac)[:3]) * 255).astype(np.uint8)

    def _clip(rr, cc):
        keep = (rr >= 0) & (rr < size) & (cc >= 0) & (cc < size)
        return rr[keep], cc[keep]

    # GT forward associations first, so detections render on top
    dt = timepoints[None, :] - timepoints[:, None]
    fwd = (assoc > 0.5) & (dt > 0) & (dt <= delta_cutoff)
    for i, j in zip(*np.nonzero(fwd)):
        rr, cc = line(
            int(round(yx[i, 0])), int(round(yx[i, 1])),
            int(round(yx[j, 0])), int(round(yx[j, 1])),
        )
        rr, cc = _clip(rr, cc)
        canvas[rr, cc] = (110, 110, 110)

    for idx in range(len(yx)):
        r_major, r_minor, orient = _ellipse_axes(
            _inertia_from_feature_row(feats[idx], mode)
        )
        # ellipse_perimeter's orientation runs opposite to skimage's region
        # orientation (verified by round-trip), so negate the major-axis angle.
        rr, cc = ellipse_perimeter(
            int(round(yx[idx, 0])), int(round(yx[idx, 1])),
            max(1, int(round(r_major))), max(1, int(round(r_minor))),
            orientation=-orient, shape=(size, size),
        )
        canvas[rr, cc] = color_of[int(timepoints[idx])]

    img = Image.fromarray(canvas)
    if title:
        ImageDraw.Draw(img).text((4, 4), title, fill=(255, 255, 255))
    return img


def save_window_debug_viz(
    batch: dict,
    sample_indices,
    out_dir: Path | str,
    *,
    mode: str = "wrfeat2",
    delta_cutoff: int = 2,
    epoch: int = 0,
    names=None,
    size: int = 512,
) -> list[Path]:
    """Render sampled training windows as feature-space tracking images.

    For each requested sample of a collated batch, every detection is drawn as an
    ellipse reconstructed from the consumed feature vector (centre + area +
    inertia tensor), coloured by timepoint ('turbo': dark blue -> red), with
    ground-truth forward associations drawn as connecting lines. Intended for
    ``--debug`` to sanity-check augmentation (coords and geometric features must
    stay consistent). The window's bounding box is centred in the image; geometry
    is in true pixels (no scaling), so a window wider than ``size`` is clipped.

    Args:
        batch: collated batch dict (``coords``, ``features``, ``timepoints``,
            ``assoc_coo``).
        sample_indices: batch rows to render.
        out_dir: base viz directory; images go to ``out_dir/epoch{E:04d}``.
        mode: feature mode; only the ``wrfeat2`` family is supported.
        delta_cutoff: maximum forward dt for a drawn GT association.
        epoch: current epoch (sub-folder + filename).
        names: optional per-sample dataset names for the title.
        size: square canvas side in pixels.

    Returns:
        Paths of the written PNGs.
    """
    import imageio.v2 as imageio

    out_dir = Path(out_dir) / f"epoch{int(epoch):04d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    coords = batch["coords"].detach().cpu().numpy()
    feats = batch["features"].detach().cpu().numpy()
    timepoints = batch["timepoints"].detach().cpu().numpy()
    bsz, n = timepoints.shape
    assoc = _densify_assoc_np(batch["assoc_coo"], bsz, n)

    paths = []
    for k, s in enumerate(sample_indices):
        valid = timepoints[s] >= 0
        name = names[k] if names is not None and k < len(names) else ""
        title = f"{name} sample{k} ep{epoch}".strip()
        img = _render_window(
            coords[s][valid],
            feats[s][valid],
            timepoints[s][valid],
            assoc[s][np.ix_(valid, valid)],
            mode=mode,
            delta_cutoff=delta_cutoff,
            size=size,
            title=title,
        )
        path = out_dir / f"sample{k}.png"
        imageio.imwrite(path, np.asarray(img))
        paths.append(path)
    return paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--img", type=Path, required=True, help="input frames folder")
    parser.add_argument("-g", "--gt", type=Path, required=True, help="ground-truth TRA folder")
    parser.add_argument("-p", "--pred", type=Path, required=True, help="prediction folder")
    parser.add_argument("-o", "--out", type=Path, default=Path("errors.mp4"))
    parser.add_argument("-s", "--size", type=int, default=512)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument(
        "--tail", type=int, default=10, help="frames a link stays visible before fading out"
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    viz_error(
        args.img,
        args.gt,
        args.pred,
        out_path=args.out,
        size=args.size,
        fps=args.fps,
        tail=args.tail,
    )
