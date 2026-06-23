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
