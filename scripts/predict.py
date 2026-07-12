"""Predict and evaluate Trackastra models on configured CTC test movies.

Two input modes:

* ``--input_test`` (normally from the config): CTC-like movie roots with ground
  truth. Tracks and evaluates against the GT ``TRA`` folder.
* ``--imgs``/``--masks``: a raw image/mask folder (or single ``T,(Z),Y,X`` tiff)
  pair with no ground truth. Tracks and writes CTC results only.
"""

import argparse
import shutil
from pathlib import Path

import configargparse
import numpy as np
import pandas as pd
from trackastra.utils import str2bool


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = configargparse.ArgumentParser(
        description="Predict configured CTC test movies and report TRA/AOGM.",
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        ignore_unknown_config_file_keys=True,
        allow_abbrev=False,
    )
    parser.add_argument("-c", "--config", is_config_file=True, help="YAML config path")
    parser.add_argument("-m", "--model", required=True, help="model folder or name")
    parser.add_argument(
        "-o", "--outdir", type=Path, default=Path("predictions"), help="CTC results"
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default=None,
        help=(
            "Name of the movie output folder, i.e. results go to OUTDIR/<model>/NAME. "
            "Defaults to the movie folder name (CTC) or the mask folder's parent "
            "(--masks). Only valid for a single input movie."
        ),
    )
    parser.add_argument(
        "--input_test",
        type=Path,
        nargs="+",
        default=None,
        help="test movie roots with ground truth (normally read from the config)",
    )
    parser.add_argument(
        "-i",
        "--imgs",
        "--img",
        dest="imgs",
        type=Path,
        default=None,
        help=(
            "image folder or single T,(Z),Y,X tiff. Use with --masks to track a movie "
            "that has no ground truth; evaluation, errors.csv and error.mp4 are skipped."
        ),
    )
    parser.add_argument(
        "--masks",
        "--mask",
        dest="masks",
        type=Path,
        default=None,
        help=(
            "mask folder or single T,(Z),Y,X tiff. Selects predict-only mode "
            "(no evaluation). --imgs may be omitted for a mask-only model."
        ),
    )
    parser.add_argument(
        "--detection-folder",
        "--detection_folders",
        dest="detection_folders",
        nargs="+",
        default=["TRA"],
        help="folder containing instance segmentation masks",
    )
    parser.add_argument(
        "--device", choices=("automatic", "cuda", "mps", "cpu"), default="automatic"
    )
    parser.add_argument(
        "--mode", choices=("greedy", "greedy_nodiv", "ilp"), default="greedy"
    )
    parser.add_argument(
        "--spatial_cutoff",
        dest="spatial_cutoff",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--spacing",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Source-to-model spatial scale, normally voxel size in micrometers. "
            "Use one value per spatial axis, e.g. 4 1 1 for anisotropic 3D. "
            "If omitted, unit spacing is assumed."
        ),
    )
    parser.add_argument(
        "--normalize-diameter",
        "--normalize_diameter",
        dest="normalize_diameter",
        type=float,
        default=None,
        help=(
            "Scale WR feature geometry so each movie's median equivalent diameter "
            "matches this value. Defaults to the model train config when present."
        ),
    )
    parser.add_argument(
        "--node_prior",
        type=float,
        default=0.0,
        help=(
            "Shared weight (lambda) applied to all node-degree priors "
            "(appear/disappear/split) in greedy or ILP tracking. 0 disables them "
            "(default). Requires a model trained with node_head=True."
        ),
    )
    for prior in ("appear", "disappear", "split"):
        parser.add_argument(
            f"--lam_{prior}",
            f"--lam-{prior}",
            dest=f"lam_{prior}",
            type=float,
            default=None,
            help=f"Weight for the {prior} node-degree prior; overrides --node_prior.",
        )
    parser.add_argument(
        "--geff",
        type=str2bool,
        default=True,
        help=(
            "also write the tracks as a GEFF zarr store next to the CTC folder, "
            "i.e. OUTDIR/<model>/<movie>.geff"
        ),
    )
    parser.add_argument("-f", "--overwrite", action="store_true")
    parser.add_argument(
        "--errormovie",
        type=str2bool,
        default=True,
        help="render an error.mp4 (TP/FP/FN tracks) into each output folder",
    )
    parser.add_argument(
        "--error-report",
        "--error_report",
        dest="error_report",
        type=str2bool,
        default=True,
        help="write errors.csv with official CTC FN/FP/WS edges and model diagnostics",
    )
    return parser.parse_args(argv)


def ctc_metrics_from_data(
    gt_data, pred_data, return_matched: bool = False
) -> dict[str, float] | tuple[dict[str, float], object]:
    """Run CTC TRA/AOGM on already-loaded traccuracy graphs.

    ``traccuracy`` annotates input graphs in place during scoring, so callers that
    run this repeatedly must pass freshly loaded or otherwise unannotated graphs.
    """
    from traccuracy import run_metrics
    from traccuracy.matchers import CTCMatcher
    from traccuracy.metrics import CTCMetrics

    results, matched = run_metrics(
        gt_data=gt_data,
        pred_data=pred_data,
        matcher=CTCMatcher(),
        metrics=[CTCMetrics()],
    )
    values = results[0]["results"]
    metric_names = (
        "TRA",
        "AOGM",
        "DET",
        "LNK",
        "fp_nodes",
        "fn_nodes",
        "ns_nodes",
        "fp_edges",
        "fn_edges",
        "ws_edges",
    )
    values = {name: float(values[name]) for name in metric_names}
    return (values, matched) if return_matched else values


def evaluate_ctc(
    gt_path: Path, pred_path: Path, return_matched: bool = False
) -> dict[str, float] | tuple[dict[str, float], object]:
    """Compute CTC-weighted TRA and AOGM with traccuracy."""
    from traccuracy.loaders import load_ctc_data

    gt_data = load_ctc_data(str(gt_path), run_checks=False)
    pred_data = load_ctc_data(str(pred_path), run_checks=False)
    return ctc_metrics_from_data(gt_data, pred_data, return_matched=return_matched)


def link_type_breakdown(matched) -> dict[str, float]:
    """Per-movie FN/FP/F1 on DIVISION links (source out-degree >= 2).

    Uses only the matched GT/pred graphs (CTC edge flags + node out-degree); no model
    internals. Rates are NaN when the movie has no division edges, so a division-free
    movie does not read as a perfect division score (callers should nanmean).
    """
    from traccuracy._tracking_graph import EdgeFlag

    gt, pred = matched.gt_graph, matched.pred_graph

    def n_division(graph, edges):
        return sum(
            1 for source, _target in edges if graph.graph.out_degree(source) >= 2
        )

    gt_div = n_division(gt, gt.edges)
    pred_div = n_division(pred, pred.edges)
    fn_div = n_division(gt, gt.get_edges_with_flag(EdgeFlag.CTC_FALSE_NEG))
    fp_div = n_division(pred, pred.get_edges_with_flag(EdgeFlag.CTC_FALSE_POS))

    def rate(num, den):
        return float(num) / den if den else float("nan")

    fn, fp = rate(fn_div, gt_div), rate(fp_div, pred_div)
    # division F1 (precision = 1 - fp rate, recall = 1 - fn rate)
    recall, precision = 1.0 - fn, 1.0 - fp
    total = precision + recall
    if total > 0:
        f1 = 2.0 * precision * recall / total
    elif recall == recall and precision == precision:  # both finite (both 0)
        f1 = 0.0
    else:
        f1 = float("nan")
    return {"fn_div": fn, "fp_div": fp, "f1_div": f1}


def _write_error_report(
    graph, details, matched, output_path: Path, ndim: int, mode: str
) -> None:
    """Write the official CTC edge errors with model diagnostics."""
    try:
        from scripts.check_errors import link_error_report
    except ImportError:
        from check_errors import link_error_report

    edf = link_error_report(graph, details, matched, ndim=ndim, mode=mode)
    edf.to_csv(output_path / "errors.csv", index=False)
    if len(edf):
        counts = ", ".join(
            f"{kind.upper()}={count}"
            for kind, count in edf.error_type.value_counts().items()
        )
        print(f"  errors.csv: {len(edf)} official edge errors ({counts})")
    else:
        print("  errors.csv: 0 edge errors")


def _node_prior_kwargs(
    mode: str,
    node_prior: float = 0.0,
    lam_appear: float | None = None,
    lam_disappear: float | None = None,
    lam_split: float | None = None,
) -> dict:
    """Map the node-degree prior weights onto the solver config for ``mode``.

    ``node_prior`` is the shared default for all three priors (appear/disappear/split);
    each ``lam_*`` overrides it individually. All zero -> current behavior (no config
    passed, i.e. the plain threshold/fixed-cost solver).
    """
    from trackastra.tracking import GreedyConfig, ILPConfig

    lam = dict(
        lam_appear=node_prior if lam_appear is None else lam_appear,
        lam_disappear=node_prior if lam_disappear is None else lam_disappear,
        lam_split=node_prior if lam_split is None else lam_split,
    )
    if not any(lam.values()):
        return {}
    if mode in ("greedy", "greedy_nodiv"):
        return {"greedy_config": GreedyConfig(**lam)}
    if mode == "ilp":
        return {"ilp_config": ILPConfig(**lam)}
    return {}


def _load_frames(path: Path) -> np.ndarray:
    """Load a time series from a folder of tiffs or a single T,(Z),Y,X tiff."""
    import tifffile
    from trackastra.data import load_tiff_timeseries

    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"No such image/mask path: {path}")
    return load_tiff_timeseries(path) if path.is_dir() else tifffile.imread(path)


def _movie_name(root: Path) -> str:
    """Self-describing output-folder name for a CTC movie root."""
    seq_dir = root.parent if root.name == "TRA" else root
    seq_name = seq_dir.name.removesuffix("_GT")
    dataset = seq_dir.parent.name.removesuffix("_GT")
    # CTC sequences are named "01", "02", ...; prefix the dataset (e.g.
    # "Fluo-N2DL-HeLa_01") so the output folder is self-describing.
    return f"{dataset}_{seq_name}" if seq_name.isdigit() and dataset else seq_name


def _movie_names(input_paths: list[Path]) -> list[str]:
    """Unique output-folder name per input root, in order."""
    names: list[str] = []
    used: set[str] = set()
    for index, root in enumerate(input_paths, start=1):
        name = _movie_name(Path(root))
        if name in used:
            name = f"{name}_{index}"
        used.add(name)
        names.append(name)
    return names


def _geff_path(output_path: Path) -> Path:
    """Sibling GEFF store for a CTC output folder: ``<movie>/`` -> ``<movie>.geff``."""
    return output_path.parent / f"{output_path.name}.geff"


def _write_geff(graph, ndim: int, path: Path) -> None:
    """Write the solution graph as a GEFF store that ``geff.read(path)`` can open.

    ``trackastra.tracking.write_to_geff`` nests the graph in a subgroup of an outer
    zarr (``<store>/tracking_graph.geff``, next to a ``segmentation`` array), so the
    outer store is not itself a GEFF group. Here the tracked masks already sit in the
    sibling CTC folder, so the GEFF group is written directly at ``path``.
    """
    import geff

    coord_names = ("z", "y", "x")[3 - ndim :]
    graph = graph.copy()  # keep the caller's graph (and its "time"/"coords") intact
    for _node, data in graph.nodes(data=True):
        coords = data.pop("coords")
        # Plain python floats -> float64 in the store. float32 coordinates survive
        # geff.read but blow up downstream: inTRACKtive JSON-encodes their mean into
        # zarr attrs, and numpy float32 is not JSON serializable.
        data.update(
            {n: float(v) for n, v in zip(coord_names, coords[-ndim:], strict=True)}
        )
        data["t"] = int(data.pop("time"))
        data["label"] = int(data["label"])
        # drop model-internal diagnostics (node-degree logits, edge weight)
        for key in ("a", "c", "s", "weight"):
            data.pop(key, None)

    geff.write(
        graph,
        store=path,
        axis_names=["t", *coord_names],
        axis_types=["time", *["space"] * ndim],
        overwrite=True,
    )


def _prepare_output(path: Path, overwrite: bool) -> None:
    generated = list(path.glob("man_track*.tif")) if path.exists() else []
    track_file = path / "man_track.txt"
    if track_file.exists():
        generated.append(track_file)
    geff = _geff_path(path)
    if (generated or geff.exists()) and not overwrite:
        raise FileExistsError(f"Results already exist for {path}; use --overwrite")
    if overwrite:
        for output in generated:
            output.unlink()
        if geff.exists():
            shutil.rmtree(geff)
    path.mkdir(parents=True, exist_ok=True)


def predict_and_evaluate(
    model,
    input_paths: list[Path],
    detection_folder: str,
    outdir: Path,
    model_name: str,
    mode: str = "greedy",
    spatial_cutoff: int | None = None,
    spacing: tuple[float, ...] | None = None,
    normalize_diameter: float | None = None,
    overwrite: bool = False,
    print_results: bool = True,
    errormovie: bool = False,
    error_report: bool = False,
    link_breakdown: bool = True,
    node_prior: float = 0.0,
    lam_appear: float | None = None,
    lam_disappear: float | None = None,
    lam_split: float | None = None,
    out_name: str | None = None,
    geff: bool = False,
) -> pd.DataFrame:
    """Track and evaluate CTC movies with an already loaded Trackastra model."""
    from trackastra.data import DetectionSequence, load_ctc_images_masks
    from trackastra.tracking import graph_to_ctc

    node_prior_kwargs = _node_prior_kwargs(
        mode, node_prior, lam_appear, lam_disappear, lam_split
    )

    if out_name is not None and len(input_paths) != 1:
        raise ValueError(
            f"--name sets a single movie output folder, but got {len(input_paths)} "
            "input movies"
        )
    names = [out_name] if out_name is not None else _movie_names(input_paths)
    output_paths = [outdir / model_name / name for name in names]
    # Check (and clear) every output up front: a collision on the last movie must not
    # surface only after the earlier ones have been tracked, which takes minutes.
    for output_path in output_paths:
        _prepare_output(output_path, overwrite)

    rows = []
    for root, name, output_path in zip(input_paths, names, output_paths, strict=True):
        root = Path(root)
        transformer = getattr(model, "transformer", None)
        config = getattr(transformer, "config", {})
        ndim = int(config.get("coord_dim", 2))
        imgs, masks, image_path, gt_path = load_ctc_images_masks(
            root,
            detection_folder=detection_folder,
            ndim=ndim,
        )

        detections = DetectionSequence.from_masks(
            imgs,
            masks,
            name=name,
            ndim=ndim,
            spacing=spacing,
            normalize_imgs=False,
            keep_masks=True,
            keep_images=True,
        )
        track_kwargs = dict(
            mode=mode, spatial_cutoff=spatial_cutoff, **node_prior_kwargs
        )
        if error_report:
            track_kwargs["return_details"] = True
        result = model.track(
            detections,
            normalize_diameter=normalize_diameter,
            **track_kwargs,
        )
        graph = result.graph
        details = (
            {
                "candidate_graph": result.candidate_graph,
                "predictions": result.predictions,
            }
            if error_report
            else None
        )
        graph_to_ctc(graph, result.masks, outdir=output_path)
        if geff:
            _write_geff(graph, ndim, _geff_path(output_path))
        need_matched = error_report or link_breakdown
        evaluated = evaluate_ctc(gt_path, output_path, return_matched=need_matched)
        values, matched = evaluated if need_matched else (evaluated, None)
        if link_breakdown and matched is not None:
            try:
                values = {**values, **link_type_breakdown(matched)}
            except Exception as error:
                print(f"Could not compute division link breakdown for {name}: {error}")

        if error_report:
            try:
                _write_error_report(
                    graph,
                    details,
                    matched,
                    output_path,
                    ndim=ndim,
                    mode=mode,
                )
            except Exception as error:
                print(f"Could not write error report for {name}: {error}")
        row = {"movie": name, "model": model_name, "mode": mode, **values}
        rows.append(row)
        # store each movie's metrics next to its tiffs and error.mp4
        pd.DataFrame([row]).to_csv(output_path / "metrics.csv", index=False)

        if errormovie:
            try:
                from scripts.utils import viz_error
            except ImportError:
                from utils import viz_error
            try:
                viz_error(image_path, gt_path, output_path, output_path / "error.mp4")
            except Exception as error:
                print(f"Could not render error movie for {name}: {error}")

    results = pd.DataFrame(rows)
    mean = {
        "movie": "Mean",
        "model": model_name,
        "mode": mode,
        **results.select_dtypes("number").mean().to_dict(),
    }
    results = pd.concat([results, pd.DataFrame([mean])], ignore_index=True)
    results.to_csv(outdir / model_name / "metrics.csv", index=False)
    if print_results:
        print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    return results


def predict_only(
    model,
    masks_path: Path,
    imgs_path: Path | None,
    outdir: Path,
    model_name: str,
    mode: str = "greedy",
    spatial_cutoff: int | None = None,
    spacing: tuple[float, ...] | None = None,
    normalize_diameter: float | None = None,
    overwrite: bool = False,
    node_prior: float = 0.0,
    lam_appear: float | None = None,
    lam_disappear: float | None = None,
    lam_split: float | None = None,
    out_name: str | None = None,
    geff: bool = False,
) -> Path:
    """Track a raw image/mask pair and write CTC results, without ground truth.

    Same tracking path as ``predict_and_evaluate``: the frames are wrapped in a
    ``DetectionSequence`` and handed to ``model.track``. Only the evaluation half is
    dropped, since there is no GT to score against.
    """
    from trackastra.data import DetectionSequence
    from trackastra.tracking import graph_to_ctc
    from trackastra.utils import normalize

    config = getattr(getattr(model, "transformer", None), "config", {})
    ndim = int(config.get("coord_dim", 2))

    # Claim the output before loading anything: a collision must not surface only
    # after the tiffs are read and the movie is tracked.
    masks_path = Path(masks_path).expanduser()
    if out_name is not None:
        name = out_name
    else:
        name = masks_path.parent.name if masks_path.is_dir() else masks_path.stem
    output_path = outdir / model_name / name
    _prepare_output(output_path, overwrite)

    masks = _load_frames(masks_path)
    imgs = _load_frames(imgs_path) if imgs_path is not None else None
    if imgs is not None:
        if len(imgs) != len(masks):
            raise ValueError(
                f"Image and mask frame counts differ: {len(imgs)} images, "
                f"{len(masks)} masks"
            )
        # Normalize per frame, as load_ctc_images_masks does. DetectionSequence's own
        # normalize_imgs=True would take percentiles over the whole movie at once,
        # which is not what the model was trained on.
        imgs = np.stack([normalize(frame) for frame in imgs])

    detections = DetectionSequence.from_masks(
        imgs,
        masks,
        name=name,
        ndim=ndim,
        spacing=spacing,
        normalize_imgs=False,  # already normalized per frame above
        keep_masks=True,
        keep_images=True,
    )
    result = model.track(
        detections,
        mode=mode,
        spatial_cutoff=spatial_cutoff,
        normalize_diameter=normalize_diameter,
        **_node_prior_kwargs(mode, node_prior, lam_appear, lam_disappear, lam_split),
    )
    graph_to_ctc(result.graph, result.masks, outdir=output_path)
    if geff:
        _write_geff(result.graph, ndim, _geff_path(output_path))
    print(
        f"{name}: {result.graph.number_of_nodes()} nodes, "
        f"{result.graph.number_of_edges()} edges -> {output_path}"
    )
    return output_path


def run(args: argparse.Namespace) -> pd.DataFrame | Path:
    from trackastra.model import Trackastra

    if args.masks is None and args.imgs is not None:
        raise ValueError("--imgs requires --masks")
    if args.masks is None and not args.input_test:
        raise ValueError(
            "Provide either --masks (with optional --imgs) to predict without ground "
            "truth, or --input_test / a config with CTC movie roots to also evaluate."
        )

    model_path = Path(args.model).expanduser()
    if model_path.is_dir():
        model_name = model_path.stem
        model = Trackastra.from_folder(model_path, device=args.device)
    else:
        model_name = args.model
        model = Trackastra.from_pretrained(args.model, device=args.device)

    if args.masks is not None:
        return predict_only(
            model=model,
            masks_path=args.masks,
            imgs_path=args.imgs,
            outdir=args.outdir,
            model_name=model_name,
            mode=args.mode,
            spatial_cutoff=args.spatial_cutoff,
            spacing=tuple(args.spacing) if args.spacing is not None else None,
            normalize_diameter=args.normalize_diameter,
            overwrite=args.overwrite,
            node_prior=args.node_prior,
            lam_appear=args.lam_appear,
            lam_disappear=args.lam_disappear,
            lam_split=args.lam_split,
            out_name=args.name,
            geff=args.geff,
        )

    if len(args.detection_folders) != 1:
        raise ValueError(
            "Prediction requires one detection folder; override the config with "
            "--detection-folder FOLDER"
        )
    return predict_and_evaluate(
        model=model,
        input_paths=args.input_test,
        detection_folder=args.detection_folders[0],
        outdir=args.outdir,
        model_name=model_name,
        mode=args.mode,
        spatial_cutoff=args.spatial_cutoff,
        spacing=tuple(args.spacing) if args.spacing is not None else None,
        normalize_diameter=args.normalize_diameter,
        overwrite=args.overwrite,
        errormovie=args.errormovie,
        error_report=getattr(args, "error_report", False),
        node_prior=args.node_prior,
        lam_appear=args.lam_appear,
        lam_disappear=args.lam_disappear,
        lam_split=args.lam_split,
        out_name=args.name,
        geff=args.geff,
    )


if __name__ == "__main__":
    run(parse_args())
