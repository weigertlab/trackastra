"""Predict and evaluate Trackastra models on configured CTC test movies."""

import argparse
from pathlib import Path

import configargparse
import pandas as pd


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = configargparse.ArgumentParser(
        description="Predict configured CTC test movies and report TRA/AOGM.",
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        ignore_unknown_config_file_keys=True,
        allow_abbrev=False,
    )
    parser.add_argument(
        "-c", "--config", required=True, is_config_file=True, help="YAML config path"
    )
    parser.add_argument("-m", "--model", required=True, help="model folder or name")
    parser.add_argument(
        "-o", "--outdir", type=Path, default=Path("predictions"), help="CTC results"
    )
    parser.add_argument(
        "--input_test",
        type=Path,
        nargs="+",
        required=True,
        help="test movie roots (normally read from the config)",
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
    parser.add_argument("--max-distance", type=int, default=128)
    parser.add_argument("-f", "--overwrite", action="store_true")
    return parser.parse_args(argv)


def resolve_ctc_paths(root: Path, detection_folder: str) -> tuple[Path, Path, Path]:
    """Resolve image, detection, and ground-truth TRA paths for common CTC layouts."""
    root = root.expanduser()
    if root.name == "TRA":
        gt_path = root
        gt_root = root.parent
        sequence = (
            gt_root.parent / gt_root.name.removesuffix("_GT")
            if gt_root.name.endswith("_GT")
            else gt_root
        )
    elif root.name.endswith("_GT"):
        gt_path = root / "TRA"
        sequence = root.parent / root.name.removesuffix("_GT")
    else:
        sequence = root
        ctc_gt = Path(f"{sequence}_GT") / "TRA"
        gt_path = ctc_gt if ctc_gt.exists() else sequence / "TRA"

    image_path = sequence / "img" if (sequence / "img").exists() else sequence
    if detection_folder == "TRA":
        mask_path = gt_path
    else:
        candidates = (
            sequence / detection_folder,
            Path(f"{sequence}_{detection_folder}"),
            Path(f"{sequence}_ST") / detection_folder,
            Path(f"{sequence}_GT") / detection_folder,
        )
        mask_path = next((path for path in candidates if path.exists()), candidates[0])

    for kind, path in (
        ("image", image_path),
        ("detection", mask_path),
        ("ground-truth TRA", gt_path),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Could not find {kind} path for {root}: {path}")
    return image_path, mask_path, gt_path


def evaluate_ctc(gt_path: Path, pred_path: Path) -> dict[str, float]:
    """Compute CTC-weighted TRA and AOGM with traccuracy."""
    from traccuracy import run_metrics
    from traccuracy.loaders import load_ctc_data
    from traccuracy.matchers import CTCMatcher
    from traccuracy.metrics import CTCMetrics

    gt_data = load_ctc_data(str(gt_path), run_checks=False)
    pred_data = load_ctc_data(str(pred_path), run_checks=False)
    results, _ = run_metrics(
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
    return {name: float(values[name]) for name in metric_names}


def _prepare_output(path: Path, overwrite: bool) -> None:
    generated = list(path.glob("man_track*.tif")) if path.exists() else []
    track_file = path / "man_track.txt"
    if track_file.exists():
        generated.append(track_file)
    if generated and not overwrite:
        raise FileExistsError(f"CTC results already exist in {path}; use --overwrite")
    if overwrite:
        for output in generated:
            output.unlink()
    path.mkdir(parents=True, exist_ok=True)


def predict_and_evaluate(
    model,
    input_paths: list[Path],
    detection_folder: str,
    outdir: Path,
    model_name: str,
    mode: str = "greedy",
    max_distance: int = 128,
    overwrite: bool = False,
    print_results: bool = True,
) -> pd.DataFrame:
    """Track and evaluate CTC movies with an already loaded Trackastra model."""
    from trackastra.tracking import graph_to_ctc

    rows = []
    used_names: set[str] = set()
    for index, root in enumerate(input_paths, start=1):
        root = Path(root)
        image_path, mask_path, gt_path = resolve_ctc_paths(root, detection_folder)
        name = root.name.removesuffix("_GT")
        if name == "TRA":
            name = root.parent.name.removesuffix("_GT")
        if name in used_names:
            name = f"{name}_{index}"
        used_names.add(name)
        output_path = outdir / model_name / name
        _prepare_output(output_path, overwrite)

        graph, masks = model.track_from_disk(
            image_path,
            mask_path,
            mode=mode,
            max_distance=max_distance,
        )
        graph_to_ctc(graph, masks, outdir=output_path)
        values = evaluate_ctc(gt_path, output_path)
        rows.append({"movie": name, "model": model_name, "mode": mode, **values})

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
        print(results.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    return results


def run(args: argparse.Namespace) -> pd.DataFrame:
    from trackastra.model import Trackastra

    if len(args.detection_folders) != 1:
        raise ValueError(
            "Prediction requires one detection folder; override the config with "
            "--detection-folder FOLDER"
        )

    model_path = Path(args.model).expanduser()
    model_name = model_path.stem
    model = (
        Trackastra.from_folder(model_path, device=args.device)
        if model_path.exists()
        else Trackastra.from_pretrained(args.model, device=args.device)
    )
    return predict_and_evaluate(
        model=model,
        input_paths=args.input_test,
        detection_folder=args.detection_folders[0],
        outdir=args.outdir,
        model_name=model_name,
        mode=args.mode,
        max_distance=args.max_distance,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    run(parse_args())
