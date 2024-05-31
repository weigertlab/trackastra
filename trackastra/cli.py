import argparse
from pathlib import Path

import torch

from .model import Trackastra
from .tracking.utils import graph_to_ctc
from .utils import str2path


def track_from_disk():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, allow_abbrev=False
    )
    p.add_argument(
        "-i",
        "--imgs",
        type=str2path,
        required=True,
        help="Directory with series of .tif files.",
    )
    p.add_argument(
        "-m",
        "--masks",
        type=str2path,
        required=True,
        help="Directory with series of .tif files.",
    )
    p.add_argument(
        "-o",
        "--outdir",
        type=str2path,
        default=None,
        help=(
            "Directory for writing results (optional). Default writes to"
            " `{masks}_tracked`."
        ),
    )
    p.add_argument(
        "--model-pretrained",
        type=str,
        default=None,
        help="Name of pretrained Trackastra model.",
    )
    p.add_argument(
        "--model-custom",
        type=str2path,
        default=None,
        help="Local folder with custom model.",
    )
    p.add_argument(
        "--mode", choices=["greedy_nodiv", "greedy", "ilp"], default="greedy"
    )
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"

    if args.model_pretrained is None == args.model_custom is None:
        raise ValueError(
            "Please pick a Trackastra model for tracking, either pretrained or a local"
            " custom model."
        )

    if args.model_pretrained is not None:
        model = Trackastra.from_pretrained(
            name=args.model_pretrained,
            device=device,
        )
    if args.model_custom is not None:
        model = Trackastra.from_folder(
            args.model_custom,
            device=device,
        )

    track_graph, masks = model.track_from_disk(
        args.imgs,
        args.masks,
        mode=args.mode,
    )

    if args.outdir is None:
        outdir = Path(f"{args.masks}_tracked")
    else:
        outdir = args.outdir

    outdir.mkdir(parents=True, exist_ok=True)
    graph_to_ctc(
        track_graph,
        masks,
        outdir=outdir,
    )
