import argparse
import sys
from pathlib import Path

import torch

from .model import Trackastra
from .tracking.utils import graph_to_ctc
from .utils import str2path


def cli():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, allow_abbrev=False
    )
    subparsers = p.add_subparsers(help="trackastra")

    p_track = subparsers.add_parser("track", help="Tracking help")
    p_track.add_argument(
        "-i",
        "--imgs",
        type=str2path,
        required=True,
        help="Directory with series of .tif files.",
    )
    p_track.add_argument(
        "-m",
        "--masks",
        type=str2path,
        required=True,
        help="Directory with series of .tif files.",
    )
    p_track.add_argument(
        "-o",
        "--outdir",
        type=str2path,
        default=None,
        help=(
            "Directory for writing results (optional). Default writes to"
            " `{masks}_tracked`."
        ),
    )
    p_track.add_argument(
        "--model-pretrained",
        type=str,
        default=None,
        help="Name of pretrained Trackastra model.",
    )
    p_track.add_argument(
        "--model-custom",
        type=str2path,
        default=None,
        help="Local folder with custom model.",
    )
    p_track.add_argument(
        "--mode", choices=["greedy_nodiv", "greedy", "ilp"], default="greedy"
    )
    p_track.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p_track.set_defaults(cmd=_track_from_disk)

    if len(sys.argv) == 1:
        p.print_help(sys.stdout)
        sys.exit(0)

    args = p.parse_args()

    args.cmd(args)


def _track_from_disk(args):
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
