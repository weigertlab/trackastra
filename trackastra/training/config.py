"""Training CLI parsing."""

from __future__ import annotations

import ast
import sys
from typing import Any

import configargparse

from trackastra.utils import str2bool


def _restore_input_config_items(items: list[Any] | None) -> list[Any] | None:
    """Restore grouped YAML input specs stringified by configargparse."""
    if items is None:
        return None
    restored = []
    for item in items:
        if not isinstance(item, str):
            restored.append(item)
            continue
        stripped = item.strip()
        if not (stripped.startswith("{") or stripped.startswith("[")):
            restored.append(item)
            continue
        try:
            restored.append(ast.literal_eval(stripped))
        except (SyntaxError, ValueError):
            restored.append(item)
    return restored


def create_train_parser() -> configargparse.ArgumentParser:
    """Create the Trackastra training parser without parsing args."""
    parser = configargparse.ArgumentParser(
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        allow_abbrev=False,
    )
    parser.add_argument(
        "-c",
        "--config",
        is_config_file=True,
        help="config file path",
        default="configs/vanvliet.yaml",
    )
    parser.add_argument("-o", "--outdir", type=str, default="runs")
    parser.add_argument("--name", type=str, help="Name to append to timestamp")
    parser.add_argument("--timestamp", type=str2bool, default=True)
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=None,
        help="load this model at start (e.g. to continue training)",
    )
    parser.add_argument(
        "--ndim", type=int, default=2, help="number of spatial dimensions"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="accelerator to train on ('auto' picks cuda when available)",
    )
    parser.add_argument("-d", "--d_model", type=int, default=256)
    parser.add_argument("-w", "--window", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument(
        "--detection_folders",
        type=str,
        nargs="+",
        default=["TRA"],
        help=(
            "Subfolders to search for detections. Defaults to `TRA`, which corresponds"
            " to using only the GT."
        ),
    )
    parser.add_argument("--input_train", type=str, nargs="+")
    parser.add_argument("--input_val", type=str, nargs="+")
    parser.add_argument("--slice_pct_train", type=float, nargs=2, default=(0.0, 1.0))
    parser.add_argument("--slice_pct_val", type=float, nargs=2, default=(0.0, 1.0))
    parser.add_argument("--downscale_temporal", type=int, default=1)
    parser.add_argument("--downscale_spatial", type=int, default=1)
    parser.add_argument("--max_distance", type=int, default=256)
    parser.add_argument("--normalize_diameter", type=float, default=None)
    parser.add_argument("--train_samples", type=int, default=10000)
    parser.add_argument("--num_encoder_layers", type=int, default=6)
    parser.add_argument("--num_decoder_layers", type=int, default=6)
    parser.add_argument("--pos_embed_per_dim", type=int, default=32)
    parser.add_argument("--feat_embed_per_dim", type=int, default=8)
    parser.add_argument(
        "--feature_embed_mode",
        choices=("fourier", "mlp"),
        default=None,
        help="feature encoder; defaults to fourier for wrfeat and mlp otherwise",
    )
    parser.add_argument("--dropout", type=float, default=0.00)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_detections", type=int, default=None)
    parser.add_argument("--delta_cutoff", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--attn_positional_bias",
        type=str,
        choices=["rope", "none"],
        default="rope",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Enable loss/grad spike dumps and per-batch dataset provenance logging "
            "under <logdir>/debug."
        ),
    )
    parser.add_argument("--attn_positional_bias_n_spatial", type=int, default=16)
    parser.add_argument("--attn_dist_mode", type=str, default="v1")
    parser.add_argument(
        "--attn_mode", type=str, choices=["dense", "sparse"], default="dense"
    )
    parser.add_argument("--max_neighbors", type=int, nargs="+", default=[16])
    parser.add_argument(
        "--sparse_knn_mode",
        type=str,
        choices=["global", "per_frame", "next_frame"],
        default="per_frame",
        help=(
            "sparse kNN budget: 'global' = K nearest over the whole window; "
            "'per_frame' = K nearest within each frame (-> F*K slots, keeps the "
            "diagonal); 'next_frame' = K nearest in the same and next frame "
            "(-> 2*K slots)"
        ),
    )
    parser.add_argument("--logit_norm", type=str2bool, default=True)
    parser.add_argument(
        "--head_mode",
        choices=["bilinear", "sparse_bilinear", "edge_star", "edge_mlp"],
        default=None,
        help="Association head; None auto-selects from --attn_mode.",
    )
    parser.add_argument(
        "--architecture_version",
        type=int,
        choices=(1, 2),
        default=2,
        help=(
            "model forward semantics; use 1 only for "
            "architecture-version-1-compatible training"
        ),
    )
    parser.add_argument(
        "--disable_abs_pos",
        action="store_true",
        help="omit input coordinate Fourier embeddings; attention RoPE is unchanged",
    )
    parser.add_argument(
        "--disable_input_norm",
        action="store_true",
        help="bypass the initial LayerNorm after input projection",
    )
    parser.add_argument("--mixedp", type=str2bool, default=True)
    parser.add_argument("--dry", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--features",
        type=str,
        choices=[
            "none",
            "intensity",
            "wrfeat",
            "wrfeat2",
            "wrfeat2_no_intensity",
        ],
        default="wrfeat2",
    )
    parser.add_argument(
        "--causal_norm",
        type=str,
        choices=["none", "linear", "softmax", "quiet_softmax"],
        default="quiet_softmax",
    )
    parser.add_argument(
        "--assoc_loss",
        choices=["bce", "child_ce"],
        default="bce",
        help=(
            "association loss: per-edge BCE or child parent-or-null CE; "
            "child_ce requires --loss_norm decision"
        ),
    )
    parser.add_argument(
        "--loss_norm",
        choices=["matrix", "decision"],
        default="matrix",
        help="loss reduction: original matrix-entry mean or per-decision mean",
    )
    parser.add_argument(
        "--focal_loss_gamma",
        type=float,
        default=0.0,
        help="binary focal-loss exponent; 0 disables focal weighting",
    )
    parser.add_argument("--div_upweight", type=float, default=2)
    parser.add_argument(
        "--grad_log_every_n_epochs",
        type=int,
        default=10,
        help="compute/log the (expensive) full-model grad norm only every N epochs",
    )
    parser.add_argument("--augment", type=int, default=3)
    parser.add_argument(
        "--detect_drop",
        type=float,
        default=0.0,
        help="probability of applying detection dropout to a training window",
    )
    parser.add_argument(
        "--detect_drop_fraction",
        type=float,
        default=0.1,
        help="fraction of detections dropped when detection dropout is applied",
    )
    parser.add_argument(
        "--tracking_frequency",
        type=int,
        default=5,
        help="run full-movie validation every N epochs; <=0 disables it",
    )
    parser.add_argument(
        "--tracking_mode",
        choices=("greedy", "greedy_nodiv", "ilp"),
        default="greedy",
        help="linking mode used for full-movie validation",
    )
    parser.add_argument(
        "--cache",
        type=str2bool,
        default=False,
        help="cache CTCData to disk use (useful for large datasets)",
    )
    parser.add_argument(
        "--cachedir",
        type=str,
        default=".cache",
        help="cache dir for CTCData if --cache is set",
    )
    parser.add_argument("--resume", type=str2bool, default=False)
    parser.add_argument(
        "--n_pool_sampler",
        type=int,
        default=8,
        help="pool size for balanced sampler (set to 1 to disable balancing)",
    )
    parser.add_argument(
        "--distributed",
        type=str2bool,
        default=False,
        help="use distributed DDP training",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb", "none"],
    )
    parser.add_argument("--wandb_project", type=str, default="trackastra-new")
    parser.add_argument(
        "--weight_by_ndivs",
        type=str2bool,
        default=True,
        help="Oversample windows that contain divisions",
    )
    parser.add_argument(
        "--weight_by_dataset",
        type=str2bool,
        default=False,
        help=(
            "Inversely weight datasets by number of samples "
            "(to counter dataset size imbalance)"
        ),
    )
    parser.add_argument(
        "--balance_batch_objects",
        type=str2bool,
        default=False,
        help=(
            "Use a variable batch size so the total detections per batch stay "
            "roughly constant (batch_size becomes an upper cap). Equalizes GPU "
            "memory across differently sized data. Ignored under DDP."
        ),
    )
    parser.add_argument(
        "--balance_pct",
        type=float,
        default=50.0,
        help=(
            "Percentile of per-window detection counts used as the reference for "
            "--balance_batch_objects (budget = batch_size * n_ref)"
        ),
    )
    return parser


def parse_train_args(
    parser: configargparse.ArgumentParser | None = None,
):
    """Parse Trackastra training CLI args from a parser users may extend."""
    parser = create_train_parser() if parser is None else parser
    args, unknown_args = parser.parse_known_args()
    args.input_train = _restore_input_config_items(args.input_train)
    args.input_val = _restore_input_config_items(args.input_val)

    allowed_unknown = ["input_test"]
    if not set(a.split("=")[0].strip("-") for a in unknown_args).issubset(
        set(allowed_unknown)
    ):
        raise ValueError(f"Unknown args: {unknown_args}")

    if args.distributed and hasattr(sys, "ps1"):
        raise ValueError(
            "Distributed training does not work in interactive mode. Run as "
            "`python train.py`."
        )

    return args
