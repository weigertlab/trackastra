"""Training CLI parsing."""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from typing import Any

import configargparse
import yaml

from trackastra.utils import str2bool


def _resolve_defaults(path: Path, stack: tuple[Path, ...] = ()) -> list[str]:
    """Flatten the `defaults:` chain of a config into files ordered low to high priority.

    Entries are resolved relative to the file declaring them, and may themselves
    declare `defaults:`. Later entries override earlier ones, the declaring file
    overrides all of them.
    """
    path = path.expanduser().resolve()
    if path in stack:
        chain = " -> ".join(str(p) for p in (*stack, path))
        raise ValueError(f"Config inheritance cycle: {chain}")
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"{path}: config root must be a mapping")

    entries = raw.get("defaults") or []
    if isinstance(entries, str):
        entries = [entries]
    if not isinstance(entries, list):
        raise TypeError(f"{path}: defaults must be a path or a list of paths")

    files = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, str) or not entry.strip():
            raise TypeError(f"{path}: defaults[{i}] must be a non-empty path string")
        base = Path(entry).expanduser()
        if not base.is_absolute():
            base = path.parent / base
        # configargparse silently ignores missing default config files, so recursing
        # here (which does check) is also what surfaces a mistyped path.
        files.extend(_resolve_defaults(base, (*stack, path)))
        files.append(str(base.resolve()))
    return files


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


def _parse_augment_details(value: Any) -> dict[str, float] | None:
    """Restore and validate the augment_details mapping from YAML/CLI input."""
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError) as e:
            raise ValueError(f"augment_details must be a mapping, got {value!r}") from e
    if not isinstance(value, dict):
        raise ValueError(f"augment_details must be a mapping, got {type(value).__name__}")
    unknown = set(value) - {"jitter", "drift", "tilt", "frame_jump", "frame_jump_p"}
    if unknown:
        raise ValueError(
            f"Unknown augment_details keys {sorted(unknown)}; "
            "supported keys are ['drift', 'frame_jump', 'frame_jump_p', "
            "'jitter', 'tilt']"
        )
    result = {}
    for key, raw in value.items():
        try:
            result[key] = float(raw)
        except (TypeError, ValueError) as e:
            raise ValueError(f"augment_details.{key} must be numeric, got {raw!r}") from e
    return result


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
    parser.add_argument(
        "--defaults",
        type=str,
        nargs="+",
        default=None,
        help=(
            "config files providing defaults for this config, resolved relative to it."
            " Later entries override earlier ones, this config overrides all of them,"
            " and the command line overrides everything. Values are replaced per"
            " top-level key, not deep-merged."
        ),
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
    parser.add_argument("--input_test", type=str, nargs="+")
    parser.add_argument("--slice_pct_train", type=float, nargs=2, default=(0.0, 1.0))
    parser.add_argument("--slice_pct_val", type=float, nargs=2, default=(0.0, 1.0))
    parser.add_argument("--downscale_temporal", type=int, default=1)
    parser.add_argument("--downscale_spatial", type=int, default=1)
    parser.add_argument(
        "--spatial_cutoff",
        dest="spatial_cutoff",
        type=int,
        default=256,
        help="Hard spatial radius in model coordinates for attention and tracking.",
    )
    parser.add_argument("--normalize_diameter", type=float, default=None)
    parser.add_argument("--train_samples", type=int, default=10000)
    parser.add_argument("--num_encoder_layers", type=int, default=6)
    parser.add_argument(
        "--num_decoder_layers",
        type=int,
        default=None,
        help="decoder depth; None mirrors --num_encoder_layers, 0 under --encoder_only",
    )
    parser.add_argument("--pos_embed_per_dim", type=int, default=32)
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
    parser.add_argument(
        "--encoder_only",
        type=str2bool,
        default=False,
        help="drop the decoder; the head sees the encoder output on both sides (y = x)",
    )
    parser.add_argument(
        "--compile",
        type=str2bool,
        default=False,
        help="wrap the training-step model forward in torch.compile",
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
        "--node_loss",
        type=float,
        default=0.0,
        help="weight (lambda) of the auxiliary node in/out-degree loss; 0 disables "
        "the node heads entirely (default: %(default)s)",
    )
    parser.add_argument(
        "--consistency_weight",
        type=float,
        default=0.1,
        help="weight of the degree-consistency loss pulling the edge-implied "
        "out-degree toward the node head's prediction; only active when "
        "--node_loss>0 (default: %(default)s)",
    )
    parser.add_argument(
        "--max_out_degree",
        type=int,
        default=2,
        help="width of the node out-degree head: out-degree in 0..max_out_degree "
        "(2 = up to a 2-way division); only used with --node_loss>0 "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--max_in_degree",
        type=int,
        default=1,
        help="width of the node in-degree head: in-degree in 0..max_in_degree "
        "(1 = a single parent); only used with --node_loss>0 (default: %(default)s)",
    )
    parser.add_argument(
        "--grad_log_every_n_epochs",
        type=int,
        default=10,
        help="compute/log the (expensive) full-model grad norm only every N epochs",
    )
    parser.add_argument("--augment", type=int, default=3)
    parser.add_argument(
        "--augment_details",
        type=_parse_augment_details,
        default=None,
        help=(
            "Optional mapping overriding preset augmentation magnitudes: "
            "jitter, drift, tilt, frame_jump, frame_jump_p."
        ),
    )
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
        "--feature_drop",
        type=float,
        default=0.0,
        help=(
            "per-window probability of masking each wrfeat2 group (intensity, shape) "
            "during training, so the model stays robust to datasets missing a group"
        ),
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
        "--oversample_divs",
        type=float,
        default=0.0,
        help=(
            "Oversample division-rich windows via the power law "
            "(1 + n_divs) ** oversample_divs; 0 disables (uniform)"
        ),
    )
    parser.add_argument(
        "--oversample_density",
        type=float,
        default=0.0,
        help=(
            "Oversample dense (harder) windows within each dataset via a power law "
            "on the per-dataset median-relative detection count; 0 disables. "
            "Renormalized per dataset so the dataset mixture is preserved"
        ),
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

    # resolve a `defaults:` chain in the selected config into configargparse's
    # default config files, which rank below the selected config, which ranks
    # below the command line.
    bootstrap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    bootstrap.add_argument("-c", "--config", type=Path, default=None)
    config = bootstrap.parse_known_args()[0].config
    if config is not None:
        parser._default_config_files = _resolve_defaults(config)

    args, unknown_args = parser.parse_known_args()
    args.input_train = _restore_input_config_items(args.input_train)
    args.input_val = _restore_input_config_items(args.input_val)
    args.input_test = _restore_input_config_items(args.input_test)

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
