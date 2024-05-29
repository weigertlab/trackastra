import os

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import pickle

import torch
import torch.multiprocessing

# torch.multiprocessing.set_sharing_strategy("file_system")
torch.set_float32_matmul_precision("medium")

import hashlib
import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from timeit import default_timer

import configargparse
import git
import lightning as pl
import numpy as np
import psutil
import trackastra
import wandb
import yaml
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.profilers import PyTorchProfiler
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from numerize import numerize
from skimage.morphology import binary_dilation, disk
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm
from trackastra.data import (
    # load_ctc_data_from_subfolders,
    BalancedBatchSampler,
    BalancedDataModule,
    CTCData,
    collate_sequence_padding,
)
from trackastra.model import TrackingTransformer
from trackastra.utils import (
    blockwise_causal_norm,
    blockwise_sum,
    normalize,
    preallocate_memory,
    random_label_cmap,
    render_label,
    seed,
    str2bool,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.simplefilter(action="ignore", category=FutureWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "mps")
np.seterr(all="ignore")


def _process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss


def _git_commit():
    """Returns the git commit hash of the current repository if it exists, otherwise None (for debugging purposes)."""
    logging.debug(f"Trackastra path: {Path(trackastra.__path__[0]).resolve()}")
    try:
        commit = str(git.Repo(Path(trackastra.__path__[0]).resolve().parent).commit())
    except:  # noqa: E722
        commit = "none"
    return commit


class WarmupCosineLRScheduler(LRScheduler):
    """A linear warmup + cosine lr scheduler."""

    def __init__(
        self,
        optimizer,
        warmup_epochs,
        max_epochs,
        cosine_final: float = 0.001,
        last_epoch=-1,
        verbose=False,
    ):
        """Use cosine_final to switch on/off the cosine annealing.

        cosine_final=0 -> reduce to 0 at the end of training
        cosine_final=1 -> dont reduce at all.
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.cosine_final = cosine_final
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`."
            )

        if self.last_epoch < self.warmup_epochs:
            # linear ramp
            initial = 1e-2
            factor = initial + (1 - initial) * self.last_epoch / self.warmup_epochs
        else:
            # cosine annealing
            epoch_rel = (self.last_epoch - self.warmup_epochs) / (
                self.max_epochs - self.warmup_epochs + 1
            )
            factor = (
                0.5 * (1 + np.cos(np.pi * epoch_rel)) * (1 - self.cosine_final)
                + self.cosine_final
            )

        logging.info(f"LRScheduler: relative lr factor {factor:.03f}")
        return [factor * base_lr for base_lr in self.base_lrs]


def log_tracking_metrics(model, _data, causal_norm: str, delta: int):
    from types import SimpleNamespace

    from predict import predict_ctc

    from tracking import tracking

    window = model.window
    args_pred = SimpleNamespace(
        t_start=0, t_end=len(_data) + window - 1, gt=False, delta=delta, thresh=0.1
    )

    with TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir)
        predict_ctc(_data, model, window, outdir, args_pred, causal_norm=causal_norm)
        args_track = SimpleNamespace(
            stop=None,
            input=outdir / "graph.pkl",
            dry=False,
            name=None,
            napari=False,
            metrics_every=1,
            img=_data.img_folder,
            feat="track",
            metrics_n_timesteps=50,
            use_distance=False,
            outdir=None,
            metrics=False,
            last_metrics=True,
            masks=_data.mask_folder,
            gt=None,
            mode="ilp",
            max_distance=50,
        )
        (
            df,
            df_metric,
            df_mot,
            masks,
            graph,
            tracks_graph,
            tracks,
            masks_original,
            viewer,
        ) = tracking(args_track)
    return df_metric


# define the LightningModule that contains the TrackingTransformer (to separate torch and lightning)
# this contains all the training/loss logic
class WrappedLightningModule(pl.LightningModule):
    def __init__(
        self,
        model,
        warmup_epochs: int = 10,
        max_epochs: int = 100,
        learning_rate: float = 1e-5,
        causal_norm: str = "none",
        delta_cutoff: int = 2,
        tracking_frequency: int = -1,  # log TRA metrics every that epochs
        batch_val_tb_idx: int = 0,  # the batch index to visualize in tensorboard
        div_upweight: float = 20,
    ):
        super().__init__()

        self.model = model
        self.causal_norm = causal_norm
        self.delta_cutoff = delta_cutoff
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.criterion_softmax = torch.nn.BCELoss(reduction="none")
        # self.train_loss = []
        # self.val_loss = []
        self.batch_val_tb_idx = batch_val_tb_idx
        self.batch_val_tb = None

        self.lr = learning_rate
        self.tracking_frequency = tracking_frequency
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.div_upweight = div_upweight

    def _common_step(self, batch, eps=torch.finfo(torch.float32).eps):
        feats = batch["features"]
        coords = batch["coords"]
        A = batch["assoc_matrix"]
        timepoints = batch["timepoints"]
        padding_mask = batch["padding_mask"]
        padding_mask = padding_mask.bool()

        A_pred = self.model(coords, feats, padding_mask=padding_mask)
        # remove inf values that might happen due to float16 numerics
        A_pred.clamp_(torch.finfo(torch.float16).min, torch.finfo(torch.float16).max)

        mask_invalid = torch.logical_or(
            padding_mask.unsqueeze(1), padding_mask.unsqueeze(2)
        )

        A_pred[mask_invalid] = 0
        loss = self.criterion(A_pred, A)

        if self.causal_norm != "none":
            # TODO speedup: I could softmax only the part of the matrix (upper triangular) that is not masked out
            A_pred_soft = torch.stack(
                [
                    blockwise_causal_norm(
                        _A, _t, mode=self.causal_norm, mask_invalid=_m
                    )
                    for _A, _t, _m in zip(A_pred, timepoints, mask_invalid)
                ]
            )
            with torch.cuda.amp.autocast(enabled=False):
                if len(A) > 0:
                    # debug
                    if torch.any(torch.isnan(A_pred_soft)):
                        print(A_pred)
                        print(
                            "AAAA pred",
                            A_pred_soft.min().item(),
                            A_pred_soft.max().item(),
                        )
                        print("AAAA pred", A_pred_soft.shape)
                        print("AAAA pred", A_pred_soft.dtype)
                        print("AAAA", A.min().item(), A.max().item())
                        print("AAAA", A.shape)
                        print("AAAA", A.dtype)
                        print("A_pred_soft has nan")
                        np.savez(
                            "runs/nan.npz",
                            A_pred=A_pred.detach().cpu().numpy(),
                            timepoints=timepoints.detach().cpu().numpy(),
                        )

                # Keep the non-softmaxed loss for numerical stability
                loss = 0.01 * loss + self.criterion_softmax(A_pred_soft, A)

        # Reweighting does not need gradients
        with torch.no_grad():
            block_sum1 = torch.stack(
                [blockwise_sum(A, t, dim=-1) for A, t in zip(A, timepoints)], 0
            )
            block_sum2 = torch.stack(
                [blockwise_sum(A, t, dim=-2) for A, t in zip(A, timepoints)], 0
            )
            block_sum = A * (block_sum1 + block_sum2)

            normal_tracks = block_sum == 2
            division_tracks = block_sum > 2

            # upweight normal (not starting or ending) tracks and division tracks
            loss_weight = 1 + 1.0 * normal_tracks + self.div_upweight * division_tracks

        loss = loss * loss_weight

        mask_valid = ~mask_invalid
        dt = timepoints.unsqueeze(1) - timepoints.unsqueeze(2)
        mask_time = torch.logical_and(dt > 0, dt <= self.delta_cutoff)

        mask = mask_time * mask_valid
        mask = mask.float()

        loss_before_reduce = loss * mask
        # Normalized by number of valid entry for each sample
        # Here I get a loss that is normalized by the number of connections to predict
        loss_normalized = loss_before_reduce / (
            mask.sum(dim=(1, 2), keepdim=True) + eps
        )
        loss_per_sample = loss_normalized.sum(dim=(1, 2))

        # Hack: weight larger samples a little more...
        prefactor = torch.pow(mask.sum(dim=(1, 2)), 0.2)

        loss = loss_per_sample * prefactor / (prefactor.sum() + eps)
        loss = loss.sum()

        # print(padding_mask.float().mean())
        return dict(
            loss=loss,
            padding_fraction=padding_mask.float().mean(),
            loss_before_reduce=loss_before_reduce,
            A_pred=A_pred,
            mask=mask,
            mask_time=mask_time,
            mask_valid=mask_valid,
        )

    def checkpoint_path(self, logdir):
        path = Path(logdir) / "checkpoints" / "last.ckpt"
        if path.exists():
            return path
        else:
            return None

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-5)
        return dict(
            optimizer=optimizer,
            lr_scheduler=WarmupCosineLRScheduler(
                optimizer, self.warmup_epochs, self.max_epochs
            ),
        )

    def training_step(self, batch, batch_idx):
        out = self._common_step(batch)
        loss = out["loss"]
        if torch.isnan(loss):
            print("NaN loss, skipping")
            return None

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        # self.train_loss.append(loss)

        self.log_dict(
            {
                "detections_per_sequence": batch["coords"].shape[1],
                "padding_fraction": out["padding_fraction"],
            },
            on_step=True,
            on_epoch=False,
        )

        return loss

    # def on_train_epoch_end(self):
    #     loss = torch.stack(self.train_loss).mean()
    #     if isinstance(self.logger, TensorBoardLogger):
    #         self.logger.experiment.add_scalars(
    #             "loss", {"train": loss}, self.current_epoch
    #         )
    #     self.train_loss.clear()

    def validation_step(self, batch, batch_idx):
        out = self._common_step(batch)
        loss = out["loss"]
        if torch.isnan(loss):
            print("NaN loss, skipping")
            return None

        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        # self.val_loss.append(loss)
        if batch_idx == self.batch_val_tb_idx:
            self.batch_val_tb = dict(batch=batch, out=out)

        return loss

    def on_validation_epoch_end(self):
        # skip if sanity checking
        if self.trainer.sanity_checking:
            return

        # val_loss = torch.stack(self.val_loss).mean()

        # if isinstance(self.logger, TensorBoardLogger):
        #     self.logger.experiment.add_scalars(
        #         "loss", {"val": val_loss}, self.current_epoch
        #     )
        # self.val_loss.clear()

        # Hack to make lightning progress bars with loss values persistent
        print(" ")

        if (
            self.tracking_frequency > 0
            and self.current_epoch % self.tracking_frequency == 0
        ):
            _data = self.trainer.val_dataloaders.dataset.datasets[0].datasets[0]
            try:
                metrics = log_tracking_metrics(
                    self.model, _data, self.causal_norm, self.delta_cutoff
                )
                self.logger.experiment.add_scalar(
                    "tra_error", 1 - metrics["TRA"].mean(), self.current_epoch
                )
            except Exception as e:
                logging.error(f"Error logging tracking metrics: {e}")

        if self.batch_val_tb is not None:
            batch = self.batch_val_tb["batch"]
            out = self.batch_val_tb["out"]

            # First sample of the batch
            sample = 0
            A_gt = batch["assoc_matrix"][sample]
            timepoints = batch["timepoints"][sample]
            A_pred = out["A_pred"][sample]
            loss_before_reduce = out["loss_before_reduce"][sample]

            if self.causal_norm != "none":
                A_pred = blockwise_causal_norm(
                    A_pred, timepoints, mode=self.causal_norm
                )
            else:
                A_pred = torch.sigmoid(A_pred)

            # create grid of timepoints for visualization
            time_grid = torch.diff(timepoints, append=timepoints[-1:]) != 0
            time_grid = time_grid.unsqueeze(0) + time_grid.unsqueeze(1)

            over = torch.stack((A_pred, A_gt, A_pred), 0)
            # add grid as blue background
            over[2, time_grid] += 0.2
            # over = over.unsqueeze(0)

            if isinstance(self.logger, TensorBoardLogger):
                self.logger.experiment.add_image(
                    "assoc matrix", over, self.current_epoch, dataformats="CHW"
                )

                self.logger.experiment.add_image(
                    "loss",
                    loss_before_reduce.unsqueeze(0),
                    self.current_epoch,
                    dataformats="CHW",
                )
                self.logger.experiment.add_image(
                    "loss_mask",
                    out["mask"][sample].unsqueeze(0),
                    self.current_epoch,
                    dataformats="CHW",
                )

                # Log losses for different delta t's (in loss_val writer)
                dt = timepoints[None, :] - timepoints[:, None]
                for _, delta in enumerate(range(1, self.delta_cutoff + 1)):
                    loss_before_reduce[dt == delta].mean()
                    # if isinstance(self.logger, TensorBoardLogger):
                    #     self.logger.experiment.add_scalar(
                    #         f"val_loss/delta_t={delta}", lt, self.current_epoch
                    #     )

                for name, enc_decs in zip(
                    ("encoder", "decoder"), (self.model.encoder, self.model.decoder)
                ):
                    for i, mod in enumerate(enc_decs):
                        if not mod.attn._mode == "bias":
                            continue

                        pos = mod.attn.pos_bias
                        temp, spat, nhead = (
                            pos.temporal_bins,
                            pos.spatial_bins,
                            pos.bias.shape[-1],
                        )

                        bias = pos.bias.view(len(temp), len(spat), nhead).transpose(
                            -1, 0
                        )
                        bias = bias.transpose(1, 2).unsqueeze(1)
                        bias = make_grid(bias, nrow=2, normalize=True)
                        self.logger.experiment.add_image(
                            f"pos bias {name} {i}",
                            bias,
                            self.current_epoch,
                            dataformats="CHW",
                        )

            elif isinstance(self.logger, WandbLogger):
                pass
                # wandb.log(
                #     {
                #         "images/assoc_matrix": wandb.Image(
                #             np.moveaxis(over.detach().cpu().numpy(), 0, -1), mode="RGB"
                #         ),
                #         "images/loss": wandb.Image(
                #             loss_before_reduce.unsqueeze(2).detach().cpu().numpy()
                #         ),
                #         "images/loss_mask": wandb.Image(
                #             out["mask"][sample].unsqueeze(2).detach().cpu().numpy()
                #         ),
                #     },
                #     step=self.current_epoch,
                # )
            elif self.logger is None:
                pass
            else:
                raise ValueError(f"Unknown logger {self.logger}")


class ExampleImages(pl.pytorch.callbacks.Callback):
    def __init__(self, n_samples=2, mode="overlay"):
        """Log example images to tensorboard.

        Args:
            n_samples (int, optional): _description_. Defaults to 5.
            mode (str, optional): grid for speed, overlay for beauty.
        """
        self._n_samples = n_samples
        self.mode = mode
        self.cmap = random_label_cmap()

    def on_train_start(self, trainer, pl_module):
        start = default_timer()

        if isinstance(trainer.train_dataloader.dataset, torch.utils.data.ConcatDataset):
            dataset = trainer.train_dataloader.dataset.datasets[0]
        else:
            dataset = trainer.train_dataloader.dataset

        for n in range(min(self._n_samples, len(dataset))):
            sample = dataset.__getitem__(n, return_dense=True)
            sample_img_normalized = np.clip(normalize(sample["img"].numpy()), 0, 1)
            assert sample_img_normalized.ndim == 3
            for i, (img, mask) in tqdm(
                enumerate(zip(sample_img_normalized, sample["mask"])),
                desc="Logging example images",
                leave=False,
            ):
                coords = sample["coords"][
                    sample["timepoints"] == sample["timepoints"].min() + i
                ]
                coords = coords[:, 1:].numpy().astype(int)
                points = np.zeros_like(img)
                points[coords[:, 0], coords[:, 1]] = 1

                points = binary_dilation(points, footprint=disk(3)).astype(float)

                if self.mode == "overlay":
                    # Overlay is pretty slow
                    overlay = render_label(
                        lbl=mask, img=img, cmap=self.cmap, normalize_img=False
                    )
                    overlay = torch.from_numpy(overlay[..., :3])

                    overlay = torch.maximum(
                        overlay, torch.as_tensor(points).unsqueeze(-1).expand(-1, -1, 3)
                    )

                    if isinstance(pl_module.logger, TensorBoardLogger):
                        pl_module.logger.experiment.add_image(
                            f"example_images/{n}_img",
                            overlay,
                            i,
                            dataformats="HWC",
                        )
                    elif isinstance(pl_module.logger, WandbLogger):
                        wandb.log(
                            {
                                f"example_images/{n}_img": wandb.Image(
                                    overlay.numpy(), mode="RGB"
                                )
                            },
                            # step=0,
                        )
                    else:
                        raise ValueError(f"Unknown logger {pl_module.logger}")

                elif self.mode == "grid":
                    img = torch.from_numpy(img).unsqueeze(0).expand(3, -1, -1)
                    mask = torch.from_numpy(self.cmap(mask)).moveaxis(-1, 0)[:3]
                    grid = torch.stack([img, mask], dim=0)
                    if isinstance(pl_module.logger, TensorBoardLogger):
                        pl_module.logger.experiment.add_images(
                            f"example_images/{n}_img",
                            grid,
                            i,
                            dataformats="NCHW",
                        )
                    if isinstance(pl_module.logger, WandbLogger):
                        raise NotImplementedError()
                else:
                    raise ValueError(f"Unknown mode {self.mode}")

        print(f"Logged example images in {(default_timer() - start):.1f} s")


# a modelcheckpoint that uses TrackingTransformer.save() to save the model
class MyModelCheckpoint(pl.pytorch.callbacks.Callback):
    def __init__(self, logdir, training_args: dict, monitor: str = "val_loss"):
        self._logdir = Path(logdir)
        self._monitor = monitor
        self._best = np.inf
        self._training_args = training_args

    def on_fit_start(self, trainer, pl_module):
        if trainer.is_global_zero:
            logging.info(f"using logdir {self._logdir}")
            self._logdir.mkdir(parents=True, exist_ok=True)
            with open(self._logdir / "train_config.yaml", "tw") as f:
                yaml.safe_dump(self._training_args, f)

    def on_validation_end(self, trainer, pl_module):
        if trainer.is_global_zero and not trainer.sanity_checking:
            value = trainer.logged_metrics[self._monitor]
            if value < self._best:
                self._best = value
                logging.info(f"saved best model with {self._monitor}={value:.5f}")
                pl_module.model.save(self._logdir)


# def weight_matrix(coords: torch.Tensor, scale: float = 100):
#     D = torch.linalg.norm(coords.unsqueeze(1) - coords.unsqueeze(2), dim=-1)
#     weight = 1 + 10 * torch.exp(-(D**2) / 2 / scale**2).to(coords.device)
#     return weight


def create_run_name(args):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # name = f"{timestamp}_{args.name}_feats_{args.features}_pos_{args.attn_positional_bias}_causal_norm_{args.causal_norm}"
    if args.timestamp:
        name = f"{timestamp}_{args.name}"
    else:
        name = args.name
    return name


def cache_class(cachedir=None):
    """A simple file cache for CTCData."""

    def make_hashable(obj):
        if isinstance(obj, (tuple, list)):
            return tuple(make_hashable(e) for e in obj)
        elif isinstance(obj, Path):
            return obj.as_posix()
        elif isinstance(obj, dict):
            return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
        else:
            return obj

    def hash_args_kwargs(*args, **kwargs):
        hashable_args = tuple(make_hashable(arg) for arg in args)
        hashable_kwargs = make_hashable(kwargs)
        combined_serialized = json.dumps(
            [hashable_args, hashable_kwargs], sort_keys=True
        )
        hash_obj = hashlib.sha256(combined_serialized.encode())
        return hash_obj.hexdigest()

    if cachedir is None:
        return CTCData
    else:
        cachedir = Path(cachedir)

        def _wrapped(*args, **kwargs):
            h = hash_args_kwargs(*args, **kwargs)
            cachedir.mkdir(exist_ok=True, parents=True)
            cache_file = cachedir / f"{h}.pkl"
            if cache_file.exists():
                logger.info(f"Loading cached dataset from {cache_file}")
                with open(cache_file, "rb") as f:
                    return pickle.load(f)
            else:
                c = CTCData(*args, **kwargs)
                logger.info(f"Saving cached dataset to {cache_file}")
                pickle.dump(c, open(cache_file, "wb"))
            return c

        return _wrapped


def find_val_batch(loader_val, n_gpus):
    # find the val batch with most divisions for vizualisation, which runs on GPU 0
    batches_val = tuple(
        tqdm(loader_val, desc="Scanning val batches for max divs", leave=False)
    )

    n_divs = []
    n_dets = 0
    for batch in batches_val[: len(batches_val) // n_gpus]:

        _n_divs = (
            (
                blockwise_sum(batch["assoc_matrix"][0], batch["timepoints"][0]).max(
                    dim=0
                )[0]
                == 2
            )
            .sum()
            .item()
        )
        n_divs.append(_n_divs)
        assert len(batch["timepoints"]) == 1
        _n_dets = len(batch["timepoints"][0])
        n_dets += _n_dets
        logger.debug(f"{_n_divs=}, {_n_dets=}")

    logger.info(
        f"Validation set division/detection ratio: {np.array(n_divs).sum() / n_dets}"
    )
    batch_val_tb_idx = np.argsort(n_divs)[-1]
    return batch_val_tb_idx


@rank_zero_only
def _init_wandb(project, name, config):
    _ = wandb.init(project=project, name=name, config=config)


def train(args):
    args.seed = seed(args.seed)
    if args.model is None:
        logger.warning("Training from scratch, this is slow!\n")

    args.warmup_epochs = min(args.warmup_epochs, args.epochs)

    if args.delta_cutoff is None:
        args.delta_cutoff = args.window

    memory = _process_memory()

    if args.features == "wrfeat" and args.feat_embed_per_dim <= 1:
        raise ValueError("For wrfeat, feat_embed_per_dim must be > 1 (e.g. 8)")

    callbacks = []
    if not args.dry:
        run_name = create_run_name(args)
        logdir = Path(args.outdir) / run_name
        # saving checkpoints in case training gets restarted
        callbacks.append(
            pl.pytorch.callbacks.ModelCheckpoint(
                dirpath=logdir / "checkpoints",
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                save_last=True,
            )
        )

        training_args = vars(args)
        training_args["git_commit"] = _git_commit()
        callbacks.append(MyModelCheckpoint(logdir, training_args, monitor="val_loss"))

        if args.logger == "tensorboard":
            train_logger = TensorBoardLogger(logdir, name="tb")
        elif args.logger == "wandb":
            train_logger = WandbLogger(name=run_name, project=args.wandb_project)

            # init here to get an alert on job failure even before training
            _init_wandb(project=args.wandb_project, name=run_name, config=vars(args))

        elif args.logger == "none":
            train_logger = False
        else:
            raise ValueError(f"Unknown logger {args.logger}")
    else:
        logdir = None
        train_logger = False

    if logdir is not None and logdir.exists() and not args.resume:
        raise ValueError(
            f'Logdir {logdir} exists, set "--resume t"  if you want to overwrite'
        )

    datasets = dict()

    CTCData = cache_class(args.cachedir if args.cache else None)

    n_gpus = torch.cuda.device_count() if args.distributed else 1

    if args.preallocate:
        logger.info("Load one dataset to preallocate memory for training")
        dummy_data = CTCData(
            root=Path(args.input_train[0]),
            ndim=args.ndim,
            detection_folders=args.detection_folders,
            window_size=args.window,
            max_tokens=args.max_tokens,
            augment=args.augment,
            features=args.features,
            slice_pct=args.slice_pct_train,
            downscale_temporal=args.downscale_temporal,
            downscale_spatial=args.downscale_spatial,
            sanity_dist=args.sanity_dist,
            crop_size=args.crop_size,
            compress=args.compress,
        )

        dummy_model = TrackingTransformer(
            coord_dim=dummy_data.ndim,
            feat_dim=dummy_data.feat_dim,
            d_model=args.d_model,
            pos_embed_per_dim=args.pos_embed_per_dim,
            feat_embed_per_dim=args.feat_embed_per_dim,
            num_encoder_layers=args.num_encoder_layers,
            num_decoder_layers=args.num_decoder_layers,
            dropout=args.dropout,
            window=args.window,
            spatial_pos_cutoff=args.spatial_pos_cutoff,
            attn_positional_bias=args.attn_positional_bias,
            attn_positional_bias_n_spatial=args.attn_positional_bias_n_spatial,
            causal_norm=args.causal_norm,
        )

        dummy_model_lightning = WrappedLightningModule(
            model=dummy_model,
            warmup_epochs=args.warmup_epochs,
            max_epochs=args.epochs,
            learning_rate=args.lr,
            delta_cutoff=args.delta_cutoff,
            causal_norm=args.causal_norm,
            tracking_frequency=args.tracking_frequency,
            batch_val_tb_idx=0,
            div_upweight=args.div_upweight,
        )
        dummy_model_lightning.to(device)
        preallocate_memory(
            dummy_data,
            dummy_model_lightning,
            args.batch_size // n_gpus,
            args.max_tokens,
            device,
        )
        del dummy_model_lightning
        del dummy_data

    for p in args.input_train + args.input_val:
        if not Path(p).exists():
            raise FileNotFoundError(f"Input folder {p} does not exist")

    for split, inps, slice_pct in zip(
        ("train", "val"),
        (args.input_train, args.input_val),
        (args.slice_pct_train, args.slice_pct_val),
    ):
        logger.info(f"Loading {split.upper()} data")
        start = default_timer()
        datasets[split] = torch.utils.data.ConcatDataset(
            CTCData(
                root=Path(inp),
                ndim=args.ndim,
                detection_folders=args.detection_folders,
                window_size=args.window,
                max_tokens=args.max_tokens,
                augment=args.augment if split == "train" else 0,
                features=args.features,
                slice_pct=slice_pct,
                downscale_temporal=args.downscale_temporal,
                downscale_spatial=args.downscale_spatial,
                sanity_dist=args.sanity_dist,
                crop_size=args.crop_size if split == "train" else None,
                compress=args.compress,
            )
            for inp in inps
        )
        logger.info(
            f"Loaded {len(datasets[split])} {split.upper()} samples (in"
            f" {(default_timer() - start):.1f} s)\n\n"
        )

    common_loader_args = dict(
        persistent_workers=True if args.num_workers > 0 else False,
        pin_memory=True,
        collate_fn=collate_sequence_padding(),
    )

    datamodule = None
    # Sampler gets wrapped with distributed sampler, which cannot sample with replacement
    if args.distributed:
        datamodule = BalancedDataModule(
            datasets["train"],
            datasets["val"],
            num_samples=args.train_samples if args.train_samples > 0 else None,
            batch_size=args.batch_size,
            n_pool=args.n_pool_sampler,
            num_workers=args.num_workers,
            weight_by_ndivs=args.weight_by_ndivs,
            weight_by_dataset=args.weight_by_dataset,
            **common_loader_args,
        )

    else:

        if args.n_pool_sampler > 0:
            batch_sampler = BalancedBatchSampler(
                datasets["train"],
                batch_size=args.batch_size,
                n_pool=args.n_pool_sampler,
                num_samples=args.train_samples if args.train_samples > 0 else None,
                weight_by_ndivs=args.weight_by_ndivs,
                weight_by_dataset=args.weight_by_dataset,
            )

            loader_train = DataLoader(
                datasets["train"],
                batch_sampler=batch_sampler,
                num_workers=args.num_workers,
                **common_loader_args,
            )
        else:
            if args.weight_by_ndivs:
                raise NotImplementedError(
                    "Weighting by ndivs only works with balanced samples (i.e."
                    " n_pool_sampler > 0)"
                )
            sampler = torch.utils.data.RandomSampler(
                datasets["train"],
                num_samples=(
                    args.train_samples
                    if args.train_samples > 0
                    else len(datasets["train"])
                ),
                replacement=True,
            )
            loader_train = DataLoader(
                datasets["train"],
                sampler=sampler,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                **common_loader_args,
            )

    loader_val = DataLoader(
        datasets["val"],
        batch_size=1,
        shuffle=False,
        num_workers=0 if args.num_workers == 0 else max(1, args.num_workers // 2),
        **common_loader_args,
    )

    for k, v in datasets.items():
        print(f"Dataset {k}: {len(v)} samples")

    batch_val_tb_idx = find_val_batch(loader_val, n_gpus)
    # batch_val_tb_idx = 0

    if train_logger:
        callbacks.append(
            pl.pytorch.callbacks.LearningRateMonitor(logging_interval="epoch")
        )

    callbacks.append(pl.pytorch.callbacks.Timer(interval="epoch"))

    if args.example_images:
        callbacks.append(ExampleImages())

    # load the model if it was given
    if args.model is not None:
        fpath = Path(args.model)

        # allow for checkpoints to be loaded too
        if fpath.is_file():
            model = TrackingTransformer.from_folder(
                Path(*fpath.parts[:2]),
                args=args,
                checkpoint_path=Path(*fpath.parts[2:]),
            )
        else:
            model = TrackingTransformer.from_folder(fpath, args=args)
    else:
        model = TrackingTransformer(
            coord_dim=datasets["train"].datasets[0].ndim,
            feat_dim=datasets["train"].datasets[0].feat_dim,
            d_model=args.d_model,
            pos_embed_per_dim=args.pos_embed_per_dim,
            feat_embed_per_dim=args.feat_embed_per_dim,
            num_encoder_layers=args.num_encoder_layers,
            num_decoder_layers=args.num_decoder_layers,
            dropout=args.dropout,
            window=args.window,
            spatial_pos_cutoff=args.spatial_pos_cutoff,
            attn_positional_bias=args.attn_positional_bias,
            attn_positional_bias_n_spatial=args.attn_positional_bias_n_spatial,
            causal_norm=args.causal_norm,
        )

    model_lightning = WrappedLightningModule(
        model=model,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.epochs,
        learning_rate=args.lr,
        delta_cutoff=args.delta_cutoff,
        causal_norm=args.causal_norm,
        tracking_frequency=args.tracking_frequency,
        batch_val_tb_idx=batch_val_tb_idx,
        div_upweight=args.div_upweight,
    )

    # if logdir already exists and --resume option is set, load the last checkpoint (eg when continuing training after crash)
    if logdir is not None and logdir.exists() and args.resume:
        logging.info("logdir exists, loading last state of model")
        fpath = model_lightning.checkpoint_path(logdir)
        if fpath is not None:
            model_lightning = WrappedLightningModule.load_from_checkpoint(
                fpath,
                model=model,
                warmup_epochs=args.warmup_epochs,
                max_epochs=args.epochs,
                learning_rate=args.lr,
                delta_cutoff=args.delta_cutoff,
                causal_norm=args.causal_norm,
                tracking_frequency=args.tracking_frequency,
                batch_val_tb_idx=batch_val_tb_idx,
                div_upweight=args.div_upweight,
            )
        else:
            logging.warning(f"No checkpoint found in {logdir}")

    model_lightning.to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model has {numerize.numerize(num_params)} parameters")

    if args.distributed:
        # strategy = "ddp"
        strategy = "ddp_find_unused_parameters_true"
    else:
        strategy = "auto"

    if args.profile:
        # profiler = AdvancedProfiler(dirpath=".", filename="profile")
        profiler = PyTorchProfiler(dirpath=".", filename="profile", skip_first=16)
    else:
        profiler = None

    trainer = pl.Trainer(
        accelerator="cuda",
        strategy=strategy,
        devices=n_gpus,
        precision="16-mixed" if args.mixedp else 32,
        logger=train_logger,
        num_nodes=1,
        max_epochs=args.epochs,
        callbacks=callbacks,
        profiler=profiler,
    )

    t = default_timer()

    if logdir is not None and args.resume:
        resume_path = model_lightning.checkpoint_path(logdir)
        logger.info(f"Resuming from {resume_path}")
    else:
        resume_path = None

    if args.epochs > 0:
        if args.distributed and datamodule is not None:
            trainer.fit(model_lightning, datamodule=datamodule, ckpt_path=resume_path)
        else:
            trainer.fit(
                model_lightning,
                train_dataloaders=loader_train,
                val_dataloaders=loader_val,
                ckpt_path=resume_path,
            )

    print(f"Time elapsed:     {(default_timer() - t)/60:.02f} min")
    print(f"CPU Memory used:  {(_process_memory() - memory)/1e9:.2f} GB")
    print(f"GPU Memory used : {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")

    return locals()


def parse_train_args():
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
    parser.add_argument("-d", "--d_model", type=int, default=256)
    parser.add_argument("-w", "--window", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--warmup_epochs", type=int, default=10)
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
    parser.add_argument("--spatial_pos_cutoff", type=int, default=256)
    parser.add_argument("--from_subfolder", action="store_true")
    parser.add_argument("--train_samples", type=int, default=10000)
    parser.add_argument("--num_encoder_layers", type=int, default=6)
    parser.add_argument("--num_decoder_layers", type=int, default=6)
    parser.add_argument("--pos_embed_per_dim", type=int, default=32)
    parser.add_argument("--feat_embed_per_dim", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.00)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--delta_cutoff", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--attn_positional_bias",
        type=str,
        choices=["rope", "bias", "none"],
        default="rope",
    )
    parser.add_argument("--attn_positional_bias_n_spatial", type=int, default=16)
    parser.add_argument("--mixedp", type=str2bool, default=True)
    parser.add_argument("--dry", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--features",
        type=str,
        choices=[
            "none",
            "regionprops",
            "regionprops2",
            "patch",
            "patch_regionprops",
            "wrfeat",
        ],
        default="wrfeat",
    )
    parser.add_argument(
        "--causal_norm",
        type=str,
        choices=["none", "linear", "softmax", "quiet_softmax"],
        default="quiet_softmax",
    )
    parser.add_argument("--div_upweight", type=float, default=2)

    parser.add_argument("--augment", type=int, default=3)
    parser.add_argument("--tracking_frequency", type=int, default=-1)

    parser.add_argument("--sanity_dist", action="store_true")
    parser.add_argument("--preallocate", type=str2bool, default=True)
    parser.add_argument(
        "--compress", type=str2bool, default=False, help="compress dataset"
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
    parser.add_argument("--resume", type=str2bool, default=True)
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
    parser.add_argument(
        "--example_images",
        type=str2bool,
        default=False,
        help="Log example images. Slow.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb", "none"],
    )
    parser.add_argument("--wandb_project", type=str, default="trackastra")
    parser.add_argument(
        "--crop_size",
        type=int,
        required=True,
        nargs="+",
        default=None,
        help="random crop size for augmentation",
    )
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
            "Inversely weight datasets by number of samples (to counter dataset size"
            " imbalance)"
        ),
    )

    args, unknown_args = parser.parse_known_args()

    # Hack to allow for --input_test
    allowed_unknown = ["input_test"]
    if not set(a.split("=")[0].strip("-") for a in unknown_args).issubset(
        set(allowed_unknown)
    ):
        raise ValueError(f"Unknown args: {unknown_args}")

    # pprint(vars(args))

    # for backward compatibility
    # if args.attn_positional_bias == "True":
    #     args.attn_positional_bias = "bias"
    # elif args.attn_positional_bias == "False":
    #     args.attn_positional_bias = False

    if args.distributed and hasattr(sys, "ps1"):
        raise ValueError(
            "Distributed training does not work in interactive mode. Run as `python"
            " train.py`."
        )

    return args


if __name__ == "__main__":
    args = parse_train_args()

    # from torch.profiler import profile, record_function, ProfilerActivity
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    #     vars = train(args)
    vars = train(args)
