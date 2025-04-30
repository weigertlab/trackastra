"""Data loading and sampling utils for distributed training."""

import hashlib
import json
import logging
import pickle
from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path
from timeit import default_timer

import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import (
    BatchSampler,
    ConcatDataset,
    DataLoader,
    Dataset,
    DistributedSampler,
)

from .data import CTCData

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def cache_class(cachedir=None):
    """A simple file cache for CTCData."""

    def make_hashable(obj):
        if isinstance(obj, tuple | list):
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


class BalancedBatchSampler(BatchSampler):
    """samples batch indices such that the number of objects in each batch is balanced
    (so to reduce the number of paddings in the batch).


    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int,
        n_pool: int = 10,
        num_samples: int | None = None,
        weight_by_ndivs: bool = False,
        weight_by_dataset: bool = False,
        drop_last: bool = False,
    ):
        """Setting n_pool =1 will result in a regular random batch sampler.

        weight_by_ndivs: if True, the probability of sampling an element is proportional to the number of divisions
        weight_by_dataset: if True, the probability of sampling an element is inversely proportional to the length of the dataset
        """
        if isinstance(dataset, CTCData):
            self.n_objects = dataset.n_objects
            self.n_divs = np.array(dataset.n_divs)
            self.n_sizes = np.ones(len(dataset)) * len(dataset)
        elif isinstance(dataset, ConcatDataset):
            self.n_objects = tuple(n for d in dataset.datasets for n in d.n_objects)
            self.n_divs = np.array(tuple(n for d in dataset.datasets for n in d.n_divs))
            self.n_sizes = np.array(
                tuple(len(d) for d in dataset.datasets for _ in range(len(d)))
            )
        else:
            raise NotImplementedError(
                f"BalancedBatchSampler: Unknown dataset type {type(dataset)}"
            )
        assert len(self.n_objects) == len(self.n_divs) == len(self.n_sizes)

        self.batch_size = batch_size
        self.n_pool = n_pool
        self.drop_last = drop_last
        self.num_samples = num_samples
        self.weight_by_ndivs = weight_by_ndivs
        self.weight_by_dataset = weight_by_dataset
        logger.debug(f"{weight_by_ndivs=}")
        logger.debug(f"{weight_by_dataset=}")

    def get_probs(self, idx):
        idx = np.array(idx)
        if self.weight_by_ndivs:
            probs = 1 + np.sqrt(self.n_divs[idx])
        else:
            probs = np.ones(len(idx))
        if self.weight_by_dataset:
            probs = probs / (self.n_sizes[idx] + 1e-6)

        probs = probs / (probs.sum() + 1e-10)
        return probs

    def sample_batches(self, idx: Iterable[int]):
        # we will split the indices into pools of size n_pool
        num_samples = self.num_samples if self.num_samples is not None else len(idx)
        # sample from the indices with replacement and given probabilites
        idx = np.random.choice(idx, num_samples, replace=True, p=self.get_probs(idx))

        n_pool = min(
            self.n_pool * self.batch_size,
            (len(idx) // self.batch_size) * self.batch_size,
        )

        batches = []
        for i in range(0, len(idx), n_pool):
            # the indices in the pool are sorted by their number of objects
            idx_pool = idx[i : i + n_pool]
            idx_pool = sorted(idx_pool, key=lambda i: self.n_objects[i])

            # such that we can create batches where each element has a similar number of objects
            jj = np.arange(0, len(idx_pool), self.batch_size)
            np.random.shuffle(jj)

            for j in jj:
                # dont drop_last, as this leads to a lot of lightning problems....
                # if j + self.batch_size > len(idx_pool):  # assume drop_last=True
                #     continue
                batch = idx_pool[j : j + self.batch_size]
                batches.append(batch)
        return batches

    def __iter__(self):
        idx = np.arange(len(self.n_objects))
        batches = self.sample_batches(idx)
        return iter(batches)

    def __len__(self):
        if self.num_samples is not None:
            return self.num_samples // self.batch_size
        else:
            return len(self.n_objects) // self.batch_size


class BalancedDistributedSampler(DistributedSampler):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        n_pool: int,
        num_samples: int,
        weight_by_ndivs: bool = False,
        weight_by_dataset: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(dataset=dataset, *args, drop_last=True, **kwargs)
        self._balanced_batch_sampler = BalancedBatchSampler(
            dataset,
            batch_size=batch_size,
            n_pool=n_pool,
            num_samples=max(1, num_samples // self.num_replicas),
            weight_by_ndivs=weight_by_ndivs,
            weight_by_dataset=weight_by_dataset,
        )

    def __len__(self) -> int:
        if self.num_samples is not None:
            return self._balanced_batch_sampler.num_samples
        else:
            return super().__len__()

    def __iter__(self):
        indices = list(super().__iter__())
        batches = self._balanced_batch_sampler.sample_batches(indices)
        for batch in batches:
            yield from batch


class BalancedDataModule(LightningDataModule):
    def __init__(
        self,
        input_train: list,
        input_val: list,
        cachedir: str,
        augment: int,
        distributed: bool, 
        dataset_kwargs: dict,
        sampler_kwargs: dict,
        loader_kwargs: dict,
    ):
        super().__init__()
        self.input_train = input_train
        self.input_val = input_val
        self.cachedir = cachedir
        self.augment = augment
        self.distributed = distributed
        self.dataset_kwargs = dataset_kwargs
        self.sampler_kwargs = sampler_kwargs
        self.loader_kwargs = loader_kwargs

    def prepare_data(self):
        """Loads and caches the datasets if not already done.

        Running on the main CPU process.
        """
        CTCData = cache_class(self.cachedir)
        datasets = dict()
        for split, inps in zip(
            ("train", "val"),
            (self.input_train, self.input_val),
        ):
            logger.info(f"Loading {split.upper()} data")
            start = default_timer()
            datasets[split] = torch.utils.data.ConcatDataset(
                CTCData(
                    root=Path(inp),
                    augment=self.augment if split == "train" else 0,
                    **self.dataset_kwargs,
                )
                for inp in inps
            )
            logger.info(
                f"Loaded {len(datasets[split])} {split.upper()} samples (in"
                f" {(default_timer() - start):.1f} s)\n\n"
            )

        del datasets

    def setup(self, stage: str):
        CTCData = cache_class(self.cachedir)
        self.datasets = dict()
        for split, inps in zip(
            ("train", "val"),
            (self.input_train, self.input_val),
        ):
            logger.info(f"Loading {split.upper()} data")
            start = default_timer()
            self.datasets[split] = torch.utils.data.ConcatDataset(
                CTCData(
                    root=Path(inp),
                    augment=self.augment if split == "train" else 0,
                    **self.dataset_kwargs,
                )
                for inp in inps
            )
            logger.info(
                f"Loaded {len(self.datasets[split])} {split.upper()} samples (in"
                f" {(default_timer() - start):.1f} s)\n\n"
            )

    def train_dataloader(self):
        loader_kwargs = self.loader_kwargs.copy()
        if self.distributed:
            sampler = BalancedDistributedSampler(
                self.datasets["train"],
                **self.sampler_kwargs,
            )
            batch_sampler = None
        else: 
            sampler = None
            batch_sampler = BalancedBatchSampler(
                self.datasets["train"],
                **self.sampler_kwargs,
            )
            if not loader_kwargs['batch_size'] == batch_sampler.batch_size:
                raise ValueError(f"Batch size in loader_kwargs ({loader_kwargs['batch_size']}) and sampler_kwargs ({batch_sampler.batch_size}) must match")            
            del loader_kwargs['batch_size']
        
        loader = DataLoader(
            self.datasets["train"],
            sampler=sampler,
            batch_sampler=batch_sampler,
            **loader_kwargs,
        )
        return loader

    def val_dataloader(self):
        val_loader_kwargs = deepcopy(self.loader_kwargs)
        val_loader_kwargs["persistent_workers"] = False
        val_loader_kwargs["num_workers"] = 1
        return DataLoader(
            self.datasets["val"],
            shuffle=False,
            **val_loader_kwargs,
        )
