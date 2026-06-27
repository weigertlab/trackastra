from pathlib import Path

import numpy as np
import torch
from trackastra.data.dataset import (
    TrackingDataset,
    _sample_detection_keep_indices,
    _sample_neighborhood_indices,
    association_distances,
    collate_sequence_padding,
    densify_assoc,
    warn_association_distances,
)
from trackastra.data.io import Segmentation, TrackingSequence


def test_detection_dropout_drops_whole_lineages():
    # two 3-frame lineages; dropping a first-frame seed removes its entire lineage,
    # so the kept set is always a complete lineage (never a severed partner).
    assoc = np.zeros((6, 6), dtype=bool)
    assoc[0, 1] = assoc[1, 2] = True  # lineage A: 0(t0) -> 1(t1) -> 2(t2)
    assoc[3, 4] = assoc[4, 5] = True  # lineage B: 3(t0) -> 4(t1) -> 5(t2)
    timepoints = np.array([0, 1, 2, 0, 1, 2])

    state = np.random.get_state()
    try:
        np.random.seed(0)
        keep = _sample_detection_keep_indices(assoc, timepoints, drop_fraction=0.5)
    finally:
        np.random.set_state(state)

    # exactly one of the two first-frame lineages survives, kept whole
    assert set(keep.tolist()) in ({0, 1, 2}, {3, 4, 5})


def test_detection_dropout_always_keeps_one_lineage():
    assoc = np.zeros((6, 6), dtype=bool)
    assoc[0, 1] = assoc[1, 2] = True
    assoc[3, 4] = assoc[4, 5] = True
    timepoints = np.array([0, 1, 2, 0, 1, 2])

    state = np.random.get_state()
    try:
        np.random.seed(0)
        keep = _sample_detection_keep_indices(assoc, timepoints, drop_fraction=1.0)
    finally:
        np.random.set_state(state)

    assert set(keep.tolist()) in ({0, 1, 2}, {3, 4, 5})


def test_neighborhood_sampling_keeps_whole_lineages(monkeypatch):
    # 3 first-frame cells, budget 2: keep the 2 closest seeds' full lineages only.
    coords = np.array(
        [[0, 0], [1, 0], [100, 0], [0, 0], [1, 0], [100, 0]], dtype=np.float32
    )
    timepoints = np.array([0, 0, 0, 1, 1, 1])
    labels = np.array([1, 2, 3, 1, 2, 3])
    assoc = labels[:, None] == labels[None, :]
    monkeypatch.setattr(np.random, "randint", lambda _n: 0)  # anchor = first seed

    keep = _sample_neighborhood_indices(coords, timepoints, assoc, max_detections=2)

    assert len(keep) == 4
    assert 2 not in keep and 5 not in keep  # far lineage dropped whole
    # no association of a kept detection points outside the kept set (no severed link)
    keep_set = set(keep.tolist())
    for i in keep:
        assert set(np.flatnonzero(assoc[i]).tolist()) <= keep_set


def test_neighborhood_sampling_leaves_small_first_frame_unchanged():
    coords = np.zeros((3, 2), dtype=np.float32)
    timepoints = np.arange(3)  # one detection in the first frame
    assoc = np.eye(3, dtype=bool)

    keep = _sample_neighborhood_indices(coords, timepoints, assoc, max_detections=3)

    assert np.array_equal(keep, np.arange(3))


def test_collate_assoc_coo_densifies_to_dense_padding():
    torch.manual_seed(0)

    def sample(n):
        return {
            "coords": torch.zeros((n, 3)),
            "features": torch.zeros((n, 1)),
            "labels": torch.arange(n),
            "timepoints": torch.zeros(n, dtype=torch.long),
            "assoc_matrix": (torch.rand(n, n) < 0.25).float(),
        }

    samples = [sample(3), sample(5), sample(2)]
    batch = collate_sequence_padding(samples)

    bsz = len(samples)
    n_max = max(len(s["coords"]) for s in samples)
    dense = densify_assoc(batch["assoc_coo"], bsz, n_max)

    # densified COO must match the old dense block-padding exactly
    ref = torch.zeros(bsz, n_max, n_max)
    for i, s in enumerate(samples):
        n = s["assoc_matrix"].shape[0]
        ref[i, :n, :n] = s["assoc_matrix"]
    assert torch.equal(dense, ref)
    # the dense (B, N, N) buffer is no longer shipped
    assert "assoc_matrix" not in batch
    assert batch["assoc_coo"].shape[1] == 3


def _single_lineage_sequence(xs: tuple[float, ...]) -> TrackingSequence:
    """One detection per frame, all the same tracklet, placed at the given x."""
    n_frames = len(xs)
    seg = Segmentation(
        name="TRA",
        n_frames=n_frames,
        coords=np.array([[x, 0.0] for x in xs], dtype=np.float32),
        labels=np.array([10 + t for t in range(n_frames)]),
        timepoints=np.arange(n_frames, dtype=np.int64),
        features={"v": np.zeros((n_frames, 1), dtype=np.float32)},
        track_indices=np.zeros(n_frames, dtype=np.int64),
    )
    return TrackingSequence(
        root=Path("synthetic"),
        ndim=2,
        segmentations=(seg,),
        lineage_relation=np.eye(1, dtype=bool),
        lineage_parents=np.full(1, -1),
    )


def test_association_distances_deduplicate_overlapping_windows():
    # 4 frames, window_size 3 -> windows [0,1,2] and [1,2,3] share the t1->t2 edge,
    # which must be counted once.
    sequence = _single_lineage_sequence((0.0, 3.0, 13.0, 30.0))
    data = TrackingDataset(sequence, window_size=3)

    distances = association_distances(data, delta_cutoff=1)

    assert sorted(distances.tolist()) == [3.0, 10.0, 17.0]


def test_association_distance_warning_ignores_values_at_cutoff(caplog):
    warn_association_distances(
        np.array([3.0, 10.0]),
        max_distance=10,
        delta_cutoff=1,
        cutoff_name="max_distance",
        dataset_name="test dataset",
    )
    assert caplog.text == ""


def test_association_distance_warning_reports_exceedances(caplog):
    import logging

    with caplog.at_level(logging.WARNING):
        warn_association_distances(
            np.array([3.0, 10.0]),
            max_distance=5,
            delta_cutoff=1,
            cutoff_name="max_distance",
            dataset_name="test dataset",
        )
    assert "1/2 (50.00%)" in caplog.text
    assert "max=10.00" in caplog.text
    assert "cannot be recovered" in caplog.text
