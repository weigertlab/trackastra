import pickle
from pathlib import Path
from types import SimpleNamespace

import joblib
import networkx as nx
import numpy as np
import pytest
import torch
from tifffile import imread, imwrite
from trackastra.data.dataset import TrackingDataset
from trackastra.data.io import (
    DetectionSequence,
    DetectionSupervision,
    LineageGraph,
    TrackingSequence,
    load_ctc_images_masks,
)
from trackastra.utils import normalize


def _lineage_graph(relation: np.ndarray, parents: np.ndarray) -> LineageGraph:
    return LineageGraph(
        coords=np.zeros((0, 2), dtype=np.float32),
        timepoints=np.zeros(0, dtype=np.int64),
        node_ids=np.zeros(0, dtype=object),
        lineage_relation=relation,
        lineage_parents=parents,
    )


def _detection_sequence(frames, name="TRA"):
    """Build flat detections and supervision from ``(timepoint, lineage_index)`` specs."""
    coords, labels, timepoints, lineage_index = [], [], [], []
    feats = {
        "equivalent_diameter_area": [],
        "intensity": [],
        "inertia_tensor": [],
        "border_dist": [],
    }
    for t, ti in frames:
        n = len(ti)
        coords.append(np.arange(n * 2, dtype=np.float32).reshape(n, 2))
        labels.append(np.arange(1, n + 1, dtype=np.int32))
        timepoints.append(np.full(n, t, dtype=np.int64))
        lineage_index.append(np.asarray(ti, dtype=np.int64))
        feats["equivalent_diameter_area"].append(np.full((n, 1), 2, np.float32))
        feats["intensity"].append(np.full((n, 1), 0.5, np.float32))
        feats["inertia_tensor"].append(
            np.tile(np.eye(2, dtype=np.float32).ravel(), (n, 1))
        )
        feats["border_dist"].append(np.zeros((n, 1), np.float32))
    return DetectionSequence(
        name=name,
        n_frames=max(t for t, _ in frames) + 1,
        coords=np.concatenate(coords),
        labels=np.concatenate(labels),
        timepoints=np.concatenate(timepoints),
        features={k: np.concatenate(v) for k, v in feats.items()},
    ), DetectionSupervision(lineage_index=np.concatenate(lineage_index))


def test_tracking_sequence_from_geff_gt_nodes(monkeypatch, tmp_path):
    graph = nx.DiGraph()
    graph.add_node(10, t=0, z=1, y=0, x=0, intensity=0.1)
    graph.add_node(20, t=1, z=2, y=0, x=0, intensity=0.2)
    graph.add_node(30, t=2, z=3, y=0, x=0, intensity=0.3)
    graph.add_node(40, t=2, z=2, y=1, x=0, intensity=0.4)
    graph.add_edges_from([(10, 20), (20, 30), (20, 40)])
    metadata = SimpleNamespace(
        axes=[
            SimpleNamespace(name="t", scale=1.0),
            SimpleNamespace(name="z", scale=4.0),
            SimpleNamespace(name="y", scale=1.0),
            SimpleNamespace(name="x", scale=1.0),
        ]
    )

    import geff

    monkeypatch.setattr(geff, "read", lambda _path: (graph, metadata))

    sequence = TrackingSequence.from_geff(
        tmp_path / "track.geff",
        spacing="auto",
        feature_columns=("intensity",),
    )

    seg = sequence.detections[0]
    assert sequence.ndim == 3
    assert seg.n_frames == 3
    np.testing.assert_allclose(
        seg.coords,
        np.array([[4, 0, 0], [8, 0, 0], [12, 0, 0], [8, 1, 0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        seg.source_coords,
        np.array([[1, 0, 0], [2, 0, 0], [3, 0, 0], [2, 1, 0]], dtype=np.float32),
    )
    assert seg.spacing == (4.0, 1.0, 1.0)
    np.testing.assert_allclose(seg.features["intensity"].ravel(), [0.1, 0.2, 0.3, 0.4])
    assert sequence.supervision[0].lineage_index.tolist() == [0, 0, 1, 2]
    assert sequence.supervision[0].gt_node_index.tolist() == [0, 1, 2, 3]
    assert sequence.gt.lineage_parents.tolist() == [-1, 0, 0]
    assert sequence.gt.node_in_degree.tolist() == [0, 1, 1, 1]
    assert sequence.gt.node_out_degree.tolist() == [1, 2, 0, 0]
    assert sequence.gt.node_predecessor_set_available.tolist() == [
        False,
        True,
        True,
        True,
    ]
    assert sequence.gt.node_successor_set_available.tolist() == [
        True,
        True,
        False,
        False,
    ]
    assert sequence.gt.lineage_relation.tolist() == [
        [True, True, True],
        [True, True, False],
        [True, False, True],
    ]


def test_tracking_sequence_from_geff_spacing_auto_and_unit(monkeypatch, tmp_path):
    graph = nx.DiGraph()
    graph.add_node(10, t=0, z=1, y=0, x=0)
    graph.add_node(20, t=1, z=2, y=0, x=0)
    graph.add_edge(10, 20)
    metadata = SimpleNamespace(
        axes=[
            SimpleNamespace(name="t", scale=1.0),
            SimpleNamespace(name="z", scale=4.0),
            SimpleNamespace(name="y", scale=1.0),
            SimpleNamespace(name="x", scale=1.0),
        ]
    )

    import geff

    monkeypatch.setattr(geff, "read", lambda _path: (graph, metadata))
    auto = TrackingSequence.from_geff(tmp_path / "track.geff", spacing="auto")
    np.testing.assert_allclose(auto.detections[0].coords, [[4, 0, 0], [8, 0, 0]])
    assert auto.detections[0].spacing == (4.0, 1.0, 1.0)

    unit = TrackingSequence.from_geff(tmp_path / "track.geff")
    np.testing.assert_allclose(unit.detections[0].coords, [[1, 0, 0], [2, 0, 0]])
    assert unit.detections[0].spacing == (1.0, 1.0, 1.0)

    monkeypatch.setattr(geff, "read", lambda _path: (graph, SimpleNamespace()))
    with pytest.raises(ValueError, match="metadata has no axes"):
        TrackingSequence.from_geff(tmp_path / "track.geff", spacing="auto")


def test_tracking_sequence_from_geff_derives_2d_axes_and_validates_override(
    monkeypatch, tmp_path
):
    graph = nx.DiGraph()
    graph.add_node(10, t=0, y=1, x=2)
    graph.add_node(20, t=1, y=3, x=4)
    graph.add_edge(10, 20)
    metadata = SimpleNamespace(
        axes=[
            SimpleNamespace(name="t", type="time", scale=1.0),
            SimpleNamespace(name="y", type="space", scale=2.0),
            SimpleNamespace(name="x", type="space", scale=3.0),
        ]
    )
    import geff

    monkeypatch.setattr(geff, "read", lambda _path: (graph, metadata))

    sequence = TrackingSequence.from_geff(
        tmp_path / "track.geff", spacing="auto", source_ndim="auto"
    )
    assert sequence.ndim == 2
    np.testing.assert_allclose(sequence.detections[0].coords, [[2, 6], [6, 12]])

    with pytest.raises(ValueError, match="source_ndim=3"):
        TrackingSequence.from_geff(tmp_path / "track.geff", source_ndim=3)
    with pytest.raises(ValueError, match="do not match metadata"):
        TrackingSequence.from_geff(tmp_path / "track.geff", coord_columns=("x", "y"))


def test_tracking_sequence_from_geff_matches_proposal_csv(monkeypatch, tmp_path):
    graph = nx.DiGraph()
    graph.add_node(10, t=0, z=1, y=0, x=0)
    graph.add_node(20, t=1, z=2, y=0, x=0)
    graph.add_edge(10, 20)
    metadata = SimpleNamespace(
        axes=[
            SimpleNamespace(name="t", scale=1.0),
            SimpleNamespace(name="z", scale=4.0),
            SimpleNamespace(name="y", scale=1.0),
            SimpleNamespace(name="x", scale=1.0),
        ]
    )

    import geff

    monkeypatch.setattr(geff, "read", lambda _path: (graph, metadata))
    proposals = tmp_path / "points.csv"
    proposals.write_text(
        "axis-0,axis-1,axis-2,axis-3,intensity\n"
        "0,1,0,0,0.1\n"
        "1,2,0,0,0.2\n"
        "1,20,0,0,0.3\n"
    )

    sequence = TrackingSequence.from_geff(
        tmp_path / "track.geff",
        detections=proposals,
        spacing="auto",
        match_max_distance=0.5,
    )

    seg = sequence.detections[0]
    np.testing.assert_allclose(
        seg.coords,
        np.array([[4, 0, 0], [8, 0, 0], [80, 0, 0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        seg.source_coords,
        np.array([[1, 0, 0], [2, 0, 0], [20, 0, 0]], dtype=np.float32),
    )
    assert seg.spacing == (4.0, 1.0, 1.0)
    assert seg.timepoints.tolist() == [0, 1, 1]
    assert seg.labels.tolist() == [1, 1, 2]
    sup = sequence.supervision[0]
    assert sup.lineage_index.tolist() == [0, 0, -1]
    assert sup.matched_gt.tolist() == [True, True, True]
    assert sup.gt_predecessor_set_available.tolist() == [True, True, True]
    assert sup.gt_successor_set_available.tolist() == [True, True, True]
    np.testing.assert_allclose(seg.features["intensity"].ravel(), [0.1, 0.2, 0.3])
    sample = TrackingDataset(sequence, window_size=2, features="intensity")[0]
    np.testing.assert_allclose(sample["features"].numpy(), [[0.1], [0.2], [0.3]])

    sparse = TrackingSequence.from_geff(
        tmp_path / "track.geff",
        detections=proposals,
        spacing="auto",
        match_max_distance=0.5,
        sparse_gt=True,
    )
    sparse_sup = sparse.supervision[0]
    assert sparse_sup.lineage_index.tolist() == [0, 0, -1]
    assert sparse_sup.matched_gt.tolist() == [True, True, False]
    assert sparse_sup.gt_predecessor_set_available.tolist() == [
        False,
        True,
        False,
    ]
    assert sparse_sup.gt_successor_set_available.tolist() == [
        True,
        False,
        False,
    ]


def _linear_geff_graph():
    graph = nx.DiGraph()
    graph.add_node(10, t=0, z=1, y=0, x=0)
    graph.add_node(20, t=1, z=2, y=0, x=0)
    graph.add_edge(10, 20)
    metadata = SimpleNamespace(
        axes=[
            SimpleNamespace(name="t", scale=1.0),
            SimpleNamespace(name="z", scale=4.0),
            SimpleNamespace(name="y", scale=1.0),
            SimpleNamespace(name="x", scale=1.0),
        ]
    )
    return graph, metadata


def test_tracking_sequence_from_geff_root_multiple_csvs(monkeypatch, tmp_path):
    graph, metadata = _linear_geff_graph()
    import geff

    monkeypatch.setattr(geff, "read", lambda _path: (graph, metadata))

    root = tmp_path / "movie"
    (root / "track.geff").mkdir(parents=True)
    (root / "a.csv").write_text("axis-0,axis-1,axis-2,axis-3\n0,1,0,0\n1,2,0,0\n")
    (root / "b.csv").write_text("axis-0,axis-1,axis-2,axis-3\n0,1,0,0\n")

    sequence = TrackingSequence.from_geff(root, spacing="auto", match_max_distance=0.5)

    # one DetectionSequence per csv, discovered by directory scan and sorted by name
    assert [d.name for d in sequence.detections] == ["a", "b"]
    a, b = sequence.detections
    # both index into the single shared GT tracklet
    assert sequence.gt.lineage_relation.tolist() == [[True]]
    assert sequence.gt.lineage_parents.tolist() == [-1]
    assert sequence.supervision[0].lineage_index.tolist() == [0, 0]
    assert sequence.supervision[1].lineage_index.tolist() == [0]


def test_tracking_sequence_from_geff_root_without_csv_is_gt_only(monkeypatch, tmp_path):
    graph, metadata = _linear_geff_graph()
    import geff

    monkeypatch.setattr(geff, "read", lambda _path: (graph, metadata))

    root = tmp_path / "movie"
    (root / "track.geff").mkdir(parents=True)

    sequence = TrackingSequence.from_geff(root, spacing="auto")

    # no proposals -> a single DetectionSequence built from the GT nodes
    assert len(sequence.detections) == 1
    gt = sequence.detections[0]
    assert sequence.supervision[0].lineage_index.tolist() == [0, 0]
    np.testing.assert_allclose(gt.coords, [[4, 0, 0], [8, 0, 0]])


def test_tracking_sequence_from_geff_root_requires_single_store(monkeypatch, tmp_path):
    root = tmp_path / "movie"
    (root / "a.geff").mkdir(parents=True)
    (root / "b.geff").mkdir()
    with pytest.raises(FileNotFoundError, match=r"exactly one \.geff store"):
        TrackingSequence.from_geff(root)


def test_detection_sequence_is_immutable_and_validates_alignment():
    seg, _sup = _detection_sequence([(0, (0, 1))])

    with pytest.raises(ValueError):
        seg.coords[0, 0] = 10
    for values in seg.features.values():
        with pytest.raises(ValueError):
            values[0] = 10
    with pytest.raises(ValueError, match="aligned"):
        DetectionSequence(
            name="TRA",
            n_frames=1,
            coords=np.zeros((2, 2)),
            labels=np.ones(1),
            timepoints=np.zeros(2),
            features={},
        )


def test_tracking_sequence_pickle_roundtrip_remains_immutable():
    seg, sup = _detection_sequence([(0, (0, 1))])
    sequence = TrackingSequence(
        root=Path("synthetic"),
        ndim=2,
        detections=(seg,),
        gt=_lineage_graph(np.eye(2, dtype=bool), np.full(2, -1)),
        supervision=(sup,),
    )

    restored = pickle.loads(pickle.dumps(sequence))

    with pytest.raises(ValueError):
        restored.detections[0].coords[0, 0] = 10
    for values in restored.detections[0].features.values():
        with pytest.raises(ValueError):
            values[0] = 10


def test_tracking_data_uses_lineage_relation_and_excludes_unmatched():
    seg, sup = _detection_sequence([(0, (0, -1)), (1, (1, 2))])
    sequence = TrackingSequence(
        root=Path("synthetic"),
        ndim=2,
        detections=(seg,),
        gt=_lineage_graph(
            np.array([[1, 1, 1], [1, 1, 0], [1, 0, 1]], dtype=bool),
            np.array([-1, 0, 0]),
        ),
        supervision=(sup,),
    )

    for mode, width in (("wrfeat2", 6), ("wrfeat2_no_intensity", 5)):
        data = TrackingDataset(sequence, window_size=2, features=mode)
        sample = data[0]
        association = sample["assoc_matrix"].bool().numpy()

        assert len(data) == 1
        assert data.n_objects == (4,)
        assert sample["features"].shape == (4, width)
        assert association[0, 2] and association[0, 3]
        assert not association[2, 3]
        assert not association[1].any()
        assert not association[:, 1].any()


def test_tracking_data_normalize_diameter_scales_window_geometry_only():
    features = {
        "equivalent_diameter_area": np.full((4, 1), 2, np.float32),
        "intensity": np.full((4, 1), 0.5, np.float32),
        "inertia_tensor": np.tile(np.eye(2, dtype=np.float32).ravel(), (4, 1)),
        "border_dist": np.full((4, 1), 3, np.float32),
    }
    coords = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
    seg = DetectionSequence(
        name="TRA",
        n_frames=2,
        coords=coords,
        labels=np.array([1, 2, 1, 2]),
        timepoints=np.array([0, 0, 1, 1], dtype=np.int64),
        features=features,
    )
    sequence = TrackingSequence(
        root=Path("synthetic"),
        ndim=2,
        detections=(seg,),
        gt=_lineage_graph(np.eye(2, dtype=bool), np.full(2, -1)),
        supervision=(DetectionSupervision(lineage_index=np.array([0, 1, 0, 1])),),
    )

    sample = TrackingDataset(
        sequence, window_size=2, features="wrfeat2", normalize_diameter=4
    )[0]

    np.testing.assert_allclose(sample["coords0"][:, 1:].numpy(), coords * 2)
    values = sample["features"].numpy()
    np.testing.assert_allclose(values[:, 0], np.log1p(4))
    np.testing.assert_allclose(values[:, 1], 0.5)
    np.testing.assert_allclose(values[:, 2], np.log1p(2 / np.pi))
    np.testing.assert_allclose(values[:, 3:5], 0)
    np.testing.assert_allclose(values[:, 5], np.log1p(6))
    np.testing.assert_allclose(seg.coords, coords)
    np.testing.assert_allclose(seg.features["equivalent_diameter_area"], 2)


def _write_ctc_fixture(base: Path, layout: str) -> Path:
    if layout == "standard":
        root = base / "Fluo-C2DL-Huh7" / "01"
        images = root
        tra = base / "Fluo-C2DL-Huh7" / "01_GT" / "TRA"
    else:
        root = base / "151031-03"
        images = root / "img"
        tra = root / "TRA"
    images.mkdir(parents=True)
    tra.mkdir(parents=True)
    masks = []
    for t in range(3):
        image = np.arange(64, dtype=np.uint8).reshape(8, 8) + t
        mask = np.zeros((8, 8), dtype=np.uint16)
        if t == 0:
            mask[2:4, 2:4] = 1
        else:
            mask[1:3, 1:3] = 2
            mask[5:7, 5:7] = 3
        imwrite(images / f"t{t:03d}.tif", image)
        imwrite(tra / f"man_track{t:03d}.tif", mask)
        masks.append(mask)
    (tra / "man_track.txt").write_text("1 0 0 0\n2 1 2 1\n3 1 2 1\n")
    return root


def _write_3d_ctc_fixture(base: Path, depth: int = 2) -> Path:
    root = base / "volume"
    images = root / "img"
    tra = root / "TRA"
    images.mkdir(parents=True)
    tra.mkdir(parents=True)
    for t in range(2):
        image = np.arange(depth * 64, dtype=np.uint16).reshape(depth, 8, 8) + t
        mask = np.zeros((depth, 8, 8), dtype=np.uint16)
        mask[:, 2:4, 2:4] = 1
        imwrite(images / f"t{t:03d}.tif", image)
        imwrite(tra / f"man_track{t:03d}.tif", mask)
    (tra / "man_track.txt").write_text("1 0 1 0\n")
    return root


@pytest.mark.parametrize("layout", ("standard", "simple"))
def test_tracking_sequence_from_ctc_supports_reference_layouts(tmp_path, layout):
    root = _write_ctc_fixture(tmp_path, layout)

    sequence = TrackingSequence.from_ctc(root, n_workers=1)
    data = TrackingDataset(sequence, window_size=2)
    sample = data[0]

    assert sequence.root == root
    assert len(sequence.detections) == 1
    seg = sequence.detections[0]
    assert seg.n_frames == 3
    counts = tuple(int((seg.timepoints == t).sum()) for t in range(seg.n_frames))
    assert counts == (1, 2, 2)
    assert np.array_equal(
        sequence.gt.lineage_relation,
        np.array([[1, 1, 1], [1, 1, 0], [1, 0, 1]], dtype=bool),
    )
    assert sequence.supervision[0].gt_node_index.tolist() == [0, 1, 2, 3, 4]
    assert sequence.gt.node_in_degree.tolist() == [0, 1, 1, 1, 1]
    assert sequence.gt.node_out_degree.tolist() == [2, 1, 1, 0, 0]
    assert sample["assoc_matrix"][0, 1]
    assert sample["assoc_matrix"][0, 2]
    assert not sample["assoc_matrix"][1, 2]


def test_tracking_sequence_from_ctc_auto_detects_2d_and_3d(tmp_path):
    root_2d = _write_ctc_fixture(tmp_path / "two_d", "simple")
    root_3d = _write_3d_ctc_fixture(tmp_path / "three_d")

    sequence_2d = TrackingSequence.from_ctc(root_2d, n_workers=1)
    sequence_3d = TrackingSequence.from_ctc(root_3d, n_workers=1)

    assert sequence_2d.ndim == 2
    assert sequence_3d.ndim == 3
    assert sequence_2d.detections[0].features["inertia_tensor"].shape[1] == 4
    assert sequence_3d.detections[0].features["inertia_tensor"].shape[1] == 9


def test_ctc_explicit_2d_squeezes_singleton_z_and_rejects_real_3d(tmp_path):
    singleton = _write_3d_ctc_fixture(tmp_path / "singleton", depth=1)
    volume = _write_3d_ctc_fixture(tmp_path / "volume", depth=2)

    sequence = TrackingSequence.from_ctc(singleton, ndim=2, n_workers=1)
    assert sequence.ndim == 2
    assert sequence.detections[0].features["inertia_tensor"].shape[1] == 4

    with pytest.raises(ValueError, match="Expected 2D .* got shape"):
        TrackingSequence.from_ctc(volume, ndim=2, n_workers=1)


def test_detection_sequence_from_masks_normalizes_each_frame():
    images = np.stack(
        [
            np.arange(64, dtype=np.float32).reshape(8, 8),
            1000 + 10 * np.arange(64, dtype=np.float32).reshape(8, 8),
        ]
    )
    masks = np.zeros((2, 8, 8), dtype=np.int32)
    masks[:, 2:6, 2:6] = 1

    detections = DetectionSequence.from_masks(
        images, masks, normalize_imgs=True, keep_images=True
    )

    expected = np.stack([normalize(image) for image in images])
    np.testing.assert_allclose(detections.images, expected)


def test_tracking_sequence_from_ctc_spacing_scales_coords_and_geometry(tmp_path):
    root = _write_ctc_fixture(tmp_path, "simple")

    unit = TrackingSequence.from_ctc(root, n_workers=1)
    spaced = TrackingSequence.from_ctc(root, n_workers=1, spacing=(2, 2))

    unit_seg = unit.detections[0]
    spaced_seg = spaced.detections[0]
    assert spaced_seg.spacing == (2.0, 2.0)
    np.testing.assert_allclose(spaced_seg.source_coords, unit_seg.coords)
    np.testing.assert_allclose(spaced_seg.coords, unit_seg.coords * 2)
    np.testing.assert_allclose(
        spaced_seg.features["equivalent_diameter_area"],
        unit_seg.features["equivalent_diameter_area"] * 2,
    )
    np.testing.assert_allclose(
        spaced_seg.features["inertia_tensor"],
        unit_seg.features["inertia_tensor"] * 4,
    )
    np.testing.assert_allclose(
        spaced_seg.features["border_dist"],
        unit_seg.features["border_dist"],
    )
    np.testing.assert_allclose(
        spaced_seg.features["intensity"],
        unit_seg.features["intensity"],
    )


def test_tracking_sequence_from_ctc_anisotropic_spacing_transforms_geometry(tmp_path):
    root = _write_ctc_fixture(tmp_path, "simple")

    unit = TrackingSequence.from_ctc(root, n_workers=1)
    spaced = TrackingSequence.from_ctc(root, n_workers=1, spacing=(2, 1))

    unit_seg = unit.detections[0]
    spaced_seg = spaced.detections[0]
    assert spaced_seg.spacing == (2.0, 1.0)
    np.testing.assert_allclose(
        spaced_seg.coords,
        unit_seg.coords * np.array([2, 1], dtype=np.float32),
    )
    np.testing.assert_allclose(
        spaced_seg.features["equivalent_diameter_area"],
        unit_seg.features["equivalent_diameter_area"] * np.sqrt(2),
    )
    assert not np.allclose(
        spaced_seg.features["inertia_tensor"],
        unit_seg.features["inertia_tensor"],
    )
    np.testing.assert_allclose(
        spaced_seg.features["border_dist"],
        unit_seg.features["border_dist"],
    )
    np.testing.assert_allclose(
        spaced_seg.features["intensity"],
        unit_seg.features["intensity"],
    )


def test_tracking_sequence_from_ctc_spacing_scales_match_distance(tmp_path):
    root = _write_ctc_fixture(tmp_path, "simple")
    res = root / "RES"
    res.mkdir()
    for f in sorted((root / "TRA").glob("*.tif")):
        mask = imread(f)
        shifted = np.zeros_like(mask)
        shifted[:, 2:] = mask[:, :-2]
        imwrite(res / f.name, shifted)

    unit = TrackingSequence.from_ctc(
        root,
        detection_folders=("RES",),
        match_threshold=0.5,
        match_max_distance=5,
        n_workers=1,
    )
    spaced = TrackingSequence.from_ctc(
        root,
        detection_folders=("RES",),
        match_threshold=0.5,
        match_max_distance=5,
        n_workers=1,
        spacing=(2, 2),
    )

    assert np.all(unit.supervision[0].lineage_index >= 0)
    assert np.all(spaced.supervision[0].lineage_index == -1)


def test_tracking_sequence_from_ctc_rejects_spacing_with_spatial_downscale(tmp_path):
    root = _write_ctc_fixture(tmp_path, "simple")

    with pytest.raises(ValueError, match="downscale_spatial=1"):
        TrackingSequence.from_ctc(
            root, n_workers=1, spacing=(1, 1), downscale_spatial=2
        )


def test_from_ctc_tolerates_partial_detection_folders(tmp_path):
    # simple layout writes masks under root/TRA; mirror them into root/RES
    root = _write_ctc_fixture(tmp_path, "simple")
    res = root / "RES"
    res.mkdir()
    for f in sorted((root / "TRA").glob("*.tif")):
        imwrite(res / f.name, imread(f))

    # a requested folder that is missing is skipped, the available one still loads
    only_tra = _write_ctc_fixture(tmp_path / "tra_only", "simple")
    seq = TrackingSequence.from_ctc(
        only_tra, detection_folders=("TRA", "RES"), n_workers=1
    )
    assert [s.name for s in seq.detections] == ["TRA"]

    # when both are present, each contributes its own segmentation (~twice the windows)
    seq_both = TrackingSequence.from_ctc(
        root, detection_folders=("TRA", "RES"), n_workers=1
    )
    assert sorted(s.name for s in seq_both.detections) == ["RES", "TRA"]

    # none of the requested folders present -> fatal
    with pytest.raises(FileNotFoundError):
        TrackingSequence.from_ctc(only_tra, detection_folders=("RES",), n_workers=1)


def test_load_ctc_images_masks_refines_tra_with_st(tmp_path):
    sequence = tmp_path / "dataset" / "01"
    gt = tmp_path / "dataset" / "01_GT" / "TRA"
    st = tmp_path / "dataset" / "01_ST" / "SEG"
    sequence.mkdir(parents=True)
    gt.mkdir(parents=True)
    st.mkdir(parents=True)

    image = np.arange(100, dtype=np.uint8).reshape(10, 10)
    tra = np.zeros((10, 10), dtype=np.uint16)
    tra[4, 4] = 1
    silver = np.zeros_like(tra)
    silver[3:6, 3:6] = 1
    imwrite(sequence / "t000.tif", image)
    imwrite(gt / "man_track000.tif", tra)
    (gt / "man_track.txt").write_text("1 0 0 0\n")
    imwrite(st / "man_seg000.tif", silver)

    imgs, masks, image_path, gt_path = load_ctc_images_masks(sequence, "TRA")

    np.testing.assert_allclose(imgs[0], normalize(image))
    np.testing.assert_array_equal(masks[0], np.maximum(tra, silver))
    assert image_path == sequence
    assert gt_path == gt

    _, seg_masks, seg_image_path, seg_gt_path = load_ctc_images_masks(sequence, "SEG")
    np.testing.assert_array_equal(seg_masks[0], silver)
    assert seg_image_path == sequence
    assert seg_gt_path == gt

    loaded = TrackingSequence.from_ctc(
        sequence, detection_folders=("SEG",), n_workers=1
    )
    assert loaded.detections[0].name == "SEG"
    assert loaded.detections[0].labels.tolist() == [1]


def test_load_images_flag_attaches_arrays_and_survives_pickle(tmp_path):
    root = _write_ctc_fixture(tmp_path, "simple")

    bare = TrackingSequence.from_ctc(root, n_workers=1)
    assert bare.images is None
    assert bare.detections[0].masks is None

    seq = TrackingSequence.from_ctc(root, n_workers=1, load_images=True)
    assert seq.images is not None and len(seq.images) == 3
    assert seq.detections[0].masks is not None
    assert len(seq.detections[0].masks) == 3
    assert not seq.images.flags.writeable

    restored = pickle.loads(pickle.dumps(seq))
    np.testing.assert_array_equal(restored.images, seq.images)
    np.testing.assert_array_equal(restored.detections[0].masks, seq.detections[0].masks)


def test_tracking_sequence_without_gt_uses_isolated_detections(tmp_path):
    root = tmp_path / "detections-only"
    images = root / "img"
    masks = root / "masks"
    images.mkdir(parents=True)
    masks.mkdir()
    for t in range(2):
        image = np.arange(64, dtype=np.uint8).reshape(8, 8) + t
        mask = np.zeros((8, 8), dtype=np.uint16)
        mask[2:4, 2:4] = 1
        imwrite(images / f"t{t:03d}.tif", image)
        imwrite(masks / f"mask{t:03d}.tif", mask)

    sequence = TrackingSequence.from_ctc(
        root, use_gt=False, detection_folders=("masks",), n_workers=1
    )
    sample = TrackingDataset(sequence, window_size=2)[0]

    assert sequence.gt.lineage_relation.shape == (2, 2)
    assert np.array_equal(sequence.gt.lineage_relation, np.eye(2, dtype=bool))
    assert torch.equal(sample["assoc_matrix"], torch.eye(2))


def test_tracking_sequence_joblib_cache_ignores_workers(tmp_path):
    root = _write_ctc_fixture(tmp_path / "data", "simple")
    memory = joblib.Memory(tmp_path / "cache", verbose=0)
    loader = memory.cache(TrackingSequence.from_ctc, ignore=["n_workers"])

    sequence = loader(root=root, n_workers=1)

    assert loader.check_call_in_cache(root=root, n_workers=4)
    cached = loader(root=root, n_workers=4)
    assert not cached.gt.lineage_relation.flags.writeable
    assert not cached.detections[0].coords.flags.writeable
    assert np.array_equal(sequence.gt.lineage_relation, cached.gt.lineage_relation)


def test_tracking_data_runtime_selection_keeps_complete_lineages():
    seg, sup = _detection_sequence([(0, (0, 1, 2)), (1, (0, 1, 2))])
    sequence = TrackingSequence(
        root=Path("synthetic"),
        ndim=2,
        detections=(seg,),
        gt=_lineage_graph(np.eye(3, dtype=bool), np.full(3, -1)),
        supervision=(sup,),
    )

    np.random.seed(0)
    capped = TrackingDataset(sequence, window_size=2, max_detections=2)[0]
    np.random.seed(0)
    dropped = TrackingDataset(
        sequence,
        window_size=2,
        detect_drop=1,
        detect_drop_fraction=0.34,
    )[0]

    assert len(capped["coords"]) == 4
    assert len(dropped["coords"]) == 4
    for sample in (capped, dropped):
        labels, counts = torch.unique(sample["labels"], return_counts=True)
        assert len(labels) == 2
        assert torch.equal(counts, torch.full((2,), 2))


@pytest.mark.parametrize("level", (1, 2, 3, 4))
def test_tracking_data_augmentation_does_not_mutate_sequence(level):
    seg, sup = _detection_sequence([(0, (0, 1)), (1, (0, 1))])
    sequence = TrackingSequence(
        root=Path("synthetic"),
        ndim=2,
        detections=(seg,),
        gt=_lineage_graph(np.eye(2, dtype=bool), np.full(2, -1)),
        supervision=(sup,),
    )
    original = seg.features["inertia_tensor"].copy()

    np.random.seed(4)
    torch.manual_seed(4)
    sample = TrackingDataset(
        sequence, window_size=2, features="wrfeat2", augment=level
    )[0]

    assert sample["features"].shape == (4, 6)
    np.testing.assert_array_equal(seg.features["inertia_tensor"], original)


def test_tracking_data_position_noise_is_bounded_global_offset():
    seg, sup = _detection_sequence([(0, (0, 1)), (1, (0, 1))])
    sequence = TrackingSequence(
        root=Path("synthetic"),
        ndim=2,
        detections=(seg,),
        gt=_lineage_graph(np.eye(2, dtype=bool), np.full(2, -1)),
        supervision=(sup,),
    )

    torch.manual_seed(0)
    sample = TrackingDataset(sequence, window_size=2, augment=1, position_noise=7.0)[0]
    offset = sample["coords"][:, 1:] - sample["coords0"][:, 1:]

    assert torch.allclose(offset, offset[:1].expand_as(offset))
    assert offset.abs().max() <= 7.0
