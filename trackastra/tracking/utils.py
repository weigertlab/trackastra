import logging
from collections import deque
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import tifffile
from skimage.measure import regionprops
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class FoundTracks(Exception):
    pass


def ctc_to_napari_tracks(segmentation: np.ndarray, man_track: pd.DataFrame):
    """Convert tracks in CTC format to tracks in napari format.

    Args:
        segmentation: Dims time, spatial_0, ... , spatial_n
        man_track: columns id, start, end, parent
    """
    tracks = []
    for t, frame in tqdm(
        enumerate(segmentation),
        total=len(segmentation),
        leave=False,
        desc="Computing centroids",
    ):
        for r in regionprops(frame):
            tracks.append((r.label, t, *r.centroid))

    tracks_graph = {}
    for idx, _, _, parent in tqdm(
        man_track.to_numpy(),
        desc="Converting CTC to napari tracks",
        leave=False,
    ):
        if parent != 0:
            tracks_graph[idx] = [parent]

    return tracks, tracks_graph


class CtcTracklet:
    def __init__(self, parent: int, nodes: list[int], start_frame: int) -> None:
        self.parent = parent
        self.nodes = nodes
        self.start_frame = start_frame

    def __lt__(self, other):
        if self.start_frame < other.start_frame:
            return True
        if self.start_frame > other.start_frame:
            return False
        if self.start_frame == other.start_frame:
            return self.parent < other.parent

    def __str__(self) -> str:
        return f"Tracklet(parent={self.parent}, nodes={self.nodes})"

    def __repr__(self) -> str:
        return str(self)


def ctc_tracklets(G: nx.DiGraph, frame_attribute: str = "time") -> list[CtcTracklet]:
    """Return all CTC tracklets in a graph, i.e.

    - first node after
        - a division (out_degree of parent = 2)
        - an appearance (in_degree=0)
        - a gap closing event (delta_t to parent node > 1)
    - inner nodes have in_degree=1 and out_degree=1, delta_t=1
    - last node:
        - before a division (out_degree = 2)
        - before a disappearance (out_degree = 0)
        - before a gap closing event (delta_t to next node > 1)
    """
    tracklets = []
    # get all nodes with out_degree == 2 (i.e. parent of a tracklet)

    # Queue of tuples(parent id, start node id)
    starts = deque()
    starts.extend(
        [(p, d) for p in G.nodes for d in G.successors(p) if G.out_degree[p] == 2]
    )
    # set parent = -1 since there is no parent
    starts.extend([(-1, n) for n in G.nodes if G.in_degree[n] == 0])
    while starts:
        _p, _s = starts.popleft()
        nodes = [_s]
        # build a tracklet
        c = _s
        while True:
            if G.out_degree[c] > 2:
                raise ValueError("More than two daughters!")
            if G.out_degree[c] == 2:
                break
            if G.out_degree[c] == 0:
                break
            t_c = G.nodes[c][frame_attribute]
            suc = next(iter(G.successors(c)))
            t_suc = G.nodes[suc][frame_attribute]
            if t_suc - t_c > 1:
                logger.debug(
                    f"Gap closing edge from `{c} (t={t_c})` to `{suc} (t={t_suc})`"
                )
                starts.append((c, suc))
                break
            # Add node to tracklet
            c = next(iter(G.successors(c)))
            nodes.append(c)

        tracklets.append(
            CtcTracklet(
                parent=_p, nodes=nodes, start_frame=G.nodes[_s][frame_attribute]
            )
        )

    return tracklets


def linear_chains(G: nx.DiGraph):
    """Find all linear chains in a tree/graph, i.e. paths that.

    i) either start/end at a node with out_degree>in_degree or and have no internal branches, or
    ii) consists of a single node

    Note that each chain includes its start/end node, i.e. they can be appear in multiple chains.
    """
    # get all nodes with out_degree>in_degree (i.e. start of chain)
    nodes = tuple(n for n in G.nodes if G.out_degree[n] > G.in_degree[n])
    single_nodes = tuple(n for n in G.nodes if G.out_degree[n] == G.in_degree[n] == 0)

    for ni in single_nodes:
        yield [ni]

    for ni in nodes:
        neighs = tuple(G.neighbors(ni))
        for child in neighs:
            path = [ni, child]
            while len(childs := tuple(G.neighbors(path[-1]))) == 1:
                path.append(childs[0])
            yield path


def graph_to_napari_tracks(
    graph: nx.DiGraph,
    properties: list[str] = [],
):
    """Convert a track graph to napari tracks."""
    # each tracklet is a linear chain in the graph
    chains = tuple(linear_chains(graph))

    track_end_to_track_id = dict()
    labels = []
    for i, cs in enumerate(chains):
        label = i + 1
        labels.append(label)
        if len(cs) == 1:
            # Non-connected node
            continue
        end = cs[-1]
        track_end_to_track_id[end] = label

    tracks = []
    tracks_graph = dict()
    tracks_props = {p: [] for p in properties}

    for label, cs in tqdm(zip(labels, chains), total=len(chains)):
        start = cs[0]
        if start in track_end_to_track_id:
            tracks_graph[label] = track_end_to_track_id[start]
            nodes = cs[1:]
        else:
            nodes = cs

        for c in nodes:
            node = graph.nodes[c]
            t = node["time"]
            coord = node["coords"]
            tracks.append([label, t, *list(coord)])

            for p in properties:
                tracks_props[p].append(node[p])

    tracks = np.array(tracks)
    return tracks, tracks_graph, tracks_props


def _check_ctc_df(df: pd.DataFrame, masks: np.ndarray):
    """Sanity check of all labels in a CTC dataframe are present in the masks."""
    # Check for empty df
    if len(df) == 0 and np.all(masks == 0):
        return True

    for t in range(df.t1.min(), df.t1.max()):
        sub = df[(df.t1 <= t) & (df.t2 >= t)]
        sub_lab = set(sub.label)
        # Since we have non-negative integer labels, we can np.bincount instead of np.unique for speedup
        masks_lab = set(np.where(np.bincount(masks[t].ravel()))[0]) - {0}
        if not sub_lab.issubset(masks_lab):
            print(f"Missing labels in masks at t={t}: {sub_lab - masks_lab}")
            return False
    return True


def graph_to_edge_table(
    graph: nx.DiGraph,
    frame_attribute: str = "time",
    edge_attribute: str = "weight",
    outpath: Path | None = None,
) -> pd.DataFrame:
    """Write edges of a graph to a table.

    The table has columns `source_frame`, `source_label`, `target_frame`, `target_label`, and `weight`.
    The first line is a header. The source and target are the labels of the objects in the
    input masks in the designated frames (0-indexed).

    Args:
        graph: With node attributes `frame_attribute`, `edge_attribute` and 'label'.
        frame_attribute: Name of the frame attribute 'graph`.
        edge_attribute: Name of the score attribute in `graph`.
        outpath: If given, save the edges in CSV file format.

    Returns:
        pd.DataFrame: Edges DataFrame with columns ['source_frame', 'source', 'target_frame', 'target', 'weight']
    """
    rows = []
    for edge in graph.edges:
        source = graph.nodes[edge[0]]
        target = graph.nodes[edge[1]]

        source_label = int(source["label"])
        source_frame = int(source[frame_attribute])
        target_label = int(target["label"])
        target_frame = int(target[frame_attribute])
        weight = float(graph.edges[edge][edge_attribute])

        rows.append([source_frame, source_label, target_frame, target_label, weight])

    df = pd.DataFrame(
        rows,
        columns=[
            "source_frame",
            "source_label",
            "target_frame",
            "target_label",
            "weight",
        ],
    )
    df = df.sort_values(
        by=["source_frame", "source_label", "target_frame", "target_label"],
        ascending=True,
    )

    if outpath is not None:
        outpath = Path(outpath)
        outpath.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        df.to_csv(outpath, index=False, header=True, sep=",")

    return df


def graph_to_ctc(
    graph: nx.DiGraph,
    masks_original: np.ndarray,
    check: bool = True,
    frame_attribute: str = "time",
    outdir: Path | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Convert graph to ctc track Dataframe and relabeled masks.

    Args:
        graph: with node attributes `frame_attribute` and "label"
        masks_original: list of masks with unique labels
        check: Check CTC format
        frame_attribute: Name of the frame attribute in the graph nodes.
        outdir: path to save results in CTC format.

    Returns:
        pd.DataFrame: track dataframe with columns ['track_id', 't_start', 't_end', 'parent_id']
        np.ndarray: masks with unique color for each track
    """
    # each tracklet is a linear chain in the graph
    tracklets = ctc_tracklets(graph, frame_attribute=frame_attribute)

    regions = tuple(
        dict((reg.label, reg.slice) for reg in regionprops(m))
        for t, m in enumerate(masks_original)
    )

    masks = np.stack([np.zeros_like(m) for m in masks_original])
    rows = []
    # To map parent references to tracklet ids. -1 means no parent, which is mapped to 0 in CTC format.
    node_to_tracklets = dict({-1: 0})

    # Sort tracklets by parent id
    for i, _tracklet in tqdm(
        enumerate(sorted(tracklets)),
        total=len(tracklets),
        desc="Converting graph to CTC results",
    ):
        _parent = _tracklet.parent
        _nodes = _tracklet.nodes
        label = i + 1

        _start, end = _nodes[0], _nodes[-1]

        t1 = _tracklet.start_frame
        # t1 = graph.nodes[start][frame_attribute]
        t2 = graph.nodes[end][frame_attribute]

        node_to_tracklets[end] = label

        # relabel masks
        for _n in _nodes:
            node = graph.nodes[_n]
            t = node[frame_attribute]
            lab = node["label"]
            ss = regions[t][lab]
            m = masks_original[t][ss] == lab
            if masks[t][ss][m].max() > 0:
                raise RuntimeError(f"Overlapping masks at t={t}, label={lab}")
            if np.count_nonzero(m) == 0:
                raise RuntimeError(f"Empty mask at t={t}, label={lab}")
            masks[t][ss][m] = label

        rows.append([label, t1, t2, node_to_tracklets[_parent]])

    df = pd.DataFrame(rows, columns=["label", "t1", "t2", "parent"], dtype=int)

    masks = np.stack(masks)

    if check:
        _check_ctc_df(df, masks)

    if outdir is not None:
        outdir = Path(outdir)
        outdir.mkdir(
            # mode=775,
            parents=True,
            exist_ok=True,
        )
        df.to_csv(outdir / "man_track.txt", index=False, header=False, sep=" ")
        for i, m in tqdm(enumerate(masks), total=len(masks), desc="Saving masks"):
            tifffile.imwrite(
                outdir / f"man_track{i:04d}.tif",
                m,
                compression="zstd",
            )

    return df, masks
