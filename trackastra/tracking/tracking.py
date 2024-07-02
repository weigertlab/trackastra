import logging
from itertools import chain

import networkx as nx
import numpy as np
import scipy
from tqdm import tqdm

from .track_graph import TrackGraph

# from trackastra.tracking import graph_to_napari_tracks, graph_to_ctc

logger = logging.getLogger(__name__)


def copy_edge(edge: tuple, source: nx.DiGraph, target: nx.DiGraph):
    if edge[0] not in target.nodes:
        target.add_node(edge[0], **source.nodes[edge[0]])
    if edge[1] not in target.nodes:
        target.add_node(edge[1], **source.nodes[edge[1]])
    target.add_edge(edge[0], edge[1], **source.edges[(edge[0], edge[1])])


def track_greedy(
    candidate_graph,
    allow_divisions=True,
    threshold=0.5,
    edge_attr="weight",
):
    """Greedy matching, global.

    Iterates over global edges sorted by weight, and keeps edge if feasible and weight above threshold.

    Args:
       allow_divisions (bool, optional):
            Whether to model divisions. Defaults to True.

    Returns:
        solution_graph: NetworkX graph of tracks
    """
    logger.info("Running greedy tracker")

    solution_graph = nx.DiGraph()

    # TODO bring back
    # if args.gt_as_dets:
    # solution_graph.add_nodes_from(candidate_graph.nodes(data=True))

    edges = candidate_graph.edges(data=True)
    edges = sorted(
        edges,
        key=lambda edge: edge[2][edge_attr],
        reverse=True,
    )

    for edge in tqdm(edges, desc="Greedily matched edges"):
        node_in, node_out, features = edge
        assert (
            features[edge_attr] <= 1.0
        ), "Edge weights are assumed to be normalized to [0,1]"
        # assumes sorted edges
        if features[edge_attr] < threshold:
            break
        # Check whether this edge is a feasible edge to add
        # i.e. no fusing
        if node_out in solution_graph.nodes and solution_graph.in_degree(node_out) > 0:
            # target node already has an incoming edge
            continue
        if node_in in solution_graph and solution_graph.out_degree(node_in) >= (
            2 if allow_divisions else 1
        ):
            # parent node already has max number of outgoing edges
            continue
        # otherwise add to solution
        copy_edge(edge, candidate_graph, solution_graph)

    # df, masks = graph_to_ctc(solution_graph, masks_original)
    # tracks, tracks_graph, _ = graph_to_napari_tracks(solution_graph)

    return solution_graph
    # TODO this should all be in a tracker class
    # return df, masks, solution_graph, tracks_graph, tracks, candidate_graph


def build_graph(
    nodes: dict,
    weights: tuple | None = None,
    use_distance: bool = False,
    max_distance: int | None = None,
    max_neighbors: int | None = None,
    delta_t=1,
):
    logger.info(f"Build candidate graph with {delta_t=}")
    G = nx.DiGraph()

    for node in nodes:
        G.add_node(
            node["id"],
            time=node["time"],
            label=node["label"],
            coords=node["coords"],
            # index=node["index"],
            weight=1,
        )

    if use_distance:
        weights = None
    if weights:
        weights = {w[0]: w[1] for w in weights}

    graph = TrackGraph(G, frame_attribute="time")
    frame_pairs = zip(
        chain(
            *[
                list(range(graph.t_begin, graph.t_end - d))
                for d in range(1, delta_t + 1)
            ]
        ),
        chain(
            *[
                list(range(graph.t_begin + d, graph.t_end))
                for d in range(1, delta_t + 1)
            ]
        ),
    )
    iterator = tqdm(
        frame_pairs,
        total=(graph.t_end - graph.t_begin) * delta_t,
        leave=False,
    )
    for t_begin, t_end in iterator:
        n_edges_t = len(G.edges)
        ni, nj = graph.nodes_by_frame(t_begin), graph.nodes_by_frame(t_end)
        pi = []
        for _ni in ni:
            pi.append(np.array(G.nodes[_ni]["coords"]))
        pi = np.stack(pi)
        pj = []
        for _nj in nj:
            pj.append(np.array(G.nodes[_nj]["coords"]))
        pj = np.stack(pj)

        dists = scipy.spatial.distance.cdist(pi, pj)

        for _i, _ni in enumerate(ni):
            inds = np.argsort(dists[_i])
            neighbors = 0
            for _j, _nj in zip(inds, np.array(nj)[inds]):
                if max_neighbors and neighbors >= max_neighbors:
                    break
                dist = dists[_i, _j]
                if max_distance is None or dist <= max_distance:
                    if weights is None:
                        G.add_edge(_ni, _nj, weight=1 - dist / max_distance)
                        neighbors += 1
                    else:
                        if (_ni, _nj) in weights:
                            G.add_edge(_ni, _nj, weight=weights[(_ni, _nj)])
                            neighbors += 1

        e_added = len(G.edges) - n_edges_t
        if e_added == 0:
            logger.warning(f"No candidate edges in frame {t_begin}")
        iterator.set_description(
            f"{e_added} edges in frame {t_begin}  Total edges: {len(G.edges)}"
        )

    logger.info(f"Added {len(G.nodes)} vertices, {len(G.edges)} edges")

    return G
