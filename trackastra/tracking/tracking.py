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
    solution_graph.add_nodes_from(candidate_graph.nodes(data=True))

    # Pull edges once, in native (insertion) order, and drive the greedy loop
    # over numpy-sorted weights + plain degree counters instead of repeatedly
    # querying networkx edge/degree views (which dominate the solver cost).
    edge_list = list(candidate_graph.edges(data=True))
    if edge_list:
        weights = np.fromiter(
            (features[edge_attr] for _, _, features in edge_list),
            dtype=float,
            count=len(edge_list),
        )
        assert weights.max() <= 1.0, (
            "Edge weights are assumed to be normalized to [0,1]"
        )
        # descending by weight; stable to match sorted(..., reverse=True) ties
        order = np.argsort(-weights, kind="stable")

        max_out_degree = 2 if allow_divisions else 1
        in_degree: dict = {}
        out_degree: dict = {}
        selected = []
        for idx in tqdm(order.tolist(), desc="Greedily matched edges"):
            # assumes sorted edges: all remaining weights are below threshold
            if weights[idx] < threshold:
                break
            node_in, node_out, features = edge_list[idx]
            # no fusing: target already has an incoming edge
            if in_degree.get(node_out, 0) > 0:
                continue
            # parent already has max number of outgoing edges
            if out_degree.get(node_in, 0) >= max_out_degree:
                continue
            in_degree[node_out] = in_degree.get(node_out, 0) + 1
            out_degree[node_in] = out_degree.get(node_in, 0) + 1
            selected.append((node_in, node_out, features))

        solution_graph.add_edges_from(selected)

    return solution_graph
    # TODO this should all be in a tracker class
    # return df, masks, solution_graph, tracks_graph, tracks, candidate_graph


def build_graph(
    nodes: dict,
    weights: tuple | None = None,
    use_distance: bool = False,
    spatial_cutoff: int | None = None,
    max_neighbors: int | None = None,
    delta_t=1,
) -> nx.DiGraph:
    logger.info(f"Build candidate graph with {delta_t=}")
    G = nx.DiGraph()

    if len(nodes) == 0:
        logger.warning("No nodes provided, returning empty graph")
        return G

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
    if weights is not None:
        weights = {w[0]: w[1] for w in weights}

    graph = TrackGraph(G, frame_attribute="time")
    frame_pairs = zip(
        chain(*[
            list(range(graph.t_begin, graph.t_end - d)) for d in range(1, delta_t + 1)
        ]),
        chain(*[
            list(range(graph.t_begin + d, graph.t_end)) for d in range(1, delta_t + 1)
        ]),
    )
    iterator = tqdm(
        frame_pairs,
        total=(graph.t_end - graph.t_begin) * delta_t,
        leave=False,
    )
    for t_begin, t_end in iterator:
        n_edges_t = len(G.edges)
        ni, nj = graph.nodes_by_frame(t_begin), graph.nodes_by_frame(t_end)

        if len(ni) == 0:
            # skip edge creation for empty frames
            logger.warning(f"No nodes in frame {t_begin}")
            continue

        if len(nj) == 0:
            # skip edge creation for empty frames
            logger.warning(f"No nodes in frame {t_end}")
            continue

        pi = np.array([G.nodes[_ni]["coords"] for _ni in ni])
        pj = np.array([G.nodes[_nj]["coords"] for _nj in nj])
        nj_arr = np.asarray(nj)

        dists = scipy.spatial.distance.cdist(pi, pj)

        for _i, _ni in enumerate(ni):
            row = dists[_i]
            order = np.argsort(row)
            row_sorted = row[order]
            if spatial_cutoff is not None:
                # sorted nearest-first; drop everything beyond the cutoff
                keep = int(np.searchsorted(row_sorted, spatial_cutoff, side="right"))
                order = order[:keep]
                row_sorted = row_sorted[:keep]
            nj_sorted = nj_arr[order].tolist()

            if weights is None:
                if max_neighbors:
                    nj_sorted = nj_sorted[:max_neighbors]
                    row_sorted = row_sorted[:max_neighbors]
                for _nj, dist in zip(nj_sorted, row_sorted):
                    G.add_edge(_ni, _nj, weight=1 - dist / spatial_cutoff)
            else:
                neighbors = 0
                for _nj in nj_sorted:
                    if max_neighbors and neighbors >= max_neighbors:
                        break
                    w = weights.get((_ni, _nj))
                    if w is not None:
                        G.add_edge(_ni, _nj, weight=w)
                        neighbors += 1

        e_added = len(G.edges) - n_edges_t
        if e_added == 0:
            logger.warning(f"No candidate edges in frame {t_begin}")
        iterator.set_description(
            f"{e_added} edges in frame {t_begin}  Total edges: {len(G.edges)}"
        )

    logger.info(f"Added {len(G.nodes)} vertices, {len(G.edges)} edges")

    return G
