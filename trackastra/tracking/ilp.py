import logging
import sys
import time
from types import SimpleNamespace

import networkx as nx
import yaml

try:
    import motile

    if sys.version_info >= (3, 12):
        raise ImportError("The optional dependency 'motile' requires Python <3.12.")
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "For tracking with an ILP, please conda install the optional `motile`"
        " dependency following https://funkelab.github.io/motile/install.html."
    )


logger = logging.getLogger(__name__)

ILP_CONFIGS = {
    "gt": SimpleNamespace(
        nodeW=0,
        nodeC=-10,  # take all nodes
        edgeW=-1,
        edgeC=0,
        appearC=0.25,
        disappearC=0.5,
        splitC=0.25,
    ),
    "deepcell_gt": SimpleNamespace(
        nodeW=0,
        nodeC=-10,  # take all nodes
        edgeW=-1,
        edgeC=0,
        appearC=0.25,
        disappearC=0.5,
        splitC=1,
    ),
    "deepcell_gt_tuned": SimpleNamespace(
        nodeW=0,
        nodeC=-10,  # take all nodes
        edgeW=-1,
        edgeC=0,
        appearC=0.5,
        disappearC=0.5,
        splitC=1,
    ),
    "deepcell_res_tuned": SimpleNamespace(
        nodeW=0,
        nodeC=0.25,
        edgeW=-1,
        edgeC=-0.25,
        appearC=0.25,
        disappearC=0.25,
        splitC=1.0,
    ),
}


def track_ilp(
    candidate_graph,
    allow_divisions: bool = True,
    ilp_config: str = "gt",
    params_file: str | None = None,
    **kwargs,
):
    candidate_graph_motile = motile.TrackGraph(candidate_graph, frame_attribute="time")

    ilp, _used_costs = solve_full_ilp(
        candidate_graph_motile,
        allow_divisions=allow_divisions,
        mode=ilp_config,
        params_file=params_file,
    )
    print_solution_stats(ilp, candidate_graph_motile)

    graph = solution_to_graph(ilp, candidate_graph_motile)

    return graph


def solve_full_ilp(
    graph,
    allow_divisions: bool,
    mode: str,
    params_file: str | None,
):
    solver = motile.Solver(graph)
    if params_file:
        with open(params_file) as f:
            p = yaml.safe_load(f)
        # TODO more checks
        p = SimpleNamespace(**p)
        logger.info(f"Using ILP parameters {p}")
    else:
        try:
            p = ILP_CONFIGS[mode]
            logger.info(f"Using `{mode}` ILP config.")
        except KeyError:
            raise ValueError(
                f"Unknown ILP mode {mode}. Choose from {list(ILP_CONFIGS.keys())} or"
                " supply custom parameters via `params_file` argument."
            )

    # Add costs
    used_costs = SimpleNamespace()

    # NODES
    solver.add_cost(
        motile.costs.NodeSelection(weight=p.nodeW, constant=p.nodeC, attribute="weight")
    )
    used_costs.nodeW = p.nodeW
    used_costs.nodeC = p.nodeC

    # EDGES
    solver.add_cost(
        motile.costs.EdgeSelection(weight=p.edgeW, constant=p.edgeC, attribute="weight")
    )
    used_costs.edgeW = p.edgeW
    used_costs.edgeC = p.edgeC

    # APPEAR
    solver.add_cost(motile.costs.Appear(constant=p.appearC))
    used_costs.appearC = p.appearC

    # DISAPPEAR
    solver.add_cost(motile.costs.Disappear(constant=p.disappearC))
    used_costs.disappearC = p.disappearC

    # DIVISION
    if allow_divisions:
        solver.add_cost(motile.costs.Split(constant=p.splitC))
        used_costs.splitC = p.splitC

    # Add constraints
    solver.add_constraint(motile.constraints.MaxParents(1))
    solver.add_constraint(motile.constraints.MaxChildren(2 if allow_divisions else 1))

    solver.solve()

    return solver, vars(used_costs)


def solution_to_graph(solver, base_graph):
    new_graph = nx.DiGraph()
    node_indicators = solver.get_variables(motile.variables.NodeSelected)
    edge_indicators = solver.get_variables(motile.variables.EdgeSelected)

    # Build nodes
    for node, index in node_indicators.items():
        if solver.solution[index] > 0.5:
            new_graph.add_node(node, **base_graph.nodes[node])

    # Build edges
    for edge, index in edge_indicators.items():
        if solver.solution[index] > 0.5:
            new_graph.add_edge(*edge, **base_graph.edges[edge])

    return new_graph


def print_solution_stats(solver, graph, gt_graph=None):
    time.sleep(0.1)  # to wait for ilpy prints
    print(
        f"\nCandidate graph\t\t{len(graph.nodes):3} nodes\t{len(graph.edges):3} edges"
    )
    if gt_graph:
        print(
            f"Ground truth graph\t{len(gt_graph.nodes):3}"
            f" nodes\t{len(gt_graph.edges):3} edges"
        )

    node_selected = solver.get_variables(motile.variables.NodeSelected)
    edge_selected = solver.get_variables(motile.variables.EdgeSelected)
    nodes = 0
    for node in graph.nodes:
        if solver.solution[node_selected[node]] > 0.5:
            nodes += 1
    edges = 0
    for u, v in graph.edges:
        if solver.solution[edge_selected[(u, v)]] > 0.5:
            edges += 1
    print(f"Solution graph\t\t{nodes:3} nodes\t{edges:3} edges")
