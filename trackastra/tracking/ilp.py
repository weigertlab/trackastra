import logging
import time
from dataclasses import dataclass, fields
from pathlib import Path

import networkx as nx
import yaml

logger = logging.getLogger(__name__)


def _require_motile():
    """Import the optional ``motile`` dependency, with an install hint on failure."""
    try:
        import motile
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "For tracking with an ILP, please conda install the optional `motile`"
            " dependency following https://funkelab.github.io/motile/install.html."
        )
    return motile


@dataclass(frozen=True)
class ILPConfig:
    """Weights and constants for the motile ILP tracking costs.

    Each ``*_w`` weight scales a per-detection/per-edge ``weight`` attribute and each
    ``*_c`` constant is a fixed cost added when the corresponding indicator is selected.
    Negative costs encourage selection. Defaults reproduce the ``"gt"`` preset.
    """

    node_w: float = 0.0
    node_c: float = -10.0  # strongly negative -> select all nodes
    edge_w: float = -1.0
    edge_c: float = 0.0
    appear_c: float = 0.25
    disappear_c: float = 0.5
    split_c: float = 0.25

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ILPConfig":
        """Load an ILP config from a YAML file whose keys are field names."""
        with open(path) as f:
            data = yaml.safe_load(f)
        allowed = {f.name for f in fields(cls)}
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(
                f"Unknown ILP config keys {sorted(unknown)}; allowed {sorted(allowed)}."
            )
        return cls(**data)


ILP_CONFIGS: dict[str, ILPConfig] = {
    "gt": ILPConfig(),
    "deepcell_gt": ILPConfig(split_c=1.0),
    "deepcell_gt_tuned": ILPConfig(appear_c=0.5, split_c=1.0),
    "deepcell_res_tuned": ILPConfig(
        node_c=0.25, edge_c=-0.25, disappear_c=0.25, split_c=1.0
    ),
}


def _resolve_ilp_config(ilp_config: "ILPConfig | str | None") -> ILPConfig:
    if ilp_config is None:
        return ILP_CONFIGS["gt"]
    if isinstance(ilp_config, ILPConfig):
        return ilp_config
    try:
        return ILP_CONFIGS[ilp_config]
    except KeyError:
        raise ValueError(
            f"Unknown ILP config {ilp_config!r}. Choose from {list(ILP_CONFIGS)}"
            " or pass an ILPConfig instance."
        )


def track_ilp(
    candidate_graph,
    allow_divisions: bool = True,
    ilp_config: "ILPConfig | str | None" = None,
) -> nx.DiGraph:
    if len(candidate_graph) == 0:
        return candidate_graph

    motile = _require_motile()
    config = _resolve_ilp_config(ilp_config)
    candidate_graph_motile = motile.TrackGraph(candidate_graph, frame_attribute="time")

    solver = solve_full_ilp(
        candidate_graph_motile,
        allow_divisions=allow_divisions,
        config=config,
    )
    print_solution_stats(solver, candidate_graph_motile)

    graph = solution_to_graph(solver, candidate_graph_motile)

    return graph


def solve_full_ilp(
    graph,
    allow_divisions: bool,
    config: ILPConfig,
):
    motile = _require_motile()
    logger.info(f"Using ILP config {config}")
    solver = motile.Solver(graph)

    solver.add_cost(
        motile.costs.NodeSelection(
            weight=config.node_w, constant=config.node_c, attribute="weight"
        )
    )
    solver.add_cost(
        motile.costs.EdgeSelection(
            weight=config.edge_w, constant=config.edge_c, attribute="weight"
        )
    )
    solver.add_cost(motile.costs.Appear(constant=config.appear_c))
    solver.add_cost(motile.costs.Disappear(constant=config.disappear_c))
    if allow_divisions:
        solver.add_cost(motile.costs.Split(constant=config.split_c))

    solver.add_constraint(motile.constraints.MaxParents(1))
    solver.add_constraint(motile.constraints.MaxChildren(2 if allow_divisions else 1))

    solver.solve()

    return solver


def solution_to_graph(solver, base_graph) -> nx.DiGraph:
    motile = _require_motile()
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
    motile = _require_motile()
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
