"""Turn a graph into the numbers a network reads.

A graph knows where its nodes are. That is not the same as knowing what to tell
the network about them. Absolute coordinates are the obvious thing to hand over
and the wrong one: with a few hundred training brackets, "stress is high at
x = 12" is easy to memorise and worthless on a bracket nobody has drawn yet.

So each node is described by things that would still mean something on a new
design:

  what kind of node am I          3   ordinary / clamped / loaded
  which way does my surface face  3   the outward normal
  how sharply does it turn here   1   1.0 flat, lower at an edge
  how far to the nearest clamp    1   mm
  how far to the nearest load     1   mm
                                  -
                                  9

and each edge by the step between its ends:

  the step from one end to the other  3
  how long that step is               1
                                      -
                                      4

Run on its own::

    python -m simjeb.features 0
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from simjeb import graph as graph_module
from simjeb import mesh as mesh_module

# The order of the node feature columns. Fixed, and never to be reordered: a
# trained model reads its input by position, so a saved checkpoint and a
# reordered feature set would silently disagree.
NODE_FEATURE_NAMES = [
    "is_ordinary",
    "is_clamped",
    "is_loaded",
    "normal_x",
    "normal_y",
    "normal_z",
    "sharpness",
    "distance_to_clamp",
    "distance_to_load",
]

EDGE_FEATURE_NAMES = ["step_x", "step_y", "step_z", "step_length"]

N_NODE_FEATURES = len(NODE_FEATURE_NAMES)
N_EDGE_FEATURES = len(EDGE_FEATURE_NAMES)

N_NODE_KINDS = 3


@dataclass
class Features:
    """One bracket, as the tensors a network consumes.

    Held in raw units. Scaling happens later and only from the training
    brackets, so these files stay usable whatever split is chosen next.
    """

    model_id: int
    node_features: np.ndarray     # (n_nodes, 9)
    edge_index: np.ndarray        # (2, n_directed_edges) sender, receiver
    edge_features: np.ndarray     # (n_directed_edges, 4)
    target: np.ndarray            # (n_nodes, 1) log1p(MPa) if configured
    stress_mpa: np.ndarray        # (n_nodes, 1) kept so scores can be in MPa

    @property
    def n_nodes(self):
        return len(self.node_features)

    @property
    def n_edges(self):
        return self.edge_index.shape[1]


def build(graph, log_target=True):
    """Graph -> features. No scaling, no shuffling, no decisions about splits."""
    normals, sharpness = mesh_module.vertex_normals(graph.positions,
                                                    graph.triangles)

    kind_one_hot = one_hot(graph.node_kind, N_NODE_KINDS)
    distance_to_clamp = distance_to_nearest(
        graph.positions, graph.node_kind == graph_module.NODE_CLAMPED)
    distance_to_load = distance_to_nearest(
        graph.positions, graph.node_kind == graph_module.NODE_LOADED)

    node_features = np.column_stack([
        kind_one_hot,                        # 0, 1, 2
        normals,                             # 3, 4, 5
        sharpness,                           # 6
        distance_to_clamp,                   # 7
        distance_to_load,                    # 8
    ])

    edge_index, edge_features = build_edges(graph)
    target, stress = build_target(graph.stress, log_target)

    return Features(
        model_id=graph.model_id,
        node_features=node_features.astype(np.float32),
        edge_index=edge_index,
        edge_features=edge_features.astype(np.float32),
        target=target.astype(np.float32),
        stress_mpa=stress.astype(np.float32),
    )


def one_hot(labels, n_kinds):
    """Turn 0/1/2 into three on/off columns.

    Handing the network 0, 1 and 2 directly would tell it that "loaded" is
    twice "clamped", and that clamped sits halfway between ordinary and loaded.
    None of that is true - they are three separate kinds with no order between
    them. Three columns, exactly one of them set, removes the false ordering.
    """
    encoded = np.zeros((len(labels), n_kinds))
    rows = np.arange(len(labels))
    encoded[rows, labels] = 1.0
    return encoded


def distance_to_nearest(positions, is_target):
    """Millimetres from every node to the nearest node of a given kind.

    Stress flows from where the load enters to where the part is held, so
    proximity to each end of that path is directly informative.

    It is also the only GLOBAL information the network gets. Message passing
    reaches a handful of neighbours per round, and the lug is dozens of hops
    from the nearest bolt hole - so without these two numbers a node in the
    middle of the bracket would have no idea where it sits in the part.

    A KD-tree answers "which of these is nearest" in a few steps instead of
    comparing against every candidate.
    """
    targets = positions[is_target]
    if len(targets) == 0:
        raise ValueError("no nodes of that kind to measure the distance to")

    tree = cKDTree(targets)
    distances, _ = tree.query(positions)
    return distances.reshape(-1, 1)


def build_edges(graph):
    """Each stored edge becomes two directed ones, plus its step vector.

    The graph stores every edge once, purely to halve the file. Messages have
    to travel both ways, so both directions are made here.

    The step vector is the reason this architecture was chosen at all. Stress
    comes from strain, and strain is a rate of change across space. A rate of
    change is built from differences between neighbouring points - exactly what
    a message carrying "the step from me to you" lets a layer compute. A graph
    network without edge features can record THAT two nodes are connected but
    not how far apart or in which direction, so it cannot represent the
    derivative at all.
    """
    first_end = graph.edges[:, 0]
    second_end = graph.edges[:, 1]

    senders = np.concatenate([first_end, second_end])
    receivers = np.concatenate([second_end, first_end])

    step = graph.positions[receivers] - graph.positions[senders]
    length = np.linalg.norm(step, axis=1, keepdims=True)

    edge_index = np.vstack([senders, receivers]).astype(np.int64)
    edge_features = np.hstack([step, length])
    return edge_index, edge_features


def build_target(stress_mpa, log_target):
    """The answer, and the same answer in MPa for reporting.

    ``log1p(v)`` is ``log(1 + v)``; the plus one keeps it defined at zero.

    Why transform at all: a linear-elastic solver reports unbounded stress at
    sharp re-entrant corners. Those values are numerical artefacts that grow
    with mesh refinement rather than settling, and under a plain squared-error
    loss a handful of them would supply most of the gradient - so the network
    would spend its capacity fitting corners instead of learning mechanics.

    log1p compresses that tail without discarding it, and makes the network
    care about RELATIVE error: 20% wrong matters the same at 200 MPa as at
    2,000, which is how an engineer would judge it.

    Every reported score inverts this first. A score computed in log space
    flatters the model, because the log shrinks exactly the large errors that
    matter most.
    """
    stress = stress_mpa.reshape(-1, 1)
    target = np.log1p(stress) if log_target else stress.copy()
    return target, stress


def invert_target(values, log_target):
    """Back to MPa. The exact opposite of :func:`build_target`."""
    return np.expm1(values) if log_target else values


def save(features, directory):
    """Write one bracket's features to ``<directory>/<model_id>.npz``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{features.model_id}.npz"

    np.savez(
        path,
        node_features=features.node_features,
        edge_index=features.edge_index.astype(np.int32),
        edge_features=features.edge_features,
        target=features.target,
        stress_mpa=features.stress_mpa,
    )
    return path


def load(directory, model_id):
    """Read back one bracket's features."""
    with np.load(Path(directory) / f"{model_id}.npz") as stored:
        return Features(
            model_id=model_id,
            node_features=stored["node_features"],
            edge_index=stored["edge_index"].astype(np.int64),
            edge_features=stored["edge_features"],
            target=stored["target"],
            stress_mpa=stored["stress_mpa"],
        )


def describe(features):
    """A per-column summary, for checking a bracket by eye."""
    lines = [f"{'column':<20}{'min':>12}{'mean':>12}{'max':>12}"]

    for index, name in enumerate(NODE_FEATURE_NAMES):
        column = features.node_features[:, index]
        lines.append(f"{name:<20}{column.min():>12.2f}{column.mean():>12.2f}"
                     f"{column.max():>12.2f}")

    for index, name in enumerate(EDGE_FEATURE_NAMES):
        column = features.edge_features[:, index]
        lines.append(f"{name:<20}{column.min():>12.2f}{column.mean():>12.2f}"
                     f"{column.max():>12.2f}")

    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import config

    model_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    index = graph_module.build_index(config.data_directories())
    graph = graph_module.build(index, model_id, config.LOAD_CASE,
                               config.MAX_COORDINATE_MISMATCH_MM)
    features = build(graph, config.LOG_TARGET)

    print(f"bracket {model_id}")
    print(f"  node_features {features.node_features.shape}")
    print(f"  edge_index    {features.edge_index.shape}")
    print(f"  edge_features {features.edge_features.shape}")
    print(f"  target        {features.target.shape}\n")
    print(describe(features))

    stress = features.stress_mpa
    print(f"\ntarget: {stress.min():.1f} to {stress.max():.1f} MPa"
          f"  ->  {features.target.min():.2f} to {features.target.max():.2f}"
          f"  (log1p = {config.LOG_TARGET})")

    # The transform must undo exactly, or every reported score is wrong.
    recovered = invert_target(features.target, config.LOG_TARGET)
    worst = np.abs(recovered - stress).max()
    print(f"inverting the transform recovers MPa to {worst:.1e} MPa")

    path = save(features, config.FEATURE_DIR)
    print(f"\nsaved {path.name} ({path.stat().st_size / 1e6:.1f} MB)")
