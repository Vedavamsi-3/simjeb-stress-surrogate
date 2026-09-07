"""Combine the deck, the mesh and the results into one graph.

This is where the three files finally meet, and where a bracket is either
accepted or excluded. Everything upstream reads a file; this decides whether
what was read hangs together.

The steps, in order:

  1. read all three, and check they describe the same bracket
  2. keep the surface nodes, drop the interior
  3. build the edges from the tetrahedra
  4. renumber the survivors 0, 1, 2, ... with no gaps
  5. shift the coordinates so the bolt holes sit in a common place

Run on its own to build one bracket::

    python -m simjeb.graph 0
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from simjeb import deck as deck_module
from simjeb import mesh as mesh_module

# The four-way node label in the results CSV. It is not a yes/no flag, and
# reading it as one would silently discard every bolt-hole and lug node -
# exactly the nodes where the physics enters the part.
SURF_INTERIOR = 0
SURF_SKIN = 1
SURF_BOLT = 2
SURF_LUG = 3

# Node kinds in the finished graph.
NODE_ORDINARY = 0
NODE_CLAMPED = 1
NODE_LOADED = 2

LOAD_CASES = ("ver", "hor", "dia", "tor")


class GraphError(Exception):
    """This bracket cannot become a usable graph."""


@dataclass
class Graph:
    """One bracket as a graph, ready for features to be built from it."""

    model_id: int
    positions: np.ndarray      # (n_nodes, 3) mm, centred on the interfaces
    edges: np.ndarray          # (n_edges, 2) node numbers, each edge once
    node_kind: np.ndarray      # (n_nodes,) ordinary / clamped / loaded
    stress: np.ndarray         # (n_nodes,) MPa, the answer
    landmarks: np.ndarray      # (5, 3) bolt-hole and lug centres, centred
    triangles: np.ndarray      # (n_triangles, 3) the skin, in graph numbering

    @property
    def n_nodes(self):
        return len(self.positions)

    @property
    def n_edges(self):
        return len(self.edges)

    def summary(self):
        clamped = int((self.node_kind == NODE_CLAMPED).sum())
        loaded = int((self.node_kind == NODE_LOADED).sum())
        return (f"{self.n_nodes:,} nodes, {self.n_edges:,} edges, "
                f"{clamped:,} clamped, {loaded:,} loaded")


def build(index, model_id, load_case="ver",
          coordinate_tolerance_mm=1e-3, landmark_tolerance_mm=5.0,
          reference_landmarks=None):
    """Build one graph, or raise explaining why the bracket is unusable.

    ``index`` is what :func:`build_index` returns - a filename lookup per
    file type. Passing the index rather than a folder is what lets the three
    file types live in three different places, and it means the directory
    listing happens once for the whole run instead of once per bracket.

    ``reference_landmarks`` is where the interfaces sit in a typical
    bracket. Pass it to reject brackets that are rotated; leave it out on
    the first pass, when there is nothing yet to compare against.
    """
    if load_case not in LOAD_CASES:
        raise GraphError(f"unknown load case {load_case!r}")

    paths = paths_for(index, model_id)

    # ---- 1. read, and check the three files agree -----------------------
    try:
        deck = deck_module.parse(paths["fem"], model_id)
        mesh = mesh_module.read(paths["vtk"])
    except (deck_module.DeckError, mesh_module.MeshError) as error:
        raise GraphError(str(error)) from error

    columns = ["surf", "x", "y", "z", f"{load_case}_stress"]
    results = pd.read_csv(paths["csv"], usecols=columns)

    try:
        mesh_module.check_against_results(
            mesh, results[["x", "y", "z"]].to_numpy(), coordinate_tolerance_mm)
        deck_module.check_against_surface_labels(
            deck, results["surf"].to_numpy())
    except (deck_module.DeckError, mesh_module.MeshError) as error:
        raise GraphError(str(error)) from error

    # ---- 2. keep the surface ----------------------------------------------
    surface_codes = results["surf"].to_numpy()
    keep = surface_codes > SURF_INTERIOR
    if not keep.any():
        raise GraphError("no nodes are flagged as surface")

    kept_rows = np.flatnonzero(keep)

    # ---- 3. edges from the tetrahedra -------------------------------------
    try:
        edges_in_mesh_numbering = mesh_module.surface_edges(mesh, keep)
    except mesh_module.MeshError as error:
        raise GraphError(str(error)) from error

    # ---- 4. renumber -------------------------------------------------------
    # The kept nodes have scattered original numbers. The graph needs them
    # numbered 0, 1, 2, ... with no gaps, because a network stores one row per
    # node and cannot leave 60% of its rows empty.
    #
    # A lookup table over the whole mesh does the translation. Dropped nodes
    # keep -1, so an edge pointing at one shows up immediately.
    mesh_to_graph = np.full(mesh.n_nodes, -1, dtype=np.int64)
    mesh_to_graph[kept_rows] = np.arange(len(kept_rows))

    edges = mesh_to_graph[edges_in_mesh_numbering]
    if edges.min() < 0:
        raise GraphError("an edge points at a node that was dropped")

    degrees = mesh_module.node_degrees(edges, len(kept_rows))
    isolated = int((degrees == 0).sum())
    if isolated > 0:
        raise GraphError(f"{isolated} surface nodes have no edges at all")

    triangles_in_mesh_numbering = mesh_module.surface_triangles(mesh)
    triangles = mesh_to_graph[triangles_in_mesh_numbering]
    if triangles.min() < 0:
        raise GraphError("a skin triangle touches a node that was dropped")

    # ---- 5. put the bracket in a common frame -----------------------------
    landmarks = interface_landmarks(mesh.points, deck)
    origin = landmarks.mean(axis=0)

    # Subtracting one row of three numbers from an array of thousands of rows
    # works because numpy repeats it down every row. Only a shift is applied:
    # the shape, the size and the load direction are all unchanged, which is
    # what makes it safe.
    #
    # Rotation is deliberately NOT corrected. The load is a fixed world
    # direction that would not turn with the part, so straightening a rotated
    # bracket would leave an aligned mesh whose answers came from a different
    # physics problem.
    centred_landmarks = landmarks - origin
    if reference_landmarks is not None:
        offset = np.abs(centred_landmarks - reference_landmarks).max()
        if offset > landmark_tolerance_mm:
            raise GraphError(
                f"interfaces sit up to {offset:.1f} mm from where they sit in "
                f"a typical bracket; this one is rotated")

    positions = (mesh.points - origin)[kept_rows]

    # ---- the node kinds and the answer ------------------------------------
    node_kind = np.full(len(kept_rows), NODE_ORDINARY, dtype=np.int64)
    node_kind[mesh_to_graph[deck.fixed_rows]] = NODE_CLAMPED
    node_kind[mesh_to_graph[deck.loaded_rows]] = NODE_LOADED

    stress = results[f"{load_case}_stress"].to_numpy()[kept_rows]
    if not np.isfinite(stress).all():
        raise GraphError("the results contain non-finite stress values")
    if (stress < 0).any():
        raise GraphError("the results contain negative von Mises stress")

    return Graph(
        model_id=model_id,
        positions=positions,
        edges=edges,
        node_kind=node_kind,
        stress=stress,
        landmarks=centred_landmarks,
        triangles=triangles,
    )


def interface_landmarks(points, deck):
    """The centre of each bolt hole and of the load lug: five points.

    These are the only features every bracket shares - the competition fixed
    the bolt pattern and the lug, and left everything between them to the
    designer. So they are the one thing that can define a common frame.

    Deliberately NOT the centre of the part. That moves with the design, so
    centring there would put the bolt holes somewhere different in every
    bracket and destroy the only regularity worth keeping.
    """
    centres = []
    for index in range(len(deck.bolt_holes)):
        rows = deck.bolt_hole_rows(index)
        centres.append(points[rows].mean(axis=0))

    centres.append(points[deck.loaded_rows].mean(axis=0))
    return np.vstack(centres)


def save(graph, directory):
    """Write one graph to ``<directory>/<model_id>.npz``.

    One file per bracket, not one big file: a crash on bracket 300 must not
    throw away the first 299.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{graph.model_id}.npz"

    np.savez(
        path,
        positions=graph.positions.astype(np.float32),
        edges=graph.edges.astype(np.int32),
        node_kind=graph.node_kind.astype(np.int8),
        stress=graph.stress.astype(np.float32),
        landmarks=graph.landmarks,
        triangles=graph.triangles.astype(np.int32),
    )
    return path


def load(directory, model_id):
    """Read back a saved graph."""
    with np.load(Path(directory) / f"{model_id}.npz") as stored:
        return Graph(
            model_id=model_id,
            positions=stored["positions"],
            edges=stored["edges"].astype(np.int64),
            node_kind=stored["node_kind"].astype(np.int64),
            stress=stored["stress"],
            landmarks=stored["landmarks"],
            triangles=stored["triangles"].astype(np.int64),
        )


def index_files(directory, suffix):
    """Map every matching filename to its path, searching subfolders too.

    Built once and reused, because a directory listing is far cheaper than
    asking the filesystem about each of 381 files individually - and
    SimJEB's results are split across two halves, one of which unpacks
    into a subfolder, so a plain listing of the top level would miss half
    the dataset.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise GraphError(f"{directory} is not a directory")

    found = {}
    for path in directory.rglob("*" + suffix):
        # A later copy wins. Duplicates across halves are the same file.
        found[path.name] = path
    return found


def build_index(directories):
    """One filename lookup per file type: {"fem": {...}, "vtk": {...}, ...}.

    ``directories`` is a dict of the three source folders. They may all be
    the same folder, or three different ones - the code below does not
    change either way.
    """
    return {
        "fem": index_files(directories["fem"], ".fem"),
        "vtk": index_files(directories["vtk"], ".vtk"),
        "csv": index_files(directories["csv"], "field.csv"),
    }


def paths_for(index, model_id):
    """The three files for one bracket. Raises if any is missing."""
    wanted = {
        "fem": f"{model_id}.fem",
        "vtk": f"{model_id}.vtk",
        "csv": f"{model_id}field.csv",
    }

    paths = {}
    for kind, filename in wanted.items():
        if filename not in index[kind]:
            raise GraphError(f"{filename} not found")
        paths[kind] = index[kind][filename]
    return paths


def find_brackets(directories):
    """Which brackets are present, discovered rather than hard-coded.

    A bracket needs all three files, and they may be in three different
    folders. One missing any of them can never be built, so it must not
    enter the loop in the first place.
    """
    index = build_index(directories)

    found = []
    for filename in index["fem"]:
        name = Path(filename).stem
        if not name.isdigit():
            continue                      # READMEs and other strays

        model_id = int(name)
        has_mesh = f"{model_id}.vtk" in index["vtk"]
        has_results = f"{model_id}field.csv" in index["csv"]
        if has_mesh and has_results:
            found.append(model_id)

    return sorted(found)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import config

    model_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    directories = config.data_directories()
    for kind, folder in directories.items():
        print(f"  {kind:<4} {folder}")

    index = build_index(directories)
    available = find_brackets(directories)
    print(f"{len(available)} brackets available, first few: "
          f"{available[:10]}")
    print()

    graph = build(index, model_id, config.LOAD_CASE,
                  config.MAX_COORDINATE_MISMATCH_MM)
    print(f"bracket {model_id}: {graph.summary()}")
    print(f"  positions   : {graph.positions.shape}")
    print(f"  triangles   : {graph.triangles.shape}")
    print(f"  stress      : {graph.stress.min():.1f} to "
          f"{graph.stress.max():.1f} MPa")

    print("\n  interface landmarks, after centring:")
    names = ["bolt hole 1", "bolt hole 2", "bolt hole 3", "bolt hole 4",
             "load lug"]
    for name, point in zip(names, graph.landmarks):
        print(f"    {name:<12} ({point[0]:8.2f}, {point[1]:8.2f}, "
              f"{point[2]:8.2f})")

    path = save(graph, config.GRAPH_DIR)
    reloaded = load(config.GRAPH_DIR, model_id)
    identical = (
        np.array_equal(reloaded.edges, graph.edges)
        and np.allclose(reloaded.positions, graph.positions)
        and np.array_equal(reloaded.node_kind, graph.node_kind)
    )
    print(f"\n  saved {path.name} ({path.stat().st_size / 1e6:.1f} MB), "
          f"reloads identical: {identical}")
