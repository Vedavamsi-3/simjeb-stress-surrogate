"""Read the physics out of a SimJEB ``.fem`` solver deck.

The mesh file says where the material is. The results file says what the stress
turned out to be. Neither says where the bracket is bolted down or how hard it
is pushed - that lives only here.

The awkward part is that the deck never names real mesh nodes directly. It says
"node 112874 is fixed", where node 112874 is an imaginary point at the centre of
a bolt hole, tied to the ring of real nodes around it by a rigid element.
Expanding that is most of this module.

Run on its own to check one bracket::

    python -m simjeb.deck 0
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Nastran fixed-format: 9 fields of 8 characters to a line. There are no
# separators - a value's meaning is decided entirely by which columns it sits
# in, which is why every offset below is written down rather than counted.
FIELD_WIDTH = 8
FIELDS_PER_LINE = 9


class DeckError(Exception):
    """The deck cannot be trusted. The caller should exclude this bracket."""


@dataclass
class Deck:
    """Everything the rest of the pipeline needs from one ``.fem`` file."""

    model_id: int
    n_mesh_nodes: int
    material: dict                        # E, nu, rho
    load_cases: list                      # names, in subcase order
    load_vectors: dict                    # case name -> 3 numbers
    fixed_nodes: np.ndarray               # 1-based node ids, all bolt holes
    loaded_nodes: np.ndarray              # 1-based node ids, the load lug
    bolt_holes: list = field(default_factory=list)   # the four, kept separate

    @property
    def fixed_rows(self):
        """The clamped nodes as array rows. Node ids start at 1, rows at 0."""
        return self.fixed_nodes - 1

    @property
    def loaded_rows(self):
        return self.loaded_nodes - 1

    def bolt_hole_rows(self, index):
        return np.asarray(self.bolt_holes[index]) - 1


# ---------------------------------------------------------------------------
# READING THE FORMAT
# ---------------------------------------------------------------------------

def split_fields(line):
    """Cut one line into its nine 8-character fields.

    Field 0 is the card name, fields 1-8 are its data. Blank fields still
    occupy their columns, so a card with a gap in the middle still has its
    later values at fixed positions.
    """
    line = line.rstrip("\n").rstrip("\r")

    fields = []
    for index in range(FIELDS_PER_LINE):
        start = index * FIELD_WIDTH
        stop = start + FIELD_WIDTH
        fields.append(line[start:stop])
    return fields


def read_number(text):
    """Read a Nastran real, including its home-made scientific notation.

    Eight characters is not much room, so Nastran drops the E from exponents:
    4.43e-9 is written 4.43-9.

    A plain float() raises on that, which is survivable. The dangerous version
    is a parser that strips the sign and reads 4.43 - a billion times too
    large, with nothing to signal the mistake.
    """
    text = text.strip()
    if not text:
        return None

    try:
        return float(text)
    except ValueError:
        pass

    # Look for a + or - that is neither the leading sign nor part of an
    # exponent already. That position is where the missing E belongs.
    for index in range(1, len(text)):
        character = text[index]
        previous = text[index - 1]
        if character in "+-" and previous not in "eE":
            return float(text[:index] + "e" + text[index:])

    raise DeckError(f"cannot read {text!r} as a number")


def read_integers(fields):
    """Every non-blank field, as integers. Used for the rigid-element node lists."""
    values = []
    for text in fields:
        if text.strip():
            values.append(int(text))
    return values


# ---------------------------------------------------------------------------
# THE PARSER
# ---------------------------------------------------------------------------

def parse(path, model_id=None):
    """Read one deck. Raises :class:`DeckError` if it cannot be trusted.

    A single pass. The file is ~40 MB and 99.9% of its lines are GRID (node
    coordinates) and CTETRA (elements), both of which come more cheaply from
    the .vtk - so those are rejected on the first character and never parsed.
    """
    path = Path(path)
    if model_id is None:
        model_id = int(path.stem)

    material = None
    subcases = []
    constraint_sets = {}         # set number -> node ids it holds fixed
    load_sets = {}               # set number -> where and how hard
    rigid_elements = []          # (reference node, [real node ids])

    # A rigid element's node list runs over many lines. This holds the one
    # currently accepting continuation lines.
    open_element = None

    with open(path, "r", errors="replace") as handle:
        for line in handle:
            if line.startswith(("GRID", "CTETRA", "$")):
                continue

            if line.startswith("+"):
                if open_element is not None:
                    fields = split_fields(line)
                    open_element[1].extend(read_integers(fields[1:]))
                continue

            open_element = None
            fields = split_fields(line)
            card = fields[0].strip()

            if card == "MAT1":
                material = read_material(fields)

            elif card == "SUBCASE":
                subcases.append({"id": int(fields[1]), "label": "",
                                 "constraints": None, "load": None})

            elif card == "SPC":
                set_number = int(fields[1])
                node_id = int(fields[2])
                constraint_sets.setdefault(set_number, []).append(node_id)

            elif card in ("FORCE", "MOMENT"):
                load_sets[int(fields[1])] = read_load(fields, card)

            elif card == "RBE2":
                # field 2 is the reference node, real nodes start at field 4
                open_element = (int(fields[2]), read_integers(fields[4:]))
                rigid_elements.append(open_element)

            elif card == "RBE3":
                # field 3 is the reference node, real nodes start at field 7
                open_element = (int(fields[3]), read_integers(fields[7:]))
                rigid_elements.append(open_element)

            else:
                read_case_control(line, subcases)

    return assemble(model_id, material, subcases, constraint_sets,
                    load_sets, rigid_elements)


def read_material(fields):
    """MAT1 gives the material. Its layout is fixed and easy to miscount::

        1 = id    2 = E    3 = G (blank here)    4 = nu    5 = rho

    Field 3 is empty in these decks but still occupies its eight columns, so nu
    is at 4 and not at 3.
    """
    return {
        "E": read_number(fields[2]),
        "nu": read_number(fields[4]),
        "rho": read_number(fields[5]),
    }


def read_load(fields, card):
    """FORCE is a push, MOMENT is a twist. Both act at one node.

    The card gives a scale factor and a direction; the load is their product.
    """
    scale = read_number(fields[4])
    if scale is None:
        scale = 1.0

    direction = []
    for index in (5, 6, 7):
        value = read_number(fields[index])
        direction.append(0.0 if value is None else value)

    return {
        "node": int(fields[2]),
        "vector": scale * np.array(direction),
        "kind": card,
    }


def read_case_control(line, subcases):
    """The indented lines under a SUBCASE: its name, and which sets it uses."""
    if not subcases:
        return

    text = line.strip()
    if text.startswith("LABEL"):
        subcases[-1]["label"] = text.split("LABEL", 1)[1].strip()
    elif text.startswith("SPC ="):
        subcases[-1]["constraints"] = int(text.split("=", 1)[1])
    elif text.startswith("LOAD ="):
        subcases[-1]["load"] = int(text.split("=", 1)[1])


# ---------------------------------------------------------------------------
# TURNING WHAT WAS READ INTO WHAT IS NEEDED
# ---------------------------------------------------------------------------

def assemble(model_id, material, subcases, constraint_sets, load_sets,
             rigid_elements):
    """Check the deck makes sense, then expand it into real node lists.

    Each check below guards a failure that would NOT crash. Boundary conditions
    would attach to the wrong nodes and the model would train happily on
    nonsense, so a bracket that fails is refused here rather than silently
    corrupting the dataset.
    """
    if material is None:
        raise DeckError("no MAT1 card: the material is unknown")
    if not subcases:
        raise DeckError("no SUBCASE blocks: the load cases are unknown")
    if not rigid_elements:
        raise DeckError("no RBE2/RBE3 elements: cannot locate the interfaces")
    if not constraint_sets:
        raise DeckError("no SPC cards: nothing is clamped")

    n_mesh_nodes = find_mesh_node_count(rigid_elements)

    constraint_set = subcases[0]["constraints"]
    if constraint_set not in constraint_sets:
        raise DeckError(f"subcase 1 names SPC set {constraint_set}, "
                        f"which does not exist")
    clamped_references = set(constraint_sets[constraint_set])

    fixed = []
    loaded = []
    bolt_holes = []

    for reference_node, members in rigid_elements:
        if reference_node in clamped_references:
            fixed.extend(members)
            bolt_holes.append(sorted(members))
        else:
            loaded.extend(members)

    if not fixed:
        raise DeckError("no rigid element is attached to a clamped node")
    if not loaded:
        raise DeckError("no rigid element carries the load")

    load_vectors = {}
    load_cases = []
    for subcase in subcases:
        name = subcase["label"] or f"case_{subcase['id']}"
        load_cases.append(name)
        if subcase["load"] in load_sets:
            load_vectors[name] = load_sets[subcase["load"]]["vector"]

    return Deck(
        model_id=model_id,
        n_mesh_nodes=n_mesh_nodes,
        material=material,
        load_cases=load_cases,
        load_vectors=load_vectors,
        fixed_nodes=np.array(sorted(fixed)),
        loaded_nodes=np.array(sorted(loaded)),
        bolt_holes=bolt_holes,
    )


def find_mesh_node_count(rigid_elements):
    """Where the real mesh nodes end and the imaginary ones begin.

    The reference nodes named by rigid elements do not exist in the mesh. They
    are numbered above the real nodes in one contiguous block, so the smallest
    of them marks the boundary.

    That block being contiguous is what makes ``row = node_id - 1`` true for
    every real node. One bracket in the full 381 breaks it, so it is checked
    rather than assumed.
    """
    reference_nodes = sorted(reference for reference, _ in rigid_elements)
    n_mesh_nodes = min(reference_nodes) - 1

    expected = list(range(n_mesh_nodes + 1,
                          n_mesh_nodes + 1 + len(reference_nodes)))
    if reference_nodes != expected:
        raise DeckError(
            f"reference nodes {reference_nodes} are not the contiguous block "
            f"above node {n_mesh_nodes}; node numbering cannot be trusted")

    return n_mesh_nodes


def check_against_surface_labels(deck, surface_codes):
    """Confirm the deck's interfaces match the ones the results CSV labels.

    Two independent routes to the same node lists. The deck reached them
    through SPC and RBE2 cards; the CSV reached them through SimJEB's own
    labelling of each node. Nothing connects the two, so agreement is real
    evidence that the field offsets, the continuation lines and the 1-based to
    0-based conversion are all correct at once.

    ``surface_codes`` is the CSV's ``surf`` column: 2 marks a bolt-hole node,
    3 marks a load-lug node.
    """
    if len(surface_codes) != deck.n_mesh_nodes:
        raise DeckError(
            f"deck says {deck.n_mesh_nodes:,} mesh nodes, "
            f"the results file has {len(surface_codes):,} rows")

    labelled_bolt = np.flatnonzero(surface_codes == 2) + 1
    labelled_lug = np.flatnonzero(surface_codes == 3) + 1

    if not np.array_equal(deck.fixed_nodes, labelled_bolt):
        raise DeckError(
            f"deck and results file disagree on the bolt holes: "
            f"{len(deck.fixed_nodes)} nodes against {len(labelled_bolt)}")

    if not np.array_equal(deck.loaded_nodes, labelled_lug):
        raise DeckError(
            f"deck and results file disagree on the load lug: "
            f"{len(deck.loaded_nodes)} nodes against {len(labelled_lug)}")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import config
    from simjeb import graph as graph_module

    model_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    index = graph_module.build_index(config.data_directories())
    paths = graph_module.paths_for(index, model_id)
    deck = parse(paths["fem"])

    print(f"bracket {deck.model_id}")
    print(f"  mesh nodes   : {deck.n_mesh_nodes:,}")
    print(f"  material     : E {deck.material['E']:,.0f} MPa, "
          f"nu {deck.material['nu']}, rho {deck.material['rho']}")
    print(f"  load cases   : {deck.load_cases}")
    for name, vector in deck.load_vectors.items():
        print(f"    {name:<12} ({vector[0]:>10,.0f}, {vector[1]:>10,.0f}, "
              f"{vector[2]:>10,.0f})")
    print(f"  clamped      : {len(deck.fixed_nodes):,} nodes "
          f"across {len(deck.bolt_holes)} bolt holes")
    print(f"  loaded       : {len(deck.loaded_nodes):,} nodes on the lug")

    import pandas as pd

    codes = pd.read_csv(paths["csv"],
                        usecols=["surf"])["surf"].to_numpy()
    check_against_surface_labels(deck, codes)
    print("  cross-check against the results file: PASSED")
