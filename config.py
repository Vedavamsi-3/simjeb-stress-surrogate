"""Every setting for the whole project, in one file.

Nothing else in this project contains a number you might want to change. If a
value would ever need editing - a path, a learning rate, how many message
rounds - it belongs here and not buried in a script.

Two reasons that matters:

  * you can read this file and know what a run did, without reading the code
  * changing an experiment means editing one file, and the change is visible
    in a diff instead of hidden three functions deep
"""

import os
from pathlib import Path


def from_environment(name, default):
    """Let an environment variable override a path, and otherwise use the
    value written here.

    Kaggle is the reason this exists. There, the dataset arrives as a
    READ-ONLY folder under /kaggle/input, and anything written has to go to
    /kaggle/working. Neither path exists on this machine, and hardcoding
    either would mean keeping two copies of this file - which is how the two
    quietly drift apart and a Kaggle run stops matching a local one.

    So the notebook sets the environment and nothing in the code changes::

        os.environ["SIMJEB_GRAPH_DIR"] = "/kaggle/input/simjeb/graphs"
        os.environ["SIMJEB_FEATURE_DIR"] = "/kaggle/input/simjeb/features"
        os.environ["SIMJEB_OUTPUT_DIR"] = "/kaggle/working/output"
    """
    value = os.environ.get(name)
    if value:
        return Path(value)
    return default


# ---------------------------------------------------------------------------
# WHERE THINGS LIVE
# ---------------------------------------------------------------------------

# WHERE THE RAW FILES ARE
#
# The three file types do not have to live together. SimJEB ships them in
# three separate folders - decks, meshes and results - and a working copy
# often ends up arranged differently again.
#
# So each type gets its own directory. Point all three at the same folder
# when they do happen to be together; that is simply the special case where
# the three values are equal.
#
# Subfolders are searched too, which matters because SimJEB splits its
# results across two halves and one of them unpacks into a subfolder.

# --- the full dataset, three separate folders -------------------------
SIMJEB_ROOT = Path(r"D:\Vamsi_courses\Projects\3D_deep_learning_project_2")

FEM_DIR = SIMJEB_ROOT / "Input_files"     # <id>.fem       the solver decks
VTK_DIR = SIMJEB_ROOT / "Output_files"    # <id>.vtk       the meshes
CSV_DIR = SIMJEB_ROOT / "Csv_files"       # <id>field.csv  the results

# --- the 8-bracket sample, all in one folder --------------------------
# Uncomment these to work on the small set instead. Nothing else changes:
# the code does not care whether the three paths differ.
#
# SAMPLE = Path(r"D:\Vamsi_courses\Projects"
#               r"\Simjeb-strucutral-gnn_understanding\multi_bracket\Data")
# FEM_DIR = VTK_DIR = CSV_DIR = SAMPLE


def data_directories():
    """The three source folders, keyed by the file type each holds."""
    return {"fem": FEM_DIR, "vtk": VTK_DIR, "csv": CSV_DIR}

# The metadata table. Two things depend on it: which brackets exist at all,
# and the category and submission each belongs to.
#
# It does NOT have to live beside the raw files, and usually should not. A
# cleaned table - one that has had corrupted rows removed - is the authority on
# which brackets the pipeline may use, so it is kept separately and pointed at
# here. Swap this line to change which dataset a run is built from.
# Relative to this file, so moving the project does not break it. The raw
# data folders above are absolute because they live outside the project;
# anything inside it should be found relative to the code.
METADATA_FILE = (Path(__file__).parent / "Data_processing_manual"
                 / "cleaned_raw_meta_data_file.csv")

# A bracket present on disk but absent from the metadata was removed during
# data cleaning. Skip it rather than building it and discarding it later.
METADATA_IS_AUTHORITATIVE = True

# Everything this project produces goes under here, nothing anywhere else.
OUTPUT_DIR = from_environment("SIMJEB_OUTPUT_DIR",
                              Path(__file__).parent / "output")

# The two big caches are named separately because on a training machine they
# are INPUTS, sitting somewhere read-only, while everything else is output.
GRAPH_DIR = from_environment("SIMJEB_GRAPH_DIR",
                             OUTPUT_DIR / "graphs")     # geometry
FEATURE_DIR = from_environment("SIMJEB_FEATURE_DIR",
                               OUTPUT_DIR / "features")  # model inputs

REPORT_DIR = OUTPUT_DIR / "reports"      # what was excluded, and why
RUN_DIR = OUTPUT_DIR / "runs"            # checkpoints and training history

# The split and the scaling travel WITH the dataset, not with the outputs: a
# checkpoint is only meaningful next to the exact numbers its inputs were
# scaled by, and a score is only meaningful next to the split it was measured
# on. Uploading the dataset means uploading these two as well.
SPLIT_FILE = from_environment("SIMJEB_SPLIT_FILE", OUTPUT_DIR / "split.json")
SCALING_FILE = from_environment("SIMJEB_SCALING_FILE",
                                OUTPUT_DIR / "scaling.npz")


# ---------------------------------------------------------------------------
# THE PROBLEM
# ---------------------------------------------------------------------------

# SimJEB solves four load cases per bracket. The results CSV holds all four;
# this picks which one to train on.
LOAD_CASE = "ver"                  # ver | hor | dia | tor

# Only surface nodes are kept. The interior is about 60% of the mesh and is
# discarded: cracks start at the surface, and dropping it makes the dataset
# small enough to work with.
SURFACE_ONLY = True


# ---------------------------------------------------------------------------
# QUALITY CONTROL
# ---------------------------------------------------------------------------
# A bracket failing any of these is excluded, with its reason written to a file.
# Silently dropping samples is how a dataset quietly becomes something other
# than what you think you are training on.

# The results CSV and the mesh must agree on every node's position, or the
# stress values are attached to the wrong nodes.
MAX_COORDINATE_MISMATCH_MM = 1e-3

# The bolt holes and load lug must land in the same place in every bracket
# after centring. A bracket further out than this is rotated - and a rotated
# bracket cannot simply be turned straight, because the load is a fixed
# world-direction that would not turn with it. It is a different problem.
MAX_LANDMARK_OFFSET_MM = 5.0


# ---------------------------------------------------------------------------
# THE SPLIT
# ---------------------------------------------------------------------------

TEST_FRACTION = 0.2
VAL_FRACTION = 0.1
SPLIT_SEED = 0

# Brackets sharing a GrabCAD submission are variants of one design. They must
# never be separated across splits: a variant in train and its twin in test
# means the model has effectively seen the test bracket.
GROUP_BY_SUBMISSION = True

# Keep each shape category in proportion across the splits, so a whole category
# cannot land entirely in one of them.
STRATIFY_BY_CATEGORY = True


# ---------------------------------------------------------------------------
# THE MODEL
# ---------------------------------------------------------------------------

HIDDEN_WIDTH = 64          # numbers describing each node internally
MESSAGE_ROUNDS = 8         # how many hops a node can see
DROPOUT = 0.0

# The stress target spans 300 to 15,000 MPa because the solver reports
# unbounded stress at sharp corners. log1p compresses that tail without
# discarding it. Every reported score inverts it back to MPa first.
LOG_TARGET = True


# ---------------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------------

BATCH_SIZE = 2             # brackets per step; limited by memory, not by choice
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3
MAX_EPOCHS = 300
GRAD_CLIP = 1.0

# Stop when the validation loss has not improved for this many epochs. Without
# it a network keeps improving on what it has seen long after it has stopped
# improving on what it has not.
PATIENCE = 30
MIN_IMPROVEMENT = 1e-5

# Wall-clock budget in hours. The loop stops cleanly and can be resumed, so a
# session shorter than the run does not lose the work.
MAX_HOURS = 10.5

SEED = 0
DEVICE = "cuda"            # falls back to cpu automatically if absent


def describe():
    """A one-screen summary of the settings, for the top of a run log."""
    lines = [
        f"decks      : {FEM_DIR}",
        f"meshes     : {VTK_DIR}",
        f"results    : {CSV_DIR}",
        f"graphs     : {GRAPH_DIR}",
        f"features   : {FEATURE_DIR}",
        f"output     : {OUTPUT_DIR}",
        f"load case  : {LOAD_CASE}",
        f"model      : {HIDDEN_WIDTH} wide, {MESSAGE_ROUNDS} message rounds",
        f"training   : batch {BATCH_SIZE}, lr {LEARNING_RATE}, "
        f"up to {MAX_EPOCHS} epochs, patience {PATIENCE}",
        f"split      : {int(100 * (1 - TEST_FRACTION - VAL_FRACTION))}"
        f"/{int(100 * VAL_FRACTION)}/{int(100 * TEST_FRACTION)} "
        f"train/val/test, seed {SPLIT_SEED}",
    ]
    return "\n".join(lines)


def make_directories():
    """Create every output directory. Safe to call repeatedly.

    The graph and feature folders are treated differently from the rest: when
    they have been pointed somewhere read-only - a Kaggle input dataset - they
    already exist and there is nothing to create. Only a path that is BOTH
    missing and uncreatable is a real problem, and it is worth saying so
    clearly rather than letting a PermissionError surface from three
    functions down.
    """
    for directory in (OUTPUT_DIR, REPORT_DIR, RUN_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    for name, directory in (("SIMJEB_GRAPH_DIR", GRAPH_DIR),
                            ("SIMJEB_FEATURE_DIR", FEATURE_DIR)):
        if directory.is_dir():
            continue
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise SystemExit(
                f"cannot use {directory} - it does not exist and cannot be "
                f"created ({error}).\nIf this is a read-only dataset folder, "
                f"check the path in ${name}.") from error


if __name__ == "__main__":
    print(describe())
    make_directories()
    print(f"\ndirectories ready under {OUTPUT_DIR}")
