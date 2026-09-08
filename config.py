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

# Train stays at 70% either way; these two only trade against each other.
# 15/15 rather than 10/20 buys a steadier early-stopping decision at the cost
# of a wider error bar on the reported test number.
#
# Why that trade is worth it here: nodes within one bracket are not
# independent - same shape, same mesh, same load path - so the effective
# sample size of a split is closer to its BRACKET count than its node count.
# At 10% the validation set was 33 brackets, and one hard bracket carried 3%
# of the number the stopping rule watches. 50 is thin but no longer fragile.
TEST_FRACTION = 0.15
VAL_FRACTION = 0.15
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

HIDDEN_WIDTH = 128         # numbers describing each node internally
MESSAGE_ROUNDS = 8         # how many hops a node can see
DROPOUT = 0.0

# Trade time for memory: throw away each block's internal activations during
# the forward pass and rebuild them during the backward one.
#
# This is here because of what the first full run said. It reached an R of
# 0.29 on the brackets it had SEEN - it was not memorising the training set,
# it could not fit it - and a network of 247,000 parameters against 8.4
# million training nodes is a plausible reason why. The fix is a wider one.
#
# But 128 wide roughly doubles the edge tensors that already set the memory
# ceiling, and the largest bracket does not fit on a T4 at that width. So the
# two settings arrive together: without the recompute, HIDDEN_WIDTH 128 is an
# out-of-memory error rather than an experiment.
#
# Costs about 30% in wall-clock. Turn it off to get the plain forward pass
# back - at HIDDEN_WIDTH 64 it is not needed.
CHECKPOINT_ACTIVATIONS = True

# The stress target spans 300 to 15,000 MPa because the solver reports
# unbounded stress at sharp corners. log1p compresses that tail without
# discarding it. Every reported score inverts it back to MPa first.
LOG_TARGET = True


# ---------------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------------

# Brackets per step. One, and it is a memory limit rather than a choice.
#
# Bracket sizes vary fourfold - 18,000 nodes to 120,174 - and batches are
# shuffled, so what matters is not the average draw but the worst one. At
# batch 2 the two largest brackets landing together needs about 20 GB, and a
# T4 has 14.56 usable. That is not a rare edge case over a long run: it
# happened at epoch 2.
#
# At batch 1 the largest single bracket projected to about 10.3 GB at
# HIDDEN_WIDTH 64, which fitted with roughly 4 GB to spare. At 128 it would
# not fit at all - see CHECKPOINT_ACTIVATIONS, which is what buys it back.
#
# The cost is noisier gradients and twice as many steps per epoch. Neither is
# a real problem here - GRAD_CLIP already caps the damage a single extreme
# bracket can do, and small batches are closer to what MeshGraphNet papers
# use anyway.
BATCH_SIZE = 1

# How many brackets contribute to one weight update.
#
# BATCH_SIZE is pinned at 1 by memory rather than by choice, and one bracket
# is a noisy thing to steer by: the gradient it gives is the reaction to one
# shape, not to the dataset. Adding the gradients up over several brackets
# before stepping recovers most of what a real batch would have given - about
# 8x less noise here - and costs nothing in memory, because the brackets are
# still processed one at a time and each one's working memory is freed before
# the next begins.
#
# What it does change is 29 weight updates an epoch instead of 232: steadier
# steps, fewer of them. Set to 1 to get the old behaviour back exactly.
ACCUMULATION_STEPS = 8

# Lowered from 1e-3 after the first full run, whose validation loss bounced
# between 0.55 and 0.65 for its last 30 epochs while the training loss crept
# steadily down - too large a step to settle.
#
# Not lowered as far as it looks. Accumulating 8 brackets already removes most
# of the noise that caused the bouncing, and a larger effective batch supports
# a LARGER learning rate, not a smaller one. Cutting this to 1e-4 while
# accumulating 8x would have been moving two dials in opposite directions.
LEARNING_RATE = 3e-4

# A penalty on the size of every weight, pulling each one slightly towards
# zero on every step. It buys smoother, less memorised answers by spending
# capacity - which is the wrong trade for a model that cannot yet fit the data
# it has already seen.
#
# Worth being honest about the size of this: AdamW multiplies the decay by the
# learning rate, so at these values it was already a gentle force, perhaps a
# ten-thousandth of the learning step. Turning it down is right, and it is not
# what will move the result.
WEIGHT_DECAY = 1e-5
# A ceiling, not a target. The run is meant to end for a reason that says
# something - the validation loss stopped improving, or the clock ran out -
# and not because an arbitrary number was reached. Set high enough that it
# never binds: at roughly 85 seconds an epoch - 128 wide and recomputing - the
# 10.5-hour budget stops the run around epoch 440 long before this does.
MAX_EPOCHS = 3000
GRAD_CLIP = 1.0

# How much extra a high-stress node counts for in the loss.
#
# Plain MSE averages over every node equally, and here that is a problem
# rather than a detail. A bracket has around 36,000 nodes, and the stress
# concentration - the only part anyone designs against - is a few dozen of
# them. Their share of the average is under a percent, so the cheapest way for
# the network to lower the loss is to predict the bulk field well and give up
# on the peaks entirely.
#
# That is not a hypothetical. The first full run predicted a peak near 900 MPa
# on every bracket in the dataset, whether its real peak was 630 or 7,800.
#
# The target is standardised log-stress, so a node this many spreads above
# average counts (1 + PEAK_WEIGHT x spreads) times. Below-average nodes stay
# at 1. Setting it to 0 gives back plain MSE exactly.
PEAK_WEIGHT = 2.0

# Which nodes get reported separately as "the peak" when scoring. Reported
# only - it has no effect on training. 0.99 is the hottest 1% of each bracket,
# measured per bracket rather than pooled so a mild bracket beside a severe
# one still contributes its own worst nodes.
PEAK_QUANTILE = 0.99

# Stop when the validation loss has not improved for this many epochs. Without
# it a network keeps improving on what it has seen long after it has stopped
# improving on what it has not.
# Raised from 30 with the learning rate drop: smaller steps mean smaller
# per-epoch improvements, and 30 was already stopping runs on the noise in a
# bouncing validation curve rather than on a real plateau.
PATIENCE = 40
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
