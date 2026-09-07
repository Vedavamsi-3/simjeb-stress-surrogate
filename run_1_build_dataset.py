"""Stage 1 - turn the raw SimJEB files into graphs and features.

Run once per dataset. Everything after this reads the cached output and never
touches the raw files again, which is what makes experimenting cheap: the raw
data for 381 brackets is over 20 GB unpacked, the cache is about 4 GB and loads
in seconds.

    python run_1_build_dataset.py

Restartable. Brackets already built are skipped, so a run that dies part-way
resumes rather than starting over. Pass --rebuild to force it.

Two passes over the brackets, and the second one is the point:

  pass 1  build every bracket, and record where its bolt holes ended up
  pass 2  reject the ones whose interfaces sit far from where everyone
          else's do - those brackets are rotated

The check cannot happen in pass 1 because "where everyone else's are" is not
known until everyone else has been built.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import config
from simjeb import features as features_module
from simjeb import graph as graph_module
from simjeb import splits as splits_module


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true",
                        help="rebuild brackets that are already cached")
    parser.add_argument("--limit", type=int, default=None,
                        help="stop after this many brackets, for a quick check")
    return parser.parse_args()


def build_one(index, model_id, reference_landmarks=None):
    """Build and save one bracket. Returns what happened, never raises."""
    graph = graph_module.build(
        index,
        model_id,
        load_case=config.LOAD_CASE,
        coordinate_tolerance_mm=config.MAX_COORDINATE_MISMATCH_MM,
        landmark_tolerance_mm=config.MAX_LANDMARK_OFFSET_MM,
        reference_landmarks=reference_landmarks,
    )
    graph_module.save(graph, config.GRAPH_DIR)

    features = features_module.build(graph, log_target=config.LOG_TARGET)
    features_module.save(features, config.FEATURE_DIR)

    return graph


def already_built(model_id):
    graph_file = config.GRAPH_DIR / f"{model_id}.npz"
    feature_file = config.FEATURE_DIR / f"{model_id}.npz"
    return graph_file.is_file() and feature_file.is_file()


def main():
    arguments = parse_arguments()
    config.make_directories()

    print(config.describe())
    print()

    directories = config.data_directories()
    for kind, folder in directories.items():
        print(f"  {kind:<4} {folder}")
    print()

    # One directory listing for the whole run, not one per bracket.
    index = graph_module.build_index(directories)
    model_ids = graph_module.find_brackets(directories)
    print(f"{len(model_ids)} brackets have all three files on disk")

    # A bracket on disk but missing from the metadata was removed during
    # data cleaning - a corrupted row, or one an outlier rule rejected.
    # Building it would waste several minutes and then fail at the split
    # stage, where every bracket is looked up by id. So the metadata
    # decides what exists.
    if config.METADATA_IS_AUTHORITATIVE:
        metadata = splits_module.read_metadata(config.METADATA_FILE)
        known = set(metadata.index)

        allowed = []
        missing = []
        for model_id in model_ids:
            if model_id in known:
                allowed.append(model_id)
            else:
                missing.append(model_id)
        model_ids = allowed

        print(f"{len(metadata)} brackets listed in "
              f"{Path(config.METADATA_FILE).name}")
        if missing:
            print(f"{len(missing)} on disk but not in the metadata, "
                  f"skipped: {missing}")

    if arguments.limit:
        model_ids = model_ids[:arguments.limit]
    print(f"{len(model_ids)} brackets to build")


    # ---- pass 1 ------------------------------------------------------------
    print("pass 1: building")
    started = time.time()

    built = {}
    excluded = {}

    for model_id in model_ids:
        if already_built(model_id) and not arguments.rebuild:
            graph = graph_module.load(config.GRAPH_DIR, model_id)
            built[model_id] = graph.landmarks
            print(f"  {model_id:>4}  already cached")
            continue

        try:
            graph = build_one(index, model_id)
            built[model_id] = graph.landmarks
            print(f"  {model_id:>4}  {graph.summary()}")

        except Exception as error:      # noqa: BLE001 - recorded, not raised
            # One unusable bracket must not end a run of 381. It is recorded
            # with its reason and the loop carries on. A pipeline that silently
            # drops samples is how a dataset quietly becomes something other
            # than what you think you are training on.
            excluded[model_id] = f"{type(error).__name__}: {error}"
            print(f"  {model_id:>4}  EXCLUDED: {error}")

    print(f"\nbuilt {len(built)} of {len(model_ids)} "
          f"in {time.time() - started:.0f}s")

    if not built:
        print("nothing was built; stopping")
        return

    # ---- pass 2 ------------------------------------------------------------
    # Where do the bolt holes and the lug sit in a typical bracket? The median
    # across everything built, rather than the mean: a couple of badly rotated
    # brackets would drag a mean towards themselves and soften the very test
    # they should fail.
    print("\npass 2: checking every bracket against the typical frame")

    all_landmarks = np.stack(list(built.values()))
    reference = np.median(all_landmarks, axis=0)

    names = ["bolt hole 1", "bolt hole 2", "bolt hole 3", "bolt hole 4",
             "load lug"]
    print("  typical interface positions, mm:")
    for name, point in zip(names, reference):
        print(f"    {name:<12} ({point[0]:8.2f}, {point[1]:8.2f}, "
              f"{point[2]:8.2f})")

    offsets = {}
    rotated = []

    for model_id, landmarks in built.items():
        offset = float(np.abs(landmarks - reference).max())
        offsets[model_id] = offset
        if offset > config.MAX_LANDMARK_OFFSET_MM:
            rotated.append(model_id)

    values = np.array(list(offsets.values()))
    print(f"\n  offset from typical: median {np.median(values):.2f} mm, "
          f"worst {values.max():.2f} mm")
    print(f"  tolerance          : {config.MAX_LANDMARK_OFFSET_MM} mm")

    for model_id in rotated:
        # The load is a fixed world direction that would not turn with the
        # part, so a rotated bracket carries force along a different internal
        # path. It is a different physical problem, not a different view - and
        # straightening it would leave a tidy mesh whose answers came from
        # somewhere else.
        excluded[model_id] = (
            f"rotated: interfaces sit up to {offsets[model_id]:.1f} mm from "
            f"the typical frame")
        for directory in (config.GRAPH_DIR, config.FEATURE_DIR):
            path = directory / f"{model_id}.npz"
            if path.is_file():
                path.unlink()
        del built[model_id]
        print(f"  {model_id:>4}  EXCLUDED: {excluded[model_id]}")

    if not rotated:
        print("  no bracket is rotated beyond the tolerance")

    # ---- the record --------------------------------------------------------
    rows = []
    for model_id, reason in sorted(excluded.items()):
        rows.append({"model_id": model_id, "reason": reason})

    report_path = config.REPORT_DIR / "excluded.csv"
    pd.DataFrame(rows, columns=["model_id", "reason"]).to_csv(report_path,
                                                              index=False)

    offset_rows = []
    for model_id in sorted(offsets):
        offset_rows.append({"model_id": model_id,
                            "offset_mm": round(offsets[model_id], 3),
                            "kept": model_id in built})
    pd.DataFrame(offset_rows).to_csv(config.REPORT_DIR / "alignment.csv",
                                     index=False)

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"  usable brackets : {len(built)}")
    print(f"  excluded        : {len(excluded)}")
    print(f"  graphs          : {config.GRAPH_DIR}")
    print(f"  features        : {config.FEATURE_DIR}")
    print(f"  reasons         : {report_path}")

    total_mb = 0.0
    for directory in (config.GRAPH_DIR, config.FEATURE_DIR):
        for path in directory.glob("*.npz"):
            total_mb += path.stat().st_size / 1e6
    print(f"  cache size      : {total_mb:.0f} MB "
          f"({total_mb / max(len(built), 1):.1f} MB per bracket)")

    print("\nnext: python run_2_make_splits.py")


if __name__ == "__main__":
    main()
