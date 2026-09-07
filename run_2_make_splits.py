"""Stage 2 - decide which brackets the model may see, and fit the scaling.

    python run_2_make_splits.py

Fast: it reads the cached features, not the raw files. Run it again with a
different seed in config.py to get a different split, and everything downstream
picks it up.

The order inside this script is the whole point of it. The split is decided
first, then the scaling is computed from the training brackets ALONE. Doing it
the other way round - or computing the scaling over everything - folds
information about the held-out brackets into the numbers every input passes
through. Nothing breaks, no error appears, and the final score comes out
slightly better than it should by an amount nobody can measure afterwards.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import config
from simjeb import features as features_module
from simjeb import scaling as scaling_module
from simjeb import splits as splits_module


def find_built_brackets():
    """Which brackets stage 1 actually produced.

    Read from the cache rather than from the raw data folder, so brackets that
    stage 1 excluded are absent here without this script needing to know why.
    """
    found = []
    for path in sorted(config.FEATURE_DIR.glob("*.npz")):
        if path.stem.isdigit():
            found.append(int(path.stem))
    return sorted(found)


def main():
    config.make_directories()

    model_ids = find_built_brackets()
    if not model_ids:
        print(f"no features found in {config.FEATURE_DIR}")
        print("run: python run_1_build_dataset.py")
        return

    metadata = splits_module.read_metadata(config.METADATA_FILE)
    print(f"{len(model_ids)} usable brackets: {model_ids}\n")

    # ---- 1. the design families -------------------------------------------
    print("=" * 70)
    print("1. DESIGN FAMILIES")
    print("=" * 70)

    groups = splits_module.leakage_groups(metadata, model_ids)

    size_of_family = {}
    for group in groups.values():
        size_of_family[group] = size_of_family.get(group, 0) + 1

    families_with_relatives = 0
    brackets_with_relatives = 0
    for size in size_of_family.values():
        if size > 1:
            families_with_relatives += 1
            brackets_with_relatives += size

    print(f"  {len(size_of_family)} families across {len(model_ids)} brackets")
    print(f"  families with more than one member : {families_with_relatives}")
    print(f"  brackets that have a relative      : {brackets_with_relatives}")
    print("\n  Brackets sharing a GrabCAD submission are variants of one")
    print("  design. They are assigned to a split together, so a variant can")
    print("  never sit in train while its twin sits in test.")

    # ---- 2. the split ------------------------------------------------------
    print()
    print("=" * 70)
    print("2. THE SPLIT")
    print("=" * 70)

    train_fraction = 1 - config.TEST_FRACTION - config.VAL_FRACTION
    print(f"  targeting {train_fraction:.0%} / {config.VAL_FRACTION:.0%} / "
          f"{config.TEST_FRACTION:.0%}, seed {config.SPLIT_SEED}\n")

    split = splits_module.make(
        metadata, model_ids,
        test_fraction=config.TEST_FRACTION,
        val_fraction=config.VAL_FRACTION,
        seed=config.SPLIT_SEED,
        group_by_submission=config.GROUP_BY_SUBMISSION,
        stratify_by_category=config.STRATIFY_BY_CATEGORY,
    )
    print(splits_module.report(split, metadata))

    check = split.verification
    if check["empty_splits"]:
        raise SystemExit(
            f"the {check['empty_splits']} split(s) came out empty. "
            f"There are only {len(model_ids)} brackets - too few for these "
            f"fractions. Add more data or change them in config.py.")
    if check["groups_straddling"]:
        raise SystemExit(f"{check['groups_straddling']} design families "
                         f"straddle a split boundary")

    print("\n  shape categories, as a share of each split:")
    for name in ("train", "val", "test"):
        shares = check["category_shares"][name]
        readable = "  ".join(f"{k} {v:.0%}" for k, v in sorted(shares.items()))
        print(f"    {name:<6} {readable}")

    # ---- 3. the scaling ----------------------------------------------------
    print()
    print("=" * 70)
    print("3. SCALING, FROM THE TRAINING BRACKETS ONLY")
    print("=" * 70)

    scalers = scaling_module.fit(config.FEATURE_DIR, split.train,
                                 features_module.load)
    print(f"  fitted on {len(split.train)} brackets\n")
    print(scaling_module.describe(scalers,
                                  features_module.NODE_FEATURE_NAMES,
                                  features_module.EDGE_FEATURE_NAMES))

    # How much a held-out bracket differs from the training average is worth
    # seeing: it is the leak visibly not happening.
    if split.test:
        held_out = split.test[0]
        features = features_module.load(config.FEATURE_DIR, held_out)
        scaled = scalers.node.apply(features.node_features)

        print(f"\n  bracket {held_out} is held out. After scaling, its columns")
        print(f"  average:")
        for index, name in enumerate(features_module.NODE_FEATURE_NAMES):
            print(f"    {name:<22}{scaled[:, index].mean():>8.2f}")
        print("\n  Near zero, but not zero. A held-out bracket landing exactly")
        print("  on zero would mean it had helped compute its own scaling.")

    # ---- 4. what the splits contain ---------------------------------------
    print()
    print("=" * 70)
    print("4. WHAT LANDED WHERE")
    print("=" * 70)

    rows = []
    for name in ("train", "val", "test"):
        for model_id in getattr(split, name):
            features = features_module.load(config.FEATURE_DIR, model_id)
            stress = features.stress_mpa
            rows.append({
                "model_id": model_id,
                "split": name,
                "category": metadata.loc[model_id, "category"],
                "nodes": features.n_nodes,
                "median_mpa": round(float(np.median(stress)), 1),
                "peak_mpa": round(float(stress.max()), 1),
            })

    table = pd.DataFrame(rows).set_index("model_id")
    print(table.to_string())
    table.to_csv(config.REPORT_DIR / "split_contents.csv")

    peaks = table["peak_mpa"]
    hardest = peaks.idxmax()
    print(f"\n  the most extreme bracket is {hardest} at "
          f"{peaks.max():,.0f} MPa, in {table.loc[hardest, 'split'].upper()}")

    if len(split.test) < 10:
        print(f"\n  WARNING: {len(split.test)} bracket(s) in test. With this")
        print("  much variation between brackets, a test set this small is")
        print("  measuring which brackets landed in it, not the model. Treat")
        print("  any score from this run as a check that the code works.")

    split.save(config.SPLIT_FILE)
    scalers.save(config.SCALING_FILE)

    print()
    print(f"saved {config.SPLIT_FILE.name} and {config.SCALING_FILE.name}")
    print("\nnext: python run_3_train.py --quick")


if __name__ == "__main__":
    main()
