"""Decide which brackets the model may learn from, and which it must not.

A split is a claim: "how well it does on these held-out brackets predicts how
well it will do on a design nobody has drawn yet." Splitting at random does not
make that claim true, it assumes it. This module tries to make it true, and
then measures whether it worked.

Two things get in the way of a random split, and both are handled here.

**Near-copies.** Designers submitted several variants of one entry. Put a
variant in train and its twin in test and the model has effectively seen the
test bracket - the score comes out better than it should by an amount nobody
can measure afterwards. So brackets sharing a submission are grouped, and whole
groups are assigned together.

**Uneven shapes.** SimJEB labels each bracket block / beam / butterfly and so
on. Let those fall where they may and a whole category can end up entirely in
one split, so the test set is measuring a different problem from the one that
was trained.

Run on its own::

    python -m simjeb.splits
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# Columns the metadata table quotes.
QUOTED_COLUMNS = ("category", "author", "author_id", "link_name")


@dataclass
class Split:
    """One frozen assignment of brackets to train, validation and test."""

    train: list
    val: list
    test: list
    seed: int
    grouped_by: str
    stratified_by: str
    verification: dict = field(default_factory=dict)

    @property
    def all_ids(self):
        return sorted(self.train + self.val + self.test)

    def which(self, model_id):
        """Which split a bracket belongs to."""
        if model_id in set(self.train):
            return "train"
        if model_id in set(self.val):
            return "val"
        if model_id in set(self.test):
            return "test"
        raise KeyError(f"bracket {model_id} is not in this split")

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))
        return path

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))


def read_metadata(path):
    """Load the metadata table, whichever way it is separated.

    SimJEB ships a tab-separated ``.tab``; a table that has been through a
    cleaning step is usually a comma-separated ``.csv``. Rather than making
    the caller remember which, the separator is taken from the extension.

    ``sep=None`` would let pandas sniff it, but it guesses - and a wrong
    guess reads every row as a single column, which fails much later and
    far more confusingly than a wrong extension would.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"no metadata table at {path}. config.METADATA_FILE decides "
            f"which brackets exist, so nothing can run without it.")

    separator = "\t" if path.suffix.lower() == ".tab" else ","
    table = pd.read_csv(path, sep=separator)

    if "id" not in table.columns:
        raise ValueError(
            f"{path.name} has no 'id' column - it may have been read with "
            f"the wrong separator. Columns found: {list(table.columns)[:5]}")

    # SimJEB's own table quotes its text fields; a cleaned one usually does
    # not. Stripping quotes from unquoted text does nothing, so this is
    # safe either way.
    for column in QUOTED_COLUMNS:
        if column in table.columns and table[column].dtype == object:
            table[column] = table[column].str.strip('"')

    return table.set_index("id", drop=False)


# ---------------------------------------------------------------------------
# FINDING THE NEAR-COPIES
# ---------------------------------------------------------------------------

def leakage_groups(metadata, model_ids, use_author=False):
    """Group brackets that must never be separated across splits.

    Returns ``{model_id: group_id}``. A bracket with no relatives is a group of
    one, which is the common case.

    ``link_name`` is the GrabCAD submission and is definitive: two models
    sharing one are variants of a single entry.

    ``use_author`` reaches much further - a designer who submitted several
    entries has them all merged - but it over-groups, because the same person
    often submitted genuinely different designs. Off by default, available as a
    stricter variant to report alongside.
    """
    signals = ["link_name"]
    if use_author:
        signals.append("author_id")

    # Start with every bracket in its own group, then merge.
    group_of = {}
    for model_id in model_ids:
        group_of[model_id] = model_id

    for signal in signals:
        seen_value = {}
        for model_id in model_ids:
            value = metadata.loc[model_id, signal]
            if value in seen_value:
                merge(group_of, model_id, seen_value[value])
            else:
                seen_value[value] = model_id

    # Renumber so the group ids are 0, 1, 2, ... which reads better in reports.
    renumbered = {}
    next_number = 0
    result = {}
    for model_id in sorted(model_ids):
        root = find_root(group_of, model_id)
        if root not in renumbered:
            renumbered[root] = next_number
            next_number += 1
        result[model_id] = renumbered[root]

    return result


def find_root(group_of, model_id):
    """Follow the chain of merges to the group's representative."""
    while group_of[model_id] != model_id:
        model_id = group_of[model_id]
    return model_id


def merge(group_of, first, second):
    """Put two brackets in the same group."""
    root_a = find_root(group_of, first)
    root_b = find_root(group_of, second)
    if root_a != root_b:
        group_of[max(root_a, root_b)] = min(root_a, root_b)


# ---------------------------------------------------------------------------
# BUILDING THE SPLIT
# ---------------------------------------------------------------------------

def make(metadata, model_ids, test_fraction=0.2, val_fraction=0.1, seed=0,
         group_by_submission=True, stratify_by_category=True):
    """Assign every bracket to train, validation or test.

    Whole groups move together, and each shape category is dealt out
    separately, so both properties hold at once.

    The method is a greedy fill: within a category, take the groups largest
    first and give each to whichever split is furthest below its target share.
    Largest first matters - place the big groups while there is still room to
    balance around them, and the small ones tidy up the remainder.
    """
    rng = np.random.default_rng(seed)

    if group_by_submission:
        group_of = leakage_groups(metadata, model_ids)
    else:
        group_of = {model_id: model_id for model_id in model_ids}

    # Collect the brackets of each group together.
    members_of_group = {}
    for model_id in model_ids:
        group = group_of[model_id]
        members_of_group.setdefault(group, []).append(model_id)

    if stratify_by_category:
        category_of_group = {}
        for group, members in members_of_group.items():
            # A group shares a submission, so its members share a category too;
            # the first is representative.
            category_of_group[group] = metadata.loc[members[0], "category"]
    else:
        category_of_group = {group: "all" for group in members_of_group}

    targets = {
        "train": 1.0 - test_fraction - val_fraction,
        "val": val_fraction,
        "test": test_fraction,
    }
    split = {"train": [], "val": [], "test": []}
    counts = {"train": 0, "val": 0, "test": 0}

    categories = sorted(set(category_of_group.values()))
    for category in categories:
        groups_here = []
        for group, group_category in category_of_group.items():
            if group_category == category:
                groups_here.append(group)

        groups_here = order_groups(groups_here, members_of_group, rng)

        for group in groups_here:
            members = members_of_group[group]
            destination = neediest_split(counts, targets)
            split[destination].extend(members)
            counts[destination] += len(members)

    for name in split:
        split[name].sort()

    result = Split(
        train=split["train"],
        val=split["val"],
        test=split["test"],
        seed=seed,
        grouped_by="link_name" if group_by_submission else "none",
        stratified_by="category" if stratify_by_category else "none",
    )
    result.verification = verify(result, metadata, group_of)
    return result


def order_groups(groups, members_of_group, rng):
    """Largest group first, with ties broken randomly.

    Randomising the ties is what makes the seed meaningful: without it the
    order would depend on dictionary insertion order, and changing the seed
    would change nothing.
    """
    shuffled = list(groups)
    rng.shuffle(shuffled)
    shuffled.sort(key=lambda group: len(members_of_group[group]), reverse=True)
    return shuffled


def neediest_split(counts, targets):
    """Whichever split is furthest below the share it is supposed to have.

    Measured as a shortfall in brackets rather than as a ratio, so an empty
    split is always the neediest and can never be skipped. That is the bug this
    replaces: an earlier version dealt round-robin and, with fewer categories
    than positions in its pattern, silently produced an empty test set.
    """
    total = sum(counts.values())
    shortfalls = {}
    for name, target in targets.items():
        deserved = target * (total + 1)
        shortfalls[name] = deserved - counts[name]

    return max(shortfalls, key=shortfalls.get)


# ---------------------------------------------------------------------------
# MEASURING WHETHER IT WORKED
# ---------------------------------------------------------------------------

def verify(split, metadata, group_of):
    """Check the split has the properties it was built to have.

    Every step above is a heuristic. This is what turns "I grouped it" into
    "here are the numbers", and it is the part worth reading before trusting
    any score.
    """
    where = {}
    for name in ("train", "val", "test"):
        for model_id in getattr(split, name):
            where[model_id] = name

    # 1. Does any group appear in more than one split?
    splits_of_group = {}
    for model_id, group in group_of.items():
        if model_id in where:
            splits_of_group.setdefault(group, set()).add(where[model_id])

    straddling = []
    for group, names in splits_of_group.items():
        if len(names) > 1:
            straddling.append({"group": int(group), "splits": sorted(names)})

    # 2. Is each shape category represented in proportion?
    shares = {}
    for name in ("train", "val", "test"):
        members = getattr(split, name)
        if not members:
            shares[name] = {}
            continue
        categories = metadata.loc[members, "category"]
        counted = categories.value_counts(normalize=True)
        shares[name] = {str(k): round(float(v), 3) for k, v in counted.items()}

    sizes = {name: len(getattr(split, name))
             for name in ("train", "val", "test")}

    return {
        "sizes": sizes,
        "groups_straddling": len(straddling),
        "straddling_detail": straddling[:10],
        "category_shares": shares,
        "empty_splits": [name for name, n in sizes.items() if n == 0],
    }


def report(split, metadata):
    """A readable summary of what the split looks like."""
    lines = []
    for name in ("train", "val", "test"):
        members = getattr(split, name)
        categories = [metadata.loc[m, "category"] for m in members]
        lines.append(f"  {name:<6} {len(members):>4} brackets  {members}")
        lines.append(f"         {categories}")

    check = split.verification
    lines.append("")
    lines.append(f"  design families straddling a boundary : "
                 f"{check['groups_straddling']}")
    lines.append(f"  empty splits                          : "
                 f"{check['empty_splits'] or 'none'}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import config
    from simjeb import graph as graph_module

    metadata = read_metadata(config.METADATA_FILE)
    model_ids = graph_module.find_brackets(config.data_directories())
    print(f"{len(model_ids)} brackets: {model_ids}\n")

    groups = leakage_groups(metadata, model_ids)
    sizes = {}
    for group in groups.values():
        sizes[group] = sizes.get(group, 0) + 1

    print("design families found by shared submission:")
    print(f"  {len(sizes)} families across {len(model_ids)} brackets")
    biggest = max(sizes.values())
    print(f"  largest family: {biggest} bracket(s)")
    if biggest == 1:
        print("  every bracket here is its own design - so grouping changes")
        print("  nothing for these 8. On the full 381 it changes a great deal:")
        print("  52 models share a submission with another.")

    split = make(
        metadata, model_ids,
        test_fraction=config.TEST_FRACTION,
        val_fraction=config.VAL_FRACTION,
        seed=config.SPLIT_SEED,
        group_by_submission=config.GROUP_BY_SUBMISSION,
        stratify_by_category=config.STRATIFY_BY_CATEGORY,
    )

    print(f"\nsplit, targeting "
          f"{1 - config.TEST_FRACTION - config.VAL_FRACTION:.0%}/"
          f"{config.VAL_FRACTION:.0%}/{config.TEST_FRACTION:.0%}:")
    print(report(split, metadata))

    assert not split.verification["empty_splits"], "a split came out empty"
    assert split.verification["groups_straddling"] == 0, "a family straddles"

    path = split.save(config.SPLIT_FILE)
    print(f"\nsaved {path.name}")
