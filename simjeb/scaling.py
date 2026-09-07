"""Put every feature on a comparable scale, without leaking the test set.

The features arrive in wildly different units: three columns that are 0 or 1,
three that run -1 to 1, and two measured in millimetres that reach 110. Left
alone, the millimetre columns dominate every calculation by sheer size, and
training is unstable.

The fix is ordinary - subtract the mean, divide by the spread - and the whole
difficulty is in one word: WHICH data the mean and spread are computed from.

    from the training brackets only.

Computing them from everything is the most common quiet mistake in machine
learning. Nothing breaks. No error appears. The reported score simply comes out
a little better than it should, by an amount nobody can measure afterwards -
because information about the held-out brackets has been folded into the
numbers every input passes through.

Run on its own::

    python -m simjeb.scaling
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Below this a column is treated as constant rather than scaled up enormously.
# A column that never varies - the material properties, or a one-hot that
# happens to be all zeros in a small split - would otherwise divide by roughly
# zero and take the whole run down.
MINIMUM_SPREAD = 1e-6


@dataclass
class Scaler:
    """Mean and spread for one array's columns."""

    mean: np.ndarray
    spread: np.ndarray

    def apply(self, values):
        """Raw numbers in, comparable numbers out."""
        return (values - self.mean) / self.spread

    def undo(self, values):
        """The exact opposite, for turning predictions back into MPa."""
        return values * self.spread + self.mean


@dataclass
class Scalers:
    """The three scalers a model needs, kept together so they travel as a set.

    Together rather than separately because a checkpoint is only usable
    alongside the exact numbers its inputs were scaled by. Splitting them up is
    how a model ends up loaded with the wrong ones.
    """

    node: Scaler
    edge: Scaler
    target: Scaler

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            node_mean=self.node.mean, node_spread=self.node.spread,
            edge_mean=self.edge.mean, edge_spread=self.edge.spread,
            target_mean=self.target.mean, target_spread=self.target.spread,
        )
        return path

    @classmethod
    def load(cls, path):
        with np.load(Path(path)) as stored:
            return cls(
                node=Scaler(stored["node_mean"], stored["node_spread"]),
                edge=Scaler(stored["edge_mean"], stored["edge_spread"]),
                target=Scaler(stored["target_mean"], stored["target_spread"]),
            )


class RunningStatistics:
    """Mean and spread of a column, accumulated one bracket at a time.

    Stacking every training bracket into one array and calling ``.mean()``
    would be simpler and works fine for 8 brackets. It does not work for 381:
    that is 14 million nodes, and the array would not fit in memory.

    Sums do fit, and give the same answer. Each bracket contributes its totals
    and is then discarded, so memory stays at one bracket regardless of how
    many there are.
    """

    def __init__(self):
        self.count = 0
        self.total = None
        self.total_of_squares = None

    def add(self, values):
        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 1:
            values = values.reshape(-1, 1)

        if self.total is None:
            n_columns = values.shape[1]
            self.total = np.zeros(n_columns)
            self.total_of_squares = np.zeros(n_columns)

        self.count += len(values)
        self.total += values.sum(axis=0)
        self.total_of_squares += (values ** 2).sum(axis=0)

    def finish(self):
        if self.count == 0:
            raise ValueError("nothing was added; cannot compute a mean")

        mean = self.total / self.count

        # variance = mean of the squares - square of the mean.
        # Clipped at zero: on a column that never varies, floating-point
        # cancellation can push this a hair below zero and the square root
        # would produce a nan.
        variance = self.total_of_squares / self.count - mean ** 2
        spread = np.sqrt(np.maximum(variance, 0.0))

        return Scaler(mean=mean.astype(np.float32),
                      spread=np.maximum(spread, MINIMUM_SPREAD).astype(np.float32))


def fit(feature_directory, model_ids, load_features):
    """Compute the scaling from the given brackets - pass the TRAINING ones.

    ``load_features`` is the loader to use, passed in rather than imported so
    this module stays independent of how features happen to be stored.
    """
    if not model_ids:
        raise ValueError("no brackets to fit the scaling on")

    node_statistics = RunningStatistics()
    edge_statistics = RunningStatistics()
    target_statistics = RunningStatistics()

    for model_id in model_ids:
        features = load_features(feature_directory, model_id)
        node_statistics.add(features.node_features)
        edge_statistics.add(features.edge_features)
        target_statistics.add(features.target)

    return Scalers(
        node=node_statistics.finish(),
        edge=edge_statistics.finish(),
        target=target_statistics.finish(),
    )


def describe(scalers, node_names, edge_names):
    """A readable table of what was fitted."""
    lines = [f"{'column':<22}{'mean':>12}{'spread':>12}"]

    for index, name in enumerate(node_names):
        lines.append(f"{name:<22}{scalers.node.mean[index]:>12.3f}"
                     f"{scalers.node.spread[index]:>12.3f}")

    for index, name in enumerate(edge_names):
        lines.append(f"{name:<22}{scalers.edge.mean[index]:>12.3f}"
                     f"{scalers.edge.spread[index]:>12.3f}")

    lines.append(f"{'target':<22}{scalers.target.mean[0]:>12.3f}"
                 f"{scalers.target.spread[0]:>12.3f}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import config
    from simjeb import features as features_module
    from simjeb import graph as graph_module
    from simjeb import splits as splits_module

    split = splits_module.Split.load(config.SPLIT_FILE)
    print(f"train {split.train}   val {split.val}   test {split.test}\n")

    # Build any features that are not on disk yet, so this can be run alone.
    for model_id in split.all_ids:
        if not (config.FEATURE_DIR / f"{model_id}.npz").is_file():
            built = graph_module.build(
                graph_module.build_index(config.data_directories()),
                model_id,
                config.LOAD_CASE,
                config.MAX_COORDINATE_MISMATCH_MM)
            features_module.save(features_module.build(built,
                                                       config.LOG_TARGET),
                                 config.FEATURE_DIR)
            print(f"  built features for bracket {model_id}")

    scalers = fit(config.FEATURE_DIR, split.train, features_module.load)

    print(f"\nfitted on the {len(split.train)} training brackets:\n")
    print(describe(scalers, features_module.NODE_FEATURE_NAMES,
                   features_module.EDGE_FEATURE_NAMES))

    # What the scaling actually does to a held-out bracket.
    held_out = split.test[0]
    features = features_module.load(config.FEATURE_DIR, held_out)
    scaled = scalers.node.apply(features.node_features)

    print(f"\nbracket {held_out} (held out), before and after scaling:")
    print(f"{'column':<22}{'raw mean':>12}{'raw max':>12}"
          f"{'scaled mean':>14}{'scaled max':>13}")
    for index, name in enumerate(features_module.NODE_FEATURE_NAMES):
        raw = features.node_features[:, index]
        new = scaled[:, index]
        print(f"{name:<22}{raw.mean():>12.2f}{raw.max():>12.2f}"
              f"{new.mean():>14.2f}{new.max():>13.2f}")

    print("\nThe scaled means are near zero but not exactly zero, and that is")
    print("the point: this bracket was not part of the numbers being used to")
    print("scale it. A held-out bracket centring exactly on zero would mean it")
    print("had helped compute its own scaling.")

    # The target scaler has to undo exactly, or every reported score is wrong.
    scaled_target = scalers.target.apply(features.target)
    recovered = scalers.target.undo(scaled_target)
    worst = np.abs(recovered - features.target).max()
    print(f"\nundoing the target scaling recovers the original to {worst:.1e}")

    path = scalers.save(config.SCALING_FILE)
    reloaded = Scalers.load(config.SCALING_FILE)
    same = np.allclose(reloaded.node.mean, scalers.node.mean)
    print(f"saved {path.name}, reloads identical: {same}")
