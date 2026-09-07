"""Stage 5 - draw the run.

    python run_5_plots.py --run full

Reads only what stages 3 and 4 wrote, and writes PNGs. Nothing here loads a
model, a GPU or the raw data, and that is the point: the four files it needs
are small enough to bring home from a training machine, so the pictures can be
redrawn as often as you like without paying for another run.

Five figures, each answering one question:

  loss_curve            did the run behave, and did it beat the baseline?
  predicted_vs_actual   where along the range of stress is the model wrong?
  per_bracket_error     is the average honest, or is one bracket carrying it?
  error_vs_peak_stress  does the error follow the highly stressed brackets?
  error_distribution    are the errors spread, or concentrated in a few nodes?

Every figure is drawn independently. A missing input file skips that one figure
and says so, rather than losing the four that could have been drawn.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")          # write files; never look for a screen

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import config
from simjeb import training as training_module

# One colour per split, fixed here so a split is the same colour in every
# figure. Reading a set of plots is much harder when it is not.
COLOUR = {"train": "tab:grey", "val": "tab:blue", "test": "tab:red"}


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="full",
                        help="which run folder to draw")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# LOADING WHAT THE EARLIER STAGES WROTE
# ---------------------------------------------------------------------------

def load_history(run_directory):
    """Rebuild the training History from the two files it saved.

    Rebuilt rather than redrawn from scratch, so the loss curve here comes out
    of the same function the training loop uses. One drawing routine, one
    appearance, and no chance of the two disagreeing about what "best epoch"
    meant.
    """
    csv_path = run_directory / "history.csv"
    if not csv_path.is_file():
        return None

    records = []
    with open(csv_path, newline="") as handle:
        for row in csv.DictReader(handle):
            records.append(training_module.EpochRecord(
                epoch=int(row["epoch"]),
                train_loss=float(row["train_loss"]),
                val_loss=float(row["val_loss"]),
                val_mae_mpa=float(row["val_mae_mpa"]),
                seconds=float(row["seconds"]),
            ))

    history = training_module.History(records=records)

    summary_path = run_directory / "history.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        history.best_epoch = summary.get("best_epoch", -1)
        history.best_val_loss = summary.get("best_val_loss", float("inf"))
        history.stopped_because = summary.get("stopped_because", "")

    return history


def load_results(run_directory):
    path = run_directory / "results.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def load_per_bracket(run_directory):
    path = run_directory / "per_bracket.csv"
    if not path.is_file():
        return None
    return pd.read_csv(path)


def load_sample(run_directory):
    """The thinned node-level predictions, as a table."""
    path = run_directory / "predictions.npz"
    if not path.is_file():
        return None

    with np.load(path) as stored:
        return pd.DataFrame({
            "model_id": stored["model_id"],
            "split": stored["split"],
            "actual": stored["actual"].astype(np.float64),
            "predicted": stored["predicted"].astype(np.float64),
        })


def splits_present(frame):
    """The splits in the data, in a sensible order rather than alphabetical."""
    found = set(frame["split"])
    return [name for name in ("train", "val", "test") if name in found]


# ---------------------------------------------------------------------------
# 2. PREDICTED AGAINST ACTUAL
# ---------------------------------------------------------------------------

def figure_predicted_vs_actual(sample, results, path):
    """A prediction against its answer, for every sampled node.

    The single most informative plot of a regression model, because a summary
    MAE hides WHERE the error is. Three different failures report the same
    average and look completely different here:

      * a cloud tilted flatter than the diagonal - the model is hedging
        towards the average and under-predicting the peaks, which for a
        stress surrogate is the dangerous direction to be wrong in
      * a cloud tight at low stress and spraying out at high - the common
        case, and the one the log target exists to soften
      * a cloud sitting off the diagonal altogether - a scaling bug, not a
        model problem

    Log axes on both, because the target spans 300 to 15,000 MPa. Hexagonal
    bins rather than dots: a million points overplot into a solid blob, and
    the question worth answering is where the DENSITY is.
    """
    names = splits_present(sample)
    figure, panels = plt.subplots(1, len(names),
                                  figsize=(4.4 * len(names), 4.3),
                                  squeeze=False)

    # One shared range, so the panels can be compared by eye.
    positive = sample[(sample["actual"] > 0) & (sample["predicted"] > 0)]
    low = max(positive["actual"].min(), 1.0)
    high = positive["actual"].max()
    limits = (low * 0.8, high * 1.25)

    dropped = len(sample) - len(positive)

    for column, name in enumerate(names):
        axes = panels[0][column]
        here = positive[positive["split"] == name]

        axes.hexbin(here["actual"], here["predicted"], gridsize=45,
                    xscale="log", yscale="log", mincnt=1, cmap="viridis",
                    bins="log")

        # The diagonal a perfect model would sit on.
        axes.plot(limits, limits, color="red", linewidth=1, linestyle="--",
                  label="perfect")

        axes.set_xlim(limits)
        axes.set_ylim(limits)
        axes.set_xlabel("actual stress, MPa")
        if column == 0:
            axes.set_ylabel("predicted stress, MPa")

        scores = results.get("scores", {}).get(name, {})
        if scores:
            subtitle = (f"{name}  -  MAE {scores['mae_mpa']:,.0f} MPa,  "
                        f"R2 {scores['r_squared']:.2f}")
        else:
            subtitle = f"{name}  -  {len(here):,} nodes"
        axes.set_title(subtitle, fontsize=10)
        axes.grid(alpha=0.25)
        axes.legend(fontsize=8, loc="upper left")

    note = "sampled nodes; colour is how many nodes fall in a bin"
    if dropped:
        note = f"{note}; {dropped:,} non-positive predictions not shown"
    figure.suptitle(f"predicted against actual  -  {note}", fontsize=10)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


# ---------------------------------------------------------------------------
# 3. ONE BAR PER BRACKET
# ---------------------------------------------------------------------------

def figure_per_bracket(table, results, path):
    """Every held-out bracket's own error, sorted worst last.

    A pooled MAE is an average over nodes, so the bracket with the most nodes
    counts for most, and one bracket with a stress singularity can move the
    headline number further than any change to the model would. This is the
    figure that shows whether that happened: a flat run of bars with two
    spikes at the end is a different result from an evenly mediocre one, and
    both report the same mean.
    """
    test = table[table["split"] == "test"].sort_values("mae_mpa")
    if test.empty:
        return None

    figure, axes = plt.subplots(figsize=(max(7, 0.16 * len(test)), 4.3))

    positions = np.arange(len(test))
    axes.bar(positions, test["mae_mpa"], color=COLOUR["test"], width=0.8)

    median = float(test["mae_mpa"].median())
    axes.axhline(median, color="black", linestyle="--", linewidth=1,
                 label=f"median bracket: {median:,.0f} MPa")

    scores = results.get("scores", {}).get("test", {})
    if scores.get("mae_mpa"):
        axes.axhline(scores["mae_mpa"], color="tab:blue", linestyle="-",
                     linewidth=1,
                     label=f"pooled over nodes: {scores['mae_mpa']:,.0f} MPa")
    if scores.get("trivial_baseline_mae_mpa"):
        axes.axhline(scores["trivial_baseline_mae_mpa"], color="grey",
                     linestyle=":", linewidth=1.2,
                     label=f"always guessing the mean: "
                           f"{scores['trivial_baseline_mae_mpa']:,.0f} MPa")

    axes.set_xticks(positions)
    axes.set_xticklabels(test["model_id"], rotation=90, fontsize=6)
    axes.set_xlabel("bracket")
    axes.set_ylabel("mean absolute error, MPa")
    axes.set_title(f"error per held-out bracket  -  "
                   f"{len(test)} in the test set", fontsize=10)
    axes.grid(alpha=0.3, axis="y")
    axes.legend(fontsize=8)

    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


# ---------------------------------------------------------------------------
# 4. ERROR AGAINST HOW HARD THE BRACKET IS
# ---------------------------------------------------------------------------

def figure_error_vs_peak_stress(table, path):
    """Does the model fail on the brackets that matter?

    A surrogate exists to compare designs, so the useful question is not "what
    is the average error" but "is the error worst exactly where the answer
    matters". Peak stress is what an engineer is looking for, so if the points
    climb steeply to the right, the model is least reliable on the brackets
    somebody would actually be worried about.

    Two panels, because they are two different worries. Absolute error says
    how many MPa out it is; error as a share of the peak says whether a large
    absolute error is simply a highly stressed bracket.
    """
    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4.3))

    for name in splits_present(table):
        here = table[table["split"] == name]
        left.scatter(here["peak_actual_mpa"], here["mae_mpa"], s=18,
                     alpha=0.75, color=COLOUR[name], label=name)
        share = 100 * here["mae_mpa"] / here["peak_actual_mpa"]
        right.scatter(here["peak_actual_mpa"], share, s=18, alpha=0.75,
                      color=COLOUR[name], label=name)

    left.set_xscale("log")
    left.set_xlabel("peak actual stress in the bracket, MPa")
    left.set_ylabel("bracket mean absolute error, MPa")
    left.set_title("absolute error", fontsize=10)

    right.set_xscale("log")
    right.set_xlabel("peak actual stress in the bracket, MPa")
    right.set_ylabel("error as a share of the peak, %")
    right.set_title("relative error", fontsize=10)

    for axes in (left, right):
        axes.grid(alpha=0.3)
        axes.legend(fontsize=8)

    figure.suptitle("error against how highly stressed the bracket is",
                    fontsize=10)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


# ---------------------------------------------------------------------------
# 5. THE SHAPE OF THE ERROR
# ---------------------------------------------------------------------------

def figure_error_distribution(sample, table, path):
    """How the error is shared out - over nodes, and over brackets.

    MAE is a mean, and a mean describes a symmetric spread. This error is not
    symmetric anywhere: most nodes are predicted well and a thin tail is
    predicted terribly. That is why RMSE comes out at more than double MAE,
    and it is worth seeing rather than inferring.

    The percentile lines are the useful part. "Half the nodes are within X"
    is a claim an engineer can act on in a way an average is not.
    """
    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4.3))

    # --- node level, test brackets only -------------------------------------
    drawn = False
    if sample is not None:
        test = sample[sample["split"] == "test"]
        error = (test["predicted"] - test["actual"]).abs()
        error = error[error > 0]

        if len(error):
            bins = np.logspace(np.log10(error.min()),
                               np.log10(error.max()), 60)
            left.hist(error, bins=bins, color=COLOUR["test"], alpha=0.85)
            left.set_xscale("log")

            for percentile, style in ((50, "-"), (90, "--"), (99, ":")):
                value = float(np.percentile(error, percentile))
                left.axvline(value, color="black", linestyle=style,
                             linewidth=1,
                             label=f"{percentile}th percentile: "
                                   f"{value:,.0f} MPa")

            left.axvline(float(error.mean()), color="tab:blue", linewidth=1.2,
                         label=f"mean (the MAE): {error.mean():,.0f} MPa")
            left.legend(fontsize=8)
            drawn = True

    if not drawn:
        left.text(0.5, 0.5, "predictions.npz not found", ha="center",
                  va="center", transform=left.transAxes, fontsize=9)

    left.set_xlabel("absolute error at a node, MPa")
    left.set_ylabel("nodes")
    left.set_title("per node, test brackets", fontsize=10)
    left.grid(alpha=0.3)

    # --- bracket level ------------------------------------------------------
    for name in splits_present(table):
        here = table[table["split"] == name]
        right.hist(here["mae_mpa"], bins=25, alpha=0.6, color=COLOUR[name],
                   label=f"{name} ({len(here)})")

    right.set_xlabel("bracket mean absolute error, MPa")
    right.set_ylabel("brackets")
    right.set_title("per bracket", fontsize=10)
    right.grid(alpha=0.3)
    right.legend(fontsize=8)

    figure.suptitle("the shape of the error, not just its average",
                    fontsize=10)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


# ---------------------------------------------------------------------------

def main():
    arguments = parse_arguments()
    run_directory = config.RUN_DIR / arguments.run

    if not run_directory.is_dir():
        print(f"no run folder at {run_directory}")
        print("the files downloaded from training are expected to sit there:")
        print("  history.csv, history.json, per_bracket.csv, results.json,")
        print("  predictions.npz")
        return

    plot_directory = run_directory / "plots"
    plot_directory.mkdir(parents=True, exist_ok=True)

    history = load_history(run_directory)
    results = load_results(run_directory)
    table = load_per_bracket(run_directory)
    sample = load_sample(run_directory)

    print("=" * 70)
    print(f"DRAWING {arguments.run}")
    print("=" * 70)
    if history:
        print(f"  history      : {len(history.records)} epochs")
    else:
        print("  history      : MISSING")
    if table is not None:
        print(f"  per bracket  : {len(table)} brackets")
    else:
        print("  per bracket  : MISSING")
    if sample is not None:
        print(f"  node sample  : {len(sample):,} nodes")
    else:
        print("  node sample  : MISSING")
    print()

    written = []

    if history:
        baseline = (results.get("scores", {}).get("train", {})
                    .get("trivial_baseline_mae_mpa"))
        written.append(training_module.plot_history(
            history, plot_directory / "loss_curve.png", baseline,
            title=arguments.run))
    else:
        print("  skipped loss_curve.png: no history.csv")

    if sample is not None:
        written.append(figure_predicted_vs_actual(
            sample, results, plot_directory / "predicted_vs_actual.png"))
    else:
        print("  skipped predicted_vs_actual.png: no predictions.npz")

    if table is not None:
        written.append(figure_per_bracket(
            table, results, plot_directory / "per_bracket_error.png"))
        written.append(figure_error_vs_peak_stress(
            table, plot_directory / "error_vs_peak_stress.png"))
        written.append(figure_error_distribution(
            sample, table, plot_directory / "error_distribution.png"))
    else:
        print("  skipped three figures: no per_bracket.csv")

    print()
    drawn = [path for path in written if path]
    for path in drawn:
        size_kb = Path(path).stat().st_size / 1000
        print(f"  {Path(path).name:<28}{size_kb:>7.0f} KB")

    print()
    print(f"{len(drawn)} figures in {plot_directory}")


if __name__ == "__main__":
    main()
