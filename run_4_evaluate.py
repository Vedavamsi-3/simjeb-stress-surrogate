"""Stage 4 - score the trained model on the test brackets.

    python run_4_evaluate.py --run quick
    python run_4_evaluate.py --run full

RUN THIS ONCE, AT THE END.

Every look at the test set during development turns it into a second
validation set. Nobody does it deliberately - you check the test score, see it
is worse than you hoped, change the dropout, check again. Three rounds of that
and the test set has quietly become the thing you tuned against, and it no
longer tells you anything about a bracket nobody has seen.

That is why this is a separate script rather than the last section of the
training one, and why it prints a warning if it is run more than once.

What it reports, and why each part
----------------------------------
**MAE in MPa**, not loss. A loss in scaled log-space is not a number anyone can
act on. MAE in MPa is: it is the average amount the prediction is out by.

**The median bracket alongside the mean.** One bracket with a stress
singularity can move the mean by more than any change to the model would.

**Per bracket**, so a surprising average can be traced to its cause.

**A trivial baseline.** Always guessing the average stress. A model that cannot
beat it has learned nothing, whatever its loss curve looked like.

**R-squared**, because it is scale-free and reads as a percentage to an
engineer who does not work in machine learning - which is most of the audience
for a surrogate model.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))

import config
from simjeb import batching as batching_module
from simjeb import features as features_module
from simjeb import model as model_module
from simjeb import scaling as scaling_module
from simjeb import splits as splits_module
from simjeb import training as training_module


# How many nodes per bracket are kept for the plots. Every node of every
# bracket is millions of numbers - too many to draw, and too many to carry off
# a training machine. A FIXED number per bracket also stops a 90,000-node
# bracket from drowning out a 20,000-node one in the picture.
SAMPLED_NODES_PER_BRACKET = 2000


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="full",
                        help="which run folder to score")
    parser.add_argument("--again", action="store_true",
                        help="score a second time, knowing what that costs")
    return parser.parse_args()


def r_squared(predicted, actual):
    """The share of the variation the model explains.

    Measured against the mean of the SET BEING SCORED, so a model that always
    outputs that mean scores exactly zero. The sign then answers a plain
    question: does this beat knowing nothing at all?
    """
    residual = np.sum((actual - predicted) ** 2)
    total = np.sum((actual - actual.mean()) ** 2)
    if total <= 0:
        return float("nan")
    return float(1.0 - residual / total)


def sample_nodes(n_nodes, rng, keep=SAMPLED_NODES_PER_BRACKET):
    """Which node rows to keep for the plots. All of them if there are few."""
    if n_nodes <= keep:
        return np.arange(n_nodes)
    return rng.choice(n_nodes, size=keep, replace=False)


def score_bracket(model, model_id, scalers, log_target, device):
    """Everything worth knowing about one bracket's predictions."""
    features = features_module.load(config.FEATURE_DIR, model_id)
    prepared = batching_module.prepare(features, scalers)
    batch = batching_module.collate([prepared]).to(device)

    with torch.no_grad():
        scaled_prediction = model.predict(batch)

    predicted = training_module.to_mpa(scaled_prediction, scalers,
                                       log_target).cpu().numpy().ravel()
    actual = batch.stress_mpa.cpu().numpy().ravel()
    error = np.abs(predicted - actual)

    return {
        "model_id": model_id,
        "nodes": len(actual),
        "mae_mpa": float(error.mean()),
        "rmse_mpa": float(np.sqrt((error ** 2).mean())),
        "r_squared": r_squared(predicted, actual),
        "worst_node_mpa": float(error.max()),
        "peak_actual_mpa": float(actual.max()),
        "peak_predicted_mpa": float(predicted.max()),
        "_predicted": predicted,
        "_actual": actual,
    }


def main():
    arguments = parse_arguments()
    run_directory = config.RUN_DIR / arguments.run

    checkpoint = run_directory / "best_model.pt"
    if not checkpoint.is_file():
        print(f"no trained model at {checkpoint}")
        print("run: python run_3_train.py --quick")
        return

    # ---- has the test set been looked at already? -------------------------
    marker = run_directory / "test_scored.json"
    if marker.is_file() and not arguments.again:
        previous = json.loads(marker.read_text())
        print("=" * 70)
        print("THE TEST SET HAS ALREADY BEEN SCORED FOR THIS RUN")
        print("=" * 70)
        print(f"  scored before, giving {previous['mae_mpa']:.1f} MPa")
        print("\n  Scoring it again is not free. If anything about the model")
        print("  changed because of what the first score said, the test set is")
        print("  now something you tuned against, and it no longer measures a")
        print("  bracket nobody has seen.")
        print("\n  Pass --again if you know that and want it anyway.")
        return

    settings_file = run_directory / "settings.json"
    settings = json.loads(settings_file.read_text()) if settings_file.is_file() else {}

    split = splits_module.Split.load(config.SPLIT_FILE)
    scalers = scaling_module.Scalers.load(config.SCALING_FILE)

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

    # The model must be rebuilt at exactly the size it was trained at, or the
    # saved weights will not load into it.
    model = model_module.MeshGraphNet(
        node_width=features_module.N_NODE_FEATURES,
        edge_width=features_module.N_EDGE_FEATURES,
        hidden_width=settings.get("HIDDEN_WIDTH", config.HIDDEN_WIDTH),
        message_rounds=settings.get("MESSAGE_ROUNDS", config.MESSAGE_ROUNDS),
        output_width=1,
        dropout=0.0,          # off for scoring, always
    )

    stored = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(stored["model"])
    model = model.to(device)
    model.eval()

    print("=" * 70)
    print(f"EVALUATING {arguments.run}")
    print("=" * 70)
    print(f"  checkpoint : epoch {stored['epoch']}, "
          f"validation loss {stored['val_loss']:.4f}")
    print(f"  network    : {model.hidden_width} wide, "
          f"{model.message_rounds} rounds, {model.n_parameters:,} parameters")
    print(f"  load case  : {config.LOAD_CASE}")
    print(f"  test set   : {len(split.test)} brackets {split.test}")

    if settings.get("NAME") == "quick":
        print("\n  WARNING: this is a QUICK-CHECK model - a fraction of the")
        print("  real size, trained for a handful of epochs. The numbers below")
        print("  show the scoring code works. They are not results.")

    # ---- the trivial baseline, from the training brackets ------------------
    # From TRAINING, not from test: at prediction time the average you would
    # actually guess is the one you learned, not one computed from the answers
    # you are trying to predict.
    def make_loader(model_ids):
        return batching_module.Loader(
            config.FEATURE_DIR, model_ids, scalers, features_module.load,
            batch_size=1, shuffle=False)

    training_mean_mpa = None
    values = []
    for batch in make_loader(split.train):
        values.append(batch.stress_mpa)
    training_stress = torch.cat(values)
    training_mean_mpa = float(training_stress.mean())

    # ---- score every split -------------------------------------------------
    print()
    print("=" * 70)
    print("RESULTS, in MPa")
    print("=" * 70)

    all_rows = []
    summary = {}

    # Node-level predictions, thinned, for run_5_plots.py. Seeded, so the same
    # nodes are kept every time and two runs can be compared node for node.
    sample_rng = np.random.default_rng(0)
    sample = {"model_id": [], "split": [], "actual": [], "predicted": []}

    for name in ("train", "val", "test"):
        members = getattr(split, name)
        if not members:
            continue

        rows = []
        for model_id in members:
            row = score_bracket(model, model_id, scalers,
                                config.LOG_TARGET, device)
            rows.append(row)

            keep = sample_nodes(len(row["_actual"]), sample_rng)
            sample["actual"].append(row["_actual"][keep])
            sample["predicted"].append(row["_predicted"][keep])
            sample["model_id"].append(np.full(len(keep), model_id))
            sample["split"].append(np.full(len(keep), name))

        # Pooled over every node of every bracket in the split - the same
        # convention the real project uses, so the numbers are comparable.
        predicted = np.concatenate([row["_predicted"] for row in rows])
        actual = np.concatenate([row["_actual"] for row in rows])
        error = np.abs(predicted - actual)

        baseline_error = np.abs(actual - training_mean_mpa).mean()
        per_bracket_mae = [row["mae_mpa"] for row in rows]

        summary[name] = {
            "brackets": len(rows),
            "nodes": len(actual),
            "mae_mpa": float(error.mean()),
            "median_bracket_mae_mpa": float(np.median(per_bracket_mae)),
            "rmse_mpa": float(np.sqrt((error ** 2).mean())),
            "r_squared": r_squared(predicted, actual),
            "trivial_baseline_mae_mpa": float(baseline_error),
        }

        for row in rows:
            record = {key: value for key, value in row.items()
                      if not key.startswith("_")}
            record["split"] = name
            all_rows.append(record)

    header = f"{'split':<8}{'MAE':>9}{'median':>9}{'RMSE':>9}{'R2':>8}{'baseline':>11}"
    print(header)
    for name, scores in summary.items():
        print(f"{name:<8}{scores['mae_mpa']:>9.1f}"
              f"{scores['median_bracket_mae_mpa']:>9.1f}"
              f"{scores['rmse_mpa']:>9.1f}"
              f"{scores['r_squared']:>8.3f}"
              f"{scores['trivial_baseline_mae_mpa']:>11.1f}")

    # ---- per bracket -------------------------------------------------------
    print()
    print("=" * 70)
    print("PER BRACKET")
    print("=" * 70)
    table = pd.DataFrame(all_rows).set_index("model_id")
    columns = ["split", "nodes", "mae_mpa", "rmse_mpa", "r_squared",
               "peak_actual_mpa", "peak_predicted_mpa"]
    print(table[columns].to_string(float_format=lambda v: f"{v:,.2f}"))

    # ---- how to read it ----------------------------------------------------
    print()
    print("=" * 70)
    print("READING THIS")
    print("=" * 70)

    test = summary.get("test")
    if test:
        beat_by = 100 * (1 - test["mae_mpa"] / test["trivial_baseline_mae_mpa"])
        print(f"  test MAE {test['mae_mpa']:.1f} MPa against a trivial "
              f"baseline of {test['trivial_baseline_mae_mpa']:.1f}")
        print(f"  -> {beat_by:+.0f}% versus always guessing the average")

        if test["rmse_mpa"] > 2 * test["mae_mpa"]:
            print(f"\n  RMSE ({test['rmse_mpa']:.0f}) is more than double MAE "
                  f"({test['mae_mpa']:.0f}). That means a small number of nodes")
            print("  carry most of the error - the stress singularities at")
            print("  sharp corners, which the log transform compresses but")
            print("  does not tame.")

        if len(split.test) < 10:
            print(f"\n  Only {len(split.test)} test bracket(s). This is a check")
            print("  that the scoring code works, not a measurement of the")
            print("  model. A test set this small reports which brackets landed")
            print("  in it.")

    # ---- write it down -----------------------------------------------------
    table[columns].to_csv(run_directory / "per_bracket.csv")

    predictions_path = run_directory / "predictions.npz"
    np.savez_compressed(
        predictions_path,
        actual=np.concatenate(sample["actual"]).astype(np.float32),
        predicted=np.concatenate(sample["predicted"]).astype(np.float32),
        model_id=np.concatenate(sample["model_id"]).astype(np.int32),
        split=np.concatenate(sample["split"]),
    )
    (run_directory / "results.json").write_text(json.dumps({
        "run": arguments.run,
        "checkpoint_epoch": stored["epoch"],
        "load_case": config.LOAD_CASE,
        "settings": settings,
        "split": {"train": split.train, "val": split.val, "test": split.test},
        "scores": summary,
    }, indent=2))

    if test:
        marker.write_text(json.dumps({"mae_mpa": test["mae_mpa"],
                                      "epoch": stored["epoch"]}))

    print()
    size_mb = predictions_path.stat().st_size / 1e6
    print(f"wrote {run_directory.name}/: results.json, per_bracket.csv, "
          f"predictions.npz ({size_mb:.1f} MB)")
    print()
    print("next: python run_5_plots.py")


if __name__ == "__main__":
    main()
