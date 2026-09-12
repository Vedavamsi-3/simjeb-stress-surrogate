"""Stage 3 - train the model.

    python run_3_train.py --quick     a few minutes, checks the code runs
    python run_3_train.py             the real thing, from config.py

--quick exists because the two questions are different. "Does this code run
end to end without crashing or filling memory" takes a tiny network and a
handful of epochs. "Is this model any good" takes the full one and a GPU. Doing
the first before the second is how you avoid discovering a typo nine hours in.

The mode is printed at the top of every run and written into the run folder, so
a quick run cannot later be mistaken for a real result.

Resumable. Run the same command again and it carries on from the last epoch,
which is what makes a run longer than a session possible.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

import config
from simjeb import batching as batching_module
from simjeb import features as features_module
from simjeb import model as model_module
from simjeb import scaling as scaling_module
from simjeb import splits as splits_module
from simjeb import training as training_module


class QuickSettings:
    """Minimal everything. For checking the code, not the model.

    A 16-wide network with 2 message rounds has roughly 2% of the real model's
    parameters, and 3 epochs is 3 nudges. Nothing about the numbers it produces
    means anything - but every line of code gets exercised, which is the point.
    """

    NAME = "quick"

    HIDDEN_WIDTH = 16
    MESSAGE_ROUNDS = 2
    DROPOUT = 0.0
    CHECKPOINT_ACTIVATIONS = False   # nothing to save on a 2-block network

    BATCH_SIZE = 1              # the smallest memory footprint there is
    ACCUMULATION_STEPS = config.ACCUMULATION_STEPS
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = config.WEIGHT_DECAY
    MAX_EPOCHS = 3
    GRAD_CLIP = config.GRAD_CLIP

    PATIENCE = 100              # high, so it never stops early in 3 epochs
    MIN_IMPROVEMENT = config.MIN_IMPROVEMENT
    MAX_HOURS = 0.5

    PEAK_WEIGHT = config.PEAK_WEIGHT
    PEAK_QUANTILE = config.PEAK_QUANTILE

    LOG_TARGET = config.LOG_TARGET
    DEVICE = "cpu"              # a tiny model gains nothing from a GPU
    SEED = config.SEED

    LIMIT_TRAIN_BRACKETS = []   # all of them
    LIMIT_VAL_BRACKETS = 0      # all of them


class OverfitSettings:
    """Can this network fit 16 brackets it is shown 1500 times?

    Identical to RealSettings except for the four things that would otherwise
    give a low score an excuse: the dataset is tiny, the epochs are many, both
    regularisers are off, and early stopping is disabled. See THE OVERFIT TEST
    in config.py for what each outcome means.
    """

    NAME = "overfit"

    HIDDEN_WIDTH = config.HIDDEN_WIDTH
    MESSAGE_ROUNDS = config.MESSAGE_ROUNDS
    DROPOUT = 0.0                     # nothing held back
    CHECKPOINT_ACTIVATIONS = config.CHECKPOINT_ACTIVATIONS

    BATCH_SIZE = config.BATCH_SIZE

    # One bracket per update, not eight. Accumulation trades update COUNT for
    # steadier updates, and on 16 brackets that trade goes the wrong way: at 8
    # it would be 2 updates an epoch, so 1500 epochs would be 3,000 updates -
    # fewer than run 2 managed in 235. At 1 it is 24,000.
    ACCUMULATION_STEPS = 1

    LEARNING_RATE = config.LEARNING_RATE
    WEIGHT_DECAY = 0.0                # nothing held back
    MAX_EPOCHS = config.OVERFIT_EPOCHS
    GRAD_CLIP = config.GRAD_CLIP

    # Memorising is the goal here, so stopping when validation stalls would
    # end the run at exactly the point it starts answering the question.
    PATIENCE = 10 ** 9
    MIN_IMPROVEMENT = config.MIN_IMPROVEMENT
    MAX_HOURS = config.MAX_HOURS

    LOG_TARGET = config.LOG_TARGET
    DEVICE = config.DEVICE
    SEED = config.SEED

    PEAK_WEIGHT = config.PEAK_WEIGHT
    PEAK_QUANTILE = config.PEAK_QUANTILE

    LIMIT_TRAIN_BRACKETS = config.OVERFIT_BRACKETS
    LIMIT_VAL_BRACKETS = config.OVERFIT_VAL_BRACKETS


class RealSettings:
    """Everything straight from config.py."""

    NAME = "full"

    HIDDEN_WIDTH = config.HIDDEN_WIDTH
    MESSAGE_ROUNDS = config.MESSAGE_ROUNDS
    DROPOUT = config.DROPOUT
    CHECKPOINT_ACTIVATIONS = config.CHECKPOINT_ACTIVATIONS

    BATCH_SIZE = config.BATCH_SIZE
    ACCUMULATION_STEPS = config.ACCUMULATION_STEPS
    LEARNING_RATE = config.LEARNING_RATE
    WEIGHT_DECAY = config.WEIGHT_DECAY
    MAX_EPOCHS = config.MAX_EPOCHS
    GRAD_CLIP = config.GRAD_CLIP

    PATIENCE = config.PATIENCE
    MIN_IMPROVEMENT = config.MIN_IMPROVEMENT
    MAX_HOURS = config.MAX_HOURS

    PEAK_WEIGHT = config.PEAK_WEIGHT
    PEAK_QUANTILE = config.PEAK_QUANTILE

    LOG_TARGET = config.LOG_TARGET
    DEVICE = config.DEVICE
    SEED = config.SEED

    LIMIT_TRAIN_BRACKETS = []         # all of them
    LIMIT_VAL_BRACKETS = 0            # all of them


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="tiny network, 3 epochs: checks the code runs")
    parser.add_argument("--overfit", action="store_true",
                        help="can it memorise 16 brackets? a diagnostic, "
                             "not a model")
    parser.add_argument("--name", default=None,
                        help="name for the run folder (default: the mode)")
    parser.add_argument("--fresh", action="store_true",
                        help="ignore any existing checkpoint and start over")
    return parser.parse_args()


def current_commit():
    """The git commit the code is sitting on, or "unknown".

    settings.json already records every NUMBER a run used. It does not record
    the CODE, and the two are not the same thing: a loss function can change
    while every setting stays identical, and then two run folders disagree for
    a reason nothing on disk explains.

    Six lines to close that. Kaggle clones the repo rather than pasting it, so
    the commit is available there too - which means every run folder, wherever
    it was produced, points back at exactly the code that produced it.

    Wrapped in try/except because a missing git, or a copy of the code that is
    not a repository at all, must not stop a ten-hour run from starting.
    """
    try:
        finished = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent, capture_output=True, text=True,
            timeout=10, check=True)
        return finished.stdout.strip() or "unknown"
    except Exception:            # noqa: BLE001 - recorded, never raised
        return "unknown"


def settings_as_dict(settings):
    """The settings, for writing into the run folder.

    A run whose settings are not recorded is a result you cannot explain later.
    """
    values = {}
    for name in dir(settings):
        if name.startswith("_"):
            continue
        value = getattr(settings, name)
        if isinstance(value, (int, float, str, bool)):
            values[name] = value

    values["GIT_COMMIT"] = current_commit()
    return values


def main():
    arguments = parse_arguments()
    if arguments.quick:
        settings = QuickSettings()
    elif arguments.overfit:
        settings = OverfitSettings()
    else:
        settings = RealSettings()

    config.make_directories()
    run_name = arguments.name or settings.NAME
    run_directory = config.RUN_DIR / run_name
    run_directory.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(settings.SEED)

    if not config.SPLIT_FILE.is_file():
        print(f"no split found at {config.SPLIT_FILE}")
        print("run: python run_2_make_splits.py")
        return

    split = splits_module.Split.load(config.SPLIT_FILE)
    scalers = scaling_module.Scalers.load(config.SCALING_FILE)

    # Narrowed only by --overfit; every other mode gets the whole split.
    #
    # The training ids are listed explicitly rather than taken as "the first
    # N", because WHICH brackets they are is the entire design of that test -
    # see OVERFIT_BRACKETS in config.py - and a slice of whatever order the
    # split file happens to be in would not be reproducible.
    train_ids = list(settings.LIMIT_TRAIN_BRACKETS) or split.train
    missing = [i for i in train_ids if i not in split.train]
    if missing:
        raise SystemExit(
            f"brackets {missing} are not in the training split. Scoring a "
            f"model on brackets it trained on is the one mistake this "
            f"project cannot make quietly.")

    val_ids = split.val
    if settings.LIMIT_VAL_BRACKETS:
        val_ids = split.val[:settings.LIMIT_VAL_BRACKETS]

    # ---- say clearly which run this is ------------------------------------
    print("=" * 70)
    if arguments.quick:
        print("QUICK CHECK - tiny network, 3 epochs. Not a result.")
    elif arguments.overfit:
        print("OVERFIT TEST - can it memorise 16 brackets? A DIAGNOSTIC.")
        print("The number to read is TRAIN R-squared. Nothing here is a result,")
        print("and the checkpoint it writes is not a model to keep.")
    else:
        print("FULL RUN")
    print("=" * 70)
    print(f"  run folder : {run_directory}")
    print(f"  commit     : {current_commit()}")
    print(f"  network    : {settings.HIDDEN_WIDTH} wide, "
          f"{settings.MESSAGE_ROUNDS} message rounds, "
          f"dropout {settings.DROPOUT}"
          + (", recomputing activations"
             if settings.CHECKPOINT_ACTIVATIONS else ""))
    print(f"  training   : batch {settings.BATCH_SIZE} x "
          f"{settings.ACCUMULATION_STEPS} accumulated, "
          f"lr {settings.LEARNING_RATE}, weight decay {settings.WEIGHT_DECAY}")
    print(f"  stopping   : up to {settings.MAX_EPOCHS} epochs, "
          f"patience {settings.PATIENCE}, budget {settings.MAX_HOURS} h")
    print(f"  brackets   : {len(train_ids)} train, {len(val_ids)} val, "
          f"{len(split.test)} test")
    if settings.LIMIT_TRAIN_BRACKETS:
        print(f"  training on: {sorted(train_ids)}")

    (run_directory / "settings.json").write_text(
        json.dumps(settings_as_dict(settings), indent=2))

    # ---- the loaders -------------------------------------------------------
    def make_loader(model_ids, shuffle):
        return batching_module.Loader(
            config.FEATURE_DIR, model_ids, scalers, features_module.load,
            batch_size=settings.BATCH_SIZE, shuffle=shuffle,
            seed=settings.SEED)

    train_loader = make_loader(train_ids, shuffle=True)
    val_loader = make_loader(val_ids, shuffle=False)

    # ---- the number to beat ------------------------------------------------
    # Worked out before training, from the training brackets. A model that
    # cannot beat "always guess the average" has learned nothing, whatever its
    # loss curve looks like.
    baseline = training_module.trivial_baseline_mpa(
        make_loader(train_ids, shuffle=False))
    print(f"  to beat    : {baseline:,.1f} MPa "
          f"(always guessing the average stress)")

    # ---- the model ---------------------------------------------------------
    model = model_module.MeshGraphNet(
        node_width=features_module.N_NODE_FEATURES,
        edge_width=features_module.N_EDGE_FEATURES,
        hidden_width=settings.HIDDEN_WIDTH,
        message_rounds=settings.MESSAGE_ROUNDS,
        output_width=1,
        dropout=settings.DROPOUT,
        checkpoint_activations=settings.CHECKPOINT_ACTIVATIONS,
    )
    print(f"  parameters : {model.n_parameters:,}")

    memory = model_module.estimate_memory_gb(
        n_nodes=60_000 * settings.BATCH_SIZE,
        n_edges=380_000 * settings.BATCH_SIZE,
        hidden_width=settings.HIDDEN_WIDTH,
        message_rounds=settings.MESSAGE_ROUNDS,
        checkpoint_activations=settings.CHECKPOINT_ACTIVATIONS,
    )
    print(f"  memory     : roughly {memory:.1f} GB per training step")
    print()

    # ---- train -------------------------------------------------------------
    history = training_module.train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        scalers=scalers,
        settings=settings,
        run_directory=run_directory,
        resume=not arguments.fresh,
        baseline_mpa=baseline,
    )

    # ---- how it did on train and validation --------------------------------
    # Test is deliberately NOT scored here. Every look at the test set during
    # development turns it into a second validation set, and model choices
    # slowly bend towards it. run_4_evaluate.py does that once, at the end.
    print()
    print("=" * 70)
    print("SCORES" + ("  (quick check - not results)" if arguments.quick else ""))
    print("=" * 70)

    device = torch.device(settings.DEVICE if torch.cuda.is_available() else "cpu")
    for name, members in (("train", train_ids), ("val", val_ids)):
        scores = training_module.evaluate(
            model, make_loader(members, shuffle=False), scalers,
            settings.LOG_TARGET, device, settings.PEAK_WEIGHT,
            settings.PEAK_QUANTILE)
        detail = "  ".join(f"{k}:{v:,.0f}"
                           for k, v in sorted(scores["per_bracket"].items()))
        print(f"  {name:<6}{scores['mae_mpa']:>8.1f} MPa  "
              f"(median bracket {scores['median_bracket_mae']:,.0f}, "
              f"hottest 1% {scores['peak_mae_mpa']:,.0f})   {detail}")

    print(f"  {'beat?':<6}{baseline:>8.1f} MPa   the trivial baseline")

    print(f"\n  best epoch {history.best_epoch}, "
          f"validation loss {history.best_val_loss:.4f}")
    print(f"  wrote {run_directory.name}/: "
          f"{[p.name for p in sorted(run_directory.iterdir())]}")

    if arguments.quick:
        print("\nThe code runs. Nothing here says anything about the model:")
        print(f"  {model.n_parameters:,} parameters against "
              f"{config.HIDDEN_WIDTH} wide x {config.MESSAGE_ROUNDS} rounds "
              f"in a real run,")
        print(f"  and {settings.MAX_EPOCHS} epochs against up to "
              f"{config.MAX_EPOCHS}.")
        print("\nnext: python run_3_train.py    (on a GPU)")
    else:
        print("\nnext: python run_4_evaluate.py")


if __name__ == "__main__":
    main()
