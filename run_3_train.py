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

    BATCH_SIZE = 1              # the smallest memory footprint there is
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = config.WEIGHT_DECAY
    MAX_EPOCHS = 3
    GRAD_CLIP = config.GRAD_CLIP

    PATIENCE = 100              # high, so it never stops early in 3 epochs
    MIN_IMPROVEMENT = config.MIN_IMPROVEMENT
    MAX_HOURS = 0.5

    LOG_TARGET = config.LOG_TARGET
    DEVICE = "cpu"              # a tiny model gains nothing from a GPU
    SEED = config.SEED


class RealSettings:
    """Everything straight from config.py."""

    NAME = "full"

    HIDDEN_WIDTH = config.HIDDEN_WIDTH
    MESSAGE_ROUNDS = config.MESSAGE_ROUNDS
    DROPOUT = config.DROPOUT

    BATCH_SIZE = config.BATCH_SIZE
    LEARNING_RATE = config.LEARNING_RATE
    WEIGHT_DECAY = config.WEIGHT_DECAY
    MAX_EPOCHS = config.MAX_EPOCHS
    GRAD_CLIP = config.GRAD_CLIP

    PATIENCE = config.PATIENCE
    MIN_IMPROVEMENT = config.MIN_IMPROVEMENT
    MAX_HOURS = config.MAX_HOURS

    LOG_TARGET = config.LOG_TARGET
    DEVICE = config.DEVICE
    SEED = config.SEED


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="tiny network, 3 epochs: checks the code runs")
    parser.add_argument("--name", default=None,
                        help="name for the run folder (default: the mode)")
    parser.add_argument("--fresh", action="store_true",
                        help="ignore any existing checkpoint and start over")
    return parser.parse_args()


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
    return values


def main():
    arguments = parse_arguments()
    settings = QuickSettings() if arguments.quick else RealSettings()

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

    # ---- say clearly which run this is ------------------------------------
    print("=" * 70)
    if arguments.quick:
        print("QUICK CHECK - tiny network, 3 epochs. Not a result.")
    else:
        print("FULL RUN")
    print("=" * 70)
    print(f"  run folder : {run_directory}")
    print(f"  network    : {settings.HIDDEN_WIDTH} wide, "
          f"{settings.MESSAGE_ROUNDS} message rounds, "
          f"dropout {settings.DROPOUT}")
    print(f"  training   : batch {settings.BATCH_SIZE}, "
          f"lr {settings.LEARNING_RATE}, weight decay {settings.WEIGHT_DECAY}")
    print(f"  stopping   : up to {settings.MAX_EPOCHS} epochs, "
          f"patience {settings.PATIENCE}, budget {settings.MAX_HOURS} h")
    print(f"  brackets   : {len(split.train)} train, {len(split.val)} val, "
          f"{len(split.test)} test")

    (run_directory / "settings.json").write_text(
        json.dumps(settings_as_dict(settings), indent=2))

    # ---- the loaders -------------------------------------------------------
    def make_loader(model_ids, shuffle):
        return batching_module.Loader(
            config.FEATURE_DIR, model_ids, scalers, features_module.load,
            batch_size=settings.BATCH_SIZE, shuffle=shuffle,
            seed=settings.SEED)

    train_loader = make_loader(split.train, shuffle=True)
    val_loader = make_loader(split.val, shuffle=False)

    # ---- the number to beat ------------------------------------------------
    # Worked out before training, from the training brackets. A model that
    # cannot beat "always guess the average" has learned nothing, whatever its
    # loss curve looks like.
    baseline = training_module.trivial_baseline_mpa(
        make_loader(split.train, shuffle=False))
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
    )
    print(f"  parameters : {model.n_parameters:,}")

    memory = model_module.estimate_memory_gb(
        n_nodes=60_000 * settings.BATCH_SIZE,
        n_edges=380_000 * settings.BATCH_SIZE,
        hidden_width=settings.HIDDEN_WIDTH,
        message_rounds=settings.MESSAGE_ROUNDS,
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
    for name in ("train", "val"):
        members = getattr(split, name)
        scores = training_module.evaluate(
            model, make_loader(members, shuffle=False), scalers,
            settings.LOG_TARGET, device)
        detail = "  ".join(f"{k}:{v:,.0f}"
                           for k, v in sorted(scores["per_bracket"].items()))
        print(f"  {name:<6}{scores['mae_mpa']:>8.1f} MPa  "
              f"(median bracket {scores['median_bracket_mae']:,.0f})   {detail}")

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
