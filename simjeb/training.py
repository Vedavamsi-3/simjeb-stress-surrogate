"""The training loop, and the scoring that decides when to stop.

The loop itself is five lines. Everything else in this module exists because a
real run is long, the machine it runs on may not survive it, and a network left
alone will happily keep improving on what it has already seen.

Three things it does that a five-line loop does not:

**Stops at the right time.** A network keeps getting better on the training
brackets long after it has stopped getting better on unseen ones. Watching a
held-out score and stopping when it stalls is what separates a model from a
memorised dataset.

**Survives a dead session.** Full state is written every epoch, so a run that
is killed at hour nine resumes at hour nine instead of starting again.

**Reports in MPa.** Everything internal is in scaled log-space. Every number
that leaves this module has been converted back, because a score computed in
log space flatters the model by shrinking exactly the large errors that matter.

Run on its own for a short smoke test::

    python -m simjeb.training
"""

import csv
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch


@dataclass
class EpochRecord:
    """What happened in one pass over the training brackets."""

    epoch: int
    train_loss: float
    val_loss: float
    val_mae_mpa: float
    seconds: float


@dataclass
class History:
    """The record of a run, written to disk every epoch."""

    records: list = field(default_factory=list)
    best_epoch: int = -1
    best_val_loss: float = float("inf")
    epochs_without_improvement: int = 0
    stopped_because: str = ""

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        summary = {
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "epochs_completed": len(self.records),
            "stopped_because": self.stopped_because,
        }
        (directory / "history.json").write_text(json.dumps(summary, indent=2))

        # Rewritten in full every epoch rather than appended, and deliberately
        # not buffered: a session killed at the wall clock still leaves a
        # complete record of everything up to that point.
        if self.records:
            with open(directory / "history.csv", "w", newline="") as handle:
                writer = csv.DictWriter(handle,
                                        fieldnames=list(asdict(self.records[0])))
                writer.writeheader()
                for record in self.records:
                    writer.writerow(asdict(record))


# ---------------------------------------------------------------------------
# SCORING
# ---------------------------------------------------------------------------

def to_mpa(scaled_prediction, scalers, log_target):
    """Undo the scaling, then undo the log. Always report in MPa."""
    in_log_space = scalers.target.undo(scaled_prediction)
    if log_target:
        return torch.expm1(in_log_space)
    return in_log_space


@torch.no_grad()
def evaluate(model, loader, scalers, log_target, device):
    """Loss and error over a set of brackets, pooled and per bracket.

    Per bracket as well as pooled, because a pooled average is dominated by
    whichever bracket has the most nodes and whichever has the largest errors.
    On this data one bracket with a stress singularity can move the mean by
    more than any change to the model would.
    """
    model.eval()

    total_loss = 0.0
    total_absolute_error = 0.0
    total_nodes = 0
    per_bracket = {}

    for batch in loader:
        batch = batch.to(device)
        prediction = model.predict(batch)

        loss = torch.nn.functional.mse_loss(prediction, batch.target)
        total_loss += float(loss) * batch.n_nodes

        predicted_mpa = to_mpa(prediction, scalers, log_target)
        error = (predicted_mpa - batch.stress_mpa).abs()

        total_absolute_error += float(error.sum())
        total_nodes += batch.n_nodes

        for position, model_id in enumerate(batch.model_ids):
            rows = batch.rows_of(position)
            per_bracket[model_id] = float(error[rows].mean())

    if total_nodes == 0:
        raise ValueError("nothing to evaluate")

    return {
        "loss": total_loss / total_nodes,
        "mae_mpa": total_absolute_error / total_nodes,
        "per_bracket": per_bracket,
        "median_bracket_mae": float(np.median(list(per_bracket.values()))),
    }


def trivial_baseline_mpa(loader):
    """The error from always guessing the average stress.

    The number every model must beat before anything else it does is worth
    discussing. A model that cannot beat this has learned nothing, whatever its
    loss curve looks like.
    """
    values = []
    for batch in loader:
        values.append(batch.stress_mpa)

    stress = torch.cat(values)
    return float((stress - stress.mean()).abs().mean())


# ---------------------------------------------------------------------------
# CHECKPOINTS
# ---------------------------------------------------------------------------

def save_checkpoint(path, model, optimizer, history, epoch, config_values):
    """Everything needed to carry on from exactly here.

    The optimizer state matters as much as the weights. Adam keeps a running
    sense of how each weight has been behaving; drop it and a resumed run takes
    a while to settle back down, which shows up as a visible bump in the loss.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "history": asdict(history),
        "config": config_values,
    }, path)
    return path


def load_checkpoint(path, model, optimizer, device):
    """Restore a run. Returns the epoch to carry on from."""
    stored = torch.load(Path(path), map_location=device, weights_only=False)

    model.load_state_dict(stored["model"])
    if optimizer is not None:
        optimizer.load_state_dict(stored["optimizer"])

    history = History(**stored["history"])
    history.records = [EpochRecord(**record) for record in history.records]

    return history, stored["epoch"] + 1


# ---------------------------------------------------------------------------
# THE PICTURE
# ---------------------------------------------------------------------------

def plot_history(history, path, baseline_mpa=None, title=None):
    """Draw the loss curves and the validation error, and save them as a PNG.

    This adds no information: ``history.csv`` already holds every number in it.
    What it adds is a glance. Whether a run was SOUND is a shape rather than a
    column of figures - a validation curve that bottoms out at epoch 200 and
    then climbs for another 200 is unmistakable in a picture and easy to miss
    in a table of 400 rows.

    Two panels, because they answer different questions:

      left   the loss both curves are actually optimising. Train below
             validation is normal; the gap widening while validation rises is
             the model memorising the training brackets.
      right  the same run in MPa, next to the number it has to beat. This is
             the panel to show someone who does not work in machine learning.

    Wrapped in try/except on purpose. This is the last thing a ten-hour run
    does, and a missing matplotlib must not be the reason that run finishes
    without a saved figure - or worse, raises after the checkpoint is written.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")      # write a file; never look for a screen
        import matplotlib.pyplot as plt
    except Exception as error:     # noqa: BLE001 - reported, never raised
        print(f"  no loss curve: matplotlib is unavailable ({error})")
        return None

    if not history.records:
        return None

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    epochs = [record.epoch for record in history.records]
    train_loss = [record.train_loss for record in history.records]
    val_loss = [record.val_loss for record in history.records]
    val_mae = [record.val_mae_mpa for record in history.records]

    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4))

    left.plot(epochs, train_loss, linewidth=1.5, label="train")
    left.plot(epochs, val_loss, linewidth=1.5, label="validation")
    left.set_xlabel("epoch")
    left.set_ylabel("mean squared error")
    left.set_title("loss, in scaled log-stress space")

    # Log scale, because almost all of the movement is in the first few epochs
    # and a linear axis flattens everything after them into one straight line.
    if min(train_loss + val_loss) > 0:
        left.set_yscale("log")

    right.plot(epochs, val_mae, linewidth=1.5, color="tab:orange",
               label="validation")
    right.set_xlabel("epoch")
    right.set_ylabel("mean absolute error, MPa")
    right.set_title("validation error, in the units of the problem")

    if baseline_mpa:
        right.axhline(baseline_mpa, color="grey", linestyle="--", linewidth=1,
                      label=f"always guessing the mean "
                            f"({baseline_mpa:,.0f} MPa)")

    # The one epoch the saved model actually came from. Everything to the right
    # of this line was computed and then thrown away.
    if history.best_epoch in epochs:
        position = epochs.index(history.best_epoch)
        for axes in (left, right):
            axes.axvline(history.best_epoch, color="black", linestyle=":",
                         linewidth=1)
        right.plot([history.best_epoch], [val_mae[position]], "o",
                   color="black", markersize=5,
                   label=f"saved model: epoch {history.best_epoch}, "
                         f"{val_mae[position]:,.0f} MPa")

    for axes in (left, right):
        axes.grid(alpha=0.3)
        axes.legend(fontsize=8)

    # Named after the run, not after the folder the PNG happens to sit in -
    # run_5_plots.py writes into a plots/ subfolder, and "plots" is not a
    # useful heading.
    heading = title or path.parent.name
    if history.stopped_because:
        heading = f"{heading}  -  {history.stopped_because}"
    figure.suptitle(heading, fontsize=10)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)

    return path


# ---------------------------------------------------------------------------
# THE LOOP
# ---------------------------------------------------------------------------

def train(model, train_loader, val_loader, scalers, settings, run_directory,
          resume=True, baseline_mpa=None):
    """Train until the validation score stops improving, or time runs out.

    ``settings`` is anything with the training attributes of the config module,
    so a caller can hand over ``config`` itself or a small stand-in for a test.

    ``baseline_mpa`` is only used for drawing: it puts the "always guess the
    average" line on the figure, so the picture answers "is this model worth
    anything" and not just "did the loss go down".
    """
    run_directory = Path(run_directory)
    run_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_directory / "checkpoint.pt"
    best_path = run_directory / "best_model.pt"

    device = torch.device(settings.DEVICE
                          if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=settings.LEARNING_RATE,
                                  weight_decay=settings.WEIGHT_DECAY)

    history = History()
    first_epoch = 0

    if resume and checkpoint_path.is_file():
        history, first_epoch = load_checkpoint(checkpoint_path, model,
                                               optimizer, device)
        print(f"resumed from epoch {first_epoch}")

    print(f"device {device} | {len(train_loader)} batches per epoch")
    print(f"{'epoch':>6}{'train loss':>13}{'val loss':>11}"
          f"{'val MAE MPa':>14}{'secs':>8}")

    started = time.time()

    for epoch in range(first_epoch, settings.MAX_EPOCHS):
        epoch_started = time.time()
        model.train()

        running_loss = 0.0
        seen_nodes = 0

        for batch in train_loader:
            batch = batch.to(device)

            # PyTorch ADDS gradients rather than replacing them, so without
            # this they pile up across steps and the weights fly off.
            optimizer.zero_grad(set_to_none=True)

            prediction = model.predict(batch)
            loss = torch.nn.functional.mse_loss(prediction, batch.target)

            # Work backwards through every operation to find, for each weight,
            # which direction reduces the loss.
            loss.backward()

            # Cap the size of the update. One bracket with an extreme stress
            # singularity can produce a gradient large enough to undo an hour
            # of progress in a single step.
            if settings.GRAD_CLIP:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               settings.GRAD_CLIP)

            optimizer.step()

            # .detach() first: loss still carries the graph that produced
            # it, and converting it directly would keep that graph alive.
            running_loss += loss.detach().item() * batch.n_nodes
            seen_nodes += batch.n_nodes

        scores = evaluate(model, val_loader, scalers, settings.LOG_TARGET,
                          device)

        record = EpochRecord(
            epoch=epoch,
            train_loss=running_loss / max(seen_nodes, 1),
            val_loss=scores["loss"],
            val_mae_mpa=scores["mae_mpa"],
            seconds=time.time() - epoch_started,
        )
        history.records.append(record)

        improved = record.val_loss < history.best_val_loss - settings.MIN_IMPROVEMENT
        if improved:
            history.best_val_loss = record.val_loss
            history.best_epoch = epoch
            history.epochs_without_improvement = 0
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_loss": record.val_loss}, best_path)
        else:
            history.epochs_without_improvement += 1

        save_checkpoint(checkpoint_path, model, optimizer, history, epoch,
                        {"hidden": model.hidden_width,
                         "rounds": model.message_rounds})
        history.save(run_directory)

        marker = "  *" if improved else ""
        print(f"{epoch:>6}{record.train_loss:>13.4f}{record.val_loss:>11.4f}"
              f"{record.val_mae_mpa:>14.1f}{record.seconds:>8.0f}{marker}")

        if history.epochs_without_improvement >= settings.PATIENCE:
            history.stopped_because = (
                f"the validation loss has not improved for "
                f"{settings.PATIENCE} epochs")
            break

        hours = (time.time() - started) / 3600
        if hours >= settings.MAX_HOURS:
            history.stopped_because = (
                f"the time budget ran out at epoch {epoch}; "
                f"run again to carry on")
            break
    else:
        history.stopped_because = f"reached the {settings.MAX_EPOCHS}-epoch limit"

    history.save(run_directory)
    print(f"\n{history.stopped_because}")
    print(f"best epoch {history.best_epoch}, "
          f"validation loss {history.best_val_loss:.4f}")

    figure_path = plot_history(history, run_directory / "loss_curve.png",
                               baseline_mpa)
    if figure_path:
        print(f"loss curve: {figure_path}")

    return history


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import config
    from simjeb import batching as batching_module
    from simjeb import features as features_module
    from simjeb import model as model_module
    from simjeb import scaling as scaling_module
    from simjeb import splits as splits_module

    torch.manual_seed(config.SEED)

    split = splits_module.Split.load(config.SPLIT_FILE)
    scalers = scaling_module.Scalers.load(config.SCALING_FILE)

    def make_loader(model_ids, shuffle):
        return batching_module.Loader(
            config.FEATURE_DIR, model_ids, scalers, features_module.load,
            batch_size=1, shuffle=shuffle, seed=config.SEED)

    train_loader = make_loader(split.train, shuffle=True)
    val_loader = make_loader(split.val, shuffle=False)

    # A deliberately tiny model and a two-epoch budget: this is checking the
    # loop runs, not training anything. See run_3_train.py for a real run.
    class SmokeTest:
        DEVICE = "cpu"
        LEARNING_RATE = 1e-3
        WEIGHT_DECAY = config.WEIGHT_DECAY
        MAX_EPOCHS = 2
        GRAD_CLIP = config.GRAD_CLIP
        PATIENCE = 10
        MIN_IMPROVEMENT = config.MIN_IMPROVEMENT
        MAX_HOURS = 1.0
        LOG_TARGET = config.LOG_TARGET

    model = model_module.MeshGraphNet(
        node_width=features_module.N_NODE_FEATURES,
        hidden_width=16, message_rounds=2)
    print(f"SMOKE TEST - {model.describe()}\n")

    run_directory = config.RUN_DIR / "smoke_test"
    history = train(model, train_loader, val_loader, scalers, SmokeTest(),
                    run_directory, resume=False)

    print("\nscores after 2 epochs (these are not results):")
    baseline = trivial_baseline_mpa(make_loader(split.train, shuffle=False))
    for name in ("train", "val", "test"):
        members = getattr(split, name)
        scores = evaluate(model, make_loader(members, shuffle=False), scalers,
                          config.LOG_TARGET, torch.device("cpu"))
        detail = "  ".join(f"{k}:{v:,.0f}"
                           for k, v in scores["per_bracket"].items())
        print(f"  {name:<6}{scores['mae_mpa']:>8.1f} MPa   {detail}")
    print(f"  always guessing the mean: {baseline:,.1f} MPa")

    print(f"\nwrote {run_directory.name}/: "
          f"{[p.name for p in sorted(run_directory.iterdir())]}")
