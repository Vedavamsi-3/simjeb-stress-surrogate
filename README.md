# SimJEB structural surrogate — clean build

A graph neural network that predicts surface von Mises stress on jet-engine
brackets, built from the raw solver files.

This is the tidied version of the walkthrough in the folders next door. Same
pipeline, organised so it runs on any number of brackets — 8 or all 381 — with
one command per stage.

```
raw files  ──▶  graphs  ──▶  features  ──▶  split + scaling  ──▶  model  ──▶  scores
```

---

## Run it

```bash
python run_1_build_dataset.py     # raw files -> graphs + features
python run_2_make_splits.py       # decide train/val/test, fit the scaling
python run_3_train.py --quick     # check the code runs (~15 s)
python run_4_evaluate.py --run quick
python run_5_plots.py --run quick # five figures from what stage 4 wrote
```

Then, when there is a GPU and real data:

```bash
python run_3_train.py             # the full model, resumable
python run_4_evaluate.py --run full
python run_5_plots.py --run full
```

Every setting lives in [`config.py`](config.py). Point `DATA_DIR` at a folder of
SimJEB files and nothing else needs editing.

Each module in `simjeb/` also runs on its own, which is how to check one piece
in isolation:

```bash
python -m simjeb.deck 0        # read one solver deck
python -m simjeb.mesh 0        # read one mesh, find its skin and edges
python -m simjeb.graph 0       # build one graph
python -m simjeb.features 0    # build one bracket's features
python -m simjeb.splits        # the split
python -m simjeb.scaling       # the scaling
python -m simjeb.batching      # gluing graphs together
python -m simjeb.model         # the network, one forward pass
python -m simjeb.training      # a two-epoch smoke test
```

---

## Layout

```
config.py                  every setting, and nothing but settings

simjeb/
  deck.py                  read the .fem solver deck: clamps, loads, material
  mesh.py                  read the .vtk: skin triangles, edges, normals
  graph.py                 combine all three files into one graph
  features.py              graph -> the numbers a network reads
  splits.py                which brackets train / validate / test
  scaling.py               fit on train, apply to everything
  batching.py              several graphs -> one disconnected graph
  model.py                 the MeshGraphNet
  training.py              the loop, early stopping, checkpoints

run_1_build_dataset.py     stage 1
run_2_make_splits.py       stage 2
run_3_train.py             stage 3
run_4_evaluate.py          stage 4
run_5_plots.py             stage 5

output/                    everything produced, nothing anywhere else
  graphs/                  one .npz per bracket, geometry
  features/                one .npz per bracket, model inputs
  reports/                 what was excluded and why
  runs/<name>/             checkpoints, history, results
    loss_curve.png         drawn by stage 3 at the end of training
    predictions.npz        thinned node-level predictions, for stage 5
    plots/                 the five figures from stage 5
  split.json
  scaling.npz
```

---

## What the model sees

Three raw files describe one bracket, and none is sufficient alone:

| file | carries | why it is needed |
|---|---|---|
| `<id>.vtk` | node positions, tetrahedra | the shape, and which nodes are joined |
| `<id>field.csv` | one row per node: `surf`, stress | **the answers** |
| `<id>.fem` | the solver deck | where it is clamped, where it is pushed, with what |

Only **surface** nodes are kept — about 40% of the mesh. Cracks start at the
surface, and dropping the interior makes the dataset small enough to work with.

**9 numbers per node**

| feature | count | |
|---|---|---|
| node kind, one-hot | 3 | ordinary / clamped / loaded |
| surface normal | 3 | which way the skin faces |
| sharpness | 1 | 1.0 flat, lower at an edge |
| distance to nearest clamp | 1 | mm |
| distance to nearest load | 1 | mm |

**4 numbers per edge** — the step from one end to the other, and its length.

The raw x, y, z are deliberately **not** features. With a few hundred training
brackets, "stress is high at x = 12" is easy to memorise and worthless on a new
design. "I am 5 mm from a bolt hole, on a surface facing up" transfers.

---

## Design decisions worth knowing

### Why edge features are not optional

Stress comes from strain, and strain is a spatial derivative of displacement. A
derivative is built from differences between neighbouring points — exactly what
a message carrying `x_j − x_i` lets a layer compute. A graph network without
edge features (GCN, GraphSAGE, GAT) can record *that* two nodes are connected
but not how far apart or in which direction, so it cannot represent the
derivative at all. That is a requirement, not a preference.

### Edges come from tetrahedra, not from distance

Every skin-triangle edge is also a tetrahedron edge, so tetrahedron edges are a
superset. The extra ones cut straight *through* the material — from one face of
a thin rib to the node opposite it — and material genuinely joins those nodes.

What it refuses to do matters as much: two surfaces facing each other across a
gap share no tetrahedron, so no edge appears. A rule like "connect anything
within 2 mm" would bridge that gap and teach the network a connection that does
not exist.

### Sharpness is free, and was being thrown away

A node's normal is the sum of its triangles' normals. If they all face the same
way the sum is long; if they disagree it partly cancels and the sum is short.
So `length of the sum / sum of the lengths` is 1.0 on a flat surface and falls
towards 0 at a sharp edge — that *is* curvature, and curvature is what sets
stress concentration.

Rescaling the normal to unit length destroys it, so it is captured first and
handed over as a 9th feature. Without it, the network has to infer sharpness
indirectly by comparing normals across several message-passing rounds.

### 64 wide × 8 rounds, not the paper's 128 × 15

The usual argument for depth is that a node should see the whole load path.
Measured on a real SimJEB graph, the load lug is **54 hops** from the nearest
bolt hole — so 15 rounds reaches a quarter of it and 54 would need over 10 GB
per graph. The paper's meshes are an order of magnitude coarser.

Spanning the load path is the wrong target anyway. Stress concentration is
**local**, set by fillet radii and thickness changes; 8 hops at ~1 mm edge
length covers that scale. The **global** picture — where a node sits between
the clamps and the load — is handed over directly by the two distance features,
needing no message passing at all. Those two features are load-bearing.

### The stress target is log-transformed

Peak von Mises across SimJEB runs to ~15,000 MPa against a Ti-6Al-4V yield of
~880. These are linear-elastic solves, so that is the signature of stress
singularities at sharp corners — numerical artefacts that grow with mesh
refinement rather than converging.

Under a plain squared-error loss a handful of those nodes would supply most of
the gradient. `log1p` compresses the tail without discarding it, and makes the
network care about *relative* error: 20% wrong matters the same at 200 MPa as
at 2,000.

**Every reported score inverts the transform first.** A score computed in log
space flatters the model by shrinking exactly the large errors that matter.

### Rotation is never corrected, only excluded

A translation changes nothing, so brackets are shifted to a common origin —
centred on the five interface landmarks, *not* the centre of the part, which
moves with the design.

A rotation is not neutral. The load is a fixed world-direction that does not
turn with the part, so a rotated bracket under that load carries force along a
different internal path. Straightening it would leave a tidy mesh whose answers
came from a different physics problem. So those brackets are dropped.

---

## The checks

Every check below guards a failure that would **not** crash. Wrong node
numbering, mismatched rows, an edge pointing at nothing — the model would train
to a plausible loss curve on scrambled labels and nothing would look odd.

| check | where | what it catches |
|---|---|---|
| CSV rows vs mesh points | `mesh.py` | files describing different brackets |
| CSV coordinates vs mesh coordinates | `mesh.py` | rows shifted by one |
| deck's `RBE2` expansion vs CSV `surf` labels | `deck.py` | wrong field offsets, wrong 1-based conversion |
| reference nodes are one contiguous block | `deck.py` | `row = id − 1` not holding |
| no edge points at a dropped node | `graph.py` | broken renumbering |
| no isolated nodes | `graph.py` | holes in the edge construction |
| interfaces near the typical frame | `run_1` | rotated brackets |
| no design family straddles a split | `splits.py` | a variant in train and its twin in test |
| scaling fitted on train only | `run_2` | test information folded into training |
| batched == one at a time | `batching.py` | wrong offset arithmetic |
| the target transform inverts exactly | `features.py` | every MPa figure being wrong |

The two cross-file checks are the valuable ones. The deck reaches the bolt-hole
nodes through `SPC` and `RBE2` cards; the CSV reaches them through SimJEB's own
labelling. Nothing connects the two routes, so agreement is real evidence that
the 8-column field offsets, the continuation lines and the off-by-one
conversion are **all** correct at once.

Anything excluded is written to `output/reports/excluded.csv` with a reason. A
pipeline that silently drops samples is how a dataset quietly becomes something
other than what you think you are training on.

---

## Honest state

Measured on the 8 brackets in `multi_bracket/Data`:

```
stage 1   8 of 8 built, 0 excluded, 49 s, 79 MB of cache
stage 2   6 train / 1 val / 1 test
stage 3   --quick: 4,977 parameters, 3 epochs, 15 s
stage 4   test MAE 80.8 MPa against a trivial baseline of 87.7
```

**None of those numbers say anything about the model.** With one test bracket,
the score reports which bracket landed there. With 4,977 parameters and 3
epochs, the model has learned "stress is about average everywhere" — visible in
the per-bracket table, where every prediction peaks near 145 MPa whether the
truth is 659 or 4,823.

What they do say is that the code runs end to end and the checks pass.

### To make it a real run

1. Point `DATA_DIR` at all 381 brackets.
2. `python run_1_build_dataset.py` — about 40 minutes, expect exclusions.
3. `python run_2_make_splits.py` — a 76-bracket test set, and the small-test
   warning will stop firing.
4. `python run_3_train.py` on a GPU — 64 × 8, up to 300 epochs, early stopping.
   Resumable, so a session shorter than the run is fine.
5. `python run_4_evaluate.py --run full` — **once**.
6. `python run_5_plots.py --run full` — the figures. Reads only
   `history.csv`, `per_bracket.csv`, `results.json` and
   `predictions.npz`, so it can be run at home on files downloaded
   from the training machine.

Stage 3 needs roughly 4 GB per training step at batch 2. On this CPU one epoch
at the full size takes minutes rather than seconds, which is the whole reason
the real project runs on Kaggle.

---

## Reference

Whalen, Beyene & Mueller, *SimJEB: Simulated Jet Engine Bracket Dataset*,
Computer Graphics Forum, 2021. [arXiv:2105.03534](https://arxiv.org/abs/2105.03534)
