"""Feed several brackets to the network at once.

Rows of a spreadsheet can be stacked because they are all the same width.
Graphs cannot: one bracket has 24,577 nodes and another 45,862, and there is no
sensible way to pad that.

The trick is to stop thinking of them as several graphs. Glue them into ONE
graph with several disconnected pieces:

    bracket A   nodes 0..41       edges point at 0..41
    bracket B   nodes 0..29   ->  nodes 42..71, and every edge number +42

    result      one graph, 72 nodes, two pieces, no edge between them

Because no edge is ever created between the pieces, a message cannot travel
from one bracket into another. Message passing runs once over the whole thing
and each bracket comes out exactly as it would have alone - which this module
proves rather than asserts.

Run on its own::

    python -m simjeb.batching
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class Batch:
    """Several brackets as one graph, plus a note of which node came from where."""

    model_ids: list
    node_features: torch.Tensor    # (total_nodes, 9)
    edge_index: torch.Tensor       # (2, total_edges)
    edge_features: torch.Tensor    # (total_edges, 4)
    target: torch.Tensor           # (total_nodes, 1)
    stress_mpa: torch.Tensor       # (total_nodes, 1)
    owner: torch.Tensor            # (total_nodes,) which bracket, 0..n-1

    @property
    def n_nodes(self):
        return self.node_features.shape[0]

    @property
    def n_edges(self):
        return self.edge_index.shape[1]

    def to(self, device):
        """Move every tensor to a device, returning a new batch."""
        return Batch(
            model_ids=self.model_ids,
            node_features=self.node_features.to(device),
            edge_index=self.edge_index.to(device),
            edge_features=self.edge_features.to(device),
            target=self.target.to(device),
            stress_mpa=self.stress_mpa.to(device),
            owner=self.owner.to(device),
        )

    def rows_of(self, position):
        """Which rows belong to the bracket at ``position`` in ``model_ids``.

        Needed to score each bracket separately after one pooled prediction. A
        single pooled average hides everything: it is dominated by whichever
        bracket has the most nodes and whichever has the largest errors.
        """
        return self.owner == position


def prepare(features, scalers):
    """One bracket's features, scaled and turned into tensors.

    The scaling comes from the training brackets and is applied unchanged to
    every bracket, whichever split it is in. A validation bracket is scaled by
    numbers it played no part in producing - exactly the position a brand new
    bracket would be in.
    """
    node_features = scalers.node.apply(features.node_features)
    edge_features = scalers.edge.apply(features.edge_features)
    target = scalers.target.apply(features.target)

    return {
        "model_id": features.model_id,
        "node_features": torch.tensor(node_features, dtype=torch.float32),
        "edge_index": torch.tensor(features.edge_index, dtype=torch.long),
        "edge_features": torch.tensor(edge_features, dtype=torch.float32),
        "target": torch.tensor(target, dtype=torch.float32),
        "stress_mpa": torch.tensor(features.stress_mpa, dtype=torch.float32),
    }


def collate(prepared_brackets):
    """Glue prepared brackets into one batch.

    Node arrays simply stack - one row per node, so more brackets means more
    rows. The edges are the part that needs care, because each bracket numbers
    its own nodes from zero and those numbers would collide.
    """
    if not prepared_brackets:
        raise ValueError("cannot make a batch from no brackets")

    node_features = []
    edge_indices = []
    edge_features = []
    targets = []
    stresses = []
    owners = []

    nodes_so_far = 0

    for position, bracket in enumerate(prepared_brackets):
        n_nodes = bracket["node_features"].shape[0]

        node_features.append(bracket["node_features"])
        edge_features.append(bracket["edge_features"])
        targets.append(bracket["target"])
        stresses.append(bracket["stress_mpa"])

        # The shift. Every node number in this bracket's edges moves up by
        # however many nodes came before it.
        edge_indices.append(bracket["edge_index"] + nodes_so_far)

        # Remember where each node came from, so per-bracket scores can be
        # recovered from a single pooled prediction.
        owners.append(torch.full((n_nodes,), position, dtype=torch.long))

        nodes_so_far += n_nodes

    return Batch(
        model_ids=[bracket["model_id"] for bracket in prepared_brackets],
        node_features=torch.cat(node_features),
        edge_index=torch.cat(edge_indices, dim=1),
        edge_features=torch.cat(edge_features),
        target=torch.cat(targets),
        stress_mpa=torch.cat(stresses),
        owner=torch.cat(owners),
    )


def check_pieces_are_separate(batch):
    """Confirm no edge joins two different brackets.

    If one did, a message would cross from one design into another and the
    network would be learning from a shape that does not exist. Cheap to check
    and impossible to notice otherwise.
    """
    owner_of_sender = batch.owner[batch.edge_index[0]]
    owner_of_receiver = batch.owner[batch.edge_index[1]]
    crossing = int((owner_of_sender != owner_of_receiver).sum())

    if crossing > 0:
        raise ValueError(f"{crossing} edges join two different brackets")
    return crossing


class Loader:
    """Hands out batches of brackets, one epoch at a time.

    Loads from disk on demand rather than holding everything in memory. Eight
    brackets would fit; 381 at ~8 MB each would not, and the whole point of
    this version is that it works at either size.
    """

    def __init__(self, feature_directory, model_ids, scalers, load_features,
                 batch_size=2, shuffle=False, seed=0):
        self.feature_directory = Path(feature_directory)
        self.model_ids = list(model_ids)
        self.scalers = scalers
        self.load_features = load_features
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        """How many batches one pass takes."""
        full, remainder = divmod(len(self.model_ids), self.batch_size)
        return full + (1 if remainder else 0)

    def __iter__(self):
        order = list(self.model_ids)

        if self.shuffle:
            # A different order every epoch, but reproducible: the seed and the
            # epoch number together decide it. The model should never be able
            # to learn anything from the order brackets arrive in.
            rng = np.random.default_rng(self.seed + self.epoch)
            rng.shuffle(order)
            self.epoch += 1

        for start in range(0, len(order), self.batch_size):
            chosen = order[start:start + self.batch_size]

            prepared = []
            for model_id in chosen:
                features = self.load_features(self.feature_directory, model_id)
                prepared.append(prepare(features, self.scalers))

            yield collate(prepared)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import config
    from simjeb import features as features_module
    from simjeb import scaling as scaling_module
    from simjeb import splits as splits_module

    split = splits_module.Split.load(config.SPLIT_FILE)
    scalers = scaling_module.Scalers.load(config.SCALING_FILE)

    pair = split.train[:2]
    prepared = []
    for model_id in pair:
        features = features_module.load(config.FEATURE_DIR, model_id)
        prepared.append(prepare(features, scalers))

    for bracket in prepared:
        print(f"bracket {bracket['model_id']:>3}: "
              f"{bracket['node_features'].shape[0]:>7,} nodes, "
              f"{bracket['edge_index'].shape[1]:>8,} edges, "
              f"numbered 0 to {bracket['node_features'].shape[0] - 1:,}")

    batch = collate(prepared)
    print(f"\nbatched   : {batch.n_nodes:>7,} nodes, {batch.n_edges:>8,} edges, "
          f"numbered 0 to {batch.n_nodes - 1:,}")

    first_shift = prepared[0]["node_features"].shape[0]
    edges_before = prepared[0]["edge_index"].shape[1]
    print(f"\nbracket {pair[1]}'s first edge alone   : "
          f"{prepared[1]['edge_index'][:, 0].tolist()}")
    print(f"bracket {pair[1]}'s first edge batched : "
          f"{batch.edge_index[:, edges_before].tolist()}   "
          f"(shifted by {first_shift:,})")

    crossing = check_pieces_are_separate(batch)
    print(f"\nedges joining two brackets: {crossing}")

    # The stronger check. Any function applied node by node must give the same
    # answer batched as separately - here a stand-in for the real network.
    def message_round(node_features, edge_index, edge_features):
        senders, receivers = edge_index[0], edge_index[1]
        messages = edge_features.sum(dim=1, keepdim=True) * 0.5
        gathered = torch.zeros(node_features.shape[0], 1)
        gathered.index_add_(0, receivers, messages)
        return node_features.sum(dim=1, keepdim=True) + gathered

    batched = message_round(batch.node_features, batch.edge_index,
                            batch.edge_features)
    separately = torch.cat([
        message_round(b["node_features"], b["edge_index"], b["edge_features"])
        for b in prepared
    ])
    worst = float((batched - separately).abs().max())
    print(f"batched vs one at a time, worst difference: {worst:.2e}")
    print("\nZero. Batching is a way to keep the machine busy - it changes no")
    print("prediction. If this were not tiny the offset arithmetic would be")
    print("wrong somewhere.")

    loader = Loader(config.FEATURE_DIR, split.train, scalers,
                    features_module.load, batch_size=config.BATCH_SIZE,
                    shuffle=True, seed=config.SEED)
    print(f"\nloader over {len(split.train)} training brackets, "
          f"batch {config.BATCH_SIZE}: {len(loader)} batches per epoch")
    for number, batch in enumerate(loader, start=1):
        print(f"  batch {number}: brackets {batch.model_ids}, "
              f"{batch.n_nodes:,} nodes")
