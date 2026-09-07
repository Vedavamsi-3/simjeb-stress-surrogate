"""The network: a MeshGraphNet, encode - process - decode.

Reference: Pfaff, Fortunato, Sanchez-Gonzalez & Battaglia, *Learning Mesh-Based
Simulation with Graph Networks*, ICLR 2021.

Adapted in two ways, both worth being able to state plainly:

  * MeshGraphNets predicts the NEXT state and rolls forward through time. A
    linear-static solve has no time dimension, so this predicts the field
    directly in one shot.
  * MeshGraphNets carries two edge types because it handles contact and
    self-collision. Nothing here collides, so there is one.

Why this family of network at all
---------------------------------
Stress comes from strain, and strain is a spatial derivative of displacement. A
derivative is built from differences between neighbouring points - exactly what
a message carrying ``x_j - x_i`` lets a layer compute. A graph network without
edge FEATURES (GCN, GraphSAGE, GAT) can record that two nodes are connected but
not how far apart or in which direction, so it cannot represent the derivative
at all. That is a requirement, not a preference, and it is what selects this
architecture.

Inputs are expected already scaled - see :mod:`simjeb.scaling`. The model owns
no statistics of its own, so a checkpoint plus its saved scalers fully
determine what it will predict.

Run on its own::

    python -m simjeb.model
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.utils import scatter


def make_mlp(input_width, output_width, hidden_width, dropout=0.0,
             layer_norm=True):
    """The small network used at every point in the model.

    ReLU is what makes depth worth having. Without a bend between the two
    linear layers they would collapse into a single linear map.

    LayerNorm keeps each stage's output at a steady scale. Stacked eight blocks
    deep, small drifts compound and training becomes unstable. The decoder
    leaves it off, because its output is a physical answer rather than an
    internal state that feeds the next layer.
    """
    layers = [nn.Linear(input_width, hidden_width), nn.ReLU()]

    if dropout > 0:
        layers.append(nn.Dropout(dropout))

    layers.append(nn.Linear(hidden_width, output_width))

    if layer_norm:
        layers.append(nn.LayerNorm(output_width))

    return nn.Sequential(*layers)


class ProcessorBlock(nn.Module):
    """One round of message passing: update every edge, then every node.

    Each block moves information exactly one step along the mesh, so the number
    of blocks decides how far a node can see.

    Both updates are residual - the block computes a CHANGE and adds it rather
    than replacing the state. At this depth that is what makes the network
    trainable at all, for the same reason ResNet uses them.
    """

    def __init__(self, hidden_width, dropout=0.0):
        super().__init__()

        # An edge looks at itself and at the nodes on both of its ends.
        self.edge_mlp = make_mlp(3 * hidden_width, hidden_width, hidden_width,
                                 dropout)

        # A node looks at itself and at the sum of the messages arriving.
        self.node_mlp = make_mlp(2 * hidden_width, hidden_width, hidden_width,
                                 dropout)

    def forward(self, node_state, edge_state, edge_index):
        senders = edge_index[0]
        receivers = edge_index[1]

        # node_state[senders] copies each sender's row out to its edges, so
        # node data and edge data can sit side by side and be concatenated.
        edge_input = torch.cat(
            [edge_state, node_state[senders], node_state[receivers]], dim=1)
        edge_change = self.edge_mlp(edge_input)

        # Add up the messages arriving at each node.
        #
        # SUM, not mean. A node with more neighbours genuinely has more
        # material attached to it, and averaging would erase that.
        #
        # PyTorch Geometric's scatter is used here rather than its
        # MessagePassing base class, and that is deliberate. MessagePassing
        # hands back only the aggregated node result, while this block also has
        # to return the updated EDGE state - so the base class does not fit.
        # scatter is the piece that does fit, and it is what the production
        # version of this project uses.
        gathered = scatter(edge_change, receivers, dim=0,
                           dim_size=node_state.shape[0], reduce="sum")

        node_input = torch.cat([node_state, gathered], dim=1)
        node_change = self.node_mlp(node_input)

        return node_state + node_change, edge_state + edge_change


class MeshGraphNet(nn.Module):
    """Encode into a latent width, pass messages, decode to one number.

    Choosing the size
    -----------------
    The paper uses 128 wide and 15 blocks. Those do not transfer here, and the
    reason is worth recording.

    The usual argument for depth is that a node's receptive field should span
    the load path. Measured on a real SimJEB surface graph, the load lug is 54
    hops from the nearest bolt hole - so 15 blocks reaches barely a quarter of
    it, and 54 blocks would need over 10 GB for a single graph. The paper's
    meshes are an order of magnitude coarser, so hop counts do not carry over.

    Spanning the load path turns out to be the wrong target anyway:

      * stress concentration is LOCAL, set by fillet radii and thickness
        changes. At about 1 mm mean edge length, 8 hops reaches roughly 8 mm,
        which covers that length scale.
      * the GLOBAL context - where a node sits between the clamps and the load
        - is handed over directly by the two distance features, with no message
        passing required at all.

    Those two features are therefore load-bearing, not conveniences.
    """

    def __init__(self, node_width, edge_width=4, hidden_width=64,
                 message_rounds=8, output_width=1, dropout=0.0):
        super().__init__()

        self.node_width = node_width
        self.edge_width = edge_width
        self.hidden_width = hidden_width
        self.message_rounds = message_rounds
        self.output_width = output_width

        self.node_encoder = make_mlp(node_width, hidden_width, hidden_width)
        self.edge_encoder = make_mlp(edge_width, hidden_width, hidden_width)

        blocks = []
        for _ in range(message_rounds):
            blocks.append(ProcessorBlock(hidden_width, dropout))
        self.blocks = nn.ModuleList(blocks)

        self.decoder = make_mlp(hidden_width, output_width, hidden_width,
                                layer_norm=False)

    def forward(self, node_features, edge_index, edge_features):
        node_state = self.node_encoder(node_features)
        edge_state = self.edge_encoder(edge_features)

        for block in self.blocks:
            node_state, edge_state = block(node_state, edge_state, edge_index)

        return self.decoder(node_state)

    def predict(self, batch):
        """Convenience wrapper for a :class:`simjeb.batching.Batch`."""
        return self.forward(batch.node_features, batch.edge_index,
                            batch.edge_features)

    @property
    def n_parameters(self):
        total = 0
        for parameter in self.parameters():
            total += parameter.numel()
        return total

    def describe(self):
        return (f"MeshGraphNet({self.node_width} node features, "
                f"{self.edge_width} edge features, {self.hidden_width} wide, "
                f"{self.message_rounds} rounds) - {self.n_parameters:,} parameters")


def estimate_memory_gb(n_nodes, n_edges, hidden_width, message_rounds,
                       bytes_per_number=4):
    """Roughly how much working memory one training step needs, in GB.

    Backpropagation keeps every block's intermediate tensors alive, so memory
    grows with depth. On these graphs the EDGE tensors dominate - there are
    about six times as many edges as nodes, and the edge update concatenates
    three hidden-width tensors across all of them. That is what limits batch
    size here, far more than in a typical graph network.

    Deliberately approximate: it counts each block's inputs and outputs, not
    every activation inside the MLPs, so treat it as a floor rather than a
    prediction. Measure on the real device before fixing a batch size.
    """
    per_block = (
        n_edges * 3 * hidden_width      # the concatenated edge input
        + n_edges * hidden_width        # the edge update
        + n_nodes * 2 * hidden_width    # the concatenated node input
        + n_nodes * hidden_width        # the node update
    )
    return per_block * message_rounds * bytes_per_number / 1e9


def build(node_width, config_module):
    """Make the model the config describes."""
    return MeshGraphNet(
        node_width=node_width,
        edge_width=4,
        hidden_width=config_module.HIDDEN_WIDTH,
        message_rounds=config_module.MESSAGE_ROUNDS,
        output_width=1,
        dropout=config_module.DROPOUT,
    )


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import config
    from simjeb import batching as batching_module
    from simjeb import features as features_module
    from simjeb import scaling as scaling_module
    from simjeb import splits as splits_module

    torch.manual_seed(config.SEED)

    split = splits_module.Split.load(config.SPLIT_FILE)
    scalers = scaling_module.Scalers.load(config.SCALING_FILE)

    model = build(features_module.N_NODE_FEATURES, config)
    print(model.describe())

    print("\nwhere the parameters are:")
    parts = [
        ("node encoder", model.node_encoder),
        ("edge encoder", model.edge_encoder),
        ("one block", model.blocks[0]),
        ("all blocks", model.blocks),
        ("decoder", model.decoder),
    ]
    for name, part in parts:
        count = sum(p.numel() for p in part.parameters())
        share = 100 * count / model.n_parameters
        print(f"  {name:<14}{count:>10,}   {share:>5.1f}%")

    # One batch through the model.
    prepared = []
    for model_id in split.train[:2]:
        features = features_module.load(config.FEATURE_DIR, model_id)
        prepared.append(batching_module.prepare(features, scalers))
    batch = batching_module.collate(prepared)

    print(f"\nbatch: brackets {batch.model_ids}, {batch.n_nodes:,} nodes, "
          f"{batch.n_edges:,} edges")

    with torch.no_grad():
        prediction = model.predict(batch)
    print(f"output {tuple(prediction.shape)} - one number per node")

    # Batching must not change a single prediction. Same check as in
    # simjeb.batching, now with the real network rather than a stand-in.
    with torch.no_grad():
        separately = torch.cat([
            model(b["node_features"], b["edge_index"], b["edge_features"])
            for b in prepared
        ])
    worst = float((prediction - separately).abs().max())
    print(f"batched vs one at a time, worst difference: {worst:.2e}")

    error = (prediction - batch.target).abs().mean()
    print(f"\nmean error, untrained: {error:.3f} in scaled units")
    print("About 1.0 means no better than guessing, which is what random")
    print("weights should give. Training is what changes it.")

    print(f"\nestimated working memory for one training step:")
    for size in (1, 2, 4):
        nodes = batch.n_nodes // 2 * size
        edges = batch.n_edges // 2 * size
        gigabytes = estimate_memory_gb(nodes, edges, config.HIDDEN_WIDTH,
                                       config.MESSAGE_ROUNDS)
        print(f"  batch of {size}: {nodes:>8,} nodes, {edges:>9,} edges"
              f"  ->  {gigabytes:>5.1f} GB")
