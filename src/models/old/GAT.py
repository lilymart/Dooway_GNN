import torch
from torch_geometric.nn import GATv2Conv, Linear


class My_GAT(torch.nn.Module):

    def __init__(self, hidden_channels=128, out_channels=2, dropout=0.3, num_layers=3):
        super().__init__()

        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()

        for _ in range(num_layers):

            self.convs.append(
                GATv2Conv((-1, -1), hidden_channels, add_self_loops=False, dropout=dropout)
            )

            self.lins.append(
                Linear(-1, hidden_channels)
            )

        # Prediction head
        self.classifier = Linear(hidden_channels, out_channels)


    def forward(self, x, edge_index, edge_attr_dict=None):

        embeddings = None

        for layer_idx in range(self.num_layers):

            h = self.convs[layer_idx](x, edge_index) + self.lins[layer_idx](x.relu())

            # Activation only between GNN layers
            if layer_idx < self.num_layers - 1:
                x = h.relu()
            else:
                embeddings = h

        logits = self.classifier(embeddings)

        return logits, embeddings