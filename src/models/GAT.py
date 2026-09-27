import torch
from torch_geometric.nn import GATv2Conv, Linear


class My_GAT(torch.nn.Module):

    def __init__(self, hidden_channels=128, out_channels=2, dropout=0.3, num_layers=4):
        super().__init__()

        # num_layers = TOTAL number of message-passing layers
        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()

        # Hidden GATv2 layers
        # Example: num_layers=4 -> 3 hidden layers + 1 output GAT layer
        for _ in range(num_layers - 1):

            self.convs.append(
                GATv2Conv((-1, -1), hidden_channels, add_self_loops=False, dropout=dropout)
            )

            self.lins.append(
                Linear(-1, hidden_channels)
            )

        # Final message-passing classification layer
        self.final_conv = GATv2Conv((-1, -1), out_channels, add_self_loops=False, dropout=dropout)
        self.final_lin = Linear(-1, out_channels)

    def forward(self, x, edge_index, edge_attr_dict=None):

        # Hidden layers except the embedding layer
        for i in range(self.num_layers - 2):

            x = self.convs[i](x, edge_index) + self.lins[i](x.relu())
            x = x.relu()

        # Last hidden GAT layer -> embeddings
        embeddings = self.convs[-1](x, edge_index) + self.lins[-1](x.relu())

        # Final GATv2 classification layer
        logits = self.final_conv(embeddings, edge_index) + self.final_lin(embeddings.relu())

        return logits, embeddings