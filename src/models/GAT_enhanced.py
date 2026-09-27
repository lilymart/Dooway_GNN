import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, Linear


class RelationGATv2Layer(torch.nn.Module):

    def __init__(self, metadata, in_channels, out_channels, edge_attr_dims=None, heads=1, dropout=0.3):
        super().__init__()

        self.node_types, self.edge_types = metadata
        self.edge_types = list(self.edge_types)

        self.edge_attr_dims = edge_attr_dims or {}

        self.convs = torch.nn.ModuleDict()
        self.edge_type_to_key = {}

        # First layer can use lazy input dimensions
        if in_channels is None:
            gat_in_channels = (-1, -1)
            linear_in_channels = -1
        else:
            gat_in_channels = (in_channels, in_channels)
            linear_in_channels = in_channels

        # One GATv2Conv for each relation
        for i, edge_type in enumerate(self.edge_types):

            key = f"rel_{i}"
            self.edge_type_to_key[edge_type] = key

            self.convs[key] = GATv2Conv(
                in_channels=gat_in_channels,
                out_channels=out_channels,
                heads=heads,
                concat=False,
                dropout=dropout,
                add_self_loops=False,
                edge_dim=self.edge_attr_dims.get(edge_type),
                share_weights=False,
            )

        # Node-type-specific residual/root transformation
        self.root_lins = torch.nn.ModuleDict()

        for node_type in self.node_types:
            self.root_lins[node_type] = Linear(linear_in_channels, out_channels)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        edge_attr_dict = edge_attr_dict or {}

        messages = {
            node_type: [] for node_type in self.node_types
        }

        for edge_type in self.edge_types:

            if edge_type not in edge_index_dict:
                continue

            src_type, _, dst_type = edge_type
            conv = self.convs[self.edge_type_to_key[edge_type]]
            edge_index = edge_index_dict[edge_type]
            x_pair = x_dict[src_type], x_dict[dst_type]

            # Edge-aware attention only where edge attributes exist
            if edge_type in self.edge_attr_dims:

                edge_attr = edge_attr_dict[edge_type]

                if edge_attr.dim() == 1:
                    edge_attr = edge_attr.view(-1, 1)

                expected_dim = self.edge_attr_dims[edge_type]

                if edge_attr.size(-1) != expected_dim:
                    raise ValueError(
                        f"Relation {edge_type}: expected "
                        f"edge_attr dimension {expected_dim}, "
                        f"got {edge_attr.size(-1)}."
                    )

                edge_attr = edge_attr.to(dtype=x_dict[src_type].dtype)
                out = conv(x_pair, edge_index, edge_attr=edge_attr)

            else:

                out = conv(x_pair, edge_index)

            messages[dst_type].append(out)

        out_dict = {}

        for node_type in self.node_types:

            node_messages = messages[node_type]


            # Same relation aggregation used by to_hetero(..., aggr="sum")
            rel_out = torch.stack(node_messages, dim=0, ).sum(dim=0)

            # Same residual/root branch as the original GAT:
            # lin(x.relu())
            root_out = self.root_lins[node_type](F.relu(x_dict[node_type]))
            out_dict[node_type] = rel_out + root_out

        return out_dict


class My_GAT_enhanced(torch.nn.Module):

    uses_edge_attr = True

    def __init__(self, metadata, target_type="user", hidden_channels=64, out_channels=15, dropout=0.3, num_layers=4, heads=1, edge_attr_dims=None):
        super().__init__()

        self.target_type = target_type
        self.num_layers = num_layers
        self.edge_attr_dims = edge_attr_dims or {}

        self.hidden_convs = torch.nn.ModuleList()

        # Hidden message-passing layers
        for layer_idx in range(num_layers - 1):

            self.hidden_convs.append(
                RelationGATv2Layer(
                    metadata=metadata,
                    in_channels=None if layer_idx == 0 else hidden_channels,
                    out_channels=hidden_channels,
                    edge_attr_dims=self.edge_attr_dims,
                    heads=heads,
                    dropout=dropout,
                )
            )

        # Final GATv2 classification layer:
        # hidden_channels -> out_channels
        self.final_conv = RelationGATv2Layer(
            metadata=metadata,
            in_channels=hidden_channels,
            out_channels=out_channels,
            edge_attr_dims=self.edge_attr_dims,
            heads=heads,
            dropout=dropout,
        )


    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        h_dict = x_dict

        # Hidden layers except the embedding layer
        for i in range(self.num_layers - 2):

            h_dict = self.hidden_convs[i](
                h_dict,
                edge_index_dict,
                edge_attr_dict=edge_attr_dict,
            )

            # EXACTLY as in the original model:
            # ReLU only between hidden layers
            h_dict = {
                node_type: h.relu()
                for node_type, h in h_dict.items()
            }

        # Last hidden layer -> embeddings
        embeddings = self.hidden_convs[-1](
            h_dict,
            edge_index_dict,
            edge_attr_dict=edge_attr_dict,
        )

        # Final message-passing classification layer
        logits_dict = self.final_conv(
            embeddings,
            edge_index_dict,
            edge_attr_dict=edge_attr_dict,
        )

        return logits_dict, embeddings