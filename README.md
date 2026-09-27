## LANTERN

Implementation of LANTERN: LM-Assisted graph attention Network for Tourist pERsoNalization. <!--, as presented in our paper: -->

<!--
```
Comito, C., Falcone, A., Forestiero, A., Martirano, L.,
LLM-Assisted Multimodal Graph Attention network for Multilabel Tourist Profiling 
Information Fusion (2026).
```
-->

> Our proposed LANTERN exploits multimodal heterogeneous graph representation learning, edge-aware relational attention, and expert-validated LLM-assisted supervision, providing a coherent and accurate framework for multi-label prediction of user profiles in tourism scenarios, shedding light on their latent behavioral patterns and preferences.


## Requirements
You can install the required dependencies by running the following command:
```
pip install -r requirements.txt
```

Please note that some libraries (pyg-lib, torch-cluster, torch-scatter, torch-sparse and torch-spline-conv) may fail their installation.
You can install each of this library by running the following:
```
pip install library_name -f https://data.pyg.org/whl/torch-2.3.1+cu121.html
```

## Data
The dataset used in this study was developed within the *Data Analytics for Tourism in Italy - DOOWAY* project, partially funded by the Italian Ministry of Enterprises and Made in Italy. Due to the terms associated with the original data collection and the underlying third-party sources, the dataset cannot currently be publicly released. 

LANTERN is not tied to the DOOWAY dataset and can be applied to other heterogeneous graph datasets by specifying the target node type for node classification.

The current data loader expects a `heterodata/` directory organized as follows:

```
heterodata/
├── node_features/
│   └── tensors/
├── edgelists/
│   └── tensors/
└── edgeattrs/
    └── tensors/
```

Node features, graph connectivity, and optional edge attributes are stored separately and assembled according to the heterogeneous node and relation schema.
The loader converts these inputs into a PyTorch Geometric *HeteroData* object used by the training pipeline. This loading procedure reflects the current implementation and can be adapted to support different input formats or preprocessing pipelines.


## Training and evaluation
For training LANTERN, you can run the script `run_LANTERN_and_HGNN_models.sh` specifying the following parameters:
- *dataset_name*: string specifying the name of the dataset to be processed.
- *model_name*: string specifying the model to be trained (default: `"LANTERN"`).
- *feats_dim*: integer specifying the dimensionality of the input node features (default: `256`).
- *target_type*: string specifying the target node type for node classification (default: `"user"`).
- *multilabel*: boolean indicating whether the prediction task is multi-label (`true`/`false`).
- *use_attrs*: boolean indicating whether node attributes are used as input features (`true`/`false`).
- *use_imgs*: boolean indicating whether visual features are included when available (`true`/`false`).
- *seed*: integer specifying the random seed used for data splitting, model initialization, and training (default: `64`).
- *num_layers*: integer specifying the number of heterogeneous GNN layers (default: `4`)
2- *hidden_channels*: integer specifying the dimensionality of the hidden node representations (default: `64`).
- *dropout*: float specifying the dropout probability applied during training (default: `0.3`).
- *learning_rate*: float specifying the optimizer learning rate (default: `0.001`).
- *robustness*: parameter controlling the robustness experiment, i.e., the amount of training supervision retained when evaluating sensitivity to reduced label availability (default: `100`).
- *results_dir*: string specifying the directory in which experimental results are stored.

  Note:
For training and evaluating the model, *run_LANTERN_and_HGNN_models.sh* can be modified with your data directory and your source code directory (the latter to be added to the PYTHONPATH environment variable).

## Reference

> Coming soon...
