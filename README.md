Implementation of LANTERN: LM-Assisted graph attention Network for Tourist pERsoNalization. #, as presented in our paper:

<!--
```
Comito, C., Falcone, A., Forestiero, A., Martirano, L.,
LLM-Assisted Multimodal Graph Attention network for Multilabel Tourist Profiling 
Information Fusion (2026).
```
-->

>Our proposed LANTERN...
The objective is to ...

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
XXX

## Training and evaluation
For training LANTERN, run the script main.py (main_late_fusion.py, resp.) specifying the following parameters:
- *dataset_name*, string specifying the name of the dataset to be processed
- *model_name*, string specifying the name of the model... (default: "LANTERN")
- *feats_dim*, the size of node features (default: 256)
- *target_type* (default: "user")
- *multilabel*, whether task is multilabel (true/false)
- *use_attrs*
- *use_imgs*
- *seed*
- *num_layers*
- *hidden_channels*
- *dropout*
- *learning_rate*
- *robustness*
- *results_dir*

  Note:
For training and evaluating the model, *run_LANTERN_and_HGNN_models.sh* can be modified with your data directory and your source code directory (the latter to be added to the PYTHONPATH environment variable).

## Reference

> Coming soon...
