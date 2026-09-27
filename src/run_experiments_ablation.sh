#!/bin/bash

export PYTHONPATH=$PYTHONPATH:/Projects/Dooway_GNN/

dataset_name="Basilicata" #1
feats_dim="256"
target_type="user" #sys.argv[3]
multilabel="true"
num_layers=3 #2 #sys.argv[5]
hidden_channels=64
dropout=0.3
learning_rate=0.001 #0.001

training_seeds=(42 96 25 2025 123)

use_imgs="false"
use_attributes=("true" "false") #"false"

results_dir="/home/jovyan/data/dooway/$dataset_name/subset/results/ablation"


for seed in "${training_seeds[@]}"
do
    echo "### Running experiment with seed: $seed ###"
    for use_attrs in "${use_attributes[@]}"
    do
      python src/main_ES.py \
          --dataset_name "$dataset_name" --seed "$seed" --num_layers "$num_layers" --hidden_channels "$hidden_channels" --learning_rate "$learning_rate" --use_attrs "$use_attrs" --use_imgs "$use_imgs" --results_dir "$results_dir"
    done
done

