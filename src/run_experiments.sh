#!/bin/bash

export PYTHONPATH=$PYTHONPATH:/Projects/Dooway_GNN/

dataset_name="Basilicata"
feats_dim="256"
target_type="user"
multilabel="true"
num_layers=4
hidden_channels=64
dropout=0.3
learning_rate=0.001

training_seeds=(42 96 25 2025 123)

robustness=(10 20 40 60 80 100)

results_dir="/home/martirano/data/dooway/$dataset_name/subset/results/robustness"


for seed in "${training_seeds[@]}"
do
    echo "### Running experiment with seed: $seed ###"
    for perc in "${robustness[@]}"
    do
      python src/main.py \
          --dataset_name "$dataset_name" --seed "$seed" --num_layers "$num_layers" --hidden_channels "$hidden_channels" --learning_rate "$learning_rate" --robusteness "$perc" --results_dir "$results_dir"
    done
done

