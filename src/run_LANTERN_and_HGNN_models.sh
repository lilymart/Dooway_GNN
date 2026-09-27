#!/bin/bash

export PYTHONPATH=$PYTHONPATH:/Projects/Dooway_GNN/

dataset_name="Basilicata"
num_layers=4
hidden_channels=64
dropout=0.3
learning_rate=0.001

training_seeds=(42 96 25 2025 123)
models=("LANTERN" "HAN" "HGT" "HGT_single_head" "SimpleHGN")

results_root="/home/jovyan/data/dooway/$dataset_name/subset/results/competitors"

for model_name in "${models[@]}"
do
    for seed in "${training_seeds[@]}"
    do
        echo "### Running $model_name with seed: $seed ###"
        python src/main.py \
            --model_name "$model_name" \
            --dataset_name "$dataset_name" \
            --seed "$seed" \
            --num_layers "$num_layers" \
            --hidden_channels "$hidden_channels" \
            --dropout "$dropout" \
            --learning_rate "$learning_rate" \
            --results_dir "$results_root/$model_name"
    done
done
