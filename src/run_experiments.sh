#!/bin/bash

dataset_name="dooway" #1
region="Basilicata"
feats_dim="256"

training_seeds=(42 123 12345 123123 2025)

base_dir="/home/martirano/$dataset_name/$region"
embeddings_dir="$base_dir/embeddings"
models_dir="$base_dir/best_models"
results_dir="$base_dir/results"

for dir in "$embeddings_dir" "$models_dir" "$results_dir"; do
    if [ ! -d "$dir" ]; then
        echo "Creating directory: $dir"
        mkdir -p "$dir"
    fi
done

export PYTHONPATH=$PYTHONPATH:/home/martirano/Projects/Dooway_GNN/src


for seed_index in "${!training_seeds[@]}"  # Iterate over seed indices
do
    seed="${training_seeds[$seed_index]}"  # Get actual seed value
    echo "### Running experiment with dataset: $dataset_name, seed: $seed ###"
    python src/main_ES.py "$dataset_name" "$feats_dim" "$target_type" "$num_layers" "$seed_index" "$seed" "$embeddings_dir" "$models_dir", "$results_dir"
done
