#!/bin/bash

mu_values=(0.001 0.01 0.1 1)

for mu in "${mu_values[@]}"
do
    python src/simulator/FLTrainer.py --device_id 2 --agg_method FedProx --proximal_mu "$mu" --batch_size 52 --wandb_project Fine_FL_Thesis
    sleep 60
done