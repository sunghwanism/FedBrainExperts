#!/bin/bash

mu_values=(1 5 10)

for mu in "${mu_values[@]}"
do
    python src/simulator/FLTrainer.py --device_id 1 --agg_method FedProx --proximal_mu "$mu" --batch_size 64
    sleep 60
done