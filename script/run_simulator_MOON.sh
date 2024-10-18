#!/bin/bash

mu_values=(1 5 10)

for mu in "${mu_values[@]}"
do
    python src/simulator/FLTrainer.py --device_id 2 --agg_method MOON --proximal_mu "$mu" --batch_size 50
    sleep 60
done