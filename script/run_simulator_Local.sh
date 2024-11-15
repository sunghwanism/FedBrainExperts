#!/bin/bash

seeds=(10 20 30)

for seed in "${seeds[@]}"
do
    python src/simulator/CentTrainer.py --device_id 2 --agg_method Local --wandb_project Fine_Local_thesis_v2 --optimizer sgd --lr 2e-6 --batch_size 64 --seed "$seed"
    sleep 30
done