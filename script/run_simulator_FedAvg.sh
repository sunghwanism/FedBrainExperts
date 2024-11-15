#!/bin/bash

seeds=(10 20 30)

for seed in "${seeds[@]}"
do
    python src/simulator/FLTrainer.py --device_id 1 --agg_method FedAvg --wandb_project Fine_FL_Thesis_v5 --optimizer sgd --lr 2e-6 --batch_size 64 --seed "$seed"
    sleep 60
done