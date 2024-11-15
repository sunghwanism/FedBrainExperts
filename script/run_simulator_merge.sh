#!/bin/bash

methods=(Center)

for m in "${methods[@]}"
do
    python src/simulator/CentTrainer.py --device_id 3 --agg_method "$m" --wandb_project Fine_Center_thesis_v2 --optimizer sgd --lr 2e-6 --batch_size 64 --seed 40
    sleep 60
done

python src/simulator/CentTrainer.py --device_id 3 --agg_method Local --wandb_project Fine_Local_thesis_v2 --optimizer sgd --lr 2e-6 --batch_size 64 --seed 40