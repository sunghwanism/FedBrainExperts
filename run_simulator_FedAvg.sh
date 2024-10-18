#!/bin/bash

epochs=(1 10 20)

for epoch in "${epochs[@]}"
do
    python src/simulator/FLTrainer.py --device_id 0 --agg_method FedAvg --batch_size 64 --epoch "$epoch"
    sleep 60
done