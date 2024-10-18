#!/usr/bin/bash

#SBATCH -J 
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=5
#SBATCH --mem-per-gpu=20G
#SBATCH -p batch_grad
#SBATCH -w ariel-v1
#SBATCH -t 1-00:00:00
#SBATCH -o /data/msh2044/logs/slurm-%A_FedProx_0.1.out


pwd
which python
hostname
python ./src/simulator/FLTrainer.py --device_id 0 --agg_method FedProx --proximal_mu 0.1

exit 0
