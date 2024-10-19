import os
import sys
sys.path.append('../../')

import torch
import numpy as np

import json
from src.simulator.utils import generate_model, init_wandb, get_client_dataset


def CKA(A, B):
    pass


def load_config(config_path, proj_name):
    PATH = os.path.join(config_path, f"config_{proj_name}.json")
    with open(PATH, 'r') as f:
        config = json.load(f)
    return config

def load_model(model_name, modelPATH, config, device):
    PATH = os.path.join(modelPATH, model_name)
    model_dict = torch.load(PATH, map_location=device)

    global_model = generate_model(config).to(device)
    if config.agg_method != ("Center" or "Local"):
        global_model.load_state_dict(model_dict['global_model'], strict=False)
        local_model_dict = model_dict['local_model']
    else:
        global_model.load_state_dict(model_dict, strict=False)
        local_model_dict = None

    return global_model, local_model_dict