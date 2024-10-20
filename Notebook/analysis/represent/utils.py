import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

import argparse

import torch
import numpy as np
from copy import deepcopy
import json
import torch.nn as nn

import matplotlib.pyplot as plt

from src.data.DataList import dataset_dict
from src.simulator.utils import generate_model, get_client_dataset, get_key_by_value


def load_config(config_path, proj_name):
    PATH = os.path.join(config_path, f"config_{proj_name}.json")
    with open(PATH, 'r') as f:
        config = json.load(f)
    return config

def load_model(model_name, modelPATH, config, device):
    device = torch.device('cpu')
    PATH = os.path.join(modelPATH, model_name)
    model_dict = torch.load(PATH, map_location=device)

    global_model = generate_model(config).to(device)
    
    if (config.agg_method != "Center") and (config.agg_method != "Local"):
        global_model.load_state_dict(model_dict['global_model'], strict=False)
        local_model_dict = model_dict['local_model']
        
    else:
        global_model.load_state_dict(model_dict['model'], strict=False)
        local_model_dict = None

    return global_model, local_model_dict


def load_all_models(model_type_dict, basePATH, device, batch_size=128, num_workers=8):
    
    result_dict = deepcopy(model_type_dict)
    
    for _type, wandb_id in model_type_dict.items():
        print(f"Loading {_type} model named [{wandb_id}]")
        ckptPATH = os.path.join(basePATH, f"{_type}/{wandb_id}")
        config = load_config(ckptPATH, wandb_id)
        
        config['batch_size'] = batch_size
        config['num_workers'] = num_workers
        config['nowandb'] = True
        config = argparse.Namespace(**config)
        
        main_name = wandb_id.split('_')[0]
        
        if _type == 'Center':
            model_name = 'Center_best_model.pth'
            glob_model, _ = load_model(model_name, ckptPATH, config, device)
            result_dict[_type] = glob_model.cpu()
        
        elif _type == 'Local':
            loc_model_list = []
            for i in range(config.num_clients):
                model_name = f'C{str(i).zfill(2)}_best_model.pth'
                client_model, _ = load_model(model_name, ckptPATH, config, device)
                loc_model_list.append(client_model.cpu())
            result_dict[_type] = loc_model_list
                
        else:
            model_name = f"{main_name}_best_round_100.pth"
            glob_model, _ = load_model(model_name, ckptPATH, config, device)
            result_dict[_type] = glob_model.cpu()
    
    return result_dict, config


def load_all_client_loader(config):
    result_list = []
    TestDataset_dict = get_client_dataset(config, config.num_clients, 
                                    _mode='test', verbose=False, 
                                    PATH=config.data_path,
                                    get_info=True)
    for i in range(config.num_clients):
        
        temp_loader = torch.utils.data.DataLoader(TestDataset_dict[i],
                                                  batch_size=config.batch_size, shuffle=False,
                                                  num_workers=0)
        result_list.append(temp_loader)
        
    return result_list
    