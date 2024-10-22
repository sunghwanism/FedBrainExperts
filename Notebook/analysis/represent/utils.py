import os
import sys
sys.path.append('../../../')

import argparse

import torch
import numpy as np
from copy import deepcopy
import json
        
import torch.nn as nn
from Notebook.analysis.represent.cka import CKACalculator
import matplotlib.pyplot as plt
from src.data.DataList import dataset_dict

from src.simulator.utils import generate_model, get_client_dataset, get_key_by_value


def load_config(config_path, proj_name):
    PATH = os.path.join(config_path, f"config_{proj_name}.json")
    with open(PATH, 'r') as f:
        config = json.load(f)
    return config

def load_model(model_name, modelPATH, config, device):
    # device = torch.device('cpu')
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
    torch.cuda.set_device(device)
    result_dict = deepcopy(model_type_dict)
    
    for _type, (wandb_id, best_round) in model_type_dict.items():
        print(f"Loading {_type} model named [{wandb_id}]")
        ckptPATH = os.path.join(basePATH, f"{_type}/{wandb_id}")
        config = load_config(ckptPATH, wandb_id)
        
        config['batch_size'] = batch_size
        config['num_workers'] = num_workers
        config['nowandb'] = True
        config = argparse.Namespace(**config)
        
        # main_name = wandb_id.split('_')[0]
        
        if _type == 'Center':
            model_name = f'Center_best_model_{wandb_id}.pth'
            glob_model, _ = load_model(model_name, ckptPATH, config, device)
            result_dict[_type] = glob_model.cpu()
            glob_model.to(device)
        
        elif _type == 'Local':
            loc_model_list = []
            for i in range(config.num_clients):
                model_name = f'C{str(i).zfill(2)}_best_model_{wandb_id}.pth'
                client_model, _ = load_model(model_name, ckptPATH, config, device)
                loc_model_list.append(client_model.cpu())
                client_model.to(device)
            result_dict[_type] = loc_model_list
                
        else:
            # model_name = f"{main_name}_best_round_{str(best_round).zfill(3)}.pth"
            model_name = f"{wandb_id}_best_model.pth"
            glob_model, _ = load_model(model_name, ckptPATH, config, device)
            result_dict[_type] = glob_model.cpu()
            glob_model.to(device)
    
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


def vizualize_cka_model(model_dict, client_loader_list, device, criterion='Local'):
    torch.cuda.set_device(device)
    fig, axs = plt.subplots(2, 5, figsize=(20, 3))
    
    layers = (nn.Conv3d, nn.Linear)
    model_types = list(model_dict.keys())
    model_types.remove(criterion)
    y_ticks = [f"{_type}" for _type in model_types]
    x_ticks = [layer_num for layer_num in range(1, 19)]
    
    for client_idx in range(len(client_loader_list)):
        ax = axs[client_idx//5, client_idx%5]
        dataLoader = client_loader_list[client_idx]
        client_cka = []
        for _type in model_types:
            calculator = CKACalculator(model1=model_dict[_type].to(device), 
                                       model2=model_dict[criterion][client_idx].to(device), 
                                       dataloader=dataLoader, 
                                       hook_layer_types=layers,
                                       num_epochs=1,
                                       epsilon=1e-5,)
            
            cka_output = calculator.calculate_cka_matrix().detach().cpu().numpy()
            cka_output = np.nan_to_num(cka_output, nan=0, posinf=1, neginf=1)
            cka_output = np.diag(cka_output)
            cka_output = np.delete(cka_output, [7, 12, 17])
            
            client_cka.append(cka_output)
            calculator.reset()
            torch.cuda.empty_cache()
        
        image = ax.imshow(client_cka, cmap='inferno', vmin=0, vmax=1)
        ax.set_xticks(range(len(x_ticks)))
        ax.set_xticklabels(x_ticks)
        ax.set_yticks(range(len(y_ticks)))
        ax.set_yticklabels(y_ticks)
        ax.set_title(f"Client {client_idx} || {get_key_by_value(dataset_dict, client_idx)} (n={len(dataLoader.dataset)*10})",
                     fontsize=12)
        
    # fig.subplots_adjust(right=0.85)
    # cbar_ax = fig.add_axes([1.03, 0.15, 0.02, 0.7])
    # cbar = fig.colorbar(image, cax=cbar_ax, fraction=0.046, pad=0.04)
    # cbar.set_label('CKA Value')
    
    plt.tight_layout()
    plt.show()
            
        
    