import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

import argparse

import torch
import numpy as np
import pandas as pd
from copy import deepcopy
import json
import torch.nn as nn

import matplotlib.pyplot as plt

from src.data.DataList import dataset_dict
from src.metric.function import MAE
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

def load_all_client_loader(config, _mode='test'):
    if _mode == 'test':
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

    else:
        train_result_list = []
        val_result_list = []
        test_result_list = []
        TrainDataset_dict = get_client_dataset(config, config.num_clients,
                                                  _mode='train', verbose=False,
                                                  PATH=config.data_path,
                                                  get_info=True)
        ValDataset_dict = get_client_dataset(config, config.num_clients,
                                             _mode='val', verbose=False,
                                             PATH=config.data_path,
                                             get_info=True)

        TestDataset_dict = get_client_dataset(config, config.num_clients, 
                                              _mode='val', verbose=False, 
                                              PATH=config.data_path,
                                              get_info=True)
        
        for i in range(config.num_clients):
            temp_loader = torch.utils.data.DataLoader(TrainDataset_dict[i],
                                                    batch_size=config.batch_size, shuffle=False,
                                                    num_workers=0)
            train_result_list.append(temp_loader)

            temp_loader = torch.utils.data.DataLoader(ValDataset_dict[i],
                                                      batch_size=config.batch_size, shuffle=False,
                                                      num_workers=0)
            val_result_list.append(temp_loader)

            temp_loader = torch.utils.data.DataLoader(TestDataset_dict[i],
                                                      batch_size=config.batch_size, shuffle=False,
                                                      num_workers=0)
            
            test_result_list.append(temp_loader)

        return train_result_list, val_result_list, test_result_list
    
def custom_collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    ages = torch.stack([item[1] for item in batch])
    
    if len(batch[0]) > 2:
        subjects = [item[2] for item in batch]
        sexes = [item[3] for item in batch]
        return images, ages, subjects, sexes
    
    return images, ages


def get_client_result(client_idx, model_dict, TrainLoader, ValLoader, TestLoader, device, savepath, model_type_dict):
    result_dict = {}

    for model_type, model in model_dict.items():
        result_dict[model_type] = None

        if model_type == 'Local':
            model = model[client_idx].to(device)
            model.eval()
        else:
            model.to(device)
            model.eval()

        result_df = pd.DataFrame()
        criterion = nn.MSELoss()

        Loaders = [TrainLoader[client_idx], ValLoader[client_idx], TestLoader[client_idx]]
        mode_list = ['Train', 'Valid', 'Test']
        
        for _mode, Loader in zip(mode_list, Loaders):
            
            pred_age = []
            true_age = []
            Subject_list = []
            Sex_list = []
            col_mode = [_mode]*len(Loader.dataset)

            mae = 0
            epoch_loss = 0

            for batch in Loader:
                with torch.no_grad():
                    (images, labels, Subject, Sex) = batch[0].to(device), batch[1].to(device), batch[2], batch[3]
                    output = model(images)
                    loss = criterion(output.squeeze(), labels.squeeze())
                    epoch_loss += loss.item()
                    mae += MAE(output.detach().cpu().numpy().squeeze(), 
                            labels.detach().cpu().numpy().squeeze())
                    
                    if output.size(0) > 1:
                        output = output.detach().cpu().numpy().squeeze()
                        labels = labels.detach().cpu().numpy().squeeze()
                        Subject = np.array(Subject).squeeze()
                        Sex = np.array(Sex).squeeze()
                    
                    else:
                        output = output.detach().cpu().numpy()
                        labels = labels.detach().cpu().numpy()
                        Subject = np.array(Subject)
                        Sex = np.array(Sex)
                    
                    pred_age.extend(output)
                    true_age.extend(labels)
                    Subject_list.extend(Subject)
                    Sex_list.extend(Sex)
            
            mae = mae / len(Loader)
            epoch_loss = epoch_loss / len(Loader)
                
            temp_df = pd.DataFrame({'Subject': Subject_list,
                                    'Sex': Sex_list,
                                    'Age': true_age,
                                    'pred_age': pred_age,
                                    'mode': col_mode})
            
            result_df = pd.concat([result_df, temp_df], axis=0)

            if _mode == 'Test':
                result_dict[model_type] = mae

            if not os.path.exists(os.path.join(savepath, model_type)):
                os.makedirs(os.path.join(savepath, model_type))
            
            if not os.path.exists(os.path.join(savepath, model_type, model_type_dict[model_type][0])):
                os.makedirs(os.path.join(savepath, model_type, model_type_dict[model_type][0]))

            SAVEPATH = os.path.join(savepath, model_type, model_type_dict[model_type][0])

            data_name = get_key_by_value(dataset_dict, client_idx)
            result_df = result_df.applymap(convert_to_float)
                
            if model_type == 'Local':
                result_df.to_csv(os.path.join(SAVEPATH,
                                            f"C{str(client_idx).zfill(2)}_{data_name}_{model_type_dict[model_type][0]}_local.csv"), index=False)
            
            elif model_type == 'Center':
                result_df.to_csv(os.path.join(SAVEPATH,
                                            f"C{str(client_idx).zfill(2)}_{data_name}_{model_type_dict[model_type][0]}_center.csv"), index=False)
            
            else:
                result_df.to_csv(os.path.join(SAVEPATH,
                                            f"C{str(client_idx).zfill(2)}_{data_name}_{model_type_dict[model_type][0]}_{model_type}.csv"), index=False)

    return result_dict


def convert_to_float(value):
    try:
        # If it's a list (like [39.02573]), extract the float
        if isinstance(value, str) and value.startswith('[') and value.endswith(']'):
            return float(value.strip('[]'))
        # Try converting directly to float otherwise
        return float(value)
    except ValueError:
        return value  # If it can't be converted, return as is (handle error case)