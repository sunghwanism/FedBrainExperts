import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import time

import torch
import torch.nn as nn
import wandb
import json

from monai.utils import set_determinism

from utils import init_wandb, get_client_dataset, MergeClientDataset
from simulator.configuration.localconfig import LocalConfig

from TrainUtils import LocalTrain, SaveBestResult


def main(config):
    assert (config.device_id is not None), 'Please specify device_id'
    total_train_time = 0
    start = time.time()

    torch.cuda.set_device(config.device_id)
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

    if not config.nowandb:
        run_wandb = init_wandb(config)
    else:
        run_wandb = None

    set_determinism(seed=config.seed)
    torch.backends.cudnn.benchmark = False

    # DataLoader
    TrainDataset_dict = get_client_dataset(config, config.num_clients, 
                                        _mode='train', verbose=False, 
                                        get_info=True, PATH=config.data_path)
    
    ValDataset_dict = get_client_dataset(config, config.num_clients, 
                                        _mode='val', verbose=False, 
                                        get_info=True, PATH=config.data_path)
    
    # TestDataset_dict = get_client_dataset(config, config.num_clients, 
    #                                     _mode='test', verbose=False, 
    #                                     PATH=config.data_path,
    #                                     get_info=True)
    run_name = None
    
    if not config.nowandb:
        run_name = wandb.run.name
        config_dict = vars(config)
        configPath = os.path.join(config.save_path, config.agg_method, f'config_{wandb.run.name}.json')
        with open(configPath, 'w') as f:
            json.dump(config_dict, f, indent=4)

    if config.agg_method == 'Local':

        for client_idx in range(config.num_clients):
            bestmodel = LocalTrain(client_idx, TrainDataset_dict, ValDataset_dict, run_wandb, config, device, run_name)
            local_train_time = time.time()-start
            total_train_time += local_train_time

    elif config.agg_method == 'Center':

        TrainDataset = MergeClientDataset(TrainDataset_dict, config.num_clients)
        ValDataset = MergeClientDataset(ValDataset_dict, config.num_clients)

        bestmodel = LocalTrain(-1, TrainDataset, ValDataset, run_wandb, config, device, run_name)
        local_train_time = time.time()-start
        total_train_time += local_train_time

    # for client_idx in range(config.num_clients):
    #     SaveBestResult(client_idx, bestmodel, TrainDataset_dict, ValDataset_dict, TestDataset_dict, run_wandb, config, device)

    if not config.nowandb:
        run_wandb.log({'Average Train Time': total_train_time/config.num_clients})
        run_wandb.finish()



if __name__ == '__main__':
    parser = LocalConfig()
    config = parser.parse_args()
    main(config)
