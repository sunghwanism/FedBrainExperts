import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import time
from tqdm import tqdm

import torch
import wandb
import json
from copy import deepcopy

from monai.utils import set_determinism

from utils import generate_model, init_wandb, get_client_dataset
from config import FLconfig
from TrainUtils import LocalUpdate, inference
from model.agg.aggmodule import Aggregator


def main(config):
    assert (config.device_id is not None), 'Please specify device_id'
    total_train_time = 0
    start = time.time()

    if config.agg_method == 'FedKLIEP':
        assert config.model == 'RepResNet', 'Only RepResNet is available for FedKLIEP'

    torch.cuda.set_device(config.device_id)
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

    if not config.nowandb:
        run_wandb = init_wandb(config)
        if not os.path.exists(config.save_path):
            os.makedirs(config.save_path)
        if not os.path.exists(os.path.join(config.save_path, config.agg_method)):
            os.makedirs(os.path.join(config.save_path, config.agg_method))
        if not os.path.exists(os.path.join(config.save_path, config.agg_method, wandb.run.name)):
            os.makedirs(os.path.join(config.save_path, config.agg_method, wandb.run.name))

    set_determinism(seed=config.seed)
    torch.backends.cudnn.benchmark = False


    # DataLoader
    TrainDataset_dict = get_client_dataset(config, config.num_clients, 
                                           _mode='train', verbose=False, 
                                           get_info=False, PATH=config.data_path)
    
    ValDataset_dict = get_client_dataset(config, config.num_clients, 
                                         _mode='val', verbose=False, 
                                         get_info=False, PATH=config.data_path)
    
    TestDataset_dict = get_client_dataset(config, config.num_clients, 
                                         _mode='test', verbose=False, 
                                         get_info=False, PATH=config.data_path)

    client_data_num_dict = {client_idx: len(TrainDataset_dict[client_idx]) for client_idx in range(config.num_clients)}
    update_weight = [client_data_num_dict[client_idx]/sum(client_data_num_dict.values()) for client_idx in range(config.num_clients)]
    update_weight_per_client = {client_idx: update_weight[client_idx] for client_idx in range(config.num_clients)}

    # Model
    global_model = generate_model(config).to(device)
    aggregator = Aggregator(global_model, device, config)

    best_valid_MAE = float('inf')

    if not config.nowandb:
        config_dict = vars(config)
        configPath = os.path.join(config.save_path, config.agg_method, wandb.run.name, f'config_{wandb.run.name}.json')
        with open(configPath, 'w') as f:
            json.dump(config_dict, f, indent=4)

    learning_rate = config.lr

    if config.agg_method == 'FedKLIEP':
        imp_w_dict = {client_idx: [1,1,1,1] for client_idx in range(config.num_clients)}
    else:
        imp_w_dict = None

    local_weights = {}

    if config.agg_method == 'MOON':
        prev_local_model = deepcopy(global_model).to(device)
    else:
        prev_local_model = None

    for _round in range(config.round):
        round_start = time.time()
        _round += 1
    
        # learning_rate *= 0.995 # learning rate scheduler
    
        for client_idx in range(config.num_clients):
            print(f"################################################################ Round {_round} | Client {client_idx} Training ################################################################")
            
            if _round > 1 :
                if config.agg_method == 'MOON':
                    prev_local_model = deepcopy(global_model)
                    prev_local_model.load_state_dict(local_weights[client_idx])

            local_model_weight, imp_w_list = LocalUpdate(client_idx, global_model, learning_rate,
                                                         TrainDataset_dict, config, device, _round, prev_local_model,
                                                         imp_w_list=imp_w_dict)

            local_weights[client_idx] = local_model_weight

            if config.agg_method == 'FedKLIEP':
                imp_w_dict[client_idx] = imp_w_list

            if config.agg_method == 'MOON':
                del local_model_weight, prev_local_model
            else:
                del local_model_weight

            torch.cuda.empty_cache()
        
        # Test the global model with Train, Validation and Test dataset
        global_model = aggregator.aggregate(local_weights, update_weight_per_client)

        round_end = time.time()
        round_time = round_end - round_start
        total_train_time += round_time

        minutes = int(round_time // 60)
        seconds = round_time % 60
        print(f"Round {_round} Time: {minutes}m {round(seconds,2)}s")

        if _round == 1 or _round % 2 == 0:
            for client_idx in range(config.num_clients):
                train_result = inference(client_idx, global_model, local_weights, 
                                        TrainDataset_dict, config, device, imp_w_dict)
                valid_result = inference(client_idx, global_model, local_weights, 
                                        ValDataset_dict, config, device, imp_w_dict)
                test_result = inference(client_idx, global_model, local_weights, 
                                        TestDataset_dict, config, device, imp_w_dict)

                if not config.nowandb:
                    run_wandb.log({
                        "round": _round,
                        f"Client_{client_idx}-Train_Loss": round(train_result[0], 3),
                        f"Client_{client_idx}-Train_MAE": round(train_result[1], 3),
                        f"Client_{client_idx}-Valid_Loss": round(valid_result[0], 3),
                        f"Client_{client_idx}-Valid_MAE": round(valid_result[1], 3),
                        f"Client_{client_idx}-Test_Loss": round(test_result[0], 3),
                        f"Client_{client_idx}-Test_MAE": round(test_result[1], 3),
                    })

            if (best_valid_MAE > valid_result[1] and _round >= 40):
                best_valid_MAE = valid_result[1]
                if config.agg_method == 'FedKLIEP':
                    save_dict = {
                        "round": _round,
                        "global_model": global_model.state_dict(),
                        "local_model": local_weights,
                        "imp_w_list": imp_w_list,
                    }

                else:
                    save_dict = {
                        "round": _round,
                        "global_model": global_model.state_dict(),
                        "local_model": local_weights,
                    }

                if not config.nowandb:
                    torch.save(save_dict, 
                            os.path.join(config.save_path, config.agg_method, wandb.run.name,
                                            f"{wandb.run.name}_best_model.pth"))
            if _round == 100:
                if config.agg_method == 'FedKLIEP':
                    save_dict = {
                        "round": _round,
                        "global_model": global_model.state_dict(),
                        "local_model": local_weights,
                        "imp_w_list": imp_w_list,
                    }
                else:
                    save_dict = {
                        "round": _round,
                        "global_model": global_model.state_dict(),
                        "local_model": local_weights,}
                    
                if not config.nowandb:
                    torch.save(save_dict, 
                            os.path.join(config.save_path, config.agg_method, wandb.run.name,
                                            f"{wandb.run.name}_round100_model.pth"))
                    
                del save_dict
                torch.cuda.empty_cache()

        del train_result, valid_result, test_result
        torch.cuda.empty_cache()

    end = time.time()

    running_time = end-start
    minutes = int(running_time // 60)
    seconds = running_time % 60
    print(f"***** Total Running Time: {minutes}m {round(seconds,2)}s *****")

    if not config.nowandb:
        run_wandb.log({'Total Train Time': total_train_time})
        run_wandb.finish()


if __name__ == '__main__':
    parser = FLconfig()
    config = parser.parse_args()
    main(config)