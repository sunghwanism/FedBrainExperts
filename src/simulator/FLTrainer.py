import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import time
from tqdm import tqdm

import torch
import wandb

from monai.utils import set_determinism

from utils import generate_model, init_wandb, get_client_dataset
from config import FLconfig
from TrainUtils import LocalUpdate, inference
from model.agg.aggmodule import Aggregator
from src.metric.function import MAE, aggregate_result


def main(config):
    assert (config.device_id is not None), 'Please specify device_id'

    start = time.time()

    torch.cuda.set_device(config.device_id)
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

    if not config.nowandb:
        run_wandb = init_wandb(config)

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
    update_weight = [round(client_data_num_dict[client_idx]/sum(client_data_num_dict.values()), 2) for client_idx in range(config.num_clients)]
    update_weight_per_client = {client_idx: update_weight[client_idx] for client_idx in range(config.num_clients)}

    # Model
    global_model = generate_model(config).to(device)
    aggregator = Aggregator(global_model, device, config)

    for _round in range(config.round):
        _round += 1

        local_weights = {}
    
        for client_idx in range(config.num_clients):
            print(f"#################################### Round {_round} | Client {client_idx} Training ####################################")
            local_model_weight = LocalUpdate(client_idx, global_model, 
                                             TrainDataset_dict, config, device)

            local_weights[client_idx] = local_model_weight

        # Test the global model with Train, Validation and Test dataset
        global_model = aggregator.aggregate(local_weights, update_weight_per_client)
        
        for client_idx in range(config.num_clients):
            train_result = inference(client_idx, global_model, local_weights, 
                                    TrainDataset_dict, config, device)
            valid_result = inference(client_idx, global_model, local_weights, 
                                    ValDataset_dict, config, device)
            test_result = inference(client_idx, global_model, local_weights, 
                                    TestDataset_dict, config, device)

            if not config.nowandb:
                run_wandb.log({
                    "round": _round,
                    'Client': client_idx,
                    f"Client_{client_idx}-Train_Loss": round(train_result[0], 3),
                    f"Client_{client_idx}-Train_MAE": round(train_result[1], 3),
                    f"Client_{client_idx}-Valid_Loss": round(valid_result[0], 3),
                    f"Client_{client_idx}-Valid_MAE": round(valid_result[1], 3),
                    f"Client_{client_idx}-Test_Loss": round(test_result[0], 3),
                    f"Client_{client_idx}-Test_MAE": round(test_result[1], 3),
                })

        del train_result, valid_result, test_result
        torch.cuda.empty_cache()

    end = time.time()

    running_time = end-start
    minutes = int(running_time // 60)
    seconds = running_time % 60        
    print(f"Total Running Time: {minutes}m {seconds}s")

    if not config.nowandb:
        run_wandb.log({'Running Time': running_time})
        run_wandb.finish()


if __name__ == '__main__':
    
    parser = FLconfig()
    config = parser.parse_args()
    main(config)