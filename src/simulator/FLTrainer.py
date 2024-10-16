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
    assert (config.device_id is not None) or (config.use_ddp), 'Please specify device_id'

    start = time.time()

    if config.use_ddp:
        import torch.distributed as dist

        dist.init_process_group(backend='nccl', init_method='env://')
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])

        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
        print(f"Using device {device} on rank {rank}")

        if rank == 0 and not config.nowandb:
            run_wandb = init_wandb(config)

    else:
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
    # print(f"Update Weight per Client", )

    # Model
    global_model = generate_model(config).to(device)

    aggregator = Aggregator(global_model, config)

    if config.use_ddp:
        global_model = torch.nn.DataParallel(global_model)

    for _round in range(config.round):
        local_results = {}
        local_weights = {}
    
        for client_idx in range(config.num_clients):
            print(f"################################### Round {_round} | Client {client_idx} Training ###################################")
            local_model_weight, local_result = LocalUpdate(client_idx, _round, global_model, 
                                                           TrainDataset_dict, ValDataset_dict,
                                                           config, device, run_wandb)

            local_weights[client_idx] = local_model_weight
            local_results[client_idx] = local_result

        global_model = aggregator.aggregate(local_weights, client_data_num_dict, update_weight_per_client)
        agg_loss, agg_mae = aggregate_result(local_results)

        if not config.nowandb:
            run_wandb.log({'round': _round,
                           'Agg-Train_Loss': agg_loss[0],
                           'Agg-Valid_Loss': agg_loss[1],
                           'Agg-Train_MAE': agg_mae[0],
                           'Agg-Valid_MAE': agg_mae[1]})
        
    for client_idx in range(config.num_clients):
        test_result = inference(client_idx, global_model, local_weights, 
                                TestDataset_dict, config, device)

        if not config.nowandb:
            run_wandb.log({'round': _round,
                            f'Client_{client_idx}-Test_Loss': test_result[0],
                            f'Client_{client_idx}-Test_MAE': test_result[1]})

    end = time.time()

    if not config.nowandb:
        running_time = end-start
        minutes = int(running_time // 60)
        seconds = running_time % 60
        run_wandb.log({'Running Time': running_time})
        print(f"Total Running Time: {minutes}m {seconds}s")
        run_wandb.finish()


if __name__ == '__main__':
    
    parser = FLconfig()
    config = parser.parse_args()
    main(config)