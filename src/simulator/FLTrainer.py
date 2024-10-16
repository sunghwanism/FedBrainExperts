import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import time
from tqdm import tqdm

import torch

from monai.utils import set_determinism

from utils import generate_model, init_wandb, get_client_dataset
from config import FLconfig



def main(config):
    assert (config.device_id is not None) or (config.use_ddp), 'Please specify device_id'

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
            init_wandb(config)

    else:
        torch.cuda.set_device(config.device_id)
        device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

        if not config.nowandb:
            init_wandb(config)

    set_determinism(seed=config.seed)
    torch.backends.cudnn.benchmark = False

    # DataLoader
    TrainDataset_dict = get_client_dataset(config.data_idx, config.num_clients, 
                                           _mode='train', verbose=False, 
                                           get_info=False, PATH=config.data_path)
    
    ValDataset_dict = get_client_dataset(config.data_idx, config.num_clients, 
                                         _mode='val', verbose=False, 
                                         get_info=False, PATH=config.data_path)
    
    TestDataset_dict = get_client_dataset(config.data_idx, config.num_clients, 
                                         _mode='test', verbose=False, 
                                         get_info=False, PATH=config.data_path)


    # Model
    global_model = generate_model(config).to(device)
    if config.use_ddp:
        global_model = torch.nn.DataParallel(global_model)
    
    global_weights = global_model.state_dict()


    train_loss = {}
    net_list = []

    for _round in range(config.round):

        local_weight, local_loss = {}, {}
        global_model.train()

        for client_idx in range(config.num_clients):
            local_model = generate_model(config).to(device)
            local_model.load_state_dict(global_weights)
            local_model.train()

            for epoch in range(config.epochs):

                LocalUpdate(client_idx, model, TrainLoader, ValLoader, config, wandb)

        


    
    

    

if __name__ == '__main__':

    start = time.time()
    parser = FLconfig()
    config = parser.parse_args()
    main(config)
    end = time.time()

    print(f"Running Time: {end-start}")