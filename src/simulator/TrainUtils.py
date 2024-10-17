import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from torch import nn
from copy import deepcopy
from tqdm import tqdm

from src.metric.function import MAE, MSE


def LocalUpdate(client_idx, global_model, learning_rate, TrainDataset_dict, config, device):
    """
    Args:
        client_idx: int
        global_model: torch.nn.Module
        learning_rate: float
        TrainDataset_dict: dict {client_idx: FLDataset}
        config: argparse.Namespace
        device: torch.device

    Returns:
        local_model.state_dict():
    """

    TrainLoader = torch.utils.data.DataLoader(TrainDataset_dict[client_idx], 
                                              batch_size=config.batch_size, 
                                              shuffle=True, num_workers=config.num_workers)

    local_model = deepcopy(global_model).to(device)

    if config.optimizer == 'adam':
        optimizer = torch.optim.Adam(local_model.parameters(), lr=learning_rate)
    elif config.optimizer == 'sgd':
        optimizer = torch.optim.SGD(local_model.parameters(), lr=learning_rate, momentum=config.momentum)
       
    criterion = nn.MSELoss()

    for epoch in range(config.epochs):
        local_model.train()
        epoch_loss = 0
        mae = 0

        progress_bar = tqdm(enumerate(TrainLoader), total=len(TrainLoader), ncols=100)
        
        for batch_idx, batch in progress_bar:
            images, labels = batch[0].to(device), batch[1].to(device)

            optimizer.zero_grad()
            output = local_model(images)

            if config.agg_method == 'FedAvg':
                loss = criterion(output.squeeze(), labels.squeeze())

            elif config.agg_method == 'FedProx':
                proximal_term = 0
                proximal_mu = config.proximal_mu
                for local_w, glob_w in zip(local_model.parameters(), global_model.parameters()):
                    proximal_term += torch.square((local_w - glob_w).norm(2))

                loss = criterion(output.squeeze(), labels.squeeze()) + (proximal_mu / 2) * proximal_term

            loss.backward()
            torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=3.0)
            optimizer.step()

            epoch_loss += loss.item()

            mae += MAE(output.detach().cpu().numpy().squeeze(), 
                             labels.detach().cpu().numpy().squeeze())

            progress_bar.set_postfix({
                                    "Client": client_idx,
                                    "[Train] epoch": epoch+1,
                                    "MSE_loss": round(epoch_loss / (batch_idx + 1), 3),
                                    "MAE_loss": round(mae / (batch_idx + 1), 3),
                                    })

    return local_model.cpu().state_dict()


def inference(client_idx, global_model, local_weight, TestDataset_dict, config, device):
    
    TestLoader = torch.utils.data.DataLoader(TestDataset_dict[client_idx],
                                             batch_size=config.batch_size, 
                                             shuffle=False, num_workers=config.num_workers)
    
    global_model.eval()
    criterion = nn.MSELoss()
    mae = 0
    test_loss = 0

    if config.personalized:
        local_model = deepcopy(global_model).to(device)
        local_model.load_state_dict(local_weight[client_idx])
        local_model.eval()

    for step, batch in enumerate(TestLoader):
        images, labels = batch[0].to(device), batch[1].to(device)

        if config.personalized:
            output = local_model(images)
        else:
            output = global_model(images)

        test_loss += criterion(output.squeeze(), labels.squeeze()).item()
        
        mae += MAE(output.detach().cpu().numpy().squeeze(), 
                   labels.detach().cpu().numpy().squeeze())
        
    return (test_loss / len(TestLoader), mae / len(TestLoader))
        


    






    

    
                                                