import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from torch import nn
from copy import deepcopy
from tqdm import tqdm

from src.metric.function import MAE, MSE


def LocalUpdate(client_idx, _round, global_model, TrainDataset_dict, ValDataset_dict, 
                config, device, run_wandb=None):
    """
    Args:
        client_idx: int
        global_model: torch.nn.Module
        TrainDataset_dict: dict {client_idx: FLDataset}
        config: argparse.Namespace
        run_wandb: wandb object
    
    Returns:
        local_model.state_dict(): dict {str: torch.Tensor}
        epoch_loss: float
    """

    TrainLoader = torch.utils.data.DataLoader(TrainDataset_dict[client_idx], 
                                              batch_size=config.batch_size, 
                                              shuffle=True, num_workers=config.num_workers)

    ValLoader = torch.utils.data.DataLoader(ValDataset_dict[client_idx], 
                                            batch_size=config.batch_size, 
                                            shuffle=False, num_workers=config.num_workers)

    local_model = deepcopy(global_model).to(device)
    local_model.train()

    if config.optimizer == 'adam':
        optimizer = torch.optim.Adam(local_model.parameters(), lr=config.lr)
    elif config.optimizer == 'sgd':
        optimizer = torch.optim.SGD(local_model.parameters(), lr=config.lr, momentum=config.momentum)
       

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
            loss = criterion(output.squeeze(), labels.squeeze())
            loss.backward()
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

        local_model.eval()
        
        train_epoch_loss = deepcopy(epoch_loss)
        train_mae = deepcopy(mae)

        epoch_loss = 0
        mae = 0
        val_progress_bar = tqdm(enumerate(ValLoader), total=len(ValLoader), ncols=100)

        for batch_idx, batch in val_progress_bar:
            images, labels = batch[0].to(device), batch[1].to(device)
            output = local_model(images)
            loss = criterion(output.squeeze(), labels.squeeze())
            epoch_loss += loss.item()

            mae += MAE(output.detach().cpu().numpy().squeeze(), 
                        labels.detach().cpu().numpy().squeeze())

            val_progress_bar.set_postfix({
                                            "Client": client_idx,
                                            "[Valid] Round": epoch+1,
                                            "MSE_loss": round(epoch_loss / (batch_idx + 1), 3),
                                            "MAE_loss": round(mae / (batch_idx + 1), 3),
                                            })
            
    if not config.nowandb:
        run_wandb.log({
            "round": _round,
            f"Client_{client_idx}-Train_Loss": round(train_epoch_loss / len(TrainLoader), 3),
            f"Client_{client_idx}-Train_MAE": round(train_mae / len(TrainLoader), 3),
            f"Client_{client_idx}-Valid_Loss": round(epoch_loss / len(ValLoader), 3),
            f"Client_{client_idx}-Valid_MAE": round(mae / len(ValLoader), 3),
        })

    return local_model.state_dict(), (train_epoch_loss / len(TrainLoader), 
                                      epoch_loss / len(ValLoader),
                                      train_mae / len(TrainLoader),
                                      mae / len(ValLoader))


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
        


    






    

    
                                                