import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from torch import nn
from copy import deepcopy
from tqdm import tqdm

from src.metric.function import MAE, MSE


def LocalUpdate(client_idx, global_model, learning_rate, TrainDataset_dict, config, device, prev_global_model=None):
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
        optimizer = torch.optim.SGD(local_model.parameters(), lr=learning_rate,
                                    momentum=config.momentum, weight_decay=0.00001)
       
    criterion = nn.MSELoss()

    if config.agg_method == 'MOON':
        contrastive_temp = config.contrastive_temp

        prev_global_model.eval()
        for param in prev_global_model.parameters():
            param.requires_grad = False
            
        prev_global_model.cuda()

        cos = torch.nn.CosineSimilarity(dim=-1)
        const_loss = torch.nn.CrossEntropyLoss().cuda()

    for epoch in range(config.epochs):
        local_model.train()
        epoch_loss = 0
        mae = 0
        epoch_const_loss = 0
        epoch_prd_loss = 0
        if config.agg_method == 'MOON':
            progress_bar = tqdm(enumerate(TrainLoader), total=len(TrainLoader), ncols=130)
        else:
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

            elif config.agg_method == 'MOON':

                output, represent_local = local_model(images, represent=True)
                _, represent_global = global_model(images, represent=True)
                
                pos_sim = cos(represent_local, represent_global).reshape(-1, 1)

                prev_global_model.to(device)
                with torch.no_grad():
                    _, represent_prev_global = prev_global_model(images, represent=True)

                neg_sim = cos(represent_local, represent_prev_global).reshape(-1, 1)
                logits = torch.cat((pos_sim, neg_sim), dim=1) / contrastive_temp
                const_labels = torch.zeros(logits.size(0)).to(device).long()

                prev_global_model.to("cpu")
                
                contrastive_loss = config.proximal_mu * const_loss(logits, const_labels)
                prd_loss = criterion(output.squeeze(), labels.squeeze())
                loss = contrastive_loss + prd_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=3.0)
            optimizer.step()

            epoch_loss += loss.item()

            if config.agg_method == 'MOON':
                epoch_const_loss += contrastive_loss.item()
                epoch_prd_loss += prd_loss.item()

            mae += MAE(output.detach().cpu().numpy().squeeze(), 
                             labels.detach().cpu().numpy().squeeze())
            
            if config.agg_method == 'MOON':
                progress_bar.set_postfix({
                                        "Client": client_idx,
                                        "[Train] epoch": epoch+1,
                                        "pred_loss": round(epoch_prd_loss / (batch_idx+1), 3),
                                        "cont_loss": round(epoch_const_loss / (batch_idx+1), 3),
                                        "MAE_loss": round(mae / (batch_idx + 1), 3),
                                        })
                
            else:
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
        


    






    

    
                                                