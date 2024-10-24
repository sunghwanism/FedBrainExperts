import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from torch import nn
from copy import deepcopy
from tqdm import tqdm

from src.metric.function import MAE
from utils import generate_model

import torch
import torch.nn as nn

import pandas as pd
import numpy as np
from src.data.DataList import dataset_dict
from src.simulator.utils import custom_collate_fn, get_key_by_value, get_activation_for_models
from src.model.KLIEP import KLIEP



def LocalUpdate(client_idx, global_model, learning_rate, TrainDataset_dict, config, device, _round, prev_local_model=None,
                imp_w_list=None):
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
                                              shuffle=True, num_workers=config.num_workers,
                                              drop_last=True)

    local_model = deepcopy(global_model).to(device)
    _global_model = deepcopy(global_model).to(device)

    if config.optimizer == 'adam':
        optimizer = torch.optim.Adam(local_model.parameters(), lr=learning_rate, weight_decay=0.0001)
    elif config.optimizer == 'sgd':
        optimizer = torch.optim.SGD(local_model.parameters(), lr=learning_rate,
                                    momentum=config.momentum, weight_decay=0.0001)
       
    criterion = nn.MSELoss()

    if config.agg_method == 'MOON':
        contrastive_temp = config.contrastive_temp
        prev_local_model.eval()
        for param in prev_local_model.parameters():
            param.requires_grad = False
            
        prev_local_model.cuda()

        cos = torch.nn.CosineSimilarity(dim=-1)
        const_loss = torch.nn.CrossEntropyLoss().cuda()
        
    if config.agg_method == 'FedKLIEP':
        kliep = KLIEP(bandwidth=config.bandwidth, device=device, steps=config.kliep_steps, lr=config.kliep_lr, verbose=True)
        kliep.importance_weight_list = imp_w_list

    for epoch in range(config.epochs):
        local_model.train()
        epoch_loss = 0
        mae = 0
        epoch_const_loss = 0
        epoch_pred_loss = 0

        if config.agg_method == 'MOON':
            progress_bar = tqdm(enumerate(TrainLoader), total=len(TrainLoader), ncols=130)
        else:
            progress_bar = tqdm(enumerate(TrainLoader), total=len(TrainLoader), ncols=100)
        
        for batch_idx, batch in progress_bar:
            images, labels = batch[0].to(device), batch[1].to(device)

            optimizer.zero_grad()

            if config.agg_method == 'FedAvg':

                output = local_model(images)
                loss = criterion(output.squeeze(), labels.squeeze())

            elif config.agg_method == 'FedProx':

                proximal_term = 0
                proximal_mu = config.proximal_mu
                output = local_model(images)

                for local_w, glob_w in zip(local_model.parameters(), _global_model.parameters()):
                    proximal_term += torch.square((local_w - glob_w).norm(2))

                loss = criterion(output.squeeze(), labels.squeeze()) + (proximal_mu / 2) * proximal_term

            elif config.agg_method == 'MOON':

                if _round > config.warmup:

                    output, represent_local = local_model(images, represent=True)
                    _, represent_global = _global_model(images, represent=True)
                    
                    pos_sim = cos(represent_local, represent_global).reshape(-1, 1)

                    prev_local_model.to(device)

                    with torch.no_grad():
                        _, represent_prev_local = prev_local_model(images, represent=True)

                    neg_sim = cos(represent_local, represent_prev_local).reshape(-1, 1)
                    logits = torch.cat((pos_sim, neg_sim), dim=1) / contrastive_temp
                    const_labels = torch.zeros(logits.size(0)).to(device).long()

                    prev_local_model.to("cpu")
                    
                    contrastive_loss = config.proximal_mu * const_loss(logits, const_labels)
                    pred_loss = criterion(output.squeeze(), labels.squeeze())
                    loss = contrastive_loss + pred_loss
                    
                else:
                    output = local_model(images, represent=False)
                    pred_loss = criterion(output.squeeze(), labels.squeeze())
                    loss = pred_loss

            elif config.agg_method == 'FedKLIEP':
                if _round > config.warmup:
                    with torch.no_grad():
                        (loc_rep_list, glob_rep_list, 
                         loc_hooks, glob_hooks) = get_activation_for_models(local_model, global_model, images)
                    
                    kliep.fit(loc_rep_list, glob_rep_list)

                    for l_hook, g_hook in zip(loc_hooks, glob_hooks):
                        l_hook.remove()
                        g_hook.remove()

                    del loc_rep_list, glob_rep_list
                    torch.cuda.empty_cache()

                    output = local_model(images, kliep.importance_weight_list)

                else:
                    output = local_model(images, kliep.importance_weight_list)

            loss = criterion(output.squeeze(), labels.squeeze())

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if config.agg_method == 'MOON':
                if _round > config.warmup:
                    epoch_const_loss += contrastive_loss.item()
                else:
                    epoch_const_loss = 0
                epoch_pred_loss += pred_loss.item()

            mae += MAE(output.detach().cpu().numpy().squeeze(), 
                             labels.detach().cpu().numpy().squeeze())
            
            if config.agg_method == 'MOON':
                progress_bar.set_postfix({
                                        "Client": client_idx,
                                        "[Train] epoch": epoch+1,
                                        "pred_loss": round(epoch_pred_loss / (batch_idx+1), 3),
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

    del _global_model, prev_local_model
    torch.cuda.empty_cache()

    if config.agg_method == 'FedKLIEP':
        return local_model.cpu().state_dict(), kliep.importance_weight_list#.cpu().detach().numpy()
    else:
        return local_model.cpu().state_dict(), None


def inference(client_idx, global_model, local_weight, TestDataset_dict, config, device, imp_w_list):
    
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
            if config.agg_method == 'FedKLIEP':
                output = local_model(images, imp_w_list)
            else:
                output = local_model(images)
        else:
            if config.agg_method == 'FedKLIEP':
                output = global_model(images, imp_w_list)
            else:
                output = global_model(images)

        test_loss += criterion(output.squeeze(), labels.squeeze()).item()
        
        mae += MAE(output.detach().cpu().numpy().squeeze(), 
                   labels.detach().cpu().numpy().squeeze())
        
    return (test_loss / len(TestLoader), mae / len(TestLoader))



def LocalTrain(client_idx, TrainDataset_dict, ValDataset_dict, run_wandb, config, device, run_name=None):

    model = generate_model(config).to(device)

    best_valid_MAE = float('inf')

    if config.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=0.0001)
    elif config.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr,
                                    momentum=config.momentum, weight_decay=0.0001)
    
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9995)

    if config.agg_method == 'Local':
        TrainLoader = torch.utils.data.DataLoader(TrainDataset_dict[client_idx], 
                                                batch_size=config.batch_size, 
                                                shuffle=True, num_workers=config.num_workers)
        
        ValLoader = torch.utils.data.DataLoader(ValDataset_dict[client_idx],
                                                batch_size=config.batch_size,
                                                shuffle=False, num_workers=config.num_workers)
    elif config.agg_method == 'Center':
        TrainLoader = torch.utils.data.DataLoader(TrainDataset_dict, 
                                                  batch_size=config.batch_size, 
                                                  shuffle=True, num_workers=config.num_workers,
                                                  collate_fn=custom_collate_fn)
    
        ValLoader = torch.utils.data.DataLoader(ValDataset_dict,
                                                batch_size=config.batch_size,
                                                shuffle=False, num_workers=config.num_workers,
                                                collate_fn=custom_collate_fn)
    
    criterion = nn.MSELoss()
    
    for epoch in range(config.epochs):
        epoch += 1
        model.train()
        mae = 0
        epoch_loss = 0
        
        progress_bar = tqdm(enumerate(TrainLoader), total=len(TrainLoader), ncols=120)
        
        for batch_idx, batch in progress_bar:
            images, labels = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output.squeeze(), labels.squeeze())
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            mae += MAE(output.detach().cpu().numpy().squeeze(), 
                        labels.detach().cpu().numpy().squeeze())            

            progress_bar.set_postfix({
                                    "Client": client_idx,
                                    "[Train] epoch": epoch,
                                    "MSE_loss": round(epoch_loss / (batch_idx + 1), 3),
                                    "MAE_loss": round(mae / (batch_idx + 1), 3),
                                    })
        
        epoch_loss /= len(TrainLoader)
        mae /= len(TrainLoader)

        if not config.nowandb:
            if config.agg_method == 'Local':
                run_wandb.log({
                    "epoch": epoch,
                    f"C{client_idx}-Train_Loss": epoch_loss,
                    f'C{client_idx}-Train_MAE': mae,
                })
            elif config.agg_method == 'Center':
                run_wandb.log({
                    "epoch": epoch,
                    "Train_Loss": epoch_loss,
                    'Train_MAE': mae,
                })

        # Validation
        mae = 0
        epoch_loss = 0
        model.eval()

        for _, batch in enumerate(ValLoader):
            with torch.no_grad():
                images, labels = batch[0].to(device), batch[1].to(device)
                output = model(images)
                loss = criterion(output.squeeze(), labels.squeeze())
                epoch_loss += loss.item()
                mae += MAE(output.detach().cpu().numpy().squeeze(), 
                            labels.detach().cpu().numpy().squeeze())
            
        mae = mae / len(ValLoader)
        epoch_loss = epoch_loss / len(ValLoader)
        
        if not config.nowandb:
            if config.agg_method == 'Local':

                run_wandb.log({
                    "epoch": epoch,
                    f"C{client_idx}-Valid_Loss": epoch_loss,
                    f'C{client_idx}-Valid_MAE': mae,
                })
                
            elif config.agg_method == 'Center':
                run_wandb.log({
                    "epoch": epoch,
                    "Valid_Loss": epoch_loss,
                    'Valid_MAE': mae,
                })

        if (mae < best_valid_MAE):
            best_valid_MAE = mae
            save_dict = {
                "epoch": epoch,
                "Client_idx": client_idx,
                "model": model.cpu().state_dict(),
            }
            if config.agg_method == 'Local':
                torch.save(save_dict, os.path.join(config.save_path, config.agg_method, run_name,
                                                   f"C{str(client_idx).zfill(2)}_best_model_{run_name}.pth"))
            elif config.agg_method == 'Center':
                torch.save(save_dict, os.path.join(config.save_path, config.agg_method, run_name,
                                                   f"Center_best_model_{run_name}.pth"))
            model.to(device)
            bestmodel = deepcopy(model)

        lr_scheduler.step()

    return bestmodel


def SaveBestResult(client_idx, bestmodel, TrainDataset_dict, ValDataset_dict, TestDataset_dict, run_wandb, config, device):
    
    bestmodel.to(device)
    bestmodel.eval()
    result_df = pd.DataFrame()

    criterion = nn.MSELoss()

    TrainLoader = torch.utils.data.DataLoader(TrainDataset_dict[client_idx], 
                                              batch_size=config.batch_size, 
                                              shuffle=False, num_workers=config.num_workers,
                                              collate_fn=custom_collate_fn
                                              )
    
    ValLoader = torch.utils.data.DataLoader(ValDataset_dict[client_idx],
                                            batch_size=config.batch_size,
                                            shuffle=False, num_workers=config.num_workers,
                                            collate_fn=custom_collate_fn)
    
    TestLoader = torch.utils.data.DataLoader(TestDataset_dict[client_idx],
                                             batch_size=config.batch_size, 
                                             shuffle=False, num_workers=config.num_workers,
                                             collate_fn=custom_collate_fn)

    Loaders = [TrainLoader, ValLoader, TestLoader]
    mode_list = ['Train', 'Valid', 'Test']

    for _mode, Loader in zip(mode_list, Loaders):
        pred_age = []
        true_age = []
        Subject_list = []
        Sex_list = []
        col_mode = [_mode]*len(Loader.dataset)

        mae = 0
        epoch_loss = 0

        for step, batch in enumerate(Loader):
            with torch.no_grad():
                (images, labels, Subject, Sex) = batch[0].to(device), batch[1].to(device), batch[2], batch[3]
                output = bestmodel(images)
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

        if not config.nowandb:
            run_wandb.log({
                            f"Best-{_mode}_Loss (c{client_idx}|{get_key_by_value(dataset_dict, client_idx)})": epoch_loss,
                            f'Best-{_mode}_MAE (c{client_idx}|{get_key_by_value(dataset_dict, client_idx)})': mae,
                            })
            
        temp_df = pd.DataFrame({'Subject': Subject_list,
                                'Sex': Sex_list,
                                'Age': true_age,
                                'pred_age': pred_age,
                                'mode': col_mode})
        
        result_df = pd.concat([result_df, temp_df], axis=0)
        
    if config.agg_method == 'Local':
        result_df.to_csv(os.path.join(config.save_path, config.agg_method,
                                  f"C{str(client_idx).zfill(2)}_{get_key_by_value(dataset_dict, client_idx)}_result_local.csv"), index=False)
    else:
        result_df.to_csv(os.path.join(config.save_path, config.agg_method,
                                  f"C{str(client_idx).zfill(2)}_{get_key_by_value(dataset_dict, client_idx)}_result_center.csv"), index=False)