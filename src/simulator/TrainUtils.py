import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from copy import deepcopy

from src.metric.function import MAE, MSE


def LocalUpdate(client_idx, global_model, TrainDataset_dict, config, wandb):

    TrainLoader = torch.utils.data.DataLoader(TrainDataset_dict[client_idx], 
                                              batch_size=config.batch_size, 
                                              shuffle=True, num_workers=config.num_workers)

    ValLoader = torch.utils.data.DataLoader(ValDataset_dict[client_idx], 
                                            batch_size=config.batch_size, 
                                            shuffle=False, num_workers=config.num_workers)

    local_model = global_model.deepcopy().to(device)
    local_model.load_state_dict(global_weights)
    local_model.train()

    if config.optimizer == 'adam':
        optimizer = torch.optim.Adam(local_model.parameters(), lr=config.lr)
    elif config.optimizer == 'sgd':
        optimizer = torch.optim.SGD(local_model.parameters(), lr=config.lr, momentum=config.momentum)
    
    criterion = nn.MSELoss()

    for epoch in range(config.epochs):
        local_model.train()
        epoch_loss = 0

        for batch_idx, (images, labels) in enumerate(TrainLoader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            output = local_model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Client: {client_idx} | Epoch: {epoch} | Loss: {epoch_loss}")

        wandb.log({f"Client_{client_idx}_Train_Loss": epoch_loss})

        for batch_idx, (images, labels) in enumerate(ValLoader):
            images, labels = images.to(device), labels.to(device)
            output = local_model(images)
            loss = criterion(output, labels)
            epoch_loss += loss.item()

        wandb.log({f"Client_{client_idx}_Val_Loss": epoch_loss})
        

        

    

    
                                                