import numpy as np
import torch



def MAE(output, target):

    if type(output) == list:
        output = np.array(output)
        target = np.array(target)

    if output.shape != target.shape:
        output = output.view(-1)
        target = target.view(-1)
        
    if type(output) == np.ndarray:
        return np.mean(np.abs(output - target))

    elif type(output) == torch.Tensor:
        return torch.mean(torch.abs(output - target))

def MSE(output, target):
    if output.shape != target.shape:
        output = output.view(-1)
        target = target.view(-1)

    if type(output) == np.ndarray:
        return np.mean(np.abs(output - target))
    elif type(output) == torch.Tensor:
        return torch.mean(torch.abs(output - target))


def aggregate_result(result):
    """
    Aggregate the MAE from multiple clients

    Args:
    - result (dict): a dictionary containing MAE values from multiple clients: 
      {client_idx: (train_loss, val_loss, train_mae, val_mae)}

    Returns:
    - agg_loss (set): the aggreagated loss value --> (train_loss, val_loss)
    - aggregated_mae (set): the aggregated MAE --> (train_mae, val_mae)
    """

    client_idx = list(result.keys())

    train_loss = np.array([result[client][0] for client in client_idx])
    val_loss = np.array([result[client][1] for client in client_idx])
    train_mae = np.array([result[client][2] for client in client_idx])
    val_mae = np.array([result[client][3] for client in client_idx])

    train_agg_loss = np.mean(train_loss)
    train_aggregated_mae = np.mean(train_mae)

    val_agg_loss = np.mean(val_loss)
    val_aggregated_mae = np.mean(val_mae)

    agg_loss = (train_agg_loss, val_agg_loss)
    aggregated_mae = (train_aggregated_mae, val_aggregated_mae)

    return agg_loss, aggregated_mae