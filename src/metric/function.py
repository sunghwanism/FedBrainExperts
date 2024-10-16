import numpy as np
import torch



def MAE(output, target):
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

