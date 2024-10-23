import torch
import torch.nn as nn



class KLIEP:

    def __init__(self, bandwidth):
        self.bandwidth = bandwidth


    def kde(self, batch_data):
        dist = torch.cdist(batch_data, batch_data)
        kernels = self.gaussian_kernel(dist)
        density = kernels.mean(dim=1)

        return density

    def gaussian_kernel(self, distance):
        pi = torch.tensor(torch.pi, device=distance.device)
        return torch.exp(-0.5 * (distance / self.bandwidth) ** 2) / (self.bandwidth * torch.sqrt(2 *pi))
    
    def __call__(self, batch_data, test_data):
        batch_density = self.kde(batch_data)
        test_density = self.kde(test_data)
        
        
        return (test_density / batch_density).mean()
    
