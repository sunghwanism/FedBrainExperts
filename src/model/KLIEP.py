import torch
import torch.nn as nn



class KLIEP:

    def __init__(self, bandwidth, device):
        self.bandwidth = bandwidth
        self.device = device
        self.softmax = nn.Softmax(dim=-1)

    def kde(self, batch_data):
        B, C, H, W, D = batch_data.shape
        batch_data = batch_data.view(B, C, -1)
        density = self.gaussian_kernel(batch_data)
        torch.cuda.empty_cache()

        return density

    def gaussian_kernel(self, distance):
        pi = torch.tensor(torch.pi, device=self.device)
        return torch.exp(-0.5 * (distance / self.bandwidth) ** 2) / (self.bandwidth * torch.sqrt(2 *pi))

    def fit(self, loc_rep, glob_rep):
        B, C, H, W, D = loc_rep.size()
        loc_density = self.kde(loc_rep)
        glob_density = self.kde(glob_rep)
        density_ratio = loc_density / glob_density
        imp_w = self.softmax(density_ratio)
        imp_w = imp_w.view(B, C, H, W, D)
        
        return imp_w