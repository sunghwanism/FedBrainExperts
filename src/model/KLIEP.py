import torch
import torch.nn as nn
import torch.optim as optim


class KLIEP:

    def __init__(self, bandwidth, device, steps, lr=0.0001, epsilon = 1e-9, verbose=False):
        self.bandwidth = bandwidth
        self.device = device
        self.epsilon = epsilon
        self.lr = lr
        self.verbose = verbose
        self.num_steps = steps


    def kde(self, batch_data):
        B, C, _, _, _ = batch_data.shape
        batch_data = batch_data.view(B, C, -1)
        density = self.gaussian_kernel(batch_data)
        torch.cuda.empty_cache()

        return density

    def gaussian_kernel(self, distance):
        pi = torch.tensor(torch.pi, device=self.device)

        return torch.exp(-0.5 * (distance / self.bandwidth) ** 2) / (self.bandwidth * torch.sqrt(2 *pi))


    def fit(self, loc_rep_list, glob_rep_list):
        importance_weight_list = []

        for i in range(len(loc_rep_list)):
            loc_rep = loc_rep_list[i]
            glob_rep = glob_rep_list[i]

            loc_density = self.kde(loc_rep)
            glob_density = self.kde(glob_rep)
            
            if self.verbose:
                print(f"Optimizing Layer {i+1}...")
            imp_w = self.optimizer(loc_density, glob_density, lr=self.lr)

            importance_weight_list.append(imp_w)
        
        del imp_w, loc_density, glob_density, loc_rep, glob_rep
        torch.cuda.empty_cache()

        return importance_weight_list


    def optimizer(self, loc_density, glob_density):

        B, C, H, W, D = loc_density.shape
        importance_weights = torch.randn(B, C, H, W, D, device=self.device, requires_grad=True)

        optimizer = optim.SGD([importance_weights], lr=self.lr)

        for step in range(self.num_steps):
            importance_weights_normalized = torch.softmax(importance_weights, dim=(2, 3, 4))
            weighted_loc_density = importance_weights_normalized * loc_density

            loss = torch.mean(weighted_loc_density - glob_density * torch.log(weighted_loc_density + self.epsilon))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if self.verbose:
                print(f"Step [{step+1}/{self.num_steps}], Loss: {loss.item():.5f}")

        del weighted_loc_density, loss, optimizer
        torch.cuda.empty_cache()

        return importance_weights_normalized