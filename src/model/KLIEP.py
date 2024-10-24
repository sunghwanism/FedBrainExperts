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
        self.importance_weight_list = None


    def kde(self, batch_data):
        B, C, H, W, D = batch_data.shape
        self.C, self.H, self.W, self.D = C, H, W, D
        batch_data = batch_data.view(B, C, -1)
        density = self.gaussian_kernel(batch_data)
        torch.cuda.empty_cache()

        return density

    def gaussian_kernel(self, distance):
        pi = torch.tensor(torch.pi, device=self.device)

        return torch.exp(-0.5 * (distance / self.bandwidth) ** 2) / (self.bandwidth * torch.sqrt(2 *pi))


    def fit(self, loc_rep_list, glob_rep_list):

        self.init_importance_weights(loc_rep_list)

        for i in range(len(loc_rep_list)):
            loc_rep = loc_rep_list[i]
            glob_rep = glob_rep_list[i]

            loc_density = self.kde(loc_rep)
            glob_density = self.kde(glob_rep)
            
            if self.verbose:
                print(f"Optimizing Layer {i+1}...")
            imp_w = self.optimizer(loc_density, glob_density, i)
            imp_w = imp_w.detach().requires_grad_(False)

            self.importance_weight_list[i] = imp_w
        
        del imp_w, loc_density, glob_density, loc_rep, glob_rep
        torch.cuda.empty_cache()

        return self.importance_weight_list


    def optimizer(self, loc_density, glob_density, layer_idx):
        importance_weights = self.importance_weight_list[layer_idx]
        importance_weights = importance_weights.detach().requires_grad_(True)

        optimizer = optim.Adam([importance_weights], lr=self.lr)

        loc_density = loc_density.view(loc_density.shape[0], -1, self.H, self.W, self.D)
        glob_density = glob_density.view(glob_density.shape[0], -1, self.H, self.W, self.D)
        
        for step in range(self.num_steps):
            importance_weights = importance_weights.view(importance_weights.shape[0], self.C, -1)
            norm_importance_weights = torch.softmax(importance_weights, dim=-1)
            norm_importance_weights = norm_importance_weights.view(norm_importance_weights.shape[0], self.C, self.H, self.W, self.D)

            weighted_loc_density = norm_importance_weights * loc_density
            
            loss = torch.mean(loc_density * (torch.log(loc_density + self.epsilon) - torch.log(glob_density + self.epsilon)))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if self.verbose:
                if (step+1) % 100 == 0:
                    print(f"Step [{step+1}/{self.num_steps}], Loss: {loss.item():.5f}")

        del weighted_loc_density, loss, optimizer
        torch.cuda.empty_cache()
        print(norm_importance_weights[4][2].sum())
        return norm_importance_weights
    
    def init_importance_weights(self, rep_list):
        if self.importance_weight_list[0] == int:
            print("Initializing Importance Weights...")
            self.importance_weight_list = []

            for rep in rep_list:
                B, C, H, W, D = rep.shape
                importance_weights = torch.empty(B, C, H, W, D, device=self.device, requires_grad=True)
                torch.nn.init.xavier_normal_(importance_weights)
                self.importance_weight_list.append(importance_weights)