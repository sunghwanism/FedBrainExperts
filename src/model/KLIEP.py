import torch
import torch.optim as optim


class KLIEP:

    def __init__(self, device, steps, lr=0.001, epsilon = 1e-8, verbose=False):
        self.device = device
        self.epsilon = epsilon
        self.lr = lr
        self.verbose = verbose
        self.num_steps = steps

    def fit(self, source, target):
        source = source.detach()
        target = target.detach()

        source = self.rbf_kernel(source, source)
        target = self.rbf_kernel(target, target)

        N = source.shape[0]
        alpha = torch.full((N, 1), 1 / N, device=self.device, requires_grad=True)
        optimizer = optim.SGD([alpha], lr=self.lr)

        for step in range(self.num_steps):
            optimizer.zero_grad()
            alpha_normalized = torch.softmax(alpha, dim=0)
            loss = self.kliep_loss(source, target, alpha_normalized)
            loss.backward()
            optimizer.step()

            if self.verbose and (step+1) % self.num_steps == 0:
                print(f"Step {step} | Loss: {round(loss.item(), 3)}")

        return alpha.T

    def kliep_loss(self, source, target, alpha):
        
        target_ratio = torch.matmul(target.T, alpha)
        log_target_ratio = torch.mean(torch.log(target_ratio + self.epsilon))
        
        source_ratio = torch.matmul(source.T, alpha)
        source_mean_ratio = torch.mean(source_ratio)
        
        loss = -(log_target_ratio - source_mean_ratio)

        return loss

    def rbf_kernel(self, X, Y, sigma=1.0):
        X_norm = (X ** 2).sum(1).view(-1, 1)
        Y_norm = (Y ** 2).sum(1).view(1, -1)
        dist = X_norm + Y_norm - 2.0 * torch.mm(X, Y.t())
        return torch.exp(-dist / (2 * sigma ** 2))