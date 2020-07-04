import torch
import torch.nn as nn

EPS=1e-9

class KLdivergence(nn.Module):
    def __init__(self):
        super(KLdivergence, self).__init__()
        
    def forward(self, mean, var, eps=EPS, batch_mean=True):
        """
        Args:
            mean (batch_size, latent_dim)
            var (batch_size, latent_dim)
        """
        loss = 1 + torch.log(var + eps) - mean**2 - var
        loss = - 0.5 * loss.sum(dim=1)

        if batch_mean:
            loss = loss.mean(dim=0)

        return loss

class BinaryCrossEntropy(nn.Module):
    def __init__(self):
        super(BinaryCrossEntropy, self).__init__()
        
    def forward(self, input, target, eps=EPS, batch_mean=True):
        """
        Args:
            input: (batch_size, n_samples, latent_dim)
            target: (batch_size, latent_dim)
        """
        target = target.unsqueeze(dim=1)
        
        loss = - target * torch.log(input + eps) - (1 - target)*torch.log(1 - input + eps)
        loss = loss.sum(dim=2).mean(dim=1)

        if batch_mean:
            loss = loss.mean(dim=0)

        return loss
