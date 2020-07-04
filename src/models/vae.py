import torch
import torch.nn as nn

class VAEbaseline(nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_dim, n_layers=3):
        super(VAEbaseline, self).__init__()
        
        self.latent_dim = latent_dim
        
        self.encoder = Encoder(in_channels, hidden_channels, latent_dim=latent_dim, n_layers=n_layers)
        self.latent_sampler = LatentSampler()
        self.decoder = Decoder(in_channels, hidden_channels, latent_dim=latent_dim, n_layers=n_layers)
        
    def forward(self, input, n_samples=1, return_params=False):
        mean, var = self.encoder(input)
        z = self.latent_sampler(mean, var, n_samples=n_samples) # z: (batch_size, n_samples, latent_dim)
        output = self.decoder(z) # output: (batch_size, n_samples, out_channels)
        
        if return_params:
            return output, z, mean, var
        
        return output
        
class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_dim, n_layers=3):
        super(Encoder, self).__init__()

        net = []
        
        for n in range(n_layers-1):
            if n == 0:
                net.append(nn.Linear(in_channels, hidden_channels))
            else:
                net.append(nn.Linear(hidden_channels, hidden_channels))
            net.append(nn.ReLU())

        self.net = nn.Sequential(*net)
        self.linear_mean = nn.Linear(hidden_channels, latent_dim)
        self.linear_var = nn.Linear(hidden_channels, latent_dim)
        self.activation_var = nn.Softplus()
        
    def forward(self, input):
        x = self.net(input)
    
        output_mean = self.linear_mean(x)
        x_var = self.linear_var(x)
        output_var = self.activation_var(x_var)
        
        return output_mean, output_var
        
class Decoder(nn.Module):
    def __init__(self, out_channels, hidden_channels, latent_dim, n_layers=3):
        super(Decoder, self).__init__()
        
        net = []
        
        for n in range(n_layers):
            if n == 0:
                net.append(nn.Linear(latent_dim, hidden_channels))
            elif n == n_layers-1:
                net.append(nn.Linear(hidden_channels, out_channels))
            else:
                net.append(nn.Linear(hidden_channels, hidden_channels))
            if n == n_layers-1:
                net.append(nn.Sigmoid())
            else:
                net.append(nn.ReLU())
        self.net = nn.Sequential(*net)
        
    def forward(self, z):
        output = self.net(z)
        
        return output
        
        
class LatentSampler(nn.Module):
    def __init__(self):
        super(LatentSampler, self).__init__()
    
    def forward(self, mean, var, n_samples=1):
        """
        Args:
            mean (batch_size, latent_dim)
        Returns:
             z (batch_size, n_samples, latent_dim)
        """
        batch_size, latent_dim = mean.size()
        mean, var = mean.unsqueeze(dim=1), var.unsqueeze(dim=1)
        
        epsilon = torch.randn((batch_size, n_samples, latent_dim)).to(mean.device)
        z = mean + torch.sqrt(var) * epsilon
            
        return z

if __name__ == '__main__':
    batch_size = 4
    latent_dim = 10
    n_layers = 4
    in_channels, hidden_channels = 28*28, 200
    size_input = (batch_size, in_channels)

    model = VAEbaseline(in_channels, hidden_channels, latent_dim=latent_dim)
    print(model)
    
    input = torch.randint(0, 256, size_input) / 256
    output = model(input)
    
    print(output.size())
