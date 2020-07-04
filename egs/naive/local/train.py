#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import random
import numpy as np
import torch
from torchvision import datasets, transforms

from driver import Trainer
from models.vae import VAEbaseline

parser = argparse.ArgumentParser("VAE")

parser.add_argument("--latent_dim", type=int, default=None, help="Latent dimension")
parser.add_argument("--hidden_channels", type=int, default=None, help="Number of hidden channels")
parser.add_argument("--n_layers", type=int, default=None, help="Number of hidden layers")
parser.add_argument("--n_samples", type=int, default=None, help="Number of samples")
parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
parser.add_argument("--save_dir", type=str, default="./tmp", help="Directory for saving models")
parser.add_argument("--seed", type=int, default=111, help="Random seed")

def main(args):
    set_seed(args.seed)
    
    args.in_channels = 28*28
    
    # Preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))])

    train_dataset = datasets.MNIST("./data/MNIST", train=True, download=True, transform=transform)
    valid_dataset = datasets.MNIST("./data/MNIST", train=False, download=True, transform=transform)

    loader = {}
    loader['train'] = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    loader['valid'] = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    model = VAEbaseline(args.in_channels, args.hidden_channels, latent_dim=args.latent_dim, n_layers=args.n_layers)
    print(model)
    
    if torch.cuda.is_available():
        model.cuda()
        print("Use CUDA")
    else:
        print("Not use CUDA")
        
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    trainer = Trainer(model, loader=loader, optimizer=optimizer, args=args)
    trainer.run()
        
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    
if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    
    main(args)
