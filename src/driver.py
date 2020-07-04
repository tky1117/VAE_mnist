import os
import matplotlib.pyplot as plt
import torch

from criterion import KLdivergence, BinaryCrossEntropy

class Trainer:
    def __init__(self, model, loader, optimizer, args):
        self.loader = loader
        
        self.model = model
        self.optimizer = optimizer
        
        self._reset(args)
        
    def _reset(self, args):
        self.n_samples = args.n_samples
        self.epochs = args.epochs
    
        self.kl_divergence = KLdivergence()
        self.reconstruction = BinaryCrossEntropy()
        
        # Loss
        self.train_loss_list = torch.Tensor(self.epochs)
        self.train_kl_loss_list = torch.Tensor(self.epochs)
        self.train_reconstruction_loss_list = torch.Tensor(self.epochs)
        self.valid_loss_list = torch.Tensor(self.epochs)
        self.valid_kl_loss_list = torch.Tensor(self.epochs)
        self.valid_reconstruction_loss_list = torch.Tensor(self.epochs)
        
        self.best_loss = float('infinity')
        self.no_improvement = 0
        
        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)
    
    def save_model(self, save_path, epoch=0):
        package = {}
        package['epoch'] = epoch + 1
        package['state_dict'] = self.model.state_dict()
        package['optim_dict'] = self.optimizer.state_dict()
        package['train_loss'] = self.train_loss_list
        package['train_kl_loss'] = self.train_kl_loss_list
        package['train_reconstruction_loss'] = self.train_reconstruction_loss_list
        package['valid_loss'] = self.valid_loss_list
        package['valid_kl_loss'] = self.valid_kl_loss_list
        package['valid_reconstruction_loss'] = self.valid_reconstruction_loss_list
        package['best_loss'] = self.best_loss
        package['no_improvement'] = self.no_improvement
        
        torch.save(package, save_path)
        
    def run(self):
        for epoch in range(self.epochs):
            train_loss, valid_loss = self.run_one_epoch(epoch)
            
            if valid_loss < self.best_loss:
                self.best_loss = valid_loss
                self.no_improvement = 0
                save_path = os.path.join(self.save_dir, "best.pth")
                self.save_model(save_path, epoch=epoch)
            else:
                self.no_improvement += 1
                if self.no_improvement%3 == 0:
                    optim_dict = self.optimizer.state_dict()
                    print("Learning rate: {} -> {}".format(optim_dict['param_groups'][0]['lr'], optim_dict['param_groups'][0]['lr']*0.5))
                    optim_dict['param_groups'][0]['lr'] = optim_dict['param_groups'][0]['lr'] * 0.5
                    self.optimizer.load_state_dict(optim_dict)
                    
            save_path = os.path.join(self.save_dir, "last.pth")
            self.save_model(save_path, epoch=epoch)
            
            if self.no_improvement >= 10:
                return
                
    def run_one_epoch(self, epoch):
        n_train_batch = len(self.loader['train'])
        train_loss = 0
        train_kl_loss = 0
        train_reconstruction_loss = 0
    
        self.model.train()
    
        for x, t in self.loader['train']:
            if torch.cuda.is_available():
                x = x.cuda()

            self.optimizer.zero_grad()

            y, z, mean, var = self.model(x, n_samples=self.n_samples, return_params=True)

            kl_loss = self.kl_divergence(mean=mean, var=var)
            reconstruction_loss = self.reconstruction(input=y, target=x)
            loss = kl_loss + reconstruction_loss

            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
            train_kl_loss += kl_loss.item()
            train_reconstruction_loss += reconstruction_loss.item()
            
        train_loss /= n_train_batch
        train_kl_loss /= n_train_batch
        train_reconstruction_loss /= n_train_batch
        self.train_loss_list[epoch] = train_loss
        self.train_kl_loss_list[epoch] = train_kl_loss
        self.train_reconstruction_loss_list[epoch] = train_reconstruction_loss
            
        n_valid = len(self.loader['valid'].dataset)
        valid_loss = 0
        valid_kl_loss = 0
        valid_reconstruction_loss = 0
        
        self.model.eval()
        
        with torch.no_grad():
            for x, t in self.loader['valid']:
                if torch.cuda.is_available():
                    x = x.cuda()

                y, z, mean, var = self.model(x, return_params=True)
                
                kl_loss = self.kl_divergence(mean=mean, var=var, batch_mean=False)
                reconstruction_loss = self.reconstruction(input=y, target=x, batch_mean=False)
                loss = kl_loss + reconstruction_loss
                
                valid_loss += loss.sum().item()
                valid_kl_loss += kl_loss.sum().item()
                valid_reconstruction_loss += reconstruction_loss.sum().item()
        
        valid_loss /= n_valid
        valid_kl_loss /= n_valid
        valid_reconstruction_loss /= n_valid
        
        self.valid_loss_list[epoch] = valid_loss
        self.valid_kl_loss_list[epoch] = valid_kl_loss
        self.valid_reconstruction_loss_list[epoch] = valid_reconstruction_loss
        
        print("[Epoch {}/{}] Train Lower Bound:{:.5f}, (KL: {:.5f}, Reconstruction: {:.5f}), Valid Lower Bound: {:.5f} (KL: {:.5f}, Reconstruction: {:.5f})".format(epoch+1, self.epochs, train_loss, train_kl_loss, train_reconstruction_loss, valid_loss, valid_kl_loss, valid_reconstruction_loss), flush=True)
        
        return train_loss, valid_loss
        
class Tester:
    def __init__(self, model, loader, args):
        self.loader = loader
        
        self.model = model
        
        self._reset(args)
        
    def _reset(self, args):
        self.n_samples = args.n_samples
    
        self.kl_divergence = KLdivergence()
        self.reconstruction = BinaryCrossEntropy()
        
        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.load_model(args.model_path)

    def load_model(self, model_path):
        package = torch.load(model_path)
        self.model.load_state_dict(package['state_dict'])
        
    def run(self):
        n_test = len(self.loader.dataset)
        test_loss = 0
        test_kl_loss = 0
        test_reconstruction_loss = 0
        
        self.model.eval()
        
        with torch.no_grad():
            # Original image
            fig = plt.figure(figsize=(10, 10))
            
            for idx in range(100):
                x, t = self.loader.dataset[idx]
                
                x = x.unsqueeze(dim=0)
                im = x.view(-1, 28, 28).permute(1, 2, 0).squeeze().numpy()
                ax = fig.add_subplot(10, 10, idx+1, xticks=[], yticks=[])
                ax.imshow(im, 'gray')
                
                

            save_path = os.path.join(self.save_dir, "original.png")
            fig.savefig(save_path, bbox_inches='tight')
                
            # Reconstruction image
            fig = plt.figure(figsize=(10, 10))

            for idx in range(100):
                x, t = self.loader.dataset[idx]

                if torch.cuda.is_available():
                    x = x.cuda()

                x = x.unsqueeze(dim=0)
                y, z, _, _ = self.model(x, return_params=True)
                im = y.view(-1, 28, 28).permute(1, 2, 0).cpu().squeeze().detach().numpy()
                ax = fig.add_subplot(10, 10, idx+1, xticks=[], yticks=[])
                ax.imshow(im, 'gray')

            save_path = os.path.join(self.save_dir, "reconstrunction.png")
            fig.savefig(save_path, bbox_inches='tight')

            # Random image
            fig = plt.figure(figsize=(10, 10))
            
            latent_dim = self.model.latent_dim
            for idx in range(100):
                z = torch.randn((1, latent_dim))
                if torch.cuda.is_available():
                    z = z.cuda()

                y = self.model.decoder(z)
                im = y.view(-1, 28, 28).permute(1, 2, 0).cpu().squeeze().detach().numpy()
                ax = fig.add_subplot(10, 10, idx+1, xticks=[], yticks=[])
                ax.imshow(im, 'gray')

            save_path = os.path.join(self.save_dir, "random.png")
            fig.savefig(save_path, bbox_inches='tight')

            fig = plt.figure(figsize=(5, 5))

            x_source, t = self.loader.dataset[0]
    
            if torch.cuda.is_available():
                x_source = x_source.cuda()
                
            x_source = x_source.unsqueeze(0)
            y_source, z_source, mean, var = self.model(x_source, return_params=True)
            im = y_source.view(-1, 28, 28).permute(1, 2, 0).cpu().squeeze().detach().numpy()
            ax_source = fig.add_subplot(1, 2, 1, xticks=[], yticks=[])
            ax_source.set_title('Source image')
            ax_source.imshow(im, 'gray')

            x_target, t = self.loader.dataset[3]

            if torch.cuda.is_available():
                x_target = x_target.cuda()
            x_target = x_target.unsqueeze(dim=0)
            y_target, z_target, mean, var = self.model(x_target, return_params=True)
            im = y1.view(-1, 28, 28).permute(1, 2, 0).cpu().squeeze().detach().numpy()
            ax_target = fig.add_subplot(1, 2, 2, xticks=[], yticks=[])
            ax_target.set_title('Target image')
            ax_target.imshow(im, 'gray')
            
            save_path = os.path.join(self.save_dir, "source-target.png")
            fig.savefig(save_path, bbox_inches='tight')
            
            z_source, z_target = z_source.squeeze(dim=1), z_target.squeeze(dim=1)
            fig = plt.figure(figsize=(15, 15))
            z_linear = torch.cat([z_target * (idx * 0.1) + z_source * ((9 - idx) * 0.1) for idx in range(10)])
            z_linear = z_linear.view((10, 1, -1))
            y = self.model.decoder(z_linear).view(-1, 28, 28)
            
            for idx, im in enumerate(y.cpu().detach().numpy()):
                ax = fig.add_subplot(1, 10, idx+1, xticks=[], yticks=[])
                ax.imshow(im, 'gray')
                
            save_path = os.path.join(self.save_dir, "interpolation.png")
            fig.savefig(save_path, bbox_inches='tight')
