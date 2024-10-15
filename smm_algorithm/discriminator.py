import torch
import torch.nn as nn
import numpy as np


class Discriminator(torch.nn.Module):
    def __init__(self,  state_dim, n_skill, env_name, featurize,
                latent_dim=16, hidden_dim=256, beta=0.5):
        super(Discriminator, self).__init__()
        self.env_name = env_name
        self.featurize = featurize
        if self.env_name == 'hopper':
            self.feature_dim = 3
        elif self.env_name == 'fetch':
            self.feature_dim = 3
        elif self.env_name == 'fetch_push':
            self.feature_dim = 2
        elif self.env_name == 'finger':
            self.feature_dim = 3
        # DIAYN discriminator
        self.l1=torch.nn.Linear(state_dim, 256) if not self.featurize else torch.nn.Linear(self.feature_dim, 256)
        # self.l1=torch.nn.Linear(state_dim, 256)
        self.l2=torch.nn.Linear(256, 64)
        self.l3=torch.nn.Linear(64, n_skill)
        # VAE 
        self.input_dim = state_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.beta = beta
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 2 * latent_dim)
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, state_dim)
        )

    
    def forward(self, s):
        x = self.featurizer(s) if self.featurize else s
        x=torch.nn.functional.relu(self.l1(x))
        x=torch.nn.functional.relu(self.l2(x))
        x=self.l3(x)
        x=torch.softmax(x, dim=1)
        return x
    
    def forward_vae(self, x):
        mu_log_var = self.encoder(x).view(-1, 2, self.latent_dim)
        mu = mu_log_var[:, 0, :]
        log_var = mu_log_var[:, 1, :]
        z = self.sample(mu, log_var)
        x_recon = self.decoder(z)
        return x_recon, mu, log_var, z
    
    def loss_vae(self, s):
        # compute reconstruction loss
        x_recon, mu, log_var, z = self.forward_vae(s)
        recon_loss = torch.nn.functional.mse_loss(x_recon, s, reduction='sum')
        # compute kl divergence
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + kl_div
    
    def loss_vae_n(self, s):
        # compute reconstruction loss
        x_recon, mu, log_var, z = self.forward_vae(s)
        recon_loss =torch.sqrt( torch.sum(torch.nn.functional.mse_loss(x_recon, s, reduction='none'),dim=-1))
        # compute kl divergence
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        return recon_loss + kl_div

    def featurizer(self, s):
        if self.env_name == 'hopper':
            return s[: , -6:-3]
        elif self.env_name == 'fetch':
            return s[: , 0:3]
        elif self.env_name == 'fetch_push':
            return s[: , 3:5]
        elif self.env_name == 'finger':
            return s[: , 0:3]
        
    def sample(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def sample_recon(self, z):
        x_recon = self.decoder(z)
        return x_recon

    def sample_latent_recon(self, x):
        mu_log_var = self.encoder(x).view(-1, 2, self.latent_dim)
        mu = mu_log_var[:, 0, :]
        log_var = mu_log_var[:, 1, :]
        z = self.sample(mu, log_var)
        x_recon = self.decoder(z)
        return z, x_recon