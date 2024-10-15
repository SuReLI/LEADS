import torch
import torch.nn as nn
import numpy as np
from torch.distributions.cauchy import Cauchy
import torch.nn.functional as F
import torch.distributions as dist
from torch.distributions import Normal





class Codebook(nn.Module):
    def __init__(self, n, d, device='cpu', std=1, learning_rate=1e-4):
        """ Initialize the codebook
        n: number of codebook vectors
        d: dimension of codebook vectors """
        super(Codebook, self).__init__()
        self.n=n
        self.d=d
        self.device=device
        self.std_gaussian_ball=std/np.sqrt(float(d))
        self.w=1.0
        self.codebook_vectors = nn.Parameter((self.sample_orthonormal_vectors(n,d)*self.w).to(device))
        # self.codebook_vectors = nn.Parameter(torch.eye(n,d).to(device))
        # self.codebook_vectors = torch.eye(n,d).to(device)*100
        self.one_hot_encoding = torch.eye(n).to(device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
    
    def orthonormal_loss(self):
        """ Compute the orthonormal loss """
        # orthonormal_loss=torch.norm(torch.matmul( self.codebook_vectors, torch.transpose(self.codebook_vectors, 0, 1))-torch.eye(self.codebook_vectors.shape[0]).to(self.codebook_vectors.device))**2
        orthonormal_loss=torch.norm(self.codebook_vectors@torch.transpose(self.codebook_vectors,0,1)/self.w**2-torch.eye(self.codebook_vectors.shape[0]).to(self.codebook_vectors.device))**2

        return(orthonormal_loss)
    
    
    def sample(self, n):
        """ Sample n codebook vectors 
        n: number of codebook vectors to sample (can be higher than the number of codebook vectors))"""
        idx = torch.randint(0, self.n, size=(n,))
        return(self.codebook_vectors[idx], idx)

    

    def sample_gaussian_around(self, n, vec=None):
        # Select n vectors from the codebook randomly
        indices = torch.randint(0, self.n, size=(n,)) 
        selected_vectors = self.codebook_vectors[indices] if vec is None else vec
        # Sample n vectors from a gaussian ball around the selected vectors
        distribution = Normal(loc=selected_vectors, scale=self.std_gaussian_ball)
        perturbed_vectors = distribution.sample()
        return perturbed_vectors
    
    def sample_orthonormal_vectors(self, n, d):
        # Generate a random matrix of size n x d
        random_matrix = torch.randn(d, n)
        # Apply QR decomposition to obtain orthogonal matrix
        Q, _ = torch.linalg.qr(random_matrix, 'complete')
        # Normalize each vector (this is usually not needed as QR decomposition 
        # should return an orthonormal matrix already)
        # Q_normalized = Q / Q.norm(dim=1, keepdim=True)
        return Q.T[:, :n].T

    def closest_z(self, batch_z): 
        """ Return the one hot encoding of closest codebook vector to z
        z: vector to compare with the codebook vectors """
        batch_z=torch.tensor(batch_z).to(self.device)
        # Reshape batch_z to (-1, d) 
        batch_z = batch_z.view(-1, self.d)
        # Calculate Euclidean distances between batch vectors and codebook vectors
        distances = torch.norm(self.codebook_vectors.unsqueeze(0) - batch_z.unsqueeze(1), dim=2)
        # Get indices of minimum distances
        _, indices = torch.min(distances, dim=1)
        return self.one_hot_encoding[indices], indices
        
    





if __name__=="__main__":
    print("Codebook")
    codebook=Codebook(3,5)
    print(codebook.codebook_vectors)
    # sample= codebook.sample_gaussian_around(10)
    # print(sample)
    # print(codebook.closest_z(sample))
    # check othonormality
    print(codebook.codebook_vectors@torch.transpose(codebook.codebook_vectors,0,1))
    print(codebook.orthonormal_loss())