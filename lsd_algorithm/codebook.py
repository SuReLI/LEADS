import torch
import torch.nn as nn
import numpy as np
from torch.distributions.cauchy import Cauchy
import torch.nn.functional as F
import torch.distributions as dist
from torch.distributions import Normal




class Codebook(nn.Module):
    def __init__(self, n, device='cpu', std=1, learning_rate=1e-4, algo ='lds', w=1.0):
        """ Initialize the codebook
        n: number of codebook vectors
        d: dimension of codebook vectors """
        super(Codebook, self).__init__()
        self.n=n
        self.device=device
        self.w=w
        self.codebook_vectors = -1/(n-1)*torch.ones((n, n)).to(device) + (1+1/(n-1))*torch.eye(n).to(device)
        
    





if __name__=="__main__":
    print("Codebook")
    codebook=Codebook(3)
    print(codebook.codebook_vectors)
   